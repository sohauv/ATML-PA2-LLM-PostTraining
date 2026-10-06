from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, set_seed
from common.models import (
    load_policy,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []

        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)

            chosen.append(
                encode_prompt_response(
                    tokenizer,
                    prompt,
                    yc,
                    max_length,
                )
            )

            rejected.append(
                encode_prompt_response(
                    tokenizer,
                    prompt,
                    yr,
                    max_length,
                )
            )

        return (
            pad_batch(tokenizer, chosen),
            pad_batch(tokenizer, rejected),
        )

    return collate


def prepare_dpo_run(
    config_path: str,
    dataset_path: str | None = None,
    beta: float | None = None,
    max_examples: int | None = None,
):
    cfg = load_yaml(config_path)

    set_seed(int(cfg["seed"]))

    path = dataset_path or cfg["paths"]["dpo_standard_train"]

    rows = read_jsonl(path)

    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])

    model = load_policy(
        cfg,
        trainable=True,
        fresh_lora=True,
    )

    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(
            tokenizer,
            int(cfg["max_sequence_length"]),
        ),
    )

    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(
            cfg["beta"]
            if beta is None
            else beta
        ),
    }


def run_training(
    config_path: str,
    run_name: str,
    dataset_path: str | None = None,
    output_path: str | None = None,
    beta: float | None = None,
    max_examples: int | None = None,
):
    bundle = prepare_dpo_run(
        config_path,
        dataset_path,
        beta,
        max_examples,
    )

    cfg = bundle["cfg"]
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    loader = bundle["loader"]
    optimizer = bundle["optimizer"]
    beta = bundle["beta"]

    output = repo_path(
        output_path or cfg["standard_output"]
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = results_dir / f"{run_name}_train.jsonl"

    # Avoid mixing logs from an older run with the current run.
    if log_path.exists():
        log_path.unlink()

    device = next(model.parameters()).device

    grad_accum_steps = int(
        cfg["grad_accum_steps"]
    )
    max_grad_norm = float(
        cfg["max_grad_norm"]
    )
    epochs = int(
        cfg["epochs"]
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    optimizer_step = 0

    for epoch in range(epochs):

        for step, (chosen, rejected) in enumerate(loader):

            chosen = {
                k: v.to(device)
                for k, v in chosen.items()
            }

            rejected = {
                k: v.to(device)
                for k, v in rejected.items()
            }

            # Reference policy:
            # same base model, but with the LoRA adapter disabled.
            with torch.no_grad():
                with reference_mode(model):

                    ref_chosen_logp, _, _ = (
                        response_sequence_logprobs(
                            model,
                            chosen,
                        )
                    )

                    ref_rejected_logp, _, _ = (
                        response_sequence_logprobs(
                            model,
                            rejected,
                        )
                    )

            # Trainable policy:
            # LoRA adapter is enabled again here.
            policy_chosen_logp, _, _ = (
                response_sequence_logprobs(
                    model,
                    chosen,
                )
            )

            policy_rejected_logp, _, _ = (
                response_sequence_logprobs(
                    model,
                    rejected,
                )
            )

            loss, diagnostics = dpo_loss(
                policy_chosen_logp,
                policy_rejected_logp,
                ref_chosen_logp,
                ref_rejected_logp,
                beta,
            )

            scaled_loss = (
                loss / grad_accum_steps
            )

            scaled_loss.backward()

            should_step = (
                (step + 1) % grad_accum_steps == 0
                or
                (step + 1) == len(loader)
            )

            grad_norm = None

            if should_step:

                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters(model),
                        max_grad_norm,
                    )
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                optimizer_step += 1

            record = {
                "run_name": run_name,
                "epoch": epoch,
                "batch_step": step,
                "optimizer_step": optimizer_step,
                "loss": float(
                    loss.detach().item()
                ),
                "logit_mean": float(
                    diagnostics[
                        "logit_mean"
                    ].item()
                ),
                "policy_margin_mean": float(
                    diagnostics[
                        "policy_margin_mean"
                    ].item()
                ),
                "preference_accuracy": float(
                    diagnostics[
                        "preference_accuracy"
                    ].item()
                ),
                "beta": float(beta),
            }

            if grad_norm is not None:
                record["grad_norm"] = float(
                    grad_norm.detach().item()
                    if torch.is_tensor(grad_norm)
                    else grad_norm
                )

            append_jsonl(
                log_path,
                record,
            )

            if step % 10 == 0:
                print(
                    f"epoch={epoch} "
                    f"step={step}/{len(loader)} "
                    f"loss={loss.item():.4f} "
                    f"pref_acc="
                    f"{diagnostics['preference_accuracy'].item():.4f}"
                )

    model.save_pretrained(output)

    tokenizer.save_pretrained(output)

    print()
    print("Training complete.")
    print(f"Adapter saved to: {output}")
    print(f"Training log saved to: {log_path}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/dpo.yaml",
    )

    ap.add_argument(
        "--run-name",
        default="standard",
    )

    ap.add_argument(
        "--dataset",
    )

    ap.add_argument(
        "--output",
    )

    ap.add_argument(
        "--beta",
        type=float,
    )

    ap.add_argument(
        "--max-examples",
        type=int,
    )

    args = ap.parse_args()

    run_training(
        args.config,
        args.run_name,
        args.dataset,
        args.output,
        args.beta,
        args.max_examples,
    )


if __name__ == "__main__":
    main()