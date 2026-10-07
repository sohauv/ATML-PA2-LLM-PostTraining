from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
)
from common.logging_utils import save_json, set_seed
from common.metrics import word_limit_compliance
from common.models import (
    clear_gpu,
    load_policy,
    load_tokenizer,
    reference_mode,
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


def filter_rows(rows, tokenizer, max_length):
    kept = []
    dropped = []

    for i, row in enumerate(rows):
        prompt = prompt_messages_from_preference(row)

        prompt_ids = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
        )

        if len(prompt_ids) >= max_length:
            dropped.append(
                {
                    "index": int(i),
                    "prompt_tokens": int(len(prompt_ids)),
                    "length_stratum": row.get("length_stratum"),
                }
            )
            continue

        kept.append(row)

    return kept, dropped


@torch.no_grad()
def evaluate_strata(
    policy,
    tokenizer,
    rows,
    cfg,
):
    device = next(policy.parameters()).device
    beta = float(cfg["beta"])

    strata = [
        "preferred_longer",
        "length_matched",
        "rejected_longer",
    ]

    results = {}

    for stratum in strata:
        stratum_rows = [
            row
            for row in rows
            if row["length_stratum"] == stratum
        ]

        loader = DataLoader(
            stratum_rows,
            batch_size=int(cfg["batch_size"]),
            shuffle=False,
            collate_fn=make_collate(
                tokenizer,
                int(cfg["max_sequence_length"]),
            ),
        )

        total = 0
        correct = 0.0
        loss_sum = 0.0

        for chosen, rejected in loader:
            chosen = {
                k: v.to(device)
                for k, v in chosen.items()
            }

            rejected = {
                k: v.to(device)
                for k, v in rejected.items()
            }

            with reference_mode(policy):
                ref_chosen_logp, _, _ = (
                    response_sequence_logprobs(
                        policy,
                        chosen,
                    )
                )

                ref_rejected_logp, _, _ = (
                    response_sequence_logprobs(
                        policy,
                        rejected,
                    )
                )

            policy_chosen_logp, _, _ = (
                response_sequence_logprobs(
                    policy,
                    chosen,
                )
            )

            policy_rejected_logp, _, _ = (
                response_sequence_logprobs(
                    policy,
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

            n = chosen["input_ids"].shape[0]

            total += n

            correct += (
                float(
                    diagnostics[
                        "preference_accuracy"
                    ].item()
                )
                * n
            )

            loss_sum += (
                float(loss.item())
                * n
            )

        results[stratum] = {
            "num_examples": int(total),
            "preference_accuracy": (
                float(correct / total)
                if total
                else float("nan")
            ),
            "dpo_loss": (
                float(loss_sum / total)
                if total
                else float("nan")
            ),
        }

    return results


@torch.no_grad()
def evaluate_word_limits(
    policy,
    tokenizer,
    rows,
    cfg,
):
    set_seed(int(cfg["seed"]))

    generation_cfg = cfg["generation"]

    batch_size = int(cfg["batch_size"])

    records = []

    for start in range(
        0,
        len(rows),
        batch_size,
    ):
        batch_rows = rows[
            start:start + batch_size
        ]

        prompts = [
            prompt_messages(row)
            for row in batch_rows
        ]

        generated = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=int(
                cfg["max_sequence_length"]
            ),
            max_new_tokens=int(
                cfg["max_generation_tokens"]
            ),
            temperature=float(
                generation_cfg["temperature"]
            ),
            top_p=float(
                generation_cfg["top_p"]
            ),
            do_sample=bool(
                generation_cfg["do_sample"]
            ),
        )

        for i, row in enumerate(batch_rows):
            response = generated["responses"][i]

            prompt_text = row["messages"][-1]["content"]

            compliance = word_limit_compliance(
                prompt_text,
                response,
            )

            records.append(
                {
                    "prompt_id": row["prompt_id"],
                    "prompt": prompt_text,
                    "response": response,
                    "response_length_tokens": int(
                        generated[
                            "response_lengths"
                        ][i]
                    ),
                    "word_limit_compliance": (
                        None
                        if compliance is None
                        else float(compliance)
                    ),
                }
            )

    lengths = np.asarray(
        [
            r["response_length_tokens"]
            for r in records
        ],
        dtype=float,
    )

    compliance_values = [
        r["word_limit_compliance"]
        for r in records
        if r["word_limit_compliance"]
        is not None
    ]

    summary = {
        "num_prompts": len(records),
        "mean_response_length_tokens":
            float(lengths.mean()),

        "std_response_length_tokens":
            float(lengths.std()),

        "word_limit_compliance_rate":
            float(np.mean(compliance_values))
            if compliance_values
            else float("nan"),

        "num_word_limit_prompts":
            len(compliance_values),
    }

    return summary, records


def evaluate_model(
    name,
    adapter,
    cfg,
    tokenizer,
    stratified_rows,
    word_limit_rows,
):
    print()
    print("=" * 80)
    print(f"Evaluating {name}")
    print("=" * 80)

    policy = load_policy(
        cfg,
        adapter_path=adapter,
        trainable=False,
    )

    stratum_metrics = evaluate_strata(
        policy,
        tokenizer,
        stratified_rows,
        cfg,
    )

    word_limit_summary, word_limit_records = (
        evaluate_word_limits(
            policy,
            tokenizer,
            word_limit_rows,
            cfg,
        )
    )

    clear_gpu(policy)

    return {
        "name": name,
        "adapter": adapter,
        "strata": stratum_metrics,
        "word_limit": word_limit_summary,
    }, word_limit_records


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/dpo.yaml",
    )

    ap.add_argument(
        "--standard-adapter",
        required=True,
    )

    ap.add_argument(
        "--balanced-adapter",
        required=True,
    )

    args = ap.parse_args()

    cfg = load_yaml(args.config)

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )

    stratified = read_jsonl(
        cfg["paths"]["dpo_length_eval"]
    )

    word_limit_rows = read_jsonl(
        cfg["paths"]["word_limit_prompts"]
    )

    max_length = int(
        cfg["max_sequence_length"]
    )

    stratified, dropped = filter_rows(
        stratified,
        tokenizer,
        max_length,
    )

    print(
        f"Length-stratified eval: "
        f"{len(stratified)} kept, "
        f"{len(dropped)} dropped "
        f"for prompt length >= {max_length}"
    )

    standard_summary, standard_records = (
        evaluate_model(
            "standard",
            args.standard_adapter,
            cfg,
            tokenizer,
            stratified,
            word_limit_rows,
        )
    )

    balanced_summary, balanced_records = (
        evaluate_model(
            "length_balanced",
            args.balanced_adapter,
            cfg,
            tokenizer,
            stratified,
            word_limit_rows,
        )
    )

    results_dir = repo_path(
        cfg["results_dir"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "max_sequence_length": max_length,
        "num_stratified_rows_kept":
            len(stratified),
        "num_stratified_rows_dropped":
            len(dropped),
        "standard":
            standard_summary,
        "length_balanced":
            balanced_summary,
    }

    save_json(
        results_dir
        / "length_confounded_summary.json",
        summary,
    )

    save_json(
        results_dir
        / "length_confounded_filter.json",
        {
            "dropped": dropped,
        },
    )

    write_jsonl(
        results_dir
        / "standard_word_limit_generations.jsonl",
        standard_records,
    )

    write_jsonl(
        results_dir
        / "length_balanced_word_limit_generations.jsonl",
        balanced_records,
    )

    print()
    print("=" * 80)
    print("Length-confounding evaluation complete")
    print("=" * 80)

    for model_name in [
        "standard",
        "length_balanced",
    ]:
        model_result = summary[
            model_name
        ]

        print()
        print(model_name)

        for stratum, metrics in (
            model_result["strata"].items()
        ):
            print(
                f"  {stratum}: "
                f"n={metrics['num_examples']} "
                f"pref_acc="
                f"{metrics['preference_accuracy']:.4f} "
                f"loss={metrics['dpo_loss']:.4f}"
            )

        wl = model_result["word_limit"]

        print(
            "  mean response length: "
            f"{wl['mean_response_length_tokens']:.2f}"
        )

        print(
            "  word-limit compliance: "
            f"{wl['word_limit_compliance_rate']:.4f}"
        )


if __name__ == "__main__":
    main()