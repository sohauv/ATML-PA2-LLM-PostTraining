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
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import save_json
from common.metrics import sampled_kl
from common.models import (
    load_policy,
    load_reward_model,
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


def load_evaluation_bundle(
    config_path: str,
    adapter: str,
):
    cfg = load_yaml(config_path)

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )

    policy = load_policy(
        cfg,
        adapter_path=adapter,
        trainable=False,
    )

    return {
        "cfg": cfg,
        "rows": read_jsonl(
            cfg["paths"]["dpo_standard_eval"]
        ),
        "tokenizer": tokenizer,
        "policy": policy,
    }


def filter_rows_for_length(
    rows,
    tokenizer,
    max_length,
):
    kept = []
    kept_indices = []
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
                }
            )
            continue

        kept.append(row)
        kept_indices.append(i)

    return kept, kept_indices, dropped


@torch.no_grad()
def evaluate_preference_pairs(
    policy,
    tokenizer,
    rows,
    cfg,
):
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=make_collate(
            tokenizer,
            int(cfg["max_sequence_length"]),
        ),
    )

    device = next(policy.parameters()).device

    total_examples = 0
    loss_sum = 0.0
    correct_sum = 0.0

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
            float(cfg["beta"]),
        )

        n = chosen["input_ids"].shape[0]

        total_examples += n
        loss_sum += float(loss.item()) * n
        correct_sum += (
            float(
                diagnostics[
                    "preference_accuracy"
                ].item()
            )
            * n
        )

    return {
        "dpo_loss":
            loss_sum / total_examples,

        "preference_accuracy":
            correct_sum / total_examples,

        "num_examples":
            total_examples,
    }


@torch.no_grad()
def evaluate_generation(
    policy,
    tokenizer,
    rows,
    cfg,
):
    generation_cfg = cfg["generation"]

    batch_size = int(cfg["batch_size"])

    all_records = []

    kl_numerator = 0.0
    kl_denominator = 0.0

    for start in range(
        0,
        len(rows),
        batch_size,
    ):
        batch_rows = rows[
            start:start + batch_size
        ]

        prompts = [
            prompt_messages_from_preference(row)
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

        policy_tok_logp, _ = (
            response_token_logprobs(
                policy,
                generated["sequences"],
                generated["attention_mask"],
                generated["prompt_width"],
                generated["response_ids"],
            )
        )

        with reference_mode(policy):
            ref_tok_logp, _ = (
                response_token_logprobs(
                    policy,
                    generated["sequences"],
                    generated["attention_mask"],
                    generated["prompt_width"],
                    generated["response_ids"],
                )
            )

        mask = generated[
            "response_mask"
        ]

        batch_kl = sampled_kl(
            policy_tok_logp,
            ref_tok_logp,
            mask,
        )

        valid_tokens = float(
            mask.sum().item()
        )

        kl_numerator += (
            float(batch_kl.item())
            * valid_tokens
        )

        kl_denominator += valid_tokens

        for i, response in enumerate(
            generated["responses"]
        ):
            all_records.append(
                {
                    "prompt_index":
                        start + i,

                    "response":
                        response,

                    "response_length":
                        int(
                            generated[
                                "response_lengths"
                            ][i]
                        ),

                    "terminated_with_eos":
                        bool(
                            generated[
                                "terminated_with_eos"
                            ][i]
                        ),

                    "truncated":
                        bool(
                            generated[
                                "truncated"
                            ][i]
                        ),
                }
            )

    lengths = np.asarray(
        [
            r["response_length"]
            for r in all_records
        ],
        dtype=float,
    )

    return all_records, {
        "kl_from_reference":
            kl_numerator
            / max(kl_denominator, 1.0),

        "mean_response_length":
            float(lengths.mean()),

        "std_response_length":
            float(lengths.std()),

        "median_response_length":
            float(np.median(lengths)),

        "iqr_response_length":
            float(
                np.percentile(lengths, 75)
                -
                np.percentile(lengths, 25)
            ),
    }


@torch.no_grad()
def evaluate_reward(
    cfg,
    rows,
    generation_records,
):
    reward_model, reward_tokenizer = (
        load_reward_model(cfg)
    )

    batch_size = int(cfg["batch_size"])

    scores = []

    for start in range(
        0,
        len(rows),
        batch_size,
    ):
        batch_rows = rows[
            start:start + batch_size
        ]

        prompts = [
            prompt_messages_from_preference(row)
            for row in batch_rows
        ]

        responses = [
            generation_records[i]["response"]
            for i in range(
                start,
                min(
                    start + batch_size,
                    len(generation_records),
                ),
            )
        ]

        batch_scores = score_reward_pairs(
            reward_model,
            reward_tokenizer,
            prompts,
            responses,
        )

        scores.extend(
            batch_scores.detach()
            .cpu()
            .tolist()
        )

    return {
        "reward_model_score_mean":
            float(np.mean(scores)),

        "reward_model_score_std":
            float(np.std(scores)),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/dpo.yaml",
    )

    ap.add_argument(
        "--adapter",
        required=True,
    )

    ap.add_argument(
        "--name",
        default="standard",
    )

    args = ap.parse_args()

    bundle = load_evaluation_bundle(
        args.config,
        args.adapter,
    )

    cfg = bundle["cfg"]
    rows = bundle["rows"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]

    max_length = int(
        cfg["max_sequence_length"]
    )

    kept_rows, kept_indices, dropped = (
        filter_rows_for_length(
            rows,
            tokenizer,
            max_length,
        )
    )

    print(
        f"DPO eval rows: "
        f"{len(rows)} total, "
        f"{len(kept_rows)} kept, "
        f"{len(dropped)} dropped "
        f"for prompt length >= {max_length}"
    )

    pair_metrics = (
        evaluate_preference_pairs(
            policy,
            tokenizer,
            kept_rows,
            cfg,
        )
    )

    generation_records, generation_metrics = (
        evaluate_generation(
            policy,
            tokenizer,
            kept_rows,
            cfg,
        )
    )

    reward_metrics = evaluate_reward(
        cfg,
        kept_rows,
        generation_records,
    )

    results_dir = repo_path(
        cfg["results_dir"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "name": args.name,

        "adapter":
            args.adapter,

        "beta":
            float(cfg["beta"]),

        "max_sequence_length":
            max_length,

        "num_original_eval_rows":
            len(rows),

        "num_eval_rows":
            len(kept_rows),

        "num_dropped_rows":
            len(dropped),

        **pair_metrics,
        **generation_metrics,
        **reward_metrics,
    }

    save_json(
        results_dir
        / f"{args.name}_summary.json",
        summary,
    )

    save_json(
        results_dir
        / f"{args.name}_eval_filter.json",
        {
            "kept_indices":
                kept_indices,

            "dropped":
                dropped,
        },
    )

    write_jsonl(
        results_dir
        / f"{args.name}_generations.jsonl",
        generation_records,
    )

    print()
    print("Evaluation complete.")
    print()

    for key, value in summary.items():
        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()