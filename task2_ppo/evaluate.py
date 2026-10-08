from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common.data import (
    load_yaml,
    prompt_messages,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import save_json, set_seed
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)


def load_evaluation_bundle(
    config_path: str,
    adapter: str,
):
    cfg = load_yaml(config_path)

    return {
        "cfg": cfg,
        "rows": read_jsonl(
            cfg["paths"]["rl_prompt_eval"]
        ),
        "tokenizer": load_tokenizer(
            cfg["base_model"]
        ),
        "policy": load_policy(
            cfg,
            adapter_path=adapter,
            trainable=False,
        ),
        "reward": load_reward_model(cfg),
    }


@torch.no_grad()
def token_entropies(logits):
    """
    Exact categorical entropy for each response position.

    logits: [batch, response_steps, vocab]
    returns: [batch, response_steps]
    """
    batch, steps, _ = logits.shape

    out = torch.empty(
        (batch, steps),
        device=logits.device,
        dtype=torch.float32,
    )

    # Avoid materializing float32 log-probs for every
    # response position simultaneously.
    chunk_size = 32

    for start in range(
        0,
        steps,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            steps,
        )

        chunk = logits[
            :,
            start:end,
            :,
        ].float()

        logp = F.log_softmax(
            chunk,
            dim=-1,
        )

        entropy = -(
            logp.exp() * logp
        ).sum(dim=-1)

        out[:, start:end] = entropy

    return out


@torch.no_grad()
def evaluate(
    config_path: str,
    adapter: str,
    name: str,
    batch_size: int = 1,
    limit: int | None = None,
):
    bundle = load_evaluation_bundle(
        config_path,
        adapter,
    )

    cfg = bundle["cfg"]
    rows = bundle["rows"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]

    reward_model, reward_tokenizer = (
        bundle["reward"]
    )

    if limit is not None:
        rows = rows[: int(limit)]

    seed = int(cfg["seed"])

    set_seed(seed)

    results_dir = repo_path(
        cfg["results_dir"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generations_path = (
        results_dir
        / f"{name}_generations.jsonl"
    )

    summary_path = (
        results_dir
        / f"{name}_eval_summary.json"
    )

    all_records = []

    total_kl_sum = 0.0
    total_entropy_sum = 0.0
    total_tokens = 0.0

    reward_values = []
    response_lengths = []

    print(
        f"PPO evaluation name={name} "
        f"examples={len(rows)} "
        f"batch_size={batch_size}"
    )

    for start in range(
        0,
        len(rows),
        batch_size,
    ):
        batch_rows = rows[
            start : start + batch_size
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
                cfg["max_prompt_length"]
            ),
            max_new_tokens=int(
                cfg[
                    "eval_max_response_length"
                ]
            ),
            temperature=float(
                cfg["generation"][
                    "temperature"
                ]
            ),
            top_p=float(
                cfg["generation"][
                    "top_p"
                ]
            ),
            do_sample=bool(
                cfg["generation"][
                    "do_sample"
                ]
            ),
        )

        # batch_generate uses inference_mode.
        # Clone before subsequent model computations.
        sequences = generated[
            "sequences"
        ].clone()

        attention_mask = generated[
            "attention_mask"
        ].clone()

        response_ids = generated[
            "response_ids"
        ].clone()

        response_mask = generated[
            "response_mask"
        ].clone()

        prompt_width = int(
            generated["prompt_width"]
        )

        policy_logp, policy_logits = (
            response_token_logprobs(
                policy,
                sequences,
                attention_mask,
                prompt_width,
                response_ids,
            )
        )

        entropy_tokens = (
            token_entropies(
                policy_logits
            )
        )

        with reference_mode(policy):
            ref_logp, _ = (
                response_token_logprobs(
                    policy,
                    sequences,
                    attention_mask,
                    prompt_width,
                    response_ids,
                )
            )

        rewards = score_reward_pairs(
            reward_model,
            reward_tokenizer,
            prompts,
            generated["responses"],
            max_length=int(
                cfg["reward_max_length"]
            ),
        )

        mask = response_mask.float()

        kl_tokens = (
            policy_logp - ref_logp
        )

        batch_token_count = float(
            mask.sum().item()
        )

        total_tokens += (
            batch_token_count
        )

        total_kl_sum += float(
            (
                kl_tokens * mask
            ).sum().item()
        )

        total_entropy_sum += float(
            (
                entropy_tokens * mask
            ).sum().item()
        )

        for i, row in enumerate(
            batch_rows
        ):
            valid_tokens = float(
                mask[i].sum().item()
            )

            if valid_tokens > 0:
                example_kl = float(
                    (
                        kl_tokens[i]
                        * mask[i]
                    ).sum().item()
                    / valid_tokens
                )

                example_entropy = float(
                    (
                        entropy_tokens[i]
                        * mask[i]
                    ).sum().item()
                    / valid_tokens
                )
            else:
                example_kl = 0.0
                example_entropy = 0.0

            reward = float(
                rewards[i].item()
            )

            response_length = int(
                generated[
                    "response_lengths"
                ][i]
            )

            reward_values.append(
                reward
            )

            response_lengths.append(
                response_length
            )

            record = {
                "eval_index":
                    start + i,

                "source_split":
                    row.get(
                        "source_split"
                    ),

                "source_index":
                    row.get(
                        "source_index"
                    ),

                "prompt_id":
                    row.get(
                        "prompt_id"
                    ),

                "prompt":
                    row.get(
                        "prompt"
                    ),

                "response":
                    generated[
                        "responses"
                    ][i],

                "reward_model_score":
                    reward,

                "kl_from_reference":
                    example_kl,

                "entropy":
                    example_entropy,

                "response_length":
                    response_length,

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

            all_records.append(
                record
            )

        print(
            f"evaluated "
            f"{min(start + len(batch_rows), len(rows))}"
            f"/{len(rows)}"
        )

        del (
            sequences,
            attention_mask,
            response_ids,
            response_mask,
            policy_logp,
            policy_logits,
            ref_logp,
            entropy_tokens,
            rewards,
        )

    lengths = np.asarray(
        response_lengths,
        dtype=float,
    )

    reward_array = np.asarray(
        reward_values,
        dtype=float,
    )

    terminated = np.asarray(
        [
            float(
                r[
                    "terminated_with_eos"
                ]
            )
            for r in all_records
        ]
    )

    truncated = np.asarray(
        [
            float(r["truncated"])
            for r in all_records
        ]
    )

    if total_tokens > 0:
        mean_kl = (
            total_kl_sum
            / total_tokens
        )

        mean_entropy = (
            total_entropy_sum
            / total_tokens
        )
    else:
        mean_kl = 0.0
        mean_entropy = 0.0

    q25, q75 = np.percentile(
        lengths,
        [25, 75],
    )

    summary = {
        "name":
            name,

        "adapter":
            adapter,

        "seed":
            seed,

        "num_examples":
            len(all_records),

        "max_prompt_length":
            int(
                cfg[
                    "max_prompt_length"
                ]
            ),

        "max_response_length":
            int(
                cfg[
                    "eval_max_response_length"
                ]
            ),

        "temperature":
            float(
                cfg["generation"][
                    "temperature"
                ]
            ),

        "top_p":
            float(
                cfg["generation"][
                    "top_p"
                ]
            ),

        "do_sample":
            bool(
                cfg["generation"][
                    "do_sample"
                ]
            ),

        "reward_model_score_mean":
            float(
                reward_array.mean()
            ),

        "reward_model_score_std":
            float(
                reward_array.std()
            ),

        "kl_from_reference":
            float(mean_kl),

        "entropy":
            float(mean_entropy),

        "mean_response_length":
            float(
                lengths.mean()
            ),

        "std_response_length":
            float(
                lengths.std()
            ),

        "median_response_length":
            float(
                np.median(lengths)
            ),

        "iqr_response_length":
            float(q75 - q25),

        "terminated_with_eos_fraction":
            float(
                terminated.mean()
            ),

        "truncated_fraction":
            float(
                truncated.mean()
            ),
    }

    write_jsonl(
        generations_path,
        all_records,
    )

    save_json(
        summary_path,
        summary,
    )

    print()
    print("Evaluation complete.")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print(
        f"Generations: "
        f"{generations_path}"
    )

    print(
        f"Summary: "
        f"{summary_path}"
    )

    return summary


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/ppo.yaml",
    )

    ap.add_argument(
        "--adapter",
        required=True,
    )

    ap.add_argument(
        "--name",
        default="standard",
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--limit",
        type=int,
    )

    args = ap.parse_args()

    evaluate(
        args.config,
        args.adapter,
        args.name,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()