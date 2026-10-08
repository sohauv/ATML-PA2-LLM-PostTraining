from __future__ import annotations

import argparse

import torch
from torch.nn.utils.rnn import pad_sequence

from common.data import (
    load_yaml,
    prompt_messages,
    read_jsonl,
    repo_path,
)
from common.generation import (
    response_token_logprobs,
)
from common.logging_utils import save_json
from common.models import (
    load_policy,
    load_tokenizer,
)
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    shaped_rewards,
)


def load_cached_rollouts(path):
    rows = torch.load(
        repo_path(path),
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Expected a non-empty list in the supplied PPO rollout cache"
        )

    normalized = []

    for row in rows:
        row = dict(row)

        if (
            "old_logprobs" not in row
            and "old_policy_logprobs" in row
        ):
            row["old_logprobs"] = row[
                "old_policy_logprobs"
            ]

        if (
            "ref_logprobs" not in row
            and "reference_logprobs" in row
        ):
            row["ref_logprobs"] = row[
                "reference_logprobs"
            ]

        normalized.append(row)

    required = {
        "source_index",
        "response",
        "old_logprobs",
        "ref_logprobs",
        "values",
        "effective_terminal_reward",
        "response_tokens",
    }

    if not required.issubset(
        normalized[0]
    ):
        raise ValueError(
            "Unexpected PPO cache schema; "
            f"need at least {sorted(required)}"
        )

    return normalized


def reconstruct_cached_example(
    tokenizer,
    prompt_row,
    cache_row,
    max_prompt_length,
):
    messages = prompt_messages(
        prompt_row
    )

    rendered = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    prompt_enc = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    )

    prompt_ids = (
        prompt_enc["input_ids"][0]
        .long()
    )

    response_ids = tokenizer(
        cache_row["response"],
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].long()

    expected_tokens = int(
        cache_row["response_tokens"]
    )

    # The cached text excludes special tokens after decoding.
    # Add EOS back when the rollout terminated normally.
    if (
        bool(
            cache_row.get(
                "terminated_with_eos",
                False,
            )
        )
        and tokenizer.eos_token_id
        is not None
        and response_ids.numel()
        < expected_tokens
    ):
        response_ids = torch.cat(
            [
                response_ids,
                torch.tensor(
                    [
                        tokenizer.eos_token_id
                    ],
                    dtype=torch.long,
                ),
            ]
        )

    if (
        response_ids.numel()
        != expected_tokens
    ):
        raise ValueError(
            "Could not exactly reconstruct "
            f"cached response for prompt_id="
            f"{cache_row.get('prompt_id')}: "
            f"expected {expected_tokens} tokens, "
            f"got {response_ids.numel()}"
        )

    sequence = torch.cat(
        [
            prompt_ids,
            response_ids,
        ]
    )

    attention_mask = torch.ones_like(
        sequence,
        dtype=torch.long,
    )

    return {
        "sequence": sequence,
        "attention_mask": attention_mask,
        "prompt_width": int(
            prompt_ids.numel()
        ),
        "response_ids": response_ids,
    }


@torch.no_grad()
def compute_current_logprobs(
    policy,
    tokenizer,
    eval_rows,
    cache_rows,
    max_prompt_length,
):
    by_prompt_id = {
        row["prompt_id"]: row
        for row in eval_rows
    }

    by_source_index = {
        row["source_index"]: row
        for row in eval_rows
    }

    all_logprobs = []

    for i, cache_row in enumerate(
        cache_rows
    ):
        prompt_row = None

        prompt_id = cache_row.get(
            "prompt_id"
        )

        if (
            prompt_id is not None
            and prompt_id in by_prompt_id
        ):
            prompt_row = (
                by_prompt_id[prompt_id]
            )

        if prompt_row is None:
            source_index = cache_row[
                "source_index"
            ]

            prompt_row = (
                by_source_index.get(
                    source_index
                )
            )

        if prompt_row is None:
            raise KeyError(
                "Could not locate cached prompt "
                f"{i}: "
                f"prompt_id={prompt_id}, "
                f"source_index="
                f"{cache_row['source_index']}"
            )

        reconstructed = (
            reconstruct_cached_example(
                tokenizer,
                prompt_row,
                cache_row,
                max_prompt_length,
            )
        )

        device = next(
            policy.parameters()
        ).device

        sequence = (
            reconstructed["sequence"]
            .unsqueeze(0)
            .to(device)
        )

        attention_mask = (
            reconstructed[
                "attention_mask"
            ]
            .unsqueeze(0)
            .to(device)
        )

        response_ids = (
            reconstructed[
                "response_ids"
            ]
            .unsqueeze(0)
            .to(device)
        )

        current_logp, _ = (
            response_token_logprobs(
                policy,
                sequence,
                attention_mask,
                reconstructed[
                    "prompt_width"
                ],
                response_ids,
            )
        )

        all_logprobs.append(
            current_logp[
                0
            ].float().cpu()
        )

    return all_logprobs


def make_padded_cache_tensors(
    cache_rows,
    current_logprobs,
):
    old = [
        row["old_logprobs"]
        .float()
        .cpu()
        for row in cache_rows
    ]

    ref = [
        row["ref_logprobs"]
        .float()
        .cpu()
        for row in cache_rows
    ]

    values = [
        row["values"]
        .float()
        .cpu()
        for row in cache_rows
    ]

    lengths = torch.tensor(
        [
            int(row["response_tokens"])
            for row in cache_rows
        ],
        dtype=torch.long,
    )

    max_len = int(
        lengths.max().item()
    )

    old_pad = pad_sequence(
        old,
        batch_first=True,
        padding_value=0.0,
    )

    ref_pad = pad_sequence(
        ref,
        batch_first=True,
        padding_value=0.0,
    )

    value_pad = pad_sequence(
        values,
        batch_first=True,
        padding_value=0.0,
    )

    current_pad = pad_sequence(
        current_logprobs,
        batch_first=True,
        padding_value=0.0,
    )

    positions = torch.arange(
        max_len
    ).unsqueeze(0)

    mask = (
        positions
        < lengths.unsqueeze(1)
    ).float()

    terminal_rewards = (
        torch.tensor(
            [
                float(
                    row[
                        "effective_terminal_reward"
                    ]
                )
                for row in cache_rows
            ],
            dtype=torch.float32,
        )
    )

    return {
        "old": old_pad,
        "ref": ref_pad,
        "current": current_pad,
        "values": value_pad,
        "mask": mask,
        "terminal_rewards":
            terminal_rewards,
        "lengths": lengths,
    }


def clipping_metrics(
    new_logp,
    old_logp,
    advantages,
    mask,
    eps,
):
    ratio = torch.exp(
        new_logp - old_logp
    )

    unclipped = (
        ratio
        * advantages
    )

    clipped_ratio = ratio.clamp(
        1.0 - eps,
        1.0 + eps,
    )

    clipped_candidate = (
        clipped_ratio
        * advantages
    )

    clipped_objective = (
        torch.minimum(
            unclipped,
            clipped_candidate,
        )
    )

    valid = mask.bool()

    clipped_surrogate = float(
        clipped_objective[
            valid
        ]
        .mean()
        .item()
    )

    unclipped_surrogate = float(
        unclipped[
            valid
        ]
        .mean()
        .item()
    )

    outside = (
        (ratio < (1.0 - eps))
        | (ratio > (1.0 + eps))
    )

    clip_fraction = float(
        outside[
            valid
        ]
        .float()
        .mean()
        .item()
    )

    # A token is actually affected when PPO's
    # min() chooses the clipped candidate rather
    # than the original surrogate.
    affected = (
        clipped_objective
        < unclipped
    )

    affected_fraction = float(
        affected[
            valid
        ]
        .float()
        .mean()
        .item()
    )

    ratio_valid = ratio[
        valid
    ]

    return {
        "epsilon":
            float(eps),

        "unclipped_surrogate":
            unclipped_surrogate,

        "clipped_surrogate":
            clipped_surrogate,

        "policy_loss":
            -clipped_surrogate,

        "clip_fraction":
            clip_fraction,

        "affected_token_fraction":
            affected_fraction,

        "ratio_mean":
            float(
                ratio_valid
                .mean()
                .item()
            ),

        "ratio_std":
            float(
                ratio_valid
                .std(
                    unbiased=False
                )
                .item()
            ),

        "ratio_min":
            float(
                ratio_valid
                .min()
                .item()
            ),

        "ratio_max":
            float(
                ratio_valid
                .max()
                .item()
            ),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/ppo.yaml",
    )

    args = ap.parse_args()

    cfg = load_yaml(
        args.config
    )

    rows = load_cached_rollouts(
        cfg["cached_rollouts"]
    )

    eval_rows = read_jsonl(
        cfg["paths"]["rl_prompt_eval"]
    )

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )

    policy = load_policy(
        cfg,
        adapter_path=cfg[
            "paths"
        ][
            "ppo_midpoint_policy"
        ],
        trainable=False,
    )

    print(
        "Cached PPO rollouts:",
        len(rows),
    )

    print(
        "Required epsilon values:",
        cfg["clip_values"],
    )

    current_logprobs = (
        compute_current_logprobs(
            policy,
            tokenizer,
            eval_rows,
            rows,
            int(
                cfg[
                    "max_prompt_length"
                ]
            ),
        )
    )

    tensors = (
        make_padded_cache_tensors(
            rows,
            current_logprobs,
        )
    )

    old_logp = tensors["old"]
    ref_logp = tensors["ref"]
    new_logp = tensors["current"]
    values = tensors["values"]
    mask = tensors["mask"]

    rewards = shaped_rewards(
        tensors[
            "terminal_rewards"
        ],
        old_logp,
        ref_logp,
        mask,
        float(cfg["kl_beta"]),
    )

    advantages, returns = (
        compute_gae(
            rewards,
            values,
            mask,
            gamma=float(
                cfg["gamma"]
            ),
            lam=float(
                cfg["gae_lambda"]
            ),
        )
    )

    advantages = (
        normalize_advantages(
            advantages,
            mask,
        )
    )

    metrics = []

    for eps in cfg[
        "clip_values"
    ]:
        result = clipping_metrics(
            new_logp,
            old_logp,
            advantages,
            mask,
            float(eps),
        )

        metrics.append(
            result
        )

        print()
        print(
            f"epsilon={result['epsilon']}"
        )
        print(
            "  clipped surrogate:",
            result[
                "clipped_surrogate"
            ],
        )
        print(
            "  clip fraction:",
            result[
                "clip_fraction"
            ],
        )
        print(
            "  affected fraction:",
            result[
                "affected_token_fraction"
            ],
        )
        print(
            "  ratio range:",
            result["ratio_min"],
            "to",
            result["ratio_max"],
        )

    summary = {
        "num_rollouts":
            len(rows),

        "num_valid_tokens":
            int(
                mask.sum().item()
            ),

        "kl_beta":
            float(
                cfg["kl_beta"]
            ),

        "gamma":
            float(
                cfg["gamma"]
            ),

        "gae_lambda":
            float(
                cfg["gae_lambda"]
            ),

        "conditions":
            metrics,
    }

    out = (
        repo_path(
            cfg["results_dir"]
        )
        / "clipping_cached_summary.json"
    )

    save_json(
        out,
        summary,
    )

    print()
    print(
        f"Saved: {out}"
    )


if __name__ == "__main__":
    main()