from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from common.data import (
    load_yaml,
    prompt_messages,
    read_jsonl,
    repo_path,
)
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import (
    append_jsonl,
    save_json,
    set_seed,
)
from common.metrics import sampled_kl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task3_grpo.grpo import (
    grpo_policy_loss,
    group_relative_advantages,
    mask_truncated_sequences,
)


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])

    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )

    reward_model, reward_tokenizer = load_reward_model(cfg)

    prompts = read_jsonl(
        cfg["paths"]["rl_prompt_train"]
    )

    # T4 + fp16: use the same stable Adam epsilon that worked for PPO.
    optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["learning_rate"]),
        eps=1e-5,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def select_prompt_rows(
    rows: list[dict],
    update_idx: int,
    prompts_per_update: int,
) -> list[dict]:
    start = update_idx * prompts_per_update
    return [
        rows[(start + offset) % len(rows)]
        for offset in range(prompts_per_update)
    ]


def expand_prompt_groups(
    rows: list[dict],
    num_generations: int,
):
    """Repeat each prompt K times and return matching integer group IDs."""
    expanded_prompts = []
    group_ids = []

    for group_id, row in enumerate(rows):
        messages = prompt_messages(row)

        for _ in range(num_generations):
            expanded_prompts.append(messages)
            group_ids.append(group_id)

    return expanded_prompts, torch.tensor(
        group_ids,
        dtype=torch.long,
    )


@torch.no_grad()
def categorical_entropy(
    logits: torch.Tensor,
    mask: torch.Tensor,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Exact mean categorical entropy over valid response tokens."""
    total = torch.zeros(
        (),
        device=logits.device,
        dtype=torch.float32,
    )

    count = (
        mask.float()
        .sum()
        .clamp_min(1.0)
    )

    for start in range(
        0,
        logits.shape[1],
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            logits.shape[1],
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
            logp.exp()
            * logp
        ).sum(dim=-1)

        total += (
            entropy
            * mask[
                :,
                start:end,
            ].float()
        ).sum()

    return total / count


def _finite_float(value) -> float:
    if torch.is_tensor(value):
        value = (
            value.detach()
            .float()
            .item()
        )

    value = float(value)

    if not math.isfinite(value):
        raise FloatingPointError(
            f"Non-finite GRPO diagnostic encountered: {value}"
        )

    return value


def _safe_corr(x, y) -> float:
    x = np.asarray(
        x,
        dtype=float,
    )
    y = np.asarray(
        y,
        dtype=float,
    )

    if (
        len(x) < 2
        or np.std(x) == 0
        or np.std(y) == 0
    ):
        return float("nan")

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def length_conditioned_signal_stats(
    seq_adv: torch.Tensor,
    train_mask: torch.Tensor,
    loss_type: str,
    max_completion_length: int,
) -> dict:
    """Length-conditioned optimization statistic for the normalization study.

    At rho=1, canonical GRPO gives an absolute per-sequence signal |A|
    independent of realized length, while Dr-GRPO gives
    |A| * T / T_max. Masked truncated completions contribute zero.

    Valid completions are also rank-split into a shorter and longer half.
    """
    lengths = (
        train_mask.sum(dim=-1)
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    advantages = (
        seq_adv.detach()
        .float()
        .abs()
        .cpu()
        .numpy()
    )

    if loss_type == "grpo":
        signals = np.where(
            lengths > 0,
            advantages,
            0.0,
        )

    elif loss_type == "dr_grpo":
        signals = (
            advantages
            * (
                lengths
                / float(
                    max_completion_length
                )
            )
        )

    else:
        raise ValueError(loss_type)

    valid = np.flatnonzero(
        lengths > 0
    )

    out = {
        "length_signal_corr":
            float("nan"),

        "short_mean_length":
            float("nan"),

        "long_mean_length":
            float("nan"),

        "short_abs_sequence_signal":
            float("nan"),

        "long_abs_sequence_signal":
            float("nan"),

        "long_short_signal_ratio":
            float("nan"),

        "sequence_train_lengths":
            lengths.astype(int).tolist(),

        "sequence_abs_signals":
            signals.astype(float).tolist(),
    }

    if len(valid) == 0:
        return out

    valid_lengths = lengths[
        valid
    ]
    valid_signals = signals[
        valid
    ]

    out[
        "length_signal_corr"
    ] = _safe_corr(
        valid_lengths,
        valid_signals,
    )

    order = valid[
        np.argsort(
            lengths[valid],
            kind="stable",
        )
    ]

    split = max(
        1,
        len(order) // 2,
    )

    short_idx = order[:split]
    long_idx = order[split:]

    if len(long_idx) == 0:
        long_idx = short_idx

    short_signal = float(
        np.mean(
            signals[short_idx]
        )
    )

    long_signal = float(
        np.mean(
            signals[long_idx]
        )
    )

    out.update(
        {
            "short_mean_length":
                float(
                    np.mean(
                        lengths[
                            short_idx
                        ]
                    )
                ),

            "long_mean_length":
                float(
                    np.mean(
                        lengths[
                            long_idx
                        ]
                    )
                ),

            "short_abs_sequence_signal":
                short_signal,

            "long_abs_sequence_signal":
                long_signal,

            "long_short_signal_ratio":
                (
                    float(
                        long_signal
                        / short_signal
                    )
                    if short_signal > 0
                    else float("nan")
                ),
        }
    )

    return out


def run_grpo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    loss_type: str = "grpo",
    run_name: str = "standard",
):
    bundle = prepare_grpo_continuation(
        config_path
    )

    cfg = bundle["cfg"]

    if updates is not None:
        cfg["updates"] = int(
            updates
        )

    tokenizer = bundle[
        "tokenizer"
    ]
    policy = bundle[
        "policy"
    ]
    reward_model = bundle[
        "reward_model"
    ]
    reward_tokenizer = bundle[
        "reward_tokenizer"
    ]
    prompt_rows = bundle[
        "prompt_rows"
    ]
    optimizer = bundle[
        "optimizer"
    ]

    num_updates = int(
        cfg["updates"]
    )

    prompts_per_update = int(
        cfg["prompts_per_update"]
    )

    num_generations = int(
        cfg["num_generations"]
    )

    policy_epochs = int(
        cfg.get(
            "policy_epochs",
            1,
        )
    )

    clip_eps = float(
        cfg["clip_epsilon"]
    )

    beta_kl = float(
        cfg["kl_beta"]
    )

    max_grad_norm = float(
        cfg["max_grad_norm"]
    )

    max_prompt_length = int(
        cfg["max_prompt_length"]
    )

    max_completion_length = int(
        cfg[
            "max_completion_length"
        ]
    )

    mask_truncated = bool(
        cfg.get(
            "mask_truncated_completions",
            True,
        )
    )

    if loss_type not in {
        "grpo",
        "dr_grpo",
    }:
        raise ValueError(
            "loss_type must be "
            "'grpo' or 'dr_grpo'"
        )

    out = repo_path(
        output or cfg["output"]
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dir = repo_path(
        cfg["results_dir"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        results_dir
        / f"{run_name}_train.jsonl"
    )

    summary_path = (
        results_dir
        / f"{run_name}_summary.json"
    )

    if log_path.exists():
        log_path.unlink()

    set_seed(
        int(cfg["seed"])
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    run_start = (
        time.perf_counter()
    )

    total_generated_tokens = 0
    total_train_tokens = 0
    used_prompt_ids = []
    used_source_indices = []

    print(
        f"GRPO run={run_name} "
        f"loss_type={loss_type} "
        f"updates={num_updates} "
        f"K={num_generations} "
        f"policy_epochs={policy_epochs} "
        f"clip_epsilon={clip_eps} "
        f"kl_beta={beta_kl}"
    )

    for update_idx in range(
        num_updates
    ):
        update_start = (
            time.perf_counter()
        )

        rows = select_prompt_rows(
            prompt_rows,
            update_idx,
            prompts_per_update,
        )

        prompt_ids = [
            row.get("prompt_id")
            for row in rows
        ]

        source_indices = [
            row.get("source_index")
            for row in rows
        ]

        used_prompt_ids.extend(
            prompt_ids
        )

        used_source_indices.extend(
            source_indices
        )

        prompts, group_ids_cpu = (
            expand_prompt_groups(
                rows,
                num_generations,
            )
        )

        generated = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=
                max_prompt_length,
            max_new_tokens=
                max_completion_length,
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

        # batch_generate uses inference_mode().
        # Clone before autograd computations.
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
        ].clone().float()

        prompt_width = int(
            generated[
                "prompt_width"
            ]
        )

        device = sequences.device

        group_ids = (
            group_ids_cpu
            .to(device)
        )

        generated_tokens = int(
            response_mask
            .sum()
            .item()
        )

        total_generated_tokens += (
            generated_tokens
        )

        if mask_truncated:
            train_mask = (
                mask_truncated_sequences(
                    response_mask,
                    generated[
                        "truncated"
                    ],
                )
            )
        else:
            train_mask = (
                response_mask.clone()
            )

        train_tokens = int(
            train_mask
            .sum()
            .item()
        )

        total_train_tokens += (
            train_tokens
        )

        # Behavior-policy and frozen-reference
        # probabilities on the sampled completions.
        policy_was_training = (
            policy.training
        )

        policy.eval()

        with torch.no_grad():
            old_logp, old_logits = (
                response_token_logprobs(
                    policy,
                    sequences,
                    attention_mask,
                    prompt_width,
                    response_ids,
                )
            )

            rollout_entropy = (
                categorical_entropy(
                    old_logits,
                    response_mask,
                )
            )

            with reference_mode(
                policy
            ):
                ref_logp, _ = (
                    response_token_logprobs(
                        policy,
                        sequences,
                        attention_mask,
                        prompt_width,
                        response_ids,
                    )
                )

        if policy_was_training:
            policy.train()

        old_logp = (
            old_logp.detach()
        )

        ref_logp = (
            ref_logp.detach()
        )

        with torch.no_grad():
            rewards = (
                score_reward_pairs(
                    reward_model,
                    reward_tokenizer,
                    prompts,
                    generated[
                        "responses"
                    ],
                    max_length=int(
                        cfg.get(
                            "reward_max_length",
                            max_prompt_length
                            + max_completion_length
                            + 256,
                        )
                    ),
                )
                .to(device)
                .float()
            )

        seq_adv = (
            group_relative_advantages(
                rewards,
                group_ids,
            )
            .detach()
        )

        # Raw within-prompt reward std determines
        # whether a group is informative.
        group_reward_stds = []

        for group_id in torch.unique(
            group_ids
        ):
            vals = rewards[
                group_ids == group_id
            ]

            group_reward_stds.append(
                vals.std(
                    unbiased=False
                )
            )

        group_reward_stds_t = (
            torch.stack(
                group_reward_stds
            )
        )

        uninformative = (
            group_reward_stds_t
            <= 1e-6
        )

        signal_stats = (
            length_conditioned_signal_stats(
                seq_adv,
                train_mask,
                loss_type,
                max_completion_length,
            )
        )

        epoch_losses = []
        epoch_policy_terms = []
        epoch_kl_penalties = []
        epoch_clip_fractions = []
        epoch_grad_norms = []
        epoch_ratio_means = []

        for _ in range(
            policy_epochs
        ):
            policy.train()

            optimizer.zero_grad(
                set_to_none=True
            )

            new_logp, _ = (
                response_token_logprobs(
                    policy,
                    sequences,
                    attention_mask,
                    prompt_width,
                    response_ids,
                )
            )

            loss, diagnostics = (
                grpo_policy_loss(
                    new_logp,
                    old_logp,
                    seq_adv,
                    train_mask,
                    ref_logp,
                    eps=clip_eps,
                    beta=beta_kl,
                    loss_type=
                        loss_type,
                    max_completion_length=
                        max_completion_length,
                )
            )

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    "Non-finite GRPO loss "
                    f"at update "
                    f"{update_idx + 1}: "
                    f"{loss.item()}"
                )

            loss.backward()

            grad_norm = (
                clip_grad_norm_(
                    trainable_parameters(
                        policy
                    ),
                    max_grad_norm,
                )
            )

            if not torch.isfinite(
                torch.as_tensor(
                    grad_norm
                )
            ):
                raise FloatingPointError(
                    "Non-finite GRPO "
                    "gradient norm at "
                    f"update "
                    f"{update_idx + 1}"
                )

            optimizer.step()

            epoch_losses.append(
                _finite_float(
                    loss
                )
            )

            epoch_policy_terms.append(
                _finite_float(
                    diagnostics[
                        "policy_term"
                    ]
                )
            )

            epoch_kl_penalties.append(
                _finite_float(
                    diagnostics[
                        "sampled_kl"
                    ]
                )
            )

            epoch_clip_fractions.append(
                _finite_float(
                    diagnostics[
                        "clip_fraction"
                    ]
                )
            )

            epoch_grad_norms.append(
                _finite_float(
                    grad_norm
                )
            )

            epoch_ratio_means.append(
                _finite_float(
                    diagnostics[
                        "ratio_mean"
                    ]
                )
            )

        # Required reporting convention:
        # sampled response-token
        # log p_policy - log p_reference.
        rollout_kl = sampled_kl(
            old_logp,
            ref_logp,
            response_mask,
        )

        train_kl = sampled_kl(
            old_logp,
            ref_logp,
            train_mask,
        )

        response_lengths = (
            response_mask
            .sum(dim=-1)
            .detach()
            .float()
        )

        update_seconds = (
            time.perf_counter()
            - update_start
        )

        record = {
            "run_name":
                run_name,

            "loss_type":
                loss_type,

            "update":
                int(
                    update_idx + 1
                ),

            "seed":
                int(
                    cfg["seed"]
                ),

            "prompt_ids":
                prompt_ids,

            "source_indices":
                source_indices,

            "num_generations":
                num_generations,

            "policy_epochs":
                policy_epochs,

            "clip_epsilon":
                clip_eps,

            "kl_beta":
                beta_kl,

            "generated_tokens":
                generated_tokens,

            "train_tokens":
                train_tokens,

            "reward_mean":
                float(
                    rewards
                    .mean()
                    .item()
                ),

            "reward_std":
                float(
                    rewards
                    .std(
                        unbiased=False
                    )
                    .item()
                ),

            "kl_from_reference":
                float(
                    rollout_kl.item()
                ),

            "train_kl_from_reference":
                float(
                    train_kl.item()
                ),

            "group_reward_std_mean":
                float(
                    group_reward_stds_t
                    .mean()
                    .item()
                ),

            "group_reward_stds":
                [
                    float(x)
                    for x
                    in (
                        group_reward_stds_t
                        .detach()
                        .cpu()
                        .tolist()
                    )
                ],

            "uninformative_group_fraction":
                float(
                    uninformative
                    .float()
                    .mean()
                    .item()
                ),

            "informative_group_fraction":
                float(
                    (~uninformative)
                    .float()
                    .mean()
                    .item()
                ),

            "policy_loss":
                float(
                    np.mean(
                        epoch_losses
                    )
                ),

            "policy_term":
                float(
                    np.mean(
                        epoch_policy_terms
                    )
                ),

            "loss_kl_penalty":
                float(
                    np.mean(
                        epoch_kl_penalties
                    )
                ),

            "clip_fraction":
                float(
                    np.mean(
                        epoch_clip_fractions
                    )
                ),

            "ratio_mean":
                float(
                    np.mean(
                        epoch_ratio_means
                    )
                ),

            "policy_grad_norm":
                float(
                    np.mean(
                        epoch_grad_norms
                    )
                ),

            "entropy":
                float(
                    rollout_entropy.item()
                ),

            "response_length_mean":
                float(
                    response_lengths
                    .mean()
                    .item()
                ),

            "response_length_std":
                float(
                    response_lengths
                    .std(
                        unbiased=False
                    )
                    .item()
                ),

            "truncated_fraction":
                float(
                    np.mean(
                        [
                            float(x)
                            for x
                            in generated[
                                "truncated"
                            ]
                        ]
                    )
                ),

            "masked_completion_fraction":
                float(
                    np.mean(
                        [
                            float(
                                mask_truncated
                                and bool(x)
                            )
                            for x
                            in generated[
                                "truncated"
                            ]
                        ]
                    )
                ),

            # Per-completion records preserve exact evidence
            # for later qualitative/normalization analysis.
            "rewards":
                rewards
                .detach()
                .cpu()
                .tolist(),

            "advantages":
                seq_adv
                .detach()
                .cpu()
                .tolist(),

            "group_ids":
                group_ids
                .detach()
                .cpu()
                .tolist(),

            "response_lengths":
                [
                    int(x)
                    for x
                    in generated[
                        "response_lengths"
                    ]
                ],

            "terminated_with_eos":
                [
                    bool(x)
                    for x
                    in generated[
                        "terminated_with_eos"
                    ]
                ],

            "truncated":
                [
                    bool(x)
                    for x
                    in generated[
                        "truncated"
                    ]
                ],

            "responses":
                list(
                    generated[
                        "responses"
                    ]
                ),

            "update_seconds":
                float(
                    update_seconds
                ),

            **signal_stats,
        }

        append_jsonl(
            log_path,
            record,
        )

        print(
            f"update="
            f"{update_idx + 1}/"
            f"{num_updates} "
            f"reward="
            f"{record['reward_mean']:.4f} "
            f"kl="
            f"{record['kl_from_reference']:.5f} "
            f"group_std="
            f"{record['group_reward_std_mean']:.4f} "
            f"uninform="
            f"{record['uninformative_group_fraction']:.3f} "
            f"loss="
            f"{record['policy_loss']:.4f} "
            f"grad="
            f"{record['policy_grad_norm']:.4f} "
            f"entropy="
            f"{record['entropy']:.4f} "
            f"len="
            f"{record['response_length_mean']:.1f} "
            f"trunc="
            f"{record['truncated_fraction']:.2f}"
        )

        del (
            sequences,
            attention_mask,
            response_ids,
            response_mask,
            train_mask,
            old_logp,
            old_logits,
            ref_logp,
            rewards,
            seq_adv,
        )

    total_seconds = (
        time.perf_counter()
        - run_start
    )

    peak_vram_bytes = 0

    if torch.cuda.is_available():
        peak_vram_bytes = int(
            torch.cuda
            .max_memory_allocated()
        )

    policy.save_pretrained(
        out
    )

    tokenizer.save_pretrained(
        out
    )

    summary = {
        "run_name":
            run_name,

        "loss_type":
            loss_type,

        "seed":
            int(
                cfg["seed"]
            ),

        "updates":
            num_updates,

        "prompts_per_update":
            prompts_per_update,

        "num_generations":
            num_generations,

        "policy_epochs":
            policy_epochs,

        "learning_rate":
            float(
                cfg["learning_rate"]
            ),

        "clip_epsilon":
            clip_eps,

        "kl_beta":
            beta_kl,

        "max_prompt_length":
            max_prompt_length,

        "max_completion_length":
            max_completion_length,

        "mask_truncated_completions":
            mask_truncated,

        "total_generated_tokens":
            int(
                total_generated_tokens
            ),

        "total_train_tokens":
            int(
                total_train_tokens
            ),

        "prompt_ids":
            used_prompt_ids,

        "source_indices":
            used_source_indices,

        "total_wall_seconds":
            float(
                total_seconds
            ),

        "peak_vram_bytes":
            peak_vram_bytes,

        "peak_vram_gb":
            float(
                peak_vram_bytes
                / (1024 ** 3)
            ),

        "policy_output":
            str(out),

        "train_log":
            str(log_path),
    }

    save_json(
        summary_path,
        summary,
    )

    print()
    print(
        "GRPO continuation complete."
    )
    print(
        f"Policy: {out}"
    )
    print(
        f"Train log: {log_path}"
    )
    print(
        f"Summary: {summary_path}"
    )
    print(
        f"Wall time: "
        f"{total_seconds:.1f}s"
    )
    print(
        f"Peak VRAM: "
        f"{summary['peak_vram_gb']:.2f} GB"
    )

    return summary


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/grpo.yaml",
    )

    ap.add_argument(
        "--output"
    )

    ap.add_argument(
        "--updates",
        type=int,
    )

    ap.add_argument(
        "--loss-type",
        choices=[
            "grpo",
            "dr_grpo",
        ],
        default="grpo",
    )

    ap.add_argument(
        "--run-name",
        default="standard",
    )

    args = ap.parse_args()

    run_grpo(
        args.config,
        args.output,
        args.updates,
        args.loss_type,
        args.run_name,
    )


if __name__ == "__main__":
    main()
