from __future__ import annotations

import argparse
import time

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
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )

    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"][
            "ppo_midpoint_policy"
        ],
        trainable=True,
    )

    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get(
            "value_train_mode",
            "head_only",
        ),
    )

    reward_model, reward_tokenizer = (
        load_reward_model(cfg)
    )

    prompts = read_jsonl(
        cfg["paths"]["rl_prompt_train"]
    )

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(
            cfg["policy_learning_rate"]
        ),
    )

    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(
                cfg[
                    "value_lora_learning_rate"
                ]
            ),
            head_lr=float(
                cfg[
                    "value_head_learning_rate"
                ]
            ),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def select_prompt_rows(
    rows,
    update_idx: int,
    prompts_per_update: int,
):
    start = (
        update_idx
        * prompts_per_update
    )

    return [
        rows[
            (start + offset)
            % len(rows)
        ]
        for offset in range(
            prompts_per_update
        )
    ]


def response_state_values(
    value_model,
    sequences,
    attention_mask,
    prompt_width: int,
    response_steps: int,
):
    all_values = token_values(
        value_model,
        sequences,
        attention_mask,
    )

    start = prompt_width - 1
    end = start + response_steps

    return all_values[
        :,
        start:end,
    ]


@torch.no_grad()
def categorical_entropy(
    logits,
    mask,
    chunk_size: int = 32,
):
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


def policy_grad_parameters(policy):
    return [
        parameter
        for parameter
        in policy.parameters()
        if parameter.requires_grad
    ]


def value_grad_parameters(
    value_model,
):
    return [
        parameter
        for parameter
        in value_model.parameters()
        if parameter.requires_grad
    ]


def run_ppo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    clip_epsilon: float | None = None,
    kl_beta: float | None = None,
    run_name: str = "standard",
):
    bundle = prepare_ppo_continuation(
        config_path
    )

    cfg = bundle["cfg"]

    if updates is not None:
        cfg["updates"] = int(
            updates
        )

    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(
            clip_epsilon
        )

    if kl_beta is not None:
        cfg["kl_beta"] = float(
            kl_beta
        )

    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    value_model = bundle[
        "value_model"
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

    policy_optimizer = bundle[
        "policy_optimizer"
    ]
    value_optimizer = bundle[
        "value_optimizer"
    ]

    num_updates = int(
        cfg["updates"]
    )

    ppo_epochs = int(
        cfg["ppo_epochs"]
    )

    prompts_per_update = int(
        cfg["prompts_per_update"]
    )

    clip_eps = float(
        cfg["clip_epsilon"]
    )

    beta_kl = float(
        cfg["kl_beta"]
    )

    gamma = float(
        cfg["gamma"]
    )

    gae_lambda = float(
        cfg["gae_lambda"]
    )

    value_coef = float(
        cfg["value_coef"]
    )

    max_grad_norm = float(
        cfg["max_grad_norm"]
    )

    missing_eos_penalty = float(
        cfg["missing_eos_penalty"]
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

    start_time = (
        time.perf_counter()
    )

    total_generated_tokens = 0
    used_prompt_ids = []

    print(
        f"PPO run={run_name} "
        f"updates={num_updates} "
        f"ppo_epochs={ppo_epochs} "
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

        prompts = [
            prompt_messages(row)
            for row in rows
        ]

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

        generated = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=int(
                cfg[
                    "max_prompt_length"
                ]
            ),
            max_new_tokens=int(
                cfg[
                    "max_response_length"
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

        sequences = generated[
            "sequences"
        ]

        attention_mask = generated[
            "attention_mask"
        ]

        response_ids = generated[
            "response_ids"
        ]

        response_mask = generated[
            "response_mask"
        ]

        prompt_width = int(
            generated[
                "prompt_width"
            ]
        )

        response_steps = (
            response_ids.shape[1]
        )

        generated_tokens = int(
            response_mask
            .sum()
            .item()
        )

        total_generated_tokens += (
            generated_tokens
        )

        # Compute behavior-policy and reference
        # probabilities with dropout disabled.
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

        if policy_was_training:
            policy.train()

        old_logp = (
            old_logp.detach()
        )

        ref_logp = (
            ref_logp.detach()
        )

        with torch.no_grad():
            raw_task_reward = (
                score_reward_pairs(
                    reward_model,
                    reward_tokenizer,
                    prompts,
                    generated[
                        "responses"
                    ],
                    max_length=int(
                        cfg[
                            "reward_max_length"
                        ]
                    ),
                )
                .to(old_logp.device)
                .float()
            )

        eos_penalty = torch.tensor(
            [
                (
                    0.0
                    if terminated
                    else missing_eos_penalty
                )
                for terminated
                in generated[
                    "terminated_with_eos"
                ]
            ],
            device=old_logp.device,
            dtype=torch.float32,
        )

        effective_task_reward = (
            raw_task_reward
            - eos_penalty
        )

        rewards = shaped_rewards(
            effective_task_reward,
            old_logp,
            ref_logp,
            response_mask,
            beta_kl,
        )

        value_was_training = (
            value_model.training
        )

        value_model.eval()

        with torch.no_grad():
            rollout_values = (
                response_state_values(
                    value_model,
                    sequences,
                    attention_mask,
                    prompt_width,
                    response_steps,
                )
                .float()
            )

        if value_was_training:
            value_model.train()

        advantages, returns = (
            compute_gae(
                rewards,
                rollout_values,
                response_mask,
                gamma=gamma,
                lam=gae_lambda,
            )
        )

        normalized_advantages = (
            normalize_advantages(
                advantages,
                response_mask,
            )
            .detach()
        )

        returns = (
            returns.detach()
        )

        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_clip_fractions = []
        epoch_policy_grad_norms = []
        epoch_value_grad_norms = []

        for _ in range(
            ppo_epochs
        ):
            policy.train()

            policy_optimizer.zero_grad(
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

            (
                policy_loss,
                _,
                clip_fraction,
            ) = ppo_policy_loss(
                new_logp,
                old_logp,
                normalized_advantages,
                response_mask,
                eps=clip_eps,
            )

            policy_loss.backward()

            policy_grad_norm = (
                clip_grad_norm_(
                    policy_grad_parameters(
                        policy
                    ),
                    max_grad_norm,
                )
            )

            policy_optimizer.step()

            epoch_policy_losses.append(
                float(
                    policy_loss
                    .detach()
                    .item()
                )
            )

            epoch_clip_fractions.append(
                float(
                    clip_fraction.item()
                )
            )

            epoch_policy_grad_norms.append(
                float(
                    policy_grad_norm
                )
            )

            value_model.train()

            value_optimizer.zero_grad(
                set_to_none=True
            )

            predicted_values = (
                response_state_values(
                    value_model,
                    sequences,
                    attention_mask,
                    prompt_width,
                    response_steps,
                )
            )

            value_loss = (
                value_mse_loss(
                    predicted_values,
                    returns,
                    response_mask,
                )
            )

            (
                value_coef
                * value_loss
            ).backward()

            value_grad_norm = (
                clip_grad_norm_(
                    value_grad_parameters(
                        value_model
                    ),
                    max_grad_norm,
                )
            )

            value_optimizer.step()

            epoch_value_losses.append(
                float(
                    value_loss
                    .detach()
                    .item()
                )
            )

            epoch_value_grad_norms.append(
                float(
                    value_grad_norm
                )
            )

        rollout_kl = sampled_kl(
            old_logp,
            ref_logp,
            response_mask,
        )

        valid_mask = (
            response_mask.bool()
        )

        valid_values = (
            rollout_values[
                valid_mask
            ]
        )

        valid_returns = (
            returns[
                valid_mask
            ]
        )

        valid_advantages = (
            advantages[
                valid_mask
            ]
        )

        update_seconds = (
            time.perf_counter()
            - update_start
        )

        record = {
            "run_name":
                run_name,

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

            "clip_epsilon":
                clip_eps,

            "kl_beta":
                beta_kl,

            "ppo_epochs":
                ppo_epochs,

            "generated_tokens":
                generated_tokens,

            "raw_reward_mean":
                float(
                    raw_task_reward
                    .mean()
                    .item()
                ),

            "effective_reward_mean":
                float(
                    effective_task_reward
                    .mean()
                    .item()
                ),

            "kl_from_reference":
                float(
                    rollout_kl.item()
                ),

            "policy_loss":
                float(
                    sum(
                        epoch_policy_losses
                    )
                    / len(
                        epoch_policy_losses
                    )
                ),

            "value_loss":
                float(
                    sum(
                        epoch_value_losses
                    )
                    / len(
                        epoch_value_losses
                    )
                ),

            "entropy":
                float(
                    rollout_entropy.item()
                ),

            "clip_fraction":
                float(
                    sum(
                        epoch_clip_fractions
                    )
                    / len(
                        epoch_clip_fractions
                    )
                ),

            "policy_grad_norm":
                float(
                    sum(
                        epoch_policy_grad_norms
                    )
                    / len(
                        epoch_policy_grad_norms
                    )
                ),

            "value_grad_norm":
                float(
                    sum(
                        epoch_value_grad_norms
                    )
                    / len(
                        epoch_value_grad_norms
                    )
                ),

            "response_length_mean":
                float(
                    response_mask
                    .sum(dim=-1)
                    .float()
                    .mean()
                    .item()
                ),

            "missing_eos_fraction":
                float(
                    (
                        eos_penalty > 0
                    )
                    .float()
                    .mean()
                    .item()
                ),

            "value_mean":
                float(
                    valid_values
                    .mean()
                    .item()
                ),

            "return_mean":
                float(
                    valid_returns
                    .mean()
                    .item()
                ),

            "advantage_mean":
                float(
                    valid_advantages
                    .mean()
                    .item()
                ),

            "advantage_std":
                float(
                    valid_advantages
                    .std(
                        unbiased=False
                    )
                    .item()
                ),

            "update_seconds":
                float(
                    update_seconds
                ),
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
            f"{record['raw_reward_mean']:.4f} "
            f"kl="
            f"{record['kl_from_reference']:.5f} "
            f"p_loss="
            f"{record['policy_loss']:.4f} "
            f"v_loss="
            f"{record['value_loss']:.4f} "
            f"entropy="
            f"{record['entropy']:.4f} "
            f"clip="
            f"{record['clip_fraction']:.4f} "
            f"len="
            f"{record['response_length_mean']:.1f}"
        )

    total_seconds = (
        time.perf_counter()
        - start_time
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

    value_out = (
        out.parent
        / f"{out.name}_value"
    )

    value_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    value_model.save_pretrained(
        value_out
    )

    summary = {
        "run_name":
            run_name,

        "seed":
            int(
                cfg["seed"]
            ),

        "updates":
            num_updates,

        "ppo_epochs":
            ppo_epochs,

        "prompts_per_update":
            prompts_per_update,

        "clip_epsilon":
            clip_eps,

        "kl_beta":
            beta_kl,

        "max_prompt_length":
            int(
                cfg[
                    "max_prompt_length"
                ]
            ),

        "max_response_length":
            int(
                cfg[
                    "max_response_length"
                ]
            ),

        "total_generated_tokens":
            int(
                total_generated_tokens
            ),

        "prompt_ids":
            used_prompt_ids,

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

        "value_output":
            str(value_out),

        "train_log":
            str(log_path),
    }

    save_json(
        summary_path,
        summary,
    )

    print()
    print(
        "PPO continuation complete."
    )
    print(
        f"Policy: {out}"
    )
    print(
        f"Value model: {value_out}"
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
        default="configs/ppo.yaml",
    )

    ap.add_argument(
        "--output",
    )

    ap.add_argument(
        "--updates",
        type=int,
    )

    ap.add_argument(
        "--clip-epsilon",
        type=float,
    )

    ap.add_argument(
        "--kl-beta",
        type=float,
    )

    ap.add_argument(
        "--run-name",
        default="standard",
    )

    args = ap.parse_args()

    run_ppo(
        args.config,
        args.output,
        args.updates,
        args.clip_epsilon,
        args.kl_beta,
        args.run_name,
    )


if __name__ == "__main__":
    main()