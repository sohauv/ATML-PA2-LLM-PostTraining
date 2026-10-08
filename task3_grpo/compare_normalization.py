from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from common.data import load_yaml, repo_path
from common.logging_utils import save_json


def read_json(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def read_jsonl(path):
    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def run_condition(
    *,
    config_path: str,
    updates: int,
    loss_type: str,
    run_name: str,
    output: str,
):
    cmd = [
        sys.executable,
        "-m",
        "task3_grpo.continue_train",
        "--config",
        config_path,
        "--updates",
        str(updates),
        "--loss-type",
        loss_type,
        "--run-name",
        run_name,
        "--output",
        output,
    ]

    print()
    print("=" * 88)
    print(
        f"Running {run_name} "
        f"(loss_type={loss_type})"
    )
    print("=" * 88)
    print(
        " ".join(cmd)
    )
    print()

    subprocess.run(
        cmd,
        check=True,
    )


def finite_mean(values):
    clean = [
        float(x)
        for x in values
        if x is not None
        and np.isfinite(
            float(x)
        )
    ]

    if not clean:
        return float("nan")

    return float(
        np.mean(clean)
    )


def summarize_training(
    summary,
    rows,
):
    return {
        "run_name":
            summary[
                "run_name"
            ],

        "loss_type":
            summary[
                "loss_type"
            ],

        "updates":
            int(
                summary[
                    "updates"
                ]
            ),

        "num_generations":
            int(
                summary[
                    "num_generations"
                ]
            ),

        "clip_epsilon":
            float(
                summary[
                    "clip_epsilon"
                ]
            ),

        "kl_beta":
            float(
                summary[
                    "kl_beta"
                ]
            ),

        "scheduled_generation_token_budget":
            int(
                summary[
                    "updates"
                ]
                * summary[
                    "prompts_per_update"
                ]
                * summary[
                    "num_generations"
                ]
                * summary[
                    "max_completion_length"
                ]
            ),

        "realized_generated_tokens":
            int(
                summary[
                    "total_generated_tokens"
                ]
            ),

        "realized_train_tokens":
            int(
                summary[
                    "total_train_tokens"
                ]
            ),

        "mean_reward":
            finite_mean(
                [
                    row[
                        "reward_mean"
                    ]
                    for row in rows
                ]
            ),

        "mean_kl":
            finite_mean(
                [
                    row[
                        "kl_from_reference"
                    ]
                    for row in rows
                ]
            ),

        "mean_response_length":
            finite_mean(
                [
                    row[
                        "response_length_mean"
                    ]
                    for row in rows
                ]
            ),

        "mean_entropy":
            finite_mean(
                [
                    row[
                        "entropy"
                    ]
                    for row in rows
                ]
            ),

        "mean_group_reward_std":
            finite_mean(
                [
                    row[
                        "group_reward_std_mean"
                    ]
                    for row in rows
                ]
            ),

        "mean_uninformative_group_fraction":
            finite_mean(
                [
                    row[
                        "uninformative_group_fraction"
                    ]
                    for row in rows
                ]
            ),

        "mean_policy_loss":
            finite_mean(
                [
                    row[
                        "policy_loss"
                    ]
                    for row in rows
                ]
            ),

        "mean_policy_grad_norm":
            finite_mean(
                [
                    row[
                        "policy_grad_norm"
                    ]
                    for row in rows
                ]
            ),

        "mean_truncated_fraction":
            finite_mean(
                [
                    row[
                        "truncated_fraction"
                    ]
                    for row in rows
                ]
            ),

        # Direct normalization diagnostic.
        # Canonical uses |A| per valid sequence at rho=1.
        # Dr-GRPO uses |A| * T/Tmax.
        "mean_length_signal_corr":
            finite_mean(
                [
                    row.get(
                        "length_signal_corr"
                    )
                    for row in rows
                ]
            ),

        "mean_short_abs_sequence_signal":
            finite_mean(
                [
                    row.get(
                        "short_abs_sequence_signal"
                    )
                    for row in rows
                ]
            ),

        "mean_long_abs_sequence_signal":
            finite_mean(
                [
                    row.get(
                        "long_abs_sequence_signal"
                    )
                    for row in rows
                ]
            ),

        "mean_long_short_signal_ratio":
            finite_mean(
                [
                    row.get(
                        "long_short_signal_ratio"
                    )
                    for row in rows
                ]
            ),

        "wall_seconds":
            float(
                summary[
                    "total_wall_seconds"
                ]
            ),

        "peak_vram_gb":
            float(
                summary[
                    "peak_vram_gb"
                ]
            ),
    }


def write_csv(
    path,
    rows,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=
                fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def validate_matched_design(
    canonical_summary,
    dr_summary,
):
    fixed_keys = [
        "seed",
        "updates",
        "prompts_per_update",
        "num_generations",
        "policy_epochs",
        "learning_rate",
        "clip_epsilon",
        "kl_beta",
        "max_prompt_length",
        "max_completion_length",
        "mask_truncated_completions",
        "prompt_ids",
        "source_indices",
    ]

    mismatches = {}

    for key in fixed_keys:
        if (
            canonical_summary.get(key)
            != dr_summary.get(key)
        ):
            mismatches[key] = {
                "canonical":
                    canonical_summary.get(
                        key
                    ),

                "dr_grpo":
                    dr_summary.get(
                        key
                    ),
            }

    if mismatches:
        raise AssertionError(
            "Canonical/Dr-GRPO design mismatch:\n"
            + json.dumps(
                mismatches,
                indent=2,
            )
        )

    scheduled_canonical = (
        canonical_summary[
            "updates"
        ]
        * canonical_summary[
            "prompts_per_update"
        ]
        * canonical_summary[
            "num_generations"
        ]
        * canonical_summary[
            "max_completion_length"
        ]
    )

    scheduled_dr = (
        dr_summary[
            "updates"
        ]
        * dr_summary[
            "prompts_per_update"
        ]
        * dr_summary[
            "num_generations"
        ]
        * dr_summary[
            "max_completion_length"
        ]
    )

    if (
        scheduled_canonical
        != scheduled_dr
    ):
        raise AssertionError(
            "Scheduled generated-token budgets differ"
        )

    return int(
        scheduled_canonical
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/grpo.yaml",
    )

    ap.add_argument(
        "--skip-training",
        action="store_true",
        help=(
            "Only validate/summarize existing "
            "normalization fork results."
        ),
    )

    args = ap.parse_args()

    cfg = load_yaml(
        args.config
    )

    updates = int(
        cfg["fork_updates"]
    )

    results_dir = repo_path(
        cfg["results_dir"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical_name = (
        "normalization_canonical"
    )

    dr_name = (
        "normalization_dr_grpo"
    )

    canonical_output = (
        "outputs/task3_grpo/"
        "normalization_canonical"
    )

    dr_output = (
        "outputs/task3_grpo/"
        "normalization_dr_grpo"
    )

    if not args.skip_training:
        run_condition(
            config_path=
                args.config,
            updates=
                updates,
            loss_type=
                "grpo",
            run_name=
                canonical_name,
            output=
                canonical_output,
        )

        run_condition(
            config_path=
                args.config,
            updates=
                updates,
            loss_type=
                "dr_grpo",
            run_name=
                dr_name,
            output=
                dr_output,
        )

    canonical_summary_path = (
        results_dir
        / (
            canonical_name
            + "_summary.json"
        )
    )

    dr_summary_path = (
        results_dir
        / (
            dr_name
            + "_summary.json"
        )
    )

    canonical_log_path = (
        results_dir
        / (
            canonical_name
            + "_train.jsonl"
        )
    )

    dr_log_path = (
        results_dir
        / (
            dr_name
            + "_train.jsonl"
        )
    )

    required = [
        canonical_summary_path,
        dr_summary_path,
        canonical_log_path,
        dr_log_path,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing normalization result files:\n"
            + "\n".join(missing)
        )

    canonical_summary = (
        read_json(
            canonical_summary_path
        )
    )

    dr_summary = (
        read_json(
            dr_summary_path
        )
    )

    scheduled_budget = (
        validate_matched_design(
            canonical_summary,
            dr_summary,
        )
    )

    canonical_rows = (
        read_jsonl(
            canonical_log_path
        )
    )

    dr_rows = (
        read_jsonl(
            dr_log_path
        )
    )

    if (
        len(canonical_rows)
        != updates
        or len(dr_rows)
        != updates
    ):
        raise AssertionError(
            "Normalization fork log length "
            "does not match fork_updates"
        )

    canonical_table = (
        summarize_training(
            canonical_summary,
            canonical_rows,
        )
    )

    dr_table = (
        summarize_training(
            dr_summary,
            dr_rows,
        )
    )

    table = [
        canonical_table,
        dr_table,
    ]

    comparison = {
        "fork_updates":
            updates,

        "matched_design":
            True,

        "scheduled_generation_token_budget_per_condition":
            scheduled_budget,

        "budget_definition":
            (
                "updates × prompts_per_update × K × "
                "max_completion_length. Realized generated "
                "tokens are saved separately because response "
                "length is an outcome of the normalization "
                "condition."
            ),

        "canonical":
            canonical_table,

        "dr_grpo":
            dr_table,

        "realized_generated_token_difference_dr_minus_canonical":
            int(
                dr_table[
                    "realized_generated_tokens"
                ]
                - canonical_table[
                    "realized_generated_tokens"
                ]
            ),

        "response_length_difference_dr_minus_canonical":
            float(
                dr_table[
                    "mean_response_length"
                ]
                - canonical_table[
                    "mean_response_length"
                ]
            ),

        "length_signal_corr_difference_dr_minus_canonical":
            float(
                dr_table[
                    "mean_length_signal_corr"
                ]
                - canonical_table[
                    "mean_length_signal_corr"
                ]
            ),

        "long_short_signal_ratio_difference_dr_minus_canonical":
            float(
                dr_table[
                    "mean_long_short_signal_ratio"
                ]
                - canonical_table[
                    "mean_long_short_signal_ratio"
                ]
            ),
    }

    save_json(
        results_dir
        / "normalization_training_comparison.json",
        comparison,
    )

    write_csv(
        results_dir
        / "normalization_training_table.csv",
        table,
    )

    print()
    print("=" * 88)
    print(
        "Matched normalization study complete"
    )
    print("=" * 88)

    print(
        "Fork updates:",
        updates,
    )

    print(
        "Scheduled generation-token budget "
        "per condition:",
        scheduled_budget,
    )

    print()
    print(
        "condition              "
        "reward      KL          len      "
        "grad      trunc    len-signal-corr  "
        "long/short"
    )

    for row in table:
        print(
            f"{row['loss_type']:<22s}"
            f"{row['mean_reward']:<12.4f}"
            f"{row['mean_kl']:<12.6f}"
            f"{row['mean_response_length']:<9.1f}"
            f"{row['mean_policy_grad_norm']:<10.4f}"
            f"{row['mean_truncated_fraction']:<9.4f}"
            f"{row['mean_length_signal_corr']:<17.4f}"
            f"{row['mean_long_short_signal_ratio']:.4f}"
        )

    print()
    print(
        "Realized generated tokens:"
    )

    print(
        "  canonical:",
        canonical_table[
            "realized_generated_tokens"
        ],
    )

    print(
        "  dr_grpo  :",
        dr_table[
            "realized_generated_tokens"
        ],
    )

    print()
    print(
        "Saved:",
        results_dir
        / "normalization_training_comparison.json",
    )

    print(
        "Saved:",
        results_dir
        / "normalization_training_table.csv",
    )


if __name__ == "__main__":
    main()
