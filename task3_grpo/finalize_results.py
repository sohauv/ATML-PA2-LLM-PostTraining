from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from common.data import load_yaml, repo_path
from common.logging_utils import save_json


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def eval_row(label: str, summary: dict) -> dict:
    return {
        "condition": label,
        "num_examples": summary["num_examples"],
        "reward_mean": summary["reward_model_score_mean"],
        "reward_std": summary["reward_model_score_std"],
        "kl_from_reference": summary["kl_from_reference"],
        "entropy": summary["entropy"],
        "mean_response_length": summary["mean_response_length"],
        "std_response_length": summary["std_response_length"],
        "median_response_length": summary["median_response_length"],
        "iqr_response_length": summary["iqr_response_length"],
        "eos_fraction": summary["terminated_with_eos_fraction"],
        "truncated_fraction": summary["truncated_fraction"],
    }


def same_eval_protocol(a: dict, b: dict) -> bool:
    keys = [
        "seed",
        "num_examples",
        "max_prompt_length",
        "max_response_length",
        "temperature",
        "top_p",
        "do_sample",
    ]
    return all(a.get(k) == b.get(k) for k in keys)


def main():
    cfg = load_yaml("configs/grpo.yaml")
    results = repo_path(cfg["results_dir"])
    assets = results / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Required result files
    # ------------------------------------------------------------------
    paths = {
        "standard_train": results / "standard_train.jsonl",
        "standard_summary": results / "standard_summary.json",
        "standard_eval": results / "standard_eval_summary.json",
        "standard_gen": results / "standard_generations.jsonl",
        "group_summary": results / "group_size_summary.json",
        "group_table": results / "group_size_table.csv",
        "group_bins": results / "group_size_difficulty_table.csv",
        "norm_train_cmp": results / "normalization_training_comparison.json",
        "canonical_train": results / "normalization_canonical_train.jsonl",
        "dr_train": results / "normalization_dr_grpo_train.jsonl",
        "canonical_eval": results / "normalization_canonical_eval_summary.json",
        "dr_eval": results / "normalization_dr_grpo_eval_summary.json",
        "canonical_gen": results / "normalization_canonical_generations.jsonl",
        "dr_gen": results / "normalization_dr_grpo_generations.jsonl",
    }

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Task 3 result files:\n" + "\n".join(missing))

    standard_train = load_jsonl(paths["standard_train"])
    standard_summary = load_json(paths["standard_summary"])
    standard_eval = load_json(paths["standard_eval"])
    standard_gen = load_jsonl(paths["standard_gen"])

    group_summary = load_json(paths["group_summary"])
    norm_train_cmp = load_json(paths["norm_train_cmp"])

    canonical_train = load_jsonl(paths["canonical_train"])
    dr_train = load_jsonl(paths["dr_train"])
    canonical_eval = load_json(paths["canonical_eval"])
    dr_eval = load_json(paths["dr_eval"])
    canonical_gen = load_jsonl(paths["canonical_gen"])
    dr_gen = load_jsonl(paths["dr_gen"])

    # ------------------------------------------------------------------
    # Audit held-out protocol and exact prompt order
    # ------------------------------------------------------------------
    if not same_eval_protocol(standard_eval, canonical_eval):
        raise AssertionError("Standard and canonical evaluation protocols differ")
    if not same_eval_protocol(canonical_eval, dr_eval):
        raise AssertionError("Canonical and Dr-GRPO evaluation protocols differ")

    def key(row):
        return (row.get("prompt_id"), row.get("source_index"))

    standard_keys = [key(r) for r in standard_gen]
    canonical_keys = [key(r) for r in canonical_gen]
    dr_keys = [key(r) for r in dr_gen]

    if standard_keys != canonical_keys or canonical_keys != dr_keys:
        raise AssertionError("Held-out prompt IDs/order differ across GRPO conditions")

    if len(standard_keys) != 200:
        raise AssertionError(f"Expected 200 held-out prompts, got {len(standard_keys)}")

    # ------------------------------------------------------------------
    # Standard 20-update trajectory table
    # ------------------------------------------------------------------
    trajectory_fields = [
        "update",
        "reward_mean",
        "kl_from_reference",
        "group_reward_std_mean",
        "uninformative_group_fraction",
        "policy_loss",
        "policy_grad_norm",
        "entropy",
        "response_length_mean",
        "truncated_fraction",
        "generated_tokens",
        "train_tokens",
        "update_seconds",
    ]

    trajectory_rows = [
        {k: row.get(k) for k in trajectory_fields}
        for row in standard_train
    ]

    write_csv(
        assets / "standard_trajectory.csv",
        trajectory_rows,
        trajectory_fields,
    )

    # ------------------------------------------------------------------
    # Held-out normalization comparison
    # ------------------------------------------------------------------
    eval_rows = [
        eval_row("canonical", canonical_eval),
        eval_row("dr_grpo", dr_eval),
    ]

    eval_fields = list(eval_rows[0].keys())

    write_csv(
        assets / "normalization_heldout_table.csv",
        eval_rows,
        eval_fields,
    )

    deltas = {
        "reward_mean_dr_minus_canonical":
            dr_eval["reward_model_score_mean"]
            - canonical_eval["reward_model_score_mean"],

        "kl_dr_minus_canonical":
            dr_eval["kl_from_reference"]
            - canonical_eval["kl_from_reference"],

        "entropy_dr_minus_canonical":
            dr_eval["entropy"]
            - canonical_eval["entropy"],

        "mean_response_length_dr_minus_canonical":
            dr_eval["mean_response_length"]
            - canonical_eval["mean_response_length"],

        "truncated_fraction_dr_minus_canonical":
            dr_eval["truncated_fraction"]
            - canonical_eval["truncated_fraction"],
    }

    save_json(
        assets / "normalization_heldout_deltas.json",
        deltas,
    )

    # ------------------------------------------------------------------
    # Direct matched-rollout normalization diagnostic
    #
    # Use the canonical fork's sampled rollouts and compute BOTH
    # normalization allocations counterfactually on exactly the same
    # rewards/advantages/lengths. This isolates normalization from
    # policy-induced changes in later rollouts.
    #
    # At rho=1:
    #   canonical total abs sequence signal = |A|
    #   Dr-GRPO total abs sequence signal   = |A| * T/Tmax
    #
    # and per-token normalization:
    #   canonical = |A|/T
    #   Dr-GRPO   = |A|/Tmax
    # ------------------------------------------------------------------
    max_len = float(cfg["max_completion_length"])
    matched_rows = []

    for update_row in canonical_train:
        advantages = update_row["advantages"]
        lengths = update_row["sequence_train_lengths"]
        response_lengths = update_row["response_lengths"]
        truncated = update_row["truncated"]

        for seq_idx, (adv, train_len, response_len, was_truncated) in enumerate(
            zip(advantages, lengths, response_lengths, truncated)
        ):
            train_len = int(train_len)
            if train_len <= 0:
                continue

            abs_adv = abs(float(adv))

            matched_rows.append(
                {
                    "update": int(update_row["update"]),
                    "sequence_index": int(seq_idx),
                    "response_length": int(response_len),
                    "train_length": train_len,
                    "truncated": bool(was_truncated),
                    "abs_advantage": abs_adv,
                    "canonical_total_abs_signal": abs_adv,
                    "dr_total_abs_signal": abs_adv * train_len / max_len,
                    "canonical_per_token_abs_weight": abs_adv / train_len,
                    "dr_per_token_abs_weight": abs_adv / max_len,
                }
            )

    if not matched_rows:
        raise AssertionError("No valid canonical rollout sequences for matched diagnostic")

    matched_fields = list(matched_rows[0].keys())

    write_csv(
        assets / "normalization_matched_rollout_signal.csv",
        matched_rows,
        matched_fields,
    )

    # Rank split valid sequences by realized length.
    order = sorted(range(len(matched_rows)), key=lambda i: matched_rows[i]["train_length"])
    split = max(1, len(order) // 2)
    short_idx = order[:split]
    long_idx = order[split:]
    if not long_idx:
        long_idx = short_idx

    def mean_for(indices, field):
        return float(np.mean([matched_rows[i][field] for i in indices]))

    direct_signal_summary = {
        "num_valid_sequences": len(matched_rows),
        "split_rule": "Rank all valid canonical-fork rollout sequences by realized train length; lower half=short, upper half=long.",
        "short_mean_length": mean_for(short_idx, "train_length"),
        "long_mean_length": mean_for(long_idx, "train_length"),
        "canonical_short_total_abs_signal": mean_for(short_idx, "canonical_total_abs_signal"),
        "canonical_long_total_abs_signal": mean_for(long_idx, "canonical_total_abs_signal"),
        "dr_short_total_abs_signal_counterfactual": mean_for(short_idx, "dr_total_abs_signal"),
        "dr_long_total_abs_signal_counterfactual": mean_for(long_idx, "dr_total_abs_signal"),
        "canonical_short_per_token_abs_weight": mean_for(short_idx, "canonical_per_token_abs_weight"),
        "canonical_long_per_token_abs_weight": mean_for(long_idx, "canonical_per_token_abs_weight"),
        "dr_short_per_token_abs_weight_counterfactual": mean_for(short_idx, "dr_per_token_abs_weight"),
        "dr_long_per_token_abs_weight_counterfactual": mean_for(long_idx, "dr_per_token_abs_weight"),
    }

    save_json(
        assets / "normalization_matched_rollout_signal_summary.json",
        direct_signal_summary,
    )

    # ------------------------------------------------------------------
    # Qualitative candidate table: preserve both responses and metrics,
    # but do not make a qualitative judgment automatically.
    # ------------------------------------------------------------------
    qualitative = []

    for c, d in zip(canonical_gen, dr_gen):
        if key(c) != key(d):
            raise AssertionError("Canonical/Dr row alignment failed")

        reward_delta = (
            float(d["reward_model_score"])
            - float(c["reward_model_score"])
        )
        length_delta = (
            int(d["response_length"])
            - int(c["response_length"])
        )

        qualitative.append(
            {
                "prompt_id": c.get("prompt_id"),
                "source_index": c.get("source_index"),
                "prompt": c.get("prompt"),
                "canonical_reward": c["reward_model_score"],
                "dr_reward": d["reward_model_score"],
                "reward_delta_dr_minus_canonical": reward_delta,
                "abs_reward_delta": abs(reward_delta),
                "canonical_length": c["response_length"],
                "dr_length": d["response_length"],
                "length_delta_dr_minus_canonical": length_delta,
                "abs_length_delta": abs(length_delta),
                "canonical_kl": c["kl_from_reference"],
                "dr_kl": d["kl_from_reference"],
                "canonical_response": c["response"],
                "dr_response": d["response"],
            }
        )

    qualitative.sort(
        key=lambda r: (
            -float(r["abs_reward_delta"]),
            -float(r["abs_length_delta"]),
        )
    )

    qual_fields = list(qualitative[0].keys())

    write_csv(
        assets / "normalization_qualitative_candidates.csv",
        qualitative,
        qual_fields,
    )

    with (assets / "normalization_qualitative_top20.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(qualitative[:20], f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Compact audit manifest
    # ------------------------------------------------------------------
    manifest = {
        "task3_standard": {
            "updates": standard_summary["updates"],
            "num_generations": standard_summary["num_generations"],
            "wall_seconds": standard_summary["total_wall_seconds"],
            "peak_vram_gb": standard_summary["peak_vram_gb"],
            "heldout_examples": standard_eval["num_examples"],
        },
        "group_size": {
            "equal_generation_budget": group_summary["equal_generation_budget"],
            "group_sizes": [row["k"] for row in group_summary["overall"]],
            "difficulty_binning_rule": group_summary["difficulty_binning_rule"],
        },
        "normalization": {
            "fork_updates": norm_train_cmp["fork_updates"],
            "matched_design": norm_train_cmp["matched_design"],
            "scheduled_generation_token_budget_per_condition":
                norm_train_cmp["scheduled_generation_token_budget_per_condition"],
            "heldout_prompt_order_identical": True,
            "heldout_evaluation_protocol_identical": True,
            "heldout_examples_per_condition": canonical_eval["num_examples"],
        },
        "generated_assets": [
            "standard_trajectory.csv",
            "normalization_heldout_table.csv",
            "normalization_heldout_deltas.json",
            "normalization_matched_rollout_signal.csv",
            "normalization_matched_rollout_signal_summary.json",
            "normalization_qualitative_candidates.csv",
            "normalization_qualitative_top20.json",
        ],
    }

    save_json(
        assets / "task3_report_manifest.json",
        manifest,
    )

    print("TASK 3 FINALIZATION AUDIT PASSED")
    print()
    print("Held-out prompt/order match: 200/200")
    print("Evaluation protocol match: yes")
    print()
    print("Canonical vs Dr-GRPO held-out:")
    print(
        f"  reward: {canonical_eval['reward_model_score_mean']:.6f}"
        f" -> {dr_eval['reward_model_score_mean']:.6f}"
        f"  delta={deltas['reward_mean_dr_minus_canonical']:+.6f}"
    )
    print(
        f"  KL:     {canonical_eval['kl_from_reference']:.8f}"
        f" -> {dr_eval['kl_from_reference']:.8f}"
        f"  delta={deltas['kl_dr_minus_canonical']:+.8f}"
    )
    print(
        f"  length: {canonical_eval['mean_response_length']:.2f}"
        f" -> {dr_eval['mean_response_length']:.2f}"
        f"  delta={deltas['mean_response_length_dr_minus_canonical']:+.2f}"
    )
    print()
    print("Matched-rollout normalization diagnostic:")
    for k, v in direct_signal_summary.items():
        print(f"  {k}: {v}")
    print()
    print("Assets:", assets)


if __name__ == "__main__":
    main()
