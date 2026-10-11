from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge


EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

# Each pair is oriented as (diagnostically better response, comparison response).
PAIR_SPECS = {
    "reasoning_only": ("clean_correct", "corrupt_reasoning_correct_final"),
    "wrong_final": ("clean_correct", "good_reasoning_wrong_final"),
    "persuasive_filler": ("clean_correct", "persuasive_filler_correct"),
    "gold_distractor": ("clean_correct", "gold_distractor_wrong_final"),
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        pid = str(row["problem_id"])
        variant = row["variant_type"]
        if variant in by_problem[pid]:
            raise ValueError(f"Problem {pid} has duplicate variant {variant}")
        by_problem[pid][variant] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        extra = set(variants) - EXPECTED_VARIANTS
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
        if extra:
            raise ValueError(f"Problem {pid} has unexpected variants: {sorted(extra)}")
    return dict(by_problem)


def _preference_from_scores(better: float, worse: float) -> str:
    if better > worse:
        return "better"
    if better < worse:
        return "wrong"
    return "tie"


def _preference_from_judge(raw: str) -> str:
    return {"A": "better", "B": "wrong", "TIE": "tie"}[raw]


def _summary_from_pair_records(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mechanism in ("rlvr", "rlaif"):
        col = f"{mechanism}_preference"
        for pair_type in PAIR_SPECS:
            subset = df[df["pair_type"] == pair_type]
            rows.append(
                {
                    "mechanism": mechanism,
                    "pair_type": pair_type,
                    "n": int(len(subset)),
                    "better_response_rate": float((subset[col] == "better").mean()),
                    "tie_rate": float((subset[col] == "tie").mean()),
                    "wrong_preference_rate": float((subset[col] == "wrong").mean()),
                }
            )
    return pd.DataFrame(rows)


def _sensitivity_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mechanism in ("rlvr", "rlaif"):
        col = f"{mechanism}_preference"
        reasoning = pair_df[pair_df["pair_type"] == "reasoning_only"]
        # Both wrong-final conditions change the designated outcome. Pool them for Soutcome,
        # while the pair table still reports them separately.
        outcome = pair_df[pair_df["pair_type"].isin(["wrong_final", "gold_distractor"])]
        filler = pair_df[pair_df["pair_type"] == "persuasive_filler"]
        distractor = pair_df[pair_df["pair_type"] == "gold_distractor"]
        rows.append(
            {
                "mechanism": mechanism,
                "Sreason": float((reasoning[col] == "better").mean()),
                "Sreason_tie_rate": float((reasoning[col] == "tie").mean()),
                "Sreason_wrong_rate": float((reasoning[col] == "wrong").mean()),
                "Soutcome": float((outcome[col] == "better").mean()),
                "Soutcome_tie_rate": float((outcome[col] == "tie").mean()),
                "Soutcome_wrong_rate": float((outcome[col] == "wrong").mean()),
                "clean_over_filler_rate": float((filler[col] == "better").mean()),
                "filler_tie_rate": float((filler[col] == "tie").mean()),
                "filler_wrong_rate": float((filler[col] == "wrong").mean()),
                "gold_distractor_robustness": float((distractor[col] == "better").mean()),
                "gold_distractor_tie_rate": float((distractor[col] == "tie").mean()),
                "gold_distractor_wrong_rate": float((distractor[col] == "wrong").mean()),
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)

    if len(groups) != 20:
        raise ValueError(f"Expected 20 diagnostic problems, found {len(groups)}")

    # Validate the fixed exact verifier against the course-provided expected reward.
    variant_records = []
    for pid, variants in groups.items():
        for variant_name in sorted(EXPECTED_VARIANTS):
            row = variants[variant_name]
            if not bool(row.get("manual_validation", False)):
                raise ValueError(f"Problem {pid}/{variant_name} is not marked manually validated")
            actual = float(exact_reward(row["response"], str(row["gold_final"])))
            expected = float(row["expected_exact_reward"])
            if actual != expected:
                raise RuntimeError(
                    f"Exact verifier mismatch for {pid}/{variant_name}: expected {expected}, got {actual}"
                )
            variant_records.append(
                {
                    "problem_id": pid,
                    "question": row["question"],
                    "gold_final": str(row["gold_final"]),
                    "variant_type": variant_name,
                    "response": row["response"],
                    "reasoning_quality": row.get("reasoning_quality"),
                    "style": row.get("style"),
                    "expected_exact_reward": expected,
                    "rlvr_reward": actual,
                }
            )

    judge = PairwiseAIJudge(cfg, outdir / "pairwise_judge_cache.json")
    try:
        # Give every one of the five variants an AI-feedback score by ranking all five
        # responses within each fixed problem group using the supplied pairwise judge.
        by_key = {(r["problem_id"], r["variant_type"]): r for r in variant_records}
        order = [
            "clean_correct",
            "corrupt_reasoning_correct_final",
            "good_reasoning_wrong_final",
            "persuasive_filler_correct",
            "gold_distractor_wrong_final",
        ]
        for n, (pid, variants) in enumerate(groups.items(), start=1):
            responses = [variants[name]["response"] for name in order]
            ai_scores = judge.group_rewards(variants["clean_correct"]["question"], responses)
            for name, score in zip(order, ai_scores):
                by_key[(pid, name)]["rlaif_group_reward"] = float(score)
            if n % 5 == 0 or n == len(groups):
                print(f"AI diagnostic group scoring: {n}/{len(groups)}", flush=True)

        pair_records = []
        for n, (pid, variants) in enumerate(groups.items(), start=1):
            problem = variants["clean_correct"]["question"]
            for pair_type, (better_name, worse_name) in PAIR_SPECS.items():
                better = variants[better_name]
                worse = variants[worse_name]
                better_exact = float(exact_reward(better["response"], str(better["gold_final"])))
                worse_exact = float(exact_reward(worse["response"], str(worse["gold_final"])))
                rlvr_pref = _preference_from_scores(better_exact, worse_exact)
                raw = judge.compare(problem, better["response"], worse["response"])
                rlaif_pref = _preference_from_judge(raw)
                pair_records.append(
                    {
                        "problem_id": pid,
                        "question": problem,
                        "pair_type": pair_type,
                        "better_variant": better_name,
                        "comparison_variant": worse_name,
                        "better_response": better["response"],
                        "comparison_response": worse["response"],
                        "better_rlvr_reward": better_exact,
                        "comparison_rlvr_reward": worse_exact,
                        "rlvr_preference": rlvr_pref,
                        "judge_raw": raw,
                        "rlaif_preference": rlaif_pref,
                        "mechanisms_agree": bool(rlvr_pref == rlaif_pref),
                    }
                )
            if n % 5 == 0 or n == len(groups):
                print(f"Controlled pair scoring: {n}/{len(groups)}", flush=True)
    finally:
        del judge
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    variant_df = pd.DataFrame(variant_records)
    pair_df = pd.DataFrame(pair_records)
    pair_summary = _summary_from_pair_records(pair_df)
    sensitivity = _sensitivity_summary(pair_df)

    variant_summary = (
        variant_df.groupby("variant_type", as_index=False)
        .agg(
            n=("problem_id", "size"),
            mean_rlvr_reward=("rlvr_reward", "mean"),
            mean_rlaif_group_reward=("rlaif_group_reward", "mean"),
        )
        .sort_values("variant_type")
        .reset_index(drop=True)
    )

    variant_df.to_csv(outdir / "diagnostic_variant_scores.csv", index=False)
    variant_summary.to_csv(outdir / "diagnostic_variant_summary.csv", index=False)
    pair_df.to_csv(outdir / "diagnostic_pair_records.csv", index=False)
    pair_summary.to_csv(outdir / "diagnostic_pair_summary.csv", index=False)
    sensitivity.to_csv(outdir / "diagnostic_sensitivity_summary.csv", index=False)

    disagreements = pair_df[pair_df["mechanisms_agree"] == False].copy()  # noqa: E712
    disagreements.to_csv(outdir / "diagnostic_mechanism_disagreements.csv", index=False)

    manifest = {
        "n_problems": len(groups),
        "n_variants": int(len(variant_df)),
        "n_controlled_pairs": int(len(pair_df)),
        "variant_summary": variant_summary.to_dict(orient="records"),
        "pair_summary": pair_summary.to_dict(orient="records"),
        "sensitivity_summary": sensitivity.to_dict(orient="records"),
        "n_mechanism_disagreements": int(len(disagreements)),
        "Soutcome_definition": "pooled clean_correct vs good_reasoning_wrong_final and clean_correct vs gold_distractor_wrong_final",
    }
    (outdir / "diagnostic_summary.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nControlled-pair summary:")
    print(pair_summary.to_string(index=False))
    print("\nSensitivity summary:")
    print(sensitivity.to_string(index=False))
    print("\nWrote Task 5 diagnostic outputs under:", outdir)


if __name__ == "__main__":
    main()
