from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import load_yaml, repo_path


POLICY_ORDER = ("sft", "rlvr", "rlaif")


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _qualitative_candidates(pair_records: pd.DataFrame) -> pd.DataFrame:
    # Keep the selection deterministic and mechanical; the student still decides which
    # examples to discuss and writes all report interpretation themselves.
    disagreements = pair_records[pair_records["mechanisms_agree"].astype(str).str.lower() == "false"].copy()
    if disagreements.empty and "mechanisms_agree" in pair_records:
        disagreements = pair_records[pair_records["mechanisms_agree"] == False].copy()  # noqa: E712
    selected = []
    for pair_type in ("reasoning_only", "persuasive_filler", "wrong_final", "gold_distractor"):
        subset = disagreements[disagreements["pair_type"] == pair_type]
        if subset.empty:
            subset = pair_records[pair_records["pair_type"] == pair_type]
        if not subset.empty:
            selected.append(subset.iloc[0].to_dict())
    return pd.DataFrame(selected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"

    gsm = _read_required_csv(outdir / "gsm_policy_summary.csv")
    transfer = _read_required_csv(outdir / "transfer_policy_summary.csv")
    diag_pairs = _read_required_csv(outdir / "diagnostic_pair_summary.csv")
    diag_sensitivity = _read_required_csv(outdir / "diagnostic_sensitivity_summary.csv")
    diag_variants = _read_required_csv(outdir / "diagnostic_variant_summary.csv")
    pair_records = _read_required_csv(outdir / "diagnostic_pair_records.csv")

    if list(gsm["policy"]) != list(POLICY_ORDER) or list(transfer["policy"]) != list(POLICY_ORDER):
        raise RuntimeError("GSM/transfer summaries do not contain SFT, RLVR, RLAIF in the expected order")

    combined = gsm.copy()
    combined = combined.rename(
        columns={c: f"gsm_{c}" for c in combined.columns if c != "policy"}
    )
    t = transfer.rename(
        columns={c: f"transfer_{c}" for c in transfer.columns if c != "policy"}
    )
    combined = combined.merge(t, on="policy", validate="one_to_one")
    combined["exact_accuracy_drop_gsm_to_transfer"] = (
        combined["gsm_exact_accuracy"] - combined["transfer_exact_accuracy"]
    )
    combined["format_compliance_drop_gsm_to_transfer"] = (
        combined["gsm_format_compliance"] - combined["transfer_format_compliance"]
    )
    combined["mean_response_tokens_change_transfer_minus_gsm"] = (
        combined["transfer_mean_response_tokens"] - combined["gsm_mean_response_tokens"]
    )
    combined["ai_win_rate_drop_gsm_to_transfer"] = (
        combined["gsm_ai_win_rate_vs_sft"] - combined["transfer_ai_win_rate_vs_sft"]
    )

    combined.to_csv(outdir / "task5_policy_comparison.csv", index=False)

    diagnostic_combined = diag_pairs.merge(
        diag_sensitivity,
        on="mechanism",
        how="left",
        validate="many_to_one",
    )
    diagnostic_combined.to_csv(outdir / "task5_diagnostic_comparison.csv", index=False)

    qualitative = _qualitative_candidates(pair_records)
    qualitative.to_csv(outdir / "qualitative_disagreement_candidates.csv", index=False)

    summary = {
        "policy_comparison": combined.replace({np.nan: None}).to_dict(orient="records"),
        "diagnostic_pair_summary": diag_pairs.to_dict(orient="records"),
        "diagnostic_sensitivity": diag_sensitivity.to_dict(orient="records"),
        "diagnostic_variant_summary": diag_variants.to_dict(orient="records"),
        "qualitative_candidate_count": int(len(qualitative)),
        "notes": {
            "ai_win_rate": "win=1, tie=0.5, loss=0 against the matched SFT response",
            "verifier_judge_agreement": "categorical agreement between binary exact-reward preference (policy/SFT/tie) and AI judge preference",
            "Soutcome": "pooled outcome-changing comparisons from clean_correct to good_reasoning_wrong_final and gold_distractor_wrong_final",
        },
    }
    (outdir / "task5_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nTask 5 in-domain / transfer comparison:")
    cols = [
        "policy",
        "gsm_exact_accuracy",
        "transfer_exact_accuracy",
        "exact_accuracy_drop_gsm_to_transfer",
        "gsm_ai_win_rate_vs_sft",
        "transfer_ai_win_rate_vs_sft",
        "gsm_mean_response_tokens",
        "transfer_mean_response_tokens",
    ]
    print(combined[cols].to_string(index=False))
    print("\nDiagnostic sensitivity:")
    print(diag_sensitivity.to_string(index=False))
    print("\nWrote:")
    print(outdir / "task5_policy_comparison.csv")
    print(outdir / "task5_diagnostic_comparison.csv")
    print(outdir / "qualitative_disagreement_candidates.csv")
    print(outdir / "task5_summary.json")


if __name__ == "__main__":
    main()
