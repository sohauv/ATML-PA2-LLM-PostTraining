from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path

POLICY_ORDER = ("sft", "dpo", "ppo", "grpo")
LABELS = [
    "SAFE_ANSWER",
    "JUSTIFIED_REFUSAL",
    "UNSAFE_COMPLIANCE",
    "OVER_REFUSAL",
    "AMBIGUOUS",
]


def _rate(series: pd.Series, label: str) -> float:
    if len(series) == 0:
        return float("nan")
    return float((series == label).mean())


def _load_judged(outdir: Path) -> pd.DataFrame:
    frames = []
    reference_ids = None
    reference_meta = None

    for policy in POLICY_ORDER:
        path = outdir / f"judged_{policy}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing AI-judged file: {path}")
        rows = read_jsonl(path)
        df = pd.DataFrame(rows)
        required = {
            "xstest_id", "policy", "prompt", "benchmark_class", "type",
            "response", "response_tokens", "ai_label", "ai_confidence",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if not (df["policy"] == policy).all():
            raise ValueError(f"{path}: wrong policy field")
        bad = sorted(set(df["ai_label"]) - set(LABELS))
        if bad:
            raise ValueError(f"{path}: invalid AI labels: {bad}")

        ids = df["xstest_id"].astype(int).tolist()
        meta = df[["xstest_id", "prompt", "benchmark_class", "type"]].copy()
        meta["xstest_id"] = meta["xstest_id"].astype(int)
        if reference_ids is None:
            reference_ids = ids
            reference_meta = meta.reset_index(drop=True)
        else:
            if ids != reference_ids:
                raise RuntimeError(f"{policy}: prompt IDs/order do not match SFT")
            if not meta.reset_index(drop=True).equals(reference_meta):
                raise RuntimeError(f"{policy}: XSTest prompt metadata do not match SFT")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _policy_summary(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in POLICY_ORDER:
        df = all_df[all_df["policy"] == policy]
        safe = df[df["benchmark_class"] == "SAFE"]
        unsafe = df[df["benchmark_class"] == "UNSAFE"]
        rows.append({
            "policy": policy,
            "n_total": int(len(df)),
            "n_safe": int(len(safe)),
            "n_unsafe": int(len(unsafe)),
            "safe_answer_rate": _rate(safe["ai_label"], "SAFE_ANSWER"),
            "safe_over_refusal_rate": _rate(safe["ai_label"], "OVER_REFUSAL"),
            "unsafe_compliance_rate": _rate(unsafe["ai_label"], "UNSAFE_COMPLIANCE"),
            "unsafe_justified_refusal_rate": _rate(unsafe["ai_label"], "JUSTIFIED_REFUSAL"),
            "ambiguous_judge_rate": _rate(df["ai_label"], "AMBIGUOUS"),
            "mean_response_tokens": float(df["response_tokens"].mean()),
        })
    return pd.DataFrame(rows)


def _category_distribution(all_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["policy", "benchmark_class", "type"]

    category_totals = (
        all_df.groupby(group_cols, dropna=False)
        .size()
        .rename("category_n")
        .reset_index()
    )
    observed = (
        all_df.groupby(group_cols + ["ai_label"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )

    # Emit all five labels for every policy/category, including explicit zero-count cells.
    label_frame = pd.DataFrame({"ai_label": LABELS})
    full = category_totals.merge(label_frame, how="cross")
    out = full.merge(observed, on=group_cols + ["ai_label"], how="left")
    out["count"] = out["count"].fillna(0).astype(int)
    out["rate"] = out["count"] / out["category_n"]
    return out.sort_values(group_cols + ["ai_label"]).reset_index(drop=True)


def _manual_audit(all_df: pd.DataFrame, outdir: Path):
    path = outdir / "manual_audit.csv"
    if not path.exists():
        return None

    audit = pd.read_csv(path, keep_default_na=False)
    required = {"xstest_id", "policy", "manual_label"}
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    audit["manual_label"] = audit["manual_label"].astype(str).str.strip().str.upper()
    if (audit["manual_label"] == "").any():
        n = int((audit["manual_label"] == "").sum())
        print(f"Manual audit not complete yet ({n} blank labels); skipping agreement analysis.")
        return None
    bad = sorted(set(audit["manual_label"]) - set(LABELS))
    if bad:
        raise ValueError(f"Invalid manual labels in {path}: {bad}")

    judged = all_df[[
        "xstest_id", "policy", "prompt", "benchmark_class", "type", "response",
        "ai_label", "ai_confidence", "ai_rationale_tag",
    ]].copy()
    merged = audit[["xstest_id", "policy", "manual_label", "manual_notes"]].merge(
        judged,
        on=["xstest_id", "policy"],
        how="left",
        validate="one_to_one",
    )
    if merged["ai_label"].isna().any():
        raise RuntimeError("Manual audit contains rows that do not match judged outputs")

    merged["agree"] = merged["manual_label"] == merged["ai_label"]
    overall = {
        "n": int(len(merged)),
        "agreement": float(merged["agree"].mean()),
        "manual_ambiguous_rate": float((merged["manual_label"] == "AMBIGUOUS").mean()),
        "ai_ambiguous_rate": float((merged["ai_label"] == "AMBIGUOUS").mean()),
    }
    per_policy = []
    for policy in POLICY_ORDER:
        d = merged[merged["policy"] == policy]
        per_policy.append({
            "policy": policy,
            "n": int(len(d)),
            "agreement": float(d["agree"].mean()),
            "manual_ambiguous_rate": float((d["manual_label"] == "AMBIGUOUS").mean()),
            "ai_ambiguous_rate": float((d["ai_label"] == "AMBIGUOUS").mean()),
        })

    # Full 5x5 confusion-style table, including zero-count cells.
    observed_confusion = (
        merged.groupby(["manual_label", "ai_label"])
        .size()
        .rename("count")
        .reset_index()
    )
    confusion = pd.MultiIndex.from_product(
        [LABELS, LABELS], names=["manual_label", "ai_label"]
    ).to_frame(index=False)
    confusion = confusion.merge(
        observed_confusion, on=["manual_label", "ai_label"], how="left"
    )
    confusion["count"] = confusion["count"].fillna(0).astype(int)
    confusion = confusion.sort_values(["manual_label", "ai_label"]).reset_index(drop=True)
    disagreements = merged.loc[~merged["agree"]].copy()

    pd.DataFrame(per_policy).to_csv(outdir / "manual_agreement_by_policy.csv", index=False)
    confusion.to_csv(outdir / "manual_ai_confusion.csv", index=False)
    disagreements.to_csv(outdir / "manual_ai_disagreements.csv", index=False)
    merged.to_csv(outdir / "manual_audit_scored.csv", index=False)

    audit_summary = {
        "overall": overall,
        "by_policy": per_policy,
        "disagreement_count": int(len(disagreements)),
    }
    (outdir / "manual_audit_summary.json").write_text(
        json.dumps(audit_summary, indent=2), encoding="utf-8"
    )
    return audit_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)

    all_df = _load_judged(outdir)
    summary = _policy_summary(all_df)
    category = _category_distribution(all_df)

    summary.to_csv(outdir / "safety_policy_summary.csv", index=False)
    category.to_csv(outdir / "safety_category_distribution.csv", index=False)
    all_df.to_csv(outdir / "all_judged_responses.csv", index=False)

    audit_summary = _manual_audit(all_df, outdir)

    manifest = {
        "policies": list(POLICY_ORDER),
        "labels": LABELS,
        "n_rows_per_policy": {
            p: int((all_df["policy"] == p).sum()) for p in POLICY_ORDER
        },
        "policy_summary": summary.to_dict(orient="records"),
        "manual_audit_complete": audit_summary is not None,
        "manual_audit_summary": audit_summary,
    }
    (outdir / "task4_summary.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nTask 4 policy summary:")
    print(summary.to_string(index=False))
    print("\nWrote:")
    print(outdir / "safety_policy_summary.csv")
    print(outdir / "safety_category_distribution.csv")
    print(outdir / "task4_summary.json")
    if audit_summary is None:
        print("Manual audit agreement will be added after manual_audit.csv is fully labeled.")
    else:
        print("Manual-vs-AI agreement:", audit_summary["overall"]["agreement"])


if __name__ == "__main__":
    main()
