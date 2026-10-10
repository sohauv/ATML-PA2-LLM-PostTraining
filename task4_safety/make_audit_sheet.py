from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path

POLICY_ORDER = ("sft", "dpo", "ppo", "grpo")
VALID_LABELS = {
    "SAFE_ANSWER",
    "JUSTIFIED_REFUSAL",
    "UNSAFE_COMPLIANCE",
    "OVER_REFUSAL",
    "AMBIGUOUS",
}


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"

    generated = {}
    for policy in POLICY_ORDER:
        src = outdir / f"generated_{policy}.jsonl"
        if not src.exists():
            raise FileNotFoundError("Generate/save responses first: " + str(src))
        generated[policy] = read_jsonl(src)

    base = generated["sft"]
    ids = fixed_audit_ids(base, int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    idset = set(ids)

    # Save the fixed prompt IDs separately as immutable audit metadata.
    pd.DataFrame({"xstest_id": ids}).to_csv(outdir / "manual_audit_ids.csv", index=False)

    rows = []
    for policy in POLICY_ORDER:
        selected = [r for r in generated[policy] if int(r["xstest_id"]) in idset]
        if len(selected) != len(ids):
            raise RuntimeError(f"{policy}: expected {len(ids)} audit prompts, found {len(selected)}")
        by_id = {int(r["xstest_id"]): r for r in selected}
        for xid in ids:
            r = by_id[xid]
            rows.append({
                "xstest_id": xid,
                "policy": policy,
                "benchmark_class": r["benchmark_class"],
                "type": r["type"],
                "prompt": r["prompt"],
                "response": r["response"],
                "manual_label": "",
                "manual_notes": "",
            })

    audit_path = outdir / "manual_audit.csv"
    if audit_path.exists():
        old = pd.read_csv(audit_path, keep_default_na=False)
        nonempty = old.get("manual_label", pd.Series(dtype=str)).astype(str).str.strip().ne("").any()
        if nonempty:
            raise RuntimeError(
                f"Refusing to overwrite {audit_path} because it already contains manual labels."
            )

    audit = pd.DataFrame(rows)
    audit.to_csv(audit_path, index=False)
    print("Wrote fixed 60-prompt audit IDs:", outdir / "manual_audit_ids.csv")
    print("Wrote blind manual audit sheet:", audit_path)
    print("Rows to label (60 prompts x 4 policies):", len(audit))
    print("Allowed manual_label values:", sorted(VALID_LABELS))
    print("Do not open judged_*.jsonl until manual labeling is finished.")


if __name__ == "__main__":
    main()
