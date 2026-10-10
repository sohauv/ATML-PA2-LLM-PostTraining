from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import load_policy, load_tokenizer


POLICY_ORDER = ("sft", "dpo", "ppo", "grpo")


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    try:
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start:start + batch_size]
            prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
            gen = batch_generate(
                model,
                tokenizer,
                prompts,
                max_prompt_length=256,
                max_new_tokens=int(cfg["safety_max_new_tokens"]),
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )
            for (_, row), response, n_tok in zip(
                chunk.iterrows(), gen["responses"], gen["response_lengths"]
            ):
                records.append({
                    "xstest_id": int(row["xstest_id"]),
                    "policy": policy_name,
                    "prompt": str(row["prompt"]),
                    "benchmark_class": str(row["benchmark_class"]),
                    "type": str(row["type"]),
                    "response": response,
                    "response_tokens": int(n_tok),
                })
            print(
                f"[{policy_name}] {min(start + len(chunk), len(df))}/{len(df)}",
                flush=True,
            )
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument(
        "--policy",
        choices=[*POLICY_ORDER, "all"],
        default="all",
        help="Generate one policy or all four fixed Task 4 policies.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate a policy even when its complete output already exists.",
    )
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    df = load_xstest(cfg)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)

    if df["xstest_id"].duplicated().any():
        raise ValueError("XSTest xstest_id values must be unique")

    policies = POLICY_ORDER if args.policy == "all" else (args.policy,)
    expected_ids = df["xstest_id"].astype(int).tolist()

    print("XSTest rows:", len(df))
    print("Policies:", list(policies))

    for policy_name in policies:
        out = outdir / f"generated_{policy_name}.jsonl"
        if out.exists() and not args.overwrite:
            from common.data import read_jsonl

            existing = read_jsonl(out)
            existing_ids = [int(r["xstest_id"]) for r in existing]
            if existing_ids == expected_ids and all(r.get("policy") == policy_name for r in existing):
                print(f"[{policy_name}] complete output already exists; skipping: {out}")
                continue
            raise RuntimeError(
                f"Existing {out} is incomplete or has the wrong prompt order. "
                "Use --overwrite only if you intend to regenerate it."
            )

        rows = generate_for_policy(cfg, policy_name, batch_size=args.batch_size)
        got_ids = [int(r["xstest_id"]) for r in rows]
        if got_ids != expected_ids:
            raise RuntimeError(f"{policy_name}: generated prompt order does not match fixed XSTest order")
        write_jsonl(out, rows)
        print(f"[{policy_name}] wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
