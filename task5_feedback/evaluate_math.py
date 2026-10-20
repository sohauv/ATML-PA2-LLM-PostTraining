from __future__ import annotations

from common.result_validation import validate_math_prefix

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


POLICY_ORDER = ("sft", "rlvr", "rlaif")


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _example_id(row: dict, dataset: str, position: int) -> str:
    if dataset == "transfer":
        return str(row.get("prompt_id", row.get("source_index", f"transfer:{position}")))
    return f"gsm:{row.get('source_index', position)}"


def _problem_text(row: dict) -> str:
    return str(row.get("question", ""))


def _validate_rows(rows: list[dict], dataset: str) -> None:
    if not rows:
        raise ValueError(f"{dataset}: evaluation set is empty")
    required = {"question", "messages", "gold_final"}
    for i, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{dataset} row {i} missing fields: {sorted(missing)}")
    ids = [_example_id(row, dataset, i) for i, row in enumerate(rows)]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{dataset}: example IDs are not unique")


def _expected_ids(rows: list[dict], dataset: str) -> list[str]:
    return [_example_id(row, dataset, i) for i, row in enumerate(rows)]


def _generation_file(outdir: Path, dataset: str, policy: str) -> Path:
    return outdir / f"generated_{dataset}_{policy}.jsonl"


def _pairwise_file(outdir: Path, dataset: str, policy: str) -> Path:
    return outdir / f"pairwise_{dataset}_{policy}_vs_sft.jsonl"


def _generation_is_complete(path: Path, expected_ids: list[str], policy: str) -> bool:
    if not path.exists():
        return False
    rows = read_jsonl(path)
    return (
        len(rows) == len(expected_ids)
        and [str(r.get("example_id")) for r in rows] == expected_ids
        and all(r.get("policy") == policy for r in rows)
    )


def _generate_policy(
    cfg: dict,
    rows: list[dict],
    tokenizer,
    dataset: str,
    policy: str,
    batch_size: int,
) -> list[dict]:
    _set_seed(int(cfg["seed"]))
    model = load_frozen_policy(cfg, policy)
    records: list[dict] = []
    max_prompt_length = int(cfg.get("math_max_prompt_length", 512))
    max_new_tokens = int(cfg["math_max_new_tokens"])
    generation_cfg = cfg.get("generation", {})

    try:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            prompts = [prompt_messages(row) for row in chunk]
            gen = batch_generate(
                model,
                tokenizer,
                prompts,
                max_prompt_length=max_prompt_length,
                max_new_tokens=max_new_tokens,
                temperature=float(generation_cfg.get("temperature", 0.7)),
                top_p=float(generation_cfg.get("top_p", 0.9)),
                do_sample=bool(generation_cfg.get("do_sample", True)),
            )

            for offset, (row, response, n_tok, terminated, truncated) in enumerate(
                zip(
                    chunk,
                    gen["responses"],
                    gen["response_lengths"],
                    gen["terminated_with_eos"],
                    gen["truncated"],
                )
            ):
                pos = start + offset
                gold = str(row["gold_final"])
                pred = extract_designated_final(response)
                records.append(
                    {
                        "dataset": dataset,
                        "eval_position": int(pos),
                        "example_id": _example_id(row, dataset, pos),
                        "source_index": row.get("source_index"),
                        "prompt_id": row.get("prompt_id"),
                        "policy": policy,
                        "question": _problem_text(row),
                        "gold_final": gold,
                        "response": response,
                        "response_tokens": int(n_tok),
                        "terminated_with_eos": bool(terminated),
                        "truncated": bool(truncated),
                        "parsed_final": pred,
                        "format_compliant": bool(pred is not None),
                        "exact_reward": float(exact_reward(response, gold)),
                    }
                )
            print(f"[{dataset}/{policy}] {min(start + len(chunk), len(rows))}/{len(rows)}", flush=True)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return records


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_generation_sets(outdir: Path, dataset: str, expected_ids: list[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for policy in POLICY_ORDER:
        path = _generation_file(outdir, dataset, policy)
        if not _generation_is_complete(path, expected_ids, policy):
            raise RuntimeError(f"Incomplete/misaligned generation file: {path}")
        out[policy] = read_jsonl(path)
    return out


def _verifier_preference(policy_reward: float, sft_reward: float) -> str:
    if policy_reward > sft_reward:
        return "policy"
    if policy_reward < sft_reward:
        return "sft"
    return "tie"


def _judge_preference(raw: str) -> str:
    return {"A": "policy", "B": "sft", "TIE": "tie"}[raw]


def _pairwise_complete(path: Path, expected_ids: list[str], policy: str) -> bool:
    if not path.exists():
        return False
    rows = read_jsonl(path)
    return (
        len(rows) == len(expected_ids)
        and [str(r.get("example_id")) for r in rows] == expected_ids
        and all(r.get("policy") == policy for r in rows)
    )


def _run_pairwise(
    cfg: dict,
    outdir: Path,
    dataset: str,
    generations: dict[str, list[dict]],
    expected_ids: list[str],
    overwrite: bool,
) -> None:
    pending = []
    for policy in ("rlvr", "rlaif"):
        path = _pairwise_file(outdir, dataset, policy)
        if overwrite and path.exists():
            path.unlink()
        if path.exists():
            validate_math_prefix(read_jsonl(path), generations[policy], generations["sft"], policy)
        if not _pairwise_complete(path, expected_ids, policy):
            pending.append(policy)
        else:
            print(f"[{dataset}/{policy}] complete pairwise output exists; skipping")

    if not pending:
        return

    judge = PairwiseAIJudge(cfg, outdir / "pairwise_judge_cache.json")
    try:
        sft_rows = generations["sft"]
        for policy in pending:
            src_rows = generations[policy]
            dest = _pairwise_file(outdir, dataset, policy)
            existing = read_jsonl(dest) if dest.exists() else []
            if len(existing) > len(src_rows):
                raise RuntimeError(f"{dest} has more rows than source generations")

            for i, old in enumerate(existing):
                if (
                    str(old.get("example_id")) != expected_ids[i]
                    or old.get("policy") != policy
                ):
                    raise RuntimeError(f"{dest} is not an exact prefix of expected comparisons")

            print(f"[{dataset}/{policy}] pairwise resuming at {len(existing)}/{len(src_rows)}")
            for i in range(len(existing), len(src_rows)):
                prow = src_rows[i]
                srow = sft_rows[i]
                if str(prow["example_id"]) != str(srow["example_id"]):
                    raise RuntimeError("Policy/SFT generation order mismatch")
                raw = judge.compare(prow["question"], prow["response"], srow["response"])
                jpref = _judge_preference(raw)
                vpref = _verifier_preference(float(prow["exact_reward"]), float(srow["exact_reward"]))
                rec = {
                    "dataset": dataset,
                    "eval_position": int(i),
                    "example_id": str(prow["example_id"]),
                    "source_index": prow.get("source_index"),
                    "prompt_id": prow.get("prompt_id"),
                    "policy": policy,
                    "question": prow["question"],
                    "gold_final": prow["gold_final"],
                    "policy_response": prow["response"],
                    "sft_response": srow["response"],
                    "policy_exact_reward": float(prow["exact_reward"]),
                    "sft_exact_reward": float(srow["exact_reward"]),
                    "verifier_preference": vpref,
                    "judge_raw": raw,
                    "judge_preference": jpref,
                    "judge_score_for_policy": 1.0 if raw == "A" else 0.5 if raw == "TIE" else 0.0,
                    "verifier_judge_agree": bool(vpref == jpref),
                }
                _append_jsonl(dest, rec)
                if (i + 1) % 10 == 0 or i + 1 == len(src_rows):
                    print(f"[{dataset}/{policy}] pairwise {i + 1}/{len(src_rows)}", flush=True)
    finally:
        del judge
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _policy_summary(generations: dict[str, list[dict]], pairwise: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for policy in POLICY_ORDER:
        df = pd.DataFrame(generations[policy])
        lengths = df["response_tokens"].astype(float)
        rec = {
            "policy": policy,
            "n": int(len(df)),
            "exact_accuracy": float(df["exact_reward"].mean()),
            "format_compliance": float(df["format_compliant"].mean()),
            "mean_response_tokens": float(lengths.mean()),
            "std_response_tokens": float(lengths.std(ddof=0)),
            "median_response_tokens": float(lengths.median()),
            "iqr_response_tokens": float(lengths.quantile(0.75) - lengths.quantile(0.25)),
            "truncation_rate": float(df["truncated"].mean()),
        }
        if policy == "sft":
            rec.update(
                {
                    "ai_win_rate_vs_sft": 0.5,
                    "judge_tie_rate_vs_sft": 1.0,
                    "verifier_judge_agreement_vs_sft": float("nan"),
                    "verifier_judge_decisive_agreement_vs_sft": float("nan"),
                    "judge_breaks_verifier_tie_rate": float("nan"),
                }
            )
        else:
            pdf = pd.DataFrame(pairwise[policy])
            verifier_decisive = pdf[pdf["verifier_preference"] != "tie"]
            verifier_ties = pdf[pdf["verifier_preference"] == "tie"]
            rec.update(
                {
                    "ai_win_rate_vs_sft": float(pdf["judge_score_for_policy"].mean()),
                    "judge_tie_rate_vs_sft": float((pdf["judge_preference"] == "tie").mean()),
                    "verifier_judge_agreement_vs_sft": float(pdf["verifier_judge_agree"].mean()),
                    "verifier_judge_decisive_agreement_vs_sft": (
                        float(verifier_decisive["verifier_judge_agree"].mean())
                        if len(verifier_decisive)
                        else float("nan")
                    ),
                    "judge_breaks_verifier_tie_rate": (
                        float((verifier_ties["judge_preference"] != "tie").mean())
                        if len(verifier_ties)
                        else float("nan")
                    ),
                }
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def _pairwise_summary(pairwise: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for policy in ("rlvr", "rlaif"):
        df = pd.DataFrame(pairwise[policy])
        for vpref in ("policy", "sft", "tie"):
            subset = df[df["verifier_preference"] == vpref]
            if len(subset) == 0:
                continue
            rows.append(
                {
                    "policy": policy,
                    "verifier_preference": vpref,
                    "n": int(len(subset)),
                    "judge_policy_rate": float((subset["judge_preference"] == "policy").mean()),
                    "judge_sft_rate": float((subset["judge_preference"] == "sft").mean()),
                    "judge_tie_rate": float((subset["judge_preference"] == "tie").mean()),
                    "agreement_rate": float(subset["verifier_judge_agree"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _failure_summary(generations: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for policy in POLICY_ORDER:
        df = pd.DataFrame(generations[policy])
        labels = np.where(
            df["exact_reward"].astype(float) == 1.0,
            "correct",
            np.where(df["format_compliant"].astype(bool), "wrong_final", "format_failure"),
        )
        counts = pd.Series(labels).value_counts()
        for failure_type in ("correct", "wrong_final", "format_failure"):
            n = int(counts.get(failure_type, 0))
            rows.append(
                {
                    "policy": policy,
                    "failure_type": failure_type,
                    "count": n,
                    "rate": float(n / len(df)),
                }
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--overwrite-generation", action="store_true")
    ap.add_argument("--overwrite-pairwise", action="store_true")
    args = ap.parse_args()

    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    _validate_rows(rows, args.dataset)
    expected_ids = _expected_ids(rows, args.dataset)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)

    print("Dataset:", args.dataset)
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))

    for policy in POLICY_ORDER:
        dest = _generation_file(outdir, args.dataset, policy)
        if args.overwrite_generation and dest.exists():
            dest.unlink()
        if _generation_is_complete(dest, expected_ids, policy):
            print(f"[{args.dataset}/{policy}] complete generation exists; skipping")
            continue
        if dest.exists():
            raise RuntimeError(
                f"{dest} exists but is incomplete/misaligned. Use --overwrite-generation to regenerate."
            )
        generated = _generate_policy(cfg, rows, tokenizer, args.dataset, policy, args.batch_size)
        got_ids = [str(r["example_id"]) for r in generated]
        if got_ids != expected_ids:
            raise RuntimeError(f"{policy}: generated IDs/order do not match fixed evaluation set")
        write_jsonl(dest, generated)
        print(f"[{args.dataset}/{policy}] wrote {len(generated)} rows -> {dest}")

    generations = _load_generation_sets(outdir, args.dataset, expected_ids)
    _run_pairwise(
        cfg,
        outdir,
        args.dataset,
        generations,
        expected_ids,
        overwrite=args.overwrite_pairwise,
    )

    pairwise = {
        policy: read_jsonl(_pairwise_file(outdir, args.dataset, policy))
        for policy in ("rlvr", "rlaif")
    }
    for policy, prows in pairwise.items():
        if len(prows) != len(rows):
            raise RuntimeError(f"{policy}: incomplete pairwise evaluation")

    summary = _policy_summary(generations, pairwise)
    pair_summary = _pairwise_summary(pairwise)
    failures = _failure_summary(generations)

    summary_path = outdir / f"{args.dataset}_policy_summary.csv"
    pair_path = outdir / f"{args.dataset}_verifier_judge_summary.csv"
    failure_path = outdir / f"{args.dataset}_failure_summary.csv"
    summary.to_csv(summary_path, index=False)
    pair_summary.to_csv(pair_path, index=False)
    failures.to_csv(failure_path, index=False)

    manifest = {
        "dataset": args.dataset,
        "n_examples": len(rows),
        "seed": int(cfg["seed"]),
        "generation": {
            "max_prompt_length": int(cfg.get("math_max_prompt_length", 512)),
            "max_new_tokens": int(cfg["math_max_new_tokens"]),
            "temperature": float(cfg.get("generation", {}).get("temperature", 0.7)),
            "top_p": float(cfg.get("generation", {}).get("top_p", 0.9)),
            "do_sample": bool(cfg.get("generation", {}).get("do_sample", True)),
        },
        "policy_summary": summary.replace({np.nan: None}).to_dict(orient="records"),
    }
    (outdir / f"{args.dataset}_summary.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nPolicy summary:")
    print(summary.to_string(index=False))
    print("\nWrote:")
    print(summary_path)
    print(pair_path)
    print(failure_path)
    print(outdir / f"{args.dataset}_summary.json")


if __name__ == "__main__":
    main()
