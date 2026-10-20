"""Validate that resumed judgements still refer to their saved source text."""
from __future__ import annotations


def validate_safety_prefix(existing, source, policy):
    if len(existing) > len(source):
        raise RuntimeError("Judgements contain more rows than source generations")
    for old, current in zip(existing, source):
        if old.get("policy") != policy or any(old.get(k) != current.get(k)
                for k in ["xstest_id", "prompt", "response", "benchmark_class", "type"]):
            raise RuntimeError("Safety judgement source changed; rerun judge_responses with --overwrite.")


def validate_math_prefix(existing, policy_rows, sft_rows, policy):
    if len(existing) > len(policy_rows) or len(policy_rows) != len(sft_rows):
        raise RuntimeError("Pairwise judgement/source lengths disagree")
    for old, current, baseline in zip(existing, policy_rows, sft_rows):
        matches = (str(old.get("example_id")) == str(current.get("example_id"))
                   == str(baseline.get("example_id"))
                   and old.get("policy") == policy
                   and old.get("question") == current.get("question") == baseline.get("question")
                   and old.get("gold_final") == current.get("gold_final") == baseline.get("gold_final")
                   and old.get("policy_response") == current.get("response")
                   and old.get("sft_response") == baseline.get("response")
                   and old.get("policy_exact_reward") == current.get("exact_reward")
                   and old.get("sft_exact_reward") == baseline.get("exact_reward"))
        if not matches:
            raise RuntimeError("Math judgement source changed; rerun evaluate_math with --overwrite-pairwise.")
