"""Check unchanged-weight likelihood ratios; never performs an optimizer step."""
from __future__ import annotations

import argparse
import gc
import math

import torch

from common.data import load_yaml, prompt_messages, read_jsonl
from common.generation import batch_generate, response_token_logprobs
from common.logging_utils import save_json, set_seed
from common.models import load_policy, load_tokenizer


def check(config, checkpoint, samples, repetitions):
    cfg = load_yaml(config)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=checkpoint, trainable=True)
    rows = read_jsonl(cfg["paths"]["rl_prompt_train"])[:samples]
    dropout = [{"name": name, "p": float(module.p)}
               for name, module in policy.named_modules()
               if isinstance(module, torch.nn.Dropout) and module.p > 0]
    records = []
    for row in rows:
        generated = batch_generate(policy, tokenizer, [prompt_messages(row)],
                                   max_prompt_length=int(cfg["max_prompt_length"]),
                                   max_new_tokens=32, do_sample=False)
        # Inference tensors from generate must be copied before an autograd forward.
        tensors = [generated[key].clone() for key in
                   ["sequences", "attention_mask", "response_ids"]]
        seq, attn, response = tensors
        mask = generated["response_mask"].clone().bool()
        width = generated["prompt_width"]
        policy.eval()
        with torch.no_grad():
            old, _ = response_token_logprobs(policy, seq, attn, width, response)
        eval_logp, _ = response_token_logprobs(policy, seq, attn, width, response)
        eval_ratio = (eval_logp.detach() - old).exp()[mask]
        policy.zero_grad(set_to_none=True)
        (-eval_logp[mask].mean()).backward()
        gradients = [p.grad for p in policy.parameters() if p.requires_grad and p.grad is not None]
        finite_gradients = bool(gradients) and all(torch.isfinite(g).all().item() for g in gradients)
        gradient_norm = sum(float(g.float().square().sum().item()) for g in gradients) ** 0.5
        eval_error = float((eval_ratio - 1).abs().max().item())
        del eval_logp, eval_ratio, gradients
        policy.zero_grad(set_to_none=True)
        train_checks = []
        for repetition in range(repetitions):
            set_seed(int(cfg["seed"]) + repetition)
            policy.train()
            with torch.no_grad():
                new, _ = response_token_logprobs(policy, seq, attn, width, response)
                ratio = (new - old).exp()[mask]
            train_checks.append({"max_abs_ratio_minus_one": float((ratio - 1).abs().max().item()),
                                 "mean_abs_ratio_minus_one": float((ratio - 1).abs().mean().item()),
                                 "outside_epsilon_005_fraction": float(((ratio < .95) | (ratio > 1.05)).float().mean().item())})
        records.append({"prompt_id": row.get("prompt_id"), "source_index": row.get("source_index"),
                        "valid_tokens": int(mask.sum().item()), "eval_max_abs_ratio_minus_one": eval_error,
                        "eval_gradients_finite": finite_gradients, "eval_gradient_norm": gradient_norm,
                        "legacy_train_mode_checks": train_checks})
    result = {"checkpoint": checkpoint, "seed": cfg["seed"], "optimizer_steps": 0,
              "generation_cap": 32, "active_dropout_modules": dropout, "records": records,
              "eval_checks_pass": all(r["eval_max_abs_ratio_minus_one"] <= 1e-5
                                      and r["eval_gradients_finite"] and math.isfinite(r["eval_gradient_norm"])
                                      and r["eval_gradient_norm"] > 0 for r in records),
              "scope": "Small unchanged-weight diagnostic; not a validation of every archived rollout."}
    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["ppo", "grpo", "all"], default="all")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--output", default="results/validation/policy_ratio_checks.json")
    args = ap.parse_args()
    if args.samples < 1 or args.repetitions < 1:
        ap.error("samples and repetitions must be positive")
    result = {}
    for task in (["ppo", "grpo"] if args.task == "all" else [args.task]):
        config = f"configs/{task}.yaml"
        cfg = load_yaml(config)
        result[task] = check(config, cfg["paths"][f"{task}_midpoint_policy"],
                             args.samples, args.repetitions)
    save_json(args.output, result)
    print(f"Saved {args.output}; no optimizer steps were performed.")
    for task, record in result.items():
        print(task, "eval checks pass:", record["eval_checks_pass"])


if __name__ == "__main__":
    main()
