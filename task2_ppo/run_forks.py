from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

from common.data import REPO_ROOT, load_yaml, repo_path


def condition_name(study: str, value: float) -> str:
    prefix = "kl_beta" if study == "kl" else "clip_eps"
    return f"{prefix}_{int(round(value * 100)):03d}"


def commands(study, cfg, config_path, train_only=False, eval_only=False,
             likelihood_mode="legacy", suffix=""):
    values = cfg["kl_values" if study == "kl" else "clip_values"]
    output_root = repo_path(cfg["output"]).parent
    for value in values:
        name = condition_name(study, float(value)) + suffix
        output = output_root / name
        if not eval_only:
            yield [sys.executable, "-m", "task2_ppo.continue_train",
                   "--config", config_path, "--updates", str(cfg["fork_updates"]),
                   "--clip-epsilon", str(value if study == "clipping" else cfg["clip_epsilon"]),
                   "--kl-beta", str(value if study == "kl" else cfg["kl_beta"]),
                   "--run-name", name, "--output", str(output),
                   "--likelihood-mode", likelihood_mode]
        if not train_only:
            yield [sys.executable, "-m", "task2_ppo.evaluate", "--config", config_path,
                   "--adapter", str(output), "--name", name]


def main(study):
    ap = argparse.ArgumentParser(description=f"Run midpoint PPO {study} forks in fresh processes.")
    ap.add_argument("--config", default="configs/ppo.yaml")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--train-only", action="store_true")
    group.add_argument("--eval-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="Allow replacement of existing fork files.")
    ap.add_argument("--likelihood-mode", choices=["legacy", "eval"], default="legacy")
    ap.add_argument("--suffix", default="", help="Separate names for a new implementation, e.g. _evalmode.")
    args = ap.parse_args()
    if args.likelihood_mode == "eval" and not args.suffix:
        ap.error("Use --suffix _evalmode to keep corrected-mode experiments separate from archived results.")
    cfg = load_yaml(args.config)
    if not args.dry_run and not args.overwrite:
        existing = []
        for value in cfg["kl_values" if study == "kl" else "clip_values"]:
            name = condition_name(study, float(value)) + args.suffix
            if not args.eval_only:
                existing += [repo_path(cfg["results_dir"]) / f"{name}_train.jsonl",
                             repo_path(cfg["output"]).parent / name]
            if not args.train_only:
                existing += [repo_path(cfg["results_dir"]) / f"{name}_generations.jsonl",
                             repo_path(cfg["results_dir"]) / f"{name}_eval_summary.json"]
        present = [str(p) for p in existing if p.exists()]
        if present:
            ap.error("Existing results found; use a suffix or --overwrite:\n" + "\n".join(present))
    for cmd in commands(study, cfg, args.config, args.train_only, args.eval_only,
                        args.likelihood_mode, args.suffix):
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
