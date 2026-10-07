from __future__ import annotations

import argparse
import subprocess
import sys

from common.data import load_yaml


def beta_name(beta: float) -> str:
    return f"beta_{int(round(beta * 100)):03d}"


def run_command(cmd):
    print()
    print("=" * 80)
    print("Running:")
    print(" ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/dpo.yaml",
    )

    ap.add_argument(
        "--train-only",
        action="store_true",
    )

    ap.add_argument(
        "--eval-only",
        action="store_true",
    )

    args = ap.parse_args()

    if args.train_only and args.eval_only:
        ap.error(
            "--train-only and --eval-only "
            "cannot be used together"
        )

    cfg = load_yaml(args.config)

    betas = [
        float(x)
        for x in cfg["betas"]
    ]

    max_examples = int(
        cfg["short_ablation_examples"]
    )

    print("DPO beta conditions:", betas)
    print(
        "Short-run raw examples per condition:",
        max_examples,
    )

    for beta in betas:
        name = beta_name(beta)

        output = (
            f"outputs/task1_dpo/{name}"
        )

        if not args.eval_only:
            run_command(
                [
                    sys.executable,
                    "-m",
                    "task1_dpo.train",
                    "--config",
                    args.config,
                    "--run-name",
                    name,
                    "--output",
                    output,
                    "--beta",
                    str(beta),
                    "--max-examples",
                    str(max_examples),
                ]
            )

        if not args.train_only:
            run_command(
                [
                    sys.executable,
                    "-m",
                    "task1_dpo.evaluate",
                    "--config",
                    args.config,
                    "--adapter",
                    output,
                    "--name",
                    name,
                    "--beta",
                    str(beta),
                ]
            )

    print()
    print("DPO beta study complete.")


if __name__ == "__main__":
    main()