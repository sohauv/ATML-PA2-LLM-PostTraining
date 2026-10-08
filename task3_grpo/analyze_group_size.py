from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np
import torch

from common.data import (
    load_yaml,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.logging_utils import save_json
from task3_grpo.grpo import group_relative_advantages


TOL = 1e-6


def load_k8_cache(path):
    rows = read_jsonl(path)

    by_prompt = defaultdict(list)

    for row in rows:
        by_prompt[str(row["source_index"])].append(row)

    # Instructor cache has eight completions per prompt.
    bad = {
        pid: len(group)
        for pid, group in by_prompt.items()
        if len(group) < 8
    }

    if bad:
        raise ValueError(
            "Expected at least K=8 cached completions "
            f"per prompt; short groups: {bad}"
        )

    for group in by_prompt.values():
        group.sort(
            key=lambda x: int(
                x.get(
                    "generation_index",
                    0,
                )
            )
        )

        prompt_ids = {
            str(row["prompt_id"])
            for row in group
        }

        if len(prompt_ids) != 1:
            raise ValueError(
                "A source_index maps to multiple prompt_ids: "
                f"{sorted(prompt_ids)}"
            )

    return by_prompt


def regroup_equal_generation_budget(
    by_prompt,
    k: int,
):
    """Partition the same eight cached generations per prompt into K-sized groups.

    K=2 -> four groups/prompt
    K=4 -> two groups/prompt
    K=8 -> one group/prompt

    Thus every K condition uses every one of the same cached completions,
    so the total generation budget is identical.
    """
    if k not in {2, 4, 8}:
        raise ValueError(
            f"Expected K in {{2,4,8}}, got {k}"
        )

    if 8 % k != 0:
        raise ValueError(
            f"K={k} does not divide the eight-completion cache"
        )

    groups = []

    for source_key in sorted(
        by_prompt,
        key=lambda x: int(x),
    ):
        # Use exactly the fixed eight generations for every prompt.
        rows = by_prompt[source_key][:8]

        for subgroup_index, start in enumerate(
            range(0, 8, k)
        ):
            chunk = rows[
                start:
                start + k
            ]

            if len(chunk) != k:
                raise RuntimeError(
                    "Unexpected short regrouped chunk"
                )

            groups.append(
                {
                    "source_index":
                        int(
                            chunk[0][
                                "source_index"
                            ]
                        ),

                    "prompt_id":
                        str(
                            chunk[0][
                                "prompt_id"
                            ]
                        ),

                    "k":
                        int(k),

                    "subgroup_index":
                        int(
                            subgroup_index
                        ),

                    "rows":
                        chunk,
                }
            )

    return groups


def build_prompt_difficulty_bins(
    by_prompt,
):
    """Rank prompts by mean reward across all eight cached completions.

    Lowest third = hard, middle third = medium, highest third = easy.
    Ranking is computed once from the full K=8 cache and reused for all K,
    so the difficulty definition does not change across conditions.
    """
    prompt_stats = []

    for source_key, rows in (
        by_prompt.items()
    ):
        fixed_rows = rows[:8]

        rewards = np.asarray(
            [
                float(row["reward"])
                for row in fixed_rows
            ],
            dtype=float,
        )

        prompt_stats.append(
            {
                "source_index":
                    int(
                        fixed_rows[0][
                            "source_index"
                        ]
                    ),

                "prompt_id":
                    str(
                        fixed_rows[0][
                            "prompt_id"
                        ]
                    ),

                "prompt_mean_reward":
                    float(
                        rewards.mean()
                    ),

                "prompt_reward_std":
                    float(
                        rewards.std(
                            ddof=0
                        )
                    ),
            }
        )

    # Deterministic tie-break by source index.
    prompt_stats.sort(
        key=lambda row: (
            row[
                "prompt_mean_reward"
            ],
            row[
                "source_index"
            ],
        )
    )

    chunks = np.array_split(
        np.arange(
            len(prompt_stats)
        ),
        3,
    )

    labels = [
        "hard",
        "medium",
        "easy",
    ]

    assignment = {}

    for label, indices in zip(
        labels,
        chunks,
    ):
        for idx in indices:
            row = prompt_stats[
                int(idx)
            ]

            row[
                "difficulty_bin"
            ] = label

            assignment[
                str(
                    row[
                        "source_index"
                    ]
                )
            ] = {
                "difficulty_bin":
                    label,

                "prompt_mean_reward":
                    row[
                        "prompt_mean_reward"
                    ],
            }

    return (
        prompt_stats,
        assignment,
    )


def attach_group_statistics(
    groups,
    difficulty_assignment,
):
    """Compute reward variation and GRPO relative signal for each group."""
    rewards_flat = []
    group_ids_flat = []

    for group_id, group in enumerate(
        groups
    ):
        for row in group["rows"]:
            rewards_flat.append(
                float(
                    row["reward"]
                )
            )

            group_ids_flat.append(
                group_id
            )

    reward_tensor = torch.tensor(
        rewards_flat,
        dtype=torch.float32,
    )

    group_id_tensor = torch.tensor(
        group_ids_flat,
        dtype=torch.long,
    )

    advantages = (
        group_relative_advantages(
            reward_tensor,
            group_id_tensor,
            eps=TOL,
        )
        .detach()
        .cpu()
        .numpy()
    )

    cursor = 0
    records = []

    for group_id, group in enumerate(
        groups
    ):
        k = int(
            group["k"]
        )

        rows = group["rows"]

        rewards = np.asarray(
            [
                float(row["reward"])
                for row in rows
            ],
            dtype=float,
        )

        signals = advantages[
            cursor:
            cursor + k
        ]

        cursor += k

        reward_std = float(
            rewards.std(
                ddof=0
            )
        )

        source_key = str(
            group[
                "source_index"
            ]
        )

        difficulty = (
            difficulty_assignment[
                source_key
            ]
        )

        record = {
            "group_id":
                int(group_id),

            "k":
                k,

            "source_index":
                int(
                    group[
                        "source_index"
                    ]
                ),

            "prompt_id":
                group[
                    "prompt_id"
                ],

            "subgroup_index":
                int(
                    group[
                        "subgroup_index"
                    ]
                ),

            "difficulty_bin":
                difficulty[
                    "difficulty_bin"
                ],

            "prompt_mean_reward":
                float(
                    difficulty[
                        "prompt_mean_reward"
                    ]
                ),

            "generation_indices":
                [
                    int(
                        row[
                            "generation_index"
                        ]
                    )
                    for row in rows
                ],

            "completion_tokens":
                [
                    int(
                        row[
                            "completion_tokens"
                        ]
                    )
                    for row in rows
                ],

            "clipped_at_max":
                [
                    bool(
                        row[
                            "clipped_at_max"
                        ]
                    )
                    for row in rows
                ],

            "rewards":
                rewards.tolist(),

            "reward_mean":
                float(
                    rewards.mean()
                ),

            "reward_std":
                reward_std,

            "informative":
                bool(
                    reward_std > TOL
                ),

            "relative_signals":
                [
                    float(x)
                    for x in signals
                ],
        }

        records.append(
            record
        )

    return records


def summarize_group_records(
    records,
):
    if not records:
        raise ValueError(
            "Cannot summarize an empty group list"
        )

    reward_stds = np.asarray(
        [
            row["reward_std"]
            for row in records
        ],
        dtype=float,
    )

    informative = np.asarray(
        [
            float(
                row["informative"]
            )
            for row in records
        ],
        dtype=float,
    )

    all_signals = np.asarray(
        [
            signal
            for row in records
            for signal in row[
                "relative_signals"
            ]
        ],
        dtype=float,
    )

    informative_signals = np.asarray(
        [
            signal
            for row in records
            if row["informative"]
            for signal in row[
                "relative_signals"
            ]
        ],
        dtype=float,
    )

    all_rewards = np.asarray(
        [
            reward
            for row in records
            for reward in row[
                "rewards"
            ]
        ],
        dtype=float,
    )

    completion_tokens = np.asarray(
        [
            n
            for row in records
            for n in row[
                "completion_tokens"
            ]
        ],
        dtype=float,
    )

    clipped = np.asarray(
        [
            float(flag)
            for row in records
            for flag in row[
                "clipped_at_max"
            ]
        ],
        dtype=float,
    )

    unique_prompts = {
        row["prompt_id"]
        for row in records
    }

    return {
        "num_prompts":
            int(
                len(
                    unique_prompts
                )
            ),

        "num_groups":
            int(
                len(records)
            ),

        "num_completions":
            int(
                len(
                    all_signals
                )
            ),

        "informative_group_rate":
            float(
                informative.mean()
            ),

        "uninformative_group_fraction":
            float(
                1.0
                - informative.mean()
            ),

        "mean_within_group_reward_std":
            float(
                reward_stds.mean()
            ),

        # Primary requested relative-signal statistic:
        # variance across every completion in the equal-budget condition.
        "relative_signal_variance":
            float(
                all_signals.var(
                    ddof=0
                )
            ),

        # Secondary audit value: variance after excluding zero-signal
        # groups. This helps show whether a change in the primary variance
        # is driven mainly by the fraction of uninformative groups.
        "relative_signal_variance_informative_only":
            (
                float(
                    informative_signals.var(
                        ddof=0
                    )
                )
                if len(
                    informative_signals
                ) > 0
                else 0.0
            ),

        "mean_reward":
            float(
                all_rewards.mean()
            ),

        "mean_completion_tokens":
            float(
                completion_tokens.mean()
            ),

        "clipped_at_max_fraction":
            float(
                clipped.mean()
            ),
    }


def write_csv(
    path,
    rows,
    fieldnames,
):
    path = repo_path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=
                fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(key)
                    for key
                    in fieldnames
                }
            )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/grpo.yaml",
    )

    args = ap.parse_args()

    cfg = load_yaml(
        args.config
    )

    by_prompt = load_k8_cache(
        cfg["group_cache"]
    )

    print(
        "Cached prompts:",
        len(by_prompt),
    )

    print(
        "Cached completions:",
        sum(
            len(rows[:8])
            for rows
            in by_prompt.values()
        ),
    )

    print(
        "Group sizes to analyze:",
        cfg["group_sizes"],
    )

    prompt_bins, difficulty_assignment = (
        build_prompt_difficulty_bins(
            by_prompt
        )
    )

    results_dir = repo_path(
        cfg["results_dir"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_group_records = []
    overall_rows = []
    difficulty_rows = []

    for k in cfg[
        "group_sizes"
    ]:
        k = int(k)

        groups = (
            regroup_equal_generation_budget(
                by_prompt,
                k,
            )
        )

        records = (
            attach_group_statistics(
                groups,
                difficulty_assignment,
            )
        )

        all_group_records.extend(
            records
        )

        overall = (
            summarize_group_records(
                records
            )
        )

        overall["k"] = k

        overall_rows.append(
            overall
        )

        for difficulty_bin in (
            "hard",
            "medium",
            "easy",
        ):
            subset = [
                row
                for row in records
                if row[
                    "difficulty_bin"
                ]
                == difficulty_bin
            ]

            summary = (
                summarize_group_records(
                    subset
                )
            )

            summary["k"] = k
            summary[
                "difficulty_bin"
            ] = difficulty_bin

            difficulty_rows.append(
                summary
            )

    # Sanity check: every K must consume exactly the same cached
    # completion budget.
    budgets = {
        row[
            "num_completions"
        ]
        for row in overall_rows
    }

    if len(budgets) != 1:
        raise AssertionError(
            "Group-size conditions do not use an equal "
            f"generation budget: {sorted(budgets)}"
        )

    summary = {
        "cache":
            cfg["group_cache"],

        "numerical_tolerance":
            TOL,

        "difficulty_binning_rule":
            (
                "Rank prompts once by mean reward across their "
                "eight fixed cached completions; lowest third=hard, "
                "middle third=medium, highest third=easy. "
                "Ties are broken by source_index."
            ),

        "regrouping_rule":
            (
                "For each prompt, sort the fixed eight completions by "
                "generation_index and partition consecutive completions "
                "into non-overlapping K-sized groups. This yields "
                "4/2/1 groups per prompt for K=2/4/8 and uses every "
                "cached completion exactly once in every condition."
            ),

        "total_cached_prompts":
            int(
                len(by_prompt)
            ),

        "equal_generation_budget":
            int(
                next(
                    iter(
                        budgets
                    )
                )
            ),

        "overall":
            overall_rows,

        "by_difficulty":
            difficulty_rows,
    }

    write_jsonl(
        results_dir
        / "group_size_groups.jsonl",
        all_group_records,
    )

    write_jsonl(
        results_dir
        / "group_size_prompt_bins.jsonl",
        prompt_bins,
    )

    save_json(
        results_dir
        / "group_size_summary.json",
        summary,
    )

    overall_fields = [
        "k",
        "num_prompts",
        "num_groups",
        "num_completions",
        "informative_group_rate",
        "uninformative_group_fraction",
        "mean_within_group_reward_std",
        "relative_signal_variance",
        "relative_signal_variance_informative_only",
        "mean_reward",
        "mean_completion_tokens",
        "clipped_at_max_fraction",
    ]

    difficulty_fields = [
        "k",
        "difficulty_bin",
        "num_prompts",
        "num_groups",
        "num_completions",
        "informative_group_rate",
        "uninformative_group_fraction",
        "mean_within_group_reward_std",
        "relative_signal_variance",
        "relative_signal_variance_informative_only",
        "mean_reward",
        "mean_completion_tokens",
        "clipped_at_max_fraction",
    ]

    write_csv(
        results_dir
        / "group_size_table.csv",
        overall_rows,
        overall_fields,
    )

    write_csv(
        results_dir
        / "group_size_difficulty_table.csv",
        difficulty_rows,
        difficulty_fields,
    )

    print()
    print(
        "Difficulty bins:"
    )

    for label in (
        "hard",
        "medium",
        "easy",
    ):
        rows = [
            row
            for row in prompt_bins
            if row[
                "difficulty_bin"
            ] == label
        ]

        means = [
            row[
                "prompt_mean_reward"
            ]
            for row in rows
        ]

        print(
            f"  {label:6s}: "
            f"n={len(rows)} "
            f"mean-reward range="
            f"[{min(means):.4f}, "
            f"{max(means):.4f}]"
        )

    print()
    print(
        "Overall equal-generation results:"
    )

    header = (
        "K  groups  completions  informative  "
        "mean_group_std  signal_var"
    )

    print(header)

    for row in overall_rows:
        print(
            f"{row['k']:<2d} "
            f"{row['num_groups']:<7d} "
            f"{row['num_completions']:<12d} "
            f"{row['informative_group_rate']:<11.4f} "
            f"{row['mean_within_group_reward_std']:<14.4f} "
            f"{row['relative_signal_variance']:.4f}"
        )

    print()
    print(
        "Difficulty-bin results:"
    )

    print(
        "K  bin     groups  informative  "
        "mean_group_std  signal_var"
    )

    for row in difficulty_rows:
        print(
            f"{row['k']:<2d} "
            f"{row['difficulty_bin']:<7s} "
            f"{row['num_groups']:<7d} "
            f"{row['informative_group_rate']:<11.4f} "
            f"{row['mean_within_group_reward_std']:<14.4f} "
            f"{row['relative_signal_variance']:.4f}"
        )

    print()
    print(
        "Saved:"
    )

    for filename in (
        "group_size_summary.json",
        "group_size_table.csv",
        "group_size_difficulty_table.csv",
        "group_size_groups.jsonl",
        "group_size_prompt_bins.jsonl",
    ):
        print(
            " ",
            results_dir
            / filename,
        )


if __name__ == "__main__":
    main()
