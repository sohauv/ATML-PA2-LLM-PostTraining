from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import save_json, set_seed
from common.metrics import sampled_kl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen = []
        rejected = []

        for row in rows:
            prompt = prompt_messages_from_preference(row)
            chosen_response, rejected_response = preference_responses(row)

            chosen.append(
                encode_prompt_response(
                    tokenizer,
                    prompt,
                    chosen_response,
                    max_length,
                )
            )

            rejected.append(
                encode_prompt_response(
                    tokenizer,
                    prompt,
                    rejected_response,
                    max_length,
                )
            )

        return (
            pad_batch(tokenizer, chosen),
            pad_batch(tokenizer, rejected),
        )

    return collate


def load_evaluation_bundle(
    config_path: str,
    adapter: str,
):
    cfg = load_yaml(config_path)

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )

    policy = load_policy(
        cfg,
        adapter_path=adapter,
        trainable=False,
    )

    rows = read_jsonl(
        cfg["paths"]["dpo_standard_eval"]
    )

    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "policy": policy,
    }


def filter_rows_for_length(
    rows,
    tokenizer,
    max_length,
):
    kept_rows = []
    kept_indices = []
    dropped = []

    for index, row in enumerate(rows):
        prompt = prompt_messages_from_preference(row)

        prompt_ids = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
        )

        if len(prompt_ids) >= max_length:
            dropped.append(
                {
                    "index": int(index),
                    "prompt_tokens": int(
                        len(prompt_ids)
                    ),
                }
            )
            continue

        kept_rows.append(row)
        kept_indices.append(index)

    return (
        kept_rows,
        kept_indices,
        dropped,
    )


@torch.no_grad()
def evaluate_preference_pairs(
    policy,
    tokenizer,
    rows,
    cfg,
    beta,
):
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=make_collate(
            tokenizer,
            int(cfg["max_sequence_length"]),
        ),
    )

    device = next(
        policy.parameters()
    ).device

    total_examples = 0
    loss_sum = 0.0
    correct_sum = 0.0

    for chosen, rejected in loader:
        chosen = {
            key: value.to(device)
            for key, value in chosen.items()
        }

        rejected = {
            key: value.to(device)
            for key, value in rejected.items()
        }

        with reference_mode(policy):
            ref_chosen_logp, _, _ = (
                response_sequence_logprobs(
                    policy,
                    chosen,
                )
            )

            ref_rejected_logp, _, _ = (
                response_sequence_logprobs(
                    policy,
                    rejected,
                )
            )

        policy_chosen_logp, _, _ = (
            response_sequence_logprobs(
                policy,
                chosen,
            )
        )

        policy_rejected_logp, _, _ = (
            response_sequence_logprobs(
                policy,
                rejected,
            )
        )

        loss, diagnostics = dpo_loss(
            policy_chosen_logp,
            policy_rejected_logp,
            ref_chosen_logp,
            ref_rejected_logp,
            beta,
        )

        batch_size = chosen[
            "input_ids"
        ].shape[0]

        total_examples += batch_size

        loss_sum += (
            float(loss.item())
            * batch_size
        )

        correct_sum += (
            float(
                diagnostics[
                    "preference_accuracy"
                ].item()
            )
            * batch_size
        )

    return {
        "dpo_loss":
            loss_sum / total_examples,

        "preference_accuracy":
            correct_sum / total_examples,

        "num_examples":
            total_examples,
    }


@torch.no_grad()
def evaluate_generation(
    policy,
    tokenizer,
    rows,
    kept_indices,
    cfg,
):
    generation_cfg = cfg["generation"]

    batch_size = int(
        cfg["batch_size"]
    )

    records = []

    kl_numerator = 0.0
    kl_denominator = 0.0

    for start in range(
        0,
        len(rows),
        batch_size,
    ):
        batch_rows = rows[
            start:start + batch_size
        ]

        batch_indices = kept_indices[
            start:start + batch_size
        ]

        prompts = [
            prompt_messages_from_preference(row)
            for row in batch_rows
        ]

        generated = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=int(
                cfg["max_sequence_length"]
            ),
            max_new_tokens=int(
                cfg["max_generation_tokens"]
            ),
            temperature=float(
                generation_cfg["temperature"]
            ),
            top_p=float(
                generation_cfg["top_p"]
            ),
            do_sample=bool(
                generation_cfg["do_sample"]
            ),
        )

        policy_logp, _ = (
            response_token_logprobs(
                policy,
                generated["sequences"],
                generated["attention_mask"],
                generated["prompt_width"],
                generated["response_ids"],
            )
        )

        with reference_mode(policy):
            reference_logp, _ = (
                response_token_logprobs(
                    policy,
                    generated["sequences"],
                    generated["attention_mask"],
                    generated["prompt_width"],
                    generated["response_ids"],
                )
            )

        response_mask = generated[
            "response_mask"
        ]

        batch_kl = sampled_kl(
            policy_logp,
            reference_logp,
            response_mask,
        )

        valid_tokens = float(
            response_mask.sum().item()
        )

        kl_numerator += (
            float(batch_kl.item())
            * valid_tokens
        )

        kl_denominator += valid_tokens

        for i, response in enumerate(
            generated["responses"]
        ):
            row = batch_rows[i]

            records.append(
                {
                    "prompt_index":
                        start + i,

                    "eval_index":
                        int(batch_indices[i]),

                    "prompt_id":
                        row.get("prompt_id"),

                    "source_index":
                        row.get(
                            "source_index",
                            batch_indices[i],
                        ),

                    "response":
                        response,

                    "response_length":
                        int(
                            generated[
                                "response_lengths"
                            ][i]
                        ),

                    "terminated_with_eos":
                        bool(
                            generated[
                                "terminated_with_eos"
                            ][i]
                        ),

                    "truncated":
                        bool(
                            generated[
                                "truncated"
                            ][i]
                        ),
                }
            )

    lengths = np.asarray(
        [
            record["response_length"]
            for record in records
        ],
        dtype=float,
    )

    metrics = {
        "kl_from_reference":
            kl_numerator
            / max(
                kl_denominator,
                1.0,
            ),

        "mean_response_length":
            float(
                lengths.mean()
            ),

        "std_response_length":
            float(
                lengths.std()
            ),

        "median_response_length":
            float(
                np.median(lengths)
            ),

        "iqr_response_length":
            float(
                np.percentile(
                    lengths,
                    75,
                )
                - np.percentile(
                    lengths,
                    25,
                )
            ),
    }

    return records, metrics


@torch.no_grad()
def score_generation_records(
    reward_model,
    reward_tokenizer,
    rows,
    generation_records,
    batch_size,
):
    scores = []

    for start in range(
        0,
        len(rows),
        batch_size,
    ):
        batch_rows = rows[
            start:start + batch_size
        ]

        prompts = [
            prompt_messages_from_preference(row)
            for row in batch_rows
        ]

        responses = [
            record["response"]
            for record in generation_records[
                start:start + batch_size
            ]
        ]

        batch_scores = (
            score_reward_pairs(
                reward_model,
                reward_tokenizer,
                prompts,
                responses,
            )
        )

        scores.extend(
            batch_scores
            .detach()
            .cpu()
            .tolist()
        )

    return [
        float(score)
        for score in scores
    ]


def attach_reward_scores(
    generation_records,
    scores,
):
    if len(generation_records) != len(scores):
        raise RuntimeError(
            "Number of generated responses "
            "does not match number of reward scores."
        )

    for record, score in zip(
        generation_records,
        scores,
    ):
        record[
            "reward_model_score"
        ] = float(score)


def reward_metrics(scores):
    scores_array = np.asarray(
        scores,
        dtype=float,
    )

    return {
        "reward_model_score_mean":
            float(
                scores_array.mean()
            ),

        "reward_model_score_std":
            float(
                scores_array.std()
            ),
    }


@torch.no_grad()
def generate_reference_records(
    cfg,
    policy,
    tokenizer,
    rows,
    kept_indices,
):
    generation_cfg = cfg["generation"]

    batch_size = int(
        cfg["batch_size"]
    )

    records = []

    for start in range(
        0,
        len(rows),
        batch_size,
    ):
        batch_rows = rows[
            start:start + batch_size
        ]

        batch_indices = kept_indices[
            start:start + batch_size
        ]

        prompts = [
            prompt_messages_from_preference(row)
            for row in batch_rows
        ]

        with reference_mode(policy):
            generated = batch_generate(
                policy,
                tokenizer,
                prompts,
                max_prompt_length=int(
                    cfg[
                        "max_sequence_length"
                    ]
                ),
                max_new_tokens=int(
                    cfg[
                        "max_generation_tokens"
                    ]
                ),
                temperature=float(
                    generation_cfg[
                        "temperature"
                    ]
                ),
                top_p=float(
                    generation_cfg[
                        "top_p"
                    ]
                ),
                do_sample=bool(
                    generation_cfg[
                        "do_sample"
                    ]
                ),
            )

        for i, response in enumerate(
            generated["responses"]
        ):
            row = batch_rows[i]

            records.append(
                {
                    "eval_index":
                        int(
                            batch_indices[i]
                        ),

                    "prompt_id":
                        row.get(
                            "prompt_id"
                        ),

                    "source_index":
                        row.get(
                            "source_index",
                            batch_indices[i],
                        ),

                    "response":
                        response,

                    "response_length":
                        int(
                            generated[
                                "response_lengths"
                            ][i]
                        ),
                }
            )

    return records


@torch.no_grad()
def build_qualitative_candidates(
    cfg,
    policy,
    tokenizer,
    reward_model,
    reward_tokenizer,
    rows,
    kept_indices,
    policy_generation_records,
    limit,
    output_path,
):
    """
    Generate reference-policy responses for the same fixed
    prompts and compare their reward-model scores with the
    evaluated DPO policy.

    The output is intended for manual inspection of
    reward-versus-quality agreement/disagreement cases.
    """

    num_examples = min(
        int(limit),
        len(rows),
    )

    if num_examples <= 0:
        return 0

    qualitative_rows = rows[
        :num_examples
    ]

    qualitative_indices = kept_indices[
        :num_examples
    ]

    policy_records = (
        policy_generation_records[
            :num_examples
        ]
    )

    # Reset the seed so the reference condition begins
    # from the same fixed RNG state as policy generation.
    set_seed(
        int(cfg["seed"])
    )

    reference_records = (
        generate_reference_records(
            cfg,
            policy,
            tokenizer,
            qualitative_rows,
            qualitative_indices,
        )
    )

    batch_size = int(
        cfg["batch_size"]
    )

    reference_scores = (
        score_generation_records(
            reward_model,
            reward_tokenizer,
            qualitative_rows,
            reference_records,
            batch_size,
        )
    )

    attach_reward_scores(
        reference_records,
        reference_scores,
    )

    candidates = []

    for i in range(num_examples):
        row = qualitative_rows[i]

        policy_record = (
            policy_records[i]
        )

        reference_record = (
            reference_records[i]
        )

        policy_score = float(
            policy_record[
                "reward_model_score"
            ]
        )

        reference_score = float(
            reference_record[
                "reward_model_score"
            ]
        )

        candidates.append(
            {
                "input_position":
                    int(i),

                "eval_index":
                    int(
                        qualitative_indices[i]
                    ),

                "prompt_id":
                    row.get(
                        "prompt_id"
                    ),

                "source_index":
                    row.get(
                        "source_index",
                        qualitative_indices[i],
                    ),

                "prompt":
                    prompt_messages_from_preference(
                        row
                    ),

                "reference_response":
                    reference_record[
                        "response"
                    ],

                "policy_response":
                    policy_record[
                        "response"
                    ],

                "reference_reward":
                    reference_score,

                "policy_reward":
                    policy_score,

                "reward_gain":
                    policy_score
                    - reference_score,

                "reference_length":
                    int(
                        reference_record[
                            "response_length"
                        ]
                    ),

                "policy_length":
                    int(
                        policy_record[
                            "response_length"
                        ]
                    ),
            }
        )

    candidates.sort(
        key=lambda candidate:
            candidate["reward_gain"],
        reverse=True,
    )

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        candidate[
            "reward_gain_rank"
        ] = int(rank)

    write_jsonl(
        output_path,
        candidates,
    )

    print(
        f"Saved {len(candidates)} "
        f"qualitative candidates to "
        f"{output_path}"
    )

    return len(candidates)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/dpo.yaml",
    )

    parser.add_argument(
        "--adapter",
        required=True,
    )

    parser.add_argument(
        "--name",
        default="standard",
    )

    parser.add_argument(
        "--beta",
        type=float,
    )

    parser.add_argument(
        "--qualitative-limit",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    bundle = load_evaluation_bundle(
        args.config,
        args.adapter,
    )

    cfg = bundle["cfg"]
    rows = bundle["rows"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]

    eval_beta = float(
        cfg["beta"]
        if args.beta is None
        else args.beta
    )

    max_length = int(
        cfg["max_sequence_length"]
    )

    (
        kept_rows,
        kept_indices,
        dropped,
    ) = filter_rows_for_length(
        rows,
        tokenizer,
        max_length,
    )

    print(
        f"DPO eval rows: "
        f"{len(rows)} total, "
        f"{len(kept_rows)} kept, "
        f"{len(dropped)} dropped "
        f"for prompt length >= "
        f"{max_length}"
    )

    pair_metrics = (
        evaluate_preference_pairs(
            policy,
            tokenizer,
            kept_rows,
            cfg,
            eval_beta,
        )
    )

    # Every DPO condition begins stochastic generation
    # from exactly the same released random seed.
    set_seed(
        int(cfg["seed"])
    )

    (
        generation_records,
        generation_metrics,
    ) = evaluate_generation(
        policy,
        tokenizer,
        kept_rows,
        kept_indices,
        cfg,
    )

    # Load the fixed reward model once and reuse it for
    # both standard scoring and qualitative diagnostics.
    reward_model, reward_tokenizer = (
        load_reward_model(cfg)
    )

    batch_size = int(
        cfg["batch_size"]
    )

    policy_reward_scores = (
        score_generation_records(
            reward_model,
            reward_tokenizer,
            kept_rows,
            generation_records,
            batch_size,
        )
    )

    attach_reward_scores(
        generation_records,
        policy_reward_scores,
    )

    generation_reward_metrics = (
        reward_metrics(
            policy_reward_scores
        )
    )

    results_dir = repo_path(
        cfg["results_dir"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    qualitative_count = 0

    if args.qualitative_limit > 0:
        qualitative_count = (
            build_qualitative_candidates(
                cfg,
                policy,
                tokenizer,
                reward_model,
                reward_tokenizer,
                kept_rows,
                kept_indices,
                generation_records,
                args.qualitative_limit,
                results_dir
                / (
                    "qualitative_reward_"
                    "candidates.jsonl"
                ),
            )
        )

    summary = {
        "name":
            args.name,

        "adapter":
            args.adapter,

        "beta":
            eval_beta,

        "seed":
            int(
                cfg["seed"]
            ),

        "max_sequence_length":
            max_length,

        "max_generation_tokens":
            int(
                cfg[
                    "max_generation_tokens"
                ]
            ),

        "num_original_eval_rows":
            len(rows),

        "num_eval_rows":
            len(kept_rows),

        "num_dropped_rows":
            len(dropped),

        "qualitative_candidate_count":
            qualitative_count,

        **pair_metrics,
        **generation_metrics,
        **generation_reward_metrics,
    }

    save_json(
        results_dir
        / f"{args.name}_summary.json",
        summary,
    )

    save_json(
        results_dir
        / f"{args.name}_eval_filter.json",
        {
            "kept_indices":
                kept_indices,

            "dropped":
                dropped,
        },
    )

    write_jsonl(
        results_dir
        / f"{args.name}_generations.jsonl",
        generation_records,
    )

    print()
    print("Evaluation complete.")
    print()

    for key, value in summary.items():
        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()