# ATML PA2 - LLM Post-Training

<!-- FINAL_STUDENT_SETUP -->

## Quick start

```bash
git clone https://github.com/sohauv/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

The fixed datasets, cached diagnostics, and supplied
continuation checkpoints are downloaded from:

https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets

Pinned release revision:

`0b350481fb03f5525a35bcdec4131bd4fe487f98`

---

This repository contains implemented pipelines and saved machine-readable experimental results for ATML PA2, built on the [course starter repository](https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining) and fixed course assets. Section 7 documents recovered experiment provenance and remaining interpretation limitations.

For Tasks 1-3, the training/ablation pipelines were completed and the deliberate objective defects in DPO, PPO, and GRPO were corrected before running the required experiments. For Task 4, the supplied frozen safety judge is used with the implemented generation, aggregation, and manual-audit pipeline. For Task 5, the supplied exact RLVR verifier and fixed pairwise RLAIF judge are used unchanged with the implemented evaluation and analysis pipeline.

Student-generated machine-readable outputs are stored under `results/`. Large downloaded checkpoints, raw course assets, model caches, and regenerated training outputs remain excluded from Git where appropriate.

## 1. Clone and install

```bash
git clone https://github.com/sohauv/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
```

## 2. Download the course assets

The large course-created checkpoints and fixed data are downloaded from the pinned course asset release on Hugging Face. After cloning, run:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

Public base/reward/judge models are downloaded from Hugging Face at runtime and are **not** included in the course asset archive.

The installer also materializes the fixed 100-example Task 5 transfer set from the official SVAMP challenge-set source if it is not already present. The tiny Task 1 word-limit prompt set is tracked directly in this repository.

## 3. Environment check

```bash
python -m scripts.check_environment
```

Run commands from the repository root. The reference environment used to prepare the release pins Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, and Tokenizers 0.22.1.

## 4. Supplied course checkpoints

After `download_assets`, these directories should exist:

```text
checkpoints/ppo_midpoint_policy/
checkpoints/ppo_midpoint_value/
checkpoints/grpo_midpoint_policy/
checkpoints/rlvr_policy/
checkpoints/rlaif_policy/
```

PPO and GRPO begin from the supplied continuation checkpoints. RLVR and RLAIF are supplied frozen evaluation policies; students do not retrain them.

The PPO value checkpoint is intentionally released as the exact staff midpoint state, including its imperfect held-out value calibration. Treat critic behavior as an analysis variable rather than assuming a perfect baseline, and start every PPO fork from the identical supplied policy/value state. The default continuation generation cap is 512 tokens for feasibility; frozen evaluation uses the larger cap specified in `configs/ppo.yaml`.

## 5. Task entry points

### Task 1 - DPO

```bash
# Standard, one-epoch DPO run and fixed held-out evaluation
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard --qualitative-limit 40

# Three fixed-budget beta forks (0.03, 0.10, 0.30)
python -m task1_dpo.ablate_beta --config configs/dpo.yaml

# Recovered replacement length-balanced run: its log and filtering record
# are saved as results/task1_dpo/length_balanced_replacement_{train.jsonl,filtered_prompt_indices.json}.
python -m task1_dpo.train \
  --config configs/dpo.yaml \
  --run-name length_balanced_replacement \
  --dataset data/dpo_length_balanced_train.jsonl \
  --output outputs/task1_dpo/length_balanced_replacement

# Compare standard versus the replacement adapter on fixed length strata
# and word-limit prompts; results are in results/task1_dpo/length_replacement/.
python -m task1_dpo.analyze_length \
  --config configs/dpo_length_replacement.yaml \
  --standard-adapter outputs/task1_dpo/standard \
  --balanced-adapter outputs/task1_dpo/length_balanced_replacement
```

### Task 2 - PPO

The **report-ready corrected experiments** use `--likelihood-mode eval` for both
rollout and differentiable-update likelihoods. Preserve the `_evalmode` suffix
so the older legacy-mode artifacts are not silently substituted.

```bash
# Standard 20-update PPO continuation from the supplied policy/value midpoint
python -m task2_ppo.continue_train \
  --config configs/ppo.yaml \
  --run-name standard_evalmode \
  --output outputs/task2_ppo/standard_evalmode \
  --likelihood-mode eval

python -m task2_ppo.evaluate \
  --config configs/ppo.yaml \
  --adapter outputs/task2_ppo/standard_evalmode \
  --name standard_evalmode

# Frozen cached-rollout clipping diagnostic: epsilon = 0.05, 0.20, 0.50
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml

# Six matched 8-update online forks, each from the same supplied midpoint.
# The wrappers also run held-out evaluation for each completed fork.
python -m task2_ppo.ablate_clipping \
  --config configs/ppo.yaml --likelihood-mode eval --suffix _evalmode
python -m task2_ppo.ablate_kl \
  --config configs/ppo.yaml --likelihood-mode eval --suffix _evalmode

# Rebuild PPO report tables and figures from these corrected logs.
python -m task2_ppo.finalize_results \
  --config configs/ppo.yaml --suffix _evalmode
```

Running the fork wrappers again when outputs exist requires deliberate handling
of their overwrite protection; do not overwrite saved results just to check commands.

### Task 3 - GRPO

The corrected standard and normalization-fork runs likewise use eval-mode
likelihoods. The current `compare_normalization` training wrapper and
`finalize_results` aggregate **legacy unsuffixed** runs; neither should be used
to regenerate or label corrected `_evalmode` results. The direct commands below
match the corrected training/evaluation entry points and write the reported
`*_evalmode_*` records.

```bash
# Corrected standard 20-update GRPO continuation
python -m task3_grpo.continue_train \
  --config configs/grpo.yaml \
  --run-name standard_evalmode \
  --output outputs/task3_grpo/standard_evalmode \
  --likelihood-mode eval

python -m task3_grpo.evaluate \
  --config configs/grpo.yaml \
  --adapter outputs/task3_grpo/standard_evalmode \
  --name standard_evalmode

# Equal-generation cached study of K = 2, 4, 8; no model training
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml

# Corrected matched 8-update normalization forks
python -m task3_grpo.continue_train \
  --config configs/grpo.yaml --updates 8 \
  --loss-type grpo \
  --run-name normalization_canonical_evalmode \
  --output outputs/task3_grpo/normalization_canonical_evalmode \
  --likelihood-mode eval

python -m task3_grpo.continue_train \
  --config configs/grpo.yaml --updates 8 \
  --loss-type dr_grpo \
  --run-name normalization_dr_grpo_evalmode \
  --output outputs/task3_grpo/normalization_dr_grpo_evalmode \
  --likelihood-mode eval

python -m task3_grpo.evaluate \
  --config configs/grpo.yaml \
  --adapter outputs/task3_grpo/normalization_canonical_evalmode \
  --name normalization_canonical_evalmode

python -m task3_grpo.evaluate \
  --config configs/grpo.yaml \
  --adapter outputs/task3_grpo/normalization_dr_grpo_evalmode \
  --name normalization_dr_grpo_evalmode
```

The saved corrected normalization summaries, trajectories, and held-out
generations are in `results/task3_grpo/` under their `_evalmode` names. The
fixed group-size diagnostic uses the same supplied completion/reward cache.

### Task 4 - Safety calibration

Safety compares the SFT baseline, standard one-epoch DPO, and the **corrected
standard PPO/GRPO** adapters. Use `configs/feedback_evalmode.yaml` for **every**
stage; `configs/feedback.yaml` still points at the archived unsuffixed PPO/GRPO
adapters. Complete the human audit without seeing AI judgments first.

```bash
# Generate 450 deterministic responses for each of four fixed policies
python -m task4_safety.generate_responses \
  --config configs/feedback_evalmode.yaml --policy all --batch-size 4

# Prepare the fixed, blind 60-prompt manual audit (240 policy-response rows)
python -m task4_safety.make_audit_sheet \
  --config configs/feedback_evalmode.yaml

# Human reviewer completes results/task4_safety/manual_audit.csv using the
# five released behavior labels before inspecting AI judgments.

python -m task4_safety.judge_responses \
  --config configs/feedback_evalmode.yaml --policy all

python -m task4_safety.evaluate_safety \
  --config configs/feedback_evalmode.yaml
```

The final stored manual audit preserves 216 existing labels only when prompt
and response text matched exactly; 24 changed PPO/GRPO responses were manually
re-labeled. Do not mix these corrected judged responses with the legacy audit.

### Task 5 - RLVR vs RLAIF

The supplied exact verifier and fixed pairwise AI judge are used unchanged. The commands below reproduce the in-domain GSM8K evaluation, controlled reward diagnostic study, fixed SVAMP transfer evaluation, and final cross-domain aggregation.

```bash
# In-domain GSM8K evaluation: SFT, RLVR, and RLAIF
python -m task5_feedback.evaluate_math \
  --config configs/feedback.yaml \
  --dataset gsm \
  --batch-size 4

# Controlled 100-response reward diagnostic study
python -m task5_feedback.score_perturbations \
  --config configs/feedback.yaml

# Fixed 100-example SVAMP transfer evaluation
python -m task5_feedback.evaluate_math \
  --config configs/feedback.yaml \
  --dataset transfer \
  --batch-size 4

# Aggregate in-domain, diagnostic, and transfer evidence
python -m task5_feedback.compare_feedback \
  --config configs/feedback.yaml
```

## 6. Reproducibility rules

- Do not alter course-provided data, cached rollouts, or supplied checkpoints.
- Start every short fork from the **same supplied midpoint checkpoint**.
- Keep prompt IDs, generated-token/update budgets, seed, and evaluation procedure matched across ablations.
- Commit your code, configs, small JSON/CSV logs, and figures. Do not commit downloaded checkpoints, raw course assets, or model caches.
- Record peak VRAM and wall-clock time for the standard PPO and GRPO continuations.

See the assignment manual for the required experiments, metrics, and report questions.

## 7. Recovered provenance and remaining interpretation checks

- **Task 1 provenance:** The standard DPO run retained 1,446 of 1,500 pairs
  (54 filtered). The earlier length-balanced run's original training record
  could not be recovered, so a **new one-epoch length-balanced replacement**
  was run from the specified initialization. Its logs and indices are present:
  `results/task1_dpo/length_balanced_replacement_train.jsonl` and
  `length_balanced_replacement_filtered_prompt_indices.json` (1,442/1,500
  kept, 58 filtered). The resulting fixed held-out and word-limit comparison,
  adapter hash provenance, and generation records are in
  `results/task1_dpo/length_replacement/`. Results attributed to length-balanced
  DPO in the report should use that **replacement**, not the earlier adapter.
- **Task 2 provenance:** The corrected standard PPO run and clipping/KL forks
  carry the `_evalmode` suffix and record `likelihood_mode: eval`. Keep them
  distinct from the older legacy-mode logs. The PPO finalizer accepts
  `--suffix _evalmode`; the corrected JSON summaries were compared with the
  Kaggle backup and were semantically identical even when byte formatting
  differed. Regenerated PNG images may have different plotting metadata or
  presentation and should be reviewed before report submission.
- **Task 3 provenance:** Corrected GRPO standard and normalization training
  logs, held-out summaries, and generations use `_evalmode`. Both normalization
  forks have eight updates and equal **scheduled maximum-token** budgets;
  **realized** generated tokens differ (9,280 canonical versus 9,578 Dr-GRPO),
  as do tokens retained for training (5,696 versus 6,506). Do not describe
  this as exactly matched realized-token compute. The released comparison and
  finalizer code presently operate on unsuffixed legacy runs; corrected
  comparisons should reference the named `_evalmode` summaries and matching
  held-out examples directly. The group-size cache study is independent of
  this likelihood-mode correction.
- **Task 4 provenance:** The corrected standard PPO and GRPO policies were
  evaluated alongside SFT and standard DPO on all 450 XSTest prompts. The
  corrected manual audit has 240 labels (60 prompt IDs × 4 policies), with
  216 response-matched labels reused and 24 newly reviewed responses; overall
  manual-versus-AI agreement is 171/240 = 71.25%. The manual audit labels
  44 over-refusals, although the frozen categorical AI judge labels none as
  over-refusals. Keep the automated and manual evidence distinguished. All
  corrected outputs use `results/task4_safety/` and
  `configs/feedback_evalmode.yaml`.
- **Task 5 provenance:** Use the fixed supplied RLVR verifier, AI judge,
  diagnostic response set, and frozen RLVR/RLAIF adapters unchanged. The final
  GSM8K, SVAMP, controlled diagnostics, and judge agreement files are in
  `results/task5_feedback/`.
- **Additional diagnostic:** On the original GPU environment, optionally run
  `python -m scripts.check_policy_ratios --task all` to verify the corrected
  train/eval likelihood handling without optimizer updates. Do not claim that
  this diagnostic has passed unless its output was actually checked and saved.

The reproducibility commands above describe fresh runs. Adapters and course
assets are intentionally excluded from Git, while machine-readable result
files, prompt IDs, configurations, and source are retained. See the report
for interpretation and limitations of each comparison.
