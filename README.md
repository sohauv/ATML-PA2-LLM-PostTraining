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

This repository contains implemented pipelines and saved machine-readable experimental results for ATML PA2. Section 7 records remaining pre-report checks. It builds on the course starter infrastructure and fixed course assets.

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
# Standard DPO
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard --qualitative-limit 40

# Beta ablations
python -m task1_dpo.ablate_beta --config configs/dpo.yaml

# Length-balanced DPO
python -m task1_dpo.train --config configs/dpo.yaml --run-name length_balanced --dataset data/dpo_length_balanced_train.jsonl --output outputs/task1_dpo/length_balanced

# Length-confounding evaluation
python -m task1_dpo.analyze_length --config configs/dpo.yaml --standard-adapter outputs/task1_dpo/standard --balanced-adapter outputs/task1_dpo/length_balanced
```

### Task 2 - PPO

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
# Cached-rollout diagnostic (does not launch online forks)
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml

# Online forks: each reloads the supplied midpoint, then evaluates its adapter.
# Existing fork files are protected; --overwrite deliberately regenerates them.
python -m task2_ppo.ablate_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml

# Reaggregate saved results, tables, trajectories, and figures; no model loading.
python -m task2_ppo.finalize_results --config configs/ppo.yaml
```

### Task 3 - GRPO

```bash
# Standard 20-update GRPO continuation
python -m task3_grpo.continue_train \
  --config configs/grpo.yaml \
  --output outputs/task3_grpo/standard \
  --run-name standard

# Standard held-out evaluation
python -m task3_grpo.evaluate \
  --config configs/grpo.yaml \
  --adapter outputs/task3_grpo/standard \
  --name standard

# Equal-generation K = 2, 4, 8 diagnostic
python -m task3_grpo.analyze_group_size \
  --config configs/grpo.yaml

# 8-update canonical / Dr-GRPO forks with equal scheduled caps
python -m task3_grpo.compare_normalization \
  --config configs/grpo.yaml

# Held-out evaluation of both normalization forks
python -m task3_grpo.evaluate \
  --config configs/grpo.yaml \
  --adapter outputs/task3_grpo/normalization_canonical \
  --name normalization_canonical

python -m task3_grpo.evaluate \
  --config configs/grpo.yaml \
  --adapter outputs/task3_grpo/normalization_dr_grpo \
  --name normalization_dr_grpo

# Validate matched evaluation protocol and produce Task 3 evidence tables
python -m task3_grpo.finalize_results
```

### Task 4 - Safety calibration

The four frozen policies are evaluated on the fixed XSTest prompt set with identical deterministic decoding. Create the blind manual-audit sheet before inspecting any AI-judge labels.

```bash
# Generate one deterministic response for every XSTest prompt from
# SFT, standard DPO, standard PPO, and standard GRPO.
python -m task4_safety.generate_responses \
  --config configs/feedback.yaml \
  --policy all \
  --batch-size 4

# Create the fixed 60-prompt blind manual-audit subset.
python -m task4_safety.make_audit_sheet \
  --config configs/feedback.yaml

# Manually fill results/task4_safety/manual_audit.csv using only:
# SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE,
# OVER_REFUSAL, or AMBIGUOUS.
# Do not inspect judged_*.jsonl before completing the manual labels.

# Apply the fixed categorical AI safety judge to all four policies.
python -m task4_safety.judge_responses \
  --config configs/feedback.yaml \
  --policy all

# Produce policy summaries, category distributions, and
# manual-vs-AI agreement/confusion outputs.
python -m task4_safety.evaluate_safety \
  --config configs/feedback.yaml
```

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

## 7. Pre-report corrections and remaining checks

The standard DPO training filter has been restored from `task1_standard_backup(1).zip`; its standard training log and adapter are byte-identical to this repository's run. The restored filter records 1,500 raw pairs, 1,446 retained pairs, and 54 exclusions. Recovery provenance is in `results/validation/pre_report_fix_status.json`. The length-balanced training log and training filter remain missing; evaluation and adapter files do not establish the training budget. Locate their originals before claiming complete Task 1 training evidence. Do not substitute a beta-run filter or synthesize a training log.

Archived PPO and GRPO results used `--likelihood-mode legacy`: rollout likelihoods in eval mode, differentiable update likelihoods in train mode, with LoRA dropout 0.05. That mode remains the default solely to reproduce those archived experiments. New logs and summaries explicitly record the mode. `--likelihood-mode eval` disables dropout for both likelihood forwards while retaining autograd. It is a corrected option, not the implementation that generated the archived results. Use new names/output directories for corrected experiments. The PPO fork wrappers require a suffix in eval mode, for example `--likelihood-mode eval --suffix _evalmode`; the report finalizer accepts the same `--suffix _evalmode` after the standard run and all six forks have been produced with that suffix and mode. Do not mix corrected-mode adapters with old generations or judgements.

Before deciding whether to replace experimental results, run this small diagnostic on the original GPU environment with the supplied midpoint checkpoints installed:

```bash
python -m scripts.check_policy_ratios --task all
```

This loads policy/tokenizer only, uses four prompts per checkpoint and three repeated train-mode forwards, checks unchanged-weight token ratios and eval-mode gradients, and performs zero optimizer steps. Save `results/validation/policy_ratio_checks.json`. The diagnostic is sample-limited and does not retroactively certify every rollout. It has not been executed in the review environment, which lacks PyTorch, PEFT, the base model, and the supplied checkpoints.

The normalization forks match eight updates and 16,384 scheduled maximum tokens each. Actual generated tokens are 8,425 versus 9,182; training-retained tokens are 4,329 versus 5,598. These are not equal actual generated-token budgets. Report the distinction and the existing matched-rollout counterfactual. If the assignment's fixed-budget requirement means exact realized tokens, that comparison still needs a revised experimental protocol and new runs; a wording change does not fulfill it.

The safety judge labels all 44 manually identified over-refusals as justified refusals. Preserve its prescribed labels and qualify automated zero-over-refusal rates using the audit. GRPO updates 5, 13, 15, and 17 contain no retained tokens. Task 5 accuracy and format failures must be reported separately; its AI judge mostly ties and does not show reasoning sensitivity in the direct controlled pairs. These observations require accurate synthesis, not changing the supplied judge/verifier.

If responses are regenerated, rejudge them too. Safety and math judgement resumption now checks source response text and metadata, not just IDs; stale source-dependent files raise an explicit overwrite instruction. Generation files still do not certify checkpoint identity from IDs alone, so deliberately regenerate them after model/config changes.

For a safe command preview without training:

```bash
python -m task2_ppo.ablate_kl --dry-run
python -m task2_ppo.ablate_clipping --dry-run
```
