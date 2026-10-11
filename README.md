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

This is the **student starter repository** for ATML PA2. The released code is intentionally incomplete: Tasks 1-3 provide model/data loading, objective helpers, checkpoint restoration, and experiment entry points, but **you must implement the training loops and ablation orchestration yourself**. Each of Tasks 1-3 also contains one deliberate algorithmic defect in its core objective code; identifying and correcting these defects is part of validating your implementation.

Task 4 supplies the fixed AI safety judge and response-generation utilities, but you must write the evaluation/aggregation code. Task 5 supplies the exact RLVR verifier, the fixed pairwise AI judge used for RLAIF evaluation, and data/model loaders; you must implement the requested evaluation and analysis.

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

If your instructor provides a direct asset URL separately, use:

```bash
python -m scripts.download_assets --url '<ASSET_URL>'
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

# Standard DPO
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard --qualitative-limit 40

# Beta ablations
python -m task1_dpo.ablate_beta --config configs/dpo.yaml

# Length-balanced DPO
python -m task1_dpo.train --config configs/dpo.yaml --run-name length_balanced --dataset data/dpo_length_balanced_train.jsonl --output outputs/task1_dpo/length_balanced

# Length-confounding evaluation
python -m task1_dpo.analyze_length --config configs/dpo.yaml --standard-adapter outputs/task1_dpo/standard --balanced-adapter outputs/task1_dpo/length_balanced

### Task 2 - PPO

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml
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

# Matched 8-update canonical GRPO / Dr-GRPO forks
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

The exact verifier and pairwise AI judge are supplied; you implement the evaluation/analysis.

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

## 6. Reproducibility rules

- Do not alter course-provided data, cached rollouts, or supplied checkpoints.
- Start every short fork from the **same supplied midpoint checkpoint**.
- Keep prompt IDs, generated-token/update budgets, seed, and evaluation procedure matched across ablations.
- Commit your code, configs, small JSON/CSV logs, and figures. Do not commit downloaded checkpoints, raw course assets, or model caches.
- Record peak VRAM and wall-clock time for the standard PPO and GRPO continuations.

See the assignment manual for the required experiments, metrics, and report questions.
