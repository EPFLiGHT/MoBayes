#!/bin/bash
# DDXPlus — Prior distribution ablation (BMBE + GPT-5.4-nano).
# Compares: empirical (default), uniform, conditional (age x sex) priors.
# n=25 per condition, 3 runs in parallel.
# Usage: bash experiments/ddxplus/prior_ablation.sh
set -euo pipefail
source .env

PROFILES="data/cases/ddxplus_patientsim_a/plain/profiles.json"
DATA_DIR="data/kbs/ddxplus"
OUTPUT="data/benchmark_results/set_a/ddxplus_patientsim/ablations"
N=25
MODEL="gpt-5.4-nano"
PATIENT="gpt-5.4-nano"

# ── 1. Empirical prior (default / baseline) ──
echo "=== [1/3] Empirical prior (baseline) ==="
python3 scripts/core/benchmark_patientsim.py \
  --profiles "$PROFILES" --data-dir "$DATA_DIR" \
  --doctor-mode mobayes --doctor-model "$MODEL" \
  --patient-model "$PATIENT" \
  --prior-mode empirical \
  --force-max-turns --n $N \
  --output "$OUTPUT/prior_empirical" &

# ── 2. Uniform prior (1/N) ──
echo "=== [2/3] Uniform prior ==="
python3 scripts/core/benchmark_patientsim.py \
  --profiles "$PROFILES" --data-dir "$DATA_DIR" \
  --doctor-mode mobayes --doctor-model "$MODEL" \
  --patient-model "$PATIENT" \
  --prior-mode uniform \
  --force-max-turns --n $N \
  --output "$OUTPUT/prior_uniform" &

# ── 3. Conditional prior (age x sex from DDXPlus train) ──
echo "=== [3/3] Conditional prior (age x sex) ==="
python3 scripts/core/benchmark_patientsim.py \
  --profiles "$PROFILES" --data-dir "$DATA_DIR" \
  --doctor-mode mobayes --doctor-model "$MODEL" \
  --patient-model "$PATIENT" \
  --prior-mode conditional \
  --force-max-turns --n $N \
  --output "$OUTPUT/prior_conditional" &

wait
echo "=== All prior ablations done ==="
