#!/bin/bash
# DDXPlus — Prior distribution ablation (BMBE + GPT-5.4-nano), n=25 per condition.
# Compares empirical (default), uniform and conditional (age x sex) priors, in parallel.
# Usage: bash experiments/ddxplus/prior_ablation.sh
set -euo pipefail
[ -f .env ] && source .env || true

PROFILES="data/cases/ddxplus_patientsim_a/plain/profiles.json"
DATA_DIR="data/kbs/ddxplus"
OUTPUT="data/benchmark_results/set_a/ddxplus_patientsim/ablations"
N=25
MODEL="gpt-5.4-nano"
PATIENT="gpt-5.4-nano"

for prior in empirical uniform conditional; do
  echo "=== ${prior} prior ==="
  python3 scripts/core/benchmark_patientsim.py \
    --profiles "$PROFILES" --data-dir "$DATA_DIR" \
    --doctor-mode mobayes --doctor-model "$MODEL" \
    --patient-model "$PATIENT" \
    --prior-mode "$prior" \
    --force-max-turns --n $N \
    --output "$OUTPUT/prior_${prior}" &
done
wait
echo "=== All prior ablations done ==="
