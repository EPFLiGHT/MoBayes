#!/bin/bash
# DDXPlus — Patient persona ablation: 4 personas x {BMBE, SA} with GPT-5.4-nano, n=25 each.
# Persona profiles: data/cases/ddxplus_patientsim_a/<persona>/profiles.json
# (made by scripts/data_prep/generate_profiles.py).
# Usage: bash experiments/ddxplus/persona_ablation.sh
set -euo pipefail
[ -f .env ] && source .env || true

DATA_DIR="data/kbs/ddxplus"
PROFILE_DIR="data/cases/ddxplus_patientsim_a"
OUTPUT="data/benchmark_results/set_a/ddxplus_patientsim/ablations"
N=25
MODEL="gpt-5.4-nano"
PATIENT="gpt-5.4-nano"

PERSONAS=("overanxious" "distrustful" "dazed" "verbose_lowrecall")

# 8 conditions, three at a time.
i=0
for mode in bmbe sa; do
  if [ "$mode" = "bmbe" ]; then
    mode_args=(--doctor-mode mobayes --doctor-model "$MODEL" --patient-model "$PATIENT" --force-max-turns)
  else
    mode_args=(--doctor-mode standalone --doctor-model "$MODEL" --patient-model "$PATIENT")
  fi
  for persona in "${PERSONAS[@]}"; do
    echo "--- ${mode} + ${persona} ---"
    python3 scripts/core/benchmark_patientsim.py \
      --profiles "$PROFILE_DIR/${persona}/profiles.json" \
      --data-dir "$DATA_DIR" \
      "${mode_args[@]}" \
      --n $N \
      --output "$OUTPUT/persona_${persona}_${mode}" &
    i=$((i + 1))
    if [ $((i % 3)) -eq 0 ]; then wait; fi
  done
done
wait
echo "=== All persona ablations done ==="
