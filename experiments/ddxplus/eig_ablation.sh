#!/bin/bash
# DDXPlus — EIG strategy ablation (BMBE + GPT-5.4-nano), n=25 per condition.
# Compares global EIG with focused top-k EIG (k, λ, θ varied).
# Usage: bash experiments/ddxplus/eig_ablation.sh
set -euo pipefail
[ -f .env ] && source .env || true

PROFILES="data/cases/ddxplus_patientsim_a/plain/profiles.json"
DATA_DIR="data/kbs/ddxplus"
OUTPUT="data/benchmark_results/set_a/ddxplus_patientsim/ablations"
N=25
MODEL="gpt-5.4-nano"
PATIENT="gpt-5.4-nano"

# run NAME [EIG flags...]: one condition, in the background.
run() {
  local name="$1"; shift
  echo "=== ${name} ==="
  python3 scripts/core/benchmark_patientsim.py \
    --profiles "$PROFILES" --data-dir "$DATA_DIR" \
    --doctor-mode mobayes --doctor-model "$MODEL" \
    --patient-model "$PATIENT" \
    --force-max-turns --n $N \
    "$@" \
    --output "$OUTPUT/${name}" &
}

# Three conditions at a time.
run eig_global           --eig-topk-weight 0.0                                                    # global EIG (λ=0)
run eig_topk3_lam03      --eig-topk-size 3 --eig-topk-weight 0.3 --eig-focus-conf-threshold 0.3
run eig_topk3_lam05      --eig-topk-size 3 --eig-topk-weight 0.5 --eig-focus-conf-threshold 0.3
wait
run eig_topk5_lam03      --eig-topk-size 5 --eig-topk-weight 0.3 --eig-focus-conf-threshold 0.3
run eig_topk3_lam03_th05 --eig-topk-size 3 --eig-topk-weight 0.3 --eig-focus-conf-threshold 0.5  # later activation
run eig_topk3_lam10      --eig-topk-size 3 --eig-topk-weight 1.0 --eig-focus-conf-threshold 0.3  # strong focus
wait
echo "=== All EIG ablations done ==="
