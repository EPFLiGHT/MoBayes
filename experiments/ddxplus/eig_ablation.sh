#!/bin/bash
# DDXPlus — EIG strategy ablation (BMBE + GPT-5.4-nano).
# Compares: global EIG, focused top-k EIG (different lambda & k values).
# n=25 per condition for faster iteration.
# Usage: bash experiments/ddxplus/eig_ablation.sh
set -euo pipefail
source .env

PROFILES="data/cases/ddxplus_patientsim_a/plain/profiles.json"
DATA_DIR="data/kbs/ddxplus"
OUTPUT="data/benchmark_results/set_a/ddxplus_patientsim/ablations"
N=25
MODEL="gpt-5.4-nano"
PATIENT="gpt-5.4-nano"

# ── 1. Global EIG (baseline, lambda=0) ──
echo "=== [1/6] Global EIG (baseline) ==="
python3 scripts/core/benchmark_patientsim.py \
  --profiles "$PROFILES" --data-dir "$DATA_DIR" \
  --doctor-mode mobayes --doctor-model "$MODEL" \
  --patient-model "$PATIENT" \
  --force-max-turns --n $N \
  --eig-topk-weight 0.0 \
  --output "$OUTPUT/eig_global" &

# ── 2. Focused top-3 EIG (lambda=0.3, threshold=0.3) ──
echo "=== [2/6] Focused top-3 EIG (λ=0.3, θ=0.3) ==="
python3 scripts/core/benchmark_patientsim.py \
  --profiles "$PROFILES" --data-dir "$DATA_DIR" \
  --doctor-mode mobayes --doctor-model "$MODEL" \
  --patient-model "$PATIENT" \
  --force-max-turns --n $N \
  --eig-topk-size 3 --eig-topk-weight 0.3 --eig-focus-conf-threshold 0.3 \
  --output "$OUTPUT/eig_topk3_lam03" &

# ── 3. Focused top-3 EIG (lambda=0.5, threshold=0.3) ──
echo "=== [3/6] Focused top-3 EIG (λ=0.5, θ=0.3) ==="
python3 scripts/core/benchmark_patientsim.py \
  --profiles "$PROFILES" --data-dir "$DATA_DIR" \
  --doctor-mode mobayes --doctor-model "$MODEL" \
  --patient-model "$PATIENT" \
  --force-max-turns --n $N \
  --eig-topk-size 3 --eig-topk-weight 0.5 --eig-focus-conf-threshold 0.3 \
  --output "$OUTPUT/eig_topk3_lam05" &

wait
echo "=== Batch 1 done (3/6) ==="

# ── 4. Focused top-5 EIG (lambda=0.3, threshold=0.3) ──
echo "=== [4/6] Focused top-5 EIG (λ=0.3, θ=0.3) ==="
python3 scripts/core/benchmark_patientsim.py \
  --profiles "$PROFILES" --data-dir "$DATA_DIR" \
  --doctor-mode mobayes --doctor-model "$MODEL" \
  --patient-model "$PATIENT" \
  --force-max-turns --n $N \
  --eig-topk-size 5 --eig-topk-weight 0.3 --eig-focus-conf-threshold 0.3 \
  --output "$OUTPUT/eig_topk5_lam03" &

# ── 5. Focused top-3 EIG (lambda=0.3, threshold=0.5) — later activation ──
echo "=== [5/6] Focused top-3 EIG (λ=0.3, θ=0.5) ==="
python3 scripts/core/benchmark_patientsim.py \
  --profiles "$PROFILES" --data-dir "$DATA_DIR" \
  --doctor-mode mobayes --doctor-model "$MODEL" \
  --patient-model "$PATIENT" \
  --force-max-turns --n $N \
  --eig-topk-size 3 --eig-topk-weight 0.3 --eig-focus-conf-threshold 0.5 \
  --output "$OUTPUT/eig_topk3_lam03_th05" &

# ── 6. Focused top-3 EIG (lambda=1.0, threshold=0.3) — strong focus ──
echo "=== [6/6] Focused top-3 EIG (λ=1.0, θ=0.3) ==="
python3 scripts/core/benchmark_patientsim.py \
  --profiles "$PROFILES" --data-dir "$DATA_DIR" \
  --doctor-mode mobayes --doctor-model "$MODEL" \
  --patient-model "$PATIENT" \
  --force-max-turns --n $N \
  --eig-topk-size 3 --eig-topk-weight 1.0 --eig-focus-conf-threshold 0.3 \
  --output "$OUTPUT/eig_topk3_lam10" &

wait
echo "=== All EIG ablations done ==="
