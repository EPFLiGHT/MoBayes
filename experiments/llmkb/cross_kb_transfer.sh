#!/bin/bash
# Cross-KB Transfer — Test each KB on the other's patients (appendix experiment).
# Usage: bash experiments/llmkb/cross_kb_transfer.sh
set -euo pipefail
[ -f .env ] && source .env || true

N=50
PARALLEL=4
PATIENT_MODEL="gpt-5.4-nano"

mkdir -p "data/benchmark_results/llmkb_cross"

# ── BMBE + GPT KB tested on Gemini patients ──
echo "=== Cross: GPT KB → Gemini patients ==="
python3 scripts/core/benchmark_patientsim.py \
  --profiles "data/cases/llmkb_gemini/plain/profiles.json" \
  --data-dir "data/kbs/llmkb_gpt" \
  --doctor-mode mobayes --doctor-model gpt-5.4-nano \
  --patient-model "$PATIENT_MODEL" \
  --force-max-turns --n $N --parallel $PARALLEL \
  --output "data/benchmark_results/llmkb_cross/gpt_kb_gemini_patients"

# ── BMBE + Gemini KB tested on GPT patients ──
echo "=== Cross: Gemini KB → GPT patients ==="
python3 scripts/core/benchmark_patientsim.py \
  --profiles "data/cases/llmkb_gpt/plain/profiles.json" \
  --data-dir "data/kbs/llmkb_gemini" \
  --doctor-mode mobayes --doctor-model "google/gemini-3.1-flash-lite-preview" \
  --doctor-base-url "https://openrouter.ai/api/v1" \
  --doctor-api-key "$OPEN_ROUTER" \
  --patient-model "$PATIENT_MODEL" \
  --force-max-turns --n $N --parallel $PARALLEL \
  --output "data/benchmark_results/llmkb_cross/gemini_kb_gpt_patients"
