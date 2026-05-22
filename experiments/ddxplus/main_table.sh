#!/bin/bash
# Main table run: 12 systems (6 SA + 6 MoBayes) on HT-sim and AgentClinic.
#
# PatientSim results already exist in set_a/ddxplus_patientsim/.
# MoBayes + gpt-5.4-nano already exists for HT-sim and AC.
# This script fills the remaining 21 cells.
#
# Usage:
#   SMOKE=1 bash experiments/ddxplus/main_table.sh          # n=3 smoke
#   bash experiments/ddxplus/main_table.sh                  # n=50 full
#   COL=htsim bash experiments/ddxplus/main_table.sh        # single column
#   ROW=sa MODEL=gpt-5.4 bash experiments/ddxplus/main_table.sh  # single cell
set -euo pipefail
[ -f .env ] && source .env || true

OR_URL="https://openrouter.ai/api/v1"
PATIENT_MODEL="gpt-5.4-nano"

if [ "${SMOKE:-}" = "1" ]; then
  N=3; echo ">>> SMOKE TEST (n=3) <<<"
else
  N=50; echo ">>> FULL RUN (n=50) <<<"
fi

# ── Model registry ──────────────────────────────────────────────
# Format: "row_name|short_tag|model_slug|routing"
#   routing: "openai" (uses OPENAI_API_KEY) or "or" (uses OPEN_ROUTER via OpenRouter)
MODELS=(
  # Standalone doctors
  "sa|gpt54|gpt-5.4|openai"
  "sa|gemini31pro|google/gemini-3.1-pro-preview|or"
  "sa|llama4maverick|meta-llama/llama-4-maverick|or"
  "sa|qwen36plus|qwen/qwen3.6-plus|or"
  "sa|kimi_k25|moonshotai/kimi-k2.5|or"
  "sa|gptoss120b|openai/gpt-oss-120b|or"
  # BMBE sensors (skip gpt-5.4-nano — already has HT-sim + AC data)
  "bmbe|geminiflashlite|google/gemini-3.1-flash-lite-preview|or"
  "bmbe|gptoss20b|openai/gpt-oss-20b:nitro|or"
  "bmbe|minimax_m25|minimax/minimax-m2.5|or"
  "bmbe|gemma4_31b|google/gemma-4-31b-it|or"
  "bmbe|llama4scout|meta-llama/llama-4-scout|or"
)

COLUMNS=("htsim" "ac")

# Allow single-cell runs via env overrides.
if [ -n "${COL:-}" ]; then COLUMNS=("$COL"); fi

run_cell() {
  local row="$1" tag="$2" model="$3" routing="$4" col="$5"

  # Optional single-cell filter.
  if [ -n "${ROW:-}" ] && [ "$row" != "$ROW" ]; then return 0; fi
  if [ -n "${MODEL:-}" ] && [ "$model" != "$MODEL" ]; then return 0; fi

  local suffix="${tag}"
  local base_url_args=()
  if [ "$routing" = "or" ]; then
    base_url_args=(--doctor-base-url "$OR_URL" --doctor-api-key "$OPEN_ROUTER")
  fi

  # BMBE AC column uses v5 (distilled) KB.
  local row_kwargs="{}"
  if [ "$row" = "bmbe" ] && [ "$col" = "ac" ]; then
    row_kwargs='{"kb_version":"v5"}'
  fi

  echo ""
  echo "═══ ${row} + ${tag} × ${col} (n=${N}) ═══"
  python3 -m bench \
    --row "$row" --column "$col" \
    --doctor-model "$model" \
    --patient-model "$PATIENT_MODEL" \
    --num-cases "$N" --max-turns 15 \
    --out-suffix "$suffix" \
    --row-kwargs "$row_kwargs" \
    ${base_url_args[@]+"${base_url_args[@]}"}
}

for col in "${COLUMNS[@]}"; do
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r row tag model routing <<< "$entry"
    run_cell "$row" "$tag" "$model" "$routing" "$col"
  done
done

echo ""
echo "═══ All Table 2 expansion runs complete ═══"
