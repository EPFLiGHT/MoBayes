#!/bin/bash
# Disease Scaling Analysis — Random disease subsets.
# K=10,20,30,40 × 3 seeds + K=49 × 1.  N = K patients (1 per disease).
# SA does NOT receive the disease list (realistic setup).
#
# Prerequisite: PYTHONPATH=. python3 experiments/analysis/scaling_analysis.py --generate
# Usage: bash experiments/ddxplus/disease_scaling.sh
set -euo pipefail
[ -f .env ] && source .env || true

PARALLEL=4
PATIENT_MODEL="gpt-5.4-nano"
OUTPUT="data/benchmark_results/set_a/scaling"
mkdir -p "$OUTPUT"

MANIFEST="data/kbs/scaling/manifest.json"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found. Run --generate first."
    exit 1
fi

# run_mode MODE NAME K PROFILES DATA_DIR: one run, skipped if it already has results.
run_mode() {
    local mode="$1" name="$2" k="$3" profiles="$4" data_dir="$5"
    local out="$OUTPUT/$name"
    if ls "$out"/*.jsonl 1>/dev/null 2>&1; then
        echo "  ${name}: already done, skipping"
        return
    fi
    echo "  ${name}..."
    python3 scripts/core/benchmark_patientsim.py \
        --profiles "$profiles" --data-dir "$data_dir" \
        --doctor-mode "$mode" --doctor-model gpt-5.4-nano \
        --patient-model "$PATIENT_MODEL" \
        --n "$k" --parallel $PARALLEL \
        --output "$out" < /dev/null
}

while read -r K SEED DATA_DIR PROFILES; do
    TAG="k${K}_s${SEED}"
    if [ ! -f "$PROFILES" ]; then
        echo "SKIP ${TAG}: profiles not found"
        continue
    fi
    echo "=== K=${K} seed=${SEED} (N=${K}) ==="
    run_mode mobayes    "bmbe_${TAG}" "$K" "$PROFILES" "$DATA_DIR"
    run_mode standalone "sa_${TAG}"   "$K" "$PROFILES" "$DATA_DIR"   # no disease list
done < <(python3 -c "
import json
for m in json.load(open('$MANIFEST')):
    print(m['k'], m['seed'], m['kb_dir'], m['profiles'])")

echo "All done. Plot: PYTHONPATH=. python3 experiments/analysis/scaling_analysis.py --plot"
