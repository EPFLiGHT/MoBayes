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
OUTPUT="data/benchmark_results/scaling_v2"
mkdir -p "$OUTPUT"

MANIFEST="data/kbs/scaling/manifest.json"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found. Run --generate first."
    exit 1
fi

N_ENTRIES=$(python3 -c "import json; print(len(json.load(open('$MANIFEST'))))")
echo "Manifest: $N_ENTRIES entries"

for i in $(seq 0 $((N_ENTRIES - 1))); do
    K=$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(m[$i]['k'])")
    SEED=$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(m[$i]['seed'])")
    DATA_DIR=$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(m[$i]['kb_dir'])")
    PROFILES=$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(m[$i]['profiles'])")
    TAG="k${K}_s${SEED}"

    if [ ! -f "$PROFILES" ]; then
        echo "SKIP ${TAG}: profiles not found"
        continue
    fi

    echo "=== K=${K} seed=${SEED} (N=${K}) ==="

    # BMBE
    BMBE_OUT="$OUTPUT/bmbe_${TAG}"
    if [ -d "$BMBE_OUT" ] && ls "$BMBE_OUT"/*.jsonl 1>/dev/null 2>&1; then
        echo "  BMBE ${TAG}: already done, skipping"
    else
        echo "  BMBE ${TAG}..."
        python3 scripts/core/benchmark_patientsim.py \
            --profiles "$PROFILES" --data-dir "$DATA_DIR" \
            --doctor-mode mobayes --doctor-model gpt-5.4-nano \
            --patient-model "$PATIENT_MODEL" \
            --n $K --parallel $PARALLEL \
            --output "$BMBE_OUT"
    fi

    # SA (NO disease list — realistic)
    SA_OUT="$OUTPUT/sa_${TAG}"
    if [ -d "$SA_OUT" ] && ls "$SA_OUT"/*.jsonl 1>/dev/null 2>&1; then
        echo "  SA ${TAG}: already done, skipping"
    else
        echo "  SA ${TAG}..."
        python3 scripts/core/benchmark_patientsim.py \
            --profiles "$PROFILES" --data-dir "$DATA_DIR" \
            --doctor-mode standalone --doctor-model gpt-5.4-nano \
            --patient-model "$PATIENT_MODEL" \
            --n $K --parallel $PARALLEL \
            --output "$SA_OUT"
    fi
done

echo "All done. Plot: PYTHONPATH=. python3 experiments/analysis/scaling_analysis.py --plot"
