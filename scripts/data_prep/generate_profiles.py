#!/usr/bin/env python3
"""
Benchmark profile generator for DDXPlus (and LLM-elicited variants).

Generates persona-specific profiles.json files from patients.jsonl for use
with the patient simulator benchmark.

Usage:
    # Stratified 50 patients
    python scripts/data_prep/generate_profiles.py \
      --patients data/kbs/ddxplus/patients.jsonl \
      --n 50 --stratify --seed 42 \
      --output data/cases/experiment_50

    # All patients
    python scripts/data_prep/generate_profiles.py \
      --patients data/kbs/ddxplus/patients.jsonl \
      --output data/cases/experiment_full

    # Specific personas only
    python scripts/data_prep/generate_profiles.py \
      --patients data/kbs/ddxplus/patients.jsonl \
      --personas plain,dazed --n 50
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from scripts.data_prep.personas import PERSONAS


# ── Sampling ─────────────────────────────────────────────────────────────────

def _get_ground_truth_id(patient: Dict) -> str:
    """Extract primary ground truth disease ID from any patient format."""
    for meta_key in ("_ddxplus_meta", "_llmkb_meta"):
        if meta_key in patient:
            gt_ids = patient[meta_key].get("ground_truth_ids", [])
            return gt_ids[0] if gt_ids else None
    gt_list = patient.get("ground_truth", [])
    return gt_list[0] if gt_list else None


def exclude_other_patients(patients: List[Dict]) -> List[Dict]:
    """Drop synthetic OOD patients whose ground truth is d_other."""
    return [p for p in patients if _get_ground_truth_id(p) != "d_other"]


def stratified_sample(patients: List[Dict], n: int = 50, seed: int = 42) -> List[Dict]:
    """
    Select n patients with stratified sampling.
    Guarantees at least 1 patient per disease, fills remaining
    slots from the most common diseases (round-robin).
    """
    rng = random.Random(seed)

    # Group by disease
    by_disease = defaultdict(list)
    for p in patients:
        gt = _get_ground_truth_id(p)
        if gt:
            by_disease[gt].append(p)

    # Shuffle within each group
    for disease in by_disease:
        rng.shuffle(by_disease[disease])

    # Phase 1: pick 1 per disease
    selected = []
    selected_ids = set()
    for disease in sorted(by_disease.keys()):
        p = by_disease[disease][0]
        pid = _get_patient_id(p)
        selected.append(p)
        selected_ids.add(pid)

    # Phase 2: fill remaining slots round-robin from diseases with most patients
    remaining = n - len(selected)
    if remaining > 0:
        diseases_by_count = sorted(by_disease.keys(),
                                    key=lambda d: len(by_disease[d]),
                                    reverse=True)
        prev_remaining = None
        while remaining > 0 and remaining != prev_remaining:
            prev_remaining = remaining
            for disease in diseases_by_count:
                if remaining <= 0:
                    break
                for p in by_disease[disease]:
                    pid = _get_patient_id(p)
                    if pid not in selected_ids:
                        selected.append(p)
                        selected_ids.add(pid)
                        remaining -= 1
                        break

    return selected


def _get_patient_id(patient: Dict) -> str:
    """Extract patient ID from DDXPlus or LLM KB format."""
    for meta_key in ("_ddxplus_meta", "_llmkb_meta"):
        if meta_key in patient:
            return patient.get("hadm_id", patient[meta_key].get("patient_id", "unknown"))
    return patient.get("patient_id", "unknown")


# ── Profile wrapping ─────────────────────────────────────────────────────────

def wrap_ddxplus_profile(patient_data: Dict, persona: Dict) -> Dict:
    """Add persona settings to a DDXPlus human-readable patient profile."""
    from scripts.data_prep.ddxplus.patientsim_format import apply_persona
    return apply_persona(patient_data, persona)


# ── Loading ──────────────────────────────────────────────────────────────────

def load_patients_jsonl(path: str) -> List[Dict]:
    """Load patients from a JSONL file."""
    patients = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                patients.append(json.loads(line))
    return patients


def load_patient_ids(path: str) -> List[str]:
    """Load selected patient IDs from a JSON file."""
    with open(path) as f:
        return json.load(f)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate DDXPlus benchmark profiles with personas')
    parser.add_argument('--patients', required=True,
                        help='Path to patients.jsonl')
    parser.add_argument('--n', type=int, default=None,
                        help='Number of patients to select (default: all)')
    parser.add_argument('--stratify', action='store_true',
                        help='Use stratified sampling (one patient per disease then fill)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: auto-generated)')
    parser.add_argument('--personas', default=None,
                        help='Comma-separated persona names (default: all)')
    parser.add_argument('--patient-ids', default=None,
                        help='Path to JSON file with selected patient IDs')
    args = parser.parse_args()

    # Determine which personas to generate
    if args.personas:
        persona_names = [p.strip() for p in args.personas.split(',')]
        selected_personas = {k: v for k, v in PERSONAS.items() if k in persona_names}
        if not selected_personas:
            print(f"ERROR: No valid personas in '{args.personas}'")
            print(f"  Available: {', '.join(PERSONAS.keys())}")
            sys.exit(1)
    else:
        selected_personas = PERSONAS

    # Load patients
    print(f"Loading patients from {args.patients}...")
    patients = load_patients_jsonl(args.patients)

    # Filter by patient IDs if specified
    if args.patient_ids:
        selected_ids = set(load_patient_ids(args.patient_ids))
        patients = [p for p in patients if _get_patient_id(p) in selected_ids]
        print(f"  Filtered to {len(patients)} patients from ID list")

    # Exclude d_other
    patients = exclude_other_patients(patients)
    print(f"  {len(patients)} patients after excluding d_other")

    # Sample
    if args.stratify:
        n = args.n or len(patients)
        patients = stratified_sample(patients, n=n, seed=args.seed)
        print(f"  Stratified sample: {len(patients)} patients")
    elif args.n:
        patients = patients[:args.n]
        print(f"  Taking first {len(patients)} patients")

    n_patients = len(patients)

    # Output directory
    output_base = args.output or os.path.join(
        BASE_DIR, 'data', 'cases',
        f'ddxplus_experiment_{n_patients}')

    os.makedirs(output_base, exist_ok=True)

    # Save selected patient IDs for reproducibility
    ids_path = os.path.join(output_base, 'selected_patient_ids.json')
    with open(ids_path, 'w') as f:
        json.dump([_get_patient_id(p) for p in patients], f, indent=2)
    print(f"  Patient IDs saved to: {ids_path}")

    # Disease distribution
    disease_counts = defaultdict(int)
    for p in patients:
        gt = _get_ground_truth_id(p)
        if gt:
            disease_counts[gt] += 1
    print(f"  Diseases represented: {len(disease_counts)}")

    # Generate profiles for each persona
    for persona_name, persona_config in selected_personas.items():
        persona_dir = os.path.join(output_base, persona_name)
        os.makedirs(persona_dir, exist_ok=True)

        profiles = [wrap_ddxplus_profile(p, persona_config) for p in patients]

        output_path = os.path.join(persona_dir, 'profiles.json')
        with open(output_path, 'w') as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)

        print(f"  {persona_name}: {len(profiles)} profiles -> {output_path}")

    # Summary
    print(f"\n{'='*55}")
    print("  Profile Generation Complete")
    print(f"{'='*55}")
    print(f"  Patients   : {n_patients}")
    print(f"  Diseases   : {len(disease_counts)}")
    print(f"  Personas   : {', '.join(selected_personas.keys())}")
    print(f"  Total files: {len(selected_personas)} x {n_patients} = "
          f"{len(selected_personas) * n_patients} profiles")
    print(f"  Output dir : {output_base}")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
