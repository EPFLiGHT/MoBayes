#!/usr/bin/env python3
"""
Compute conditional priors P(disease | age_group, sex) from DDXPlus train patients.

Age groups: young (0-30), middle (31-60), old (61+)
Sex groups: male, female
Total: 6 demographic groups.

Reads:  data/kbs/ddxplus/patients.jsonl
Writes: data/kbs/ddxplus/conditional_priors.csv
        Columns: age_group, sex, disease_id, prior

Usage:
    python3 scripts/data_prep/ddxplus/compute_conditional_priors.py
    python3 scripts/data_prep/ddxplus/compute_conditional_priors.py --data-dir data/kbs/ddxplus
"""

import os
import sys
import json
import argparse
import csv
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def age_to_group(age: int) -> str:
    """Map numeric age to age group."""
    if age <= 30:
        return "young"
    elif age <= 60:
        return "middle"
    else:
        return "old"


def sex_normalize(gender: str) -> str:
    """Normalize gender string to male/female."""
    g = gender.strip().lower()
    if g in ("male", "m"):
        return "male"
    elif g in ("female", "f"):
        return "female"
    return g  # fallback


def main():
    parser = argparse.ArgumentParser(description="Compute conditional priors from DDXPlus train data")
    parser.add_argument("--data-dir", default=None, help="Path to DDXPlus data dir")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(BASE_DIR, "data", "kbs", "ddxplus")
    patients_path = os.path.join(data_dir, "patients.jsonl")
    output_path = os.path.join(data_dir, "conditional_priors.csv")

    if not os.path.exists(patients_path):
        print(f"ERROR: {patients_path} not found", file=sys.stderr)
        sys.exit(1)

    # Count occurrences: (age_group, sex) -> disease_id -> count
    group_disease_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    group_totals: Counter = Counter()

    # Also collect all disease IDs seen
    all_diseases: set[str] = set()

    n_patients = 0
    n_skipped = 0

    with open(patients_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            patient = json.loads(line)
            age = patient.get("age")
            gender = patient.get("gender", "")
            meta = patient.get("_ddxplus_meta", {})
            gt_ids = meta.get("ground_truth_ids", [])

            if age is None or not gender or not gt_ids:
                n_skipped += 1
                continue

            age_group = age_to_group(int(age))
            sex = sex_normalize(gender)
            group_key = (age_group, sex)

            # Each patient contributes 1 count to their group
            group_totals[group_key] += 1
            for did in gt_ids:
                group_disease_counts[group_key][did] += 1
                all_diseases.add(did)

            n_patients += 1

    print(f"Processed {n_patients} patients ({n_skipped} skipped)")
    print(f"Diseases: {len(all_diseases)}, Groups: {len(group_totals)}")
    for gk, cnt in sorted(group_totals.items()):
        print(f"  {gk}: {cnt} patients")

    # Write conditional priors
    # P(disease | group) = count(disease, group) / count(group)
    # Apply Laplace smoothing: (count + 1) / (total + N_diseases)
    n_diseases = len(all_diseases)
    sorted_diseases = sorted(all_diseases)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["age_group", "sex", "disease_id", "prior"])

        for group_key in sorted(group_totals.keys()):
            age_group, sex = group_key
            total = group_totals[group_key]
            counts = group_disease_counts[group_key]

            for did in sorted_diseases:
                # Laplace-smoothed probability
                prior = (counts.get(did, 0) + 1) / (total + n_diseases)
                writer.writerow([age_group, sex, did, f"{prior:.8f}"])

    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
