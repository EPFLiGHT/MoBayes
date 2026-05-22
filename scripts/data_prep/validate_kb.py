#!/usr/bin/env python3
"""
Knowledge Base Validator.

Validates the integrity and consistency of a CSV-based knowledge base.
Checks all constraints required by data_loader.py.

Usage:
    python scripts/data_prep/validate_kb.py data/kbs/ddxplus/
"""
import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def validate_kb(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Validates all CSV files in a knowledge base directory.
    Returns (errors, warnings).  errors = hard failures, warnings = sparse data notes.
    """
    errors = []
    warnings = []

    def _check_file(name, required=True):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            if required:
                errors.append(f"MISSING: {name}")
            else:
                warnings.append(f"OPTIONAL MISSING: {name} (not required for core inference)")
            return None
        return pd.read_csv(path)

    # 1. diseases.csv  (REQUIRED)
    df_d = _check_file("diseases.csv")
    if df_d is not None:
        if "disease_id" not in df_d.columns:
            errors.append("diseases.csv: missing 'disease_id' column")
        if "name" not in df_d.columns:
            errors.append("diseases.csv: missing 'name' column")

        all_parents = set(df_d["parent_id"].dropna().unique())
        leaf_diseases = df_d[~df_d["disease_id"].isin(all_parents)]["disease_id"].tolist()

        # d_other is optional — excluded by data_loader anyway
        if "d_other" in leaf_diseases:
            warnings.append("diseases.csv: 'd_other' present (will be excluded by data_loader)")

        # Detect phantom categories: nodes that have no parent AND no children
        has_children = set(df_d["parent_id"].dropna().unique())
        for _, row in df_d.iterrows():
            did = row["disease_id"]
            pid = row["parent_id"]
            if pd.isna(pid) and did not in has_children and did != "d_other":
                errors.append(
                    f"diseases.csv: '{did}' has no parent and no children "
                    f"(phantom node - remove or connect to hierarchy)"
                )
    else:
        leaf_diseases = []

    # Filter out d_other for downstream checks (data_loader excludes it)
    leaf_diseases_active = [d for d in leaf_diseases if d != "d_other"]

    # 2. features.csv  (REQUIRED)
    df_f = _check_file("features.csv")
    feature_ids = []
    feature_values = {}
    if df_f is not None:
        for col in ("feature_id", "name", "type", "values"):
            if col not in df_f.columns:
                errors.append(f"features.csv: missing '{col}' column")

        for _, row in df_f.iterrows():
            fid = row["feature_id"]
            feature_ids.append(fid)
            try:
                vals = json.loads(row["values"])
                feature_values[fid] = vals
                if len(vals) < 2:
                    errors.append(f"features.csv: {fid} has <2 values")
            except (json.JSONDecodeError, TypeError):
                errors.append(f"features.csv: {fid} has invalid values JSON")

    # 3. likelihood_counts.csv  (REQUIRED)
    df_lc = _check_file("likelihood_counts.csv")
    if df_lc is not None and leaf_diseases_active and feature_values:
        for col in ("disease_id", "feature_id", "value", "count"):
            if col not in df_lc.columns:
                errors.append(f"likelihood_counts.csv: missing '{col}' column")
                break
        else:
            # Check completeness — missing entries are WARNINGs (Dirichlet handles sparse data)
            missing_count = 0
            for did in leaf_diseases_active:
                for fid, vals in feature_values.items():
                    for val in vals:
                        matches = df_lc[
                            (df_lc["disease_id"] == did) &
                            (df_lc["feature_id"] == fid) &
                            (df_lc["value"] == val)
                        ]
                        if len(matches) == 0:
                            missing_count += 1
            if missing_count > 0:
                total_expected = len(leaf_diseases_active) * sum(len(v) for v in feature_values.values())
                coverage = 1 - (missing_count / total_expected) if total_expected > 0 else 0
                warnings.append(
                    f"likelihood_counts.csv: {missing_count} missing entries "
                    f"({coverage:.1%} coverage) - Dirichlet smoothing will handle these"
                )

            # Check counts are non-negative
            if (df_lc["count"] < 0).any():
                errors.append("likelihood_counts.csv: negative counts found")

            # Check for completely empty disease-feature pairs (all counts = 0)
            empty_pairs = 0
            for did in leaf_diseases_active:
                for fid in feature_values:
                    subset = df_lc[(df_lc["disease_id"] == did) & (df_lc["feature_id"] == fid)]
                    total = subset["count"].sum()
                    if total == 0 and len(subset) > 0:
                        empty_pairs += 1
            if empty_pairs > 0:
                warnings.append(
                    f"likelihood_counts.csv: {empty_pairs} disease-feature pairs with all-zero counts"
                )

    # 4. priors.csv  (REQUIRED)
    df_p = _check_file("priors.csv")
    if df_p is not None and leaf_diseases_active:
        prior_diseases = set(df_p["disease_id"].tolist())
        for did in leaf_diseases_active:
            if did not in prior_diseases:
                errors.append(f"priors.csv: missing prior for {did}")

        total = df_p["prior"].sum()
        if total <= 0:
            errors.append("priors.csv: priors sum to 0 or negative")

        if not np.isclose(total, 1.0, atol=0.05):
            warnings.append(
                f"priors.csv: priors sum to {total:.4f}, expected ~1.0 "
                f"(data_loader normalizes at runtime)"
            )

        if (df_p["prior"] < 0).any():
            errors.append("priors.csv: negative prior values found")

    # 5-8. Optional action/prevention files
    _check_file("actions.csv", required=False)
    df_ar = _check_file("action_rules.csv", required=False)
    if df_ar is not None and feature_ids:
        for _, row in df_ar.iterrows():
            if row["feature_id"] not in feature_ids:
                errors.append(f"action_rules.csv: references unknown feature {row['feature_id']}")

    _check_file("prevention_map.csv", required=False)
    _check_file("disease_action_map.csv", required=False)

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(
        description="Validate a CSV-based MoBayes knowledge base directory.")
    parser.add_argument("data_dir",
                        help="KB directory containing diseases.csv, features.csv, etc.")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"Error: {data_dir} is not a directory")
        sys.exit(1)

    print(f"Validating knowledge base in: {data_dir}")
    errors, warnings = validate_kb(data_dir)

    if warnings:
        print(f"\n{len(warnings)} WARNING(S):")
        for w in warnings:
            print(f"  [WARN] {w}")

    if errors:
        print(f"\n{len(errors)} ERROR(S):")
        for e in errors:
            print(f"  [ERROR] {e}")
        sys.exit(1)
    else:
        print(f"\nAll checks passed. ({len(warnings)} warnings)")
        sys.exit(0)


if __name__ == "__main__":
    main()
