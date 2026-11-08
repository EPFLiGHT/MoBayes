#!/usr/bin/env python3
"""
Convert LLM-generated KB (feature + distribution JSONs) into the MoBayes
internal KB format.

Input:  generated_features.json + generated_distributions.json
Output: diseases.csv, features.csv, likelihood_counts.csv, priors.csv

Usage:
    python3 scripts/data_prep/llmkb/convert_llmkb.py \
        --features  data/raw/llmkb/output/generated_features_diseases18.json \
        --distributions data/raw/llmkb/output/generated_distributions_diseases18.json \
        --output data/kbs/llmkb_gpt/
"""
import argparse
import csv
import json
import os

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_FEATURES = "data/raw/llmkb/output/generated_features_diseases18.json"
DEFAULT_DISTRIBUTIONS = "data/raw/llmkb/output/generated_distributions_diseases18.json"
DEFAULT_OUTPUT = "data/llmkb"
SCALE = 100  # match DDXPlus convention


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── ICD-10 & category mapping (curated) ─────────────────────────────────────
# Diseases without an entry here get empty icd10/category.

DISEASE_META = {
    # Respiratory
    "d_resp_ar":   {"icd10": "J30.4",  "cat": "d_cat_respiratory", "cat_name": "Respiratory"},
    "d_resp_a1ad": {"icd10": "E88.01", "cat": "d_cat_respiratory", "cat_name": "Respiratory"},
    "d_resp_a":    {"icd10": "J45.9",  "cat": "d_cat_respiratory", "cat_name": "Respiratory"},
    "d_resp_copd": {"icd10": "J44.9",  "cat": "d_cat_respiratory", "cat_name": "Respiratory"},
    "d_resp_ipf":  {"icd10": "J84.1",  "cat": "d_cat_respiratory", "cat_name": "Respiratory"},
    "d_resp_p":    {"icd10": "J93.9",  "cat": "d_cat_respiratory", "cat_name": "Respiratory"},
    # Infectious — viral hepatitis hierarchy
    "d_acqu_hepa":      {"icd10": "B19",   "cat": "d_cat_infectious", "cat_name": "Infectious",
                         "parent": "d_cat_infectious"},
    "d_acqu_hepaB":     {"icd10": "B16.9", "cat": "d_cat_infectious", "cat_name": "Infectious",
                         "parent": "d_acqu_hepa"},
    "d_acqu_hepaC":     {"icd10": "B17.1", "cat": "d_cat_infectious", "cat_name": "Infectious",
                         "parent": "d_acqu_hepa"},
    # Infectious — other
    "d_acqu_immunodef": {"icd10": "B20",   "cat": "d_cat_infectious", "cat_name": "Infectious"},
    "d_acqu_asper":     {"icd10": "B44.9", "cat": "d_cat_infectious", "cat_name": "Infectious"},
    "d_acqu_bruc":      {"icd10": "A23.9", "cat": "d_cat_infectious", "cat_name": "Infectious"},
    "d_acqu_candi":     {"icd10": "B37.9", "cat": "d_cat_infectious", "cat_name": "Infectious"},
    "d_acqu_chlo":      {"icd10": "A00.9", "cat": "d_cat_infectious", "cat_name": "Infectious"},
    "d_acqu_chorio":    {"icd10": "O41.12","cat": "d_cat_infectious", "cat_name": "Infectious"},
    "d_acqu_diph":      {"icd10": "A36.9", "cat": "d_cat_infectious", "cat_name": "Infectious"},
    "d_acqu_gastro":    {"icd10": "A09",   "cat": "d_cat_infectious", "cat_name": "Infectious"},
    "d_acqu_giardi":    {"icd10": "A07.1", "cat": "d_cat_infectious", "cat_name": "Infectious"},
}


def build_diseases_csv(distributions: dict, output_dir: str) -> list:
    """Build diseases.csv with ICD-10 codes and category hierarchy."""
    # Emit category rows first (parents)
    seen_cats = set()
    cat_rows = []
    diseases = []

    for r in distributions["results"]:
        did = r["disease_id"]
        meta = DISEASE_META.get(did, {})
        cat_id = meta.get("cat", "")
        cat_name = meta.get("cat_name", "")
        parent = meta.get("parent", cat_id)  # default parent = category

        # Add category row once
        if cat_id and cat_id not in seen_cats:
            seen_cats.add(cat_id)
            cat_rows.append({
                "disease_id": cat_id,
                "name": cat_name,
                "parent_id": "",
                "icd10": "",
            })

        diseases.append({
            "disease_id": did,
            "name": r["disease_name"],
            "parent_id": parent,
            "icd10": meta.get("icd10", ""),
        })

    all_rows = cat_rows + diseases
    path = os.path.join(output_dir, "diseases.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["disease_id", "name",
                                          "parent_id", "icd10"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"  diseases.csv: {len(all_rows)} rows ({len(cat_rows)} categories + {len(diseases)} leaves)")
    return [d["disease_id"] for d in diseases]


def build_features_csv(features_data: dict, output_dir: str) -> list:
    """Build features.csv from generated features catalog."""
    rows = []
    for f in features_data["features"]:
        ftype = f["type"]  # "binary" or "numerical"
        if ftype == "binary":
            values = json.dumps(["yes", "no"])
        else:
            values = json.dumps(["0", "1", "2", "3", "4", "5"])

        rows.append({
            "feature_id": f["feature_id"],
            "name": f["name"],
            "type": ftype,
            "values": values,
            "missing_token": "unknown",
        })

    path = os.path.join(output_dir, "features.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature_id", "name",
                                          "type", "values", "missing_token"])
        w.writeheader()
        w.writerows(rows)
    print(f"  features.csv: {len(rows)} rows")
    return rows


def build_likelihood_counts(distributions: dict, feature_rows: list,
                            output_dir: str) -> int:
    """Convert P(feature|disease) probabilities → pseudo-counts scaled to SCALE."""
    feature_meta = {r["feature_id"]: r for r in feature_rows}
    all_feature_ids = [r["feature_id"] for r in feature_rows]

    lc_rows = []
    for result in distributions["results"]:
        did = result["disease_id"]
        feats = result.get("features", {})

        for fid in all_feature_ids:
            meta = feature_meta[fid]
            values = json.loads(meta["values"])

            if fid in feats:
                info = feats[fid]
                if info["type"] == "binary":
                    p_yes = info["prob_yes"]
                    counts = _prob_to_counts({"yes": p_yes, "no": 1.0 - p_yes})
                else:
                    # discrete_0_5 distribution
                    dist = info.get("distribution", {})
                    counts = _prob_to_counts({str(k): v for k, v in dist.items()})
            else:
                # Feature not relevant to this disease → uniform
                counts = {v: max(1, SCALE // len(values)) for v in values}

            for v in values:
                lc_rows.append({
                    "disease_id": did,
                    "feature_id": fid,
                    "value": v,
                    "count": counts.get(v, 1),
                })

    path = os.path.join(output_dir, "likelihood_counts.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["disease_id", "feature_id",
                                          "value", "count"])
        w.writeheader()
        w.writerows(lc_rows)
    print(f"  likelihood_counts.csv: {len(lc_rows)} rows")
    return len(lc_rows)


def _prob_to_counts(prob_dict: dict) -> dict:
    """Convert probability dict → integer pseudo-counts summing to SCALE."""
    total_p = sum(prob_dict.values())
    if total_p <= 0:
        return {k: max(1, SCALE // len(prob_dict)) for k in prob_dict}

    counts = {}
    for k, p in prob_dict.items():
        counts[k] = max(1, round(SCALE * p / total_p))
    return counts


def build_priors(disease_ids: list, output_dir: str):
    """Uniform prior — no population data for LLM-generated KB."""
    n = len(disease_ids)
    path = os.path.join(output_dir, "priors.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["disease_id", "prior"])
        w.writeheader()
        for did in disease_ids:
            w.writerow({"disease_id": did, "prior": round(1.0 / n, 6)})
    print(f"  priors.csv: {n} rows (uniform)")


def build_patients(distributions: dict, features_data: dict, output_dir: str,
                   n_per_disease: int = 30, seed: int = 42):
    """Sample synthetic patients from P(feature|disease) distributions."""
    import numpy as np
    rng = np.random.default_rng(seed)

    feature_meta = {f["feature_id"]: f for f in features_data["features"]}
    # disease_feature_map tells which features are relevant per disease
    dfm = features_data.get("disease_feature_map", {})

    patients = []
    pid_counter = 0

    for result in distributions["results"]:
        did = result["disease_id"]
        dname = result["disease_name"]
        feats = result.get("features", {})
        relevant_fids = dfm.get(did, list(feats.keys()))

        for _ in range(n_per_disease):
            pid_counter += 1
            hadm_id = f"llm_{pid_counter:06d}"
            feature_map = {}
            positive_symptoms = []

            for fid in relevant_fids:
                if fid not in feats:
                    continue
                info = feats[fid]
                fname = feature_meta.get(fid, {}).get("name", fid)

                if info["type"] == "binary":
                    val = "yes" if rng.random() < info["prob_yes"] else "no"
                    feature_map[fid] = f"{fname}: {val.upper()}"
                    if val == "yes":
                        positive_symptoms.append(fname)
                else:
                    dist = info.get("distribution", {})
                    vals = list(dist.keys())
                    probs = np.array([dist[v] for v in vals])
                    probs = probs / probs.sum()
                    chosen = rng.choice(vals, p=probs)
                    feature_map[fid] = f"{fname}: {chosen}/5"
                    if chosen != "0":
                        positive_symptoms.append(f"{fname} ({chosen}/5)")

            patient = {
                "hadm_id": hadm_id,
                "age": int(rng.integers(18, 80)),
                "gender": rng.choice(["Male", "Female"]),
                "race": "Unknown",
                "Social History": {},
                "Medical History": {},
                "Current Visit": {
                    "chiefcomplaint": ", ".join(positive_symptoms[:3]) or dname,
                    "diagnosis": dname,
                },
                "Present Illness": {
                    "present_illness_positive": ", ".join(positive_symptoms),
                },
                "_llmkb_meta": {
                    "patient_id": hadm_id,
                    "ground_truth_ids": [did],
                    "diagnosis_name": dname,
                    "num_positive_symptoms": len(positive_symptoms),
                    "feature_map": feature_map,
                },
            }
            patients.append(patient)

    # Shuffle
    rng.shuffle(patients)

    path = os.path.join(output_dir, "patients.jsonl")
    with open(path, "w") as f:
        for p in patients:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  patients.jsonl: {len(patients)} patients ({n_per_disease} per disease)")
    return len(patients)


def main():
    parser = argparse.ArgumentParser(
        description="Convert LLM-generated KB to MoBayes format")
    parser.add_argument("--features", default=DEFAULT_FEATURES,
                        help="Path to generated_features.json")
    parser.add_argument("--distributions", default=DEFAULT_DISTRIBUTIONS,
                        help="Path to generated_distributions.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output directory for KB CSVs")
    parser.add_argument("--patients-per-disease", type=int, default=30,
                        help="Synthetic patients per disease (default: 30)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  LLM KB → MoBayes Format Converter")
    print("=" * 60)

    features_data = load_json(args.features)
    distributions = load_json(args.distributions)

    meta = distributions.get("_meta", {})
    print(f"  Source model: {meta.get('model', '?')}")
    print(f"  Diseases: {meta.get('num_diseases', '?')}")
    print(f"  Features: {meta.get('num_features', '?')}")
    print()

    os.makedirs(args.output, exist_ok=True)

    disease_ids = build_diseases_csv(distributions, args.output)
    feature_rows = build_features_csv(features_data, args.output)
    build_likelihood_counts(distributions, feature_rows, args.output)
    build_priors(disease_ids, args.output)
    build_patients(distributions, features_data, args.output,
                   n_per_disease=args.patients_per_disease, seed=args.seed)

    print()
    print(f"  Done → {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
