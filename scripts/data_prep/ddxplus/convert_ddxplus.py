#!/usr/bin/env python3
"""
DDXPlus → Knowledge Base Converter.

Downloads the DDXPlus dataset (NeurIPS 2022) and converts it to the CSV/JSONL
format consumed by data_loader.py. Computes empirical P(evidence|disease)
likelihoods from 1.3M patient records.

The final patients.jsonl is output in human-readable PatientSim format
(V_codes resolved, questions mapped to symptom descriptions, structured history).

Usage:
    python scripts/data_prep/ddxplus/convert_ddxplus.py --output-dir data/kbs/ddxplus/
"""
import os
import sys
import json
import ast
import re
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from scripts.data_prep.ddxplus.patientsim_format import (
    convert_patient_data, load_features, load_diseases,
)

# ── ICD-10 Chapter Grouping ──────────────────────────────────────────────────

ICD10_GROUPS = {
    "d_cat_respiratory":  (["J"],           "Respiratory Diseases"),
    "d_cat_infectious":   (["A", "B"],      "Infectious & Parasitic Diseases"),
    "d_cat_circulatory":  (["I"],           "Circulatory Diseases"),
    "d_cat_digestive":    (["K"],           "Digestive Diseases"),
    "d_cat_neuro":        (["G"],           "Neurological Diseases"),
    "d_cat_musculo":      (["M"],           "Musculoskeletal Diseases"),
    "d_cat_skin":         (["L"],           "Skin Diseases"),
    "d_cat_mental":       (["F"],           "Mental & Behavioural"),
    "d_cat_other_sys":    (["R", "N", "D", "E", "H", "O", "S", "T"],
                           "Other System Diseases"),
}

# Maximum values to keep for multi-choice evidences (body location selectors)
MAX_MULTI_CHOICE_VALUES = 20


class DDXPlusConverter:
    """Converts DDXPlus dataset to our KB format."""

    def __init__(self, output_dir: str, cache_dir: str = "data/raw/ddxplus_cache",
                 seed: int = 42, max_patients: int = 500):
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.seed = seed
        self.max_patients = max_patients
        self.rng = np.random.default_rng(seed)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        # Will be populated
        self.conditions = {}      # condition_name → metadata
        self.evidences = {}       # evidence_id → metadata
        self.pathology_to_did = {}  # "Pneumonia" → "d_pneumonia"
        self.did_to_icd = {}      # "d_pneumonia" → "J18.9"
        self.did_to_cat = {}      # "d_pneumonia" → "d_cat_respiratory"
        self.feature_list = []    # [{feature_id, name, type, values}]
        self.feature_id_set = set()
        self.evidence_to_features = {}  # E_xx → feature mapping info
        self.multi_choice_top_values = {}  # E_xx → [top N value codes]

    # ── Phase 1: Load ─────────────────────────────────────────────────────

    def _load_metadata(self):
        """Load conditions.json and evidences.json from cache or Figshare."""
        import requests

        figshare_files = {
            "release_conditions.json": "https://ndownloader.figshare.com/files/40278010",
            "release_evidences.json":  "https://ndownloader.figshare.com/files/40278013",
        }

        for fname, url in figshare_files.items():
            path = os.path.join(self.cache_dir, fname)
            if not os.path.exists(path):
                print(f"  Downloading {fname} from Figshare...")
                r = requests.get(url)
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)

        with open(os.path.join(self.cache_dir, "release_conditions.json")) as f:
            self.conditions = json.load(f)

        with open(os.path.join(self.cache_dir, "release_evidences.json")) as f:
            self.evidences = json.load(f)

        print(f"  Loaded {len(self.conditions)} conditions, "
              f"{len(self.evidences)} evidences")

    def _load_patient_data(self):
        """Load DDXPlus from HuggingFace datasets."""
        from datasets import load_dataset

        cache_path = os.path.join(self.cache_dir, "ddxplus_hf")
        print("  Loading DDXPlus from HuggingFace (may take a few minutes)...")
        self.ds = load_dataset("aai530-group6/ddxplus", cache_dir=cache_path)
        print(f"  Train: {len(self.ds['train'])} | "
              f"Test: {len(self.ds['test'])} | "
              f"Validate: {len(self.ds['validate'])}")

    # ── Phase 2: Build Mappings ───────────────────────────────────────────

    def _sanitize_id(self, name: str) -> str:
        """Convert condition/evidence name to a valid ID."""
        s = name.lower().strip()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        s = s.strip('_')
        return s[:40]

    def _build_disease_mapping(self):
        """Map DDXPlus conditions to disease IDs with ICD-10 taxonomy."""
        categories_used = defaultdict(list)

        for cond_name, cdata in self.conditions.items():
            icd = cdata.get("icd10-id", "R99").upper()
            first_letter = icd[0]

            # Find category
            cat_id = "d_cat_other_sys"
            for cid, (prefixes, _) in ICD10_GROUPS.items():
                if first_letter in prefixes:
                    cat_id = cid
                    break

            did = f"d_{self._sanitize_id(cond_name)}"
            self.pathology_to_did[cond_name] = did
            self.did_to_icd[did] = icd
            self.did_to_cat[did] = cat_id
            categories_used[cat_id].append(did)

        # Drop empty categories
        self.categories = {}
        for cat_id, (_, cat_name) in ICD10_GROUPS.items():
            if cat_id in categories_used:
                self.categories[cat_id] = cat_name

        print(f"  Mapped {len(self.pathology_to_did)} conditions to "
              f"{len(self.categories)} categories")
        for cat_id, cat_name in self.categories.items():
            n = len(categories_used[cat_id])
            print(f"    {cat_id}: {cat_name} ({n} diseases)")

    def _find_top_multi_choice_values(self):
        """Scan training data to find top N values for multi-choice evidences."""
        print("  Scanning for top multi-choice values...")
        mc_evidences = {eid for eid, edata in self.evidences.items()
                        if edata.get("data_type") == "M"}

        value_counts = defaultdict(lambda: defaultdict(int))
        for i, row in enumerate(self.ds["train"]):
            if i >= 200000:  # sample 200K for speed
                break
            evs = ast.literal_eval(row["EVIDENCES"])
            for ev in evs:
                if "_@_" in ev:
                    base, val = ev.split("_@_", 1)
                    if base in mc_evidences:
                        value_counts[base][val] += 1

        for eid in mc_evidences:
            counts = value_counts.get(eid, {})
            top = sorted(counts.items(), key=lambda x: -x[1])[:MAX_MULTI_CHOICE_VALUES]
            self.multi_choice_top_values[eid] = [v for v, c in top]

    def _build_feature_mapping(self):
        """Convert DDXPlus evidences to feature definitions."""
        self._find_top_multi_choice_values()

        for eid, edata in self.evidences.items():
            dtype = edata.get("data_type", "B")
            q_en = edata.get("question_en", eid)
            is_ant = edata.get("is_antecedent", False)
            prefix = "f_ant" if is_ant else "f"

            if dtype == "B":
                fid = f"{prefix}_{self._sanitize_id(eid)}"
                self.feature_list.append({
                    "feature_id": fid,
                    "name": q_en,
                    "type": "binary",
                    "values": json.dumps(["yes", "no"]),
                    "missing_token": "unknown",
                })
                self.feature_id_set.add(fid)
                self.evidence_to_features[eid] = {"type": "binary", "fid": fid}

            elif dtype == "C":
                fid = f"{prefix}_{self._sanitize_id(eid)}"
                pv = edata.get("possible-values", [])
                vm = edata.get("value_meaning", {})
                # Resolve V-codes: vm structure is {code: {en: label, fr: label}}
                yn_map = {"Y": "yes", "N": "no", "NA": "not_applicable"}
                def _resolve_vcode(code):
                    entry = vm.get(code, {})
                    if isinstance(entry, dict):
                        label = entry.get("en", code)
                    else:
                        label = str(entry) if entry else code
                    return yn_map.get(label, label)

                values = [_resolve_vcode(v) for v in pv]
                values.append("absent")
                code_to_readable = {v: _resolve_vcode(v) for v in pv}
                self.feature_list.append({
                    "feature_id": fid,
                    "name": q_en,
                    "type": "categorical",
                    "values": json.dumps(values),
                    "missing_token": "unknown",
                })
                self.feature_id_set.add(fid)
                self.evidence_to_features[eid] = {
                    "type": "categorical", "fid": fid,
                    "code_to_readable": code_to_readable,
                }

            elif dtype == "M":
                top_vals = self.multi_choice_top_values.get(eid, [])
                vm_raw = edata.get("value_meaning", {})
                sub_features = {}
                for val_code in top_vals:
                    vm_entry = vm_raw.get(val_code, {})
                    meaning = vm_entry.get("en", val_code) if isinstance(vm_entry, dict) else val_code
                    sub_fid = f"{prefix}_{self._sanitize_id(eid)}_{self._sanitize_id(val_code)}"
                    self.feature_list.append({
                        "feature_id": sub_fid,
                        "name": f"{q_en} [{meaning}]",
                        "type": "binary",
                        "values": json.dumps(["yes", "no"]),
                        "missing_token": "unknown",
                    })
                    self.feature_id_set.add(sub_fid)
                    sub_features[val_code] = sub_fid
                self.evidence_to_features[eid] = {
                    "type": "multi", "sub_features": sub_features,
                }

        print(f"  Built {len(self.feature_list)} features "
              f"(from {len(self.evidences)} evidences)")

    # ── Phase 3: Parse & Compute ──────────────────────────────────────────

    def _parse_evidences(self, evidence_str: str) -> Dict[str, str]:
        """Convert DDXPlus EVIDENCES string to {feature_id: value}."""
        result = {}
        evs = ast.literal_eval(evidence_str)

        for ev in evs:
            if "_@_" in ev:
                base, val = ev.split("_@_", 1)
                mapping = self.evidence_to_features.get(base)
                if not mapping:
                    continue

                if mapping["type"] == "categorical":
                    fid = mapping["fid"]
                    readable = mapping["code_to_readable"].get(val, val)
                    result[fid] = readable

                elif mapping["type"] == "multi":
                    sub = mapping["sub_features"]
                    if val in sub:
                        result[sub[val]] = "yes"
            else:
                mapping = self.evidence_to_features.get(ev)
                if mapping and mapping["type"] == "binary":
                    result[mapping["fid"]] = "yes"

        return result

    # Negated [nowhere] features: "no" means the opposite (e.g. "has swelling")
    NEGATED_FEATURES = {"f_e_57_v_123", "f_e_133_v_123", "f_e_152_v_123"}

    def _fill_absent(self, parsed: Dict[str, str],
                     skip_negated: bool = False) -> Dict[str, str]:
        """Fill absent features with default values."""
        full = {}
        for f in self.feature_list:
            fid = f["feature_id"]
            if fid in parsed:
                full[fid] = parsed[fid]
            elif skip_negated and fid in self.NEGATED_FEATURES:
                continue  # don't fill negated features
            elif f["type"] == "binary":
                full[fid] = "no"
            else:
                vals = json.loads(f["values"])
                full[fid] = vals[-1]  # "absent" is last
        return full

    def _compute_likelihood_counts(self) -> pd.DataFrame:
        """Compute empirical P(evidence|disease) from training data."""
        print("  Computing likelihoods from training data...")
        raw_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        disease_totals = defaultdict(int)

        total = len(self.ds["train"])
        for i, row in enumerate(self.ds["train"]):
            if i % 200000 == 0:
                print(f"    Processing {i}/{total}...")

            pathology = row["PATHOLOGY"]
            did = self.pathology_to_did.get(pathology)
            if not did:
                continue

            disease_totals[did] += 1
            parsed = self._parse_evidences(row["EVIDENCES"])

            # Fill absent features with defaults, EXCEPT negated [nowhere]
            # features where "no" would mean the opposite (e.g. "has swelling").
            full = self._fill_absent(parsed, skip_negated=True)
            for fid, val in full.items():
                raw_counts[did][fid][val] += 1

        print(f"    Processed {total} patients")

        # Scale to sum=100 with Laplace floor
        rows = []

        for did in self.pathology_to_did.values():
            for f in self.feature_list:
                fid = f["feature_id"]
                vals = json.loads(f["values"])
                n_vals = len(vals)

                total_raw = sum(raw_counts[did][fid].get(v, 0) for v in vals)
                for v in vals:
                    raw = raw_counts[did][fid].get(v, 0)
                    if total_raw > 0:
                        scaled = max(1, round(100 * raw / total_raw))
                    else:
                        scaled = max(1, 100 // n_vals)
                    rows.append({"disease_id": did, "feature_id": fid,
                                 "value": v, "count": scaled})

                # Fix sum to exactly 100
                subset = rows[-n_vals:]
                current = sum(r["count"] for r in subset)
                if current != 100:
                    diff = 100 - current
                    max_row = max(subset, key=lambda r: r["count"])
                    max_row["count"] = max(1, max_row["count"] + diff)

        return pd.DataFrame(rows)

    def _compute_priors(self) -> pd.DataFrame:
        """Compute disease priors from training frequency."""
        freq = defaultdict(int)
        for row in self.ds["train"]:
            did = self.pathology_to_did.get(row["PATHOLOGY"])
            if did:
                freq[did] += 1

        total = sum(freq.values())
        rows = []
        for did, count in freq.items():
            rows.append({"disease_id": did, "prior": round(count / total, 6)})

        return pd.DataFrame(rows)

    # ── Phase 4: Generate Support Files ───────────────────────────────────

    def _generate_diseases_csv(self) -> pd.DataFrame:
        """Generate diseases.csv with taxonomy and ICD-10 codes."""
        # Chapter range strings for category rows
        cat_icd_ranges = {
            "d_cat_respiratory":  "J00-J99",
            "d_cat_infectious":   "A00-B99",
            "d_cat_circulatory":  "I00-I99",
            "d_cat_digestive":    "K00-K95",
            "d_cat_neuro":        "G00-G99",
            "d_cat_musculo":      "M00-M99",
            "d_cat_skin":         "L00-L99",
            "d_cat_mental":       "F00-F99",
            "d_cat_other_sys":    "R00-R99",
        }

        rows = []
        # Categories
        for cat_id, cat_name in self.categories.items():
            rows.append({"disease_id": cat_id, "name": cat_name,
                         "parent_id": "", "icd10": cat_icd_ranges.get(cat_id, "")})

        # Leaf diseases
        for cond_name, did in self.pathology_to_did.items():
            cat_id = self.did_to_cat[did]
            eng_name = self.conditions[cond_name].get("cond-name-eng", cond_name)
            icd10 = self.did_to_icd.get(did, "")
            rows.append({"disease_id": did, "name": eng_name,
                         "parent_id": cat_id, "icd10": icd10})

        return pd.DataFrame(rows)

    def _generate_features_csv(self) -> pd.DataFrame:
        return pd.DataFrame(self.feature_list)

    # ── Phase 5: Convert Patients ─────────────────────────────────────────

    def _convert_patients(self) -> List[Dict[str, Any]]:
        """Sample patients from DDXPlus test set."""
        print(f"  Converting {self.max_patients} test patients...")
        patients = []
        test_data = self.ds["test"]
        n = len(test_data)
        indices = self.rng.choice(n, size=min(self.max_patients, n), replace=False)

        for idx in indices:
            row = test_data[int(idx)]
            pathology = row["PATHOLOGY"]
            did = self.pathology_to_did.get(pathology)
            if not did:
                continue

            parsed = self._parse_evidences(row["EVIDENCES"])
            full = self._fill_absent(parsed)

            patients.append({
                "patient_id": f"ddx_{int(idx):06d}",
                "meta": {
                    "age": row["AGE"],
                    "sex": row["SEX"],
                    "initial_evidence": row.get("INITIAL_EVIDENCE", ""),
                },
                "ground_truth": [did],
                "features": full,
                "features_observed": parsed,
            })

        return patients

    def _convert_to_patientsim(self, raw_patients: List[Dict]) -> List[Dict]:
        """Convert raw patient records to human-readable PatientSim format."""
        features_csv = os.path.join(self.output_dir, "features.csv")
        diseases_csv = os.path.join(self.output_dir, "diseases.csv")

        features = load_features(features_csv)
        diseases = load_diseases(diseases_csv)

        profiles = []
        for patient in raw_patients:
            profile = convert_patient_data(patient, features, diseases)
            profiles.append(profile)

        return profiles

    # ── Phase 6: Stratified Eval Splits ──────────────────────────────────

    def _stratified_sample_from_split(self, split_name: str, n: int,
                                       seed: int) -> List[Dict[str, Any]]:
        """Stratified sample n patients from an HF split.

        Guarantees min 2 patients per disease (if available), then fills
        remaining slots proportionally to disease prevalence.
        Excludes d_other (synthetic). Uses a dedicated seed.
        """
        rng = np.random.default_rng(seed)
        split_data = self.ds[split_name]

        # Group by pathology
        by_disease: Dict[str, List[int]] = defaultdict(list)
        for idx in range(len(split_data)):
            pathology = split_data[idx]["PATHOLOGY"]
            did = self.pathology_to_did.get(pathology)
            if did:
                by_disease[did].append(idx)

        # Shuffle within each disease
        for did in by_disease:
            rng.shuffle(by_disease[did])

        diseases = sorted(by_disease.keys())
        n_diseases = len(diseases)
        min_per = 2

        # Phase 1: pick min_per per disease
        selected_indices = []
        selected_set = set()
        for did in diseases:
            available = by_disease[did]
            pick = min(min_per, len(available))
            for i in range(pick):
                selected_indices.append(available[i])
                selected_set.add(available[i])

        # Phase 2: fill remaining proportionally
        remaining = n - len(selected_indices)
        if remaining > 0:
            weights = np.array([len(by_disease[d]) for d in diseases],
                               dtype=float)
            weights /= weights.sum()
            extra_per = np.round(weights * remaining).astype(int)

            diff = remaining - extra_per.sum()
            if diff != 0:
                order = np.argsort(-weights)
                for i in range(abs(diff)):
                    idx = order[i % n_diseases]
                    extra_per[idx] += 1 if diff > 0 else -1

            for i, did in enumerate(diseases):
                available = by_disease[did]
                count = 0
                for idx in available:
                    if count >= extra_per[i]:
                        break
                    if idx not in selected_set:
                        selected_indices.append(idx)
                        selected_set.add(idx)
                        count += 1

        # Convert to patient records (sparse only — no dense "no" padding)
        patients = []
        for idx in selected_indices:
            row = split_data[int(idx)]
            pathology = row["PATHOLOGY"]
            did = self.pathology_to_did.get(pathology)
            if not did:
                continue

            parsed = self._parse_evidences(row["EVIDENCES"])

            patients.append({
                "patient_id": f"ddx_{int(idx):06d}",
                "meta": {
                    "age": row["AGE"],
                    "sex": row["SEX"],
                    "initial_evidence": row.get("INITIAL_EVIDENCE", ""),
                },
                "ground_truth": [did],
                "features_observed": parsed,
            })

        return patients

    def generate_eval_splits(self, val_n: int = 250, test_n: int = 250,
                              val_seed: int = 100, test_seed: int = 200):
        """Generate stratified validation and test patient sets."""
        print(f"\n  Generating eval splits: val={val_n}, test={test_n}")

        val_patients = self._stratified_sample_from_split(
            "validate", val_n, val_seed)
        test_patients = self._stratified_sample_from_split(
            "test", test_n, test_seed)

        # Verify no overlap (different HF splits, so guaranteed)
        val_ids = {p["patient_id"] for p in val_patients}
        test_ids = {p["patient_id"] for p in test_patients}
        assert val_ids.isdisjoint(test_ids), "Val/test overlap detected!"

        # Disease distribution summary
        for name, pats in [("Validation", val_patients), ("Test", test_patients)]:
            disease_counts = defaultdict(int)
            for p in pats:
                disease_counts[p["ground_truth"][0]] += 1
            print(f"\n  {name}: {len(pats)} patients, "
                  f"{len(disease_counts)} diseases")
            for did, cnt in sorted(disease_counts.items(),
                                    key=lambda x: -x[1])[:5]:
                print(f"    {did}: {cnt}")
            print(f"    ... (min={min(disease_counts.values())}, "
                  f"max={max(disease_counts.values())})")

        # Save
        val_path = os.path.join(self.output_dir, "validation_patients.jsonl")
        test_path = os.path.join(self.output_dir, "test_patients.jsonl")

        for path, pats in [(val_path, val_patients),
                           (test_path, test_patients)]:
            with open(path, "w") as f:
                for p in pats:
                    f.write(json.dumps(p) + "\n")
            print(f"  Saved {path}: {len(pats)} patients")

        return val_patients, test_patients

    # ── Orchestration ─────────────────────────────────────────────────────

    def run(self):
        """Full conversion pipeline."""
        print("=" * 60)
        print("  DDXPlus → Knowledge Base Converter")
        print("=" * 60)

        print("\n[1/6] Loading metadata...")
        self._load_metadata()

        print("\n[2/6] Loading patient data...")
        self._load_patient_data()

        print("\n[3/6] Building mappings...")
        self._build_disease_mapping()
        self._build_feature_mapping()

        print("\n[4/6] Computing likelihoods...")
        lc_df = self._compute_likelihood_counts()

        print("\n[5/6] Computing priors...")
        priors_df = self._compute_priors()

        print("\n[6/6] Generating output files...")
        csvs = {
            "diseases.csv": self._generate_diseases_csv(),
            "features.csv": self._generate_features_csv(),
            "likelihood_counts.csv": lc_df,
            "priors.csv": priors_df,
        }

        for fname, df in csvs.items():
            path = os.path.join(self.output_dir, fname)
            df.to_csv(path, index=False)
            print(f"  Saved {fname}: {len(df)} rows")

        # Convert patients and write as human-readable PatientSim format
        raw_patients = self._convert_patients()
        profiles = self._convert_to_patientsim(raw_patients)

        patients_path = os.path.join(self.output_dir, "patients.jsonl")
        with open(patients_path, "w") as f:
            for p in profiles:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"  Saved patients.jsonl: {len(profiles)} patients (human-readable)")

        print("\n" + "=" * 60)
        n_diseases = len(self.pathology_to_did)
        n_features = len(self.feature_list)
        print(f"  Done: {n_diseases} diseases, {n_features} features")
        print(f"  Output: {self.output_dir}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert DDXPlus dataset to KB format")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for CSV files")
    parser.add_argument("--cache-dir", default="data/raw/ddxplus_cache",
                        help="Cache directory for downloaded files")
    parser.add_argument("--max-patients", type=int, default=500,
                        help="Max patients to convert from test set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--gen-eval-splits", action="store_true",
                        help="Generate stratified validation/test patient splits")
    parser.add_argument("--val-n", type=int, default=250,
                        help="Number of validation patients (default: 250)")
    parser.add_argument("--test-n", type=int, default=250,
                        help="Number of test patients (default: 250)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only generate eval splits (skip full KB conversion)")

    args = parser.parse_args()
    converter = DDXPlusConverter(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        max_patients=args.max_patients,
        seed=args.seed,
    )

    if args.eval_only:
        print("=" * 60)
        print("  DDXPlus — Eval Split Generation (only)")
        print("=" * 60)
        print("\n[1/3] Loading metadata...")
        converter._load_metadata()
        print("\n[2/3] Loading patient data...")
        converter._load_patient_data()
        print("\n[3/3] Building mappings...")
        converter._build_disease_mapping()
        converter._build_feature_mapping()
        converter.generate_eval_splits(val_n=args.val_n, test_n=args.test_n)
    else:
        converter.run()
        if args.gen_eval_splits:
            converter.generate_eval_splits(val_n=args.val_n,
                                            test_n=args.test_n)


if __name__ == "__main__":
    main()
