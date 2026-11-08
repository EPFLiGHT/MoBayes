from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from world_model.core.engine import WorldModel

from world_model.core.likelihood_model import DirichletLikelihoodModel, FeatureSchema
from world_model.knowledge.taxonomy import DiseaseTaxonomy

EXCLUDED_DISEASE_IDS = {"d_other"}


def load_disease_names(data_dir: str) -> dict:
    """Load disease_id → name mapping from diseases.csv."""
    import csv
    names = {}
    with open(os.path.join(data_dir, "diseases.csv")) as f:
        for row in csv.DictReader(f):
            names[row["disease_id"]] = row["name"]
    return names


def load_icd_map(data_dir: str) -> Tuple[List[str], dict]:
    """Load leaf disease names list and ICD-10 → name mapping from diseases.csv.

    Returns (disease_name_list, icd_map) excluding categories and d_other.
    """
    import csv
    names, icd_map = [], {}
    with open(os.path.join(data_dir, "diseases.csv")) as f:
        for row in csv.DictReader(f):
            if row["disease_id"].startswith("d_cat_") or row["disease_id"] in EXCLUDED_DISEASE_IDS:
                continue
            names.append(row["name"])
            icd = row.get("icd10", "").strip()
            if icd:
                icd_map[icd] = row["name"]
    return names, icd_map


def parse_values(val_str: str) -> list:
    try:
        if val_str and isinstance(val_str, str):
            return json.loads(val_str)
        return []
    except json.JSONDecodeError:
        return [x.strip() for x in val_str.strip('[]"\'').split(',')]


def _age_to_group(age: int) -> str:
    """Map numeric age to age group (matches compute_conditional_priors.py)."""
    if age <= 30:
        return "young"
    elif age <= 60:
        return "middle"
    else:
        return "old"


def _load_conditional_prior(
    data_dir: str,
    disease_to_idx: dict,
    num_diseases: int,
    patient_meta: dict = None,
) -> np.ndarray:
    """Load conditional_priors.csv and return the prior vector for a patient's demographic group."""
    import csv

    cond_path = os.path.join(data_dir, "conditional_priors.csv")
    if not os.path.exists(cond_path):
        raise FileNotFoundError(
            f"conditional_priors.csv not found at {cond_path}. "
            "Run scripts/data_prep/ddxplus/compute_conditional_priors.py first."
        )

    meta = patient_meta or {}
    age = meta.get("age", 30)
    sex_raw = meta.get("sex") or meta.get("gender") or "male"
    age_group = _age_to_group(int(age))
    sex = sex_raw.strip().lower()
    if sex in ("f", "female"):
        sex = "female"
    else:
        sex = "male"

    prior = np.ones(num_diseases) / num_diseases  # fallback
    found = False
    with open(cond_path) as f:
        for row in csv.DictReader(f):
            if row["age_group"] == age_group and row["sex"] == sex:
                did = row["disease_id"]
                if did in disease_to_idx:
                    prior[disease_to_idx[did]] = float(row["prior"])
                    found = True

    if found and np.sum(prior) > 0:
        prior /= np.sum(prior)
    return prior


def load_world_model_from_schema(
    data_dir: str,
    prior_mode: str = "empirical",
    patient_meta: dict = None,
) -> Tuple[WorldModel, List[str]]:
    """
    Loads the WorldModel entirely from the CSV data schema.
    Returns the instantiated WorldModel and the ordered list of leaf diseases.

    Args:
        data_dir: Path to the knowledge-base directory (diseases.csv, features.csv, …).
        prior_mode: How to set the disease prior —
            'empirical' (default): use priors.csv as-is.
            'uniform': set all priors to 1/N.
            'conditional': load conditional_priors.csv and select the
                           prior vector matching the patient's age_group × sex.
        patient_meta: Dict with 'age' (int) and 'sex'/'gender' (str) keys.
                      Required when prior_mode='conditional'.
    """
    # 1. Load Diseases & Hierarchy
    df_diseases = pd.read_csv(os.path.join(data_dir, "diseases.csv"))

    # Identify leaves (diseases that are NOT parents)
    all_parents = set(df_diseases["parent_id"].dropna().unique())
    df_leaves = df_diseases[~df_diseases["disease_id"].isin(all_parents)].copy()
    df_leaves = df_leaves[~df_leaves["disease_id"].isin(EXCLUDED_DISEASE_IDS)].copy()

    leaf_diseases = df_leaves["disease_id"].tolist()
    num_diseases = len(leaf_diseases)
    disease_to_idx = {d: i for i, d in enumerate(leaf_diseases)}

    # Build taxonomy mapping
    parent_mapping = {}
    for _, row in df_diseases.iterrows():
        pid = row["parent_id"]
        cid = row["disease_id"]
        if pd.notna(pid):
            if pid not in parent_mapping:
                parent_mapping[pid] = []
            parent_mapping[pid].append(cid)

    taxonomy = DiseaseTaxonomy(leaf_diseases=leaf_diseases, parent_mapping=parent_mapping)

    # 2. Load Features (Schema)
    df_features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    schema = {}
    for _, row in df_features.iterrows():
        fid = row["feature_id"]
        ftype = row["type"]
        values = parse_values(row.get("values", "[]"))
        schema[fid] = FeatureSchema(
            feature_id=fid,
            ftype=ftype,
            values=values,
            metadata={"name": row["name"], "missing_token": row.get("missing_token", "unknown")}
        )

    # 3. Load Priors
    if prior_mode == "uniform":
        base_prior = np.ones(num_diseases) / num_diseases
    elif prior_mode == "conditional":
        base_prior = _load_conditional_prior(data_dir, disease_to_idx, num_diseases, patient_meta)
    else:
        # empirical (default)
        df_priors = pd.read_csv(os.path.join(data_dir, "priors.csv"))
        base_prior = np.zeros(num_diseases)
        for _, row in df_priors.iterrows():
            did = row["disease_id"]
            if did in disease_to_idx:
                base_prior[disease_to_idx[did]] = row["prior"]

        if np.sum(base_prior) > 0:
            base_prior /= np.sum(base_prior)
        else:
            base_prior = np.ones(num_diseases) / num_diseases

    # 4. Load Likelihoods (Dirichlet Counts)
    df_counts = pd.read_csv(os.path.join(data_dir, "likelihood_counts.csv"))

    alpha_table = {}
    for fid, fschema in schema.items():
        if fschema.ftype in ['categorical', 'binary']:
            alpha_table[fid] = {}
            for val in fschema.values:
                alpha_table[fid][val] = np.ones(num_diseases).tolist()

    for _, row in df_counts.iterrows():
        did = row["disease_id"]
        fid = row["feature_id"]
        val = row["value"]
        count = row["count"]

        if did in disease_to_idx and fid in alpha_table and val in alpha_table[fid]:
            idx = disease_to_idx[did]
            alpha_table[fid][val][idx] = count + 1.0

    likelihood_model = DirichletLikelihoodModel(
        alpha_table=alpha_table,
        num_diseases=num_diseases,
        schema=schema,
    )

    # 5. Initialize Engine (lazy import to avoid circular dependency)
    from world_model.core.engine import WorldModel
    wm = WorldModel(
        num_diseases=num_diseases,
        likelihood_model=likelihood_model,
        taxonomy=taxonomy,
        base_prior=base_prior,
        leaf_diseases=leaf_diseases,
    )

    return wm, leaf_diseases
