"""DDxPlus 50 (PatientSim) column.

Cases    : data/cases/ddxplus_patientsim_a/plain/profiles.json
Patient  : `scripts.core.benchmark_patientsim.PatientSim` — our adversarial
           wrapper around DDxPlus rules (plain persona).
Universe : DDxPlus catalogue (49 leaves + 9 categories).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


NAME = "patientsim"

REPO = Path(__file__).resolve().parents[2]
_PROFILES = REPO / "data/cases/ddxplus_patientsim_a/plain/profiles.json"
_DISEASES_CSV = REPO / "data/kbs/ddxplus/diseases.csv"


def load_cases() -> List[Dict[str, Any]]:
    with _PROFILES.open() as f:
        return json.load(f)


def make_patient(profile: Dict[str, Any], patient_model: str):
    """Wrap a DDxPlus profile in PatientSim with the standard interface."""
    import os
    from scripts.core.benchmark_patientsim import SimplePatientAgent
    from config import MoBayesConfig

    sim = SimplePatientAgent(profile=profile, api_key=os.environ.get("OPENAI_API_KEY"),
                              model=patient_model, config=MoBayesConfig())
    chief = (profile.get("Current Visit", {}).get("chiefcomplaint")
             or "presenting complaint")
    age = profile.get("age", "?")
    sex = profile.get("gender", "?")
    sim.initial_info = (f"Age: {age}\nSex: {sex}\nChief Complaint: {chief}")
    return sim


def disease_universe() -> List[str]:
    return pd.read_csv(_DISEASES_CSV)["name"].tolist()


def output_dir(row_name: str, suffix: str = "") -> Path:
    sfx = f"_{suffix}" if suffix else ""
    return REPO / f"data/benchmark_results/ddxplus_patientsim_50_{row_name}{sfx}"


def matches_gt(prediction: str, ground_truth: str) -> bool:
    norm = lambda s: (s or "").lower().strip()
    return norm(prediction) == norm(ground_truth)


def gt_of(case: Dict[str, Any]) -> str:
    """Profile's ground truth — DDxPlus diagnosis name."""
    meta = case.get("_ddxplus_meta", {}) or {}
    return (meta.get("diagnosis_name")
            or case.get("Current Visit", {}).get("diagnosis", "Unknown"))
