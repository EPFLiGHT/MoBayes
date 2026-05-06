"""AgentClinic-MedQA column.

Cases    : data/cases/agentclinic_a/selected_cases.jsonl (50, frequency-weighted from 213)
Patient  : AC's published PatientAgent via baselines/agentclinic/adapters/base.py
Universe : 31 diseases from data/kbs/agentclinic_a/diseases.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


NAME = "ac"

REPO = Path(__file__).resolve().parents[2]
_CASES_PATH = REPO / "data/cases/agentclinic_a/selected_cases.jsonl"
_DISEASES_CSV = REPO / "data/kbs/agentclinic_a/diseases.csv"


def load_cases() -> List[Dict[str, Any]]:
    with _CASES_PATH.open() as f:
        return [json.loads(l) for l in f]


def make_patient(case: Dict[str, Any], patient_model: str):
    """Wraps the AC OSCE case in AC's PatientAgent (modern openai 1.x)."""
    from baselines.agentclinic.adapters.base import AgentClinicCaseWrapper
    return AgentClinicCaseWrapper(case, patient_model=patient_model)


def disease_universe() -> List[str]:
    return pd.read_csv(_DISEASES_CSV)["name"].tolist()


def output_dir(row_name: str, suffix: str = "") -> Path:
    sfx = f"_{suffix}" if suffix else ""
    return REPO / f"data/benchmark_results/agentclinic_50_{row_name}{sfx}"


def matches_gt(prediction: str, ground_truth: str) -> bool:
    """Case- and whitespace-insensitive exact match (matches AC moderator)."""
    norm = lambda s: (s or "").lower().strip()
    return norm(prediction) == norm(ground_truth)


def gt_of(case: Dict[str, Any]) -> str:
    osce = case.get("OSCE_Examination", case)
    return osce["Correct_Diagnosis"]
