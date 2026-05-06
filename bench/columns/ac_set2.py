"""AgentClinic-MedQA Set 2 column.

Cases    : data/cases/agentclinic_b/selected_cases.jsonl (50, non-overlapping with set1)
Patient  : AC's published PatientAgent via baselines/agentclinic/adapters/base.py
Universe : 50 diseases from data/kbs/agentclinic_b/diseases.csv
KB       : Continuous-scale distributions (anti mode-collapse, gpt-5.4)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


NAME = "ac_set2"

REPO = Path(__file__).resolve().parents[2]
_CASES_PATH = REPO / "data/cases/agentclinic_b/selected_cases.jsonl"
_DISEASES_CSV = REPO / "data/kbs/agentclinic_b/diseases.csv"


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
    return REPO / f"data/benchmark_results/agentclinic_50_set2_{row_name}{sfx}"


def matches_gt(prediction: str, ground_truth: str) -> bool:
    norm = lambda s: (s or "").lower().strip()
    return norm(prediction) == norm(ground_truth)


def gt_of(case: Dict[str, Any]) -> str:
    osce = case.get("OSCE_Examination", case)
    return osce["Correct_Diagnosis"]
