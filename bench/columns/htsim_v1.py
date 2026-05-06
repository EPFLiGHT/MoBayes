"""DDxPlus 50 v1 (MEDDxAgent HT-sim) — second-occurrence sample (disjoint from canonical)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


NAME = "htsim_v1"

REPO = Path(__file__).resolve().parents[2]
_CASES_PATH = REPO / "data/cases/ddxplus_htsim_b/selected_cases.jsonl"
_DISEASES_CSV = REPO / "data/kbs/ddxplus/diseases.csv"


def load_cases() -> List[Dict[str, Any]]:
    with _CASES_PATH.open() as f:
        return [json.loads(l) for l in f]


def make_patient(case: Dict[str, Any], patient_model: str):
    from baselines.streambench.adapters.base import LLMPatientWrapper
    return LLMPatientWrapper(case, patient_model=patient_model)


def disease_universe() -> List[str]:
    return pd.read_csv(_DISEASES_CSV)["name"].tolist()


def output_dir(row_name: str, suffix: str = "") -> Path:
    sfx = f"_{suffix}" if suffix else ""
    return REPO / f"data/benchmark_results/set_b/ddxplus_htsim/{row_name}{sfx}"


def matches_gt(prediction: str, ground_truth: str) -> bool:
    norm = lambda s: (s or "").lower().strip()
    return norm(prediction) == norm(ground_truth)


def gt_of(case: Dict[str, Any]) -> str:
    return case.get("PATHOLOGY", "")
