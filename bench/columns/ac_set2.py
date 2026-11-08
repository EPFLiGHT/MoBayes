"""AgentClinic-MedQA Set 2 column.

Cases    : data/cases/agentclinic_b/selected_cases.jsonl (50, non-overlapping with set1)
Patient  : AC's published PatientAgent via bench/patients/agentclinic.py
Universe : 50 diseases from data/kbs/agentclinic_b/diseases.csv
KB       : Continuous-scale distributions (anti mode-collapse, gpt-5.4)

Same wrappers as `ac.py`; case path, disease list and output dir differ.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from bench.columns.ac import gt_of, make_patient, matches_gt  # noqa: F401  (column interface)


NAME = "ac_set2"

REPO = Path(__file__).resolve().parents[2]
_CASES_PATH = REPO / "data/cases/agentclinic_b/selected_cases.jsonl"
_DISEASES_CSV = REPO / "data/kbs/agentclinic_b/diseases.csv"


def load_cases() -> List[Dict[str, Any]]:
    with _CASES_PATH.open() as f:
        return [json.loads(l) for l in f]


def disease_universe() -> List[str]:
    return pd.read_csv(_DISEASES_CSV)["name"].tolist()


def output_dir(row_name: str, suffix: str = "") -> Path:
    sfx = f"_{suffix}" if suffix else ""
    return REPO / f"data/benchmark_results/agentclinic_50_set2_{row_name}{sfx}"
