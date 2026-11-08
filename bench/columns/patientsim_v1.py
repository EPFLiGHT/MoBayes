"""DDxPlus 50 v1 (PatientSim) — set_b sample (disjoint from canonical).

Same wrappers as `patientsim.py`, only profile path + output dir differ.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from bench.columns.patientsim import (  # noqa: F401  (column interface)
    disease_universe, gt_of, make_patient, matches_gt,
)


NAME = "patientsim_v1"

REPO = Path(__file__).resolve().parents[2]
_PROFILES = REPO / "data/cases/ddxplus_patientsim_b/plain/profiles.json"


def load_cases() -> List[Dict[str, Any]]:
    with _PROFILES.open() as f:
        return json.load(f)


def output_dir(row_name: str, suffix: str = "") -> Path:
    sfx = f"_{suffix}" if suffix else ""
    return REPO / f"data/benchmark_results/set_b/ddxplus_patientsim/{row_name}{sfx}"
