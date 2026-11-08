"""DDxPlus 50 v1 (MEDDxAgent HT-sim) — second-occurrence sample (disjoint from canonical).

Same wrappers as `htsim.py`, only case path + output dir differ.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from bench.columns.htsim import (  # noqa: F401  (column interface)
    disease_universe, gt_of, make_patient, matches_gt,
)


NAME = "htsim_v1"

REPO = Path(__file__).resolve().parents[2]
_CASES_PATH = REPO / "data/cases/ddxplus_htsim_b/selected_cases.jsonl"


def load_cases() -> List[Dict[str, Any]]:
    with _CASES_PATH.open() as f:
        return [json.loads(l) for l in f]


def output_dir(row_name: str, suffix: str = "") -> Path:
    sfx = f"_{suffix}" if suffix else ""
    return REPO / f"data/benchmark_results/set_b/ddxplus_htsim/{row_name}{sfx}"
