"""MEDDxAgent bench class that feeds our 50-case JSONL instead of the full DDxPlus test split."""

from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset

from ddxdriver.benchmarks.ddxplus_kb import DDxPlus


def _resolve_cases_path() -> Path:
    env = os.environ.get("DDXPLUS_HTSIM_CASES_PATH")
    if env:
        return Path(env)
    # Fallback: locate relative to this module
    # baselines/streambench/htsim_meddxagent_bench.py → BMBE_ROOT
    bmbe_root = Path(__file__).resolve().parents[2]
    return bmbe_root / "data/cases/ddxplus_htsim_a/selected_cases.jsonl"


class DDxPlusHTSim50(DDxPlus):
    DATASET_NAME = "ddxplus"

    def load_dataset(self) -> Dataset:
        path = _resolve_cases_path()
        with path.open() as f:
            rows = [json.loads(l) for l in f]
        # Optional: skip first N cases (set DDXPLUS_HTSIM_START_IDX) to resume
        # after a partial run.
        start = int(os.environ.get("DDXPLUS_HTSIM_START_IDX", "0") or "0")
        if start > 0:
            rows = rows[start:]
        return Dataset.from_list(rows)

    def get_patient_id(self, patient_entry):
        return patient_entry.get("patient_id", patient_entry.get("id"))
