"""MEDDxAgent patient adapter for HT-sim — builds the patient_agent_cfg dict pointing at our selected_cases.jsonl."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Make MEDDxAgent importable
MEDDXAGENT_REPO = Path(os.environ.get("MEDDXAGENT_REPO", ""))
if str(MEDDXAGENT_REPO) not in sys.path:
    sys.path.insert(0, str(MEDDXAGENT_REPO))


def build_streambench_bench_cfg(selected_cases_jsonl: str | Path,
                                 doctor_model: str = "gpt-5.4-nano") -> Dict[str, Any]:
    """Return a MEDDxAgent bench_cfg that loads our 50 StreamBench cases.

    Reuses MEDDxAgent's `ddxdriver.benchmarks.ddxplus_kb.DDxPlus` as the base —
    the only thing we override is `load_dataset`, which is handled by a tiny
    subclass declared in the runner that reads from the JSONL.
    """
    return {
        "class_name": "baselines.streambench.adapters.meddxagent.StreamBench50Bench",
        "config": {
            "selected_cases_path": str(selected_cases_jsonl),
            "knn_search_cfg": {
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "pooling": "cls",
                "order": "similar_at_top",
                "seed": 42,
                "precompute_new_embeddings": False,
            },
        },
        "num_patients": 50,
        "enforce_diagnosis_options": True,
    }


# A thin Bench subclass that loads from our jsonl instead of HuggingFace
def _make_bench_class():
    """Lazy-import to avoid breaking outside the meddxagent venv."""
    from datasets import Dataset
    from ddxdriver.benchmarks.ddxplus_kb import DDxPlus

    class StreamBench50Bench(DDxPlus):
        DATASET_NAME = "ddxplus"

        def __init__(self, selected_cases_path: str, *args, **kwargs):
            self._cases_path = Path(selected_cases_path).expanduser()
            super().__init__(*args, **kwargs)

        def load_dataset(self):
            import json
            with self._cases_path.open() as f:
                rows = [json.loads(l) for l in f]
            return Dataset.from_list(rows)

        def get_patient_id(self, patient_entry):
            return patient_entry.get("patient_id", patient_entry.get("id"))

    return StreamBench50Bench


# Eagerly-bind name so MEDDxAgent's class_name resolution works at runtime
try:
    StreamBench50Bench = _make_bench_class()
except Exception:
    # Outside the meddxagent venv — fine; the runner always invokes inside it
    StreamBench50Bench = None  # type: ignore
