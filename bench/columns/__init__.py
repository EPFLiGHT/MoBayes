"""Column adapters: per-benchmark case loader + patient + disease universe."""

from importlib import import_module
from typing import Protocol, runtime_checkable, List, Dict, Any
from pathlib import Path


@runtime_checkable
class Column(Protocol):
    """Contract every column module must satisfy."""

    NAME: str
    """Short id, used in output dir."""

    def load_cases(self) -> List[Dict[str, Any]]:
        """Return the list of cases (one dict per case)."""

    def make_patient(self, case: Dict[str, Any], patient_model: str):
        """Return a patient object exposing `respond(question)->str`,
        plus `initial_info: str`, `token_usage: dict`, and (optional)
        a `_scenario` attribute used by AC native row."""

    def disease_universe(self) -> List[str]:
        """Closed-world disease list shown to the doctor."""

    def output_dir(self, row_name: str, suffix: str = "") -> Path:
        """Where to write per-case JSONs for this (row, column)."""

    def matches_gt(self, prediction: str, ground_truth: str) -> bool:
        """Whether `prediction` counts as a hit for `ground_truth`."""

    def gt_of(self, case: Dict[str, Any]) -> str:
        """Extract the ground-truth diagnosis from a case dict."""


def get(name: str) -> Column:
    """Lazy-load by name: get('ac') → bench.columns.ac module."""
    return import_module(f"bench.columns.{name}")
