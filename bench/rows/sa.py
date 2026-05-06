"""Standalone LLM doctor row.

Pure LLM doctor with no external reasoning aid. The model controls
questioning, stopping, and diagnosis autonomously.
"""

from __future__ import annotations

from typing import Any, List


NAME = "sa"


def make_doctor(
    disease_universe: List[str],
    doctor_model: str,
    api_key: str,
    max_turns: int,
    column_name: str,
    **kwargs: Any,
):
    from scripts.core.benchmark_patientsim import StandaloneDoctorAgent
    from config import MoBayesConfig
    return StandaloneDoctorAgent(
        model=doctor_model,
        api_key=api_key,
        max_turns=max_turns,
        base_url=kwargs.get("base_url"),
        config=MoBayesConfig(),
    )
