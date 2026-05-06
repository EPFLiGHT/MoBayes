"""Savage 2024 — Differential Diagnosis CoT row.

Reasoning prompt: list candidate differentials, supporting/refuting
evidence, then commit. From Savage et al., npj Digital Medicine 2024.
"""

from __future__ import annotations

from typing import Any, List


NAME = "savage_dd"


def make_doctor(
    disease_universe: List[str],
    doctor_model: str,
    api_key: str,
    max_turns: int,
    column_name: str,
    **kwargs: Any,
):
    from baselines.savage_doctor import DifferentialDiagnosisDoctorAgent
    from config import MoBayesConfig
    return DifferentialDiagnosisDoctorAgent(
        disease_names=disease_universe,
        model=doctor_model,
        api_key=api_key,
        max_turns=max_turns,
        config=MoBayesConfig(),
    )
