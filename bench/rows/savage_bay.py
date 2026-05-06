"""Savage 2024 — Bayesian Inference CoT row.

Reasoning prompt: prior over diseases, update with each evidence,
posterior. From Savage et al., npj Digital Medicine 2024.
"""

from __future__ import annotations

from typing import Any, List


NAME = "savage_bay"


def make_doctor(
    disease_universe: List[str],
    doctor_model: str,
    api_key: str,
    max_turns: int,
    column_name: str,
    **kwargs: Any,
):
    from baselines.savage_doctor import BayesianReasoningDoctorAgent
    from config import MoBayesConfig
    return BayesianReasoningDoctorAgent(
        disease_names=disease_universe,
        model=doctor_model,
        api_key=api_key,
        max_turns=max_turns,
        config=MoBayesConfig(),
    )
