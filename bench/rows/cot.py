"""Chain-of-Thought row.

Closed-World doctor + zero-shot CoT instruction. Same dialog interface
as CW (`get_question` / `parse_ddx` / `parse_possible`).
"""

from __future__ import annotations

from typing import Any, List


NAME = "cot"


def make_doctor(
    disease_universe: List[str],
    doctor_model: str,
    api_key: str,
    max_turns: int,
    column_name: str,
    **kwargs: Any,
):
    from baselines.cot_doctor import ChainOfThoughtDoctorAgent
    from config import MoBayesConfig
    return ChainOfThoughtDoctorAgent(
        disease_names=disease_universe,
        model=doctor_model,
        api_key=api_key,
        max_turns=max_turns,
        config=MoBayesConfig(),
    )
