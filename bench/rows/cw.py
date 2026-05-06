"""Closed-World LLM doctor row.

Standalone doctor + KB disease list in the system prompt. Same dialog
loop, parsing, and abstention behaviour as the standalone baseline; the
only delta is that the prompt enumerates the closed disease set.
"""

from __future__ import annotations

from typing import Any, List


NAME = "cw"


def make_doctor(
    disease_universe: List[str],
    doctor_model: str,
    api_key: str,
    max_turns: int,
    column_name: str,
    **kwargs: Any,
):
    from baselines.closed_world_doctor import ClosedWorldDoctorAgent
    from config import MoBayesConfig
    return ClosedWorldDoctorAgent(
        disease_names=disease_universe,
        model=doctor_model,
        api_key=api_key,
        max_turns=max_turns,
        config=MoBayesConfig(),
    )
