"""Bench rows for the prompt-based doctors: each builds a doctor and
bench/runner.py runs the generic per-case loop."""

from __future__ import annotations

from importlib import import_module
from typing import Any, List


class PromptRow:
    def __init__(self, name: str, doctor: str, closed_world: bool = True) -> None:
        self.NAME = name
        self._doctor = doctor            # "module:Class"
        self._closed_world = closed_world

    def make_doctor(self, disease_universe: List[str], doctor_model: str, api_key: str,
                    max_turns: int, column_name: str, **kwargs: Any):
        from config import MoBayesConfig
        module, cls = self._doctor.split(":")
        doctor_cls = getattr(import_module(module), cls)
        extra = {"disease_names": disease_universe} if self._closed_world else {}
        return doctor_cls(
            model=doctor_model,
            api_key=api_key,
            max_turns=max_turns,
            base_url=kwargs.get("base_url"),
            config=MoBayesConfig(),
            **extra,
        )


# Standalone LLM doctor: no disease list, the model controls questioning,
# stopping and diagnosis on its own.
SA = PromptRow("sa", "scripts.core.benchmark_patientsim:StandaloneDoctorAgent", closed_world=False)
CW = PromptRow("cw", "baselines.prompting.doctors:ClosedWorldDoctorAgent")
COT = PromptRow("cot", "baselines.prompting.doctors:ChainOfThoughtDoctorAgent")
SAVAGE_DD = PromptRow("savage_dd", "baselines.prompting.doctors:DifferentialDiagnosisDoctorAgent")
SAVAGE_BAY = PromptRow("savage_bay", "baselines.prompting.doctors:BayesianReasoningDoctorAgent")
