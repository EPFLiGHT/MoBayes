"""Standalone-doctor patient adapter for HT-sim — drop-in for SimplePatientAgent that routes through LLMPatient."""

from __future__ import annotations

from typing import Any, Dict

from .base import LLMPatientWrapper


class SBSimplePatient(LLMPatientWrapper):
    """Drop-in for `scripts.core.benchmark_patientsim.SimplePatientAgent`.

    Adds the surface attributes the BMBE/standalone loops read:
        .token_usage   (call counter only — LLMPatient doesn't expose tokens)
        .messages      (mirror of doctor/patient turns for logging)
    """

    def __init__(self, case: Dict[str, Any], patient_model: str = "gpt-5.4-nano",
                 **_kwargs):
        super().__init__(case, patient_model=patient_model)
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        self.messages = []

    def respond(self, question: str) -> str:
        out = super().respond(question)
        self.token_usage["calls"] += 1
        self.messages.append({"role": "user", "content": question})
        self.messages.append({"role": "assistant", "content": out})
        return out

    def reset(self):
        super().reset()
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        self.messages = []
