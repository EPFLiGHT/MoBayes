"""Adapter for benchmark_patientsim.py-style baselines (sa, cw, cot, savage_*).

These baselines all instantiate `SimplePatientAgent(profile, ...)` and call
`patient.respond(question)` in a loop. We replace SimplePatientAgent with
`ACSimplePatient` which is fed an AgentClinic OSCE case and routes responses
through the AgentClinic PatientAgent.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import AgentClinicCaseWrapper


class ACSimplePatient(AgentClinicCaseWrapper):
    """Drop-in replacement for `scripts.core.benchmark_patientsim.SimplePatientAgent`.

    Matches the SimplePatientAgent interface:
        .respond(question) -> str
        .reset()
        .token_usage  (dict with prompt/completion totals)
        .messages     (list — exposed for downstream logging if needed)
    """

    def __init__(self, ac_case: Dict[str, Any], patient_model: str = "gpt-5.4-nano",
                 bias_present: Optional[str] = None, **_kwargs):
        super().__init__(ac_case, patient_model=patient_model, bias_present=bias_present)
        # SimplePatientAgent exposes these for the BMBE/standalone loops.
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        self.messages = []

    def respond(self, question: str) -> str:
        out = super().respond(question)
        self.token_usage["calls"] += 1
        # AC's PatientAgent doesn't expose per-call usage; we count calls only
        # (Run-level token usage will be approximated from log volume.)
        self.messages.append({"role": "user", "content": question})
        self.messages.append({"role": "assistant", "content": out})
        return out

    def reset(self):
        super().reset()
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        self.messages = []
