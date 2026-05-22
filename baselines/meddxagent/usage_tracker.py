"""Module-level usage accumulators for MEDDxAgent runs.

MEDDxAgent's own OpenAIChat throws away the usage dict. Their orchestrator
spins up multiple Model instances (one per agent), so we can't simply keep
a self.total_usage on the doctor side. Patient-side MoBayesPatientAgent has a
single OpenAILLMClient instance but the runner can't easily reach it after
run_ddxdriver() returns.

Approach: two module-level dicts (DOCTOR / PATIENT). The patched OpenAIChat
adds to DOCTOR; MoBayesPatientAgent adds to PATIENT after each chat. The runner
imports `snapshot()` after run_ddxdriver() and writes results to disk.
"""

from __future__ import annotations

from typing import Dict


DOCTOR: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
PATIENT: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def reset() -> None:
    for d in (DOCTOR, PATIENT):
        d["prompt_tokens"] = 0
        d["completion_tokens"] = 0
        d["calls"] = 0


def add_doctor(prompt: int, completion: int) -> None:
    DOCTOR["prompt_tokens"] += int(prompt or 0)
    DOCTOR["completion_tokens"] += int(completion or 0)
    DOCTOR["calls"] += 1


def add_patient(prompt: int, completion: int) -> None:
    PATIENT["prompt_tokens"] += int(prompt or 0)
    PATIENT["completion_tokens"] += int(completion or 0)
    PATIENT["calls"] += 1


def snapshot() -> Dict[str, Dict[str, int]]:
    return {
        "doctor": dict(DOCTOR),
        "patient": dict(PATIENT),
    }
