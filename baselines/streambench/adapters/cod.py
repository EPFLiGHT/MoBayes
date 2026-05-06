"""CoD / DiagnosisGPT-34B patient adapter for HT-sim — factory returning (respond, get_usage, gt, init_info)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from .base import LLMPatientWrapper


def make_sb_cod_patient_callable(case: Dict[str, Any],
                                 patient_model: str = "gpt-5.4-nano",
                                 ) -> Tuple[Callable[[str], str], Callable[[], Dict[str, int]], str, str]:
    wrapper = LLMPatientWrapper(case, patient_model=patient_model)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def respond(question: str) -> str:
        out = wrapper.respond(question)
        usage["calls"] += 1
        return out

    def get_usage() -> Dict[str, int]:
        return dict(usage)

    return respond, get_usage, wrapper.ground_truth, wrapper.initial_info
