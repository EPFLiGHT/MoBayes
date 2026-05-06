"""Adapter for CoD / DiagnosisGPT-34B running on AgentClinic-MedQA cases.

CoD's runner (`scripts/baselines/run_cod_diagnosisgpt.py`) takes a
`make_patient_callable(profile, patient_model, api_key)` factory and uses the
returned `(respond_fn, get_usage_fn)` closure inside its chat loop.

We provide an AC-equivalent factory below.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from .base import AgentClinicCaseWrapper


def make_ac_cod_patient_callable(ac_case: Dict[str, Any],
                                  patient_model: str = "gpt-5.4-nano",
                                  bias_present: Optional[str] = None,
                                  ) -> Tuple[Callable[[str], str], Callable[[], Dict[str, int]], str, str]:
    """Returns (respond, get_usage, ground_truth, initial_info).

    Mirror of `run_cod_diagnosisgpt.make_patient_callable` for the AC backend.
    """
    wrapper = AgentClinicCaseWrapper(
        ac_case, patient_model=patient_model, bias_present=bias_present,
    )

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def respond(question: str) -> str:
        out = wrapper.respond(question)
        usage["calls"] += 1
        return out

    def get_usage() -> Dict[str, int]:
        return dict(usage)

    return respond, get_usage, wrapper.ground_truth, wrapper.initial_info
