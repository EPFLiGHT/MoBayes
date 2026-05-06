"""MediQ-Expert patient adapter for HT-sim — builds an MCQ sample dict over the DDxPlus 49-disease space."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

from .base import LLMPatientWrapper


# Make MediQ importable. Clone https://github.com/stellalisy/mediQ next to this
# repo, or set MEDIQ_REPO to its path.
MEDIQ_REPO = Path(os.environ.get("MEDIQ_REPO", ""))
if str(MEDIQ_REPO / "src") not in sys.path:
    sys.path.insert(0, str(MEDIQ_REPO / "src"))


def build_mediq_sample(case: Dict[str, Any], options_dict: Dict[str, str] | None = None) -> Dict[str, Any]:
    """StreamBench row → MediQ sample dict.

    Default options_dict is a degenerate 1-MCQ where the only option is the GT
    (so any commit is a hit). Pass a dict over our DDxPlus 49 disease space if
    you want a real MCQ; the runner will map the letter back to a disease name.
    """
    pathology = case.get("PATHOLOGY", "")
    chief = case.get("INITIAL_EVIDENCE_ENG", "")
    initial_info = (
        f"Age: {case.get('AGE', '?')}; Sex: {case.get('SEX', '?')}; "
        f"Chief complaint: {chief}"
    )

    if options_dict is None:
        options_dict = {"A": pathology}
        answer_idx = "A"
    else:
        answer_idx = None
        for letter, dis in options_dict.items():
            if dis.lower().strip() == pathology.lower().strip():
                answer_idx = letter
                break
        if answer_idx is None:
            next_letter = chr(ord("A") + len(options_dict))
            options_dict[next_letter] = pathology
            answer_idx = next_letter

    return {
        "id": case.get("patient_id", "sb_unknown"),
        "hadm_id": case.get("patient_id", "sb_unknown"),
        "question": "What is the most likely diagnosis?",
        "context": [initial_info],
        "context_len": 1,
        "initial_info": initial_info,
        "options": options_dict,
        "answer": pathology,
        "answer_idx": answer_idx,
        "explanation": "",
        "facts": [],
        "patient": {"gender": case.get("SEX", ""), "age": str(case.get("AGE", "")),
                     "specialties": [], "subspecialties": []},
        "_streambench_case": case,
    }


def make_sb_mediq_patient_class(patient_model: str = "gpt-5.4-nano"):
    """Factory returning a LLMPatient-backed MediQ Patient subclass."""
    from patient import Patient  # noqa: WPS433

    class SBMediQPatient(Patient):
        def __init__(self, args, sample):
            super().__init__(args, sample)
            case = sample.get("_streambench_case")
            if case is None:
                raise ValueError("StreamBench MediQ sample missing '_streambench_case' key")
            self._wrapper = LLMPatientWrapper(case, patient_model=patient_model)

        def respond(self, question: str) -> str:
            answer = self._wrapper.respond(question)
            self.update_state(question, answer)
            return answer

    return SBMediQPatient
