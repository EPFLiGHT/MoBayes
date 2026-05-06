"""Adapter for MediQ-Expert running on AgentClinic-MedQA cases.

MediQ's pipeline expects samples in iMEDQA-style MCQ format with a Patient
subclass driving the patient side. We construct degenerate single-option MCQ
samples (the "options" dict has one entry — the GT) so MediQ commits to a
letter that is then matched post-hoc against the AC free-text GT by the
runner.

This file provides:
    build_mediq_sample(ac_case, hadm_id) — converts an OSCE case to MediQ's
                                            sample dict format.
    ACMediQPatient(Patient)              — Patient subclass that routes
                                            responses through AC's PatientAgent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

from .base import AgentClinicCaseWrapper


# Make MediQ importable. Set MEDIQ_REPO to a local clone of
# https://github.com/stellalisy/mediQ
MEDIQ_REPO = Path(os.environ.get("MEDIQ_REPO", ""))
if str(MEDIQ_REPO / "src") not in sys.path:
    sys.path.insert(0, str(MEDIQ_REPO / "src"))


def build_mediq_sample(ac_case: Dict[str, Any], hadm_id: str,
                       options_dict: Dict[str, str] | None = None) -> Dict[str, Any]:
    """Convert an AgentClinic OSCE case to a MediQ sample dict.

    By default we make a degenerate 1-option MCQ where the only option IS the
    correct answer (so any commit is a hit). MediQ-Expert's Likert-abstention
    pipeline still runs question generation; the GT-matching is what changes.

    For a fairer evaluation, pass options_dict mapping letter -> disease_name
    where the letters are e.g. our 31-disease universe (so MediQ has to pick
    among real distractors).
    """
    osce = ac_case.get("OSCE_Examination", ac_case)
    pa = osce.get("Patient_Actor", {})
    history = pa.get("History", "")
    demo = pa.get("Demographics", "")
    chief_complaint = pa.get("Symptoms", {}).get("Primary_Symptom", "")
    gt = osce.get("Correct_Diagnosis", "")

    if options_dict is None:
        options_dict = {"A": gt}
        answer_idx = "A"
    else:
        # Find which letter holds the GT (case-insensitive match)
        answer_idx = None
        for letter, dis in options_dict.items():
            if dis.lower().strip() == gt.lower().strip():
                answer_idx = letter
                break
        if answer_idx is None:
            # GT not in options → add it as last entry
            next_letter = chr(ord("A") + len(options_dict))
            options_dict[next_letter] = gt
            answer_idx = next_letter

    return {
        "id": hadm_id,
        "hadm_id": hadm_id,
        "question": "What is the most likely diagnosis?",
        "context": [demo, history],
        "context_len": 2,
        "initial_info": f"{demo}. {chief_complaint}.",
        "options": options_dict,
        "answer": gt,
        "answer_idx": answer_idx,
        "explanation": "",
        "facts": [history] if history else [],
        "patient": {"gender": demo, "age": "", "specialties": [], "subspecialties": []},
        "_ac_case": ac_case,  # kept so the patient class can rebuild the wrapper
    }


def make_ac_mediq_patient_class(patient_model: str = "gpt-5.4-nano"):
    """Factory returning an AC-backed MediQ Patient subclass.

    We build the class lazily because MediQ's `Patient` import requires the
    mediQ src dir to be on sys.path (handled at module-load time above).
    """
    from patient import Patient  # noqa: WPS433

    class ACMediQPatient(Patient):
        def __init__(self, args, sample):
            super().__init__(args, sample)
            ac_case = sample.get("_ac_case")
            if ac_case is None:
                raise ValueError("AC sample missing '_ac_case' key — use build_mediq_sample")
            self._wrapper = AgentClinicCaseWrapper(
                ac_case, patient_model=patient_model
            )

        def respond(self, question: str) -> str:
            answer = self._wrapper.respond(question)
            self.update_state(question, answer)
            return answer

    return ACMediQPatient
