"""BMBE patient adapter for HT-sim — feeds LLMPatient through SimplePatient with a minimal profile dict."""

from __future__ import annotations

from typing import Any, Dict

from .standalone import SBSimplePatient


SBBMBEPatient = SBSimplePatient


def build_minimal_profile_for_bmbe(case: Dict[str, Any]) -> Dict[str, Any]:
    """BMBE's loop reads chief complaint and basic demographics off the
    profile dict; the rest of the diagnostic state lives in the engine + KB.
    """
    chief = case.get("INITIAL_EVIDENCE_ENG", "")
    return {
        "hadm_id": case.get("patient_id", "sb_unknown"),
        "_streambench_case": case,
        "Current Visit": {"chiefcomplaint": chief},
        "_meta": {
            "ground_truth_name": case.get("PATHOLOGY", ""),
            "age": case.get("AGE"),
            "sex": case.get("SEX"),
        },
    }
