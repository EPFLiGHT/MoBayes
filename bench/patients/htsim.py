"""HT-sim patient: MEDDxAgent's LLMPatient (history-taking simulator over
DDxPlus cases from the StreamBench split) wrapped behind a
``respond(question) -> str`` interface. Used by the ``htsim`` / ``htsim_v1``
bench columns and by the HT-sim runners of other baselines."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


# Make MEDDxAgent importable
MEDDXAGENT_REPO = Path(os.environ.get("MEDDXAGENT_REPO", ""))
if str(MEDDXAGENT_REPO) not in sys.path:
    sys.path.insert(0, str(MEDDXAGENT_REPO))


def _load_llm_patient_class():
    from ddxdriver.patient_agents.llm_patient import LLMPatient  # noqa: WPS433
    from ddxdriver.utils import Patient  # noqa: WPS433
    return LLMPatient, Patient


def _build_patient_object(case: Dict[str, Any]):
    """Build the MEDDxAgent `Patient` dataclass from a StreamBench row.

    StreamBench fields used:
        AGE, SEX, INITIAL_EVIDENCE_ENG, PATIENT_PROFILE,
        DIFFERENTIAL_DIAGNOSIS, PATHOLOGY
    The PATIENT_PROFILE field already contains the full self-narrative the
    LLMPatient prompt expects.
    """
    _, Patient = _load_llm_patient_class()
    pathology = case.get("PATHOLOGY", "")
    initial_info = (
        f"Age: {case.get('AGE', '?')}\n"
        f"Sex: {case.get('SEX', '?')}\n"
        f"Chief Complaint: {case.get('INITIAL_EVIDENCE_ENG', '')}"
    )
    profile_str = case["PATIENT_PROFILE"]
    if isinstance(profile_str, str) and '""' in profile_str:
        profile_str = profile_str.split('""')[1]

    # gt_ddx — parse from DIFFERENTIAL_DIAGNOSIS
    gt_ddx = []
    raw_dd = case.get("DIFFERENTIAL_DIAGNOSIS", "")
    if raw_dd:
        from ast import literal_eval
        try:
            parsed = literal_eval(raw_dd) if isinstance(raw_dd, str) else raw_dd
            gt_ddx = [item[0] if isinstance(item, list) else item for item in parsed]
        except Exception:
            gt_ddx = [pathology]

    return Patient(
        patient_id=case.get("patient_id", "sb_unknown"),
        patient_initial_info=initial_info,
        gt_pathology=pathology,
        patient_profile=profile_str,
        gt_ddx=gt_ddx,
    )


# ── Common wrapper ────────────────────────────────────────────────────────

class LLMPatientWrapper:
    """Wraps a single HT-sim case (StreamBench DDxPlus row) in MEDDxAgent's LLMPatient.

    Exposes:
        .respond(question) -> str
        .ground_truth      (PATHOLOGY)
        .initial_info      (Age / Sex / Chief Complaint string)
        .reset()
    """

    def __init__(self, case: Dict[str, Any], patient_model: str = "gpt-5.4-nano",
                 base_url: Optional[str] = None) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY not set; required for LLMPatient")

        # MEDDxAgent expects OAI_KEY too; mirror it from OPENAI_API_KEY
        os.environ.setdefault("OAI_KEY", os.environ["OPENAI_API_KEY"])

        LLMPatient, _Patient = _load_llm_patient_class()

        # patient_agent_cfg is what `self.config` becomes inside LLMPatient.
        # LLMPatient reads self.config["model"]["class_name"] etc., so the
        # "model" key must live at the top level (no extra "config" wrapper).
        cfg = {
            "model": {
                "class_name": "ddxdriver.models.oai_chat.OpenAIChat",
                "config": {"model_name": patient_model},
            }
        }

        self._patient_obj = _build_patient_object(case)
        self._llm_patient = LLMPatient(cfg)
        # LLMPatient.reset() takes the patient — must be called before use
        self._llm_patient.reset(self._patient_obj)

        self._gt = self._patient_obj.gt_pathology
        self._initial_info = self._patient_obj.patient_initial_info
        self._case = case

    @property
    def ground_truth(self) -> str:
        return self._gt

    @property
    def initial_info(self) -> str:
        return self._initial_info

    @property
    def patient_id(self) -> str:
        return self._case.get("patient_id", "sb_unknown")

    def respond(self, question: str) -> str:
        return self._llm_patient(question)

    def reset(self):
        self._llm_patient.reset(self._patient_obj)
