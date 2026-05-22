"""Adapter for MEDDxAgent running on AgentClinic-MedQA cases.

MEDDxAgent's orchestrator drives the patient via a `PatientAgent` subclass
with a `__call__(question) -> answer` interface (mirrors our existing
MoBayesPatientAgent). Here we provide an AC-backed equivalent.
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from .base import AgentClinicCaseWrapper


# Make MEDDxAgent importable
MEDDXAGENT_REPO = Path(os.environ.get("MEDDXAGENT_REPO", ""))
if str(MEDDXAGENT_REPO) not in sys.path:
    sys.path.insert(0, str(MEDDXAGENT_REPO))


def make_ac_meddxagent_patient_class(patient_model: str = "gpt-5.4-nano"):
    """Factory: returns a PatientAgent subclass that wraps an AC case.

    The MEDDxAgent runner calls patient_agent_cfg → instantiate; this
    subclass reads the AC case path from its config dict, loads the case,
    and bridges to AC's PatientAgent via AgentClinicCaseWrapper.
    """
    from ddxdriver.patient_agents.base import PatientAgent  # noqa: WPS433

    class ACMEDDxAgentPatient(PatientAgent):
        def __init__(self, patient_agent_cfg):
            super().__init__(patient_agent_cfg)
            cases_path = self.config["selected_cases_path"]
            import json
            with open(cases_path) as f:
                self._cases_by_id = {
                    self._case_key(json.loads(l)): json.loads(l)
                    for l in open(cases_path)
                }
            # Keep wrapper handle once reset() is called for a given patient.
            self._wrapper: AgentClinicCaseWrapper | None = None

        @staticmethod
        def _case_key(case: Dict[str, Any]) -> str:
            # AC cases don't have hadm_ids — we use the disease name + index
            # assigned at sample time (sample_cases.py adds _disease_id).
            osce = case.get("OSCE_Examination", case)
            return f"{osce.get('Correct_Diagnosis')}::{case.get('_case_idx', 0)}"

        def reset(self, patient) -> None:
            super().reset(patient)
            patient_id = str(patient.patient_id)
            if patient_id not in self._cases_by_id:
                raise KeyError(
                    f"AC case id {patient_id!r} not found in {self.config['selected_cases_path']}"
                )
            self._wrapper = AgentClinicCaseWrapper(
                ac_case=deepcopy(self._cases_by_id[patient_id]),
                patient_model=patient_model,
            )

        def __call__(self, question: str) -> str:
            self.dialogue_history.add_dialogue(("doctor", question))
            answer = self._wrapper.respond(question)
            self.dialogue_history.add_dialogue(("patient", answer))
            return answer

    return ACMEDDxAgentPatient
