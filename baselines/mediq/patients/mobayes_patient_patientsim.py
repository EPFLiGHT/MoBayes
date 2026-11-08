"""BMBE PatientSim wrapper for MediQ.

MediQ's default FactSelectPatient is incompatible with DDxPlus-derived MCQs
because DDxPlus uses a *closed-world* assumption — only positive evidences
(~12 per patient) are stored, and any feature in the DDxPlus catalog that is
NOT in the patient's positive set is implicitly NEGATIVE. FactSelectPatient
does not know this rule, so it answers "I cannot answer this question" for
~80% of doctor questions, hard-capping any expert at low accuracy.

This wrapper replaces FactSelectPatient with BMBE's PatientSim
(`build_patient_system_prompt`), which:
  - Receives the full DDxPlus feature_map (sparse positives) in the system
    prompt, with explicit closed-world rules.
  - Maintains a single multi-turn chat history per patient.
  - Reuses MediQ's existing LLM helper (already GPT-5.x compatible), so the
    same backbone model (e.g. gpt-5.4-nano) is used for the patient sim.

Apples-to-apples: MediQ-Expert runs against the SAME patient simulator as
the BMBE / closed_world / savage / MEDDxAgent baselines.
"""

import json
import sys
from pathlib import Path

from patient import Patient

# Make MoBayes imports resolvable
_REPO_ROOT = Path(__file__).resolve().parents[3]  # MoBayes checkout (this file: baselines/mediq/patients/)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_interface.prompt_templates import build_patient_system_prompt  # noqa: E402

_PROFILES_PATHS = [
    _REPO_ROOT / "data/cases/ddxplus_patientsim_a/plain/profiles.json",
    _REPO_ROOT / "data/cases/ddxplus_patientsim_b/plain/profiles.json",
]
_PROFILE_CACHE = None


def _load_profiles_by_hadm():
    global _PROFILE_CACHE
    if _PROFILE_CACHE is None:
        merged = {}
        for path in _PROFILES_PATHS:
            if not path.exists():
                continue
            with open(path) as f:
                for p in json.load(f):
                    merged[p["hadm_id"]] = p
        _PROFILE_CACHE = merged
    return _PROFILE_CACHE


class BMBEPatientSim(Patient):
    """Patient subclass that delegates response generation to BMBE's PatientSim.

    Requires the converted MediQ row to carry `hadm_id`, which keys into the
    BMBE DDxPlus profile bundle. The full BMBE profile (positives + persona +
    closed-world rules) drives a single chat history kept in `self.messages`.
    """

    def __init__(self, args, sample):
        super().__init__(args, sample)
        hadm_id = sample.get("hadm_id")
        if not hadm_id:
            raise ValueError(
                "BMBEPatientSim requires sample['hadm_id']; "
                "regenerate the JSONL with baselines/mediq/row.py."
            )
        profiles = _load_profiles_by_hadm()
        if hadm_id not in profiles:
            raise KeyError(f"hadm_id {hadm_id} not found in BMBE profiles bundle")
        self.profile = profiles[hadm_id]

        self.system_prompt = build_patient_system_prompt(self.profile)
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def respond(self, question):
        self.messages.append({"role": "user", "content": question})
        response, _log_probs, _usage = self.get_response(self.messages)
        self.messages.append({"role": "assistant", "content": response})
        self.update_state(question, response)
        return response
