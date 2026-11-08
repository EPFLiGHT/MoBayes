"""DDxPlus HT-simulator patient adapter for MediQ-Expert.

Mirror of mobayes_patient_agentclinic.py / mobayes_patient_patientsim.py:
    - Loads cases from data/cases/ddxplus_htsim_{a,b}/selected_cases.jsonl
      keyed by hadm_id of the form ``htsim_NNNN`` (test-split index).
    - Wraps MEDDxAgent's published LLMPatient (history-taking simulator) via
      the MoBayes LLMPatientWrapper (bench/patients/htsim.py).

Apples-to-apples: every baseline in the DDxPlus HT-simulator column drives
the same MEDDxAgent LLMPatient.
"""

import atexit
import json
import os
import sys
from pathlib import Path

from patient import Patient


_REPO_ROOT = Path(__file__).resolve().parents[3]  # MoBayes checkout (this file: baselines/mediq/patients/)
_MEDDXAGENT_ROOT = Path(os.environ.get("MEDDXAGENT_REPO", ""))

for p in (_REPO_ROOT, _MEDDXAGENT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


_CASES_PATHS = [
    _REPO_ROOT / "data/cases/ddxplus_htsim_a/selected_cases.jsonl",
    _REPO_ROOT / "data/cases/ddxplus_htsim_b/selected_cases.jsonl",
]
_CASES_BY_HADM = None

_PATIENT_TOKEN_USAGE: dict = {}


def _flush_patient_tokens() -> None:
    out_path = os.environ.get("MEDIQ_PATIENT_TOKENS_OUT")
    if not out_path or not _PATIENT_TOKEN_USAGE:
        return
    try:
        with open(out_path, "w") as f:
            json.dump(_PATIENT_TOKEN_USAGE, f, indent=2)
    except Exception:
        pass


atexit.register(_flush_patient_tokens)


def _load_cases_by_hadm():
    global _CASES_BY_HADM
    if _CASES_BY_HADM is None:
        _CASES_BY_HADM = {}
        for path in _CASES_PATHS:
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    d = json.loads(line)
                    _CASES_BY_HADM[d["patient_id"]] = d
    return _CASES_BY_HADM


class DDxHTSimMediQPatient(Patient):
    """Patient class that delegates to MEDDxAgent's LLMPatient (HT simulator)."""

    def __init__(self, args, sample):
        super().__init__(args, sample)
        hadm_id = sample.get("hadm_id")
        if not hadm_id:
            raise ValueError(
                "DDxHTSimMediQPatient requires sample['hadm_id'] of form 'htsim_NNNN'"
            )
        cases = _load_cases_by_hadm()
        if hadm_id not in cases:
            raise KeyError(f"hadm_id {hadm_id!r} not found in HT-sim cases")
        self._case = cases[hadm_id]
        self._hadm_id = hadm_id

        from bench.patients.htsim import LLMPatientWrapper
        from baselines.meddxagent import usage_tracker  # patches MEDDxAgent OpenAIChat
        self._wrapper = LLMPatientWrapper(self._case, patient_model=self.model_name)
        self._tracker = usage_tracker

    def respond(self, question):
        before = self._tracker.snapshot()["patient"]
        answer = self._wrapper.respond(question)
        after = self._tracker.snapshot()["patient"]
        entry = _PATIENT_TOKEN_USAGE.setdefault(
            self._hadm_id,
            {"input_tokens": 0, "output_tokens": 0, "calls": 0},
        )
        entry["input_tokens"] += after["prompt_tokens"] - before["prompt_tokens"]
        entry["output_tokens"] += after["completion_tokens"] - before["completion_tokens"]
        entry["calls"] += 1
        self.update_state(question, answer)
        return answer
