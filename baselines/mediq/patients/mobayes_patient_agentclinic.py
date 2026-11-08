"""AgentClinic-MedQA patient adapter for MediQ-Expert.

Parallel to mobayes_patient_patientsim.py but for AgentClinic cases:
  - Loads cases by hadm_id from MEDIQ_AC_CASES (set by baselines/mediq/row.py
    to the running column's selected_cases.jsonl)
  - Wraps AgentClinic's PatientAgent (paper-faithful patient simulator)
  - Bypasses AC's legacy openai 0.x SDK with a modern OpenAI 1.x call
    (same prompts, same conversation history mechanism — just different HTTP)

Apples-to-apples: MediQ-Expert runs against AgentClinic's published patient
simulator, while every other AC baseline (BMBE, CW, savage_*, etc.) does the
same — single source of truth for the patient side.
"""

import atexit
import json
import os
import re
import sys
from pathlib import Path

from patient import Patient


# Per-case patient-side token usage. Flushed at process exit to the path in
# MEDIQ_PATIENT_TOKENS_OUT (set by baselines/mediq/row.py). Doctor-side tokens
# are already in interactive_system.temp_additional_info[*].usage.
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

# Paths
_REPO_ROOT = Path(__file__).resolve().parents[3]  # MoBayes checkout (this file: baselines/mediq/patients/)
_AGENTCLINIC_REPO = Path(os.environ.get("AGENTCLINIC_REPO", "/tmp/agentclinic-tmp"))

if str(_AGENTCLINIC_REPO) not in sys.path:
    sys.path.insert(0, str(_AGENTCLINIC_REPO))


_CASES_PATH = Path(os.environ.get(
    "MEDIQ_AC_CASES", _REPO_ROOT / "data/cases/agentclinic_a/selected_cases.jsonl"))
_CASES_BY_HADM = None


def _load_cases_by_hadm():
    """Load AgentClinic-50 cases keyed by an `ac_NNN`-style hadm_id.
    The hadm_id is assigned in MCQ row generation (baselines/mediq/row.py).
    """
    global _CASES_BY_HADM
    if _CASES_BY_HADM is None:
        _CASES_BY_HADM = {}
        with open(_CASES_PATH) as f:
            for idx, line in enumerate(f):
                d = json.loads(line)
                _CASES_BY_HADM[f"ac_{idx:03d}"] = d
    return _CASES_BY_HADM


class ACMediQPatient(Patient):
    """Patient class that delegates to AgentClinic's PatientAgent.

    MediQ's pipeline calls .respond(question); we route each call through
    AC's PatientAgent (system_prompt + agent_hist) using a modern OpenAI 1.x
    client because AC's bundled query_model dispatcher uses the legacy
    openai 0.x SDK and a hardcoded model whitelist that excludes gpt-5.4-nano.
    """

    def __init__(self, args, sample):
        super().__init__(args, sample)
        hadm_id = sample.get("hadm_id")
        if not hadm_id:
            raise ValueError(
                "ACMediQPatient requires sample['hadm_id'] of the form 'ac_NNN'; "
                "regenerate the JSONL with baselines/mediq/row.py."
            )
        cases = _load_cases_by_hadm()
        if hadm_id not in cases:
            raise KeyError(f"hadm_id {hadm_id!r} not found in AgentClinic cases")
        self._case = cases[hadm_id]
        self._hadm_id = hadm_id
        self._wrapper = self._build_ac_agent(self._case, self.model_name)

    @staticmethod
    def _build_ac_agent(case, patient_model):
        import agentclinic
        ac_case = case if "OSCE_Examination" in case else {"OSCE_Examination": case}
        scenario = agentclinic.ScenarioMedQAExtended(ac_case)
        return agentclinic.PatientAgent(
            scenario=scenario,
            backend_str=patient_model,
            bias_present=None,
        )

    def respond(self, question):
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        agent = self._wrapper

        system = agent.system_prompt()
        user = (
            "\nHere is a history of your dialogue: " + agent.agent_hist
            + "\n Here was the doctor response: " + question
            + "Now please continue your dialogue\nPatient: "
        )

        is_reasoning = agent.backend.startswith(("gpt-5", "o"))
        kwargs = {
            "model": agent.backend,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if is_reasoning:
            kwargs["max_completion_tokens"] = 200
        else:
            kwargs["temperature"] = 0.05
            kwargs["max_tokens"] = 200

        resp = client.chat.completions.create(**kwargs)
        answer = (resp.choices[0].message.content or "").strip()
        answer = re.sub(r"\s+", " ", answer)
        agent.agent_hist += question + "\n\n" + answer + "\n\n"

        try:
            usage = resp.usage
            entry = _PATIENT_TOKEN_USAGE.setdefault(
                self._hadm_id,
                {"input_tokens": 0, "output_tokens": 0, "calls": 0},
            )
            entry["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            entry["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            entry["calls"] += 1
        except Exception:
            pass

        self.update_state(question, answer)
        return answer
