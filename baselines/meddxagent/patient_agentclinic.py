"""MEDDxAgent patient adapter for AgentClinic-MedQA cases.

Parallel to BMBEPatientAgent (patient.py) but loads AgentClinic cases and
delegates each `__call__` to AgentClinic's own PatientAgent. Same wiring as
the MediQ adapter (mediQ/src/patient_agentclinic.py): we bypass AC's legacy
openai 0.x SDK with a modern OpenAI 1.x call but keep AC's prompts and
agent_hist intact.
"""

from __future__ import annotations

import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict

from ddxdriver.patient_agents.base import PatientAgent

# Make AC repo importable
_AGENTCLINIC_REPO = Path(os.environ.get("AGENTCLINIC_REPO", "/tmp/agentclinic-tmp"))
if str(_AGENTCLINIC_REPO) not in sys.path:
    sys.path.insert(0, str(_AGENTCLINIC_REPO))


class ACMEDDxAgentPatient(PatientAgent):
    """PatientAgent subclass that drives AgentClinic's PatientAgent under the hood.

    Config keys expected (passed via patient_agent_cfg):
      - selected_cases_path : path to AC selected_cases.jsonl
      - patient_model       : LLM backbone (e.g. gpt-5.4-nano)
      - api_key             : optional, falls back to OPENAI_API_KEY env
    """

    def __init__(self, patient_agent_cfg):
        super().__init__(patient_agent_cfg)
        self._load_env_file()

        self.patient_model = self.config.get("patient_model", "gpt-5.4-nano")
        self.api_key = self.config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY not set; required for AC PatientAgent")
        os.environ.setdefault("OAI_KEY", self.api_key)

        cases_path = Path(self.config["selected_cases_path"]).expanduser()
        with cases_path.open() as f:
            cases = [json.loads(l) for l in f]
        # Index by ac_NNN id (assigned by the runner)
        self.cases_by_id: Dict[str, dict] = {f"ac_{i:03d}": c for i, c in enumerate(cases)}

        self._ac_agent = None  # built per-case in reset()

    def _load_env_file(self) -> None:
        modn_root = Path(self.config.get("modn_root") or
                          Path(__file__).resolve().parents[2])
        env_path = modn_root / ".env"
        if not env_path.exists():
            return
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except Exception:
            return

    def _build_ac_agent(self, case: dict):
        import agentclinic
        ac_case = case if "OSCE_Examination" in case else {"OSCE_Examination": case}
        scenario = agentclinic.ScenarioMedQAExtended(ac_case)
        return agentclinic.PatientAgent(
            scenario=scenario,
            backend_str=self.patient_model,
            bias_present=None,
        )

    def reset(self, patient) -> None:
        super().reset(patient)
        patient_id = str(patient.patient_id)
        if patient_id not in self.cases_by_id:
            raise KeyError(f"AC case id {patient_id!r} not found in {self.config['selected_cases_path']}")
        self._ac_agent = self._build_ac_agent(deepcopy(self.cases_by_id[patient_id]))

    def __call__(self, question: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        agent = self._ac_agent

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

        # Track usage in shared module (matches BMBEPatientAgent behaviour)
        try:
            from baselines.meddxagent.usage_tracker import add_patient
            usage = resp.usage
            add_patient(prompt=getattr(usage, "prompt_tokens", 0),
                        completion=getattr(usage, "completion_tokens", 0))
        except Exception:
            pass

        self.dialogue_history.add_dialogue(("doctor", question))
        self.dialogue_history.add_dialogue(("patient", answer))
        return answer
