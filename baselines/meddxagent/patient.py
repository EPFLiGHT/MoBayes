"""MEDDxAgent patient adapter backed by MoBayes's PatientSim profiles."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict

from ddxdriver.patient_agents.base import PatientAgent

from config import MoBayesConfig
from llm_interface.llm_client import OpenAILLMClient, sanitize_openai_text
from llm_interface.prompt_templates import build_patient_system_prompt


class BMBEPatientAgent(PatientAgent):
    """Patient agent that answers from our raw benchmark profiles."""

    def __init__(self, patient_agent_cfg):
        super().__init__(patient_agent_cfg)
        self.modn_root = Path(
            self.config.get("modn_root") or Path(__file__).resolve().parents[2]
        )
        self._load_env_file()

        cfg = MoBayesConfig()
        self.patient_model = self.config.get("patient_model", cfg.default_model)
        self.patient_temperature = float(
            self.config.get("patient_temperature", cfg.patient_temperature)
        )
        self.patient_max_tokens = int(
            self.config.get("patient_max_tokens", cfg.patient_max_tokens)
        )
        self.base_url = self.config.get("base_url")
        self.api_key = self.config.get("api_key")

        profiles_path = Path(self.config["profiles_path"]).expanduser()
        with profiles_path.open() as f:
            profiles = json.load(f)
        self.profiles_by_id: Dict[str, dict] = {
            str(profile["hadm_id"]): profile for profile in profiles
        }

        self.profile = None
        self.messages = []
        self.llm = OpenAILLMClient(
            model=self.patient_model,
            api_key=self.api_key,
            temperature=self.patient_temperature,
            base_url=self.base_url,
        )

    def _load_env_file(self) -> None:
        env_path = self.modn_root / ".env"
        if not env_path.exists():
            return
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path)
        except Exception:
            return

    def reset(self, patient) -> None:
        super().reset(patient)
        patient_id = str(patient.patient_id)
        if patient_id not in self.profiles_by_id:
            raise KeyError(
                f"Patient id {patient_id!r} not found in profiles file "
                f"{self.config['profiles_path']}"
            )

        self.profile = deepcopy(self.profiles_by_id[patient_id])
        system_prompt = build_patient_system_prompt(self.profile)
        self.messages = [{"role": "system", "content": system_prompt}]

    def __call__(self, question: str) -> str:
        self.dialogue_history.add_dialogue(("doctor", question))
        self.messages.append(
            {"role": "user", "content": sanitize_openai_text(question)}
        )
        answer, usage = self.llm.chat(
            self.messages,
            max_tokens=self.patient_max_tokens,
        )
        # Track patient-side tokens via the shared accumulator
        try:
            from baselines.meddxagent.usage_tracker import add_patient
            add_patient(
                prompt=usage.get("prompt_tokens", 0),
                completion=usage.get("completion_tokens", 0),
            )
        except Exception:  # noqa: BLE001
            pass
        self.messages.append({"role": "assistant", "content": answer})
        self.dialogue_history.add_dialogue(("patient", answer))
        return answer
