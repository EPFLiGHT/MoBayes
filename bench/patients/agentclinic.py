"""AgentClinic-MedQA patient: AgentClinic's own PatientAgent (Schmidgall et
al., 2024) wrapped behind a ``respond(question) -> str`` interface.

Used by the ``ac`` / ``ac_set2`` bench columns and by the CoD AgentClinic
runner. AgentClinic's ``query_model`` only supports a fixed model whitelist
on the legacy openai 0.x SDK, so ``respond`` keeps AC's prompts and history
but sends the request through the openai 1.x client.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


# Make AgentClinic source importable. Override with AGENTCLINIC_REPO env var.
AGENTCLINIC_REPO = Path(os.environ.get("AGENTCLINIC_REPO", "/tmp/agentclinic-tmp"))
if str(AGENTCLINIC_REPO) not in sys.path:
    sys.path.insert(0, str(AGENTCLINIC_REPO))


def _load_ac_classes():
    """Lazy-import to avoid breaking when running outside AC repo."""
    import agentclinic  # noqa: WPS433
    return agentclinic.PatientAgent, agentclinic.ScenarioMedQAExtended


def _ensure_openai_key():
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not set; required for AgentClinic PatientAgent"
        )


# ── Core wrapper used by every baseline adapter ───────────────────────────

class AgentClinicCaseWrapper:
    """Wraps a single AgentClinic OSCE case.

    Holds an internal AgentClinic PatientAgent instance (their LLM-based
    patient sim). Exposes a simple `respond(question) -> answer` method and
    metadata accessors. Subclasses adapt this to baseline-specific interfaces.
    """

    def __init__(self, ac_case: Dict[str, Any], patient_model: str = "gpt-5.4-nano",
                 bias_present: Optional[str] = None) -> None:
        _ensure_openai_key()
        PatientAgent, ScenarioMedQAExtended = _load_ac_classes()

        # ScenarioMedQAExtended expects a dict with the OSCE_Examination key
        if "OSCE_Examination" not in ac_case:
            ac_case = {"OSCE_Examination": ac_case}
        self._scenario = ScenarioMedQAExtended(ac_case)
        self._gt = self._scenario.diagnosis_information()
        self._initial_info = self._scenario.examiner_information()

        self._patient_agent = PatientAgent(
            scenario=self._scenario,
            backend_str=patient_model,
            bias_present=bias_present,
        )
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    @property
    def ground_truth(self) -> str:
        return self._gt

    @property
    def initial_info(self) -> str:
        """Objective_for_Doctor — what the doctor sees up-front."""
        return self._initial_info

    def respond(self, question: str) -> str:
        """One-step: doctor asks, patient answers. History kept internally.

        We bypass AgentClinic's `query_model` dispatcher because it hardcodes
        a list of supported model names (gpt4o, gpt-4o-mini, ...) and uses the
        legacy openai 0.x SDK. Instead we route directly through our standard
        OpenAI client, preserving AC's prompt structure (system_prompt +
        history-augmented user prompt).
        """
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        system = self._patient_agent.system_prompt()
        user = (
            "\nHere is a history of your dialogue: " + self._patient_agent.agent_hist
            + "\n Here was the doctor response: " + question
            + "Now please continue your dialogue\nPatient: "
        )

        # gpt-5.x reasoning models reject `temperature` and `max_tokens`;
        # use `max_completion_tokens` and skip temperature.
        is_reasoning = self._patient_agent.backend.startswith(("gpt-5", "o"))
        kwargs = {"model": self._patient_agent.backend,
                   "messages": [
                       {"role": "system", "content": system},
                       {"role": "user", "content": user},
                   ]}
        if is_reasoning:
            kwargs["max_completion_tokens"] = 200
        else:
            kwargs["temperature"] = 0.05
            kwargs["max_tokens"] = 200

        resp = client.chat.completions.create(**kwargs)
        answer = (resp.choices[0].message.content or "").strip()
        # Mirror AC's post-processing
        import re
        answer = re.sub(r"\s+", " ", answer)
        # Update AC's internal history (so next turn sees this exchange)
        self._patient_agent.agent_hist += question + "\n\n" + answer + "\n\n"
        # Track token usage (patient side)
        try:
            u = resp.usage
            self.token_usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            self.token_usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            self.token_usage["calls"] += 1
        except Exception:
            pass
        return answer

    def reset(self):
        self._patient_agent.reset()
