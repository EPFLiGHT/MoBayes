"""Adapter: AgentClinic's published DoctorAgent → our doctor interface.

AgentClinic's `DoctorAgent` (agentclinic.py:455) is the AC paper's own
self-baseline doctor: an LLM that asks 1-3 sentence questions, can REQUEST
TEST/IMAGES, commits with "DIAGNOSIS READY: <diagnosis>" when ready.

This wrapper exposes the same `get_question(patient_response) → (str, bool)`
+ `parse_ddx` + `parse_possible` API used by every other AC baseline in
batch_smoke_baselines.py. The internals (inference_doctor, agent_hist,
system_prompt, MAX_INFS) are AC's published code, untouched.

The HTTP call goes through a modern OpenAI 1.x client (same bypass we do
for AC's PatientAgent) because AC's `query_model` uses the legacy SDK +
hardcoded model whitelist that excludes gpt-5.4-nano.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


_AGENTCLINIC_REPO = Path(os.environ.get("AGENTCLINIC_REPO", "/tmp/agentclinic-tmp"))
if str(_AGENTCLINIC_REPO) not in sys.path:
    sys.path.insert(0, str(_AGENTCLINIC_REPO))


class ACNativeDoctorAgent:
    """Drives AgentClinic's published DoctorAgent. Disease list is unused
    (AC doctor is open-vocabulary; we map the free-text final to our 31
    closed-set via name matching downstream)."""

    def __init__(self, scenario, disease_names: List[str], model: str = "gpt-5.4-nano",
                 api_key: str | None = None, max_turns: int = 20,
                 bias_present: str | None = None):
        import agentclinic
        self.scenario = scenario
        self.disease_names = disease_names
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.max_turns = max_turns
        self._doctor = agentclinic.DoctorAgent(
            scenario=scenario,
            backend_str=model,
            max_infs=max_turns,
            bias_present=bias_present,
            img_request=False,
        )
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def _ac_inference(self, question: str) -> str:
        """Replace AC's query_model (legacy openai 0.x) with modern 1.x."""
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        agent = self._doctor
        if agent.infs >= agent.MAX_INFS:
            return "Maximum inferences reached"
        system = agent.system_prompt()
        user = (
            "\nHere is a history of your dialogue: " + agent.agent_hist
            + "\n Here was the patient response: " + question
            + "Now please continue your dialogue\nDoctor: "
        )
        is_reasoning = self.model.startswith(("gpt-5", "o"))
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if is_reasoning:
            kwargs["max_completion_tokens"] = 400
        else:
            kwargs["temperature"] = 0.05
            kwargs["max_tokens"] = 400
        resp = client.chat.completions.create(**kwargs)
        answer = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.token_usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.token_usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            self.token_usage["calls"] += 1
        agent.agent_hist += question + "\n\n" + answer + "\n\n"
        agent.infs += 1
        return answer

    def get_question(self, patient_response: str) -> Tuple[str, bool]:
        out = self._ac_inference(patient_response)
        if "DIAGNOSIS READY:" in out.upper():
            # Translate AC's free-text final into the [DDX] / [ABSTAIN]
            # protocol that batch_smoke_baselines.run_one() recognises.
            m = re.search(r"DIAGNOSIS READY:\s*(.+?)(?:\n|$)", out, flags=re.I)
            free_dx = (m.group(1) if m else "").strip()
            mapped = self._map_to_closed(free_dx)
            if mapped:
                ddx_block = "\n".join(f"{i+1}. {d}" for i, d in enumerate(mapped[:3]))
                wrapped = f"{out}\n\n[DDX]\n{ddx_block}"
            else:
                wrapped = (f"{out}\n\n[ABSTAIN] free-text diagnosis "
                           f"{free_dx!r} did not map to the closed disease set")
            return wrapped, True
        return out, False

    def _norm(self, s: str) -> str:
        return (s or "").lower().strip()

    def _map_to_closed(self, free_text: str) -> List[str]:
        """Map AC's free-text diagnosis to the closed disease set.

        Returns only diseases that substring-match the free text (no
        alphabetical fill — that biases everything to the first disease
        when free_text is empty/generic). Empty list ⇒ caller treats as
        unmapped (top1=Unknown, miss).
        """
        norm_in = self._norm(free_text)
        if not norm_in or len(norm_in) < 4:
            return []
        ranked, seen = [], set()
        for d in self.disease_names:
            n = self._norm(d)
            if len(n) < 4:
                continue
            if (n in norm_in) or (len(norm_in) >= 5 and norm_in in n):
                if d not in seen:
                    ranked.append(d); seen.add(d)
        return ranked

    def parse_ddx(self, text: str) -> List[str]:
        m = re.search(r"DIAGNOSIS READY:\s*(.+?)(?:\n|$)", text, flags=re.I)
        if not m:
            return []
        return self._map_to_closed(m.group(1))

    def parse_possible(self, text: str) -> List[str]:
        return self.parse_ddx(text)
