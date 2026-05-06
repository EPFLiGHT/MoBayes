"""Closed-world LLM doctor: standalone baseline + KB disease list in prompt.

Identical to the standalone baseline in every other respect (turn loop, parsing,
abstention behavior). The only difference is that the system prompt lists the
diseases present in the knowledge base, so the LLM operates with the same
candidate set as BMBE.
"""

from typing import List

from scripts.core.benchmark_patientsim import StandaloneDoctorAgent
from llm_interface.prompt_templates import build_closed_world_doctor_prompt


class ClosedWorldDoctorAgent(StandaloneDoctorAgent):
    def __init__(self, disease_names: List[str], **kwargs):
        self.disease_names = list(disease_names)
        super().__init__(**kwargs)

    def reset(self):
        prompt = build_closed_world_doctor_prompt(
            self.max_turns, self.disease_names
        )
        self.messages = [{"role": "system", "content": prompt}]
        self.turn = 0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
