"""Chain-of-Thought LLM doctor: closed-world prompt + zero-shot CoT.

Standard zero-shot Chain-of-Thought (Wei et al., NeurIPS 2022) applied to the
closed-world doctor baseline. Same dialog loop, parsing, and abstention behavior
as the standalone / closed-world doctor; the only difference is the system
prompt instructs the LLM to reason step-by-step (wrapped in [REASONING] tags)
before each question, [DDX], or [ABSTAIN].
"""

from typing import List

from scripts.core.benchmark_patientsim import StandaloneDoctorAgent
from llm_interface.prompt_templates import build_chain_of_thought_doctor_prompt


class ChainOfThoughtDoctorAgent(StandaloneDoctorAgent):
    def __init__(self, disease_names: List[str], **kwargs):
        self.disease_names = list(disease_names)
        super().__init__(**kwargs)

    def reset(self):
        prompt = build_chain_of_thought_doctor_prompt(
            self.max_turns, self.disease_names
        )
        self.messages = [{"role": "system", "content": prompt}]
        self.turn = 0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
