"""Savage et al. (npj Digital Medicine 2024) diagnostic reasoning prompts.

Two variants from Savage et al., "Diagnostic reasoning prompts reveal the
potential for large language model interpretability in medicine"
(https://www.nature.com/articles/s41746-024-01010-1), Table 2:

  - DifferentialDiagnosisDoctorAgent  — Differential Diagnosis CoT
  - BayesianReasoningDoctorAgent      — Bayesian Inference CoT

Both extend the closed-world doctor with the corresponding reasoning instruction
in the system prompt. The original Savage prompts include two single-turn
USMLE worked examples; we omit them because our setting is multi-turn dialogue.
"""

from typing import List

from scripts.core.benchmark_patientsim import StandaloneDoctorAgent
from llm_interface.prompt_templates import (
    build_savage_differential_doctor_prompt,
    build_savage_bayesian_doctor_prompt,
)


class DifferentialDiagnosisDoctorAgent(StandaloneDoctorAgent):
    """Savage et al. 2024 — Differential Diagnosis reasoning."""

    def __init__(self, disease_names: List[str], **kwargs):
        self.disease_names = list(disease_names)
        super().__init__(**kwargs)

    def reset(self):
        prompt = build_savage_differential_doctor_prompt(
            self.max_turns, self.disease_names
        )
        self.messages = [{"role": "system", "content": prompt}]
        self.turn = 0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}


class BayesianReasoningDoctorAgent(StandaloneDoctorAgent):
    """Savage et al. 2024 — Bayesian Inference reasoning."""

    def __init__(self, disease_names: List[str], **kwargs):
        self.disease_names = list(disease_names)
        super().__init__(**kwargs)

    def reset(self):
        prompt = build_savage_bayesian_doctor_prompt(
            self.max_turns, self.disease_names
        )
        self.messages = [{"role": "system", "content": prompt}]
        self.turn = 0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
