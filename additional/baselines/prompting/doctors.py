"""Prompt-based LLM doctors.

All four share the standalone doctor's dialogue loop, [DDX]/[ABSTAIN] parsing
and abstention behaviour (``StandaloneDoctorAgent``); they differ only in the
system prompt, which lists the KB's diseases (closed world) and, for the
reasoning variants, adds a reasoning instruction:

    ClosedWorldDoctorAgent             closed-world disease list (row cw)
    ChainOfThoughtDoctorAgent          + zero-shot CoT, Wei et al. 2022 (row cot)
    DifferentialDiagnosisDoctorAgent   + Differential Diagnosis CoT (row savage_dd)
    BayesianReasoningDoctorAgent       + Bayesian Inference CoT (row savage_bay)

The two Savage et al. variants follow "Diagnostic reasoning prompts reveal the
potential for large language model interpretability in medicine" (npj Digital
Medicine 2024, https://www.nature.com/articles/s41746-024-01010-1), Table 2.
Their two single-turn USMLE worked examples are omitted because our setting
is multi-turn dialogue.
"""

from typing import List

from scripts.core.benchmark_patientsim import StandaloneDoctorAgent
from llm_interface.prompt_templates import (
    build_chain_of_thought_doctor_prompt,
    build_closed_world_doctor_prompt,
    build_savage_bayesian_doctor_prompt,
    build_savage_differential_doctor_prompt,
)


class _ClosedWorldPromptDoctor(StandaloneDoctorAgent):
    """Standalone doctor whose system prompt is ``build_prompt(max_turns, disease_names)``."""

    build_prompt = None

    def __init__(self, disease_names: List[str], **kwargs):
        self.disease_names = list(disease_names)
        super().__init__(**kwargs)

    def reset(self):
        prompt = self.build_prompt(self.max_turns, self.disease_names)
        self.messages = [{"role": "system", "content": prompt}]
        self.turn = 0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}


class ClosedWorldDoctorAgent(_ClosedWorldPromptDoctor):
    build_prompt = staticmethod(build_closed_world_doctor_prompt)


class ChainOfThoughtDoctorAgent(_ClosedWorldPromptDoctor):
    build_prompt = staticmethod(build_chain_of_thought_doctor_prompt)


class DifferentialDiagnosisDoctorAgent(_ClosedWorldPromptDoctor):
    build_prompt = staticmethod(build_savage_differential_doctor_prompt)


class BayesianReasoningDoctorAgent(_ClosedWorldPromptDoctor):
    build_prompt = staticmethod(build_savage_bayesian_doctor_prompt)
