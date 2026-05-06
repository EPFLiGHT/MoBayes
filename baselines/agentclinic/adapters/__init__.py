"""AgentClinic patient adapters for our baselines.

Each module wraps the AgentClinic PatientAgent in the interface a particular
baseline expects:

    base.AgentClinicCaseWrapper      — common LLM-patient wrapper
    standalone.ACSimplePatient       — sa / cw / cot / savage_*  (benchmark_patientsim loop)
    mediq.ACMediQPatient             — MediQ-Expert (Patient subclass)
    meddxagent.ACMEDDxAgentPatient   — MEDDxAgent (PatientAgent subclass)
    cod.make_ac_cod_patient_callable — CoD / DiagnosisGPT (factory closure)
    bmbe.ACBMBEPatient               — BMBE benchmark loop wrapper
"""

from .base import (
    AgentClinicCaseWrapper,
    load_selected_cases,
    matches_ac_gt,
)
from .standalone import ACSimplePatient
from .bmbe import ACBMBEPatient, build_minimal_profile_for_bmbe
from .cod import make_ac_cod_patient_callable
# mediq + meddxagent factories are lazy-imported by their consumers because
# they require the corresponding sibling repos on sys.path.
