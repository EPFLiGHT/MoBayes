"""HT-sim patient-side adapters — wrap MEDDxAgent's LLMPatient for each baseline's expected interface."""

from .base import LLMPatientWrapper, load_selected_cases
from .standalone import SBSimplePatient
from .bmbe import SBBMBEPatient
from .cod import make_sb_cod_patient_callable
# mediq + meddxagent factories are lazy-imported by their consumers.
