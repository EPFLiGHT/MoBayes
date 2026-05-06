"""Adapter for BMBE running on AgentClinic-MedQA cases.

BMBE's benchmark loop (`run_mobayes_session` in
`scripts/core/benchmark_patientsim.py`) instantiates a `SimplePatientAgent`
from a profile dict and uses the AgentClinic 31-disease KB built under
`data/cases/agentclinic_a/`. The patient side is identical to other baselines —
we use ACSimplePatient. The doctor side runs unchanged but reads the
AgentClinic KB instead of DDxPlus.

The ground-truth comparison is more involved because BMBE outputs a
disease_id from the 31-disease universe; we map that to the AgentClinic
free-text GT via name matching against the disease catalogue.
"""

from __future__ import annotations

from typing import Any, Dict

from .standalone import ACSimplePatient


# Re-export the same patient the standalone baselines use — BMBE's loop also
# treats the patient as a generic respond-callable.
ACBMBEPatient = ACSimplePatient


def build_minimal_profile_for_bmbe(ac_case: Dict[str, Any], hadm_id: str) -> Dict[str, Any]:
    """Build the minimal profile dict that BMBE's benchmark loop needs.

    BMBE only reads a few fields off the profile (chief complaint, age, sex,
    initial info). The rest of the diagnostic state lives in the Bayesian
    engine / KB. We therefore pass a lightweight dict — the AC PatientAgent
    handles the actual LLM patient simulation.
    """
    osce = ac_case.get("OSCE_Examination", ac_case)
    pa = osce.get("Patient_Actor", {})
    demo = pa.get("Demographics", "")
    chief = pa.get("Symptoms", {}).get("Primary_Symptom", "")
    objective = osce.get("Objective_for_Doctor", "")
    gt = osce.get("Correct_Diagnosis", "")
    return {
        "hadm_id": hadm_id,
        "_ac_case": ac_case,
        "Current Visit": {"chiefcomplaint": chief or objective},
        "_ac_meta": {
            "ground_truth_name": gt,
            "demographics": demo,
        },
    }
