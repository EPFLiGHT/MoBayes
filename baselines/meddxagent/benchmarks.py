"""MEDDxAgent (ddxdriver) bench classes, one per bench column. Each feeds our
cases to MEDDxAgent through its DDxPlus bench interface:

    MoBayesDDxPlus50   PatientSim profiles (data/cases/ddxplus_patientsim_*)
    DDxPlusHTSim50     HT-sim cases (data/cases/ddxplus_htsim_*)
    AgentClinic50      AgentClinic-MedQA cases (data/cases/agentclinic_*)

Referenced by class path from baselines/meddxagent/run.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

from datasets import Dataset

from ddxdriver.benchmarks._fewshot_utils import make_and_save_embeddings
from ddxdriver.benchmarks._knn_search import KnnSearch
from ddxdriver.benchmarks.ddxplus import DDxPlus


# ── PatientSim ────────────────────────────────────────────────────────────
# DDxPlus50 loader for MEDDxAgent using the MoBayes benchmark patient set.
#
# Unlike the ad-hoc sibling-repo loader, this version keeps MEDDxAgent's normal
# few-shot machinery available so we can toggle `none/static/dynamic` from the
# runner.
#
# Paper-faithful fewshot pool: dynamic and static fewshot both pull from the
# DDxPlus *validation* split (parent DDxPlus.get_fewshot_dataset() returns a
# 500-patient random sample of validate). Our 50-patient test set is disjoint
# from that pool by construction. We override make_fewshot_embeddings so the
# dynamic path uses the same 500-patient validate pool instead of self.patients
# (which would otherwise be our 50 — leave-one-out from too small a pool).

_DEFAULT_PROFILES_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "cases" / "ddxplus_patientsim_a" / "plain" / "profiles.json"
)


def _gender_to_sex(gender: str) -> str:
    g = (gender or "").strip().lower()
    if g.startswith("m"):
        return "M"
    if g.startswith("f"):
        return "F"
    return "U"


def _format_patient_profile(profile: dict) -> str:
    sex = _gender_to_sex(profile.get("gender", ""))
    age = profile.get("age", "?")
    sex_word = "Male" if sex == "M" else ("Female" if sex == "F" else "Unknown")

    current_visit = profile.get("Current Visit", {}) or {}
    present_illness = profile.get("Present Illness", {}) or {}
    social_history = profile.get("Social History", {}) or {}
    medical_history = profile.get("Medical History", {}) or {}

    bullets = []
    chief = current_visit.get("chiefcomplaint", "")
    pos = present_illness.get("present_illness_positive", "")
    neg = present_illness.get("present_illness_negative", "")

    if chief:
        bullets.append(f"- Chief complaint: {chief}")
    if pos:
        bullets.append(f"- I am experiencing: {pos}")
    if neg:
        bullets.append(f"- I am NOT experiencing: {neg}")

    for key in ("tobacco", "alcohol", "occupation", "travel"):
        value = social_history.get(key)
        if value and str(value).lower() != "unknown":
            bullets.append(f"- {key.title()}: {value}")

    for key in ("medical_history", "family_medical_history", "allergies"):
        value = medical_history.get(key)
        if value and str(value).lower() != "unknown":
            bullets.append(f"- {key.replace('_', ' ').title()}: {value}")

    body_lines = [f"Sex: {sex_word}, Age: {age}"] + bullets
    body = "\n".join(body_lines)
    return f'patient profile: ""{body}""'


def _profile_to_streambench_row(profile: dict, idx: int) -> dict:
    meta = profile.get("_ddxplus_meta", {}) or {}
    diagnosis = meta.get("diagnosis_name") or ""
    current_visit = profile.get("Current Visit", {}) or {}
    chief = current_visit.get("chiefcomplaint", "")

    initial_evidence = chief.split(",")[0].split(" and ")[0].strip()[:80]
    if not initial_evidence:
        initial_evidence = "presenting complaint"

    differential = str([[diagnosis, 1.0]])
    return {
        "id": idx,
        "hadm_id": profile.get("hadm_id"),
        "AGE": profile.get("age"),
        "SEX": _gender_to_sex(profile.get("gender", "")),
        "INITIAL_EVIDENCE_ENG": initial_evidence,
        "EVIDENCES_ENG": [],
        "PATIENT_PROFILE": _format_patient_profile(profile),
        "DIFFERENTIAL_DIAGNOSIS": differential,
        "PATHOLOGY": diagnosis,
        "INITIAL_EVIDENCE": "",
        "EVIDENCES": [],
    }


class MoBayesDDxPlus50(DDxPlus):
    DATASET_NAME = "ddxplus"

    # Separate cache filename so we don't collide with MEDDxAgent's own cached
    # embeddings (which were built from a different 1764-patient pool).
    FEWSHOT_EMBEDDINGS_FILE = "fewshot_embeddings_mobayes50.json"

    def __init__(self, profiles_path: str | None = None, *args, **kwargs):
        self.profiles_path = (
            Path(profiles_path).expanduser() if profiles_path else _DEFAULT_PROFILES_PATH
        )
        super().__init__(*args, **kwargs)

    def load_dataset(self) -> Dataset:
        if not self.profiles_path.exists():
            raise FileNotFoundError(f"BMBE profiles file not found: {self.profiles_path}")
        with self.profiles_path.open() as f:
            profiles = json.load(f)
        rows = [_profile_to_streambench_row(profile, i) for i, profile in enumerate(profiles)]
        return Dataset.from_list(rows)

    def get_patient_id(self, patient_entry):
        return patient_entry.get("hadm_id", patient_entry.get("id"))

    def make_fewshot_embeddings(self, knn_search: KnnSearch, filename: Union[str, Path]):
        """Build embeddings from the 500-patient validate-split fewshot pool
        (parent DDxPlus.get_fewshot_dataset()), NOT from our 50 test patients.

        Without this override, MEDDxAgent's base.py uses self.patients for
        dynamic-fewshot embeddings — which on our 50-patient test set means
        leave-one-out from a tiny pool.
        """
        fewshot_patients = self.get_fewshot_dataset()
        make_and_save_embeddings(fewshot_patients, knn_search, filename)


# ── HT-sim ────────────────────────────────────────────────────────────────
# MEDDxAgent bench class that feeds our 50-case JSONL instead of the full DDxPlus test split.

def _resolve_cases_path() -> Path:
    env = os.environ.get("DDXPLUS_HTSIM_CASES_PATH")
    if env:
        return Path(env)
    # Fallback: set a of the HT-sim column
    return Path(__file__).resolve().parents[2] / "data/cases/ddxplus_htsim_a/selected_cases.jsonl"


class DDxPlusHTSim50(DDxPlus):
    DATASET_NAME = "ddxplus"

    def load_dataset(self) -> Dataset:
        path = _resolve_cases_path()
        with path.open() as f:
            rows = [json.loads(l) for l in f]
        # Optional: skip first N cases (set DDXPLUS_HTSIM_START_IDX) to resume
        # after a partial run.
        start = int(os.environ.get("DDXPLUS_HTSIM_START_IDX", "0") or "0")
        if start > 0:
            rows = rows[start:]
        return Dataset.from_list(rows)

    def get_patient_id(self, patient_entry):
        return patient_entry.get("patient_id", patient_entry.get("id"))


# ── AgentClinic-MedQA ─────────────────────────────────────────────────────
# AgentClinic-50 benchmark loader for MEDDxAgent.
#
# Parallel to MoBayesDDxPlus50 but reads our AgentClinic selected_cases.jsonl
# and converts each case into MEDDxAgent's internal patient row schema. The
# diagnosis_options are pinned to our 31-disease AC-50 KB list so the
# diagnosis agent picks from a closed set congruent with our setup.
#
# For dynamic fewshot (paper-best), we override ``make_fewshot_embeddings``
# to index the full AC-MedQA-Extended source pool (213 cases = 50 selected
# + 163 held-out) instead of just self.patients (50 selected). At retrieval
# time MEDDxAgent's ``get_dynamic_fewshot`` does self-exclusion via patient
# id, so for each evaluation case the 49 same-distribution selected cases +
# 163 held-out are candidates. This mirrors DDxPlus paper-best where the
# pool == eval set with self-exclusion (1,764 test patients), only here we
# expand to also include held-out cases.
#
# For static / no-fewshot variants we keep the existing behavior
# (``get_fewshot_dataset`` returns []).

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CASES_PATH = _REPO_ROOT / "data" / "cases" / "agentclinic_a" / "selected_cases.jsonl"
_DEFAULT_DISEASES_CSV = _REPO_ROOT / "data" / "kbs" / "agentclinic_a" / "diseases.csv"
# Full AC-MedQA-Extended source — 213 cases. Used as the dynamic-fewshot
# pool for paper-best (50 selected as eval + 163 held-out as expansion;
# self-exclusion at retrieval keeps things honest).
_DEFAULT_SOURCE_POOL_PATH = Path(
    "/tmp/agentclinic-tmp/agentclinic_medqa_extended.jsonl"
)


def _ac_case_to_streambench_row(case: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Convert one AC case to MEDDxAgent's internal patient row schema."""
    osce = case if "OSCE_Examination" in case else {"OSCE_Examination": case}
    inner = osce["OSCE_Examination"]
    pa = inner.get("Patient_Actor", {})
    demo = pa.get("Demographics", "")
    history = pa.get("History", "")
    chief = pa.get("Symptoms", {}).get("Primary_Symptom", "")
    secondary = pa.get("Symptoms", {}).get("Secondary_Symptoms", []) or []
    pmh = pa.get("Past_Medical_History", "")
    sh = pa.get("Social_History", "")
    diagnosis = inner.get("Correct_Diagnosis", "")

    # Extract age / sex from demographics if possible
    age = ""
    sex = "U"
    import re
    m = re.search(r"(\d+)[-\s]*year[-\s]*old", demo, flags=re.I)
    if m:
        age = m.group(1)
    if "female" in demo.lower() or "woman" in demo.lower():
        sex = "F"
    elif "male" in demo.lower() or "man" in demo.lower():
        sex = "M"

    profile_lines = [demo, history]
    if chief:
        profile_lines.append(f"Primary symptom: {chief}")
    if secondary:
        profile_lines.append("Secondary symptoms: " + ", ".join(secondary))
    if pmh:
        profile_lines.append(f"Past medical history: {pmh}")
    if sh:
        profile_lines.append(f"Social history: {sh}")
    profile_str = '\n'.join(p for p in profile_lines if p)
    patient_profile = f'patient profile: ""{profile_str}""'

    initial_evidence = chief or "presenting complaint"
    differential = str([[diagnosis, 1.0]])

    return {
        "id": idx,
        "hadm_id": f"ac_{idx:03d}",
        "AGE": age,
        "SEX": sex,
        "INITIAL_EVIDENCE_ENG": initial_evidence,
        "EVIDENCES_ENG": [],
        "PATIENT_PROFILE": patient_profile,
        "DIFFERENTIAL_DIAGNOSIS": differential,
        "PATHOLOGY": diagnosis,
        "INITIAL_EVIDENCE": "",
        "EVIDENCES": [],
    }


class AgentClinic50(DDxPlus):
    """MEDDxAgent benchmark class backed by our AC-50 selected cases."""

    DATASET_NAME = "agentclinic"
    FEWSHOT_EMBEDDINGS_FILE = "fewshot_embeddings_ac50.json"
    SPECIALIST_PREFACE = (
        "You are a specialist diagnosing patients across 31 conditions ranging "
        "from gastroenterology, dermatology, oncology, neurology, "
        "rheumatology, and infectious disease."
    )

    def __init__(self, cases_path: str | None = None,
                 diseases_csv: str | None = None,
                 source_pool_path: str | None = None,
                 *args, **kwargs):
        self.cases_path = Path(cases_path).expanduser() if cases_path else _DEFAULT_CASES_PATH
        self.diseases_csv = Path(diseases_csv).expanduser() if diseases_csv else _DEFAULT_DISEASES_CSV
        self.source_pool_path = (
            Path(source_pool_path).expanduser() if source_pool_path
            else _DEFAULT_SOURCE_POOL_PATH
        )
        super().__init__(*args, **kwargs)

    # Parent DDxPlus.__init__ calls self.load_disease_mapping() which expects
    # a diagnosis_options.txt file under the dataset cache. We bypass and
    # return our 31-disease list from the KB CSV.
    def load_disease_mapping(self) -> List[str]:
        import pandas as pd
        return pd.read_csv(self.diseases_csv)["name"].tolist()

    def load_diagnosis_options(self) -> List[str]:
        """Override Bench default — read from KB CSV instead of bundled txt."""
        import pandas as pd
        return pd.read_csv(self.diseases_csv)["name"].tolist()

    def load_dataset(self) -> Dataset:
        if not self.cases_path.exists():
            raise FileNotFoundError(f"AC cases not found: {self.cases_path}")
        with self.cases_path.open() as f:
            cases = [json.loads(l) for l in f]
        rows = [_ac_case_to_streambench_row(c, i) for i, c in enumerate(cases)]
        return Dataset.from_list(rows)

    def get_fewshot_dataset(self) -> List:
        """Static fewshot pool — full 213-case AC source.

        Used only when fewshot.type == "static". Dynamic routing flows
        through ``make_fewshot_embeddings`` (see below). Eval cases are
        filtered out by MEDDxAgent's static-fewshot self-exclusion.
        """
        return self._build_full_source_patients()

    def get_patient_id(self, patient_entry):
        return patient_entry.get("hadm_id", patient_entry.get("id"))

    # ------------------------------------------------------------------ #
    #  Paper-best dynamic fewshot pool                                    #
    #                                                                     #
    #  MEDDxAgent's default ``make_fewshot_embeddings`` (base.py:235)     #
    #  embeds ``self.patients`` (i.e. the eval set) and self-excludes at  #
    #  retrieval. We override to embed the FULL AC-MedQA-Extended source  #
    #  (213 cases = 50 selected + 163 held-out). Self-exclusion at the    #
    #  retrieval site (base.py:217) keeps the test case out of its own    #
    #  KNN. Disease overlap with the test universe is preserved (cases    #
    #  with freq>=2 in source contribute multiple same-disease shots).   #
    # ------------------------------------------------------------------ #
    def _eval_case_keys(self):
        """Return a dict mapping a stable case-content key (the inner OSCE
        json) → eval hadm_id, for the 50 selected cases. Used to detect
        when a source-pool case is the same as one we are evaluating, so
        we can assign it the eval hadm_id (enabling proper self-exclusion).
        """
        with self.cases_path.open() as f:
            selected = [json.loads(l) for l in f]
        keys = {}
        for idx, c in enumerate(selected):
            inner = c.get("OSCE_Examination", c)
            # Stable content key: GT diagnosis + first 60 chars of demographics
            pa = inner.get("Patient_Actor", {})
            demo = (pa.get("Demographics", "") or "")[:60]
            key = (inner.get("Correct_Diagnosis", "").lower().strip(), demo)
            keys[key] = f"ac_{idx:03d}"
        return keys

    def _build_full_source_patients(self):
        """Load the 213 AC-MedQA-Extended source cases as MEDDxAgent Patients.

        Source cases that match an eval case (same Correct_Diagnosis +
        Demographics) get the eval hadm_id "ac_NNN" so that self-exclusion
        at retrieval correctly drops the test case from its own KNN.
        Held-out cases get a distinct id "ac_src_NNNN".
        """
        if not self.source_pool_path.exists():
            raise FileNotFoundError(
                f"AC source pool missing: {self.source_pool_path}. "
                "Set AGENTCLINIC_REPO or pass --source-pool-path."
            )
        with self.source_pool_path.open() as f:
            source = [json.loads(l) for l in f]
        eval_keys = self._eval_case_keys()

        patients = []
        n_eval_in_pool = 0
        for src_idx, src in enumerate(source):
            inner = src.get("OSCE_Examination", src)
            pa = inner.get("Patient_Actor", {})
            demo = (pa.get("Demographics", "") or "")[:60]
            key = (inner.get("Correct_Diagnosis", "").lower().strip(), demo)
            row = _ac_case_to_streambench_row(src, src_idx)
            if key in eval_keys:
                row["hadm_id"] = eval_keys[key]   # mirror eval id → self-excl
                row["id"] = int(eval_keys[key].split("_")[1])
                n_eval_in_pool += 1
            else:
                row["hadm_id"] = f"ac_src_{src_idx:04d}"
                row["id"] = 100000 + src_idx
            patients.append(self.create_patient(patient_entry=row))
        # Sanity check: all 50 eval cases should be findable in source.
        if n_eval_in_pool != len(eval_keys):
            import warnings
            warnings.warn(
                f"AC pool: only {n_eval_in_pool}/{len(eval_keys)} eval cases "
                "found in source by content key — self-exclusion may be partial."
            )
        return patients

    def make_fewshot_embeddings(self, knn_search, filename):
        """Override base default — embed the full 213-case AC source pool
        instead of just self.patients (the 50 eval cases). Self-exclusion
        at retrieval handles eval-test leakage cleanly.
        """
        from pathlib import Path as _P
        from ddxdriver.benchmarks._fewshot_utils import make_and_save_embeddings
        _P(filename).parent.mkdir(parents=True, exist_ok=True)
        make_and_save_embeddings(self._build_full_source_patients(), knn_search, filename)
