"""DDxPlus50 loader for MEDDxAgent using the MoBayes benchmark patient set.

Unlike the ad-hoc sibling-repo loader, this version keeps MEDDxAgent's normal
few-shot machinery available so we can toggle `none/static/dynamic` from the
runner.

Paper-faithful fewshot pool: dynamic and static fewshot both pull from the
DDxPlus *validation* split (parent DDxPlus.get_fewshot_dataset() returns a
500-patient random sample of validate). Our 50-patient test set is disjoint
from that pool by construction. We override make_fewshot_embeddings so the
dynamic path uses the same 500-patient validate pool instead of self.patients
(which would otherwise be our 50 — leave-one-out from too small a pool).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from datasets import Dataset

from ddxdriver.benchmarks._fewshot_utils import make_and_save_embeddings
from ddxdriver.benchmarks._knn_search import KnnSearch
from ddxdriver.benchmarks.ddxplus_kb import DDxPlus


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


class BMBEDDxPlus50(DDxPlus):
    DATASET_NAME = "ddxplus"

    # Separate cache filename so we don't collide with MEDDxAgent's own cached
    # embeddings (which were built from a different 1764-patient pool).
    FEWSHOT_EMBEDDINGS_FILE = "fewshot_embeddings_bmbe50.json"

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
