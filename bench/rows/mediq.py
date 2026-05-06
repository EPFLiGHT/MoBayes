"""MediQ-Expert row — drives stellalisy/mediQ in a subprocess.

Doctor protocol:
  - ScaleExpert (Likert 1-5 confidence per turn)
  - max_questions=10 (paper default)
  - rationale_generation=True, self_consistency=3
  - abstain_threshold=4.0

Two paper rows in Table 4:
  - "force-final" (paper-faithful, Cov=100%) — letter_choice picked at end
  - "strict abstain" (timeouts → abstain)

Aggregator: scripts/analysis/mediq_force_vs_strict.py reads the raw mediQ
JSONL and produces both numbers from the same run.

This row is BATCH-mode: builds one JSONL with N cases, runs ONE subprocess,
post-processes per-case JSONs into our schema. Per-case loop in runner.py
is bypassed.

Currently supports column='ac' (AC-MedQA). HT-sim / PatientSim use distinct
mediQ patient adapters; extend by adding another `_mediq_patient_for(column)`
case here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


NAME = "mediq"

_MEDIQ_REPO = Path(os.environ.get("MEDIQ_REPO", Path.home() / "Documents/GitHub/mediQ"))


def make_doctor(*args, **kwargs):
    raise NotImplementedError("mediq is batch-mode; uses run_batch.")


def _mediq_patient_module(column_name: str) -> tuple[str, str]:
    """Returns (patient_module, patient_class) for mediQ subprocess.

    Adapter classes live in mediQ/src/ — already implemented for all 3 columns.
    """
    return {
        "ac":            ("patient_agentclinic",   "ACMediQPatient"),
        "ac_set2":       ("patient_agentclinic",   "ACMediQPatient"),
        "htsim":         ("patient_ddxplus_htsim", "DDxHTSimMediQPatient"),
        "htsim_v1":      ("patient_ddxplus_htsim", "DDxHTSimMediQPatient"),
        "patientsim":    ("patient_patientsim",    "BMBEPatientSim"),
        "patientsim_v1": ("patient_patientsim",    "BMBEPatientSim"),
    }[column_name]


def _hadm_id(column_name: str, idx: int, case: Dict[str, Any]) -> str:
    """Per-column hadm_id format expected by the mediQ patient adapter."""
    if column_name in ("ac", "ac_set2"):
        return f"ac_{idx:03d}"
    if column_name in ("htsim", "htsim_v1"):
        return case.get("patient_id", f"htsim_{idx:04d}")
    if column_name in ("patientsim", "patientsim_v1"):
        return case.get("hadm_id", f"patientsim_{idx:04d}")
    raise ValueError(column_name)


def _initial_info_for(column_name: str, case: Dict[str, Any]) -> tuple[str, str, str]:
    """Per-column initial_info / context extraction for the MediQ JSONL row.

    Returns (initial_info, demo, history). Demo/history are passed in
    `context[]` and `facts[]`; the patient adapter on the mediQ side may
    or may not use them (most adapters look up via hadm_id).
    """
    if column_name in ("ac", "ac_set2"):
        inner = case.get("OSCE_Examination", case)
        pa = inner.get("Patient_Actor", {}) or {}
        demo    = pa.get("Demographics", "")
        history = pa.get("History", "")
        chief   = pa.get("Symptoms", {}).get("Primary_Symptom", "")
        return f"{demo}. {chief}.".strip(". "), demo, history
    if column_name in ("htsim", "htsim_v1"):
        chief = case.get("INITIAL_EVIDENCE_ENG", "")
        demo  = f"Age {case.get('AGE','?')}, Sex {case.get('SEX','?')}"
        prof  = case.get("PATIENT_PROFILE", "")
        if isinstance(prof, str) and '""' in prof:
            prof = prof.split('""')[1]
        return f"{demo}. {chief}.".strip(". "), demo, prof
    if column_name in ("patientsim", "patientsim_v1"):
        chief = case.get("Current Visit", {}).get("chiefcomplaint", "")
        demo  = f"Age {case.get('age','?')}, Sex {case.get('gender','?')}"
        history = json.dumps(case.get("Medical History", {}))[:1000]
        return f"{demo}. {chief}.".strip(". "), demo, history
    raise ValueError(column_name)


def _build_mediq_jsonl(cases: List[Dict[str, Any]], column, start_idx: int,
                        diseases: List[str], jsonl_path: Path) -> None:
    """Convert column cases → MediQ-format JSONL (id, hadm_id, options, ...).

    For HT-sim/PatientSim the disease list may include category headers; we
    only need leaf-disease names for the MCQ. We restrict the option pool to
    pathologies that actually appear in the test cases (mirroring the
    existing per-column runners).
    """
    used = sorted({column.gt_of(c) for c in cases if column.gt_of(c)})
    seen = {d.lower().strip() for d in used}
    extras = [d for d in diseases if d.lower().strip() not in seen]
    options_pool = used + extras
    # Extend letter alphabet enough for any plausible pool size (A..Z, AA..ZZ).
    letters = ([chr(ord("A") + i) for i in range(26)]
               + [chr(ord("A") + i) + chr(ord("A") + j)
                  for i in range(26) for j in range(26)])
    options_base = {letters[i]: d for i, d in enumerate(options_pool[:len(letters)])}

    with jsonl_path.open("w") as f:
        for offset, case in enumerate(cases):
            idx = start_idx + offset
            gt = column.gt_of(case)
            initial_info, demo, history = _initial_info_for(column.NAME, case)

            # Find or insert GT into options (so MediQ can pick the right letter)
            opts = dict(options_base)
            answer_idx = next(
                (l for l, d in opts.items() if d.lower().strip() == gt.lower().strip()),
                None,
            )
            if answer_idx is None:
                next_letter = (letters[len(opts)]
                               if len(opts) < len(letters) else f"X{len(opts)}")
                opts[next_letter] = gt
                answer_idx = next_letter

            sample = {
                "id": idx,
                "hadm_id": _hadm_id(column.NAME, idx, case),
                "question": "What is the most likely diagnosis?",
                "context": [demo, history],
                "context_len": 2,
                "initial_info": initial_info,
                "options": opts,
                "answer": gt,
                "answer_idx": answer_idx,
                "explanation": "",
                "facts": [history] if history else [],
                "patient": {"gender": demo, "age": "", "specialties": [], "subspecialties": []},
            }
            f.write(json.dumps(sample) + "\n")


def run_batch(*, cases: List[Dict[str, Any]], start_idx: int, column,
              doctor_model: str, patient_model: str, max_turns: int,
              api_key: str, out_dir: Path, row_kwargs: Dict[str, Any]) -> None:
    """Build MediQ JSONL → subprocess → convert results to per-case JSONs."""
    if not _MEDIQ_REPO.exists():
        sys.exit(f"ERROR: MEDIQ_REPO={_MEDIQ_REPO} not found. "
                 "Set MEDIQ_REPO env var to the mediQ checkout.")

    diseases = column.disease_universe()
    pat_module, pat_class = _mediq_patient_module(column.NAME)

    # MediQ expects its inputs under <repo>/data, outputs under <repo>/logs/<name>
    data_name = row_kwargs.get(
        "mediq_data_name", f"{column.NAME}_{start_idx}_{start_idx+len(cases)}_mediq.jsonl",
    )
    log_subdir = row_kwargs.get(
        "mediq_log_subdir", f"{column.NAME}_{start_idx}_{start_idx+len(cases)}",
    )
    mediq_data = _MEDIQ_REPO / "data" / data_name
    mediq_data.parent.mkdir(parents=True, exist_ok=True)
    out_log_dir = _MEDIQ_REPO / "logs" / log_subdir
    out_log_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = out_log_dir / "results.jsonl"
    if output_jsonl.exists():
        output_jsonl.unlink()
    patient_tokens_path = out_log_dir / "patient_tokens.json"
    if patient_tokens_path.exists():
        patient_tokens_path.unlink()

    _build_mediq_jsonl(cases, column, start_idx, diseases, mediq_data)
    print(f"[mediq] wrote {len(cases)} samples → {mediq_data}", flush=True)

    cmd = [
        str(_MEDIQ_REPO / ".venv/bin/python"),
        "mediQ_benchmark.py",
        "--expert_module", "expert",
        "--expert_class", "ScaleExpert",
        "--expert_model", doctor_model,
        "--expert_model_question_generator", doctor_model,
        "--patient_module", pat_module,
        "--patient_class", pat_class,
        "--patient_model", patient_model,
        "--use_api", "openai",
        "--api_account", "openai",
        "--data_dir", "../data",
        "--dev_filename", data_name,
        "--output_filename", str(output_jsonl),
        "--log_filename", str(out_log_dir / "run.log"),
        "--max_questions", str(row_kwargs.get("max_questions", 10)),
        "--abstain_threshold", str(row_kwargs.get("abstain_threshold", 4.0)),
        "--rationale_generation",
        "--self_consistency", str(row_kwargs.get("self_consistency", 3)),
        "--max_tokens", "256",
    ]
    env = os.environ.copy()
    env["MEDIQ_PATIENT_TOKENS_OUT"] = str(patient_tokens_path)

    print(f"[mediq] running mediQ_benchmark.py ({len(cases)} cases) ...", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_MEDIQ_REPO / "src"), env=env)
    print(f"[mediq] exit={proc.returncode}, elapsed {time.time()-t0:.0f}s", flush=True)
    if not output_jsonl.exists():
        sys.exit(f"ERROR: mediQ produced no output. See {out_log_dir/'run.log'}")

    patient_tokens = (json.loads(patient_tokens_path.read_text())
                      if patient_tokens_path.exists() else {})

    # Per-case JSON convert (matches our schema)
    rows = [json.loads(l) for l in open(output_jsonl)]
    case_by_id = {start_idx + i: c for i, c in enumerate(cases)}
    n_hits = 0
    for r in rows:
        idx = r["id"]
        case = case_by_id.get(idx)
        if case is None:
            continue
        gt = column.gt_of(case)
        chosen_letter = r["interactive_system"]["letter_choice"]
        chosen_disease = r["info"]["options"].get(chosen_letter, "?")
        hit = column.matches_gt(chosen_disease, gt)
        n_hits += int(hit)

        doctor_in = doctor_out = doctor_calls = 0
        for tia in r["interactive_system"].get("temp_additional_info", []):
            u = (tia or {}).get("usage") or {}
            doctor_in += int(u.get("input_tokens", 0) or 0)
            doctor_out += int(u.get("output_tokens", 0) or 0)
            if u:
                doctor_calls += 1

        hadm_id = _hadm_id(column.NAME, idx, case)
        p = patient_tokens.get(hadm_id, {})

        out_path = out_dir / f"case_{idx:03d}.json"
        with out_path.open("w") as f:
            json.dump({
                "case_idx": idx,
                "baseline": NAME,
                "column": column.NAME,
                "ground_truth": gt,
                "patient_actor_hidden": (
                    case.get("OSCE_Examination", case).get("Patient_Actor", {})
                ),
                "mediq_letter_choice": chosen_letter,
                "mediq_options_count": len(r["info"]["options"]),
                "questions": r["interactive_system"]["questions"],
                "answers": r["interactive_system"]["answers"],
                "intermediate_choices": r["interactive_system"]["intermediate_choices"],
                "final": {
                    "predicted": chosen_disease,
                    "top1_hit": hit,
                    "top3_hit": None,           # MediQ has no Top-3 (single MCQ)
                    "abstained": False,         # force-final default; aggregator splits modes
                    "turns": len(r["interactive_system"]["questions"]),
                },
                "token_usage": {
                    "doctor":  {"input_tokens": doctor_in,
                                 "output_tokens": doctor_out,
                                 "calls": doctor_calls},
                    "patient": {"input_tokens": int(p.get("input_tokens", 0) or 0),
                                 "output_tokens": int(p.get("output_tokens", 0) or 0),
                                 "calls": int(p.get("calls", 0) or 0)},
                },
            }, f, indent=2, ensure_ascii=False)

    print(f"\n[mediq] {n_hits}/{len(rows)} = {n_hits/max(len(rows),1):.0%} top-1 (force-final). "
          f"Per-case: {out_dir}", flush=True)
