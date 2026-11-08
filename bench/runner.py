"""Generic per-case loop shared by every (row, column).

Replaces the per-baseline batch_smoke_*.py / run_*.py runners. Doctor
output schema, token tracking, abstention parsing, and per-case JSON
layout all live here so every row × column writes the same shape.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from bench import columns, rows


def _git_sha(repo_path: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _write_manifest(out_dir: Path, **fields: Any) -> None:
    """Persist run config (max_turns, models, row_kwargs, git SHA, ...) next
    to the case_*.json files so reproductions don't need the launch command."""
    manifest = {
        "started_at": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "git_sha": _git_sha(Path(__file__).resolve().parents[1]),
        **fields,
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)


GREETING = "Hello, I'm your doctor today. What brings you in today?"
FORCE_FINAL = "Please provide your final [DDX] or [ABSTAIN] now."
_REMAP_MODEL = "gpt-5.4-nano"


def _normalize_to_closed_set(
    predictions: List[str],
    disease_universe: List[str],
    api_key: str,
) -> List[str]:
    """Map free-text doctor predictions to the column's closed disease set.

    Uses a single LLM call identical to the prompt in
    ``llm_interface.prompt_templates.PromptTemplates.render_disease_matching_prompt``
    so that scoring is consistent with the original PatientSim pipeline
    (``scripts.core.benchmark_patientsim.normalize_standalone_predictions``).
    """
    if not predictions or not disease_universe:
        return predictions
    try:
        from openai import OpenAI
        from llm_interface.prompt_templates import PromptTemplates
        oai_key = os.environ.get("OPENAI_API_KEY") or api_key
        client = OpenAI(api_key=oai_key)
        prompt = PromptTemplates.render_disease_matching_prompt(
            predictions[:5], disease_universe)
        resp = client.chat.completions.create(
            model=_REMAP_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        name_lookup = {n.lower(): n for n in disease_universe}
        matched = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(".", 1)
            name = parts[1].strip() if len(parts) == 2 else line.strip()
            if name.upper() == "NO_MATCH":
                idx = len(matched)
                matched.append(predictions[idx] if idx < len(predictions) else "Unknown")
            elif name.lower() in name_lookup:
                matched.append(name_lookup[name.lower()])
            else:
                idx = len(matched)
                matched.append(predictions[idx] if idx < len(predictions) else name)
        while len(matched) < len(predictions):
            matched.append(predictions[len(matched)])
        return matched[: len(predictions)]
    except Exception:
        return predictions


def _doctor_tokens(doctor) -> Dict[str, Any]:
    """Most doctors expose tokens via doctor.llm.total_usage; some
    (e.g. AC native) via doctor.token_usage directly."""
    return (
        getattr(getattr(doctor, "llm", None), "total_usage", None)
        or getattr(doctor, "token_usage", None)
        or {}
    )


def run_one_case(
    case: Dict[str, Any],
    case_idx: int,
    *,
    column,
    row,
    doctor_model: str,
    patient_model: str,
    max_turns: int,
    api_key: str,
    out_dir: Path,
    row_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """One full dialog session. Writes <out_dir>/case_NNN.json and returns log.

    Rows with a non-standard dialog flow (e.g. ac_native's MeasurementAgent +
    moderator, bmbe's Bayesian engine + EIG + parsing patch) can define their
    own ``run_one_case`` and override the default loop entirely.
    """
    if hasattr(row, "run_one_case"):
        return row.run_one_case(
            case, case_idx,
            column=column,
            doctor_model=doctor_model,
            patient_model=patient_model,
            max_turns=max_turns,
            api_key=api_key,
            out_dir=out_dir,
            row_kwargs=row_kwargs or {},
        )

    gt = column.gt_of(case)
    osce = case.get("OSCE_Examination", case)
    patient = column.make_patient(case, patient_model)

    # Some rows (AC native) need the patient's underlying scenario object.
    extra = dict(row_kwargs or {})
    scenario = getattr(patient, "_scenario", None)
    if scenario is not None:
        extra.setdefault("scenario", scenario)

    doctor = row.make_doctor(
        disease_universe=column.disease_universe(),
        doctor_model=doctor_model,
        api_key=api_key,
        max_turns=max_turns,
        column_name=column.NAME,
        **extra,
    )

    log: Dict[str, Any] = {
        "case_idx": case_idx,
        "baseline": row.NAME,
        "column": column.NAME,
        "ground_truth": gt,
        "patient_actor_hidden": osce.get("Patient_Actor", {}),
        "objective_for_doctor": getattr(patient, "initial_info", ""),
        "turns": [],
        "final": {},
    }

    pa = patient.respond(GREETING)
    log["turns"].append({"role": "doctor", "content": GREETING, "turn": 0})
    log["turns"].append({"role": "patient", "content": pa, "turn": 0})

    stop_reason = "MAX_TURNS"
    last_doctor_msg = ""
    for turn in range(max_turns + 5):
        question, is_final = doctor.get_question(pa)
        log["turns"].append({"role": "doctor", "content": question, "turn": turn + 1})
        last_doctor_msg = question
        if is_final:
            stop_reason = "DDX_PROVIDED"
            break
        pa = patient.respond(question)
        log["turns"].append({"role": "patient", "content": pa, "turn": turn + 1})

    abstained = "[ABSTAIN]" in last_doctor_msg
    if "[DDX]" in last_doctor_msg:
        preds = doctor.parse_ddx(last_doctor_msg)
        stop_reason = "DDX_PROVIDED"
    elif abstained:
        preds = doctor.parse_possible(last_doctor_msg)
        stop_reason = "ABSTAINED"
    else:
        ddx_msg, _ = doctor.get_question(FORCE_FINAL)
        log["turns"].append({"role": "doctor", "content": ddx_msg, "turn": -1})
        if "[DDX]" in ddx_msg:
            preds = doctor.parse_ddx(ddx_msg)
            stop_reason = "DDX_PROVIDED"
        else:
            abstained = True
            preds = doctor.parse_possible(ddx_msg)
            stop_reason = "ABSTAINED"

    # Normalize free-text predictions to the column's closed disease set
    # (mirrors the PatientSim pipeline's normalize_standalone_predictions).
    raw_preds = list(preds)
    disease_univ = column.disease_universe()
    if preds and disease_univ:
        preds = _normalize_to_closed_set(preds, disease_univ, api_key)

    top1 = preds[0] if preds else "Unknown"
    raw_hit = column.matches_gt(top1, gt)
    raw_in_top3 = any(column.matches_gt(d, gt) for d in (preds or [])[:3])
    hit = bool(raw_hit)
    in_top3 = bool(raw_in_top3)

    log["final"] = {
        "predicted": "ABSTAIN" if abstained else top1,
        "predicted_set": list(preds[:5]),
        "predicted_raw": list(raw_preds[:5]),
        "top1_hit": hit,
        "top3_hit": in_top3,
        "raw_top1_hit": bool(raw_hit),
        "raw_top3_hit": bool(raw_in_top3),
        "abstained": bool(abstained),
        "stop_reason": stop_reason,
        "turns": sum(
            1 for t in log["turns"] if t["role"] == "doctor" and t.get("turn", 0) > 0
        ),
    }
    log["token_usage"] = {
        "doctor": dict(_doctor_tokens(doctor)),
        "patient": getattr(patient, "token_usage", {}).copy() or {},
    }

    out_path = out_dir / f"case_{case_idx:03d}.json"
    with out_path.open("w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    return log


def run(
    *,
    column_name: str,
    row_name: str,
    num_cases: int = 50,
    start_idx: int = 0,
    max_turns: int = 15,
    doctor_model: str = "gpt-5.4-nano",
    patient_model: str = "gpt-5.4-nano",
    out_suffix: str = "",
    row_kwargs: Dict[str, Any] | None = None,
    doctor_base_url: str | None = None,
    doctor_api_key: str | None = None,
) -> None:
    """End-to-end driver: load cases, build per-row doctor, loop, save.

    Dispatch order (rows can define any one of):
      1. ``run_batch(...)``     — full-column subprocess (mediq, meddxagent, cod)
      2. ``run_one_case(...)``  — custom per-case loop (ac_native, bmbe)
      3. ``make_doctor(...)``   — generic per-case loop (cw, cot, savage)
    """
    api_key = doctor_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set and --doctor-api-key not provided")

    # Inject base_url into row_kwargs so both make_doctor and run_one_case
    # receive it via **kwargs / row_kwargs dict.
    row_kwargs = dict(row_kwargs or {})
    if doctor_base_url:
        row_kwargs.setdefault("base_url", doctor_base_url)

    column = columns.get(column_name)
    row = rows.get(row_name)

    all_cases = column.load_cases()
    cases = all_cases[start_idx:num_cases]
    out_dir = column.output_dir(row_name, out_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        out_dir,
        row=row.NAME,
        column=column.NAME,
        out_dir_name=out_dir.name,
        n_cases=len(cases),
        start_idx=start_idx,
        num_cases=num_cases,
        max_turns=max_turns,
        doctor_model=doctor_model,
        patient_model=patient_model,
        doctor_base_url=doctor_base_url,
        out_suffix=out_suffix,
        row_kwargs=row_kwargs,
    )
    print(
        f"[run] {column.NAME} × {row.NAME} | "
        f"cases [{start_idx},{num_cases}) ({len(cases)}) | "
        f"max_turns={max_turns} | doctor={doctor_model} | out={out_dir.name}",
        flush=True,
    )

    # Dispatch 1: full-column batch (subprocess / remote rows)
    if hasattr(row, "run_batch"):
        row.run_batch(
            cases=cases, start_idx=start_idx,
            column=column,
            doctor_model=doctor_model, patient_model=patient_model,
            max_turns=max_turns, api_key=api_key, out_dir=out_dir,
            row_kwargs=row_kwargs or {},
        )
        return

    # Dispatch 2 / 3: per-case loop (overridden or generic)
    t0 = time.time()
    for offset, case in enumerate(cases):
        idx = start_idx + offset
        ts = time.time()
        try:
            log = run_one_case(
                case, idx,
                column=column, row=row,
                doctor_model=doctor_model, patient_model=patient_model,
                max_turns=max_turns, api_key=api_key, out_dir=out_dir,
                row_kwargs=row_kwargs,
            )
            f = log["final"]
            tag = "HIT" if f.get("top1_hit") else ("ABS" if f.get("abstained") else "miss")
            n_turns = f.get("turns") or sum(
                1 for t in log.get("turns", []) if t.get("kind") == "qa"
            )
            print(
                f"  [{offset+1}/{len(cases)}] (idx={idx}) {tag:>4} "
                f"GT={log['ground_truth'][:30]:<30} → "
                f"{str(f.get('predicted', '?'))[:30]:<30} "
                f"({n_turns} turns, {time.time()-ts:.0f}s)",
                flush=True,
            )
        except Exception as e:  # one bad case shouldn't kill the run
            print(f"  [{offset+1}/{len(cases)}] (idx={idx}) ERROR: {e}", flush=True)

    print(f"\n[done] {len(cases)} cases in {time.time()-t0:.0f}s | {out_dir}")
