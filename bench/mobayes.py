"""BMBE row — Bayesian engine + EIG question selection.

Different dialog protocol from `[DDX]` baselines: the engine drives
question selection (not the LLM), and per-turn we record posterior
trajectory + entropy for post-hoc τ-sweep.

Defaults (paper protocol):
  - force_max_turns=True  (no early commit; full max-questions budget)
  - lenient parsing prompts patched in for AC (AC patient hedges)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


NAME = "bmbe"


def _kb_dir(column_name: str) -> str:
    """Pick the KB directory for the given column."""
    if column_name == "ac":
        return "data/kbs/agentclinic_a"
    if column_name == "ac_set2":
        return "data/kbs/agentclinic_b"
    if column_name in ("patientsim", "patientsim_v1", "htsim", "htsim_v1"):
        return "data/kbs/ddxplus"
    raise ValueError(f"BMBE: no KB for column={column_name!r}")


def make_doctor(*args, **kwargs):
    raise NotImplementedError("bmbe uses run_one_case override; no make_doctor.")


def run_one_case(
    case: Dict[str, Any],
    case_idx: int,
    *,
    column,
    doctor_model: str,
    patient_model: str,
    max_turns: int,
    api_key: str,
    out_dir: Path,
    row_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    from scripts.core.benchmark_patientsim import MoBayesDoctor
    from config import MoBayesConfig
    from world_model import StoppingRules

    row_kwargs = row_kwargs or {}
    force_max_turns = row_kwargs.get("force_max_turns", True)

    # AC column hedges in patient utterances → patch parser prompts.
    if column.NAME in ("ac", "ac_set2"):
        from llm_interface.prompt_templates import PromptTemplates
        from llm_interface.lenient_parsing_prompts import (
            render_parsing_prompt_lenient, render_bulk_intake_prompt_lenient,
        )
        PromptTemplates.render_parsing_prompt = staticmethod(render_parsing_prompt_lenient)
        PromptTemplates.render_bulk_intake_prompt = staticmethod(render_bulk_intake_prompt_lenient)

    gt = column.gt_of(case)
    osce = case.get("OSCE_Examination", case)
    patient = column.make_patient(case, patient_model)

    repo = Path(__file__).resolve().parents[1]
    data_dir = str(repo / _kb_dir(column.NAME))

    # Posterior-aware EIG (off by default in MoBayesDoctor):
    #   topk_weight=0    → pure global EIG (selects features informative across all 31 diseases)
    #   topk_weight>0    → blend in focused EIG (informative among current top-K candidates)
    #   focus_conf_threshold (0..1) gate: only activate focused mode once
    #     max(posterior) ≥ threshold (so we don't over-focus too early)
    eig_topk_weight = float(row_kwargs.get("eig_topk_weight", 0.0))
    eig_focus_conf_threshold = float(row_kwargs.get("eig_focus_conf_threshold", 1.01))
    eig_topk_size = int(row_kwargs.get("eig_topk_size", 3))

    doctor = MoBayesDoctor(
        data_dir=data_dir,
        llm_model=doctor_model,
        api_key=api_key,
        base_url=row_kwargs.get("base_url"),
        max_questions=max_turns,
        prior_mode="uniform",
        eig_topk_size=eig_topk_size,
        eig_topk_weight=eig_topk_weight,
        eig_focus_conf_threshold=eig_focus_conf_threshold,
        config=MoBayesConfig(),
    )
    doctor.reset({})
    if force_max_turns:
        doctor.stopping = StoppingRules(
            confidence_threshold=1.01,             # never trigger early commit
            max_questions=max_turns,
            min_questions_before_abstain=999,      # disable abstain gate
        )

    log: Dict[str, Any] = {
        "case_idx": case_idx,
        "baseline": NAME,
        "column": column.NAME,
        "ground_truth": gt,
        "patient_actor_hidden": osce.get("Patient_Actor", {}),
        "objective_for_doctor": getattr(patient, "initial_info", ""),
        "kb_dir": _kb_dir(column.NAME),
        "turns": [],
        "final": {},
    }

    opening_q = ("Hello, what brings you in today? Could you tell me about "
                 "yourself and what's been going on?")
    opening_a = patient.respond(opening_q)
    bulk = doctor.bulk_extract_from_text(opening_a)

    posterior = doctor.wm.posterior()
    top5 = posterior.argsort()[::-1][:5]
    top5_named = [
        (doctor.disease_id_to_name(doctor.leaf_diseases[i]), float(posterior[i]))
        for i in top5
    ]
    log["turns"].append({
        "turn": 0, "kind": "opening",
        "doctor_opening": opening_q, "patient_opening": opening_a,
        "extracted_features": bulk, "top5_posterior": top5_named,
    })

    actual_stop = "MAX_QUESTIONS"
    for turn in range(max_turns):
        qr = doctor.get_next_question()
        if qr["status"] == "stopped":
            actual_stop = qr.get("reason", "STOPPED")
            log["stop_reason"] = actual_stop
            break
        ans = patient.respond(qr["question"])
        extracted, conf = doctor.process_response(
            qr["feature_id"], ans, qr["schema_values"],
        )
        posterior = doctor.wm.posterior()
        top3 = posterior.argsort()[::-1][:3]
        top3_named = [
            (doctor.disease_id_to_name(doctor.leaf_diseases[i]), float(posterior[i]))
            for i in top3
        ]
        log["turns"].append({
            "turn": turn + 1, "kind": "qa",
            "feature_id": qr["feature_id"],
            "doctor_question": qr["question"],
            "patient_answer": ans,
            "extracted_value": extracted,
            "extraction_confidence": float(conf),
            "top3_posterior": top3_named,
            "entropy": float(doctor.wm.entropy()),
        })

    results = doctor.get_results()
    hit = column.matches_gt(results["top_disease"], gt)
    in_top3 = any(column.matches_gt(d, gt) for d in results["top_diseases"][:3])
    log["final"] = {
        "predicted": results["top_disease"],
        "predicted_top3": list(results["top_diseases"][:3]),
        "confidence": float(results["confidence"]),
        "final_entropy": float(results["entropy"]),
        "top1_hit": bool(hit),
        "top3_hit": bool(in_top3),
        "stop_reason": log.get("stop_reason", actual_stop),
    }
    log["token_usage"] = {
        "doctor": getattr(getattr(doctor, "llm", None), "total_usage", {}).copy() or {},
        "patient": getattr(patient, "token_usage", {}).copy() or {},
    }

    out_path = out_dir / f"case_{case_idx:03d}.json"
    with out_path.open("w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    return log
