"""DDxPlus HT-sim runner for prompt baselines (cw, cot, savage_dd, savage_bay) and AgentClinic native (ac_native)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _matches(predicted, ground_truth) -> bool:
    p = str(predicted) if predicted is not None else ""
    g = str(ground_truth) if ground_truth is not None else ""
    if not p or not g:
        return False
    return p.strip().lower() == g.strip().lower()


class _HTSimScenario:
    """Minimal AgentClinic-compatible scenario built from a DDxPlus case.

    AC's DoctorAgent reads only `examiner_information()` (stored in
    `self.presentation`). HT-sim case has chief complaint + age/sex; that's
    enough for a doctor briefing. Anything AC's MeasurementAgent needs
    (tests, physical exams) is irrelevant — DoctorAgent doesn't touch them.
    """

    def __init__(self, case: dict):
        chief = case.get("INITIAL_EVIDENCE_ENG", "") or "Patient presents with unspecified concerns."
        age = case.get("AGE", "?")
        sex = case.get("SEX", "?")
        self._examiner = (
            f"A {age}-year-old {sex} patient presents to the clinic. "
            f"Chief complaint: {chief}."
        )

    def examiner_information(self) -> str:
        return self._examiner


def build_doctor(baseline: str, disease_names, doctor_model: str, api_key: str,
                 max_turns: int, case: dict | None = None):
    from scripts.core.benchmark_patientsim import StandaloneDoctorAgent
    from config import MoBayesConfig
    cfg = MoBayesConfig()
    if baseline == "sa":
        return StandaloneDoctorAgent(model=doctor_model, api_key=api_key,
                                      max_turns=max_turns, config=cfg)
    if baseline == "cw":
        from baselines.closed_world_doctor import ClosedWorldDoctorAgent
        return ClosedWorldDoctorAgent(disease_names=disease_names, model=doctor_model,
                                       api_key=api_key, config=cfg)
    if baseline == "cot":
        from baselines.cot_doctor import ChainOfThoughtDoctorAgent
        return ChainOfThoughtDoctorAgent(disease_names=disease_names, model=doctor_model,
                                          api_key=api_key, config=cfg)
    if baseline == "savage_dd":
        from baselines.savage_doctor import DifferentialDiagnosisDoctorAgent
        return DifferentialDiagnosisDoctorAgent(disease_names=disease_names, model=doctor_model,
                                                 api_key=api_key, config=cfg)
    if baseline == "savage_bay":
        from baselines.savage_doctor import BayesianReasoningDoctorAgent
        return BayesianReasoningDoctorAgent(disease_names=disease_names, model=doctor_model,
                                             api_key=api_key, config=cfg)
    if baseline == "ac_native":
        from baselines.agentclinic.ac_native_doctor import ACNativeDoctorAgent
        if case is None:
            raise ValueError("ac_native baseline requires `case` argument")
        return ACNativeDoctorAgent(scenario=_HTSimScenario(case),
                                    disease_names=disease_names,
                                    model=doctor_model, api_key=api_key,
                                    max_turns=max_turns)
    raise ValueError(f"Unknown baseline: {baseline}")


def run_one(case, case_idx, baseline, doctor_model, patient_model, max_turns,
             disease_names, api_key):
    from baselines.streambench.adapters.base import LLMPatientWrapper
    from baselines.meddxagent import usage_tracker

    gt = case["PATHOLOGY"]
    patient_id = case.get("patient_id", f"htsim_{case_idx:04d}")

    usage_tracker.reset()
    patient = LLMPatientWrapper(case, patient_model=patient_model)
    doctor = build_doctor(baseline, disease_names, doctor_model, api_key, max_turns, case=case)

    dialog = []
    greeting = "Hello, I'm your doctor today. What brings you in today?"
    pa = patient.respond(greeting)
    dialog.append({"role": "doctor", "content": greeting, "turn": 0})
    dialog.append({"role": "patient", "content": pa, "turn": 0})

    stop_reason = "MAX_TURNS"
    last_doctor_msg = ""
    final_via_callback = False
    # AC native paper-faithful behaviour (agentclinic.py:630-631 / our
    # run_ac_native.py:177): prepend a "this is your final question, please
    # provide a diagnosis" hint to the patient utterance the doctor will see
    # on its very last turn, so AC has a chance to commit instead of
    # silently maxing out.
    ac_total_inferences = max_turns if baseline == "ac_native" else None
    for turn in range(max_turns + 5):
        if ac_total_inferences is not None and turn == ac_total_inferences - 1:
            pa = (pa or "") + "\n\nThis is the final question. Please provide a diagnosis."
        question, is_final = doctor.get_question(pa)
        dialog.append({"role": "doctor", "content": question, "turn": turn + 1})
        last_doctor_msg = question
        if is_final:
            stop_reason = "DDX_PROVIDED"
            final_via_callback = True
            break
        pa = patient.respond(question)
        dialog.append({"role": "patient", "content": pa, "turn": turn + 1})

    abstained = "[ABSTAIN]" in last_doctor_msg
    if final_via_callback:
        # ac_native commits with DIAGNOSIS READY (no [DDX]/[ABSTAIN] markers).
        preds = doctor.parse_ddx(last_doctor_msg)
        stop_reason = "DDX_PROVIDED"
    elif "[DDX]" in last_doctor_msg:
        preds = doctor.parse_ddx(last_doctor_msg)
        stop_reason = "DDX_PROVIDED"
    elif abstained:
        preds = doctor.parse_possible(last_doctor_msg)
        stop_reason = "ABSTAINED"
    else:
        ddx_msg, _ = doctor.get_question("Please provide your final [DDX] or [ABSTAIN] now.")
        dialog.append({"role": "doctor", "content": ddx_msg, "turn": -1})
        if "[DDX]" in ddx_msg:
            preds = doctor.parse_ddx(ddx_msg); stop_reason = "DDX_PROVIDED"
        elif "DIAGNOSIS READY" in ddx_msg.upper():
            preds = doctor.parse_ddx(ddx_msg); stop_reason = "DDX_PROVIDED"
        else:
            abstained = True
            preds = doctor.parse_possible(ddx_msg); stop_reason = "ABSTAINED"

    top1 = preds[0] if preds else "Unknown"
    hit = _matches(top1, gt)
    in_top3 = any(_matches(d, gt) for d in (preds or [])[:3])

    patient_usage_snap = usage_tracker.snapshot()["patient"]

    return {
        "patient_id": patient_id,
        "ground_truth": gt,
        "predicted": "ABSTAIN" if abstained else top1,
        "predicted_set": list(preds[:5]),
        "top1_hit": bool(hit),
        "top3_hit": bool(in_top3),
        "abstained": bool(abstained),
        "stop_reason": stop_reason,
        "turns": sum(1 for d in dialog if d["role"] == "doctor" and d["turn"] > 0),
        "doctor_mode": baseline,
        "token_usage": {
            "doctor": dict(getattr(doctor, "token_usage", {})),
            "patient": {
                "prompt_tokens": patient_usage_snap.get("prompt_tokens", 0),
                "completion_tokens": patient_usage_snap.get("completion_tokens", 0),
                "calls": patient_usage_snap.get("calls", 0),
            },
        },
        "dialog_history": dialog,
    }


def _ac_llm_judge(gt: str, dialog, model: str = "gpt-5.4-nano") -> tuple[bool, str]:
    """AgentClinic-style LLM-judge match: ask gpt 'are these the same disease? Yes/No'.

    Returns (judged_hit, free_text_dx_seen). Mirrors agentclinic.py:565-567.
    """
    import re
    from openai import OpenAI
    free_dx = ""
    for d in reversed(dialog):
        if d.get("role") != "doctor":
            continue
        m = re.search(r"DIAGNOSIS READY:\s*(.+?)(?:\n|$)", d.get("content", ""), flags=re.I)
        if m:
            free_dx = m.group(1).strip(); break
    if not free_dx or free_dx.lower() in ("abstain", "unknown"):
        return False, free_dx
    client = OpenAI()
    is_reasoning = model.startswith(("gpt-5", "o"))
    kwargs = {"model": model, "messages": [
        {"role": "system", "content": (
            "You are responsible for determining if the correct diagnosis "
            "and the doctor diagnosis are the same disease. Please respond "
            "only with Yes or No. Nothing else.")},
        {"role": "user", "content": (
            f"\nHere is the correct diagnosis: {gt}"
            f"\nHere is the doctor's diagnosis: {free_dx}"
            f"\nAre these the same disease? Please respond only with Yes or No.")},
    ]}
    if is_reasoning:
        kwargs["max_completion_tokens"] = 10
    else:
        kwargs["temperature"] = 0.0; kwargs["max_tokens"] = 10
    try:
        resp = client.chat.completions.create(**kwargs)
        out = (resp.choices[0].message.content or "").strip().lower()
        return out.startswith("yes"), free_dx
    except Exception:
        return False, free_dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True,
                    choices=["sa", "cw", "cot", "savage_dd", "savage_bay", "ac_native"])
    ap.add_argument("--num-cases", type=int, default=50)
    ap.add_argument("--start-idx", type=int, default=0,
                    help="Skip first N cases (for resuming after partial sweep)")
    ap.add_argument("--max-questions", type=int, default=15)
    ap.add_argument("--llm-judge", action="store_true",
                    help="Add per-case AgentClinic-style LLM-judge match (mirrors compare_results); "
                         "writes top1_hit_judged + judged_dx_text fields. Mostly meaningful for ac_native.")
    ap.add_argument("--patient-model", default="gpt-5.4-nano")
    ap.add_argument("--doctor-model", default="gpt-5.4-nano")
    ap.add_argument("--cases-path", default="data/cases/ddxplus_htsim_a/selected_cases.jsonl")
    ap.add_argument("--out", default=None,
                    help="If unset: data/benchmark_results/set_a/ddxplus_htsim/<baseline>_gpt54nano.jsonl")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr); sys.exit(1)

    from baselines.streambench.adapters.base import load_selected_cases
    import pandas as pd

    diseases_csv = REPO / "data/kbs/ddxplus/diseases.csv"
    disease_names = pd.read_csv(diseases_csv)["name"].tolist()
    print(f"[setup] {len(disease_names)} disease names from DDxPlus KB")

    all_cases = load_selected_cases(REPO / args.cases_path)
    cases = all_cases[args.start_idx : args.start_idx + args.num_cases]
    out_path = (Path(args.out) if args.out else
                REPO / f"data/benchmark_results/set_a/ddxplus_htsim/{args.baseline}_gpt54nano.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {args.baseline.upper()} | {len(cases)} cases | doctor={args.doctor_model} (HT sim patient)")
    print(f"[out] {out_path}")

    api_key = os.environ["OPENAI_API_KEY"]
    out_f = out_path.open("w")
    rows = []
    t0 = time.time()
    for local_idx, case in enumerate(cases):
        idx = args.start_idx + local_idx
        ts = time.time()
        try:
            r = run_one(case, idx, args.baseline, args.doctor_model, args.patient_model,
                         args.max_questions, disease_names, api_key)
            r["case_idx"] = idx
            r["elapsed_seconds"] = time.time() - ts
            if args.llm_judge:
                judged_hit, judged_dx = _ac_llm_judge(r["ground_truth"], r["dialog_history"])
                r["top1_hit_judged"] = bool(judged_hit)
                r["judged_dx_text"] = judged_dx
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n"); out_f.flush()
            tag = ("HIT " if r["top1_hit"] else
                   ("ABS " if r["abstained"] else "miss"))
            print(f"  [{local_idx+1}/{len(cases)} | abs#{idx}] {tag} GT={r['ground_truth'][:35]:<35} → "
                  f"{r['predicted'][:35]:<35} ({r['turns']} turns, {r['elapsed_seconds']:.0f}s)",
                  flush=True)
            rows.append(r)
        except Exception as e:
            err = {"case_idx": idx, "patient_id": case.get("patient_id"),
                   "ground_truth": case.get("PATHOLOGY"), "error": str(e)}
            out_f.write(json.dumps(err) + "\n"); out_f.flush()
            print(f"  [{local_idx+1}/{len(cases)} | abs#{idx}] ERROR: {e}", flush=True)
    out_f.close()

    valid = [r for r in rows if "error" not in r]
    if valid:
        top1 = sum(r["top1_hit"] for r in valid)
        top3 = sum(r["top3_hit"] for r in valid)
        abs_n = sum(r["abstained"] for r in valid)
        d_in = sum(r["token_usage"]["doctor"].get("prompt_tokens", 0) for r in valid)
        d_out = sum(r["token_usage"]["doctor"].get("completion_tokens", 0) for r in valid)
        p_in = sum(r["token_usage"]["patient"].get("prompt_tokens", 0) for r in valid)
        p_out = sum(r["token_usage"]["patient"].get("completion_tokens", 0) for r in valid)
        print()
        print(f"  {args.baseline.upper()} on DDxPlus HT-sim {len(valid)}/{len(cases)} cases")
        print(f"    Top-1 (lenient): {top1}/{len(valid)} = {top1/len(valid):.0%}")
        print(f"    Top-3 (lenient): {top3}/{len(valid)} = {top3/len(valid):.0%}")
        print(f"    Abstain        : {abs_n}/{len(valid)} = {abs_n/len(valid):.0%}")
        print(f"    Tokens — doctor in/out={d_in}/{d_out}, patient in/out={p_in}/{p_out}")
        print(f"    Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
