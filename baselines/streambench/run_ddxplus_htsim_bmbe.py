"""DDxPlus HT-sim runner for BMBE."""

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


def _safe_name(x) -> str:
    return str(x) if x is not None else ""


def run_one(case, case_idx, doctor_factory, args):
    from baselines.streambench.adapters.base import LLMPatientWrapper
    from baselines.meddxagent import usage_tracker

    gt = case["PATHOLOGY"]
    patient_id = case.get("patient_id", f"htsim_{case_idx:04d}")

    # Reset patient token accumulator (MEDDxAgent OpenAIChat → DOCTOR slot,
    # which here is patient-only since BMBE doctor uses MoBayes's OpenAILLMClient).
    usage_tracker.reset()

    patient = LLMPatientWrapper(case, patient_model=args.patient_model)
    doctor = doctor_factory()
    doctor.reset({})

    if args.force_max_turns:
        from world_model import StoppingRules
        doctor.stopping = StoppingRules(
            confidence_threshold=1.01,
            max_questions=args.max_questions,
            min_questions_before_abstain=999,
        )

    dialog = []
    opening_q = ("Hello, what brings you in today? "
                 "Could you tell me about yourself and what's been going on?")
    opening_a = patient.respond(opening_q)
    dialog.append({"role": "doctor", "content": opening_q, "turn": 0})
    dialog.append({"role": "patient", "content": opening_a, "turn": 0})
    doctor.bulk_extract_from_text(opening_a)

    actual_stop = "MAX_QUESTIONS"
    for turn in range(args.max_questions):
        qr = doctor.get_next_question()
        if qr["status"] == "stopped":
            actual_stop = qr.get("reason", "STOPPED")
            break
        ans = patient.respond(qr["question"])
        doctor.process_response(qr["feature_id"], ans, qr["schema_values"])
        dialog.append({"role": "doctor", "content": qr["question"], "turn": turn + 1})
        dialog.append({"role": "patient", "content": ans, "turn": turn + 1})

    results = doctor.get_results()
    top1 = _safe_name(results.get("top_disease"))
    top_diseases = [_safe_name(d) for d in results.get("top_diseases", [])][:5]
    hit = _matches(top1, gt)
    in_top3 = any(_matches(d, gt) for d in top_diseases[:3])

    patient_usage_snap = usage_tracker.snapshot()["patient"]

    out = {
        "patient_id": patient_id,
        "ground_truth": gt,
        "predicted": top1,
        "predicted_set": top_diseases,
        "top1_hit": bool(hit),
        "top3_hit": bool(in_top3),
        "abstained": False,            # BMBE force_max_turns disables abstain
        "stop_reason": actual_stop,
        "turns": sum(1 for d in dialog if d["role"] == "doctor" and d["turn"] > 0),
        "doctor_mode": "bmbe_closed_world",
        "confidence": float(results.get("confidence", 0.0)),
        "final_entropy": float(results.get("entropy", 0.0)),
        "token_usage": {
            "doctor": dict(getattr(doctor, "token_usage",
                                    getattr(doctor.llm, "total_usage", {}))),
            "patient": {
                "prompt_tokens": patient_usage_snap.get("prompt_tokens", 0),
                "completion_tokens": patient_usage_snap.get("completion_tokens", 0),
                "calls": patient_usage_snap.get("calls", 0),
            },
        },
        "dialog_history": dialog,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-cases", type=int, default=50)
    ap.add_argument("--start-idx", type=int, default=0,
                    help="Skip first N cases (for resuming after partial sweep)")
    ap.add_argument("--max-questions", type=int, default=15)
    ap.add_argument("--patient-model", default="gpt-5.4-nano")
    ap.add_argument("--doctor-model", default="gpt-5.4-nano")
    ap.add_argument("--kb-dir", default="data/kbs/ddxplus")
    ap.add_argument("--cases-path", default="data/cases/ddxplus_htsim_a/selected_cases.jsonl")
    ap.add_argument("--out", default="data/benchmark_results/set_a/ddxplus_htsim/mobayes/bmbe_gpt54nano.jsonl")
    ap.add_argument("--force-max-turns", action="store_true", default=True)
    ap.add_argument("--allow-early-stop", dest="force_max_turns", action="store_false")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from baselines.streambench.adapters.base import load_selected_cases
    from scripts.core.benchmark_patientsim import MoBayesDoctor
    from config import MoBayesConfig

    all_cases = load_selected_cases(REPO / args.cases_path)
    cases = all_cases[args.start_idx : args.start_idx + args.num_cases]
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] BMBE | {len(cases)} cases | KB={args.kb_dir} | "
          f"max_q={args.max_questions} | doctor={args.doctor_model} (HT sim patient)")
    print(f"[out] {out_path}")

    config = MoBayesConfig()
    data_dir = str(REPO / args.kb_dir)
    api_key = os.environ["OPENAI_API_KEY"]

    def doctor_factory():
        return MoBayesDoctor(
            data_dir=data_dir,
            llm_model=args.doctor_model,
            api_key=api_key,
            max_questions=args.max_questions,
                        prior_mode="uniform",
            config=config,
        )

    out_f = out_path.open("w")
    rows = []
    t0 = time.time()
    for local_idx, case in enumerate(cases):
        idx = args.start_idx + local_idx
        ts = time.time()
        try:
            r = run_one(case, idx, doctor_factory, args)
            r["case_idx"] = idx
            r["elapsed_seconds"] = time.time() - ts
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n"); out_f.flush()
            tag = "HIT " if r["top1_hit"] else "miss"
            print(f"  [{local_idx+1}/{len(cases)} | abs#{idx}] {tag} GT={r['ground_truth'][:40]:<40} → "
                  f"{r['predicted'][:35]:<35} (conf={r['confidence']:.2f}, "
                  f"{r['elapsed_seconds']:.0f}s)", flush=True)
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
        d_in = sum(r["token_usage"]["doctor"].get("prompt_tokens", 0) for r in valid)
        d_out = sum(r["token_usage"]["doctor"].get("completion_tokens", 0) for r in valid)
        p_in = sum(r["token_usage"]["patient"].get("prompt_tokens", 0) for r in valid)
        p_out = sum(r["token_usage"]["patient"].get("completion_tokens", 0) for r in valid)
        print()
        print(f"  BMBE on DDxPlus HT-sim {len(valid)}/{len(cases)} cases")
        print(f"    Top-1: {top1}/{len(valid)} = {top1/len(valid):.0%}")
        print(f"    Top-3: {top3}/{len(valid)} = {top3/len(valid):.0%}")
        print(f"    Tokens — doctor in/out={d_in}/{d_out}, patient in/out={p_in}/{p_out}")
        print(f"    Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
