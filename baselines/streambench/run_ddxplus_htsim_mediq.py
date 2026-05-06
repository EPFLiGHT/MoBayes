"""DDxPlus HT-sim runner for MediQ-Expert (subprocess into the mediQ repo)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Set MEDIQ_REPO to a local clone of https://github.com/stellalisy/mediQ
MEDIQ_REPO = Path(os.environ.get("MEDIQ_REPO", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-cases", type=int, default=50)
    ap.add_argument("--start-idx", type=int, default=0,
                    help="Skip first N cases (resume after partial sweep)")
    ap.add_argument("--max-questions", type=int, default=10)
    ap.add_argument("--patient-model", default="gpt-5.4-nano")
    ap.add_argument("--doctor-model", default="gpt-5.4-nano")
    ap.add_argument("--cases-path", default="data/cases/ddxplus_htsim_a/selected_cases.jsonl")
    ap.add_argument("--out", default="data/benchmark_results/set_a/ddxplus_htsim/baselines/mediq_expert_gpt54nano.jsonl")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr); sys.exit(1)

    sys.path.insert(0, str(REPO))

    cases_path = REPO / args.cases_path
    all_cases = [json.loads(l) for l in open(cases_path)]
    cases = all_cases[args.start_idx : args.start_idx + args.num_cases]
    pathologies = sorted(set(c["PATHOLOGY"] for c in all_cases))
    letters = ([chr(ord("A") + c) for c in range(26)]
               + ["A" + chr(ord("A") + c) for c in range(26)])
    options = {letters[i]: d for i, d in enumerate(pathologies)}
    print(f"[setup] {len(options)} MCQ options (DDxPlus pathologies in test split).")

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mediq_data = MEDIQ_REPO / "data" / "ddxplus_htsim_full_mediq.jsonl"
    with mediq_data.open("w") as f:
        for case in cases:
            gt = case["PATHOLOGY"]
            answer_idx = next(
                (l for l, d in options.items() if d.lower().strip() == gt.lower().strip()),
                None,
            )
            if answer_idx is None:
                raise RuntimeError(f"GT {gt!r} not in options")
            chief = case.get("INITIAL_EVIDENCE_ENG", "")
            demo = f"Age {case.get('AGE', '?')}, Sex {case.get('SEX', '?')}"
            sample = {
                "id": case["_test_index"],
                "hadm_id": case["patient_id"],
                "question": "What is the most likely diagnosis?",
                "context": [demo, f"Chief complaint: {chief}"],
                "context_len": 2,
                "initial_info": f"{demo}. Chief complaint: {chief}.",
                "options": options,
                "answer": gt,
                "answer_idx": answer_idx,
                "explanation": "",
                "facts": [],
                "patient": {"gender": case.get("SEX", ""), "age": str(case.get("AGE", "")),
                             "specialties": [], "subspecialties": []},
            }
            f.write(json.dumps(sample) + "\n")
    print(f"[setup] Wrote {len(cases)} MediQ samples → {mediq_data}")

    out_log_dir = MEDIQ_REPO / "logs" / "ddxplus_htsim_full"
    out_log_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = out_log_dir / "results.jsonl"
    if output_jsonl.exists():
        output_jsonl.unlink()

    cmd = [
        str(MEDIQ_REPO / ".venv/bin/python"),
        "mediQ_benchmark.py",
        "--expert_module", "expert", "--expert_class", "ScaleExpert",
        "--expert_model", args.doctor_model,
        "--expert_model_question_generator", args.doctor_model,
        "--patient_module", "patient_ddxplus_htsim",
        "--patient_class", "DDxHTSimMediQPatient",
        "--patient_model", args.patient_model,
        "--use_api", "openai", "--api_account", "openai",
        "--data_dir", "../data",
        "--dev_filename", "ddxplus_htsim_full_mediq.jsonl",
        "--output_filename", str(output_jsonl),
        "--log_filename", str(out_log_dir / "run.log"),
        "--max_questions", str(args.max_questions),
        "--abstain_threshold", "4.0",
        "--rationale_generation",
        "--self_consistency", "3",
        "--max_tokens", "256",
    ]
    print(f"[run] cd {MEDIQ_REPO}/src && mediQ_benchmark.py ... ({len(cases)} cases)")
    t0 = time.time()
    env = os.environ.copy()
    patient_tokens_path = out_log_dir / "patient_tokens.json"
    if patient_tokens_path.exists():
        patient_tokens_path.unlink()
    env["MEDIQ_PATIENT_TOKENS_OUT"] = str(patient_tokens_path)
    proc = subprocess.run(cmd, cwd=str(MEDIQ_REPO / "src"), env=env)
    print(f"[run] MediQ exit={proc.returncode}, elapsed {time.time()-t0:.1f}s")

    if not output_jsonl.exists():
        print("ERROR: no output produced — check logs at " + str(out_log_dir / "run.log"))
        sys.exit(1)

    patient_tokens = {}
    if patient_tokens_path.exists():
        patient_tokens = json.loads(patient_tokens_path.read_text())

    # Build mediq_id -> case lookup
    case_by_mediqid = {c["_test_index"]: c for c in cases}

    rows = [json.loads(l) for l in open(output_jsonl)]
    print(f"\n[summary] {len(rows)} cases written by mediQ")
    out_f = out_path.open("w")
    top1 = 0; abs_strict = 0
    total_d_in = total_d_out = total_p_calls = 0
    for r in rows:
        chosen_letter = r["interactive_system"]["letter_choice"]
        chosen_disease = r["info"]["options"].get(chosen_letter, "?")
        gt = r["info"]["correct_answer"]
        questions = r["interactive_system"]["questions"]
        answers = r["interactive_system"]["answers"]
        intermediate = r["interactive_system"].get("intermediate_choices", [])

        # Native (force-final) hit
        hit = chosen_disease.lower().strip() == gt.lower().strip()

        # Strict-abstain semantics: was Likert >= 4 ever crossed?
        # mediQ writes intermediate confidence info; we infer abstain from
        # whether the run hit max_questions without confidence crossing.
        # If `temp_additional_info[-1]` doesn't show abstain in choice,
        # treat as forced commit ⇒ abstain under strict semantics.
        strict_abstain_flag = False
        tia = r["interactive_system"].get("temp_additional_info", [])
        # Heuristic: if questions == max_questions, it's a forced commit ⇒ abstain (strict)
        if len(questions) >= 10:
            strict_abstain_flag = True

        # Doctor token aggregation
        d_in = d_out = d_calls = 0
        for t in tia:
            u = (t or {}).get("usage") or {}
            d_in += int(u.get("input_tokens", 0) or 0)
            d_out += int(u.get("output_tokens", 0) or 0)
            if u: d_calls += 1
        total_d_in += d_in; total_d_out += d_out

        hadm_id = r["info"]["initial_info"]
        # better: lookup by id
        case_id = r["id"]
        case = case_by_mediqid.get(case_id, {})
        patient_id = case.get("patient_id", f"htsim_{case_id:04d}")
        p = patient_tokens.get(patient_id, {})
        p_in = int(p.get("input_tokens", 0) or 0)
        p_out = int(p.get("output_tokens", 0) or 0)
        p_calls = int(p.get("calls", 0) or 0)
        total_p_calls += p_calls

        top1 += int(hit)
        abs_strict += int(strict_abstain_flag)

        out_row = {
            "patient_id": patient_id,
            "ground_truth": gt,
            "predicted": chosen_disease,
            "predicted_set": [chosen_disease],
            "top1_hit": bool(hit),
            "top3_hit": None,                  # MediQ MCQ → no native top-3
            "abstained": False,                # native (force-final) commits
            "abstained_strict": bool(strict_abstain_flag),
            "stop_reason": "MAX_Q_FORCED" if strict_abstain_flag else "DDX_PROVIDED",
            "turns": len(questions),
            "doctor_mode": "mediq_expert",
            "letter_choice": chosen_letter,
            "options_count": len(r["info"]["options"]),
            "intermediate_choices": intermediate,
            "questions": questions,
            "answers": answers,
            "token_usage": {
                "doctor": {"prompt_tokens": d_in, "completion_tokens": d_out, "calls": d_calls},
                "patient": {"prompt_tokens": p_in, "completion_tokens": p_out, "calls": p_calls},
            },
        }
        out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n"); out_f.flush()
    out_f.close()
    print(f"  Top-1 (force-final): {top1}/{len(rows)} = {top1/max(1,len(rows)):.0%}")
    print(f"  Strict-abstain count: {abs_strict}/{len(rows)} = {abs_strict/max(1,len(rows)):.0%}")
    print(f"  Tokens — doctor in/out={total_d_in}/{total_d_out}, patient calls={total_p_calls}")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
