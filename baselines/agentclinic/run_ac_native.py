"""Run AgentClinic's published baseline VERBATIM on AC-MedQA-50.

This is a faithful port of agentclinic.main()'s per-case loop
(agentclinic.py:600-660) including:

  * Final-turn forcing (line 629-631): patient dialog gets
    "This is the final question. Please provide a diagnosis." prepended
    on the LAST inference, forcing the doctor to commit.
  * REQUEST TEST routing through MeasurementAgent (line 646-649) — the
    doctor's "REQUEST TEST: <test>" requests are intercepted, served by
    AC's own MeasurementAgent (LLM that simulates a measurement reader
    using scenario.exam_information()), and added to BOTH the patient's
    and the measurement agent's histories. Without this, the patient
    LLM gets a malformed dialog turn.
  * compare_results LLM moderator (line 565-567) — AC's own correctness
    check (gpt-5.4-nano asked "are these the same disease? Yes/No") runs
    on the doctor's free-text DIAGNOSIS READY line.

We do NOT modify AC's code. The only change is replacing AC's
query_model (openai 0.x SDK + hardcoded model whitelist) with a thin
bypass that uses openai 1.x and supports gpt-5.4-nano. Outputs are
saved one JSON per case under data/benchmark_results/agentclinic_50_ac_native/
with the same shape as the other AC-50 runners.

Run:
    OPENAI_API_KEY=... AGENTCLINIC_REPO=/tmp/agentclinic-tmp \
        PYTHONPATH=. python3 baselines/agentclinic/run_ac_native.py \
        --num-cases 50 --total-inferences 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Make AgentClinic importable
AGENTCLINIC_REPO = Path(os.environ.get("AGENTCLINIC_REPO", "/tmp/agentclinic-tmp"))
if str(AGENTCLINIC_REPO) not in sys.path:
    sys.path.insert(0, str(AGENTCLINIC_REPO))


def ac_query_openai(model: str, prompt: str, system_prompt: str,
                    api_key: str, max_completion: int = 400) -> tuple[str, dict]:
    """Modern openai 1.x bypass for AC's query_model.

    Used by all three AC agents (DoctorAgent, PatientAgent, MeasurementAgent)
    AND the moderator (compare_results). Returns (text, usage_dict).
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    is_reasoning = model.startswith(("gpt-5", "o"))
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    if is_reasoning:
        kwargs["max_completion_tokens"] = max_completion
    else:
        kwargs["temperature"] = 0.05
        kwargs["max_tokens"] = max_completion
    resp = client.chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip()
    text = re.sub(r"\s+", " ", text)
    u = resp.usage
    usage = {
        "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
    }
    return text, usage


def run_one_case(case, case_idx, args, out_dir):
    """Verbatim port of AC's main() per-case loop (agentclinic.py:600-660)."""
    import agentclinic

    scenario = agentclinic.ScenarioMedQAExtended(
        case if "OSCE_Examination" in case else {"OSCE_Examination": case}
    )
    gt = scenario.diagnosis_information()

    doctor = agentclinic.DoctorAgent(
        scenario=scenario, backend_str=args.doctor_model,
        max_infs=args.total_inferences, img_request=False,
    )
    patient = agentclinic.PatientAgent(
        scenario=scenario, backend_str=args.patient_model,
    )
    measurement = agentclinic.MeasurementAgent(
        scenario=scenario, backend_str=args.measurement_model,
    )

    # Per-agent token tracking
    tok = {
        "doctor":      {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
        "patient":     {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
        "measurement": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
        "moderator":   {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
    }

    def _inf_doctor(pi_dialogue: str) -> str:
        # Mirrors DoctorAgent.inference_doctor (agentclinic.py:511-518)
        if doctor.infs >= doctor.MAX_INFS:
            return "Maximum inferences reached"
        prompt = ("\nHere is a history of your dialogue: " + doctor.agent_hist
                  + "\n Here was the patient response: " + pi_dialogue
                  + "Now please continue your dialogue\nDoctor: ")
        text, usage = ac_query_openai(args.doctor_model, prompt, doctor.system_prompt(),
                                       args.api_key, max_completion=400)
        doctor.agent_hist += pi_dialogue + "\n\n" + text + "\n\n"
        doctor.infs += 1
        tok["doctor"]["prompt_tokens"]     += usage["prompt_tokens"]
        tok["doctor"]["completion_tokens"] += usage["completion_tokens"]
        tok["doctor"]["calls"]             += 1
        return text

    def _inf_patient(doctor_dialogue: str) -> str:
        # Mirrors PatientAgent.inference_patient (agentclinic.py:434-437)
        prompt = ("\nHere is a history of your dialogue: " + patient.agent_hist
                  + "\n Here was the doctor response: " + doctor_dialogue
                  + "Now please continue your dialogue\nPatient: ")
        text, usage = ac_query_openai(args.patient_model, prompt, patient.system_prompt(),
                                       args.api_key, max_completion=200)
        patient.agent_hist += doctor_dialogue + "\n\n" + text + "\n\n"
        tok["patient"]["prompt_tokens"]     += usage["prompt_tokens"]
        tok["patient"]["completion_tokens"] += usage["completion_tokens"]
        tok["patient"]["calls"]             += 1
        return text

    def _inf_measurement(doctor_dialogue: str) -> str:
        # Mirrors MeasurementAgent.inference_measurement (agentclinic.py:546-550)
        prompt = ("\nHere is a history of the dialogue: " + measurement.agent_hist
                  + "\n Here was the doctor measurement request: " + doctor_dialogue)
        text, usage = ac_query_openai(args.measurement_model, prompt, measurement.system_prompt(),
                                       args.api_key, max_completion=200)
        measurement.agent_hist += doctor_dialogue + "\n\n" + text + "\n\n"
        tok["measurement"]["prompt_tokens"]     += usage["prompt_tokens"]
        tok["measurement"]["completion_tokens"] += usage["completion_tokens"]
        tok["measurement"]["calls"]             += 1
        return text

    def _moderate(doctor_dialogue: str) -> bool:
        # Mirrors compare_results (agentclinic.py:565-567)
        prompt = ("\nHere is the correct diagnosis: " + gt
                  + "\n Here was the doctor dialogue: " + doctor_dialogue
                  + "\nAre these the same?")
        sysp = ("You are responsible for determining if the corrent diagnosis "
                "and the doctor diagnosis are the same disease. Please respond "
                "only with Yes or No. Nothing else.")
        text, usage = ac_query_openai(args.moderator_model, prompt, sysp,
                                       args.api_key, max_completion=10)
        tok["moderator"]["prompt_tokens"]     += usage["prompt_tokens"]
        tok["moderator"]["completion_tokens"] += usage["completion_tokens"]
        tok["moderator"]["calls"]             += 1
        return text.lower().strip().startswith("yes")

    # --- Verbatim port of agentclinic.py:621-659 ---
    pi_dialogue = ""
    doctor_dialogue = ""
    diagnosis_provided = False
    correct = False
    turns_log = []

    for inf_id in range(args.total_inferences):
        # Final-turn forcing (line 630-631)
        if inf_id == args.total_inferences - 1:
            pi_dialogue += "This is the final question. Please provide a diagnosis.\n"
        # Doctor inference (line 636)
        doctor_dialogue = _inf_doctor(pi_dialogue)
        turns_log.append({
            "turn": inf_id + 1, "role": "doctor", "content": doctor_dialogue,
            "final_turn_forced": (inf_id == args.total_inferences - 1),
        })
        # Diagnosis check (line 639-644)
        if "DIAGNOSIS READY" in doctor_dialogue:
            diagnosis_provided = True
            correct = _moderate(doctor_dialogue)
            break
        # REQUEST TEST routing (line 646-649)
        if "REQUEST TEST" in doctor_dialogue:
            pi_dialogue = _inf_measurement(doctor_dialogue)
            patient.add_hist(pi_dialogue)
            turns_log.append({"turn": inf_id + 1, "role": "measurement", "content": pi_dialogue})
        else:
            pi_dialogue = _inf_patient(doctor_dialogue)
            measurement.add_hist(pi_dialogue)
            turns_log.append({"turn": inf_id + 1, "role": "patient", "content": pi_dialogue})

    # Persist
    log = {
        "case_idx": case_idx,
        "baseline": "ac_native_published",
        "ground_truth": gt,
        "patient_actor_hidden": (case.get("OSCE_Examination", case)).get("Patient_Actor", {}),
        "objective_for_doctor": scenario.examiner_information(),
        "turns": turns_log,
        "final": {
            "predicted": doctor_dialogue,                      # raw free-text
            "diagnosis_provided": diagnosis_provided,
            "moderator_correct": correct,
            "top1_hit": bool(correct and diagnosis_provided),
            "top3_hit": bool(correct and diagnosis_provided),  # AC has no Top-3
            "abstained": (not diagnosis_provided),
            "stop_reason": "DIAGNOSIS_READY" if diagnosis_provided else "MAX_INFERENCES",
            "turns": doctor.infs,
        },
        "token_usage": tok,
    }
    out_path = out_dir / f"case_{case_idx:03d}.json"
    with out_path.open("w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-cases", type=int, default=50)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--total-inferences", type=int, default=20,
                    help="AC paper default is 20")
    ap.add_argument("--doctor-model",      default="gpt-5.4-nano")
    ap.add_argument("--patient-model",     default="gpt-5.4-nano")
    ap.add_argument("--measurement-model", default="gpt-5.4-nano")
    ap.add_argument("--moderator-model",   default="gpt-5.4-nano")
    ap.add_argument("--out-suffix",        default="agentclinic_50_ac_native")
    args = ap.parse_args()
    args.api_key = os.environ.get("OPENAI_API_KEY")
    if not args.api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr); sys.exit(1)

    from baselines.agentclinic.adapters.base import load_selected_cases
    all_cases = load_selected_cases()
    cases = all_cases[args.start_idx: args.num_cases]

    out_dir = REPO / f"data/benchmark_results/{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] AC-native (verbatim port of agentclinic.main loop) | "
          f"cases [{args.start_idx},{args.num_cases}) ({len(cases)}) | "
          f"total_inferences={args.total_inferences} | doctor={args.doctor_model}")

    rows = []
    t0 = time.time()
    for offset, case in enumerate(cases):
        idx = args.start_idx + offset
        ts = time.time()
        try:
            log = run_one_case(case, idx, args, out_dir)
            f = log["final"]
            tag = ("HIT" if f["top1_hit"] else
                   ("ABS" if f["abstained"] else "miss"))
            print(f"  [{offset+1}/{len(cases)}] (idx={idx}) {tag:>4} "
                  f"GT={log['ground_truth'][:30]:<30} "
                  f"({f['turns']} turns, {time.time()-ts:.0f}s)")
            rows.append({"idx": idx, "top1": f["top1_hit"], "abstain": f["abstained"],
                         "turns": f["turns"]})
        except Exception as e:
            print(f"  [{offset+1}/{len(cases)}] (idx={idx}) ERROR: {e}")
            rows.append({"idx": idx, "error": str(e)})

    valid = [r for r in rows if "error" not in r]
    if valid:
        n = len(valid)
        t1 = sum(r["top1"] for r in valid)
        ab = sum(r["abstain"] for r in valid)
        mt = sum(r["turns"] for r in valid) / n
        print(f"\n  AC-native (verbatim) on {n} cases\n"
              f"  Top-1   : {t1}/{n} = {t1/n:.0%}\n"
              f"  Abstain : {ab}/{n} = {ab/n:.0%}\n"
              f"  Mean turns: {mt:.1f}\n"
              f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
