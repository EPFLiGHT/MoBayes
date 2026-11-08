"""AgentClinic native row — verbatim port of agentclinic.main() per-case loop.

Wraps AC's published `DoctorAgent` + `PatientAgent` + `MeasurementAgent` +
`compare_results` moderator. Includes:
  * Final-turn forcing (agentclinic.py:630-631).
  * `REQUEST TEST` → MeasurementAgent routing (line 646-649).
  * LLM moderator for correctness check (compare_results, line 565-567).

Defines its own ``run_one_case`` (the dialog flow doesn't fit the standard
[DDX]/[ABSTAIN] protocol).

Only modification vs AC's code: a thin OpenAI 1.x bypass for query_model
(AC's legacy 0.x SDK doesn't support gpt-5.4-nano).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict


NAME = "ac_native"

_AC_REPO = Path(os.environ.get("AGENTCLINIC_REPO", "/tmp/agentclinic-tmp"))
if str(_AC_REPO) not in sys.path:
    sys.path.insert(0, str(_AC_REPO))


def _ac_query_openai(model: str, prompt: str, system_prompt: str,
                      api_key: str, max_completion: int = 400):
    """openai>=1.x bypass for AC's query_model. Returns (text, usage)."""
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
    text = re.sub(r"\s+", " ", (resp.choices[0].message.content or "").strip())
    u = resp.usage
    usage = {
        "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
    }
    return text, usage


def make_doctor(*args, **kwargs):
    """Not used — ac_native overrides run_one_case entirely."""
    raise NotImplementedError("ac_native uses run_one_case override; no make_doctor.")


class _HTSimScenario:
    """Minimal AC-compatible scenario built from a DDxPlus HT-sim case.

    AC's DoctorAgent only reads ``examiner_information()``. HT-sim provides
    age + sex + chief complaint; that's enough for the doctor briefing.
    """
    def __init__(self, case: Dict[str, Any]) -> None:
        chief = case.get("INITIAL_EVIDENCE_ENG", "") or "Patient presents with unspecified concerns."
        age = case.get("AGE", "?")
        sex = case.get("SEX", "?")
        self._examiner = (f"A {age}-year-old {sex} patient presents to the clinic. "
                          f"Chief complaint: {chief}.")

    def examiner_information(self) -> str:
        return self._examiner


class _PatientSimScenario:
    """AC-compatible scenario built from a DDxPlus PatientSim profile.

    Same minimal contract as ``_HTSimScenario``: AC's DoctorAgent only reads
    ``examiner_information()``. PatientSim profile carries age / gender + chief
    complaint; that's enough for the doctor briefing.
    """
    def __init__(self, profile: Dict[str, Any]) -> None:
        chief = (profile.get("Current Visit", {}).get("chiefcomplaint")
                 or "presenting complaint")
        age   = profile.get("age", "?")
        sex   = profile.get("gender", "?")
        self._examiner = (f"A {age}-year-old {sex} patient presents to the clinic. "
                          f"Chief complaint: {chief}.")

    def examiner_information(self) -> str:
        return self._examiner


# ── DDxPlus columns (PatientSim / HT-sim): AC's DoctorAgent over our patient ──

def _run_one_case_ddxplus(
    case: Dict[str, Any],
    case_idx: int,
    *,
    column,
    doctor_model: str,
    patient_model: str,
    max_turns: int,
    api_key: str,
    out_dir: Path,
    row_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """AC's DoctorAgent (via ACNativeDoctorAgent) against a DDxPlus patient.

    Differences from the AgentClinic branch in ``run_one_case``:
      * Patient   = the column's patient (PatientSim or MEDDxAgent's LLMPatient);
                    no MeasurementAgent.
      * Scenario  = synthetic age/sex/chief-complaint briefing.
      * Final-turn forcing appended to the patient utterance.
      * Scoring   = string match of the mapped diagnosis against the DDxPlus
                    list (no compare_results moderator).
      * Tokens    = doctor via ``ACNativeDoctorAgent.token_usage``; patient via
                    MEDDxAgent's ``usage_tracker`` (HT-sim) or the patient
                    object's ``token_usage`` (PatientSim).
    """
    from baselines.agentclinic.doctor import ACNativeDoctorAgent

    htsim = column.NAME in ("htsim", "htsim_v1")
    if htsim:
        from baselines.meddxagent import usage_tracker
        usage_tracker.reset()
        scenario = _HTSimScenario(case)
    else:
        scenario = _PatientSimScenario(case)

    gt = column.gt_of(case)
    disease_names = column.disease_universe()
    patient = column.make_patient(case, patient_model)
    doctor = ACNativeDoctorAgent(scenario=scenario,
                                  disease_names=disease_names,
                                  model=doctor_model, api_key=api_key,
                                  max_turns=max_turns)

    dialog = []
    greeting = "Hello, I'm your doctor today. What brings you in today?"
    pa = patient.respond(greeting)
    dialog.append({"role": "doctor", "content": greeting, "turn": 0})
    dialog.append({"role": "patient", "content": pa, "turn": 0})

    stop_reason = "MAX_TURNS"
    last_doctor_msg = ""
    final_via_callback = False
    for turn in range(max_turns + 5):
        # AC paper-faithful final-turn forcing (agentclinic.py:630-631):
        if turn == max_turns - 1:
            pa = (pa or "") + "\n\nThis is the final question. Please provide a diagnosis."
        question, is_final = doctor.get_question(pa)
        dialog.append({"role": "doctor", "content": question, "turn": turn + 1})
        last_doctor_msg = question
        if is_final:
            stop_reason = "DDX_PROVIDED"; final_via_callback = True; break
        pa = patient.respond(question)
        dialog.append({"role": "patient", "content": pa, "turn": turn + 1})

    abstained = "[ABSTAIN]" in last_doctor_msg
    if final_via_callback or "[DDX]" in last_doctor_msg:
        preds = doctor.parse_ddx(last_doctor_msg); stop_reason = "DDX_PROVIDED"
    elif abstained:
        preds = doctor.parse_possible(last_doctor_msg); stop_reason = "ABSTAINED"
    else:
        ddx_msg, _ = doctor.get_question("Please provide your final [DDX] or [ABSTAIN] now.")
        dialog.append({"role": "doctor", "content": ddx_msg, "turn": -1})
        if "[DDX]" in ddx_msg or "DIAGNOSIS READY" in ddx_msg.upper():
            preds = doctor.parse_ddx(ddx_msg); stop_reason = "DDX_PROVIDED"
        else:
            abstained = True
            preds = doctor.parse_possible(ddx_msg); stop_reason = "ABSTAINED"

    top1 = preds[0] if preds else "Unknown"
    raw_hit = column.matches_gt(top1, gt)
    hit = bool(raw_hit)
    # AC native commits with a single "DIAGNOSIS READY: <X>"; no ranked
    # differential is produced. Match the AC-MedQA path: empty
    # predicted_set, top3_hit mirrors top1_hit (Top-3 metric undefined).
    in_top3 = hit

    if htsim:
        pat = usage_tracker.snapshot()["patient"]
        patient_tokens = {"prompt_tokens": pat.get("prompt_tokens", 0),
                          "completion_tokens": pat.get("completion_tokens", 0),
                          "calls": pat.get("calls", 0)}
    else:
        patient_tokens = dict(getattr(patient, "token_usage", {}))

    log = {
        "case_idx": case_idx,
        "baseline": NAME,
        "column": column.NAME,
        "ground_truth": gt,
    }
    if htsim:
        log["patient_actor_hidden"] = case
    log.update({
        "objective_for_doctor": scenario.examiner_information(),
        "turns": dialog,
        "final": {
            "predicted":     "ABSTAIN" if abstained else top1,
            "predicted_set": [],
            "top1_hit":      hit,
            "top3_hit":      in_top3,
            "raw_top1_hit":  bool(raw_hit),
            "abstained":     bool(abstained),
            "stop_reason":   stop_reason,
            "turns": sum(1 for d in dialog if d["role"] == "doctor" and d.get("turn", 0) > 0),
        },
        "token_usage": {
            "doctor":  dict(getattr(doctor, "token_usage", {})),
            "patient": patient_tokens,
        },
    })
    out_path = out_dir / f"case_{case_idx:03d}.json"
    with out_path.open("w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    return log


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
    """Dispatch by column:

      * AC      → verbatim port of agentclinic.main() (MeasurementAgent +
                  compare_results moderator; this function below).
      * PatientSim / HT-sim → ACNativeDoctorAgent over the column's patient
                  (no MeasurementAgent, string-match scoring; see
                  ``_run_one_case_ddxplus``).
    """
    if column.NAME in ("htsim", "htsim_v1", "patientsim", "patientsim_v1"):
        return _run_one_case_ddxplus(
            case, case_idx, column=column,
            doctor_model=doctor_model, patient_model=patient_model,
            max_turns=max_turns, api_key=api_key, out_dir=out_dir,
            row_kwargs=row_kwargs or {},
        )
    if column.NAME not in ("ac", "ac_set2"):
        raise NotImplementedError(
            f"ac_native row does not yet support column={column.NAME!r}")

    import agentclinic
    row_kwargs = row_kwargs or {}
    measurement_model = row_kwargs.get("measurement_model", doctor_model)
    moderator_model   = row_kwargs.get("moderator_model",   doctor_model)
    total_inferences  = row_kwargs.get("total_inferences",  max_turns)

    scenario = agentclinic.ScenarioMedQAExtended(
        case if "OSCE_Examination" in case else {"OSCE_Examination": case}
    )
    gt = scenario.diagnosis_information()

    doctor      = agentclinic.DoctorAgent(scenario=scenario, backend_str=doctor_model,
                                           max_infs=total_inferences, img_request=False)
    patient     = agentclinic.PatientAgent(scenario=scenario, backend_str=patient_model)
    measurement = agentclinic.MeasurementAgent(scenario=scenario, backend_str=measurement_model)

    tok = {k: {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
           for k in ("doctor", "patient", "measurement", "moderator")}

    def _accumulate(actor: str, usage: Dict[str, int]) -> None:
        tok[actor]["prompt_tokens"]     += usage["prompt_tokens"]
        tok[actor]["completion_tokens"] += usage["completion_tokens"]
        tok[actor]["calls"]             += 1

    def _inf_doctor(pi_dialogue: str) -> str:
        if doctor.infs >= doctor.MAX_INFS:
            return "Maximum inferences reached"
        prompt = ("\nHere is a history of your dialogue: " + doctor.agent_hist
                  + "\n Here was the patient response: " + pi_dialogue
                  + "Now please continue your dialogue\nDoctor: ")
        text, usage = _ac_query_openai(doctor_model, prompt, doctor.system_prompt(),
                                        api_key, max_completion=400)
        doctor.agent_hist += pi_dialogue + "\n\n" + text + "\n\n"
        doctor.infs += 1
        _accumulate("doctor", usage)
        return text

    def _inf_patient(doctor_dialogue: str) -> str:
        prompt = ("\nHere is a history of your dialogue: " + patient.agent_hist
                  + "\n Here was the doctor response: " + doctor_dialogue
                  + "Now please continue your dialogue\nPatient: ")
        text, usage = _ac_query_openai(patient_model, prompt, patient.system_prompt(),
                                        api_key, max_completion=200)
        patient.agent_hist += doctor_dialogue + "\n\n" + text + "\n\n"
        _accumulate("patient", usage)
        return text

    def _inf_measurement(doctor_dialogue: str) -> str:
        prompt = ("\nHere is a history of the dialogue: " + measurement.agent_hist
                  + "\n Here was the doctor measurement request: " + doctor_dialogue)
        text, usage = _ac_query_openai(measurement_model, prompt, measurement.system_prompt(),
                                        api_key, max_completion=200)
        measurement.agent_hist += doctor_dialogue + "\n\n" + text + "\n\n"
        _accumulate("measurement", usage)
        return text

    def _moderate(doctor_dialogue: str) -> bool:
        prompt = ("\nHere is the correct diagnosis: " + gt
                  + "\n Here was the doctor dialogue: " + doctor_dialogue
                  + "\nAre these the same?")
        sysp = ("You are responsible for determining if the corrent diagnosis "
                "and the doctor diagnosis are the same disease. Please respond "
                "only with Yes or No. Nothing else.")
        text, usage = _ac_query_openai(moderator_model, prompt, sysp,
                                        api_key, max_completion=10)
        _accumulate("moderator", usage)
        return text.lower().strip().startswith("yes")

    # --- Verbatim port of agentclinic.py:621-659 ---
    pi_dialogue = ""
    doctor_dialogue = ""
    diagnosis_provided = False
    correct = False
    turns_log = []

    for inf_id in range(total_inferences):
        if inf_id == total_inferences - 1:
            pi_dialogue += "This is the final question. Please provide a diagnosis.\n"
        doctor_dialogue = _inf_doctor(pi_dialogue)
        turns_log.append({
            "turn": inf_id + 1, "role": "doctor", "content": doctor_dialogue,
            "final_turn_forced": (inf_id == total_inferences - 1),
        })
        if "DIAGNOSIS READY" in doctor_dialogue:
            diagnosis_provided = True
            correct = _moderate(doctor_dialogue)
            break
        if "REQUEST TEST" in doctor_dialogue:
            pi_dialogue = _inf_measurement(doctor_dialogue)
            patient.add_hist(pi_dialogue)
            turns_log.append({"turn": inf_id + 1, "role": "measurement",
                              "content": pi_dialogue})
        else:
            pi_dialogue = _inf_patient(doctor_dialogue)
            measurement.add_hist(pi_dialogue)
            turns_log.append({"turn": inf_id + 1, "role": "patient",
                              "content": pi_dialogue})

    log = {
        "case_idx": case_idx,
        "baseline": NAME,
        "column": column.NAME,
        "ground_truth": gt,
        "patient_actor_hidden": (case.get("OSCE_Examination", case)).get("Patient_Actor", {}),
        "objective_for_doctor": scenario.examiner_information(),
        "turns": turns_log,
        "final": {
            "predicted": doctor_dialogue,
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
