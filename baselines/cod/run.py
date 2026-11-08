"""Run DiagnosisGPT-34B (Chain-of-Diagnosis) as the doctor on one bench column.

DiagnosisGPT runs as a chatbot (CoD's ``cod_cli.DiagnosisChatbot``): the
first turn is the patient's chief complaint; the model then either asks a
follow-up question or commits with a "Diagnosis: <disease>" line plus a
confidence distribution over its own 9,604-disease retrieval space. CoD
predicts freely; the free-text diagnosis is then mapped to the column's
closed disease list.

    --column patientsim   our PatientSim; LLM mapper to the 49 DDxPlus leaves
    --column htsim        MEDDxAgent's LLMPatient; name matching to the DDxPlus list
    --column agentclinic  AgentClinic's PatientAgent; LLM mapper, plus an LLM
                          commit-or-abstain step when CoD never diagnosed
                          (``--posthoc-from-jsonl`` re-runs only that step on a
                          saved run, without the model)

Needs a GPU node, the DiagnosisGPT-34B checkpoint and a CoD clone; the Slurm
script that calls this is written by baselines/cod/row.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.cod.force_final import (  # noqa: E402
    force_final_via_llm,
    map_to_closed,
    posthoc_recompute as _shared_posthoc,
)

# Per-column defaults (only used when the flag is not given).
_DEFAULTS = {
    "patientsim": dict(num_cases=50, max_inquiries=10),
    "htsim": dict(num_cases=50, max_inquiries=15),
    "agentclinic": dict(num_cases=10, max_inquiries=10),
}


def parse_diagnosis_line(text: str) -> str:
    """Text after 'Diagnosis:' up to the end of that line (HT-sim, AgentClinic)."""
    m = re.search(r"Diagnosis:\s*(.+?)(?:\n|$)", text)
    return m.group(1).strip() if m else ""


# ── PatientSim ────────────────────────────────────────────────────────────

def build_chief_complaint(profile: Dict) -> str:
    """First user turn — what the patient tells the doctor up front."""
    age = profile.get("age", "?")
    gender = profile.get("gender", "?")
    cv = profile.get("Current Visit", {}) or {}
    chief = cv.get("chiefcomplaint", "I am not feeling well")
    return f"I am a {age}-year-old {gender}. {chief}"


def make_patientsim_callable(profile: Dict, patient_model: str, api_key: str | None):
    """Return (respond, get_usage) driving our PatientSim with DDxPlus
    closed-world rules (same patient as every other PatientSim baseline)."""
    from llm_interface.prompt_templates import build_patient_system_prompt
    from llm_interface.llm_client import OpenAILLMClient, sanitize_openai_text

    system_prompt = build_patient_system_prompt(profile)
    messages = [{"role": "system", "content": system_prompt}]
    llm = OpenAILLMClient(model=patient_model, api_key=api_key, temperature=0.7)

    def respond(question: str) -> str:
        messages.append({"role": "user", "content": sanitize_openai_text(question)})
        answer, _usage = llm.chat(messages, max_tokens=200)
        messages.append({"role": "assistant", "content": answer})
        return answer

    return respond, lambda: dict(llm.total_usage)


def parse_patientsim_diagnosis(model_text: str) -> str:
    """Extract diagnosis name after '## Diagnosis:' (cut at . , ;) or 'Diagnosis:'."""
    if "## Diagnosis:" in model_text:
        tail = model_text.split("## Diagnosis:", 1)[1].strip()
        first_line = tail.split("\n", 1)[0].strip()
        for delim in (".", ",", ";"):
            first_line = first_line.split(delim, 1)[0].strip()
        return first_line
    if "Diagnosis:" in model_text:
        tail = model_text.split("Diagnosis:", 1)[1].strip()
        return tail.split("\n", 1)[0].strip()
    return ""


_LLM_MAP_PROMPT = """You are a medical-coding assistant. Given a free-form diagnosis from one model and a closed list of {n} diseases, return EXACTLY ONE disease name from the list that semantically matches the free-form output, OR "Unknown" if no listed disease is a semantic match.

Closed disease list:
{disease_list}

Free-form diagnosis to map:
---
diagnosis_text: {diag_text}
top confidence diseases (model's own ranked candidates): {top_conf}
---

Rules:
- Output ONLY the disease name (one of the {n} above) or "Unknown" — no explanation.
- A semantic match is when the free-form output refers to the same disease entity as a listed one, even if wording differs (e.g. "COPD" matches "Acute COPD exacerbation / infection"; "Acute pyelonephritis with sepsis" matches "Pyelonephritis").
- If fundamentally different from any listed disease, return "Unknown".
- If the free-form output is empty, vague, or just instructions/format text, return "Unknown".

Answer:"""


def map_to_ddxplus_llm(prediction: str, dist: Dict[str, float], ddxplus_diseases: List[str],
                       llm_client=None, llm_model: str = "gpt-5.4-nano") -> List[str]:
    """LLM-based semantic mapping from CoD free-text → DDxPlus 49-disease set.

    Substring matching is brittle: CoD outputs ("Esophageal Spasm", verbose
    differentials) rarely substring-match the canonical 49-name DDxPlus set,
    and an alphabetical fallback would silently corrupt every unmapped case.
    One ~80-token call per case asks which of the 49 best matches, with
    "Unknown" as a safe fallback for genuinely off-list predictions.

    Returns [chosen_disease] (length 1) or [] if Unknown / mapping skipped.
    """
    if llm_client is None:
        return []

    top_conf = ", ".join(
        f"{d}({c:.2f})" for d, c in sorted(dist.items(), key=lambda kv: -kv[1])[:5]
    ) or "(none)"

    msg = _LLM_MAP_PROMPT.format(
        n=len(ddxplus_diseases),
        disease_list="\n".join(f"- {d}" for d in ddxplus_diseases),
        diag_text=(prediction or "")[:600],
        top_conf=top_conf,
    )
    is_reasoning = llm_model.startswith(("gpt-5", "o"))
    kwargs = {"model": llm_model, "messages": [{"role": "user", "content": msg}]}
    if is_reasoning:
        kwargs["max_completion_tokens"] = 80
    else:
        kwargs["temperature"] = 0.0
        kwargs["max_tokens"] = 60
    try:
        resp = llm_client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"  [warn] mapper LLM call failed: {e}", flush=True)
        return []
    raw = (resp.choices[0].message.content or "").strip().strip('"').strip("'")
    raw = raw.split("\n")[0].strip()
    if not raw or raw.lower() == "unknown":
        return []
    for d in ddxplus_diseases:
        if d.lower() == raw.lower():
            return [d]
    return []


def run_patientsim_case(profile: Dict, chatbot, patient_model: str, api_key: str | None,
                        max_inquiries: int = 10, ddxplus_diseases: List[str] = None,
                        mapper_client=None, mapper_model: str = "gpt-5.4-nano") -> Dict:
    """One CoD diagnostic session for a single PatientSim profile."""
    chief = build_chief_complaint(profile)
    respond, get_usage = make_patientsim_callable(profile, patient_model, api_key)

    history: list = []
    dialog: list = []
    dialog.append({"role": "patient", "content": chief, "turn": 0})

    cur_query = chief
    confidence = {}
    diagnosis_text = ""
    asked = 0

    for step in range(max_inquiries + 1):
        # CoD chatbot inference returns (cur_generate, history, confidence_distribution)
        try:
            generated, history, conf_dist = chatbot.inference(cur_query, history=history)
        except Exception as e:
            return {
                "patient_id": profile.get("hadm_id"),
                "error": f"inference failed: {e}",
                "dialog": dialog,
            }

        dialog.append({"role": "doctor", "content": generated, "turn": step + 1,
                       "confidence_distribution": conf_dist})

        if conf_dist:
            confidence = conf_dist

        # Check if model committed to a diagnosis
        if "## Diagnosis:" in generated or "Diagnosis:" in generated and "Inadequate" not in generated:
            diagnosis_text = parse_patientsim_diagnosis(generated)
            break

        # Otherwise model is asking a follow-up question
        if asked >= max_inquiries:
            break

        # The question follows "Ask for symptoms:" if present, else the last line with '?'
        if "Ask for symptoms:" in generated:
            question = generated.split("Ask for symptoms:", 1)[1].strip()
        else:
            question_lines = [ln for ln in generated.split("\n") if "?" in ln]
            question = question_lines[-1].strip() if question_lines else generated.strip()

        patient_response = respond(question)
        dialog.append({"role": "patient", "content": patient_response, "turn": step + 1})
        cur_query = patient_response
        asked += 1

    ranked = map_to_ddxplus_llm(diagnosis_text, confidence, ddxplus_diseases or [],
                                llm_client=mapper_client, llm_model=mapper_model)
    return {
        "patient_id": profile.get("hadm_id"),
        "ground_truth": profile.get("_ddxplus_meta", {}).get("diagnosis_name"),
        "ground_truth_id": profile.get("_ddxplus_meta", {}).get("ground_truth_ids", []),
        "cod_diagnosis_text": diagnosis_text,
        "cod_confidence_distribution": confidence,
        "predicted_set": ranked,
        "predicted": ranked[0] if ranked else "Unknown",
        "turns": asked,
        "dialog_history": dialog,
        "patient_token_usage": get_usage(),
    }


# ── HT-sim ────────────────────────────────────────────────────────────────

def map_to_ddxplus_by_name(prediction: str, dist: Dict[str, float], ddx_diseases: List[str]) -> List[str]:
    """Map CoD's free-form prediction + confidence dist into DDxPlus disease space.

    Returns a (possibly empty) ranked list of DDxPlus diseases. Empty list ⇒
    CoD's output could not be mapped to any of the 49 — caller should treat
    as miss (no alphabetical fallback, since that biases everything to "Acute COPD").
    """
    norm = lambda s: (s or "").lower().strip()
    lookup = {norm(d): d for d in ddx_diseases}
    ranked, seen = [], set()

    def _try_match(text: str) -> List[str]:
        nt = norm(text)
        if not nt:
            return []
        hits = []
        for k, original in lookup.items():
            if len(k) < 4:
                continue
            # disease name appears in candidate text
            if k in nt:
                hits.append(original); continue
            # candidate text appears in disease name (only for substantial cands)
            if len(nt) >= 5 and nt in k:
                hits.append(original); continue
        return hits

    # Pass 1: scan free-form prediction text for any DDxPlus name mention
    for original in _try_match(prediction):
        if original not in seen:
            ranked.append(original); seen.add(original)

    # Pass 2: confidence distribution sorted high→low
    for cand, _ in sorted(dist.items(), key=lambda kv: -kv[1]):
        for original in _try_match(cand):
            if original not in seen:
                ranked.append(original); seen.add(original)

    return ranked


def run_htsim_case(case: Dict, chatbot, patient_model: str,
                   max_inquiries: int, ddx_diseases: List[str]) -> Dict:
    from bench.patients.htsim import LLMPatientWrapper

    wrapper = LLMPatientWrapper(case, patient_model=patient_model)
    gt = wrapper.ground_truth
    chief = wrapper.initial_info or "I'm not feeling well."

    history, dialog = [], []
    dialog.append({"role": "patient", "content": chief, "turn": 0})
    cur_query = chief
    confidence: Dict[str, float] = {}
    diagnosis_text = ""
    asked = 0

    for step in range(max_inquiries + 1):
        try:
            generated, history, conf_dist = chatbot.inference(cur_query, history=history)
        except Exception as e:
            return {"patient_id": case.get("patient_id"), "ground_truth": gt,
                    "error": f"inference failed: {e}", "dialog": dialog}

        dialog.append({"role": "doctor", "content": generated, "turn": step + 1,
                       "confidence_distribution": conf_dist})
        if conf_dist:
            confidence = conf_dist

        if "Diagnosis:" in generated and "Inadequate" not in generated:
            diagnosis_text = parse_diagnosis_line(generated)
            break

        question = generated.strip().split("\n")[0]
        ans = wrapper.respond(question)
        dialog.append({"role": "patient", "content": ans, "turn": step + 1})
        cur_query = ans
        asked += 1

    if not diagnosis_text and confidence:
        diagnosis_text = max(confidence.items(), key=lambda kv: kv[1])[0]

    ranked = map_to_ddxplus_by_name(diagnosis_text, confidence, ddx_diseases)
    norm = lambda s: (s or "").lower().strip()
    mapping_failed = len(ranked) == 0
    top1 = ranked[0] if ranked else "Unknown"
    return {
        "patient_id": case.get("patient_id"),
        "ground_truth": gt,
        "predicted": top1,
        "predicted_set": ranked[:5],
        "top1_hit": (not mapping_failed) and norm(top1) == norm(gt),
        "top3_hit": (not mapping_failed) and any(norm(d) == norm(gt) for d in ranked[:3]),
        "abstained": False,           # CoD does not output abstentions
        "mapping_failed": mapping_failed,
        "stop_reason": "DDX_PROVIDED" if diagnosis_text else "MAX_TURNS",
        "turns": asked,
        "diagnosis_text": diagnosis_text,
        "confidence_distribution": confidence,
        "dialog": dialog,
    }


# ── AgentClinic-MedQA ─────────────────────────────────────────────────────

def make_agentclinic_callable(ac_case: Dict, patient_model: str = "gpt-5.4-nano",
                              bias_present: Optional[str] = None,
                              ) -> Tuple[Callable[[str], str], Callable[[], Dict[str, int]], str, str]:
    """Return (respond, get_usage, ground_truth, initial_info) driving
    AgentClinic's PatientAgent."""
    from bench.patients.agentclinic import AgentClinicCaseWrapper

    wrapper = AgentClinicCaseWrapper(ac_case, patient_model=patient_model, bias_present=bias_present)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def respond(question: str) -> str:
        out = wrapper.respond(question)
        usage["calls"] += 1
        return out

    return respond, lambda: dict(usage), wrapper.ground_truth, wrapper.initial_info


def run_agentclinic_case(ac_case: Dict, chatbot, patient_model: str,
                         max_inquiries: int, ac_diseases: List[str],
                         mapper_client=None, mapper_model: str = "gpt-5.4-nano") -> Dict:
    respond, get_usage, gt, initial_info = make_agentclinic_callable(
        ac_case, patient_model=patient_model,
    )

    # Chief complaint = AC's initial_info (age + chief complaint)
    chief = initial_info or "I'm not feeling well."
    history, dialog = [], []
    dialog.append({"role": "patient", "content": chief, "turn": 0})
    cur_query = chief
    confidence = {}
    diagnosis_text = ""
    asked = 0

    for step in range(max_inquiries + 1):
        try:
            generated, history, conf_dist = chatbot.inference(cur_query, history=history)
        except Exception as e:
            return {"case_id": ac_case.get("OSCE_Examination", ac_case).get("Correct_Diagnosis"),
                    "error": f"inference failed: {e}", "dialog": dialog}

        dialog.append({"role": "doctor", "content": generated, "turn": step + 1,
                       "confidence_distribution": conf_dist})
        if conf_dist:
            confidence = conf_dist

        if "Diagnosis:" in generated and "Inadequate" not in generated:
            diagnosis_text = parse_diagnosis_line(generated)
            break

        # Otherwise CoD asked a follow-up — route to AC patient
        question = generated.strip().split("\n")[0]
        ans = respond(question)
        dialog.append({"role": "patient", "content": ans, "turn": step + 1})
        cur_query = ans
        asked += 1

    # If CoD's natural loop never committed, force a final-turn diagnose-or-abstain
    # via an LLM call over the saved dialogue (matches the "force-final" semantics
    # used by other baselines on AC; abstain becomes a real signal here).
    forced = ""
    if not diagnosis_text:
        forced = force_final_via_llm(dialog, ac_diseases,
                                     llm_client=mapper_client, llm_model=mapper_model)

    if forced:
        ranked = [forced]
        diagnosis_text = forced
        abstained = False
    elif diagnosis_text:
        ranked = map_to_closed(diagnosis_text, confidence, ac_diseases,
                               llm_client=mapper_client, llm_model=mapper_model)
        abstained = not ranked
    else:
        # CoD didn't commit AND force-final returned Abstain (or no dialog)
        # -> fall back to top-confidence entry; if even that fails to map, abstain.
        if confidence:
            diagnosis_text = max(confidence.items(), key=lambda kv: kv[1])[0]
            ranked = map_to_closed(diagnosis_text, confidence, ac_diseases,
                                   llm_client=mapper_client, llm_model=mapper_model)
        else:
            ranked = []
        abstained = not ranked

    norm = lambda s: (s or "").lower().strip()
    top1 = ranked[0] if ranked else "Unknown"
    top1_hit = (not abstained) and norm(top1) == norm(gt)
    top3_hit = (not abstained) and any(norm(d) == norm(gt) for d in ranked[:3])

    return {
        "ground_truth": gt,
        "predicted": top1,
        "predicted_set": ranked[:5],
        "top1_hit": bool(top1_hit),
        "top3_hit": bool(top3_hit),
        "abstained": bool(abstained),
        "stop_reason": ("DDX_PROVIDED" if diagnosis_text and not forced
                        else ("FORCE_FINAL_LLM" if forced else "MAX_TURNS_ABSTAIN")),
        "turns": asked,
        "diagnosis_text": diagnosis_text,
        "confidence_distribution": confidence,
        "dialog": dialog,
        "patient_token_usage": get_usage(),
    }


def posthoc_recompute(saved, ac_diseases, mapper_client, mapper_model="gpt-5.4-nano"):
    """Re-run only the post-loop step on a saved AgentClinic row (no model)."""
    return _shared_posthoc(saved, ac_diseases, mapper_client, mapper_model,
                           dialog_field="dialog",
                           diag_text_field="diagnosis_text",
                           conf_field="confidence_distribution")


# ── Driver ────────────────────────────────────────────────────────────────

def _load_diseases(column: str, path: str) -> List[str]:
    if column == "patientsim":
        if path.endswith(".csv"):
            import pandas as pd
            kb = pd.read_csv(path)
            return sorted(kb.loc[~kb["disease_id"].isin(kb["parent_id"].dropna()), "name"])
        return sorted(set(json.load(open(path)).keys()))
    import pandas as pd
    return pd.read_csv(path)["name"].tolist()


def _load_cases(column: str, path: str, n: int) -> List[Dict]:
    if column == "patientsim":
        return json.load(open(path))[:n]
    return [json.loads(l) for l in open(path)][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--column", required=True, choices=sorted(_DEFAULTS))
    ap.add_argument("--cod-repo", default=None,
                    help="Path to FreedomIntelligence/Chain-of-Diagnosis clone")
    ap.add_argument("--model-dir", default=None,
                    help="Path to DiagnosisGPT-34B local checkout")
    ap.add_argument("--bmbe-root", default=str(REPO_ROOT),
                    help="Path to this repository (for llm_interface / bench)")
    ap.add_argument("--meddxagent-root", default=None,
                    help="htsim: local meddxagent checkout (provides ddxdriver)")
    ap.add_argument("--cases", "--profiles", dest="cases", default=None,
                    help="patientsim: profiles.json; htsim / agentclinic: selected_cases.jsonl")
    ap.add_argument("--diseases", required=True,
                    help="Closed disease list: the KB's diseases.csv (patientsim also accepts "
                         "DDxPlus release_conditions.json; leaf rows are used)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patient-model", default="gpt-5.4-nano")
    ap.add_argument("--num-cases", "--num-patients", dest="num_cases", type=int, default=None)
    ap.add_argument("--confidence-threshold", type=float, default=0.5)
    ap.add_argument("--max-inquiries", type=int, default=None)
    ap.add_argument("--mapper-model", default="gpt-5.4-nano",
                    help="patientsim / agentclinic: OpenAI model for the ontology mapper "
                         "(and the agentclinic force-final step)")
    ap.add_argument("--posthoc-from-jsonl", default=None,
                    help="agentclinic: skip CoD inference and reprocess the saved dialogs in "
                         "this jsonl (adds the force-final + map step retroactively).")
    args = ap.parse_args()
    for key, value in _DEFAULTS[args.column].items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    sys.path.insert(0, str(Path(args.bmbe_root)))
    if args.column == "htsim":
        if not args.meddxagent_root:
            sys.exit("--column htsim requires --meddxagent-root")
        sys.path.insert(0, str(Path(args.meddxagent_root)))
        os.environ.setdefault("MEDDXAGENT_REPO", str(Path(args.meddxagent_root)))

    diseases = _load_diseases(args.column, args.diseases)
    print(f"{args.column}: {len(diseases)} diseases in the closed list", flush=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    mapper_client = None
    if args.column in ("patientsim", "agentclinic"):
        from openai import OpenAI
        mapper_client = OpenAI(api_key=api_key)
        print(f"Ontology mapper model: {args.mapper_model}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.posthoc_from_jsonl:
        if args.column != "agentclinic":
            sys.exit("--posthoc-from-jsonl is only supported for --column agentclinic")
        rows = [json.loads(l) for l in open(args.posthoc_from_jsonl)]
        print(f"Posthoc mode: reprocessing {len(rows)} rows from {args.posthoc_from_jsonl}", flush=True)
        with open(args.out, "w") as out_f:
            for i, row in enumerate(rows):
                t0 = time.time()
                res = posthoc_recompute(row, diseases, mapper_client, args.mapper_model)
                res["elapsed_seconds"] = time.time() - t0
                out_f.write(json.dumps(res) + "\n"); out_f.flush()
                print(f"  [{i+1}/{len(rows)}] gt={(res.get('ground_truth') or '')[:30]:<30} "
                      f"pred={res.get('predicted')!r} hit={res.get('top1_hit')} "
                      f"abst={res.get('abstained')} reason={res.get('stop_reason')}", flush=True)
        print(f"\nDone. Output: {args.out}", flush=True)
        return

    if not args.cod_repo or not args.model_dir or not args.cases:
        sys.exit("Live mode requires --cod-repo, --model-dir and --cases.")
    # Search order: (meddxagent,) this repo, then the CoD clone.
    sys.path.insert(0, str(Path(args.cod_repo)))
    sys.path.insert(0, str(Path(args.bmbe_root)))
    if args.column == "htsim":
        sys.path.insert(0, str(Path(args.meddxagent_root)))
    from cod_cli import DiagnosisChatbot

    print(f"Loading DiagnosisGPT from {args.model_dir} ...", flush=True)
    t0 = time.time()
    chatbot = DiagnosisChatbot(args.model_dir, confidence_threshold=args.confidence_threshold)
    print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)

    cases = _load_cases(args.column, args.cases, args.num_cases)
    with open(args.out, "w") as out_f:
        for i, case in enumerate(cases):
            print(f"[{i+1}/{len(cases)}] ...", flush=True)
            t0 = time.time()
            if args.column == "patientsim":
                result = run_patientsim_case(case, chatbot, args.patient_model, api_key,
                                             args.max_inquiries, diseases,
                                             mapper_client, args.mapper_model)
            elif args.column == "htsim":
                result = run_htsim_case(case, chatbot, args.patient_model,
                                        args.max_inquiries, diseases)
                result["case_idx"] = i
            else:
                result = run_agentclinic_case(case, chatbot, args.patient_model,
                                              args.max_inquiries, diseases,
                                              mapper_client, args.mapper_model)
                result["case_idx"] = i
            result["elapsed_seconds"] = time.time() - t0
            out_f.write(json.dumps(result) + "\n"); out_f.flush()
            print(f"  pred={result.get('predicted')!r} | gt={result.get('ground_truth')!r} | "
                  f"turns={result.get('turns')} | {result['elapsed_seconds']:.1f}s", flush=True)
    print(f"\nDone. Output: {args.out}", flush=True)


if __name__ == "__main__":
    main()
