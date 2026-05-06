"""Run DiagnosisGPT-34B (Chain-of-Diagnosis) on the BMBE 50-patient DDxPlus set.

Architecture (from CoD's cod_cli.py):
- DiagnosisGPT runs as a chatbot. First turn: patient describes chief complaint.
- Model alternates: ask follow-up symptom questions OR commit to diagnosis when
  internal confidence (over 9,604-disease retrieval candidates) > threshold.
- Output ends with a "Diagnosis: <disease>" line + a confidence_distribution
  JSON over the top retrieved diseases.

Apples-to-apples integration:
- CoD asks symptom questions; we route them to OUR PatientSim (so the patient
  side matches our other baselines: closed-world DDxPlus rules + persona).
- CoD's retriever uses its own 9,604-disease English database. We don't restrict
  it to our 49 DDxPlus diseases — we let CoD predict freely, then map predictions
  to the 49-space via name matching (or LLM-based mapping if needed).

CoD library is imported from a local clone. Model + retriever assets live on
clariden scratch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List


def _setup_imports(cod_repo: Path, bmbe_root: Path) -> None:
    """Make CoD's cod_cli + our llm_interface importable."""
    sys.path.insert(0, str(cod_repo))
    sys.path.insert(0, str(bmbe_root))


def build_chief_complaint(profile: Dict) -> str:
    """First user turn — what the patient tells the doctor up front."""
    age = profile.get("age", "?")
    gender = profile.get("gender", "?")
    cv = profile.get("Current Visit", {}) or {}
    chief = cv.get("chiefcomplaint", "I am not feeling well")
    return f"I am a {age}-year-old {gender}. {chief}"


def make_patient_callable(bmbe_profile: Dict, patient_model: str, api_key: str | None):
    """Return a callable(question) -> answer that uses our PatientSim with
    DDxPlus closed-world rules. Mirrors what BMBEPatientSim does for MediQ
    and BMBEPatientAgent does for MEDDxAgent."""
    from llm_interface.prompt_templates import build_patient_system_prompt
    from llm_interface.llm_client import OpenAILLMClient, sanitize_openai_text

    system_prompt = build_patient_system_prompt(bmbe_profile)
    messages = [{"role": "system", "content": system_prompt}]
    llm = OpenAILLMClient(model=patient_model, api_key=api_key, temperature=0.7)

    def respond(question: str) -> str:
        messages.append({"role": "user", "content": sanitize_openai_text(question)})
        answer, _usage = llm.chat(messages, max_tokens=200)
        messages.append({"role": "assistant", "content": answer})
        return answer

    return respond, lambda: dict(llm.total_usage)


def looks_like_question(model_text: str) -> bool:
    """CoD inserts 'Inadequate for diagnosis. Ask for symptoms:' before its
    follow-up question, or it asks naturally. Detect by either marker or '?'.
    """
    if "Inadequate for diagnosis" in model_text:
        return True
    if "Ask for symptoms" in model_text:
        return True
    # Heuristic: if it contains '?' and does NOT contain 'Diagnosis:' then it's a question
    if "Diagnosis:" not in model_text and "?" in model_text:
        return True
    return False


def parse_final_diagnosis(model_text: str) -> str:
    """Extract diagnosis name after '## Diagnosis:' header."""
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


def parse_confidence_distribution(model_text: str) -> Dict[str, float]:
    """Pull the JSON dict that follows '## Diagnostic confidence:'."""
    import re
    m = re.search(r"##\s*Diagnostic confidence:\s*(\{[^}]*\})", model_text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except Exception:
        return {}


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


def map_to_ddxplus(prediction: str, dist: Dict[str, float], ddxplus_diseases: List[str],
                    llm_client=None, llm_model: str = "gpt-5.4-nano") -> List[str]:
    """LLM-based semantic mapping from CoD free-text → DDxPlus 49-disease set.

    Substring matching is brittle: CoD outputs ("Esophageal Spasm", verbose
    differentials) rarely substring-match the canonical 49-name DDxPlus set,
    and an alphabetical fallback would silently corrupt every unmapped case
    (top-1 = first disease alphabetically). One ~80-token gpt-5.4-nano call
    per case asks which of the 49 best matches, with "Unknown" as a safe
    fallback for genuinely off-list predictions.

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


def run_one_patient(profile: Dict, chatbot, patient_model: str, api_key: str | None,
                    max_inquiries: int = 10, ddxplus_diseases: List[str] = None,
                    mapper_client=None, mapper_model: str = "gpt-5.4-nano") -> Dict:
    """One CoD diagnostic session for a single patient."""
    chief = build_chief_complaint(profile)
    respond, get_usage = make_patient_callable(profile, patient_model, api_key)

    history: list = []
    dialog: list = []
    dialog.append({"role": "patient", "content": chief, "turn": 0})

    # First turn: chief complaint
    cur_query = chief
    final_text = ""
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
            diagnosis_text = parse_final_diagnosis(generated)
            final_text = generated
            break

        # Otherwise model is asking a follow-up question
        if asked >= max_inquiries:
            # Force final answer by not feeding more responses
            break

        # Extract the question CoD asked (everything after "Ask for symptoms:" if present, else last sentence)
        if "Ask for symptoms:" in generated:
            question = generated.split("Ask for symptoms:", 1)[1].strip()
        else:
            # Heuristic: last line containing '?'
            question_lines = [ln for ln in generated.split("\n") if "?" in ln]
            question = question_lines[-1].strip() if question_lines else generated.strip()

        patient_response = respond(question)
        dialog.append({"role": "patient", "content": patient_response, "turn": step + 1})
        cur_query = patient_response
        asked += 1

    ranked = map_to_ddxplus(diagnosis_text, confidence, ddxplus_diseases or [],
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cod-repo", required=True,
                    help="Path to FreedomIntelligence/Chain-of-Diagnosis clone")
    ap.add_argument("--model-dir", required=True,
                    help="Path to DiagnosisGPT-34B local checkout")
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--bmbe-root", required=True,
                    help="Path to MoBayes/hybrid-diagnostic-system (for llm_interface)")
    ap.add_argument("--diseases", required=True,
                    help="Path to release_conditions.json for the 49-disease DDxPlus list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patient-model", default="gpt-5.4-nano")
    ap.add_argument("--num-patients", type=int, default=50)
    ap.add_argument("--confidence-threshold", type=float, default=0.5)
    ap.add_argument("--max-inquiries", type=int, default=10)
    ap.add_argument("--mapper-model", default="gpt-5.4-nano",
                    help="OpenAI model for inline ontology mapper (free-text → 49 closed set)")
    args = ap.parse_args()

    _setup_imports(Path(args.cod_repo), Path(args.bmbe_root))

    from cod_cli import DiagnosisChatbot

    print(f"Loading DiagnosisGPT from {args.model_dir} ...", flush=True)
    t0 = time.time()
    chatbot = DiagnosisChatbot(args.model_dir, confidence_threshold=args.confidence_threshold)
    print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)

    profiles = json.load(open(args.profiles))[: args.num_patients]
    diseases = sorted(set(json.load(open(args.diseases)).keys()))
    print(f"DDxPlus diseases: {len(diseases)}", flush=True)

    api_key = os.environ.get("OPENAI_API_KEY")

    # OpenAI client for inline ontology mapper (free-text → DDxPlus 49)
    from openai import OpenAI
    mapper_client = OpenAI(api_key=api_key)
    print(f"Ontology mapper model: {args.mapper_model}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "w")

    for i, profile in enumerate(profiles):
        print(f"[{i+1}/{len(profiles)}] {profile.get('hadm_id')} ...", flush=True)
        t0 = time.time()
        result = run_one_patient(
            profile, chatbot,
            patient_model=args.patient_model,
            api_key=api_key,
            max_inquiries=args.max_inquiries,
            ddxplus_diseases=diseases,
            mapper_client=mapper_client,
            mapper_model=args.mapper_model,
        )
        result["elapsed_seconds"] = time.time() - t0
        out_f.write(json.dumps(result) + "\n")
        out_f.flush()
        print(f"  pred={result.get('predicted')!r} | gt={result.get('ground_truth')!r} | turns={result.get('turns')} | {result['elapsed_seconds']:.1f}s", flush=True)

    out_f.close()
    print(f"\nDone. Output: {args.out}", flush=True)


if __name__ == "__main__":
    main()
