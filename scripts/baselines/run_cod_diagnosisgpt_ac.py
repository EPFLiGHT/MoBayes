"""Run DiagnosisGPT-34B (Chain-of-Diagnosis) on AgentClinic-MedQA cases.

Sister of `run_cod_diagnosisgpt.py` (DDxPlus version). The chatbot loop is
identical; only the patient side changes:
    DDxPlus  → BMBE PatientSim       (closed-world DDxPlus rules + persona)
    AC-MedQA → AgentClinic PatientAgent  (paper-faithful AC simulator,
                                            via baselines/agentclinic/adapters/cod.py)

CoD's free-text final diagnosis is mapped to the AC-50 KB's 31-disease space
via name matching against its confidence_distribution (same logic as the
DDxPlus runner — only the disease list changes).

Hardware: same as the DDxPlus run — GPU node, DiagnosisGPT-34B local checkout
(~70GB), CoD repo clone for `cod_cli.DiagnosisChatbot`.

Run:
    OPENAI_API_KEY=... python3 scripts/baselines/run_cod_diagnosisgpt_ac.py \\
        --cod-repo /path/to/Chain-of-Diagnosis \\
        --model-dir /path/to/DiagnosisGPT-34B \\
        --bmbe-root hybrid-diagnostic-system \\
        --cases data/cases/agentclinic_a/selected_cases.jsonl \\
        --diseases data/kbs/agentclinic_a/diseases.csv \\
        --out data/benchmark_results/agentclinic_10_cod/results.jsonl \\
        --num-cases 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make local dir importable for the shared CoD helper
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path: sys.path.insert(0, str(_HERE))
from typing import Dict, List


def _setup_imports(cod_repo: Path, bmbe_root: Path) -> None:
    sys.path.insert(0, str(cod_repo))
    sys.path.insert(0, str(bmbe_root))


# Shared CoD post-loop helpers (see _cod_force_final.py)
from _cod_force_final import (
    map_to_closed as map_to_ac,
    force_final_via_llm,
    finalise as _cod_finalise,
    posthoc_recompute as _shared_posthoc,
)


def parse_final_diagnosis(text: str) -> str:
    import re
    m = re.search(r"Diagnosis:\s*(.+?)(?:\n|$)", text)
    return m.group(1).strip() if m else ""


def parse_confidence(text: str) -> Dict[str, float]:
    import re
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m: return {}
    try:
        return json.loads(m.group(1))
    except Exception:
        return {}


def run_one_case(ac_case: Dict, chatbot, patient_model: str,
                 max_inquiries: int, ac_diseases: List[str],
                 mapper_client=None, mapper_model: str = "gpt-5.4-nano") -> Dict:
    from baselines.agentclinic.adapters.cod import make_ac_cod_patient_callable
    respond, get_usage, gt, initial_info = make_ac_cod_patient_callable(
        ac_case, patient_model=patient_model,
    )

    # Chief complaint = AC's initial_info (age + chief complaint)
    chief = initial_info or "I'm not feeling well."
    history, dialog = [], []
    dialog.append({"role": "patient", "content": chief, "turn": 0})
    cur_query = chief
    final_text = ""
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
        if conf_dist: confidence = conf_dist

        if "Diagnosis:" in generated and "Inadequate" not in generated:
            diagnosis_text = parse_final_diagnosis(generated)
            final_text = generated
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
        ranked = map_to_ac(diagnosis_text, confidence, ac_diseases,
                            llm_client=mapper_client, llm_model=mapper_model)
        abstained = not ranked
    else:
        # CoD didn't commit AND force-final returned Abstain (or no dialog)
        # -> fall back to top-confidence entry; if even that fails to map, abstain.
        if confidence:
            diagnosis_text = max(confidence.items(), key=lambda kv: kv[1])[0]
            ranked = map_to_ac(diagnosis_text, confidence, ac_diseases,
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


def posthoc_recompute(saved, ac_diseases, mapper_client, mapper_model='gpt-5.4-nano'):
    """Thin wrapper: delegate to _cod_force_final.posthoc_recompute with AC field names."""
    return _shared_posthoc(saved, ac_diseases, mapper_client, mapper_model,
                           dialog_field='dialog',
                           diag_text_field='diagnosis_text',
                           conf_field='confidence_distribution')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cod-repo", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--bmbe-root", required=True)
    ap.add_argument("--cases", default=None,
                    help="data/cases/agentclinic_a/selected_cases.jsonl (not needed in posthoc mode)")
    ap.add_argument("--diseases", required=True,
                    help="data/kbs/agentclinic_a/diseases.csv (column 'name')")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patient-model", default="gpt-5.4-nano")
    ap.add_argument("--num-cases", type=int, default=10)
    ap.add_argument("--confidence-threshold", type=float, default=0.5)
    ap.add_argument("--max-inquiries", type=int, default=10)
    ap.add_argument("--mapper-model", default="gpt-5.4-nano",
                    help="OpenAI model for the inline ontology mapper and force-final step")
    ap.add_argument("--posthoc-from-jsonl", default=None,
                    help="If set, skip CoD inference and reprocess the saved dialogs in this jsonl "
                         "(adds the force-final + map step retroactively).")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.bmbe_root)))
    import pandas as pd
    ac_diseases = pd.read_csv(args.diseases)["name"].tolist()
    print(f"AC-50 KB diseases: {len(ac_diseases)}", flush=True)

    from openai import OpenAI
    mapper_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    print(f"Ontology mapper / force-final model: {args.mapper_model}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # ── Posthoc mode: no CoD inference; reprocess saved dialogs ──
    if args.posthoc_from_jsonl:
        rows = [json.loads(l) for l in open(args.posthoc_from_jsonl)]
        print(f"Posthoc mode: reprocessing {len(rows)} rows from {args.posthoc_from_jsonl}",
              flush=True)
        out_f = open(args.out, "w")
        for i, row in enumerate(rows):
            t0 = time.time()
            res = posthoc_recompute(row, ac_diseases, mapper_client, args.mapper_model)
            res["elapsed_seconds"] = time.time() - t0
            out_f.write(json.dumps(res) + "\n"); out_f.flush()
            print(f"  [{i+1}/{len(rows)}] gt={(res.get('ground_truth') or '')[:30]:<30} "
                  f"pred={res.get('predicted')!r} hit={res.get('top1_hit')} "
                  f"abst={res.get('abstained')} reason={res.get('stop_reason')}", flush=True)
        out_f.close()
        print(f"\nDone. Output: {args.out}", flush=True)
        return

    # ── Live inference mode (requires GPU model) ──
    if not args.cod_repo or not args.model_dir or not args.cases:
        sys.exit("Live mode requires --cod-repo, --model-dir, and --cases. "
                 "Use --posthoc-from-jsonl to reprocess a saved jsonl instead.")

    _setup_imports(Path(args.cod_repo), Path(args.bmbe_root))
    from cod_cli import DiagnosisChatbot

    print(f"Loading DiagnosisGPT from {args.model_dir} ...", flush=True)
    t0 = time.time()
    chatbot = DiagnosisChatbot(args.model_dir, confidence_threshold=args.confidence_threshold)
    print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)

    cases = [json.loads(l) for l in open(args.cases)][:args.num_cases]

    out_f = open(args.out, "w")

    for i, case in enumerate(cases):
        gt = case.get("OSCE_Examination", case)["Correct_Diagnosis"]
        print(f"[{i+1}/{len(cases)}] gt={gt[:30]} ...", flush=True)
        t0 = time.time()
        result = run_one_case(case, chatbot, args.patient_model,
                              args.max_inquiries, ac_diseases,
                              mapper_client=mapper_client,
                              mapper_model=args.mapper_model)
        result["case_idx"] = i
        result["elapsed_seconds"] = time.time() - t0
        out_f.write(json.dumps(result) + "\n"); out_f.flush()
        print(f"  pred={result.get('predicted')!r} hit={result.get('top1_hit')} "
              f"turns={result.get('turns')} {result['elapsed_seconds']:.1f}s", flush=True)

    out_f.close()
    print(f"\nDone. Output: {args.out}", flush=True)


if __name__ == "__main__":
    main()
