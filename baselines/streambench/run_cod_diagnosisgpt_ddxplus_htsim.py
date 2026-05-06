"""DDxPlus HT-sim runner for CoD / DiagnosisGPT-34B (open-vocab outputs mapped to DDxPlus 49 via LLM remap)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List


def _setup_imports(cod_repo: Path, bmbe_root: Path, meddxagent_root: Path) -> None:
    sys.path.insert(0, str(cod_repo))
    sys.path.insert(0, str(bmbe_root))
    sys.path.insert(0, str(meddxagent_root))
    os.environ.setdefault("MEDDXAGENT_REPO", str(meddxagent_root))


def map_to_ddxplus(prediction: str, dist: Dict[str, float], ddx_diseases: List[str]) -> List[str]:
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


def parse_final_diagnosis(text: str) -> str:
    """Extract the diagnosis label from a 'Diagnosis: <stuff>' line.

    CoD sometimes emits a long sentence ('Diagnosis: It seems that you may
    have ...'). We capture up to first newline, but downstream mapping does
    substring search against the 49-disease space, so a verbose capture is
    fine — just don't truncate too aggressively.
    """
    import re
    m = re.search(r"Diagnosis:\s*(.+?)(?:\n|$)", text)
    return m.group(1).strip() if m else ""


def run_one_case(case: Dict, chatbot, patient_model: str,
                 max_inquiries: int, ddx_diseases: List[str]) -> Dict:
    from baselines.streambench.adapters.base import LLMPatientWrapper

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
            diagnosis_text = parse_final_diagnosis(generated)
            break

        question = generated.strip().split("\n")[0]
        ans = wrapper.respond(question)
        dialog.append({"role": "patient", "content": ans, "turn": step + 1})
        cur_query = ans
        asked += 1

    if not diagnosis_text and confidence:
        diagnosis_text = max(confidence.items(), key=lambda kv: kv[1])[0]

    ranked = map_to_ddxplus(diagnosis_text, confidence, ddx_diseases)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cod-repo", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--bmbe-root", required=True)
    ap.add_argument("--meddxagent-root", required=True,
                    help="Path to local meddxagent checkout (provides ddxdriver)")
    ap.add_argument("--cases", required=True,
                    help="data/cases/ddxplus_htsim_a/selected_cases.jsonl")
    ap.add_argument("--diseases", required=True,
                    help="data/kbs/ddxplus/diseases.csv (column 'name')")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patient-model", default="gpt-5.4-nano")
    ap.add_argument("--num-cases", type=int, default=50)
    ap.add_argument("--confidence-threshold", type=float, default=0.5)
    ap.add_argument("--max-inquiries", type=int, default=15)
    args = ap.parse_args()

    _setup_imports(Path(args.cod_repo), Path(args.bmbe_root),
                    Path(args.meddxagent_root))
    from cod_cli import DiagnosisChatbot

    print(f"Loading DiagnosisGPT from {args.model_dir} ...", flush=True)
    t0 = time.time()
    chatbot = DiagnosisChatbot(args.model_dir, confidence_threshold=args.confidence_threshold)
    print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)

    cases = [json.loads(l) for l in open(args.cases)][: args.num_cases]
    import pandas as pd
    ddx_diseases = pd.read_csv(args.diseases)["name"].tolist()
    print(f"DDxPlus KB diseases: {len(ddx_diseases)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "w")

    for i, case in enumerate(cases):
        gt = case["PATHOLOGY"]
        print(f"[{i+1}/{len(cases)}] gt={gt[:30]} ...", flush=True)
        t0 = time.time()
        result = run_one_case(case, chatbot, args.patient_model,
                               args.max_inquiries, ddx_diseases)
        result["case_idx"] = i
        result["elapsed_seconds"] = time.time() - t0
        out_f.write(json.dumps(result) + "\n"); out_f.flush()
        print(f"  pred={result.get('predicted')!r} hit={result.get('top1_hit')} "
              f"turns={result.get('turns')} {result['elapsed_seconds']:.1f}s", flush=True)

    out_f.close()
    print(f"\nDone. Output: {args.out}", flush=True)


if __name__ == "__main__":
    main()
