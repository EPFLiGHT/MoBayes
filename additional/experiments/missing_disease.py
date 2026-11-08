"""Missing-disease experiment (appendix C.15), live run.

For each DDxPlus PatientSim case (plain persona) the true disease is removed
from the candidate space and the full dialogue is rerun.

  mobayes    : the true disease id is excluded from the KB at load time (via
               data_loader.EXCLUDED_DISEASE_IDS), so it is absent from the
               priors, likelihoods and EIG candidates. Full turn budget, no
               early commit; abstention is decided at the end by comparing
               the final max-posterior with tau.
  standalone : ClosedWorldDoctorAgent with the same reduced disease list and
               an explicit instruction to answer [ABSTAIN] if none matches.
               Outcomes per case:
                 abstain             : correct behaviour
                 hallucinated_inlist : commit to an in-list disease (wrong)
                 named_true_offlist  : names the excluded true disease
                 offlist_other       : commits to another off-list disease

Usage (from this folder, OPENAI_API_KEY set):
    PYTHONPATH=. python3 experiments/missing_disease.py --arm both --num-cases 50 --out-suffix rerun
Existing case files in the output folder are skipped, so a rerun into the
same folder only reprints the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

PROFILES = REPO / "data/cases/ddxplus_patientsim_a/plain/profiles.json"
KB_DIR = REPO / "data/kbs/ddxplus"
OUT_ROOT = REPO / "data/benchmark_results"

GREETING = "Hello, I'm your doctor today. What brings you in today?"
FORCE_FINAL = "Please provide your final [DDX] or [ABSTAIN] now."
REMAP_MODEL = "gpt-5.4-nano"


# ---------------------------------------------------------------- data loading

def load_cases() -> List[Dict]:
    with PROFILES.open() as f:
        return json.load(f)


def gt_of(case: Dict) -> str:
    meta = case.get("_ddxplus_meta", {}) or {}
    return (meta.get("diagnosis_name")
            or case.get("Current Visit", {}).get("diagnosis", "Unknown"))


def load_disease_tables() -> Tuple[Dict[str, str], List[str]]:
    """Returns (name_lower -> disease_id, leaf-only universe name list).

    Universe is the DDxPlus leaf diagnoses ONLY (49); parent/category
    taxonomy nodes (e.g. "Respiratory Diseases") and d_other are excluded.
    This deliberately differs from bench.columns.patientsim.disease_universe(),
    which dumps every diseases.csv name including the 8 category headers: here
    the standalone closed-world prompt must see exactly the same candidate set
    MoBayes reasons over (its posterior is defined over the 49 leaves).
    """
    name_to_id: Dict[str, str] = {}
    universe: List[str] = []
    with (KB_DIR / "diseases.csv").open() as f:
        rows = list(csv.DictReader(f))
    parents = {r["parent_id"] for r in rows if r.get("parent_id")}
    for r in rows:
        if r["disease_id"] in parents or r["disease_id"] == "d_other":
            continue
        universe.append(r["name"])
        name_to_id[r["name"].lower().strip()] = r["disease_id"]
    return name_to_id, universe


def make_patient(profile: Dict, patient_model: str):
    from scripts.core.benchmark_patientsim import SimplePatientAgent
    from config import MoBayesConfig
    sim = SimplePatientAgent(profile=profile, api_key=os.environ.get("OPENAI_API_KEY"),
                             model=patient_model, config=MoBayesConfig())
    chief = (profile.get("Current Visit", {}).get("chiefcomplaint")
             or "presenting complaint")
    sim.initial_info = (f"Age: {profile.get('age', '?')}\n"
                        f"Sex: {profile.get('gender', '?')}\n"
                        f"Chief Complaint: {chief}")
    return sim


# ---------------------------------------------------------------- mobayes arm

def run_mobayes_case(case: Dict, case_idx: int, gt: str, gt_id: str, *,
                     doctor_model: str, patient_model: str, max_turns: int,
                     api_key: str, tau: float, prior_mode: str = "empirical") -> Dict:
    from scripts.core.benchmark_patientsim import MoBayesDoctor
    from config import MoBayesConfig
    from world_model import StoppingRules
    from world_model.knowledge import data_loader

    patient = make_patient(case, patient_model)

    data_loader.EXCLUDED_DISEASE_IDS.add(gt_id)
    try:
        doctor = MoBayesDoctor(
            data_dir=str(KB_DIR),
            llm_model=doctor_model,
            api_key=api_key,
            max_questions=max_turns,
            prior_mode=prior_mode,
            config=MoBayesConfig(),
        )
        doctor.reset({})
    finally:
        data_loader.EXCLUDED_DISEASE_IDS.discard(gt_id)

    assert gt_id not in doctor.leaf_diseases, (
        f"exclusion failed: {gt_id} still in candidate space")

    # Paper protocol: no early commit, abstention decided post-hoc at tau.
    doctor.stopping = StoppingRules(
        confidence_threshold=1.01,
        max_questions=max_turns,
        min_questions_before_abstain=999,
    )

    log: Dict = {
        "case_idx": case_idx, "arm": "mobayes",
        "ground_truth": gt, "excluded_disease_id": gt_id,
        "n_candidates": len(doctor.leaf_diseases),
        "turns": [], "final": {},
    }

    opening_q = ("Hello, what brings you in today? Could you tell me about "
                 "yourself and what's been going on?")
    opening_a = patient.respond(opening_q)
    bulk = doctor.bulk_extract_from_text(opening_a)
    log["turns"].append({"turn": 0, "kind": "opening",
                         "patient_opening": opening_a,
                         "extracted_features": bulk})

    for turn in range(max_turns):
        qr = doctor.get_next_question()
        if qr["status"] == "stopped":
            log["stop_reason"] = qr.get("reason", "STOPPED")
            break
        ans = patient.respond(qr["question"])
        extracted, conf = doctor.process_response(
            qr["feature_id"], ans, qr["schema_values"])
        log["turns"].append({
            "turn": turn + 1, "kind": "qa",
            "feature_id": qr["feature_id"],
            "doctor_question": qr["question"],
            "patient_answer": ans,
            "extracted_value": extracted,
            "extraction_confidence": float(conf),
        })

    results = doctor.get_results()
    confidence = float(results["confidence"])
    abstained = confidence < tau
    log["final"] = {
        "predicted": "ABSTAIN" if abstained else results["top_disease"],
        "top_disease_if_forced": results["top_disease"],
        "predicted_top3": list(results["top_diseases"][:3]),
        "confidence": confidence,
        "final_entropy": float(results["entropy"]),
        "tau": tau,
        "abstained": bool(abstained),
        "outcome": "abstain" if abstained else "hallucinated_inlist",
    }
    log["token_usage"] = {
        "doctor": getattr(getattr(doctor, "llm", None), "total_usage", {}).copy() or {},
        "patient": getattr(patient, "token_usage", {}).copy() or {},
    }
    return log


# ------------------------------------------------------------- standalone arm

def build_strict_cw_prompt(max_turns: int, disease_names: List[str]) -> str:
    from llm_interface.prompt_templates import build_standalone_doctor_prompt
    base = build_standalone_doctor_prompt(max_turns)
    disease_list = "\n".join(f"  - {n}" for n in sorted(set(disease_names)))
    addendum = (
        "\n\nThe knowledge base includes the following disease list:\n"
        f"{disease_list}\n\n"
        "Your final [DDX] must use disease names from this list. If you "
        "believe the patient's condition does not match any disease in this "
        "list, do not guess: respond with [ABSTAIN]."
    )
    return base + addendum


def normalize_to_list(predictions: List[str], disease_universe: List[str],
                      api_key: str) -> List[str]:
    """Map free-text predictions onto the closed disease list (same remap as the benchmark runner)."""
    if not predictions or not disease_universe:
        return predictions
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or api_key)
        pred_list = "\n".join(f"{i+1}. {p}" for i, p in enumerate(predictions[:5]))
        name_list = "\n".join(f"- {n}" for n in sorted(disease_universe))
        prompt = (
            "Match each diagnosis below to the closest disease from the reference list.\n"
            "If a diagnosis does not match any disease in the list, write NO_MATCH.\n\n"
            f"DIAGNOSES:\n{pred_list}\n\n"
            f"REFERENCE DISEASE LIST:\n{name_list}\n\n"
            "Output ONLY a numbered list with the matched disease name "
            "(exactly as written in the reference list) or NO_MATCH.\n"
            "Example format:\n1. Pneumonia\n2. NO_MATCH\n3. Gastroenteritis"
        )
        resp = client.chat.completions.create(
            model=REMAP_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        lookup = {n.lower(): n for n in disease_universe}
        matched: List[str] = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(".", 1)
            name = parts[1].strip() if len(parts) == 2 else line
            if name.upper() == "NO_MATCH":
                idx = len(matched)
                matched.append(predictions[idx] if idx < len(predictions) else "Unknown")
            elif name.lower() in lookup:
                matched.append(lookup[name.lower()])
            else:
                idx = len(matched)
                matched.append(predictions[idx] if idx < len(predictions) else name)
        while len(matched) < len(predictions):
            matched.append(predictions[len(matched)])
        return matched[: len(predictions)]
    except Exception:
        return predictions


def run_standalone_case(case: Dict, case_idx: int, gt: str, gt_id: str, *,
                        doctor_model: str, patient_model: str, max_turns: int,
                        api_key: str, universe: List[str]) -> Dict:
    from baselines.prompting.doctors import ClosedWorldDoctorAgent
    from config import MoBayesConfig

    reduced = [n for n in universe if n.lower().strip() != gt.lower().strip()]
    assert len(reduced) == len(universe) - 1, (
        f"expected exactly one removal for GT={gt!r}")

    patient = make_patient(case, patient_model)
    doctor = ClosedWorldDoctorAgent(
        disease_names=reduced, model=doctor_model, api_key=api_key,
        max_turns=max_turns, config=MoBayesConfig(),
    )
    # Swap in the strict prompt (must pick from list or [ABSTAIN]).
    doctor.messages[0]["content"] = build_strict_cw_prompt(max_turns, reduced)

    log: Dict = {
        "case_idx": case_idx, "arm": "standalone",
        "ground_truth": gt, "excluded_disease_id": gt_id,
        "n_candidates_in_prompt": len(reduced),
        "turns": [], "final": {},
    }

    pa = patient.respond(GREETING)
    log["turns"].append({"role": "doctor", "content": GREETING, "turn": 0})
    log["turns"].append({"role": "patient", "content": pa, "turn": 0})

    last_msg = ""
    for turn in range(max_turns + 5):
        text, is_final = doctor.get_question(pa)
        log["turns"].append({"role": "doctor", "content": text, "turn": turn + 1})
        last_msg = text
        if is_final:
            break
        pa = patient.respond(text)
        log["turns"].append({"role": "patient", "content": pa, "turn": turn + 1})

    abstained = "[ABSTAIN]" in last_msg
    if "[DDX]" in last_msg:
        preds = doctor.parse_ddx(last_msg)
    elif abstained:
        preds = doctor.parse_possible(last_msg)
    else:
        ddx_msg, _ = doctor.get_question(FORCE_FINAL)
        log["turns"].append({"role": "doctor", "content": ddx_msg, "turn": -1})
        if "[DDX]" in ddx_msg:
            preds = doctor.parse_ddx(ddx_msg)
        else:
            abstained = True
            preds = doctor.parse_possible(ddx_msg)

    raw_preds = list(preds)
    norm_preds = normalize_to_list(preds, reduced, api_key) if preds else []

    reduced_lower = {n.lower().strip() for n in reduced}
    gt_lower = gt.lower().strip()
    top1_raw = (raw_preds[0] if raw_preds else "").lower().strip()
    top1_norm = (norm_preds[0] if norm_preds else "").lower().strip()

    named_true = bool(top1_raw) and (gt_lower in top1_raw or top1_raw in gt_lower)
    if abstained:
        outcome = "abstain"
    elif named_true:
        outcome = "named_true_offlist"
    elif top1_norm in reduced_lower:
        outcome = "hallucinated_inlist"
    else:
        outcome = "offlist_other"

    log["final"] = {
        "predicted": "ABSTAIN" if abstained else (norm_preds[0] if norm_preds else "Unknown"),
        "predicted_set": norm_preds[:5],
        "predicted_raw": raw_preds[:5],
        "abstained": bool(abstained),
        "true_disease_in_raw_preds": any(
            gt_lower in p.lower() or p.lower().strip() in gt_lower
            for p in raw_preds[:5]),
        "outcome": outcome,
    }
    log["token_usage"] = {
        "doctor": dict(getattr(doctor, "token_usage", {})),
        "patient": getattr(patient, "token_usage", {}).copy() or {},
    }
    return log


# ------------------------------------------------------------------ orchestration

def out_dir_for(arm: str, suffix: str) -> Path:
    sfx = f"_{suffix}" if suffix else ""
    return OUT_ROOT / f"missing_disease_{arm}{sfx}"


def run_arm(arm: str, args, cases: List[Dict], name_to_id: Dict[str, str],
            universe: List[str], api_key: str) -> None:
    out_dir = out_dir_for(arm, args.out_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "arm": arm, "n_cases": args.num_cases, "start_idx": args.start_idx,
        "max_turns": args.max_turns, "tau": args.tau,
        "doctor_model": args.doctor_model, "patient_model": args.patient_model,
        "profiles": str(PROFILES.relative_to(REPO)), "kb": str(KB_DIR.relative_to(REPO)),
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    todo = cases[args.start_idx:args.num_cases]
    print(f"[{arm}] {len(todo)} cases | max_turns={args.max_turns} "
          f"| doctor={args.doctor_model} | out={out_dir.name}", flush=True)

    for offset, case in enumerate(todo):
        idx = args.start_idx + offset
        out_path = out_dir / f"case_{idx:03d}.json"
        if out_path.exists():
            print(f"  [{offset+1}/{len(todo)}] (idx={idx}) skip (exists)", flush=True)
            continue
        gt = gt_of(case)
        gt_id = name_to_id.get(gt.lower().strip())
        if gt_id is None:
            print(f"  [{offset+1}/{len(todo)}] (idx={idx}) SKIP: GT {gt!r} "
                  "not a KB leaf", flush=True)
            continue
        ts = time.time()
        try:
            if arm == "mobayes":
                log = run_mobayes_case(
                    case, idx, gt, gt_id,
                    doctor_model=args.doctor_model, patient_model=args.patient_model,
                    max_turns=args.max_turns, api_key=api_key, tau=args.tau,
                    prior_mode=args.prior_mode)
            else:
                log = run_standalone_case(
                    case, idx, gt, gt_id,
                    doctor_model=args.doctor_model, patient_model=args.patient_model,
                    max_turns=args.max_turns, api_key=api_key, universe=universe)
            with out_path.open("w") as f:
                json.dump(log, f, indent=2, ensure_ascii=False, default=str)
            fin = log["final"]
            print(f"  [{offset+1}/{len(todo)}] (idx={idx}) {fin['outcome']:<20} "
                  f"GT={gt[:28]:<28} → {str(fin['predicted'])[:28]:<28} "
                  f"({time.time()-ts:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{offset+1}/{len(todo)}] (idx={idx}) ERROR: {e}", flush=True)


def summarize(arm: str, suffix: str, tau: float) -> None:
    out_dir = out_dir_for(arm, suffix)
    files = sorted(out_dir.glob("case_*.json"))
    if not files:
        print(f"[{arm}] no results in {out_dir}")
        return
    outcomes: Dict[str, int] = {}
    confidences: List[float] = []
    for p in files:
        with p.open() as f:
            c = json.load(f)
        outcomes[c["final"]["outcome"]] = outcomes.get(c["final"]["outcome"], 0) + 1
        if "confidence" in c["final"]:
            confidences.append(float(c["final"]["confidence"]))
    n = len(files)
    print(f"\n=== {arm} (n={n}, dir={out_dir.name}) ===")
    for k in sorted(outcomes):
        print(f"  {k:<22} {outcomes[k]:>3}  ({100.0*outcomes[k]/n:.1f}%)")
    if confidences:
        arr = np.array(confidences)
        print(f"  mean final max-posterior: {arr.mean():.3f}")
        sweep = {t: float((arr < t).mean()) for t in
                 (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)}
        print("  abstention rate by tau : "
              + "  ".join(f"τ={t}: {100*v:.0f}%" for t, v in sweep.items()))
        print(f"  (paper tau={tau}: {100.0*float((arr < tau).mean()):.1f}% abstain)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=["mobayes", "standalone", "both"], default="both")
    ap.add_argument("--num-cases", type=int, default=50)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--tau", type=float, default=0.90)
    ap.add_argument("--prior-mode", default="empirical",
                    choices=["empirical", "uniform", "conditional"])
    ap.add_argument("--doctor-model", default="gpt-5.4-nano")
    ap.add_argument("--patient-model", default="gpt-5.4-nano")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set")

    cases = load_cases()
    name_to_id, universe = load_disease_tables()
    arms = ["mobayes", "standalone"] if args.arm == "both" else [args.arm]
    for arm in arms:
        run_arm(arm, args, cases, name_to_id, universe, api_key)
    for arm in arms:
        summarize(arm, args.out_suffix, args.tau)


if __name__ == "__main__":
    main()
