"""
Detailed Trace of a Single Diagnostic Session.

Shows every step of MoBayes's diagnostic process:
  1. Patient profile (symptoms, history, chief complaint)
  2. Bulk extraction from chief complaint
  3. Each turn: EIG selection → question → patient answer → LLM extraction → Bayesian update
  4. Belief state changes, top diseases, entropy after each step

Usage:
    PYTHONPATH=. python3 scripts/utils/trace_session.py \
        --profiles data/cases/ddxplus_patientsim_a/plain/profiles.json                 # First patient
    PYTHONPATH=. python3 scripts/utils/trace_session.py \
        --profiles data/cases/ddxplus_patientsim_a/plain/profiles.json --patient-index 3   # 4th patient
"""

import os
import sys
import json
import numpy as np
from typing import Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

_env_path = os.path.join(BASE_DIR, '.env')
if os.path.exists(_env_path):
    from dotenv import load_dotenv
    load_dotenv(_env_path)

sys.path.insert(0, os.path.join(BASE_DIR, 'scripts', 'core'))

from benchmark_patientsim import SimplePatientAgent, MoBayesDoctor
from llm_interface.ddxplus_display import (
    is_numeric_scale, has_vcodes, readable_values, NUMERIC_SCALE_INFO,
)


# ── Pretty printing helpers ──────────────────────────────────────────────────

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def header(text: str, char="═"):
    width = 70
    print(f"\n{BOLD}{char * width}")
    print(f"  {text}")
    print(f"{char * width}{RESET}\n")


def subheader(text: str):
    print(f"\n{BOLD}{CYAN}── {text} {'─' * max(0, 60 - len(text))}{RESET}\n")


def print_top_diseases(doctor: MoBayesDoctor, n: int = 5):
    """Print current top-N diseases with probabilities."""
    posterior = doctor.wm.posterior()
    ranked = np.argsort(posterior)[::-1][:n]
    for i, idx in enumerate(ranked):
        did = doctor.leaf_diseases[idx]
        name = doctor.disease_id_to_name(did)
        prob = posterior[idx]
        bar = "█" * int(prob * 40)
        marker = " ◄" if i == 0 else ""
        color = GREEN if i == 0 else (YELLOW if i < 3 else DIM)
        print(f"  {color}{i+1}. {name:<35} {prob:6.2%} {bar}{marker}{RESET}")


def print_belief_change(before: np.ndarray, after: np.ndarray,
                        doctor: MoBayesDoctor, top_n: int = 3):
    """Show how belief changed for top diseases."""
    ranked_after = np.argsort(after)[::-1][:top_n]
    changes = []
    for idx in ranked_after:
        name = doctor.disease_id_to_name(doctor.leaf_diseases[idx])
        delta = after[idx] - before[idx]
        sign = "+" if delta >= 0 else ""
        changes.append(f"{name}: {before[idx]:.2%} → {after[idx]:.2%} ({sign}{delta:.2%})")
    print(f"  {DIM}Belief Δ: {' | '.join(changes)}{RESET}")


# ── Main trace ───────────────────────────────────────────────────────────────

def trace_session(profile: Dict, data_dir: str,
                  patient_model: str = "gpt-4o-mini",
                  doctor_llm: str = "gpt-4o-mini",
                  confidence_threshold: float = None):
    """Run and trace a single diagnostic session in full detail."""

    ground_truth = profile["_ddxplus_meta"]["diagnosis_name"]
    patient_id = profile["hadm_id"]

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 0: Patient Profile
    # ═══════════════════════════════════════════════════════════════════
    header(f"PATIENT PROFILE: {patient_id}")

    print(f"  {BOLD}Ground Truth:{RESET} {GREEN}{ground_truth}{RESET}")
    print(f"  Age: {profile.get('age', '?')}, Gender: {profile.get('gender', '?')}")
    print()

    chief = profile.get("Current Visit", {}).get("chiefcomplaint", "")
    print(f"  {BOLD}Chief Complaint:{RESET} \"{chief}\"")
    print()

    pos = profile.get("Present Illness", {}).get("present_illness_positive", "")
    print(f"  {BOLD}Symptoms Present:{RESET}")
    for s in pos.split(", "):
        print(f"    • {s}")
    print()

    med_hist = profile.get("Medical History", {}).get("medical_history", "")
    print(f"  {BOLD}Medical History:{RESET} {med_hist}")

    cat = profile.get("_ddxplus_meta", {}).get("categorical_findings", "")
    if cat:
        print(f"  {BOLD}Categorical Findings:{RESET} {cat}")

    persona = profile.get("Persona Settings", {})
    cefr = profile.get("Language Proficiency (CEFR Levels)", {}).get("cefr", "B")
    print(f"\n  {BOLD}Persona:{RESET} CEFR={cefr}, personality={persona.get('personality', 'plain')}, "
          f"recall={persona.get('recall_level', 'high')}, dazed={persona.get('dazed_level', 'normal')}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Initialize Agents
    # ═══════════════════════════════════════════════════════════════════
    header("INITIALIZATION")

    print(f"  Loading patient agent (model={patient_model})...")
    patient = SimplePatientAgent(profile, model=patient_model)

    print(f"  Loading MoBayes doctor (model={doctor_llm})...")
    from config import MoBayesConfig as _Cfg
    _cfg = _Cfg()
    if confidence_threshold is not None:
        _cfg.confidence_threshold = confidence_threshold
    doctor = MoBayesDoctor(data_dir, llm_model=doctor_llm, config=_cfg)
    doctor.reset(profile.get("meta", {}))

    print(f"  World model: {doctor.wm.num_diseases} diseases, "
          f"{len(doctor.wm.likelihood_model.schema)} features")
    print(f"  Initial entropy: {doctor.wm.entropy():.2f} bits")
    print()

    subheader("Initial Prior (Top 5)")
    print_top_diseases(doctor)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Bulk Extraction from Chief Complaint
    # ═══════════════════════════════════════════════════════════════════
    header("BULK EXTRACTION FROM CHIEF COMPLAINT")

    print(f"  Input text: \"{chief}\"")
    print(f"  Sending {len(doctor.wm.likelihood_model.schema)} features to LLM...")
    print()

    belief_before = doctor.wm.posterior().copy()
    bulk_results = doctor.bulk_extract_from_text(chief)

    if bulk_results:
        print(f"  {GREEN}Extracted {len(bulk_results)} features:{RESET}")
        for fid, val in bulk_results.items():
            schema = doctor.wm.likelihood_model.schema.get(fid)
            fname = schema.metadata.get('name', fid) if schema else fid
            print(f"    {fid}: {fname[:50]} → {BOLD}{val}{RESET}")
        print()
        belief_after = doctor.wm.posterior()
        print(f"  Entropy: {np.sum(-belief_before[belief_before > 0] * np.log2(belief_before[belief_before > 0])):.2f} → {doctor.wm.entropy():.2f} bits")
        print_belief_change(belief_before, belief_after, doctor)
    else:
        print(f"  {YELLOW}No features extracted from chief complaint.{RESET}")

    print()
    subheader("After Bulk Extraction (Top 5)")
    print_top_diseases(doctor)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: Diagnostic Loop
    # ═══════════════════════════════════════════════════════════════════
    header("DIAGNOSTIC CONVERSATION")

    for turn in range(_cfg.max_questions + 10):
        # Check stopping
        q_result = doctor.get_next_question()
        if q_result["status"] == "stopped":
            print(f"\n  {BOLD}{RED}⛔ STOPPED: {q_result['reason']}{RESET}")
            break

        feature_id = q_result["feature_id"]
        feature_name = q_result["feature_name"]
        question = q_result["question"]
        schema_values = q_result["schema_values"]
        entropy_before = q_result["entropy"]

        # ── Turn header ──
        subheader(f"Turn {turn + 1}")

        # EIG selection info
        print(f"  {DIM}EIG selected: {feature_id} ({feature_name[:60]}){RESET}")

        # Show what kind of feature this is
        if is_numeric_scale(schema_values):
            scale_info = NUMERIC_SCALE_INFO.get(feature_id, {})
            print(f"  {DIM}Type: NUMERIC 0-10 (0={scale_info.get('low','?')}, 10={scale_info.get('high','?')}){RESET}")
        elif has_vcodes(schema_values):
            labels, _ = readable_values(schema_values)
            print(f"  {DIM}Type: CATEGORICAL → {labels}{RESET}")
        else:
            print(f"  {DIM}Type: BINARY {schema_values}{RESET}")

        # Doctor asks
        print(f"\n  {BLUE}🩺 Doctor:{RESET} {question}")

        # Patient responds
        patient_response = patient.respond(question)
        print(f"  {YELLOW}🤒 Patient:{RESET} {patient_response}")

        # LLM extraction
        belief_before = doctor.wm.posterior().copy()
        extracted, jeff_weight = doctor.process_response(feature_id, patient_response, schema_values)
        belief_after = doctor.wm.posterior()
        entropy_after = doctor.wm.entropy()

        # Map jeffrey weight back to confidence label for display
        _conf_labels = {1.0: "very_likely", 0.80: "likely", 0.50: "uncertain",
                        0.25: "unlikely", 0.05: "very_unlikely"}
        conf_label = _conf_labels.get(jeff_weight, f"w={jeff_weight:.2f}")
        print(f"\n  {DIM}LLM extracted: \"{extracted}\" | confidence: {conf_label} (jeffrey_w={jeff_weight:.2f}){RESET}")

        # Follow-up: if unknown, re-ask with clarification context
        if extracted in ("unknown", "clarification_needed"):
            print(f"  {YELLOW}⟳ Follow-up (unknown → re-asking with context){RESET}")
            followup_q = doctor._generate_question(
                feature_id, feature_name, schema_values,
                last_user_query=patient_response,
            )
            print(f"  {BLUE}🩺 Doctor (follow-up):{RESET} {followup_q}")
            followup_resp = patient.respond(followup_q)
            print(f"  {YELLOW}🤒 Patient:{RESET} {followup_resp}")
            extracted2, jeff_weight2 = doctor.process_response(feature_id, followup_resp, schema_values)
            if extracted2 not in ("unknown", "clarification_needed"):
                extracted = extracted2
                jeff_weight = jeff_weight2
                belief_after = doctor.wm.posterior()
                entropy_after = doctor.wm.entropy()
                c2_label = _conf_labels.get(jeff_weight2, f"w={jeff_weight2:.2f}")
                print(f"  {DIM}Follow-up extracted: \"{extracted2}\" | {c2_label}{RESET}")

        print(f"  {DIM}Entropy: {entropy_before:.3f} → {entropy_after:.3f} "
              f"(Δ = {entropy_after - entropy_before:+.3f} bits){RESET}")

        # Belief change
        print_belief_change(belief_before, belief_after, doctor, top_n=3)

        print()
        print(f"  {BOLD}Current Top 5:{RESET}")
        print_top_diseases(doctor)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4: Final Result
    # ═══════════════════════════════════════════════════════════════════
    header("FINAL DIAGNOSIS")

    results = doctor.get_results()
    predicted = results["top_disease"]
    confidence = results["confidence"]
    is_correct = predicted.lower() == ground_truth.lower()

    color = GREEN if is_correct else RED
    print(f"  {BOLD}Ground Truth:{RESET}  {ground_truth}")
    print(f"  {BOLD}Predicted:{RESET}     {color}{predicted} ({confidence:.1%}){RESET}")
    print(f"  {BOLD}Correct:{RESET}       {color}{'YES ✓' if is_correct else 'NO ✗'}{RESET}")
    print()

    print("  Top 5 Differential:")
    for i, name in enumerate(results["top_diseases"][:5]):
        marker = " ◄ GT" if name.lower() == ground_truth.lower() else ""
        print(f"    {i+1}. {name}{marker}")

    total_turns = len(doctor.feature_history)
    print(f"  Total turns: {total_turns}")
    print(f"  Final entropy: {results['entropy']:.3f} bits")

    header("SESSION COMPLETE")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    import argparse
    from config import MoBayesConfig
    _defaults = MoBayesConfig()

    parser = argparse.ArgumentParser(description='Trace a single diagnostic session in detail')
    parser.add_argument('--profiles', default=None,
                        help='Path to profiles.json')
    parser.add_argument('--data-dir', default=None,
                        help='Path to KB data dir')
    parser.add_argument('--patient-index', type=int, default=0,
                        help='Index of the patient in profiles.json (default: 0)')
    parser.add_argument('--patient-id', default=None,
                        help='Patient ID to trace (e.g. ddx_085153)')
    parser.add_argument('--patient-model', default=_defaults.default_model)
    parser.add_argument('--doctor-model', default=_defaults.default_model)
    parser.add_argument('--confidence-threshold', type=float, default=None,
                        help='Stopping confidence threshold (default: from config)')
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(BASE_DIR, _defaults.data_dir)
    profiles_path = args.profiles or os.path.join(BASE_DIR, _defaults.cases_dir, 'profiles.json')

    with open(profiles_path) as f:
        profiles = json.load(f)

    # Select patient
    if args.patient_id:
        profile = next((p for p in profiles if p["hadm_id"] == args.patient_id), None)
        if not profile:
            print(f"Patient {args.patient_id} not found. Available:")
            for p in profiles[:20]:
                print(f"  {p['hadm_id']} ({p['_ddxplus_meta']['diagnosis_name']})")
            return
    else:
        if args.patient_index >= len(profiles):
            print(f"Index {args.patient_index} out of range. Max: {len(profiles)-1}")
            return
        profile = profiles[args.patient_index]

    trace_session(profile, data_dir,
                  patient_model=args.patient_model,
                  doctor_llm=args.doctor_model,
                  confidence_threshold=args.confidence_threshold)


if __name__ == '__main__':
    main()
