#!/usr/bin/env python3
"""
Interactive diagnostic session via terminal.

Uses the full MoBayes pipeline:
  - Bulk extraction from initial free-text intake (LLM)
  - EIG-based feature selection
  - StructuredParser for response parsing (LLM)
  - Jeffrey's conditioning for confidence-weighted updates

Usage:
    PYTHONPATH=. python3 scripts/core/run_session.py
    PYTHONPATH=. python3 scripts/core/run_session.py --data-dir data/kbs/ddxplus --model gpt-4o
    PYTHONPATH=. python3 scripts/core/run_session.py --max-questions 30 --no-llm
"""
import os
import sys
import argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

_env_path = os.path.join(BASE_DIR, '.env')
if os.path.exists(_env_path):
    from dotenv import load_dotenv
    load_dotenv(_env_path)

from world_model import load_world_model_from_schema, load_disease_names, StoppingRules, resolve_confidence
from config import MoBayesConfig
from decision_policy.policy import EIGPolicy
from llm_interface.prompt_templates import PromptTemplates
from llm_interface.structured_parser import StructuredParser, enforce_schema
from llm_interface.llm_client import OpenAILLMClient
from llm_interface.ddxplus_display import prepare_display, normalize_extracted_value, resolve_negated_name


# ── ANSI Colors ──────────────────────────────────────────────────────────────

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


# ── Display helpers ──────────────────────────────────────────────────────────

def print_top_diseases(posterior, leaf_diseases, disease_names, n=5):
    ranked = np.argsort(posterior)[::-1][:n]
    for i, idx in enumerate(ranked):
        did = leaf_diseases[idx]
        name = disease_names.get(did, did)
        prob = posterior[idx]
        bar = "█" * int(prob * 40)
        color = GREEN if i == 0 else (YELLOW if i < 3 else DIM)
        marker = " ◄" if i == 0 else ""
        print(f"  {color}{i+1}. {name:<35} {prob:6.2%} {bar}{marker}{RESET}")


def print_entropy(wm):
    H = wm.entropy()
    print(f"  {DIM}Entropy: {H:.2f} bits{RESET}")


# ── Session ──────────────────────────────────────────────────────────────────

def run_session(data_dir: str, max_questions: int = 20,
                model: str = "gpt-4o-mini", use_llm: bool = True,
                confidence_threshold: float = None):
    cfg = MoBayesConfig()
    if confidence_threshold is not None:
        cfg.confidence_threshold = confidence_threshold

    # Load KB
    wm, leaf_diseases = load_world_model_from_schema(data_dir)

    disease_names = load_disease_names(data_dir)

    # LLM interface
    llm_interface = None
    if use_llm:
        llm_client = OpenAILLMClient(model=model, temperature=cfg.llm_temperature)
        templates = PromptTemplates()
        llm_interface = {
            'llm': llm_client,
            'templates': templates,
            'parser': StructuredParser(llm_client, templates),
        }

    policy = EIGPolicy()
    stopping = StoppingRules(
        confidence_threshold=cfg.confidence_threshold,
        max_questions=max_questions,
        min_questions_before_abstain=cfg.min_questions_before_abstain,
    )

    # ── Header ──
    print(f"\n{BLUE}{'═' * 60}{RESET}")
    print(f"{BLUE}{BOLD}  MoBayes Diagnostic Session{RESET}")
    print(f"{BLUE}  {len(leaf_diseases)} diseases, "
          f"{len(wm.likelihood_model.schema)} features"
          f"  |  LLM: {model if use_llm else 'off'}{RESET}")
    print(f"{BLUE}{'═' * 60}{RESET}\n")

    print()
    print_entropy(wm)

    # ── Initial Intake (bulk extraction) ──
    print(f"\n{BOLD}What brings you in today?{RESET}")
    chief_complaint = input(f"  {BOLD}> {RESET}").strip()

    if chief_complaint and llm_interface:
        print(f"\n  {DIM}Analyzing initial statement...{RESET}")

        # Build feature list (resolve V-codes to human-readable labels)
        feature_list = []
        for fid, schema in wm.likelihood_model.schema.items():
            name = schema.metadata.get('name', fid)
            _, display_vals, _ = prepare_display(fid, name, schema.values)
            feature_list.append({
                "id": fid,
                "name": name,
                "values": display_vals,
            })

        extraction_result = llm_interface['parser'].bulk_extract(chief_complaint, feature_list)

        # Process extraction (same as main_loop.run_initial_intake)
        extracted_count = 0
        for fid, raw_val in extraction_result.items():
            if fid == "demographics" or fid not in wm.likelihood_model.schema:
                continue
            if isinstance(raw_val, dict):
                val = str(raw_val.get("value", raw_val))
                conf_label = raw_val.get("confidence", "likely")
            else:
                val = str(raw_val).strip()
                if '|' in val:
                    parts = val.split('|', 1)
                    val = parts[0].strip()
                    conf_label = parts[1].strip()
                else:
                    conf_label = "likely"

            schema_vals = wm.likelihood_model.schema[fid].values
            val = normalize_extracted_value(fid, val, schema_vals)
            normalized = enforce_schema(val, schema_vals)
            if normalized == "unknown":
                continue

            confidence = resolve_confidence(conf_label)
            confidence = min(confidence, cfg.bulk_extraction_confidence_cap)
            wm.update(fid, normalized, confidence=confidence, provenance="bulk_intake")
            fname = wm.likelihood_model.schema[fid].metadata.get('name', fid)
            print(f"    {GREEN}✓{RESET} {fname[:55]} → {BOLD}{normalized}{RESET}")
            extracted_count += 1

        if extracted_count:
            print(f"\n  {GREEN}Extracted {extracted_count} features from intake{RESET}")
        else:
            print(f"  {YELLOW}No features extracted{RESET}")

        print()
        print_entropy(wm)
        print()
        print(f"  {BOLD}Current assessment:{RESET}")
        print_top_diseases(wm.posterior(), leaf_diseases, disease_names)

    # ── Diagnostic Loop ──
    turn = 0
    while True:
        # Check stopping
        num_asked = len(wm.evidence())
        stopped, reason = stopping.should_stop_wm(wm, num_asked)
        if stopped:
            print(f"\n  {YELLOW}[Stop: {reason.name}]{RESET}")
            break

        # Select next feature
        next_feature = policy.select_feature(wm)
        if next_feature is None:
            print(f"\n  {YELLOW}[All features exhausted]{RESET}")
            break

        feature_schema = wm.likelihood_model.schema[next_feature]
        feature_name = feature_schema.metadata.get("name", next_feature)
        schema_values = feature_schema.values
        turn += 1

        # Resolve display names (V-codes → labels, negated → positive)
        display_name = resolve_negated_name(next_feature, feature_name)
        display_name, display_schema, _ = prepare_display(
            next_feature, display_name, schema_values
        )

        # Generate question
        if llm_interface:
            max_prob = float(np.max(wm.posterior()))
            confidence_level = ("low" if max_prob < cfg.confidence_display_low
                                else "medium" if max_prob < cfg.confidence_display_high
                                else "high")
            control_payload = {
                "next_feature": display_name,
                "answer_schema": display_schema,
                "confidence_level": confidence_level,
                "last_user_query": None,
            }
            prompt = llm_interface['templates'].render_question_prompt(control_payload)
            question = llm_interface['llm'].generate(prompt).strip()
        else:
            question = display_name

        # Ask
        print(f"\n{CYAN}Q{turn}:{RESET} {question}")
        if not use_llm:
            print(f"  {DIM}Options: {', '.join(str(v) for v in display_schema)}{RESET}")
        answer = input(f"  {BOLD}> {RESET}").strip()

        if not answer:
            answer = "unknown"
            confidence = 1.0
        elif llm_interface:
            # StructuredParser: extract value|confidence (use display labels)
            value, conf_label = llm_interface['parser'].extract_feature_value_with_confidence(
                answer, display_name, display_schema
            )
            # Normalize back to schema domain (V-code reverse, numeric clamp, negation)
            value = normalize_extracted_value(next_feature, value, schema_values)
            value = enforce_schema(value, schema_values)
            confidence = resolve_confidence(conf_label)
            answer = value
            print(f"  {DIM}→ extracted: \"{answer}\" (confidence: {conf_label}){RESET}")

            # Follow-up: if unknown, re-ask with explicit options and clarification
            if answer in ("unknown", "clarification_needed"):
                print(f"  {YELLOW}Could not parse — following up with options{RESET}")
                print(f"  {DIM}Options: {', '.join(str(v) for v in display_schema)}{RESET}")
                followup_answer = input(f"  {BOLD}> {RESET}").strip()
                if followup_answer:
                    v2, c2 = llm_interface['parser'].extract_feature_value_with_confidence(
                        followup_answer, display_name, display_schema
                    )
                    v2 = normalize_extracted_value(next_feature, v2, schema_values)
                    v2 = enforce_schema(v2, schema_values)
                    if v2 != "unknown":
                        answer = v2
                        confidence = resolve_confidence(c2)
                        print(f"  {DIM}→ follow-up extracted: \"{answer}\" (confidence: {c2}){RESET}")
        else:
            # Manual mode: match to schema
            if answer.lower() in ("u", "?", "skip", "unknown", "bilmiyorum"):
                answer = "unknown"
            else:
                matched = [v for v in schema_values if v.lower() == answer.lower()]
                if matched:
                    answer = matched[0]
                else:
                    partial = [v for v in schema_values if answer.lower() in v.lower()]
                    if len(partial) == 1:
                        answer = partial[0]
                    else:
                        print(f"  {RED}'{answer}' not recognized → unknown{RESET}")
                        answer = "unknown"
            confidence = 1.0

        wm.update(next_feature, answer, confidence=confidence, provenance="terminal_input")

        # Show belief state
        print_entropy(wm)
        print(f"  {DIM}Top: ", end="")
        posterior = wm.posterior()
        top3 = np.argsort(posterior)[::-1][:3]
        for i, idx in enumerate(top3):
            name = disease_names.get(leaf_diseases[idx], leaf_diseases[idx])
            print(f"{name} ({posterior[idx]:.0%})", end="")
            if i < 2:
                print(" | ", end="")
        print(f"{RESET}")

    # ── Final Result ──
    posterior = wm.posterior()

    print(f"\n{GREEN}{'═' * 60}{RESET}")
    print(f"{GREEN}{BOLD}  DIAGNOSTIC RESULT{RESET}")
    print(f"{GREEN}{'═' * 60}{RESET}")
    print(f"\n  Questions asked: {turn}")
    print_entropy(wm)
    print()
    print_top_diseases(posterior, leaf_diseases, disease_names, n=5)

    print(f"\n{GREEN}{'═' * 60}{RESET}\n")


def main():
    _defaults = MoBayesConfig()
    parser = argparse.ArgumentParser(description="Interactive MoBayes diagnostic session")
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, _defaults.data_dir),
                        help="Knowledge base directory")
    parser.add_argument("--max-questions", type=int, default=_defaults.max_questions,
                        help="Maximum questions to ask")
    parser.add_argument("--model", default=_defaults.default_model,
                        help="LLM model for question generation and parsing")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM (raw schema questions, manual input)")
    parser.add_argument("--confidence-threshold", type=float, default=_defaults.confidence_threshold,
                        help="Stopping confidence threshold (default: %(default)s)")
    args = parser.parse_args()

    run_session(
        data_dir=args.data_dir,
        max_questions=args.max_questions,
        model=args.model,
        use_llm=not args.no_llm,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    main()
