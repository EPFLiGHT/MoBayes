#!/usr/bin/env python3
"""
Failure Mode Analysis v2 — Three-category taxonomy.

Categories:
  1. KB Failure:  Oracle test gap < 0.80 → KB cannot discriminate this disease pair
  2. LLM Failure: Evidence collection error from the LLM pipeline
     - FP (False Positive): >2 turns where feature NOT in GT profile but extracted="yes"
     - WrEv (Wrong Evidence): feature IN GT profile but extracted value ≠ GT value
  3. Inference Failure: KB OK + LLM OK, but engine still got it wrong
     - Close: GT in top-3 at session end (insufficient evidence)
     - Diverged: GT not in top-3 (engine went completely wrong direction)

Usage:
    cd hybrid-diagnostic-system
    PYTHONPATH=. python3 experiments/analysis/failure_analysis_v2.py \
        --data-dir data/kbs/ddxplus \
        --profiles data/cases/ddxplus_patientsim_a/plain/profiles.json \
        --results data/benchmark_results/set_a/ddxplus_patientsim/mobayes/bmbe_gpt54nano.jsonl \
                  data/benchmark_results/set_a/ddxplus_patientsim/mobayes/bmbe_geminiflashlite.jsonl \
                  data/benchmark_results/set_a/ddxplus_patientsim/mobayes/bmbe_llama4scout.jsonl \
                  data/benchmark_results/set_a/ddxplus_patientsim/mobayes/bmbe_gptoss20b.jsonl \
                  data/benchmark_results/set_a/ddxplus_patientsim/mobayes/bmbe_gemma4_31b.jsonl \
                  data/benchmark_results/set_a/ddxplus_patientsim/mobayes/bmbe_minimax_m25.jsonl \
        --output data/benchmark_results/set_a/ddxplus_patientsim/failure_analysis_v2.json
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from world_model import load_world_model_from_schema


# ── Oracle Test ────────────────────────────────────────────────────────────

def compute_oracle_gap(data_dir: str, profile: dict) -> Tuple[float, bool]:
    """
    Run oracle: supply all GT features at c=1.0.
    Returns (gap between top-1 and top-2 posterior, oracle_pass).
    """
    meta = profile.get("_ddxplus_meta") or profile.get("_llmkb_meta") or {}
    gt_ids = meta.get("ground_truth_ids", [])
    gt_id = gt_ids[0] if gt_ids else ""
    feature_map = meta.get("feature_map", {})

    wm, leaf_diseases = load_world_model_from_schema(data_dir)

    for fid, val_str in feature_map.items():
        if ": YES" in val_str:
            val = "yes"
        elif ": NO" in val_str:
            val = "no"
        else:
            val = val_str.split(": ")[-1].lower() if ": " in val_str else val_str
        try:
            wm.update(fid, val, confidence=1.0, provenance="oracle")
        except Exception:
            pass

    posterior = wm.posterior()
    ranked = np.argsort(posterior)[::-1]
    p1 = float(posterior[ranked[0]])
    p2 = float(posterior[ranked[1]])
    top1_disease = leaf_diseases[ranked[0]]

    return p1 - p2, top1_disease == gt_id


# ── LLM Evidence Analysis ─────────────────────────────────────────────────

def analyze_evidence(session: dict, gt_feature_map: dict) -> dict:
    """
    Compare extracted evidence against GT feature map.
    Returns counts of false positives and wrong evidence.
    """
    fp_turns = []  # (turn, feature_id) — feature not in GT, extracted=yes
    we_turns = []  # (turn, feature_id, extracted, gt_val) — feature in GT, extracted ≠ GT

    asked_fid = None
    for entry in session.get("dialog_history", []):
        if entry.get("role") == "doctor":
            asked_fid = entry.get("feature_id")
        elif entry.get("role") == "patient" and entry.get("turn", 0) >= 1 and asked_fid:
            extracted = str(entry.get("extracted_value", "unknown")).lower().strip()
            turn = entry.get("turn", 0)

            if asked_fid not in gt_feature_map:
                # Feature not in patient profile
                if extracted == "yes":
                    fp_turns.append((turn, asked_fid))
            else:
                # Feature in patient profile — check match
                gt_val_raw = gt_feature_map[asked_fid]
                if ": YES" in gt_val_raw:
                    gt_parsed = "yes"
                elif ": NO" in gt_val_raw:
                    gt_parsed = "no"
                else:
                    gt_parsed = "unknown"

                if extracted != gt_parsed and extracted != "unknown" and gt_parsed != "unknown":
                    we_turns.append((turn, asked_fid, extracted, gt_parsed))

    return {
        "fp_count": len(fp_turns),
        "fp_turns": fp_turns,
        "we_count": len(we_turns),
        "we_turns": we_turns,
    }


# ── Main Analysis ─────────────────────────────────────────────────────────

def run_analysis(data_dir: str, profiles_path: str, results_paths: List[str],
                  selective: bool = False):
    """Run 3-category failure analysis on all results."""

    # Load profiles
    with open(profiles_path) as f:
        profiles = json.load(f)
    prof_map = {}
    for p in profiles:
        meta = p.get("_ddxplus_meta") or p.get("_llmkb_meta") or {}
        pid = meta.get("patient_id") or p.get("hadm_id")
        if pid:
            prof_map[pid] = p

    # Precompute oracle gaps
    print("Computing oracle gaps...")
    oracle_cache = {}
    for pid, profile in prof_map.items():
        gap, passed = compute_oracle_gap(data_dir, profile)
        oracle_cache[pid] = {"gap": gap, "pass": passed}
    print(f"  Oracle pass: {sum(1 for v in oracle_cache.values() if v['pass'])}/{len(oracle_cache)}")

    # Load all results, grouped by source file (= model)
    model_results = {}
    for path in results_paths:
        model_name = Path(path).stem  # e.g. bmbe_gpt54nano
        sessions = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    sessions.append(json.loads(line))
        model_results[model_name] = sessions

    # Analyze
    all_analysis = []

    for model_name, sessions in model_results.items():
        if selective:
            tau = find_optimal_tau(sessions)
            committed = [s for s in sessions if s.get("confidence", 0) >= tau]
            failures = [s for s in committed if not s.get("top1_hit", False)]
            n_total = len(sessions)
            n_committed = len(committed)
            n_fail = len(failures)
            print(f"  {model_name}: tau={tau:.3f}, committed={n_committed}/{n_total}, failures={n_fail}")
        else:
            failures = [s for s in sessions if not s.get("top1_hit", False)]
            n_total = len(sessions)
            n_fail = len(failures)

        model_summary = {
            "model": model_name,
            "n_total": n_total,
            "n_fail": n_fail,
            "kb": 0, "llm": 0, "llm_fp": 0, "llm_we": 0,
            "inf": 0, "inf_close": 0, "inf_diverged": 0,
            "cases": [],
        }

        for s in failures:
            pid = s["patient_id"]
            profile = prof_map.get(pid)
            if not profile:
                continue

            meta = profile.get("_ddxplus_meta") or profile.get("_llmkb_meta") or {}
            gt_fm = meta.get("feature_map", {})
            gt = s.get("ground_truth", "?")
            pred = s.get("predicted", "?")
            top3_hit = s.get("top3_hit", False)

            # 1. KB Failure
            oracle = oracle_cache.get(pid, {"gap": 1.0, "pass": True})
            is_kb = oracle["gap"] < 0.80

            # 2. LLM Failure
            ev = analyze_evidence(s, gt_fm)
            is_fp = ev["fp_count"] > 2
            is_we = ev["we_count"] > 0
            is_llm = is_fp or is_we

            # 3. Inference Failure (residual)
            is_inf = not is_kb and not is_llm

            # Subtypes
            category = "kb" if is_kb else "llm" if is_llm else "inference"
            subtype = ""
            if is_kb:
                subtype = "confusable_pair"
                model_summary["kb"] += 1
            elif is_llm:
                model_summary["llm"] += 1
                if is_fp:
                    subtype = "false_positive"
                    model_summary["llm_fp"] += 1
                else:
                    subtype = "wrong_evidence"
                    model_summary["llm_we"] += 1
            else:
                model_summary["inf"] += 1
                if top3_hit:
                    subtype = "close"
                    model_summary["inf_close"] += 1
                else:
                    subtype = "diverged"
                    model_summary["inf_diverged"] += 1

            case = {
                "patient_id": pid,
                "ground_truth": gt,
                "predicted": pred,
                "category": category,
                "subtype": subtype,
                "oracle_gap": oracle["gap"],
                "oracle_pass": oracle["pass"],
                "fp_count": ev["fp_count"],
                "we_count": ev["we_count"],
                "top3_hit": top3_hit,
            }
            model_summary["cases"].append(case)

        all_analysis.append(model_summary)

    # Print summary
    print(f"\n{'Model':>20} | {'Fail':>4} | {'KB':>3} | {'LLM':>4} {'FP':>3} {'WE':>3} | {'Inf':>4} {'close':>5} {'div':>4}")
    print("=" * 75)
    t = {"fail": 0, "kb": 0, "llm": 0, "fp": 0, "we": 0, "inf": 0, "close": 0, "div": 0}
    for m in all_analysis:
        name = m["model"]
        print(f"{name:>20} | {m['n_fail']:>4} | {m['kb']:>3} | {m['llm']:>4} {m['llm_fp']:>3} {m['llm_we']:>3} | {m['inf']:>4} {m['inf_close']:>5} {m['inf_diverged']:>4}")
        t["fail"] += m["n_fail"]
        t["kb"] += m["kb"]
        t["llm"] += m["llm"]
        t["fp"] += m["llm_fp"]
        t["we"] += m["llm_we"]
        t["inf"] += m["inf"]
        t["close"] += m["inf_close"]
        t["div"] += m["inf_diverged"]
    print("-" * 75)
    print(f"{'TOTAL':>20} | {t['fail']:>4} | {t['kb']:>3} | {t['llm']:>4} {t['fp']:>3} {t['we']:>3} | {t['inf']:>4} {t['close']:>5} {t['div']:>4}")
    if t["fail"] > 0:
        print(f"{'%':>20} |      | {100*t['kb']/t['fail']:>3.0f}% | {100*t['llm']/t['fail']:>3.0f}% {100*t['fp']/t['fail']:>3.0f}% {100*t['we']/t['fail']:>3.0f}% | {100*t['inf']/t['fail']:>3.0f}% {100*t['close']/t['fail']:>4.0f}% {100*t['div']/t['fail']:>3.0f}%")

    # Per-case detail
    print(f"\n{'Model':>20} | {'GT':>25} | {'Pred':>25} | {'Cat':>5} | {'Sub':>8} | FP WE")
    print("-" * 110)
    for m in all_analysis:
        for c in m["cases"]:
            print(f"{m['model']:>20} | {c['ground_truth'][:25]:>25} | {c['predicted'][:25]:>25} | {c['category']:>5} | {c['subtype']:>8} | {c['fp_count']:>2} {c['we_count']:>2}")

    return all_analysis


def find_optimal_tau(sessions):
    """Find DHS-optimal threshold tau for a set of sessions."""
    best_tau, best_dhs = 0.0, 0.0
    n = len(sessions)
    for tau in [t / 200 for t in range(200)]:
        committed = [s for s in sessions if s.get("confidence", 0) >= tau]
        if not committed:
            continue
        sa = sum(1 for s in committed if s.get("top1_hit")) / len(committed) * 100
        cov = len(committed) / n * 100
        dhs = 2 * sa * cov / (sa + cov) if (sa + cov) > 0 else 0
        if dhs > best_dhs:
            best_tau, best_dhs = tau, dhs
    return best_tau


def main():
    parser = argparse.ArgumentParser(description="Failure mode analysis v2 (3-category)")
    parser.add_argument("--data-dir", required=True, help="KB data directory")
    parser.add_argument("--profiles", required=True, help="Benchmark profiles JSON")
    parser.add_argument("--results", nargs="+", required=True, help="BMBE result JSONL file(s)")
    parser.add_argument("--output", default=None, help="Save analysis JSON to this path")
    parser.add_argument("--selective", action="store_true",
                        help="Only analyze committed failures (confidence >= DHS-optimal tau)")
    args = parser.parse_args()

    analysis = run_analysis(args.data_dir, args.profiles, args.results,
                            selective=args.selective)

    if args.output and analysis:
        with open(args.output, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nSaved analysis to {args.output}")


if __name__ == "__main__":
    main()
