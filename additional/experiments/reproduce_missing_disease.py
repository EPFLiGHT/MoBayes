#!/usr/bin/env python3
"""Missing-disease numbers (appendix C.15) from the saved per-case JSONs.

The true disease is removed from the candidate space, so every committed
diagnosis is wrong by construction and the correct behaviour is to abstain.
Prints the outcome split and the tau sweep. Deterministic, no API calls.

Usage (from this folder):
    python3 experiments/reproduce_missing_disease.py
"""
from __future__ import annotations
import glob, json, os
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_results")
BASE = os.path.abspath(BASE)
MOBAYES = os.path.join(BASE, "missing_disease_mobayes")
STANDALONE = os.path.join(BASE, "missing_disease_standalone")


def load(d):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "case_*.json")))]


def main():
    print(f"MoBayes    JSONs: {MOBAYES}")
    print(f"Standalone JSONs: {STANDALONE}\n")
    mob, sa = load(MOBAYES), load(STANDALONE)

    print("Outcomes (n=50 OOD cases, true disease excluded):")
    print(f"{'outcome':<22}{'MoBayes':>10}{'Standalone':>12}")
    mo = Counter(r["final"]["outcome"] for r in mob)
    so = Counter(r["final"]["outcome"] for r in sa)
    for k in ["abstain", "hallucinated_inlist", "named_true_offlist"]:
        pm = f"{mo.get(k,0)} ({mo.get(k,0)*2}%)"
        ps = f"{so.get(k,0)} ({so.get(k,0)*2}%)"
        print(f"{k:<22}{pm:>10}{ps:>12}")

    # tau-sweep: MoBayes commits (=hallucinates, all wrong) when final
    # confidence >= tau. Standalone has no tau knob -> fixed rate.
    print("\ntau-sweep, MoBayes hallucination rate (committed = necessarily wrong):")
    confs = [r["final"]["confidence"] for r in mob]
    sa_halluc = so.get("hallucinated_inlist", 0) + so.get("named_true_offlist", 0)
    print(f"{'tau':>6}{'MoBayes halluc%':>18}{'Standalone halluc%':>20}")
    for tau in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        commit = sum(1 for c in confs if c >= tau)
        print(f"{tau:>6}{commit/len(confs)*100:>17.0f}%{sa_halluc/len(sa)*100:>19.0f}%")
    print("\nAll committed diagnoses are wrong by construction, so committed == hallucination.")


if __name__ == "__main__":
    main()
