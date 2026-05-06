"""Reviewer M4: leave-one-out cross-validation for tau*.

The headline BMBE DHS in Table 2 uses tau* tuned on the full 50-patient
DDxPlus50 set, evaluated on the same set. With a 96-point tau grid this
incurs a multiple-comparisons / tuning-on-test optimism.

LOOCV: for each patient i, retune tau on the other 49 (DHS-maximising
on that subset), then evaluate patient i with that tau_(-i). Aggregate
the 50 outcomes for an unbiased DHS estimate.

Compare headline (full-data optimal tau) vs LOOCV; the difference
quantifies the tuning optimism.
"""

import json
import numpy as np
from pathlib import Path

PAPER_CANONICAL = Path(
    ""
    "hybrid-diagnostic-system/data/benchmark_results/set_a/ddxplus_patientsim"
)


def load(name):
    with open(PAPER_CANONICAL / name) as fh:
        return [json.loads(l) for l in fh]


def metrics_threshold(recs, tau):
    if not recs:
        return None
    abst = np.array([r["confidence"] < tau for r in recs])
    cov = 1.0 - abst.mean()
    comm = [r for r, a in zip(recs, abst) if not a]
    sel = float(np.mean([r["top1_hit"] for r in comm])) if comm else 0.0
    dhs = 2 * sel * cov / (sel + cov) if (sel + cov) > 0 else 0.0
    return {"sel": sel * 100, "cov": cov * 100, "dhs": dhs * 100}


def find_optimal_tau(recs, taus=None):
    if taus is None:
        taus = np.linspace(0, 0.95, 96)
    best_tau, best_dhs = 0.0, -1.0
    for t in taus:
        m = metrics_threshold(recs, t)
        if m and m["dhs"] > best_dhs:
            best_dhs, best_tau = m["dhs"], float(t)
    return best_tau, best_dhs


def loocv_dhs(recs):
    """Unbiased DHS via leave-one-out tau tuning."""
    n = len(recs)
    committed = 0
    correct = 0
    taus_used = []
    for i in range(n):
        rest = recs[:i] + recs[i + 1:]
        tau_i, _ = find_optimal_tau(rest)
        taus_used.append(tau_i)
        if recs[i]["confidence"] >= tau_i:
            committed += 1
            if recs[i]["top1_hit"]:
                correct += 1
    cov = committed / n
    sel = correct / committed if committed else 0.0
    dhs = 2 * sel * cov / (sel + cov) if (sel + cov) > 0 else 0.0
    return {
        "sel": sel * 100,
        "cov": cov * 100,
        "dhs": dhs * 100,
        "committed": committed,
        "correct": correct,
        "tau_mean": float(np.mean(taus_used)),
        "tau_std": float(np.std(taus_used)),
    }


BMBE_SYSTEMS = [
    ("BMBE+GPT-5.4-nano", "bmbe_gpt54nano.jsonl"),
    ("BMBE+Gemini FL", "bmbe_geminiflashlite.jsonl"),
    ("BMBE+GPT-OSS-20B", "bmbe_gptoss20b.jsonl"),
    ("BMBE+MiniMax M2.5", "bmbe_minimax_m25.jsonl"),
    ("BMBE+Gemma 4 31B", "bmbe_gemma4_31b.jsonl"),
    ("BMBE+Llama-4-Scout", "bmbe_llama4scout.jsonl"),
]


def main():
    print(
        f"{'System':<22} {'Headline DHS':<22} {'LOOCV DHS':<22} "
        f"{'Optimism':<10} {'tau range':<14}"
    )
    print("=" * 95)
    rows = []
    for name, fn in BMBE_SYSTEMS:
        recs = load(fn)
        tau_full, dhs_full = find_optimal_tau(recs)
        loocv = loocv_dhs(recs)
        opt = dhs_full - loocv["dhs"]
        print(
            f"{name:<22} "
            f"{dhs_full:>5.1f}  (tau={tau_full:.2f})    "
            f"{loocv['dhs']:>5.1f}  (sel={loocv['sel']:.0f},cov={loocv['cov']:.0f})  "
            f"{opt:+5.1f} pp   "
            f"[{max(0,loocv['tau_mean']-loocv['tau_std']):.2f}, "
            f"{min(1,loocv['tau_mean']+loocv['tau_std']):.2f}]"
        )
        rows.append((name, dhs_full, loocv["dhs"], opt))

    print("\nMean optimism: "
          f"{np.mean([r[3] for r in rows]):.1f} pp  "
          f"(std {np.std([r[3] for r in rows]):.1f} pp)")


if __name__ == "__main__":
    main()
