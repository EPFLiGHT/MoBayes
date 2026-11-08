"""Reviewer M7: bootstrap CIs and paired-bootstrap significance.

For each system on DDxPlus50, compute bootstrap CIs (B=2000) for top-1,
top-3, selective accuracy, coverage, and DHS. For BMBE rows, the
DHS-optimal tau* is recovered on the full data and held fixed across
bootstrap replicates.

Then, for each within-family pairing (BMBE+sensor vs SA-frontier of the
same family), compute paired-bootstrap delta DHS on the same resampled
patient indices (B=5000). Report 95% CI of delta and one-sided p-value
P(delta <= 0).

Reproduces the numbers reported in Appendix C.12.
"""

import json
import numpy as np
from pathlib import Path

PAPER_CANONICAL = (Path(__file__).resolve().parents[2]
                   / "data/benchmark_results/set_a/ddxplus_patientsim")


def _path(name):
    sub = "mobayes" if name.startswith("bmbe_") else "standalone"
    return PAPER_CANONICAL / sub / name


def load(name):
    with open(_path(name)) as fh:
        return [json.loads(l) for l in fh]


def metrics_threshold(recs, tau):
    """Apply tau-thresholding to BMBE: abstain if confidence < tau."""
    n = len(recs)
    if n == 0:
        return None
    abst = np.array([r["confidence"] < tau for r in recs])
    cov = 1.0 - abst.mean()
    comm = [r for r, a in zip(recs, abst) if not a]
    sel = float(np.mean([r["top1_hit"] for r in comm])) if comm else 0.0
    dhs = 2 * sel * cov / (sel + cov) if (sel + cov) > 0 else 0.0
    top1 = float(np.mean([r["top1_hit"] for r in recs]))
    top3 = float(np.mean([r["top3_hit"] for r in recs]))
    return {
        "top1": top1 * 100,
        "top3": top3 * 100,
        "sel": sel * 100,
        "cov": cov * 100,
        "dhs": dhs * 100,
    }


def metrics_standalone(recs):
    """Standalone: use saved abstained field."""
    abst = np.array([r.get("abstained", False) for r in recs])
    cov = 1.0 - abst.mean()
    comm = [r for r, a in zip(recs, abst) if not a]
    sel = float(np.mean([r["top1_hit"] for r in comm])) if comm else 0.0
    dhs = 2 * sel * cov / (sel + cov) if (sel + cov) > 0 else 0.0
    top1 = float(np.mean([r["top1_hit"] for r in recs]))
    top3 = float(np.mean([r["top3_hit"] for r in recs]))
    return {
        "top1": top1 * 100,
        "top3": top3 * 100,
        "sel": sel * 100,
        "cov": cov * 100,
        "dhs": dhs * 100,
    }


def find_optimal_tau(recs, taus=None):
    if taus is None:
        taus = np.linspace(0, 0.95, 96)
    best_tau, best_dhs = 0.0, -1.0
    for t in taus:
        m = metrics_threshold(recs, t)
        if m and m["dhs"] > best_dhs:
            best_dhs, best_tau = m["dhs"], float(t)
    return best_tau, best_dhs


def bootstrap_ci(recs, score_fn, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(recs)
    out = {k: [] for k in ["top1", "top3", "sel", "cov", "dhs"]}
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        sub = [recs[i] for i in idx]
        m = score_fn(sub)
        for k in out:
            out[k].append(m[k])
    return {
        k: (np.percentile(out[k], 2.5), np.percentile(out[k], 97.5))
        for k in out
    }


def paired_bootstrap_dhs(recs_b, recs_s, tau_b, B=5000, seed=42):
    """Paired BMBE vs Standalone delta DHS, same patient indices."""
    rng = np.random.default_rng(seed)
    n = len(recs_b)
    assert len(recs_s) == n
    deltas = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        sub_b = [recs_b[i] for i in idx]
        sub_s = [recs_s[i] for i in idx]
        d_b = metrics_threshold(sub_b, tau_b)["dhs"]
        d_s = metrics_standalone(sub_s)["dhs"]
        deltas.append(d_b - d_s)
    deltas = np.array(deltas)
    return {
        "mean": float(np.mean(deltas)),
        "ci": (
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)),
        ),
        "p_le0": float(np.mean(deltas <= 0)),
    }


SA_SYSTEMS = [
    ("SA GPT-5.4", "sa_gpt54.jsonl"),
    ("SA Gemini 3.1 Pro", "sa_gemini31pro.jsonl"),
    ("SA Llama-4-Maverick", "sa_llama4maverick.jsonl"),
    ("SA Qwen 3.6+", "sa_qwen36plus.jsonl"),
    ("SA Kimi K2.5", "sa_kimi_k25.jsonl"),
    ("SA GPT-OSS-120B", "sa_gptoss120b.jsonl"),
]
BMBE_SYSTEMS = [
    ("BMBE+GPT-5.4-nano", "bmbe_gpt54nano.jsonl"),
    ("BMBE+Gemini FL", "bmbe_geminiflashlite.jsonl"),
    ("BMBE+GPT-OSS-20B", "bmbe_gptoss20b.jsonl"),
    ("BMBE+MiniMax M2.5", "bmbe_minimax_m25.jsonl"),
    ("BMBE+Gemma 4 31B", "bmbe_gemma4_31b.jsonl"),
    ("BMBE+Llama-4-Scout", "bmbe_llama4scout.jsonl"),
]
WITHIN_FAMILY = [
    ("BMBE+GPT-5.4-nano", "bmbe_gpt54nano.jsonl",
     "SA GPT-5.4", "sa_gpt54.jsonl"),
    ("BMBE+GPT-OSS-20B", "bmbe_gptoss20b.jsonl",
     "SA GPT-OSS-120B", "sa_gptoss120b.jsonl"),
    ("BMBE+Gemini FL", "bmbe_geminiflashlite.jsonl",
     "SA Gemini 3.1 Pro", "sa_gemini31pro.jsonl"),
]


def main():
    fmt = lambda v, c: f"{v:.0f} [{c[0]:.0f},{c[1]:.0f}]"
    print(f"{'System':<22} {'Top-1':<14} {'Top-3':<14} "
          f"{'Sel':<14} {'Cov':<14} {'DHS':<14}")
    print("=" * 100)
    for name, fn in SA_SYSTEMS:
        recs = load(fn)
        m = metrics_standalone(recs)
        ci = bootstrap_ci(recs, metrics_standalone)
        print(f"{name:<22} {fmt(m['top1'],ci['top1']):<14} "
              f"{fmt(m['top3'],ci['top3']):<14} "
              f"{fmt(m['sel'],ci['sel']):<14} "
              f"{fmt(m['cov'],ci['cov']):<14} "
              f"{fmt(m['dhs'],ci['dhs']):<14}")
    print()
    for name, fn in BMBE_SYSTEMS:
        recs = load(fn)
        tau_star, _ = find_optimal_tau(recs)
        m = metrics_threshold(recs, tau_star)
        ci = bootstrap_ci(recs, lambda r, t=tau_star: metrics_threshold(r, t))
        print(f"{name:<22} {fmt(m['top1'],ci['top1']):<14} "
              f"{fmt(m['top3'],ci['top3']):<14} "
              f"{fmt(m['sel'],ci['sel']):<14} "
              f"{fmt(m['cov'],ci['cov']):<14} "
              f"{fmt(m['dhs'],ci['dhs']):<14}  tau={tau_star:.2f}")

    print("\n=== Paired-bootstrap within-family significance ===")
    print(f"{'Comparison':<55} {'delta DHS':<10} {'95% CI':<20} {'p (delta<=0)':<10}")
    print("=" * 100)
    for nb, fb, ns, fs in WITHIN_FAMILY:
        rb = load(fb)
        rs = load(fs)
        # align by patient_id
        s_map = {r["patient_id"]: r for r in rs}
        rs = [s_map[r["patient_id"]] for r in rb]
        tau_b, _ = find_optimal_tau(rb)
        out = paired_bootstrap_dhs(rb, rs, tau_b)
        print(f"{nb} vs {ns:<28}  {out['mean']:>+6.1f} pp  "
              f"[{out['ci'][0]:>+5.1f}, {out['ci'][1]:>+5.1f}]    "
              f"{out['p_le0']:.3f}")


if __name__ == "__main__":
    main()
