#!/usr/bin/env python3
"""Model-matched pairings (appendix C.18) from the saved per-case outputs.

For three model families, recomputes Top-1, coverage, selective accuracy and
DHS for the small and large model as standalone doctor and as MoBayes sensor.
MoBayes rows use the DHS-optimal tau*; standalone rows use their natural
operating point. Bootstrap CIs use a fixed seed, so every run prints the same
numbers. The `paper` column checks that cells shared with Table 2 fall inside
the recomputed CI.

The dialogues themselves are stochastic (patient temperature 0.7), so the raw
runs are not bit-reproducible; this script reproduces the reported numbers
from the saved outputs.

Usage (from this folder):
    python3 experiments/analysis/reproduce_w1_2x2.py
"""
from __future__ import annotations
import glob, json, os, random

BASE = os.path.join(os.path.dirname(__file__), "..", "..",
                    "data", "benchmark_results", "table2_crossed_canonical")
BASE = os.path.abspath(BASE)

# (family, config, dir, mode, n, paper_DHS)  paper_DHS=None for new cells.
CELLS = [
    ("GPT-5.4",   "Small standalone",       "sa_gpt54nano",         "standalone", 50, None),
    ("GPT-5.4",   "Large standalone",       "sa_gpt54",             "standalone", 50, 70.4),
    ("GPT-5.4",   "Small sensor + MoBayes", "bmbe_gpt54nano_fixed", "mobayes",    50, 88.0),
    ("GPT-5.4",   "Large sensor + MoBayes", "bmbe_gpt54",           "mobayes",    50, None),
    ("Gemini 3.1","Small standalone",       "sa_geminiflashlite",   "standalone", 50, None),
    ("Gemini 3.1","Large standalone",       "sa_gemini31pro",       "standalone", 25, 75.7),
    ("Gemini 3.1","Small sensor + MoBayes", "bmbe_geminiflashlite", "mobayes",    50, 85.5),
    ("Gemini 3.1","Large sensor + MoBayes", "bmbe_gemini31pro_n25", "mobayes",    25, None),
    ("GPT-OSS",   "Small standalone",       "sa_gptoss20b",         "standalone", 50, None),
    ("GPT-OSS",   "Large standalone",       "sa_gptoss120b",        "standalone", 50, 57.6),
    ("GPT-OSS",   "Small sensor + MoBayes", "bmbe_gptoss20b",       "mobayes",    50, 76.8),
    ("GPT-OSS",   "Large sensor + MoBayes", "bmbe_gptoss120b",      "mobayes",    50, None),
]


def dhs(sel, cov, a=0.5):
    return 0.0 if (sel <= 0 or cov <= 0) else 1.0 / (a / sel + (1 - a) / cov)


def load(cell_dir):
    """Merge every *.jsonl and case_*.json in the dir, dedup by patient_id."""
    seen = {}
    d = os.path.join(BASE, cell_dir)
    for jf in glob.glob(os.path.join(d, "*.jsonl")):
        for line in open(jf):
            try:
                r = json.loads(line); seen[r.get("patient_id")] = r
            except Exception:
                pass
    for cf in glob.glob(os.path.join(d, "case_*.json")):
        try:
            r = json.load(open(cf))
            seen[r.get("patient_id") or r.get("case_idx")] = r
        except Exception:
            pass
    return list(seen.values())


def _final(r):
    """Records store fields either at top level (jsonl) or under 'final' (json)."""
    return r.get("final", r)


def natural_dhs(recs):
    """Standalone: DHS at the model's own operating point (its abstentions)."""
    n = len(recs); ab = t1 = 0
    for r in recs:
        f = _final(r)
        if f.get("abstained"): ab += 1; continue
        if f.get("top1_hit"): t1 += 1
    cov = (n - ab) / n if n else 0
    sel = t1 / (n - ab) if (n - ab) else 0
    return t1 / n * 100 if n else 0, cov * 100, sel * 100, dhs(sel, cov) * 100


def tuned_dhs(recs):
    """MoBayes: DHS-optimal tau* swept over the final max-posterior confidence."""
    scored = [(_final(r).get("confidence", 0.0), bool(_final(r).get("top1_hit"))) for r in recs]
    n = len(scored); t1 = sum(1 for _, h in scored if h)
    best = (-1.0, None, None, None)
    for tau in sorted(set([s for s, _ in scored] + [0.0])):
        comm = [h for s, h in scored if s >= tau]
        c = len(comm)
        if not c: continue
        sel = sum(comm) / c; cov = c / n
        d = dhs(sel, cov) * 100
        if d > best[0]: best = (d, tau, cov * 100, sel * 100)
    return t1 / n * 100 if n else 0, best[2], best[3], best[0], best[1]


def boot_ci(recs, scorer, B=3000, seed=20260728):
    # DHS is at index 3 for both natural_dhs (t1,cov,sel,DHS) and
    # tuned_dhs (t1,cov,sel,DHS,tau*). tuned_dhs re-optimises tau* on each
    # resample, so the CI reflects tuning variability too.
    rng = random.Random(seed)
    n = len(recs); ds = []
    for _ in range(B):
        s = [recs[rng.randrange(n)] for _ in range(n)]
        ds.append(scorer(s)[3])
    ds.sort()
    return ds[int(0.025 * B)], ds[int(0.975 * B)]


def main():
    print(f"BASE = {BASE}\n")
    hdr = f"{'family':<11}{'configuration':<24}{'dir':<24}{'n':>4}{'Top1':>6}{'Cov':>6}{'SelA':>7}{'DHS':>7}{'  95% CI':>14}{'  paper':>9}"
    print(hdr); print("-" * len(hdr))
    for fam, cfg, d, mode, n_exp, paper in CELLS:
        recs = load(d)
        n = len(recs)
        if mode == "mobayes":
            t1, cov, sel, D, tau = tuned_dhs(recs)
            lo, hi = boot_ci(recs, tuned_dhs)
        else:
            t1, cov, sel, D = natural_dhs(recs)
            lo, hi = boot_ci(recs, natural_dhs)
        nflag = "" if n == n_exp else f"!={n_exp}"
        inci = "" if paper is None else (f"{paper} OK" if lo <= paper <= hi else f"{paper} OUT!")
        print(f"{fam:<11}{cfg:<24}{d:<24}{n:>4}{nflag}{t1:>6.0f}{cov:>6.0f}{sel:>7.1f}{D:>7.1f}   [{lo:>4.1f},{hi:>4.1f}]{inci:>9}")
    print("\nMoBayes rows: DHS at tau*.  Standalone rows: natural operating point.")
    print("paper OK = published Table-2 DHS falls within this cell's 95% bootstrap CI.")


if __name__ == "__main__":
    main()
