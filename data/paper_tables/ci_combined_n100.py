"""ci_combined_n100.py — Bootstrap 95% CIs for the combined n=100 table.

For each (method, benchmark) cell in the combined n=100 view:
  - Read the canonical 50 + v1 50 records via build_combined_n100.cell_records
  - For BMBE: lock τ_canonical and τ_v1 at their LOOCV-derived values on the
    *original* data (not re-tuned per replicate — paper convention).
  - Bootstrap B=2000 replicates: resample 100 indices with replacement from
    {0..99}; for each replicate compute T-1, T-3, SA, Cov, DHS.
  - Report point estimate (full-data) and 95% percentile CI.

Output: markdown table with point ± [lo, hi] for each metric.

Usage:
    python3 scripts/ci_combined_n100.py
    python3 scripts/ci_combined_n100.py --B 5000 --seed 123
    python3 scripts/ci_combined_n100.py --latex > tables/tab_combined_n100_ci.body.tex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sibling module importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_combined_n100 import (  # noqa: E402
    METHODS, BENCHES, cell_records, find_tau_loocv,
)

import numpy as np

# ─── Per-replicate metric ───────────────────────────────────────────────
def metrics_replicate(records: list, tau_per_half: tuple = (None, None),
                      half_sizes: tuple = (50, 50)) -> dict:
    """Compute T-1/T-3/SA/Cov/DHS on a resampled list of records.

    For BMBE (tau_per_half not (None, None)): commit iff conf >= τ for that half.
    The "half" of each record is encoded in r['_half'] ∈ {'can', 'v1'}.
    For non-BMBE: use r['ab'] flag.
    """
    n = len(records)
    if n == 0:
        return {k: 0.0 for k in ("t1", "t3", "sa", "cov", "dhs")}
    t1 = sum(1 for r in records if r["t1"])
    t3 = sum(1 for r in records if r["t3"])
    if tau_per_half == (None, None):
        com = sum(1 for r in records if not r["ab"])
        cor = sum(1 for r in records if not r["ab"] and r["t1"])
    else:
        tau_can, tau_v1 = tau_per_half
        def committed(r):
            tau = tau_can if r.get("_half") == "can" else tau_v1
            if tau is None:
                # No conf → fall back to abstain flag (typically False for BMBE)
                return not r["ab"]
            c = r.get("conf")
            return False if c is None else c >= tau
        com = sum(1 for r in records if committed(r))
        cor = sum(1 for r in records if committed(r) and r["t1"])
    cov = com / n if n else 0.0
    sa  = cor / com if com else 0.0
    dhs = 2 * sa * cov / (sa + cov) if (sa + cov) > 0 else 0.0
    return {"t1": 100 * t1 / n, "t3": 100 * t3 / n,
            "sa": 100 * sa, "cov": 100 * cov, "dhs": 100 * dhs}


def bootstrap_cell(records, tau_per_half=(None, None), B=2000, seed=42):
    """Resample n indices with replacement from [0,n); compute metric per replicate.

    Returns: dict {metric: (point, lo95, hi95)}.
    """
    rng = np.random.default_rng(seed)
    n = len(records)
    point = metrics_replicate(records, tau_per_half)
    samples = {k: np.empty(B) for k in ("t1", "t3", "sa", "cov", "dhs")}
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        rep = [records[i] for i in idx]
        m = metrics_replicate(rep, tau_per_half)
        for k in samples:
            samples[k][b] = m[k]
    out = {}
    for k in samples:
        lo = float(np.percentile(samples[k], 2.5))
        hi = float(np.percentile(samples[k], 97.5))
        out[k] = (point[k], lo, hi)
    return out


# ─── Drive the combined cells ───────────────────────────────────────────
def cell_data(method_key: str, bench_key: str):
    """Return (records_list, tau_per_half) ready for bootstrap."""
    can, v1 = cell_records(method_key, bench_key)
    if not can or not v1:
        return None, None
    # Tag each record with its half for BMBE
    for r in can: r["_half"] = "can"
    for r in v1:  r["_half"] = "v1"
    combined = can + v1

    if method_key == "BMBE":
        tau_can = find_tau_loocv(can) if any(r["conf"] is not None for r in can) else None
        tau_v1  = find_tau_loocv(v1)  if any(r["conf"] is not None for r in v1)  else None
        return combined, (tau_can, tau_v1)
    return combined, (None, None)


def fmt(point, lo, hi, prec=1):
    """Format point [lo, hi] with one decimal."""
    return f"{point:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000, help="Bootstrap replicates")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latex", action="store_true", help="Output LaTeX rows (compact)")
    args = ap.parse_args()

    if not args.latex:
        print(f"# Bootstrap 95% CIs for combined n=100 (B={args.B}, seed={args.seed})\n")
        print("Format: point [2.5%, 97.5%] (one-decimal float).\n")

    for label, mkey in METHODS:
        if not args.latex:
            print(f"\n## {label}")
        for blabel, bkey in BENCHES:
            recs, taus = cell_data(mkey, bkey)
            if not recs:
                if not args.latex:
                    print(f"  - {blabel}: SOURCE MISSING")
                continue
            ci = bootstrap_cell(recs, tau_per_half=taus, B=args.B, seed=args.seed)
            if not args.latex:
                tau_info = ""
                if mkey == "BMBE":
                    tc = "—" if taus[0] is None else f"{taus[0]:.2f}"
                    tv = "—" if taus[1] is None else f"{taus[1]:.2f}"
                    tau_info = f"  (τ_can={tc}, τ_v1={tv})"
                print(f"  - {blabel}{tau_info}")
                for k in ("t1", "t3", "sa", "cov", "dhs"):
                    p, lo, hi = ci[k]
                    print(f"      {k.upper():>4s}: {fmt(p, lo, hi)}")
            else:
                # LaTeX: point with subscript [lo,hi]
                cells = []
                for k in ("t1", "t3", "sa", "cov", "dhs"):
                    p, lo, hi = ci[k]
                    cells.append(f"{p:.1f}\\,\\textsubscript{{[{lo:.1f},{hi:.1f}]}}")
                # only emit per-bench inside one row at a time; collect all 3 benches
                # We'll print a row line at the end of all benches; for now, gather.
                if 'row' not in dir():
                    pass
        # latex row mode: build per-method row across 3 benches
        if args.latex:
            row_parts = []
            for blabel, bkey in BENCHES:
                recs, taus = cell_data(mkey, bkey)
                if not recs:
                    row_parts.extend(["--"] * 5); continue
                ci = bootstrap_cell(recs, tau_per_half=taus, B=args.B, seed=args.seed)
                for k in ("t1", "t3", "sa", "cov", "dhs"):
                    p, lo, hi = ci[k]
                    row_parts.append(f"{p:.1f}\\,\\textsubscript{{[{lo:.1f},{hi:.1f}]}}")
            print(f"{label:<32s} & " + " & ".join(row_parts) + r" \\")


if __name__ == "__main__":
    main()
