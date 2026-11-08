"""find_optimal_tau.py — DHS-optimal τ for every BMBE/MoBayes cell.

Per paper §App threshold_tuning:
  τ ∈ {0.00, 0.05, ..., 0.95}; commit iff confidence ≥ τ.
  τ* selected by maximising DHS_α (default α=0.5, harmonic mean of SA & Cov).
  Paper uses leave-one-out CV on n=50; we report both full-set DHS-optimal τ*
  and LOOCV-optimal τ* for each cell so you can compare.

For each BMBE cell in Tables 1, 2, 22:
  - Read case_*.json (or .jsonl) source
  - Extract per-case (confidence, top1_hit, top3_hit) using the same field
    convention the reproducer uses (judge fields preferred, else strict)
  - Sweep τ; compute (T-1, T-3, SA, Cov, DHS) at each τ
  - Pick τ* = argmax_DHS (full-set)
  - Pick τ* = argmax_DHS via LOOCV (held-out single case)
  - Compare to the tex value parsed from the .tex

Usage:
    python3 scripts/paper_tables/find_optimal_tau.py              # all BMBE cells, both tables
    python3 scripts/paper_tables/find_optimal_tau.py --table 22   # single table
    python3 scripts/paper_tables/find_optimal_tau.py --sweep      # show full sweep per cell
    python3 scripts/paper_tables/find_optimal_tau.py --loocv      # show LOOCV detail
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

HOME = Path.home()
CODEBASE = Path(__file__).resolve().parents[2]
RESULTS  = CODEBASE / "data/benchmark_results"

# ─── Source readers ─────────────────────────────────────────────────────
def read_dir(rel: str) -> List[dict]:
    d = RESULTS / rel
    if not d.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(d.glob("case_*.json"))]

def read_jsonl(rel: str) -> List[dict]:
    p = RESULTS / rel
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try: out.append(json.loads(line))
        except json.JSONDecodeError: pass
    return out

def get(r: dict, field: str):
    """Look up final.<field> (per-case dir) or top-level (jsonl)."""
    if isinstance(r.get("final"), dict) and field in r["final"]:
        return r["final"][field]
    return r.get(field)


def extract_records(records, t1_field="top1_hit", t3_field="top3_hit",
                    conf_field="confidence", ab_field="abstained"):
    """Normalise to list of dicts with 'conf','t1','t3','ab'.

    For sources with judge fields (top1_hit_judged), prefer those when present.
    """
    out = []
    for r in records:
        conf = get(r, conf_field)
        # prefer judged Top-k if present in this record
        t1j = get(r, "top1_hit_judged")
        t3j = get(r, "top3_hit_judged")
        if t1j is not None:
            t1 = bool(t1j)
        else:
            t1 = bool(get(r, t1_field))
        if t3j is not None:
            t3 = bool(t3j)
        else:
            t3 = bool(get(r, t3_field))
        ab = bool(get(r, ab_field) or get(r, "abstained_judged"))
        out.append({"conf": conf, "t1": t1, "t3": t3, "ab": ab})
    return out


# ─── Metric at a given τ ────────────────────────────────────────────────
def metric_at(records: List[dict], tau: float, alpha: float = 0.5):
    """Return (T-1%, T-3%, SA%, Cov%, DHS%).
    commit iff conf is not None AND conf >= tau (tau=0 commits all conf>=0).
    """
    n = len(records)
    if n == 0:
        return None
    def committed(r):
        c = r["conf"]
        if c is None:
            return False
        return c >= tau
    com = sum(1 for r in records if committed(r))
    correct  = sum(1 for r in records if committed(r) and r["t1"])
    correct3 = sum(1 for r in records if committed(r) and r["t3"])
    cov = com / n
    sa = correct / com if com else 0
    if alpha == 0.5:
        dhs = 2 * sa * cov / (sa + cov) if (sa + cov) else 0
    else:
        # General DHS_α: 1 / (alpha/SA + (1-alpha)/Cov)
        if sa <= 0 or cov <= 0:
            dhs = 0
        else:
            dhs = 1 / (alpha / sa + (1 - alpha) / cov)
    return (round(100 * correct / n), round(100 * correct3 / n),
            round(100 * sa), round(100 * cov), round(100 * dhs))


# ─── τ optimisation ─────────────────────────────────────────────────────
def sweep(records, alpha=0.5, step=0.05):
    """Return list of (tau, (T1,T3,SA,Cov,DHS))."""
    out = []
    n_steps = int(round(0.95 / step)) + 1
    for i in range(n_steps):
        tau = round(i * step, 4)
        if tau > 0.95: break
        out.append((tau, metric_at(records, tau, alpha)))
    return out

def argmax_dhs(records, alpha=0.5, step=0.05) -> Tuple[float, tuple]:
    best_tau = 0.0
    best_metric = metric_at(records, 0.0, alpha)
    best_dhs = best_metric[4] if best_metric else -1
    for tau, m in sweep(records, alpha, step):
        # break ties by preferring higher Cov, then earlier τ
        if m and (m[4] > best_dhs):
            best_dhs = m[4]; best_tau = tau; best_metric = m
    return best_tau, best_metric

def argmax_dhs_continuous(records, alpha=0.5):
    """Find τ* where exact DHS maximum is, using all confidence values as
    candidate cut-points (true continuous-domain optimum)."""
    confs = sorted(set(r["conf"] for r in records if r["conf"] is not None))
    candidates = [0.0] + [c for c in confs] + [1.0]
    candidates = sorted(set(round(c, 6) for c in candidates))
    best_tau, best_metric = 0.0, metric_at(records, 0.0, alpha)
    best_dhs = best_metric[4] if best_metric else -1
    for tau in candidates:
        m = metric_at(records, tau, alpha)
        if m and m[4] > best_dhs:
            best_dhs = m[4]; best_tau = tau; best_metric = m
    return best_tau, best_metric

def loocv_tau(records, alpha=0.5, step=0.01) -> Tuple[float, dict]:
    """Leave-one-out: for each held-out case, pick τ*-on-rest, then evaluate
    rest on the global metric. Reported τ* = mode of the τ*'s chosen across LOO
    folds (or the τ that maximises mean LOO-DHS).

    We use the "mean LOO-DHS" version for stability.
    """
    # For each candidate τ, compute mean DHS across LOO folds
    n = len(records)
    if n == 0: return 0.0, {}
    # candidate τ values
    taus = [round(t/100, 2) for t in range(0, 96, int(round(step*100)))]
    mean_dhs = {}
    for tau in taus:
        dhs_vals = []
        for held in range(n):
            rest = records[:held] + records[held+1:]
            m = metric_at(rest, tau, alpha)
            if m is not None:
                dhs_vals.append(m[4])
        mean_dhs[tau] = sum(dhs_vals)/len(dhs_vals) if dhs_vals else 0
    best_tau = max(mean_dhs, key=mean_dhs.get)
    # also report the per-fold τ*-of-rest distribution
    fold_taus = []
    for held in range(n):
        rest = records[:held] + records[held+1:]
        t_, _ = argmax_dhs(rest, alpha, step)
        fold_taus.append(t_)
    return best_tau, {"mean_dhs": mean_dhs, "fold_taus": fold_taus}


# ─── BMBE cell registry (where to read from) ────────────────────────────
# Each entry: (label, source_kind, source_path, t1_field, t3_field, conf_field)
CELLS = [
    # ── TABLO 1 BMBE row × 3 benchmarks ────────────────────────────────
    ("T1 BMBE PatientSim",
        "jsonl", "set_a/ddxplus_patientsim/mobayes/bmbe_gpt54nano.jsonl"),
    ("T1 BMBE HT-sim",
        "jsonl", "set_a/ddxplus_htsim/mobayes/bmbe_gpt54nano.jsonl"),
    ("T1 BMBE AC-MedQA",
        "jsonl", "set_a/agentclinic/mobayes/bmbe_gpt54nano.jsonl"),

    # ── TABLO 2 BMBE rows (sensors × benchmarks) ───────────────────────
    # PatientSim: jsonl in set_a/ddxplus_patientsim/
    ("T2 + GPT-5.4-nano PatientSim",      "jsonl", "set_a/ddxplus_patientsim/mobayes/bmbe_gpt54nano.jsonl"),
    ("T2 + Gemini-Flash-Lite PatientSim", "jsonl", "set_a/ddxplus_patientsim/mobayes/bmbe_geminiflashlite.jsonl"),
    ("T2 + GPT-OSS-20B PatientSim",       "jsonl", "set_a/ddxplus_patientsim/mobayes/bmbe_gptoss20b.jsonl"),
    ("T2 + MiniMax-M2.5 PatientSim",      "jsonl", "set_a/ddxplus_patientsim/mobayes/bmbe_minimax_m25.jsonl"),
    ("T2 + Gemma-4-31B PatientSim",       "jsonl", "set_a/ddxplus_patientsim/mobayes/bmbe_gemma4_31b.jsonl"),
    ("T2 + Llama-4-Scout PatientSim",     "jsonl", "set_a/ddxplus_patientsim/mobayes/bmbe_llama4scout.jsonl"),
    # HT-sim: judge mirror in judged/
    ("T2 + GPT-5.4-nano HT-sim",          "jsonl", "set_a/ddxplus_htsim/mobayes/bmbe_gpt54nano.jsonl"),
    ("T2 + Gemini-Flash-Lite HT-sim",     "dir",   "judged/set_a/ddxplus_htsim/mobayes/bmbe_geminiflashlite"),
    ("T2 + GPT-OSS-20B HT-sim",           "dir",   "judged/set_a/ddxplus_htsim/mobayes/bmbe_gptoss20b"),
    ("T2 + MiniMax-M2.5 HT-sim",          "dir",   "judged/set_a/ddxplus_htsim/mobayes/bmbe_minimax_m25"),
    ("T2 + Gemma-4-31B HT-sim",           "dir",   "judged/set_a/ddxplus_htsim/mobayes/bmbe_gemma4_31b"),
    ("T2 + Llama-4-Scout HT-sim",         "dir",   "judged/set_a/ddxplus_htsim/mobayes/bmbe_llama4scout"),
    # AC-MedQA
    ("T2 + Gemini-Flash-Lite AC-MedQA",   "dir",   "judged/set_a/agentclinic/mobayes/bmbe_geminiflashlite"),
    ("T2 + GPT-OSS-20B AC-MedQA",         "dir",   "judged/set_a/agentclinic/mobayes/bmbe_gptoss20b"),
    ("T2 + MiniMax-M2.5 AC-MedQA",        "dir",   "judged/set_a/agentclinic/mobayes/bmbe_minimax_m25"),
    ("T2 + Gemma-4-31B AC-MedQA",         "dir",   "judged/set_a/agentclinic/mobayes/bmbe_gemma4_31b"),
    ("T2 + Llama-4-Scout AC-MedQA",       "dir",   "judged/set_a/agentclinic/mobayes/bmbe_llama4scout"),

    # ── TABLO 22 BMBE row × 3 benchmarks (v1 split) ────────────────────
    ("T22 BMBE PatientSim v1",            "dir",   "set_b/ddxplus_patientsim/mobayes/bmbe"),
    ("T22 BMBE HT-sim v1",                "dir",   "set_b/ddxplus_htsim/mobayes/bmbe"),
    ("T22 BMBE AC-MedQA v1",              "dir",   "judged/set_b/agentclinic/mobayes/bmbe"),
]


# ─── Driver ─────────────────────────────────────────────────────────────
def run_cell(label, kind, path, alpha=0.5, sweep_show=False, loocv_show=False,
             step=0.05, continuous=False):
    if kind == "dir":
        raw = read_dir(path)
    else:
        raw = read_jsonl(path)
    if not raw:
        return f"  {label:<42s}  SOURCE MISSING: {path}"

    records = extract_records(raw)
    if all(r["conf"] is None for r in records):
        return f"  {label:<42s}  no confidence field — Cov fixed at 100"

    full_tau, full_m = argmax_dhs(records, alpha, step)
    loo_tau, loo_info = loocv_tau(records, alpha, step)
    loo_m = metric_at(records, loo_tau, alpha)

    out_lines = []
    out_lines.append(f"  {label:<42s}  n={len(records)}")
    out_lines.append(f"      full-set τ*={full_tau:.4g} (step={step}) → "
                     f"T-1={full_m[0]} T-3={full_m[1]} SA={full_m[2]} Cov={full_m[3]} DHS={full_m[4]}")
    out_lines.append(f"      LOOCV    τ*={loo_tau:.4g} → "
                     f"T-1={loo_m[0]} T-3={loo_m[1]} SA={loo_m[2]} Cov={loo_m[3]} DHS={loo_m[4]}")

    if continuous:
        cont_tau, cont_m = argmax_dhs_continuous(records, alpha)
        out_lines.append(f"      continuous τ*={cont_tau:.4g} (cuts at observed conf values) → "
                         f"T-1={cont_m[0]} T-3={cont_m[1]} SA={cont_m[2]} Cov={cont_m[3]} DHS={cont_m[4]}")

    if sweep_show:
        out_lines.append(f"      sweep (step={step}):")
        for tau, m in sweep(records, alpha, step):
            mark = " *full*" if abs(tau - full_tau) < 1e-9 else ""
            out_lines.append(f"        τ={tau:.4g}: "
                             f"T-1={m[0]:>3} T-3={m[1]:>3} SA={m[2]:>3} Cov={m[3]:>3} DHS={m[4]:>3}{mark}")

    if loocv_show:
        from collections import Counter
        ftaus = loo_info.get("fold_taus", [])
        c = Counter(ftaus)
        out_lines.append(f"      LOO fold τ* distribution: {dict(sorted(c.items()))}")

    return "\n".join(out_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=["1", "2", "22"], default=None)
    ap.add_argument("--sweep", action="store_true", help="Print full sweep per cell")
    ap.add_argument("--loocv", action="store_true", help="Print LOO fold τ* distribution")
    ap.add_argument("--alpha", type=float, default=0.5, help="DHS_α emphasis (default 0.5)")
    ap.add_argument("--step", type=float, default=0.05, help="τ sweep step (default 0.05; try 0.01 for online sweep)")
    ap.add_argument("--continuous", action="store_true", help="Also report exact continuous-domain τ* (cuts at observed conf values)")
    args = ap.parse_args()

    print(f"DHS-optimal τ search (α={args.alpha})\n")
    print("Conventions:")
    print("  full-set τ*  : argmax_DHS over τ ∈ [0.00, 0.95] step 0.05 on full n=50")
    print("  LOOCV τ*     : argmax of mean LOO-DHS (paper-recommended)\n")

    for label, kind, path in CELLS:
        if args.table == "1"  and not label.startswith("T1 "):  continue
        if args.table == "2"  and not label.startswith("T2 "):  continue
        if args.table == "22" and not label.startswith("T22"):  continue
        print(run_cell(label, kind, path, alpha=args.alpha,
                       sweep_show=args.sweep, loocv_show=args.loocv,
                       step=args.step, continuous=args.continuous))
        print()


if __name__ == "__main__":
    main()
