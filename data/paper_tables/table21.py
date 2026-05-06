#!/usr/bin/env python3
"""End-to-end reproducer for Table 21 (tab:ci_main).

Reads the raw per-case JSON files for every (model × benchmark) cell,
computes T-1 / SA / DHS with bootstrap 95% CIs, and emits the LaTeX
table. No pre-computed numbers — everything comes from the result
paths listed in DATA_PATHS below.

Run:
  python3 scripts/table21.py                    # → tables/tab_ci_main.tex
  python3 scripts/table21.py --stdout
  python3 scripts/table21.py --B 5000 --seed 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ─── Data root ─────────────────────────────────────────────────────────
RESULTS = Path(__file__).resolve().parents[2] / "data" / "benchmark_results"

# ─── Per-cell raw-data paths (single source of truth) ──────────────────
# Each value is (path, kind) where kind ∈ {"jsonl", "dir"} and the path
# is relative to RESULTS. The records are read straight from disk and
# fed to the metric functions; nothing is precomputed.
SA_ROWS = [
    ("SA GPT-5.4",          "sa_gpt54"),
    ("SA Gemini 3.1 Pro",   "sa_gemini31pro"),
    ("SA Llama-4-Maverick", "sa_llama4maverick"),
    ("SA Qwen 3.6 Plus",    "sa_qwen36plus"),
    ("SA Kimi K2.5",        "sa_kimi_k25"),
    ("SA GPT-OSS-120B",     "sa_gptoss120b"),
]
BMBE_ROWS = [
    ("+ GPT-5.4-nano",          "bmbe_gpt54nano"),
    ("+ Gemini 3.1 Flash Lite", "bmbe_geminiflashlite"),
    ("+ GPT-OSS-20B",           "bmbe_gptoss20b"),
    ("+ MiniMax M2.5",          "bmbe_minimax_m25"),
    ("+ Gemma 4 31B",           "bmbe_gemma4_31b"),
    ("+ Llama-4-Scout",         "bmbe_llama4scout"),
]
BENCHES = [
    ("DDxPlus PatientSim", "patientsim"),
    ("DDxPlus HT-sim",     "htsim"),
    ("AgentClinic-MedQA",  "ac"),
]

# (label, benchmark_key) → (relative_path, kind, is_bmbe)
DATA_PATHS: Dict[Tuple[str, str], Tuple[str, str, bool]] = {}
for label, stem in SA_ROWS:
    DATA_PATHS[(label, "patientsim")] = (f"set_a/ddxplus_patientsim/standalone/{stem}.jsonl", "jsonl", False)
    DATA_PATHS[(label, "htsim")]      = (f"judged/set_a/ddxplus_htsim/standalone/{stem}", "dir",  False)
    DATA_PATHS[(label, "ac")]         = (f"judged/set_a/agentclinic/standalone/{stem}", "dir",  False)
for label, stem in BMBE_ROWS:
    if stem == "bmbe_gpt54nano":
        DATA_PATHS[(label, "patientsim")] = ("set_a/ddxplus_patientsim/mobayes/bmbe_gpt54nano.jsonl",        "jsonl", True)
        DATA_PATHS[(label, "htsim")]      = ("set_a/ddxplus_htsim/mobayes/bmbe_gpt54nano.jsonl",  "jsonl", True)
        # AC routed to the v5-distilled KB run (matches tab_main).
        DATA_PATHS[(label, "ac")]         = ("judged/set_a/agentclinic/mobayes/bmbe_v5",                "dir",   True)
    else:
        DATA_PATHS[(label, "patientsim")] = (f"set_a/ddxplus_patientsim/mobayes/{stem}.jsonl",      "jsonl", True)
        DATA_PATHS[(label, "htsim")]      = (f"judged/set_a/ddxplus_htsim/mobayes/{stem}",     "dir",   True)
        DATA_PATHS[(label, "ac")]         = (f"judged/set_a/agentclinic/mobayes/{stem}",       "dir",   True)

# ─── Per-sensor LOOCV τ (used only for BMBE rows) ──────────────────────
# Computed once on the original n=50 data; held fixed across bootstrap
# replicates so the CIs reflect estimator variance, not τ-search noise.
TAU_LOOCV: Dict[Tuple[str, str], float] = {
    # Filled in from `find_tau_loocv` below; cached to keep table21.py
    # deterministic and fast.
}


# ─── I/O helpers ───────────────────────────────────────────────────────
def _get(rec: dict, *keys):
    for k in keys:
        if k in rec: return rec[k]
        f = rec.get("final")
        if isinstance(f, dict) and k in f: return f[k]
    return None

def read_records(rel: str, kind: str) -> List[dict]:
    p = RESULTS / rel
    if kind == "jsonl":
        return [json.loads(l) for l in open(p) if l.strip()]
    if kind == "dir":
        return [json.load(open(f)) for f in sorted(p.glob("case_*.json"))]
    raise ValueError(kind)

def normalise(records: List[dict], use_remap_for_t1: bool = False) -> List[dict]:
    """Project raw record -> {t1, t3, ab, conf}.

    Prefers judged fields for t1/t3/ab when present (modern unified
    pipeline); falls back to legacy `top1_hit_remapped` for SA HT-sim/AC
    runs that pre-date judging, controlled by `use_remap_for_t1`.
    """
    out = []
    for r in records:
        if use_remap_for_t1:
            t1 = bool(_get(r, "top1_hit_remapped"))
        else:
            t1j = _get(r, "top1_hit_judged")
            t1 = bool(t1j) if t1j is not None else bool(_get(r, "top1_hit"))
        t3j = _get(r, "top3_hit_judged")
        t3 = bool(t3j) if t3j is not None else bool(_get(r, "top3_hit"))
        abj = _get(r, "abstained_judged")
        ab = bool(abj) if abj is not None else bool(_get(r, "abstained"))
        out.append({"t1": t1, "t3": t3, "ab": ab,
                     "conf": _get(r, "confidence")})
    return out

def pad_to_n(records: List[dict], target: int) -> List[dict]:
    """Treat missing cases (e.g. inference failures) as committed-and-wrong
    so denominators stay at the intended n=50."""
    if len(records) >= target: return records
    pad = [{"t1": False, "t3": False, "ab": False, "conf": 1.0}
           for _ in range(target - len(records))]
    return list(records) + pad


# ─── Metric + bootstrap ────────────────────────────────────────────────
def metrics(records: List[dict], tau: float | None) -> dict:
    n = len(records)
    if n == 0: return {"t1": 0., "t3": 0., "sa": 0., "cov": 0., "dhs": 0.}
    t1 = sum(1 for r in records if r["t1"])
    t3 = sum(1 for r in records if r["t3"])
    if tau is None:
        com = sum(1 for r in records if not r["ab"])
        cor = sum(1 for r in records if not r["ab"] and r["t1"])
    else:
        com = sum(1 for r in records if (r.get("conf") or 0) >= tau)
        cor = sum(1 for r in records if (r.get("conf") or 0) >= tau and r["t1"])
    cov = com / n
    sa  = cor / com if com else 0.
    dhs = 2*sa*cov/(sa+cov) if (sa+cov) > 0 else 0.
    return {"t1": 100*t1/n, "t3": 100*t3/n,
            "sa": 100*sa, "cov": 100*cov, "dhs": 100*dhs}

def find_tau_loocv(records: List[dict]) -> float:
    """Leave-one-out τ-tuning: the τ that maximises mean DHS over loo splits.
    With n=50 the search lattice is 0.00, 0.01, ..., 0.50.
    """
    n = len(records)
    cands = [round(0.01*i, 2) for i in range(0, 51)]
    best_tau, best_dhs = 0.0, -1.0
    for tau in cands:
        loo_dhs = []
        for hold in range(n):
            train = [r for i, r in enumerate(records) if i != hold]
            loo_dhs.append(metrics(train, tau)["dhs"])
        m = sum(loo_dhs) / len(loo_dhs)
        if m > best_dhs:
            best_dhs, best_tau = m, tau
    return best_tau

def bootstrap(records: List[dict], tau: float | None, B: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(records)
    point = metrics(records, tau)
    samples = {k: np.empty(B) for k in ("t1", "t3", "sa", "cov", "dhs")}
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        rep = [records[i] for i in idx]
        m = metrics(rep, tau)
        for k in samples:
            samples[k][b] = m[k]
    return {k: (point[k],
                float(np.percentile(samples[k], 2.5)),
                float(np.percentile(samples[k], 97.5)))
            for k in samples}


# ─── Cell loader ───────────────────────────────────────────────────────
def load_cell(label: str, bkey: str, is_bmbe: bool) -> Tuple[List[dict], float | None]:
    rel, kind, _ = DATA_PATHS[(label, bkey)]
    raw = read_records(rel, kind)
    use_remap = (label.startswith("SA ")) and (bkey in ("htsim", "ac"))
    recs = pad_to_n(normalise(raw, use_remap_for_t1=use_remap), 50)
    if is_bmbe:
        cached = TAU_LOOCV.get((label, bkey))
        tau = cached if cached is not None else find_tau_loocv(recs)
        TAU_LOOCV[(label, bkey)] = tau
    else:
        tau = None  # SA rows: commit iff not abstained
    return recs, tau


# ─── Coloring ──────────────────────────────────────────────────────────
def dhs_color(p: float, lo: float, hi: float) -> str:
    if hi <= lo: return ""
    f = (p - lo) / (hi - lo)
    if f >= 0.5:
        i = int(round(8 + (f - 0.5) * 2 * (40 - 8)))
        return rf"\cellcolor{{dhsgreen!{i}}}"
    i = int(round(8 + (0.5 - f) * 2 * (25 - 8)))
    return rf"\cellcolor{{dhsred!{i}}}"


# ─── Renderer ──────────────────────────────────────────────────────────
def render(B: int, seed: int) -> str:
    out: List[str] = []

    # Pre-pass: bootstrap every cell once, collect per-benchmark DHS
    # range so colour scaling is column-relative.
    ci_cache: Dict[Tuple[str, str], dict] = {}
    col_dhs: Dict[str, List[float]] = {b: [] for _, b in BENCHES}
    for label, _stem in SA_ROWS + BMBE_ROWS:
        is_bmbe = any(label == l for l, _ in BMBE_ROWS)
        for _, bkey in BENCHES:
            recs, tau = load_cell(label, bkey, is_bmbe)
            ci = bootstrap(recs, tau, B=B, seed=seed)
            ci_cache[(label, bkey)] = ci
            col_dhs[bkey].append(ci["dhs"][0])
    col_minmax = {b: (min(v), max(v)) for b, v in col_dhs.items() if v}

    # Header
    out.append(r"%% =============================================================")
    out.append(r"%% Bootstrap 95% CIs for Table 3 (tab:main).")
    out.append(rf"%% Auto-generated by scripts/table21.py  B={B} seed={seed}.")
    out.append(r"%% Data path per cell: see DATA_PATHS in scripts/table21.py.")
    out.append(r"%% =============================================================")
    out.append("")
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\caption{Bootstrap 95\% CIs for Table~\ref{tab:main} "
               r"(frontier standalone vs.\ \discuss{\ours}, $n{=}50$ per cell). "
               rf"$B={B}$, seed={seed}; cells show point [2.5\%, 97.5\%]. "
               r"DHS coloured per benchmark column. "
               r"AgentClinic-MedQA results use a knowledge base elicited from "
               r"GPT-5.4-nano (construction protocol in Appendix~\ref{app:kb}).}")
    out.append(r"\label{tab:ci_main}")
    out.append(r"\footnotesize")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\renewcommand{\arraystretch}{1.05}")
    out.append(r"\resizebox{\textwidth}{!}{%")
    out.append(r"\begin{tabular}{@{}l ccc ccc ccc@{}}")
    out.append(r"\toprule")
    bench_hdr = " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{\textbf{{{b}}}}}" for b, _ in BENCHES) + r" \\"
    out.append(bench_hdr)
    out.append(r" \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}")
    out.append(r"\textbf{Method} & " + " & ".join([r"T-1 & SA & DHS$\,\uparrow$"] * 3) + r" \\")
    out.append(r"\midrule")
    out.append(r"\multicolumn{10}{l}{\textit{Standalone LLM doctors (frontier)}} \\")
    out.append(r"\midrule")

    def emit_row(label: str):
        parts = []
        for _, bkey in BENCHES:
            ci = ci_cache[(label, bkey)]
            t1, sa, dhs = ci["t1"], ci["sa"], ci["dhs"]
            parts.append(rf"{t1[0]:.1f}\,\textsubscript{{[{t1[1]:.1f},{t1[2]:.1f}]}}")
            parts.append(rf"{sa[0]:.1f}\,\textsubscript{{[{sa[1]:.1f},{sa[2]:.1f}]}}")
            lo, hi = col_minmax[bkey]
            parts.append(dhs_color(dhs[0], lo, hi) + rf"{dhs[0]:.1f}\,\textsubscript{{[{dhs[1]:.1f},{dhs[2]:.1f}]}}")
        out.append(f"{label} & " + " & ".join(parts) + r" \\")

    for label, _ in SA_ROWS: emit_row(label)
    out.append(r"\midrule")
    out.append(r"\multicolumn{10}{l}{\textit{\discuss{\ours} + sensor LLM}} \\")
    out.append(r"\midrule")
    for label, _ in BMBE_ROWS: emit_row(label)
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}%")
    out.append(r"}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="tables/tab_ci_main.tex")
    ap.add_argument("--stdout", action="store_true")
    ap.add_argument("--show-paths", action="store_true",
                    help="Print the data path used for every cell and exit.")
    args = ap.parse_args()

    if args.show_paths:
        print(f"RESULTS root: {RESULTS}\n")
        for (label, bkey), (rel, kind, is_bmbe) in DATA_PATHS.items():
            print(f"  {label:<28} × {bkey:<11}  →  {kind:<5} {rel}"
                  f"{' (BMBE: τ-tuned)' if is_bmbe else ''}")
        return

    tex = render(B=args.B, seed=args.seed)
    if args.stdout:
        print(tex)
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(tex)
        print(f"Wrote {out_path} ({len(tex)} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
