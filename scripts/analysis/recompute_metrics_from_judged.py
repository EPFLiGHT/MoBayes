"""Recompute T-1 / T-3 / SA / Cov / DHS for every judged per-method dir under
data/benchmark_results/judged/, using the LLM-judge fields:
    top1_hit_judged, top3_hit_judged, abstained_judged.

Falls back to the strict fields (top1_hit, top3_hit, abstained) if a case
is missing the judged ones.

Usage:
    python3 scripts/analysis/recompute_metrics_from_judged.py
        # tabular markdown to stdout

    python3 scripts/analysis/recompute_metrics_from_judged.py --latex
        # LaTeX cell strings (system | T-1 & SA & Cov & DHS) for paste-in
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data/benchmark_results/judged"


# (label, dir_stem) per row in the cross-benchmark Table 2 / Table 21.
# dir_stem is the suffix after the benchmark prefix.
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
COLUMNS = [
    ("DDxPlus PatientSim", "ddxplus_patientsim_50_"),
    ("DDxPlus HT-sim",      "ddxplus_htsim_50_"),
    ("AgentClinic-MedQA",   "agentclinic_50_"),
]


def _read_case(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _flag(case: dict, judged_key: str, fallback_key: str) -> Optional[bool]:
    """Prefer judged field; fall back to original strict field."""
    fin = case.get("final", {})
    if judged_key in fin:
        return bool(fin[judged_key])
    if fallback_key in fin:
        return bool(fin[fallback_key])
    if fallback_key in case:
        return bool(case[fallback_key])
    return None


def metrics_for_dir(d: Path) -> Optional[Tuple[int, float, float, float, float, int]]:
    """(n, T1%, T3%, SA%, Cov%, DHS%) — None if dir empty/missing."""
    cases = sorted(d.glob("case_*.json"))
    if not cases:
        return None
    n = len(cases)
    t1 = t3 = committed = correct_committed = 0
    for c in cases:
        case = _read_case(c)
        h1 = _flag(case, "top1_hit_judged", "top1_hit")
        h3 = _flag(case, "top3_hit_judged", "top3_hit")
        ab = _flag(case, "abstained_judged", "abstained")
        if h1: t1 += 1
        if h3: t3 += 1
        if ab is False:
            committed += 1
            if h1:
                correct_committed += 1
    t1_pct = 100.0 * t1 / n
    t3_pct = 100.0 * t3 / n
    cov_pct = 100.0 * committed / n
    sa_pct = (100.0 * correct_committed / committed) if committed else 0.0
    dhs_pct = (2 * sa_pct * cov_pct / (sa_pct + cov_pct)) if (sa_pct + cov_pct) > 0 else 0.0
    return (n, t1_pct, t3_pct, sa_pct, cov_pct, dhs_pct)


def collect_table() -> List[List[Optional[Tuple]]]:
    """rows × columns matrix of metric tuples (or None)."""
    out: List[List[Optional[Tuple]]] = []
    for label, stem in SA_ROWS + BMBE_ROWS:
        row: List[Optional[Tuple]] = [label]
        for col_label, col_prefix in COLUMNS:
            d = ROOT / f"{col_prefix}{stem}"
            row.append(metrics_for_dir(d) if d.exists() else None)
        out.append(row)
    return out


def render_markdown(matrix: List[List]) -> str:
    lines = []
    headers = ["System"] + [f"{c} (T1/T3/SA/Cov/DHS)" for c, _ in COLUMNS]
    lines.append(" | ".join(headers))
    lines.append(" | ".join(["---"] * len(headers)))
    for row in matrix:
        cells = [row[0]]
        for m in row[1:]:
            if m is None:
                cells.append("--")
            else:
                n, t1, t3, sa, cov, dhs = m
                cells.append(f"{t1:.0f}/{t3:.0f}/{sa:.0f}/{cov:.0f}/{dhs:.0f} (n={n})")
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def render_latex_row(label: str, m: Optional[Tuple]) -> str:
    if m is None:
        return "-- & -- & -- & --"
    _, t1, _, sa, cov, dhs = m
    return f"{t1:.0f} & {sa:.0f} & {cov:.0f} & {dhs:.0f}"


def render_latex(matrix: List[List]) -> str:
    """Cell strings ready to paste into tab_main.tex (T-1 / SA / Cov / DHS per benchmark)."""
    lines = []
    for row in matrix:
        label = row[0]
        cells = [render_latex_row(label, m) for m in row[1:]]
        lines.append(f"{label:<25} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true",
                    help="Output LaTeX cell strings instead of markdown")
    args = ap.parse_args()
    matrix = collect_table()
    if args.latex:
        print(render_latex(matrix))
    else:
        print(render_markdown(matrix))


if __name__ == "__main__":
    main()
