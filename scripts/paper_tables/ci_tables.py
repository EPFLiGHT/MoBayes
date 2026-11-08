"""ci_tables.py — Bootstrap 95% CIs for every row of paper Tables 2 and 3.

Tables in scope:
  - main       (tab_main.tex, paper Table 2): SA frontier × 6 + MoBayes sensor × 6,
                3 benchmarks, n=50 canonical only, metrics: T-1 / SA / Cov / DHS.
  - combined   (tab_combined_n100.tex, paper Table 3): 12 baseline rows + MoBayes,
                3 benchmarks, n=100 (canonical 50 + v1 50), metrics: T-1 / T-3 / SA / Cov / DHS.

Methodology (per cell):
  1. Read the per-case records (top1_hit, top3_hit, abstained, confidence).
  2. For BMBE: derive τ* once via LOOCV on the original data (per half for
     'combined'); lock it across all bootstrap replicates.
  3. Bootstrap B=2000: resample n indices with replacement; recompute metrics.
  4. Report point estimate (full data) and 95% percentile CI.

Output:
  - Default: markdown summary, one row per (method × benchmark)
  - --latex: LaTeX rows where each cell is "point [lo, hi]" (subscript-friendly)

Usage:
  python3 scripts/paper_tables/ci_tables.py --table main
  python3 scripts/paper_tables/ci_tables.py --table combined --B 5000
  python3 scripts/paper_tables/ci_tables.py --latex
  python3 scripts/paper_tables/ci_tables.py --appendix
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Reuse helpers from build_combined_n100.py for the combined table
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_combined_n100 import (   # noqa: E402
    METHODS as COMBINED_METHODS,
    BENCHES,
    cell_records as combined_cell_records,
    find_tau_loocv,
    normalise, normalise_remap, read_jsonl, read_dir,
    _read_mediq,
)


# ─── Table 2 (tab_main.tex) cell definitions ──────────────────────────
T2_SA_ROWS = [
    ("SA GPT-5.4",          "sa_gpt54"),
    ("SA Gemini 3.1 Pro",   "sa_gemini31pro"),
    ("SA Llama-4-Maverick", "sa_llama4maverick"),
    ("SA Qwen 3.6 Plus",    "sa_qwen36plus"),
    ("SA Kimi K2.5",        "sa_kimi_k25"),
    ("SA GPT-OSS-120B",     "sa_gptoss120b"),
]
T2_BMBE_ROWS = [
    ("+ GPT-5.4-nano",          "bmbe_gpt54nano"),
    ("+ Gemini 3.1 Flash Lite", "bmbe_geminiflashlite"),
    ("+ GPT-OSS-20B",           "bmbe_gptoss20b"),
    ("+ MiniMax M2.5",          "bmbe_minimax_m25"),
    ("+ Gemma 4 31B",           "bmbe_gemma4_31b"),
    ("+ Llama-4-Scout",         "bmbe_llama4scout"),
]
T2_BENCHES = [
    ("PatientSim", "patientsim"),
    ("HT-sim",     "htsim"),
    ("AC-MedQA",   "ac"),
]


def _pad_to_n(records: list, target: int) -> list:
    """Pad to `target` with worst-case missed-and-committed dummy records.

    Used for cells that lost runs to inference failure (e.g. BMBE +
    GPT-OSS-20B PatientSim is missing one case). Counting the missing
    case as a committed-and-wrong record (t1=False, ab=False, conf=1.0)
    gives the pessimistic per-method accounting the user asked for and
    keeps the denominator at the intended n=50 (so percentages don't
    surface as 30/49 = 61.2%).
    """
    if len(records) >= target:
        return records
    pad = [{"t1": False, "t3": False, "ab": False, "conf": 1.0}
           for _ in range(target - len(records))]
    return list(records) + pad


def t2_records(stem: str, benchmark: str) -> list:
    """Resolve canonical n=50 records for a Table 2 cell.

    SA rows: PatientSim → set_a jsonl (strict t1); HT-sim/AC → raw
             dirs with `top1_hit_remapped` (legacy lenient remap).
    BMBE rows: PatientSim → set_a jsonl (with confidence for τ);
               HT-sim/AC → judged dir (judge fields).
    """
    # SA × benchmark
    if stem.startswith("sa_"):
        if benchmark == "patientsim":
            return _pad_to_n(normalise(read_jsonl(f"set_a/ddxplus_patientsim/standalone/{stem}.jsonl")), 50)
        if benchmark == "htsim":
            return _pad_to_n(normalise_remap(read_dir(f"judged/set_a/ddxplus_htsim/standalone/{stem}")), 50)
        if benchmark == "ac":
            return _pad_to_n(normalise_remap(read_dir(f"judged/set_a/agentclinic/standalone/{stem}")), 50)

    # BMBE × benchmark
    if stem == "bmbe_gpt54nano":
        if benchmark == "patientsim":
            return _pad_to_n(normalise(read_jsonl("set_a/ddxplus_patientsim/mobayes/bmbe_gpt54nano.jsonl")), 50)
        if benchmark == "htsim":
            return _pad_to_n(normalise(read_jsonl("set_a/ddxplus_htsim/mobayes/bmbe_gpt54nano.jsonl")), 50)
        if benchmark == "ac":
            # AC uses the v5-distilled KB run (the path tab_main was hand-set
            # against; old set_a jsonl is from a pre-v5 run).
            return _pad_to_n(normalise(read_dir("judged/set_a/agentclinic/mobayes/bmbe_v5")), 50)

    # Other BMBE sensors
    if benchmark == "patientsim":
        return _pad_to_n(normalise(read_jsonl(f"set_a/ddxplus_patientsim/mobayes/{stem}.jsonl")), 50)
    if benchmark == "htsim":
        return _pad_to_n(normalise(read_dir(f"judged/set_a/ddxplus_htsim/mobayes/{stem}")), 50)
    if benchmark == "ac":
        return _pad_to_n(normalise(read_dir(f"judged/set_a/agentclinic/mobayes/{stem}")), 50)
    return []


# ─── Per-replicate metric ─────────────────────────────────────────────
def metric_replicate(records: list, tau_per_half=(None, None)) -> dict:
    """Compute (T-1, T-3, SA, Cov, DHS) on a (resampled) record list.

    For BMBE with tau_per_half: commit iff conf >= τ (chosen per record's _half tag,
    or the single τ if both halves use the same value).
    For non-BMBE / no τ: commit iff not r['ab'].
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
            tau = tau_can if r.get("_half") != "v1" else tau_v1
            if tau is None:
                return not r["ab"]
            c = r.get("conf")
            return False if c is None else c >= tau
        com = sum(1 for r in records if committed(r))
        cor = sum(1 for r in records if committed(r) and r["t1"])
    cov = com / n
    sa = cor / com if com else 0.0
    dhs = 2 * sa * cov / (sa + cov) if (sa + cov) > 0 else 0.0
    return {"t1": 100*t1/n, "t3": 100*t3/n, "sa": 100*sa, "cov": 100*cov, "dhs": 100*dhs}


def bootstrap_cell(records, tau_per_half=(None, None), B=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(records)
    point = metric_replicate(records, tau_per_half)
    samples = {k: np.empty(B) for k in ("t1", "t3", "sa", "cov", "dhs")}
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        rep = [records[i] for i in idx]
        m = metric_replicate(rep, tau_per_half)
        for k in samples:
            samples[k][b] = m[k]
    return {k: (point[k],
                float(np.percentile(samples[k], 2.5)),
                float(np.percentile(samples[k], 97.5)))
            for k in samples}


# ─── Drivers ──────────────────────────────────────────────────────────
def main_table_cells():
    """Yield (label, benchmark_label, records, tau_per_half) for Table 2."""
    for label, stem in T2_SA_ROWS:
        for blabel, bkey in T2_BENCHES:
            recs = t2_records(stem, bkey)
            yield ("main", label, blabel, recs, (None, None), stem, bkey)
    for label, stem in T2_BMBE_ROWS:
        for blabel, bkey in T2_BENCHES:
            recs = t2_records(stem, bkey)
            tau = None
            if recs and any(r["conf"] is not None for r in recs):
                tau = find_tau_loocv(recs)
            yield ("main", label, blabel, recs, (tau, tau), stem, bkey)


def combined_table_cells():
    """Yield (label, benchmark_label, records, tau_per_half, mkey, bkey) for Table 3 (combined n=100)."""
    for label, mkey in COMBINED_METHODS:
        for blabel, bkey in BENCHES:
            can, v1 = combined_cell_records(mkey, bkey)
            if not can or not v1:
                yield ("combined", label, blabel, [], (None, None), mkey, bkey); continue
            for r in can: r["_half"] = "can"
            for r in v1:  r["_half"] = "v1"
            recs = can + v1
            tau_per_half = (None, None)
            if mkey == "BMBE":
                tau_can = find_tau_loocv(can) if any(r["conf"] is not None for r in can) else None
                tau_v1  = find_tau_loocv(v1)  if any(r["conf"] is not None for r in v1)  else None
                tau_per_half = (tau_can, tau_v1)
            yield ("combined", label, blabel, recs, tau_per_half, mkey, bkey)


def fmt_cell(point, lo, hi, prec=1):
    return f"{point:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]"

def _tokens_main(stem: str, bkey: str) -> int:
    """Token total for a Tablo 2 cell (canonical n=50). Routes by SA/BMBE stem.
    Returns 0 if doctor-side tokens were not tracked."""
    from build_combined_n100 import read_jsonl, read_dir
    if stem.startswith("sa_"):
        if bkey == "patientsim": rs = read_jsonl(f"set_a/ddxplus_patientsim/standalone/{stem}.jsonl")
        elif bkey == "htsim":    rs = read_dir(f"judged/set_a/ddxplus_htsim/standalone/{stem}")
        else:                    rs = read_dir(f"judged/set_a/agentclinic/standalone/{stem}")
    elif stem == "bmbe_gpt54nano":
        if bkey == "patientsim": rs = read_jsonl("set_a/ddxplus_patientsim/mobayes/bmbe_gpt54nano.jsonl")
        elif bkey == "htsim":    rs = read_jsonl("set_a/ddxplus_htsim/mobayes/bmbe_gpt54nano.jsonl")
        else:                    rs = read_jsonl("set_a/agentclinic/mobayes/bmbe_gpt54nano.jsonl")
    else:
        cat = "mobayes" if stem.startswith("bmbe") else "baselines"
        if bkey == "patientsim": rs = read_jsonl(f"set_a/ddxplus_patientsim/{cat}/{stem}.jsonl")
        elif bkey == "htsim":    rs = read_dir(f"judged/set_a/ddxplus_htsim/{cat}/{stem}")
        else:                    rs = read_dir(f"judged/set_a/agentclinic/{cat}/{stem}")
    if not _doctor_tracked(rs):
        return 0
    return sum(_tokens_one_record(r) for r in rs)


def _tokens_one_record(r: dict, doctor_only: bool = False, patient_only: bool = False) -> int:
    """Sum tokens for a single raw record. Handles three shapes:

      (a) bench/baseline runner: top-level `token_usage = {doctor:{...}, patient:{...}}`
          with prompt_tokens/completion_tokens (OpenAI naming).
      (b) MediQ raw log: `interactive_system.temp_additional_info[].usage` with
          input_tokens/output_tokens (Anthropic naming) per question turn.
      (c) Mixed: some records have nested final.token_usage.

    `doctor_only=True` returns only the doctor-side tokens; `patient_only=True`
    only patient-side. Default sums both.
    """
    total = 0

    # (a)/(c) — top-level or final-nested token_usage. Sum across ALL roles
    # found in the dict (doctor, patient, measurement, moderator, ...) so we
    # don't miss AgentClinic's auxiliary agents (measurement/moderator).
    tu = r.get("token_usage") or (r.get("final", {}) or {}).get("token_usage") or {}
    if doctor_only:
        roles_to_sum = ("doctor",)
    elif patient_only:
        roles_to_sum = ("patient",)
    else:
        # Include every role that has a dict-shaped sub-entry with token counts.
        roles_to_sum = tuple(k for k, v in tu.items()
                             if isinstance(v, dict) and any(
                                 kk in v for kk in ("prompt_tokens", "completion_tokens",
                                                    "input_tokens", "output_tokens")))
    for role in roles_to_sum:
        t = tu.get(role) or {}
        total += int(t.get("prompt_tokens", 0) or t.get("input_tokens", 0) or 0)
        total += int(t.get("completion_tokens", 0) or t.get("output_tokens", 0) or 0)

    # (b) — MediQ raw logs: per-question usage under interactive_system
    # MediQ patient simulator doesn't go through the same path; usage entries are
    # the doctor's confidence-grading LLM calls. Treat as doctor-side.
    if not patient_only:
        isys = r.get("interactive_system") or {}
        for entry in isys.get("temp_additional_info") or []:
            if not entry:
                continue
            u = entry.get("usage") or {}
            total += int(u.get("prompt_tokens", 0) or u.get("input_tokens", 0) or 0)
            total += int(u.get("completion_tokens", 0) or u.get("output_tokens", 0) or 0)

    return total

def _doctor_tracked(records: list) -> bool:
    """Whether doctor-side tokens were tracked in this set of raw records.
    Returns False if every record has 0 doctor.prompt_tokens (i.e., runner bug
    or self-hosted model)."""
    if not records:
        return False
    sum_doctor = sum(_tokens_one_record(r, doctor_only=True) for r in records)
    return sum_doctor > 0

# Auxiliary token_usage.json files for MEDDxAgent (and other HPC runs that
# don't store tokens in their per-case results). Path is relative to RESULTS root.
# Each file has shape {"usage": {"doctor": {...}, "patient": {...}}, ...config...}.
MEDDX_AUX_TOKENS = {
    # canonical
    ("MEDDxAgent-pb", "patientsim", "can"): None,  # already in jsonl
    ("MEDDxAgent-pb", "htsim",      "can"): "set_a/ddxplus_htsim/_raw/meddxagent_paperbest/20260502_201308/token_usage.json",
    ("MEDDxAgent-pb", "ac",         "can"): "set_a/agentclinic/_raw/meddxagent_paperbest/token_usage.json",
    ("MEDDxAgent-cw", "patientsim", "can"): None,  # already in jsonl
    ("MEDDxAgent-cw", "htsim",      "can"): "set_a/ddxplus_htsim/_raw/meddxagent_cwabl/20260502_233138/token_usage.json",
    ("MEDDxAgent-cw", "ac",         "can"): "set_a/agentclinic/_raw/meddxagent_cwabl/20260503_001840/token_usage.json",
    # v1
    ("MEDDxAgent-pb", "patientsim", "v1"):  "set_b/ddxplus_patientsim/baselines/meddxagent_hpc_raw/20260505_123929/token_usage.json",
    ("MEDDxAgent-pb", "htsim",      "v1"):  "set_b/ddxplus_htsim/baselines/meddxagent_hpc_raw/20260505_133301/token_usage.json",
    ("MEDDxAgent-pb", "ac",         "v1"):  "set_b/agentclinic/baselines/meddxagent_hpc_raw/20260505_171151/token_usage.json",
    ("MEDDxAgent-cw", "patientsim", "v1"):  "set_b/ddxplus_patientsim/baselines/meddxagent_cw_raw/20260504_214724/token_usage.json",
    ("MEDDxAgent-cw", "htsim",      "v1"):  "set_b/ddxplus_htsim/baselines/meddxagent_cw_raw/20260504_214724/token_usage.json",
    ("MEDDxAgent-cw", "ac",         "v1"):  "set_b/agentclinic/baselines/meddxagent_cw_raw/20260505_204941/token_usage.json",
}

def _aux_tokens(rel_path: str) -> int:
    """Read MEDDxAgent's token_usage.json `usage` subfield."""
    if not rel_path:
        return 0
    from build_combined_n100 import RESULTS as _R
    p = _R / rel_path
    if not p.exists():
        return 0
    try:
        d = json.loads(p.read_text())
    except Exception:
        return 0
    u = d.get("usage") or {}
    total = 0
    for role in ("doctor", "patient"):
        t = u.get(role) or {}
        total += int(t.get("prompt_tokens", 0) or t.get("input_tokens", 0) or 0)
        total += int(t.get("completion_tokens", 0) or t.get("output_tokens", 0) or 0)
    return total

def cell_tokens(mkey: str, bkey: str, half: str = "both") -> int:
    """Sum tokens across canonical 50 + v1 50 (or one half) for a cell.

    Returns 0 (-> displayed as `--`) if doctor-side tokens were not tracked
    by the runner (UoT runner only logs patient sim; CoD-34B is GPU-local
    with explicit `note: 'per-call usage not tracked'`). Patient-only numbers
    would be misleadingly low — better to mark as not tracked.
    MEDDxAgent has auxiliary `token_usage.json` files that DO contain real
    doctor + patient totals, so we use those.
    """
    raw_can, raw_v1 = _raw_records(mkey, bkey)
    total = 0
    has_aux = mkey in ("MEDDxAgent-pb", "MEDDxAgent-cw")
    if half in ("both", "can"):
        if _doctor_tracked(raw_can) or has_aux:
            total += sum(_tokens_one_record(r) for r in raw_can)
        if has_aux:
            total += _aux_tokens(MEDDX_AUX_TOKENS.get((mkey, bkey, "can")) or "")
    if half in ("both", "v1"):
        if _doctor_tracked(raw_v1) or has_aux:
            total += sum(_tokens_one_record(r) for r in raw_v1)
        if has_aux:
            total += _aux_tokens(MEDDX_AUX_TOKENS.get((mkey, bkey, "v1")) or "")
    return total


def cell_n_with_tokens(mkey: str, bkey: str, half: str = "both") -> int:
    """Number of cases that contributed to cell_tokens — i.e. the correct
    denominator for tok/case averaging. Halves whose runner did not track
    tokens (and which have no aux file) are excluded so that cells like
    BMBE PatientSim canonical (50/50 records empty, no aux) don't deflate
    the combined per-case average.
    """
    raw_can, raw_v1 = _raw_records(mkey, bkey)
    n = 0
    has_aux = mkey in ("MEDDxAgent-pb", "MEDDxAgent-cw")
    if half in ("both", "can"):
        if _doctor_tracked(raw_can) or has_aux:
            n += len(raw_can)
    if half in ("both", "v1"):
        if _doctor_tracked(raw_v1) or has_aux:
            n += len(raw_v1)
    return n

def _raw_records(mkey: str, bkey: str):
    """Return raw (canonical, v1) record lists with token_usage preserved."""
    M, B = mkey, bkey
    if B == "patientsim":
        if M == "CW":           return read_jsonl("set_a/ddxplus_patientsim/baselines/cw_gpt54nano.jsonl"), read_dir("set_b/ddxplus_patientsim/baselines/cw")
        if M == "CoT":          return read_jsonl("set_a/ddxplus_patientsim/baselines/cot_gpt54nano.jsonl"), read_dir("set_b/ddxplus_patientsim/baselines/cot")
        if M == "DDx-CoT":      return read_jsonl("set_a/ddxplus_patientsim/baselines/savage_dd_gpt54nano.jsonl"), read_dir("set_b/ddxplus_patientsim/baselines/savage_dd")
        if M == "BCoT":         return read_jsonl("set_a/ddxplus_patientsim/baselines/savage_bay_gpt54nano.jsonl"), read_dir("set_b/ddxplus_patientsim/baselines/savage_bay")
        if M == "UoT":          return read_jsonl("set_a/ddxplus_patientsim/baselines/uot_gpt54nano.jsonl"), read_dir("set_b/ddxplus_patientsim/baselines/uot")
        if M == "MediQ-force":  return _read_mediq("set_a/ddxplus_patientsim/baselines/mediq_raw/results.jsonl") or [], _read_mediq("set_b/ddxplus_patientsim/baselines/mediq_raw/results.jsonl") or []
        if M == "MediQ-strict": return _read_mediq("set_a/ddxplus_patientsim/baselines/mediq_raw/results.jsonl") or [], _read_mediq("set_b/ddxplus_patientsim/baselines/mediq_raw/results.jsonl") or []
        if M == "MEDDxAgent-pb":return read_jsonl("set_a/ddxplus_patientsim/baselines/meddxagent_pubmed_dynamic_gpt54nano.jsonl"), read_dir("judged/set_b/ddxplus_patientsim/baselines/meddxagent")
        if M == "MEDDxAgent-cw":return read_jsonl("set_a/ddxplus_patientsim/baselines/meddxagent_gpt54nano.jsonl"), read_dir("set_b/ddxplus_patientsim/baselines/meddxagent_cw")
        if M == "AC-native":    return read_dir("judged/set_a/ddxplus_patientsim/baselines/ac_native"), read_dir("set_b/ddxplus_patientsim/baselines/ac_native")
        if M == "CoD":          return read_jsonl("set_a/ddxplus_patientsim/baselines/cod_diagnosisgpt34b.jsonl"), read_dir("judged/set_b/ddxplus_patientsim/baselines/cod_v1")
        if M == "BMBE":         return read_jsonl("set_a/ddxplus_patientsim/mobayes/bmbe_gpt54nano.jsonl"), read_dir("set_b/ddxplus_patientsim/mobayes/bmbe")
    if B == "htsim":
        if M == "CW":           return read_jsonl("set_a/ddxplus_htsim/baselines/cw_gpt54nano_remapped.jsonl"), read_dir("set_b/ddxplus_htsim/baselines/cw")
        if M == "CoT":          return read_jsonl("set_a/ddxplus_htsim/baselines/cot_gpt54nano_remapped.jsonl"), read_dir("set_b/ddxplus_htsim/baselines/cot")
        if M == "DDx-CoT":      return read_jsonl("set_a/ddxplus_htsim/baselines/savage_dd_gpt54nano_remapped.jsonl"), read_dir("set_b/ddxplus_htsim/baselines/savage_dd")
        if M == "BCoT":         return read_jsonl("set_a/ddxplus_htsim/baselines/savage_bay_gpt54nano_remapped.jsonl"), read_dir("set_b/ddxplus_htsim/baselines/savage_bay")
        if M == "UoT":          return read_dir("judged/set_a/ddxplus_htsim/baselines/uot"), read_dir("set_b/ddxplus_htsim/baselines/uot")
        if M == "MediQ-force":  return read_jsonl("set_a/ddxplus_htsim/baselines/mediq_expert_gpt54nano_remapped.jsonl"), _read_mediq("set_b/ddxplus_htsim/baselines/mediq_raw/results.jsonl") or []
        if M == "MediQ-strict": return read_jsonl("set_a/ddxplus_htsim/baselines/mediq_expert_gpt54nano.jsonl"), _read_mediq("set_b/ddxplus_htsim/baselines/mediq_raw/results.jsonl") or []
        if M == "MEDDxAgent-pb":return read_jsonl("set_a/ddxplus_htsim/baselines/meddxagent_pubmed_dynamic_gpt54nano_remapped.jsonl"), read_dir("judged/set_b/ddxplus_htsim/baselines/meddxagent")
        if M == "MEDDxAgent-cw":return read_jsonl("set_a/ddxplus_htsim/baselines/meddxagent_closedworld_gpt54nano_remapped.jsonl"), read_dir("set_b/ddxplus_htsim/baselines/meddxagent_cw")
        if M == "AC-native":    return read_dir("judged/set_a/ddxplus_htsim/baselines/ac_native"), read_dir("set_b/ddxplus_htsim/baselines/ac_native")
        if M == "CoD":          return read_jsonl("set_a/ddxplus_htsim/baselines/cod_diagnosisgpt34b_remapped.jsonl"), read_dir("judged/set_b/ddxplus_htsim/baselines/cod_v1")
        if M == "BMBE":         return read_jsonl("set_a/ddxplus_htsim/mobayes/bmbe_gpt54nano.jsonl"), read_dir("set_b/ddxplus_htsim/mobayes/bmbe")
    if B == "ac":
        if M == "CW":           return read_jsonl("set_a/agentclinic/baselines/cw_gpt54nano.jsonl"), read_dir("judged/set_b/agentclinic/baselines/cw")
        if M == "CoT":          return read_jsonl("set_a/agentclinic/baselines/cot_gpt54nano.jsonl"), read_dir("judged/set_b/agentclinic/baselines/cot")
        if M == "DDx-CoT":      return read_jsonl("set_a/agentclinic/baselines/savage_dd_gpt54nano.jsonl"), read_dir("judged/set_b/agentclinic/baselines/savage_dd")
        if M == "BCoT":         return read_jsonl("set_a/agentclinic/baselines/savage_bay_gpt54nano.jsonl"), read_dir("judged/set_b/agentclinic/baselines/savage_bay")
        if M == "UoT":          return read_dir("judged/set_a/agentclinic/baselines/uot"), read_dir("judged/set_b/agentclinic/baselines/uot")
        if M == "MediQ-force":  return (_read_mediq("set_a/agentclinic/baselines/mediq_raw/results_0_10.jsonl", "set_a/agentclinic/baselines/mediq_raw/results_10_50.jsonl") or []), _read_mediq("set_b/agentclinic/baselines/mediq_raw/results.jsonl") or []
        if M == "MediQ-strict": return (_read_mediq("set_a/agentclinic/baselines/mediq_raw/results_0_10.jsonl", "set_a/agentclinic/baselines/mediq_raw/results_10_50.jsonl") or []), _read_mediq("set_b/agentclinic/baselines/mediq_raw/results.jsonl") or []
        if M == "MEDDxAgent-pb":return read_jsonl("set_a/agentclinic/baselines/meddxagent_pubmed_dynamic_gpt54nano.jsonl"), read_dir("judged/set_b/agentclinic/baselines/meddxagent")
        if M == "MEDDxAgent-cw":return read_jsonl("set_a/agentclinic/baselines/meddxagent_closedworld_gpt54nano.jsonl"), read_dir("judged/set_b/agentclinic/baselines/meddxagent_cw")
        if M == "AC-native":    return read_jsonl("set_a/agentclinic/baselines/ac_native_gpt54nano.jsonl"), read_dir("judged/set_b/agentclinic/baselines/ac_native")
        if M == "CoD":          return read_jsonl("set_a/agentclinic/baselines/cod_diagnosisgpt34b_v2.jsonl"), read_dir("judged/set_b/agentclinic/baselines/cod")
        if M == "BMBE":         return read_jsonl("set_a/agentclinic/mobayes/bmbe_gpt54nano.jsonl"), read_dir("judged/set_b/agentclinic/mobayes/bmbe_v5")
    return [], []

def _fmt_tokens(n: int) -> str:
    if n <= 0: return "--"
    if n >= 1_000_000: return f"{n/1_000_000:.2f}M"
    if n >= 1_000:     return f"{n/1_000:.0f}K"
    return str(n)


def render_md(table_id: str, cells, B: int, seed: int):
    print(f"# Bootstrap 95% CIs (B={B}, seed={seed}) — {table_id}\n")
    print("Format: point [2.5%, 97.5%] (one decimal).\n")
    print("τ for BMBE locked at LOOCV(full-data) and held fixed across replicates.\n")
    cur_method = None
    for tid, label, blabel, recs, taus, _mkey, _bkey in cells:
        if label != cur_method:
            print(f"\n## {label}")
            cur_method = label
        if not recs:
            print(f"  - {blabel}: SOURCE MISSING"); continue
        ci = bootstrap_cell(recs, tau_per_half=taus, B=B, seed=seed)
        tau_info = ""
        if taus != (None, None):
            tc = "—" if taus[0] is None else f"{taus[0]:.2f}"
            tv = "—" if taus[1] is None else f"{taus[1]:.2f}"
            tau_info = f"  (τ_can={tc}, τ_v1={tv})" if tc != tv else f"  (τ*={tc})"
        print(f"  - {blabel} (n={len(recs)}){tau_info}")
        # Table 2 (main) shows only T-1; the rest show T-1 and T-3
        metrics_to_show = ("t1", "sa", "cov", "dhs") if tid == "main" else ("t1", "t3", "sa", "cov", "dhs")
        for k in metrics_to_show:
            p, lo, hi = ci[k]
            label_k = "T-1" if k == "t1" else "T-3" if k == "t3" else k.upper()
            print(f"      {label_k:>4s}: {fmt_cell(p, lo, hi)}")


def render_latex(table_id: str, cells, B: int, seed: int):
    print(f"% Bootstrap 95% CIs (B={B}, seed={seed}) — {table_id}\n")
    cur_method = None
    row_parts: list = []
    for tid, label, blabel, recs, taus, _mkey, _bkey in cells:
        if label != cur_method and cur_method is not None:
            print(f"{cur_method:<32s} & " + " & ".join(row_parts) + r" \\")
            row_parts = []
        cur_method = label
        if not recs:
            metric_count = 4 if tid == "main" else 5
            row_parts.extend(["--"] * metric_count); continue
        ci = bootstrap_cell(recs, tau_per_half=taus, B=B, seed=seed)
        # Table 2 (main) shows only T-1; the rest show T-1 and T-3
        metrics_to_show = ("t1", "sa", "cov", "dhs") if tid == "main" else ("t1", "t3", "sa", "cov", "dhs")
        for k in metrics_to_show:
            p, lo, hi = ci[k]
            row_parts.append(f"{p:.1f}\\,\\textsubscript{{[{lo:.1f},{hi:.1f}]}}")
    if cur_method is not None:
        print(f"{cur_method:<32s} & " + " & ".join(row_parts) + r" \\")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=["main", "combined", "all"], default="all",
                    help="main=Table 2, combined=Table 3, all=both")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--appendix", action="store_true",
                    help="Emit a full appendix tex with both CI tables (each with caption + label).")
    args = ap.parse_args()

    cells_main = list(main_table_cells())
    cells_combined = list(combined_table_cells())

    sections = [
        ("main",      "paper Table 2 (tab_main.tex, n=50 canonical)",      cells_main),
        ("combined",  "paper Table 3 (tab_combined_n100.tex, n=100)",      cells_combined),
    ]
    if args.appendix:
        render_appendix_tex(sections, args.B, args.seed)
        return
    render = render_latex if args.latex else render_md
    for tag, title, cells in sections:
        if args.table != "all" and args.table != tag:
            continue
        render(title, cells, args.B, args.seed)
        print()


def _tokens_for_cell(tag: str, mkey: str, bkey: str) -> int:
    """Token total for a cell. tag: 'baselines'=canonical only, 'v1'=v1 only,
    'combined'=both, 'main'=canonical (t2_records), via stem mkey."""
    if tag == "main":
        return _tokens_main(mkey, bkey)
    if tag == "baselines":
        return cell_tokens(mkey, bkey, half="can")
    if tag == "v1":
        return cell_tokens(mkey, bkey, half="v1")
    if tag == "combined":
        return cell_tokens(mkey, bkey, half="both")
    return 0


def _ncases_for_cell(tag: str, mkey: str, bkey: str) -> int:
    """Number of cases that actually contributed to cell_tokens (the
    correct denominator for tok/case). Halves with no token tracking
    AND no aux file are excluded so they don't deflate the average.
    Falls back to 50/100 only when the cell has no contributing half."""
    if tag in ("baselines", "main"):
        n = cell_n_with_tokens(mkey, bkey, half="can")
        return n if n > 0 else 50
    if tag == "v1":
        n = cell_n_with_tokens(mkey, bkey, half="v1")
        return n if n > 0 else 50
    if tag == "combined":
        n = cell_n_with_tokens(mkey, bkey, half="both")
        return n if n > 0 else 100
    return 0


def _fmt_tok_per_case(tokens: int, n_cases: int) -> str:
    """Format avg tokens-per-case (full conversation). Returns '--' if either is 0."""
    if tokens <= 0 or n_cases <= 0:
        return "--"
    avg = tokens / n_cases
    if avg >= 1_000_000:
        return f"{avg/1_000_000:.2f}M"
    if avg >= 1000:
        return f"{avg/1000:.1f}K"
    return f"{int(round(avg))}"


# ─── Dialog-tokenize estimate fallback for UoT and CoD ──────────────────
_EST_TOKENS_CACHE: Optional[dict] = None
def _load_estimated_tokens() -> dict:
    global _EST_TOKENS_CACHE
    if _EST_TOKENS_CACHE is not None:
        return _EST_TOKENS_CACHE
    from build_combined_n100 import CODEBASE as _C
    p = _C / "data" / "benchmark_results" / "costs" / "estimated_tokens.json"
    if not p.exists():
        _EST_TOKENS_CACHE = {}
    else:
        _EST_TOKENS_CACHE = json.loads(p.read_text())
    return _EST_TOKENS_CACHE


# ─── Real-token samples (costs/) ──────────────────────────────────
# UoT: 5-case real-token samples per (benchmark, half) under costs/
#  uot_<bench>_<half>/case_*.json. Falls back to uot_patientsim_can if a
#  cell-specific sample is missing (older state when only PS canonical
#  was sampled).
# CoD: Yi-34B offline tokenizer pass over saved dialogs (per cell).
_UOT_REAL_CACHE: Dict[str, float] = {}

_BENCH_DIR = {"patientsim": "ddxplus_patientsim", "htsim": "ddxplus_htsim", "ac": "agentclinic"}
_SET_LETTER = {"can": "a", "v1": "b"}

def _uot_dir_for(bkey: str, tag: str) -> str:
    """Map (bench, half-tag) to the data/benchmark_results/costs/uot_samples/ subdir name."""
    half = "v1" if tag == "v1" else ("can" if tag in ("baselines", "main") else "combined")
    if half == "combined":
        # combined cell averages can+v1; we still pick a primary subdir;
        # _uot_real_avg_per_case will average across both when present.
        return f"{_BENCH_DIR[bkey]}_a"
    return f"{_BENCH_DIR[bkey]}_{_SET_LETTER[half]}"

def _uot_avg_in_dir(dirname: str) -> Optional[float]:
    from build_combined_n100 import RESULTS as _R
    d = _R / "costs" / "uot_samples" / dirname
    if not d.exists():
        return None
    files = sorted(d.glob("case_*.json"))
    if not files:
        return None
    totals = []
    for f in files:
        rec = json.loads(f.read_text())
        dtu = (rec.get("token_usage") or {}).get("doctor") or {}
        t = dtu.get("total_tokens") or (
            (dtu.get("prompt_tokens", 0) or 0) +
            (dtu.get("completion_tokens", 0) or 0))
        if t > 0:
            totals.append(t)
    return (sum(totals) / len(totals)) if totals else None

def _uot_real_avg_per_case(bkey: str = "patientsim", tag: str = "baselines"
                             ) -> Optional[float]:
    """Per-cell tok/case from costs/uot_<bench>_<half>/.

    For combined-table cells, averages can + v1 sample dirs when both
    exist; falls back to whichever is present. Final fallback is
    uot_patientsim_can (the original 5-case smoke).
    """
    cache_key = f"{bkey}|{tag}"
    if cache_key in _UOT_REAL_CACHE:
        v = _UOT_REAL_CACHE[cache_key]
        return v if v > 0 else None

    if tag == "combined":
        a = _uot_avg_in_dir(f"{_BENCH_DIR[bkey]}_a")
        b = _uot_avg_in_dir(f"{_BENCH_DIR[bkey]}_b")
        if a is not None and b is not None:
            v = (a + b) / 2
        elif a is not None:  v = a
        elif b is not None:  v = b
        else:                v = None
    else:
        v = _uot_avg_in_dir(_uot_dir_for(bkey, tag))

    if v is None:
        # last-resort fallback: original 5-case PS canonical smoke
        v = _uot_avg_in_dir("ddxplus_patientsim_a")

    _UOT_REAL_CACHE[cache_key] = v if v else -1.0
    return v

_COD_YI_CACHE: Optional[dict] = None
def _load_cod_yi() -> dict:
    """Per-cell tok/case from CoD Yi-34B offline pass.

    Prefer v2 (reconstructs CoD's <|user|>\\n...\\n<|assistant|>\\n template
    + 3 internal passes per turn + retrieved-disease overhead), fall back to
    v1 (naive turn-by-turn sum) if v2 is missing.
    """
    global _COD_YI_CACHE
    if _COD_YI_CACHE is not None:
        return _COD_YI_CACHE
    from build_combined_n100 import CODEBASE as _C
    p = _C / "data" / "benchmark_results" / "costs" / "cod_yi34b_token_pass.json"
    if p.exists():
        _COD_YI_CACHE = json.loads(p.read_text())
        return _COD_YI_CACHE
    _COD_YI_CACHE = {}
    return _COD_YI_CACHE

def _cod_yi_per_case(bkey: str, tag: str) -> Optional[float]:
    """Look up CoD Yi-34B offline tok/case for a (benchmark, half) cell."""
    yi = _load_cod_yi()
    halves = ("can",) if tag in ("baselines", "main") else \
             ("v1",)  if tag == "v1" else ("can", "v1")
    vals = []
    for h in halves:
        e = yi.get(f"CoD_{bkey}_{h}")
        if isinstance(e, dict) and e.get("status") == "ok" and e.get("n_cases"):
            vals.append(e["tok_per_case"])
    return (sum(vals) / len(vals)) if vals else None

def _estimated_tok_per_case(method_label: str, bkey: str, tag: str) -> Optional[float]:
    """For UoT and CoD: real-sample or tokenizer-pass estimate.

    Priority:
      UoT  → costs/uot_samples/ avg (5-case real run, applied
             uniformly to all UoT cells — cost driver is benchmark-agnostic).
      CoD  → _cod_yi34b_token_pass.json (Yi-34B offline; covers PS-can,
             HT-can; falls back to cl100k_base estimator for other cells).
      else → estimated_tokens.json (cl100k_base lower-bound).
    """
    label_to_key = {
        "UoT":                       "UoT",
        "CoD / DiagnosisGPT-34B":    "CoD",
    }
    method = label_to_key.get(method_label.replace("~\\cite{hu2024uot}", "").strip())
    if method is None:
        return None
    if method == "UoT":
        v = _uot_real_avg_per_case(bkey, tag)
        if v is not None:
            return v
    if method == "CoD":
        v = _cod_yi_per_case(bkey, tag)
        if v is not None:
            return v
    est = _load_estimated_tokens()
    halves = ("can",) if tag in ("baselines","main") else \
             ("v1",)  if tag == "v1" else ("can", "v1")
    tot_tok = tot_n = 0
    for h in halves:
        e = est.get(f"{method}_{bkey}_{h}")
        if not e: continue
        tot_tok += int(e.get("total_tokens_estimated", 0))
        tot_n   += int(e.get("n_cases", 0))
    if tot_tok <= 0 or tot_n <= 0:
        return None
    return tot_tok / tot_n


def _fmt_tok_per_case_with_estimate(method_label, bkey, tag, tok, n_cases) -> str:
    if tok > 0 and n_cases > 0:
        return _fmt_tok_per_case(tok, n_cases)
    est = _estimated_tok_per_case(method_label, bkey, tag)
    if est is None:
        return "--"
    if est >= 1_000_000: s = f"{est/1_000_000:.2f}M"
    elif est >= 1000:    s = f"{est/1000:.1f}K"
    else:                s = f"{int(round(est))}"
    return s


def render_appendix_tex(sections, B, seed):
    """Emit a full appendix tex file with both CI tables.

    Compact format per benchmark: T-1 [CI] | DHS [CI] | Tokens.
    """
    print(r"%% ============================================================")
    print(r"%% Bootstrap 95% CIs for the main and combined tables.")
    print(r"%% Auto-generated by scripts/paper_tables/ci_tables.py --appendix.")
    print(rf"%% B={B}, seed={seed}.  Cells show T-1, DHS (point [lo, hi]) plus token count.")
    print(r"%% ============================================================")
    print()

    table_meta = [
        ("main",      "tab:ci_main",
         r"Bootstrap 95\% CIs for Table~\ref{tab:main} (frontier standalone vs.\ \discuss{\ours}, $n{=}50$ per cell)."),
        ("combined",  "tab:ci_combined",
         r"Bootstrap 95\% CIs and token totals for Table~\ref{tab:combined_n100} (combined $n{=}100$)."),
    ]
    bench_labels = ["DDxPlus PatientSim", "DDxPlus HT-sim", "AgentClinic-MedQA"]
    # Per-table metric layout. ci_main shows T-1 + SA + DHS (no Tok/case;
    # cost analysis lives elsewhere). The SA column makes commit-aware
    # accuracy explicit so abstain-heavy frontier models aren't penalised
    # by naive T-1 (cf. GPT-5.4 vs GPT-OSS-120B on HT-sim: T-1 close, SA gap +21).
    METRIC_LAYOUT = {
        "main":      {"n_metrics": 3, "metric_hdr": r"T-1 & SA & DHS$\,\uparrow$",       "show_sa": True,  "show_tok": False, "color_dhs": True},
        "combined":  {"n_metrics": 3, "metric_hdr": r"T-1 & DHS$\,\uparrow$ & Tok/case", "show_sa": False, "show_tok": True,  "color_dhs": True},
    }

    def _dhs_color_relative(p: float, col_min: float, col_max: float) -> str:
        """Per-column gradient: best DHS in this benchmark → dhsgreen!40,
        worst → dhsred!25, linear blend in between via cell intensity.

        Uses two distinct colour anchors so neither saturates: top half maps
        to green intensity 8 -> 40, bottom half to red intensity 8 -> 25.
        """
        if col_max <= col_min:
            return ""
        frac = (p - col_min) / (col_max - col_min)  # 0 (worst) -> 1 (best)
        if frac >= 0.5:
            # green half: 0.5 → !8, 1.0 → !40
            intensity = int(round(8 + (frac - 0.5) * 2 * (40 - 8)))
            return rf"\cellcolor{{dhsgreen!{intensity}}}"
        else:
            # red half: 0.5 → !8, 0.0 → !25
            intensity = int(round(8 + (0.5 - frac) * 2 * (25 - 8)))
            return rf"\cellcolor{{dhsred!{intensity}}}"
    # Category groupings for the row order (matches MAIN tables 1, 22)
    CATEGORIES = [
        ("Prompt-based", {"Closed-World (CW)", "Chain-of-Thought (CoT)", "DDx-CoT", "Bayesian-CoT"}),
        ("Information-pursuit", {"UoT"}),
        ("Agent-based", {"MediQ-Expert (force-final)", "MediQ-Expert (strict abstain)",
                          "MEDDxAgent (paper-best)", "MEDDxAgent (closed-world abl.)", "AgentClinic"}),
        ("Fine-tuned (specialised model)", {"CoD / DiagnosisGPT-34B"}),
        ("Ours", {"\\textbf{\\ours} (gpt-5.4-nano)"}),
    ]
    # Tablo 2 (main) — different grouping: SA frontier vs MoBayes sensors
    CATEGORIES_MAIN = [
        ("Standalone LLM doctors (frontier)", set(label for label, _ in T2_SA_ROWS)),
        ("\\discuss{\\ours} + sensor LLM", set(label for label, _ in T2_BMBE_ROWS)),
    ]
    def _category_for(method_label, cats):
        for cat_name, members in cats:
            if method_label in members:
                return cat_name
        return None

    for (tag, label, caption), (_t, _title, cells) in zip(table_meta, sections):
        layout = METRIC_LAYOUT[tag]
        n_metrics = layout["n_metrics"]
        metric_hdr = layout["metric_hdr"]
        show_tok = layout["show_tok"]
        show_sa = layout.get("show_sa", False)
        color_dhs = layout.get("color_dhs", False)
        col_spec = "@{}l" + (" ccc" if n_metrics == 3 else " cc") * 3 + "@{}"
        n_total_cols = 1 + 3 * n_metrics
        cmidrule = (r" \cmidrule(lr){2-" + str(1+n_metrics) + r"}"
                    r" \cmidrule(lr){" + str(2+n_metrics) + "-" + str(1+2*n_metrics) + r"}"
                    r" \cmidrule(lr){" + str(2+2*n_metrics) + "-" + str(1+3*n_metrics) + r"}")

        print(r"\begin{table}[t]")
        print(r"\centering")
        tok_clause = (
            r" Tok/case is the average doctor + patient prompt+completion tokens per"
            r" diagnostic conversation."
        ) if show_tok else ""
        # Compact caption: keep only the load-bearing facts.
        if show_sa:
            cap_body = (caption +
                        r" $B=" + str(B) + r"$" +
                        r"; cells show point [2.5\%, 97.5\%]."
                        r" DHS coloured per benchmark column.")
        else:
            cap_body = (caption +
                        r" $B=" + str(B) + r"$" +
                        r"; cells show point [2.5\%, 97.5\%]." +
                        tok_clause)
        print(r"\caption{" + cap_body + r"}")
        print(r"\label{" + label + r"}")
        print(r"\footnotesize")
        print(r"\setlength{\tabcolsep}{4pt}")
        print(r"\renewcommand{\arraystretch}{1.05}")
        print(r"\resizebox{\textwidth}{!}{%")
        print(r"\begin{tabular}{" + col_spec + r"}")
        print(r"\toprule")
        bench_hdr = " & " + " & ".join(r"\multicolumn{" + str(n_metrics) + r"}{c}{\textbf{" + b + r"}}" for b in bench_labels) + r" \\"
        print(bench_hdr)
        print(cmidrule)
        print(r"\textbf{Method} & " + metric_hdr + " & " + metric_hdr + " & " + metric_hdr + r" \\")
        print(r"\midrule")

        # Pre-pass: bootstrap every cell once and compute per-benchmark
        # DHS min/max for relative colour scaling.
        ci_cache: dict = {}
        col_dhs: dict = {}  # bkey -> list of point DHS
        for cell in cells:
            tid, m_label, _b, recs, taus, mkey, bkey = cell
            if not recs:
                ci_cache[(m_label, bkey)] = None
                continue
            ci = bootstrap_cell(recs, tau_per_half=taus, B=B, seed=seed)
            ci_cache[(m_label, bkey)] = ci
            col_dhs.setdefault(bkey, []).append(ci["dhs"][0])
        col_minmax = {b: (min(v), max(v)) for b, v in col_dhs.items() if v}

        # Choose category set per table tag
        cats = CATEGORIES_MAIN if tag == "main" else CATEGORIES

        # Optional: sort methods by avg DHS (descending) WITHIN each
        # category. Enabled for ci_combined; categories are preserved.
        sort_by_dhs = (tag == "combined")
        if sort_by_dhs:
            avg_dhs = {}
            for cell in cells:
                _tid, m_label, _b, _r, _t, _m, bkey = cell
                ci = ci_cache.get((m_label, bkey))
                if ci is None: continue
                avg_dhs.setdefault(m_label, []).append(ci["dhs"][0])
            avg_dhs = {m: sum(v)/len(v) for m, v in avg_dhs.items() if v}
            cat_order = {c[0]: i for i, c in enumerate(cats)}
            def cat_rank(m):
                cat = _category_for(m, cats)
                return cat_order.get(cat, 999)
            # Preserve benchmark column order: c[6] is bkey but BENCHES
            # defines the rendered order; sort within row by index there.
            bench_order = {bkey: i for i, (_, bkey) in enumerate(BENCHES)}
            cells = sorted(cells, key=lambda c: (cat_rank(c[1]),
                                                   -avg_dhs.get(c[1], 0),
                                                   bench_order.get(c[6], 999)))

        cur_method = None
        cur_category = None
        row_parts: list = []
        for cell in cells:
            tid, m_label, _b, recs, taus, mkey, bkey = cell
            if m_label != cur_method and cur_method is not None:
                print(f"{cur_method} & " + " & ".join(row_parts) + r" \\")
                row_parts = []
            # Insert a category header row when the method's category changes
            if m_label != cur_method:
                cat = _category_for(m_label, cats)
                if cat and cat != cur_category:
                    if cur_category is not None:
                        print(r"\midrule")
                    print(r"\multicolumn{" + str(n_total_cols) + r"}{l}{\textit{" + cat + r"}} \\")
                    print(r"\midrule")
                    cur_category = cat
            cur_method = m_label
            ci = ci_cache.get((m_label, bkey))
            if ci is None:
                row_parts.extend(["--"] * n_metrics); continue
            tok = _tokens_for_cell(tid, mkey, bkey)
            t1 = ci["t1"]; dhs = ci["dhs"]; sa = ci["sa"]
            row_parts.append(f"{t1[0]:.1f}\\,\\textsubscript{{[{t1[1]:.1f},{t1[2]:.1f}]}}")
            if show_sa:
                row_parts.append(f"{sa[0]:.1f}\\,\\textsubscript{{[{sa[1]:.1f},{sa[2]:.1f}]}}")
            dhs_cell = f"{dhs[0]:.1f}\\,\\textsubscript{{[{dhs[1]:.1f},{dhs[2]:.1f}]}}"
            if color_dhs and bkey in col_minmax:
                lo, hi = col_minmax[bkey]
                dhs_cell = _dhs_color_relative(dhs[0], lo, hi) + dhs_cell
            row_parts.append(dhs_cell)
            if show_tok:
                # Tok/case = total tokens / n_cases (full conversation cost per diagnosis)
                n_cases = _ncases_for_cell(tid, mkey, bkey)
                row_parts.append(_fmt_tok_per_case_with_estimate(m_label, bkey, tid, tok, n_cases))
        if cur_method is not None:
            print(f"{cur_method} & " + " & ".join(row_parts) + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}%")
        print(r"}")
        print(r"\end{table}")
        print()


if __name__ == "__main__":
    main()
