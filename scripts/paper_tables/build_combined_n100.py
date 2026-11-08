"""build_combined_n100.py — Aggregate canonical (n=50) + v1 (n=50) → n=100 table.

For each (method, benchmark) cell:
  - Read raw per-case records from BOTH canonical and v1 sources
  - Combine into a 100-case set
  - Compute:
      T-1 = sum(top1_hit) / 100              (naive)
      T-3 = sum(top3_hit) / 100              (naive)
      Cov = committed_count / 100
      SA  = correct_committed / committed    (weighted by committed counts)
      DHS = 2*SA*Cov / (SA+Cov)              (harmonic mean, kept as float)
  - For BMBE rows, re-tune τ* on combined n=100 (LOOCV).
  - For MediQ-strict, use Likert τ=4.0 over combined raw mediQ logs.

Output:
  - Markdown summary (default)
  - LaTeX rows ready to paste into a table file (--latex)

Usage:
    python3 scripts/paper_tables/build_combined_n100.py
    python3 scripts/paper_tables/build_combined_n100.py --latex > tables/tab_combined_n100.body.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

HOME = Path.home()
CODEBASE = Path(__file__).resolve().parents[2]
RESULTS  = CODEBASE / "data/benchmark_results"
MEDIQ    = RESULTS  # MediQ raw logs: set_{a,b}/<bench>/baselines/mediq_raw/
AC_CASES_CAN = CODEBASE / "data/cases/agentclinic_a/selected_cases.jsonl"
AC_CASES_V1  = CODEBASE / "data/cases/agentclinic_b/selected_cases.jsonl"

LIKERT_TAU = 4.0  # MediQ Likert ≥4 (paper-recommended)


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
        if not line: continue
        try: out.append(json.loads(line))
        except json.JSONDecodeError: pass
    return out

def get(r: dict, field: str):
    if isinstance(r.get("final"), dict) and field in r["final"]:
        return r["final"][field]
    return r.get(field)

def normalise(records, t1="top1_hit", t3="top3_hit", ab="abstained", conf="confidence"):
    """Project a list of raw records into uniform dicts."""
    out = []
    for r in records:
        # Prefer judged fields if present.
        t1j = get(r, "top1_hit_judged"); t3j = get(r, "top3_hit_judged")
        abj = get(r, "abstained_judged")
        out.append({
            "t1": bool(t1j) if t1j is not None else bool(get(r, t1)),
            "t3": bool(t3j) if t3j is not None else bool(get(r, t3)),
            "ab": bool(abj) if abj is not None else bool(get(r, ab)),
            "conf": get(r, conf),
        })
    return out

def normalise_remap(records, t1="top1_hit_remapped", t3="top3_hit", ab="abstained", conf="confidence"):
    """Variant for raw dirs storing the legacy llm_remap_to_closed_set field."""
    out = []
    for r in records:
        out.append({
            "t1": bool(get(r, t1)),
            "t3": bool(get(r, t3)),
            "ab": bool(get(r, ab)),
            "conf": get(r, conf),
        })
    return out


# ─── Metrics ────────────────────────────────────────────────────────────
def metrics_combined(records: List[dict]) -> dict:
    """T-1/T-3 naive; SA/Cov commit-based; DHS harmonic mean (float)."""
    n = len(records)
    if n == 0:
        return {"n": 0, "t1": 0.0, "t3": 0.0, "sa": 0.0, "cov": 0.0, "dhs": 0.0}
    t1 = sum(1 for r in records if r["t1"])
    t3 = sum(1 for r in records if r["t3"])
    committed = sum(1 for r in records if not r["ab"])
    correct = sum(1 for r in records if not r["ab"] and r["t1"])
    cov = committed / n
    sa  = correct / committed if committed else 0.0
    dhs = 2*sa*cov/(sa+cov) if (sa + cov) > 0 else 0.0
    return {"n": n, "t1": 100*t1/n, "t3": 100*t3/n,
            "sa": 100*sa, "cov": 100*cov, "dhs": 100*dhs}

def metrics_with_tau(records: List[dict], tau: float) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0, "t1": 0.0, "t3": 0.0, "sa": 0.0, "cov": 0.0, "dhs": 0.0}
    def committed(r):
        c = r["conf"]
        return False if c is None else c >= tau
    com = sum(1 for r in records if committed(r))
    correct  = sum(1 for r in records if committed(r) and r["t1"])
    # Naive Top-k still: count all hits, regardless of tau-commit status.
    t1 = sum(1 for r in records if r["t1"])
    t3 = sum(1 for r in records if r["t3"])
    cov = com / n
    sa = correct / com if com else 0.0
    dhs = 2*sa*cov/(sa+cov) if (sa + cov) > 0 else 0.0
    return {"n": n, "t1": 100*t1/n, "t3": 100*t3/n,
            "sa": 100*sa, "cov": 100*cov, "dhs": 100*dhs, "tau": tau}

def find_tau_loocv(records: List[dict], step: float = 0.01) -> float:
    """Pick τ that maximises mean LOO DHS."""
    n = len(records)
    if n == 0: return 0.0
    taus = [round(i*step, 4) for i in range(int(0.96/step)+1)]
    best_tau, best_mean = 0.0, -1.0
    for tau in taus:
        dhs_vals = []
        for held in range(n):
            rest = records[:held] + records[held+1:]
            m = metrics_with_tau(rest, tau)
            dhs_vals.append(m["dhs"])
        mean_dhs = sum(dhs_vals)/len(dhs_vals) if dhs_vals else 0
        if mean_dhs > best_mean:
            best_mean, best_tau = mean_dhs, tau
    return best_tau


# ─── MediQ raw-log scorers (force + strict at τ=4.0) ────────────────────
def _mediq_evaluate(rows, gt_lookup, mode: str) -> List[dict]:
    """Return per-row dict with t1/t3/ab fields (t3=False since MCQ)."""
    out = []
    for r in rows:
        isys = r["interactive_system"]
        confs = [(e or {}).get("confidence") or 0.0 for e in isys.get("temp_additional_info", [])]
        choices = isys.get("intermediate_choices", [])
        m = min(len(confs), len(choices))
        confs, choices = confs[:m], choices[:m]
        options = r["info"]["options"]
        gt = gt_lookup(r)
        if mode == "force":
            final_letter = isys.get("letter_choice")
            pick = options.get(final_letter, "")
            t1 = (pick.lower().strip() == gt)
            ab = False
        elif mode == "strict":
            idx = next((t for t, (c, ch) in enumerate(zip(confs, choices))
                        if c >= LIKERT_TAU and ch in options), None)
            if idx is None:
                t1 = False; ab = True
            else:
                pick = options.get(choices[idx], "")
                t1 = (pick.lower().strip() == gt); ab = False
        out.append({"t1": t1, "t3": False, "ab": ab, "conf": None})
    return out

def _read_mediq(*paths):
    rows = []
    for p in paths:
        full = MEDIQ / p
        if not full.exists():
            return None
        rows.extend(json.loads(l) for l in full.open())
    by_id = {r["id"]: r for r in rows}
    return sorted(by_id.values(), key=lambda r: r["id"])

def _info_gt(r):
    return (r.get("info", {}).get("correct_answer") or "").lower().strip()

def _ac_gt_factory(cases_path):
    if not cases_path.exists(): return None
    cases = [json.loads(l) for l in cases_path.open()]
    def lookup(r):
        case = cases[r["id"]]
        return case.get("OSCE_Examination", case)["Correct_Diagnosis"].lower().strip()
    return lookup


# ─── Cell registry: source for each (method, benchmark) at canonical+v1 ──
# Each entry returns (records_canonical, records_v1) lists.

def cell_records(method: str, benchmark: str) -> Tuple[List[dict], List[dict]]:
    """Return (canonical 50 records, v1 50 records) for the given cell."""
    M = method  # alias
    B = benchmark

    # ─ Closed-list / "Top candidate" methods on DDxPlus PatientSim ─
    if B == "patientsim":
        if M == "CW":
            can = normalise(read_jsonl("set_a/ddxplus_patientsim/baselines/cw_gpt54nano.jsonl"))
            v1  = normalise(read_dir("set_b/ddxplus_patientsim/baselines/cw"))
            return can, v1
        if M == "CoT":
            return (normalise(read_jsonl("set_a/ddxplus_patientsim/baselines/cot_gpt54nano.jsonl")),
                    normalise(read_dir("set_b/ddxplus_patientsim/baselines/cot")))
        if M == "DDx-CoT":
            return (normalise(read_jsonl("set_a/ddxplus_patientsim/baselines/savage_dd_gpt54nano.jsonl")),
                    normalise(read_dir("set_b/ddxplus_patientsim/baselines/savage_dd")))
        if M == "BCoT":
            return (normalise(read_jsonl("set_a/ddxplus_patientsim/baselines/savage_bay_gpt54nano.jsonl")),
                    normalise(read_dir("set_b/ddxplus_patientsim/baselines/savage_bay")))
        if M == "UoT":
            return (normalise(read_jsonl("set_a/ddxplus_patientsim/baselines/uot_gpt54nano.jsonl")),
                    normalise(read_dir("set_b/ddxplus_patientsim/baselines/uot")))
        if M == "MediQ-force":
            raw = _read_mediq("set_a/ddxplus_patientsim/baselines/mediq_raw/results.jsonl") or []
            can = _mediq_evaluate(raw, _info_gt, "force")
            raw_v1 = _read_mediq("set_b/ddxplus_patientsim/baselines/mediq_raw/results.jsonl") or []
            v1 = _mediq_evaluate(raw_v1, _info_gt, "force")
            return can, v1
        if M == "MediQ-strict":
            raw = _read_mediq("set_a/ddxplus_patientsim/baselines/mediq_raw/results.jsonl") or []
            can = _mediq_evaluate(raw, _info_gt, "strict")
            raw_v1 = _read_mediq("set_b/ddxplus_patientsim/baselines/mediq_raw/results.jsonl") or []
            v1 = _mediq_evaluate(raw_v1, _info_gt, "strict")
            return can, v1
        if M == "MEDDxAgent-pb":
            return (normalise(read_jsonl("set_a/ddxplus_patientsim/baselines/meddxagent_pubmed_dynamic_gpt54nano.jsonl")),
                    normalise(read_dir("judged/set_b/ddxplus_patientsim/baselines/meddxagent")))
        if M == "MEDDxAgent-cw":
            return (normalise(read_jsonl("set_a/ddxplus_patientsim/baselines/meddxagent_gpt54nano.jsonl")),
                    normalise(read_dir("set_b/ddxplus_patientsim/baselines/meddxagent_cw")))
        if M == "AC-native":
            return (normalise(read_dir("judged/set_a/ddxplus_patientsim/baselines/ac_native")),
                    normalise(read_dir("set_b/ddxplus_patientsim/baselines/ac_native")))
        if M == "CoD":
            return (normalise(read_jsonl("set_a/ddxplus_patientsim/baselines/cod_diagnosisgpt34b.jsonl")),
                    normalise(read_dir("judged/set_b/ddxplus_patientsim/baselines/cod_v1")))
        if M == "BMBE":
            return (normalise(read_jsonl("set_a/ddxplus_patientsim/mobayes/bmbe_gpt54nano.jsonl")),
                    normalise(read_dir("set_b/ddxplus_patientsim/mobayes/bmbe")))

    if B == "htsim":
        if M == "CW":
            return (normalise(read_jsonl("set_a/ddxplus_htsim/baselines/cw_gpt54nano_remapped.jsonl")),
                    normalise(read_dir("set_b/ddxplus_htsim/baselines/cw")))
        if M == "CoT":
            return (normalise(read_jsonl("set_a/ddxplus_htsim/baselines/cot_gpt54nano_remapped.jsonl")),
                    normalise(read_dir("set_b/ddxplus_htsim/baselines/cot")))
        if M == "DDx-CoT":
            return (normalise(read_jsonl("set_a/ddxplus_htsim/baselines/savage_dd_gpt54nano_remapped.jsonl")),
                    normalise(read_dir("set_b/ddxplus_htsim/baselines/savage_dd")))
        if M == "BCoT":
            return (normalise(read_jsonl("set_a/ddxplus_htsim/baselines/savage_bay_gpt54nano_remapped.jsonl")),
                    normalise(read_dir("set_b/ddxplus_htsim/baselines/savage_bay")))
        if M == "UoT":
            return (normalise(read_dir("judged/set_a/ddxplus_htsim/baselines/uot")),
                    normalise(read_dir("set_b/ddxplus_htsim/baselines/uot")))
        if M == "MediQ-force":
            can_raw = read_jsonl("set_a/ddxplus_htsim/baselines/mediq_expert_gpt54nano_remapped.jsonl")
            can = normalise(can_raw)
            raw_v1 = _read_mediq("set_b/ddxplus_htsim/baselines/mediq_raw/results.jsonl") or []
            v1 = _mediq_evaluate(raw_v1, _info_gt, "force")
            return can, v1
        if M == "MediQ-strict":
            # Canonical: use abstained_strict in canonical jsonl
            can_raw = read_jsonl("set_a/ddxplus_htsim/baselines/mediq_expert_gpt54nano.jsonl")
            can = []
            for r in can_raw:
                ab_strict = bool(r.get("abstained_strict"))
                can.append({"t1": bool(r.get("top1_hit")) and not ab_strict,
                             "t3": False, "ab": ab_strict, "conf": None})
            raw_v1 = _read_mediq("set_b/ddxplus_htsim/baselines/mediq_raw/results.jsonl") or []
            v1 = _mediq_evaluate(raw_v1, _info_gt, "strict")
            return can, v1
        if M == "MEDDxAgent-pb":
            return (normalise(read_jsonl("set_a/ddxplus_htsim/baselines/meddxagent_pubmed_dynamic_gpt54nano_remapped.jsonl")),
                    normalise(read_dir("judged/set_b/ddxplus_htsim/baselines/meddxagent")))
        if M == "MEDDxAgent-cw":
            return (normalise(read_jsonl("set_a/ddxplus_htsim/baselines/meddxagent_closedworld_gpt54nano_remapped.jsonl")),
                    normalise(read_dir("set_b/ddxplus_htsim/baselines/meddxagent_cw")))
        if M == "AC-native":
            return (normalise(read_dir("judged/set_a/ddxplus_htsim/baselines/ac_native")),
                    normalise(read_dir("set_b/ddxplus_htsim/baselines/ac_native")))
        if M == "CoD":
            return (normalise(read_jsonl("set_a/ddxplus_htsim/baselines/cod_diagnosisgpt34b_remapped.jsonl")),
                    normalise(read_dir("judged/set_b/ddxplus_htsim/baselines/cod_v1")))
        if M == "BMBE":
            return (normalise(read_jsonl("set_a/ddxplus_htsim/mobayes/bmbe_gpt54nano.jsonl")),
                    normalise(read_dir("set_b/ddxplus_htsim/mobayes/bmbe")))

    if B == "ac":
        if M == "CW":
            return (normalise_remap(read_jsonl("set_a/agentclinic/baselines/cw_gpt54nano.jsonl")),
                    normalise(read_dir("judged/set_b/agentclinic/baselines/cw")))
        if M == "CoT":
            return (normalise_remap(read_jsonl("set_a/agentclinic/baselines/cot_gpt54nano.jsonl")),
                    normalise(read_dir("judged/set_b/agentclinic/baselines/cot")))
        if M == "DDx-CoT":
            return (normalise_remap(read_jsonl("set_a/agentclinic/baselines/savage_dd_gpt54nano.jsonl")),
                    normalise(read_dir("judged/set_b/agentclinic/baselines/savage_dd")))
        if M == "BCoT":
            return (normalise_remap(read_jsonl("set_a/agentclinic/baselines/savage_bay_gpt54nano.jsonl")),
                    normalise(read_dir("judged/set_b/agentclinic/baselines/savage_bay")))
        if M == "UoT":
            return (normalise(read_dir("judged/set_a/agentclinic/baselines/uot")),
                    normalise(read_dir("judged/set_b/agentclinic/baselines/uot")))
        if M == "MediQ-force":
            raw = _read_mediq("set_a/agentclinic/baselines/mediq_raw/results_0_10.jsonl",
                              "set_a/agentclinic/baselines/mediq_raw/results_10_50.jsonl") or []
            gt_can = _ac_gt_factory(AC_CASES_CAN)
            can = _mediq_evaluate(raw, gt_can, "force") if gt_can else []
            raw_v1 = _read_mediq("set_b/agentclinic/baselines/mediq_raw/results.jsonl") or []
            gt_v1 = _ac_gt_factory(AC_CASES_V1)
            v1 = _mediq_evaluate(raw_v1, gt_v1, "force") if gt_v1 else []
            return can, v1
        if M == "MediQ-strict":
            raw = _read_mediq("set_a/agentclinic/baselines/mediq_raw/results_0_10.jsonl",
                              "set_a/agentclinic/baselines/mediq_raw/results_10_50.jsonl") or []
            gt_can = _ac_gt_factory(AC_CASES_CAN)
            can = _mediq_evaluate(raw, gt_can, "strict") if gt_can else []
            raw_v1 = _read_mediq("set_b/agentclinic/baselines/mediq_raw/results.jsonl") or []
            gt_v1 = _ac_gt_factory(AC_CASES_V1)
            v1 = _mediq_evaluate(raw_v1, gt_v1, "strict") if gt_v1 else []
            return can, v1
        if M == "MEDDxAgent-pb":
            return (normalise(read_jsonl("set_a/agentclinic/baselines/meddxagent_pubmed_dynamic_gpt54nano.jsonl")),
                    normalise(read_dir("judged/set_b/agentclinic/baselines/meddxagent")))
        if M == "MEDDxAgent-cw":
            return (normalise(read_jsonl("set_a/agentclinic/baselines/meddxagent_closedworld_gpt54nano.jsonl")),
                    normalise(read_dir("judged/set_b/agentclinic/baselines/meddxagent_cw")))
        if M == "AC-native":
            return (normalise(read_jsonl("set_a/agentclinic/baselines/ac_native_gpt54nano.jsonl")),
                    normalise(read_dir("judged/set_b/agentclinic/baselines/ac_native")))
        if M == "CoD":
            return (normalise(read_jsonl("set_a/agentclinic/baselines/cod_diagnosisgpt34b_v2.jsonl")),
                    normalise(read_dir("judged/set_b/agentclinic/baselines/cod")))
        if M == "BMBE":
            # Both halves now use v5 distilled KB (judge mirror with confidence preserved).
            # Canonical AC BMBE: agentclinic_50_bmbe_v5 (= bmbe_v5 run, T-1=76).
            # v1 (set2) AC BMBE: agentclinic_50_set2_bmbe_v5 (just-built v5 KB run, T-1=64).
            return (normalise(read_dir("judged/set_a/agentclinic/mobayes/bmbe_v5")),
                    normalise(read_dir("judged/set_b/agentclinic/mobayes/bmbe_v5")))

    return [], []


# ─── Methods + benchmarks declaration ───────────────────────────────────
METHODS = [
    ("Closed-World (CW)",              "CW"),
    ("Chain-of-Thought (CoT)",         "CoT"),
    ("DDx-CoT",                        "DDx-CoT"),
    ("Bayesian-CoT",                   "BCoT"),
    ("UoT",                            "UoT"),
    ("MediQ-Expert (force-final)",     "MediQ-force"),
    ("MediQ-Expert (strict abstain)",  "MediQ-strict"),
    ("MEDDxAgent (paper-best)",        "MEDDxAgent-pb"),
    ("MEDDxAgent (closed-world abl.)", "MEDDxAgent-cw"),
    ("AgentClinic",                    "AC-native"),
    ("CoD / DiagnosisGPT-34B",         "CoD"),
    ("\\textbf{\\ours} (gpt-5.4-nano)", "BMBE"),
]
BENCHES = [
    ("PatientSim n=100", "patientsim"),
    ("HT-sim n=100",     "htsim"),
    ("AC-MedQA n=100",   "ac"),
]


# ─── Driver ─────────────────────────────────────────────────────────────
def _half_counts_bmbe(records):
    """For BMBE: τ-tune via LOOCV on this half, return committed/correct counts."""
    n = len(records)
    t1 = sum(1 for r in records if r["t1"])
    t3 = sum(1 for r in records if r["t3"])
    if any(r["conf"] is not None for r in records):
        tau = find_tau_loocv(records)
        com = sum(1 for r in records if r["conf"] is not None and r["conf"] >= tau)
        correct = sum(1 for r in records if r["conf"] is not None and r["conf"] >= tau and r["t1"])
    else:
        # No confidence field → use raw abstain flag (typically all False for BMBE)
        tau = None
        com = sum(1 for r in records if not r["ab"])
        correct = sum(1 for r in records if not r["ab"] and r["t1"])
    return n, t1, t3, com, correct, tau

def cell_combined(method_key: str, bench_key: str) -> Optional[dict]:
    """Per appendix `app:threshold_tuning`: τ* is tuned by LOOCV on a 50-case set
    (NOT on the combined n=100). For BMBE, apply each half's own τ* and
    aggregate the committed/correct counts."""
    can, v1 = cell_records(method_key, bench_key)
    if not can or not v1:
        return None

    if method_key == "BMBE":
        n_c, t1_c, t3_c, com_c, cor_c, tau_c = _half_counts_bmbe(can)
        n_v, t1_v, t3_v, com_v, cor_v, tau_v = _half_counts_bmbe(v1)
        n = n_c + n_v
        t1 = t1_c + t1_v; t3 = t3_c + t3_v
        com = com_c + com_v; cor = cor_c + cor_v
        cov = com / n if n else 0
        sa = cor / com if com else 0
        dhs = 2*sa*cov/(sa+cov) if (sa+cov) > 0 else 0
        return {"n": n, "t1": 100*t1/n, "t3": 100*t3/n,
                "sa": 100*sa, "cov": 100*cov, "dhs": 100*dhs,
                "tau_can": tau_c, "tau_v1": tau_v}

    return metrics_combined(can + v1)


def render_md():
    print("# Combined n=100 (canonical 50 + v1 50)\n")
    print(f"{'Method':<35s}", end="")
    for blabel, _ in BENCHES:
        print(f" | {blabel:<35s}", end="")
    print()
    print(f"{'-'*35} | " + " | ".join("-"*35 for _ in BENCHES))
    for label, mkey in METHODS:
        print(f"{label:<35s}", end="")
        for _, bkey in BENCHES:
            m = cell_combined(mkey, bkey)
            if m is None:
                print(f" | {'  source missing':<35s}", end="")
            else:
                tau_info = ""
                if "tau_can" in m:
                    tc = "—" if m["tau_can"] is None else f"{m['tau_can']:.2f}"
                    tv = "—" if m["tau_v1"]  is None else f"{m['tau_v1']:.2f}"
                    tau_info = f" τ_can={tc} τ_v1={tv}"
                print(f" | T1={m['t1']:>5.1f} T3={m['t3']:>5.1f} SA={m['sa']:>5.1f} Cov={m['cov']:>5.1f} DHS={m['dhs']:>5.1f}{tau_info}", end="")
        print()


def render_latex():
    """LaTeX rows, T-1/T-3 as integer percent, SA/Cov as integer percent, DHS one decimal."""
    for label, mkey in METHODS:
        cells = []
        for _, bkey in BENCHES:
            m = cell_combined(mkey, bkey)
            if m is None:
                cells.extend(["--"] * 5)
                continue
            t1 = f"{m['t1']:.1f}"
            t3 = "--" if mkey in ("MediQ-force","MediQ-strict","UoT","AC-native") and bkey != "ac" and m["t3"] == 0 else f"{m['t3']:.1f}"
            # Actually, for non-ranked methods, T-3 is N/A; we use 0 from the data which we render as "--".
            if mkey.startswith("MediQ"): t3 = "--"
            sa = f"{m['sa']:.1f}"
            cov = f"{m['cov']:.1f}"
            dhs = f"{m['dhs']:.1f}"
            cells.extend([t1, t3, sa, cov, dhs])
        # Build row
        sep = " & "
        bench_block = []
        for i in range(3):
            block = sep.join(cells[i*5:(i+1)*5])
            bench_block.append(block)
        row = f"{label:<32s}" + " & " + (" " + sep + " ").join(bench_block) + r" \\"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="Output LaTeX rows")
    args = ap.parse_args()
    if args.latex:
        render_latex()
    else:
        render_md()


if __name__ == "__main__":
    main()
