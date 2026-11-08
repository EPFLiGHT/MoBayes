#!/usr/bin/env python3
"""End-to-end reproducer for Table 21's Tok/case column (combined n=100).

For every (method, benchmark, half) cell, reads the raw JSON files and
computes Tok/case from the in-record `token_usage` field. Special-cases:

  - UoT  : reads costs/uot_<bench>_<half>/ (5-case real-token samples
           collected after the token-tracking patch). Falls back to the
           PS canonical sample for any cell whose dir is empty.
  - CoD  : reads costs/cod_yi34b_token_pass.json
           (offline Yi-34B tokenizer pass that respects CoD's 1-pass /
           3-pass branch).
  - MEDDxAgent: pulls totals from auxiliary token_usage.json files
           because the per-case JSONs don't store them.

Usage:
  python3 scripts/paper_tables/tokentable.py                # validation report → stdout
  python3 scripts/paper_tables/tokentable.py --show-paths   # list every file path
  python3 scripts/paper_tables/tokentable.py --diff         # diff vs current Tab 21
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

import build_combined_n100 as bcn   # noqa: E402

RESULTS = bcn.RESULTS
COSTS   = RESULTS / "costs"

METHODS = bcn.METHODS         # [(label, mkey)]
BENCHES = bcn.BENCHES         # [(label, bkey)]


# ─── Per-cell raw record paths (mirrors ci_tables._raw_records) ────────
def raw_paths(mkey: str, bkey: str) -> tuple[tuple[str, str] | None, tuple[str, str] | None]:
    """Return (canonical_path_kind, v1_path_kind) where kind ∈ {jsonl, dir}."""
    PC = "set_a"; LR = "judged"; AV = "set_b"
    M, B = mkey, bkey

    if B == "patientsim":
        if M == "CW":              return (f"{PC}/ddxplus_patientsim/baselines/cw_gpt54nano.jsonl", "jsonl"), (f"{AV}/ddxplus_patientsim/baselines/cw", "dir")
        if M == "CoT":             return (f"{PC}/ddxplus_patientsim/baselines/cot_gpt54nano.jsonl", "jsonl"), (f"{AV}/ddxplus_patientsim/baselines/cot", "dir")
        if M == "DDx-CoT":         return (f"{PC}/ddxplus_patientsim/baselines/savage_dd_gpt54nano.jsonl", "jsonl"), (f"{AV}/ddxplus_patientsim/baselines/savage_dd", "dir")
        if M == "BCoT":            return (f"{PC}/ddxplus_patientsim/baselines/savage_bay_gpt54nano.jsonl", "jsonl"), (f"{AV}/ddxplus_patientsim/baselines/savage_bay", "dir")
        if M == "UoT":             return (f"{PC}/ddxplus_patientsim/baselines/uot_gpt54nano.jsonl", "jsonl"), (f"{AV}/ddxplus_patientsim/baselines/uot", "dir")
        if M in ("MediQ-force", "MediQ-strict"): return ("(MediQ custom format)", "mediq"), ("(MediQ custom format)", "mediq")
        if M == "MEDDxAgent-pb":   return (f"{PC}/ddxplus_patientsim/baselines/meddxagent_pubmed_dynamic_gpt54nano.jsonl", "jsonl"), (f"{LR}/set_b/ddxplus_patientsim/baselines/meddxagent", "dir")
        if M == "MEDDxAgent-cw":   return (f"{PC}/ddxplus_patientsim/baselines/meddxagent_gpt54nano.jsonl", "jsonl"), (f"{AV}/ddxplus_patientsim/baselines/meddxagent_cw", "dir")
        if M == "AC-native":       return (f"{LR}/set_a/ddxplus_patientsim/baselines/ac_native", "dir"), (f"{AV}/ddxplus_patientsim/baselines/ac_native", "dir")
        if M == "CoD":             return (f"{PC}/ddxplus_patientsim/baselines/cod_diagnosisgpt34b.jsonl", "jsonl"), (f"{LR}/set_b/ddxplus_patientsim/baselines/cod_v1", "dir")
        if M == "BMBE":            return (f"{PC}/ddxplus_patientsim/mobayes/bmbe_gpt54nano.jsonl", "jsonl"), (f"{AV}/ddxplus_patientsim/mobayes/bmbe", "dir")
    if B == "htsim":
        if M == "CW":              return (f"{PC}/ddxplus_htsim/baselines/cw_gpt54nano_remapped.jsonl", "jsonl"), (f"{AV}/ddxplus_htsim/baselines/cw", "dir")
        if M == "CoT":             return (f"{PC}/ddxplus_htsim/baselines/cot_gpt54nano_remapped.jsonl", "jsonl"), (f"{AV}/ddxplus_htsim/baselines/cot", "dir")
        if M == "DDx-CoT":         return (f"{PC}/ddxplus_htsim/baselines/savage_dd_gpt54nano_remapped.jsonl", "jsonl"), (f"{AV}/ddxplus_htsim/baselines/savage_dd", "dir")
        if M == "BCoT":            return (f"{PC}/ddxplus_htsim/baselines/savage_bay_gpt54nano_remapped.jsonl", "jsonl"), (f"{AV}/ddxplus_htsim/baselines/savage_bay", "dir")
        if M == "UoT":             return (f"{LR}/set_a/ddxplus_htsim/baselines/uot", "dir"), (f"{AV}/ddxplus_htsim/baselines/uot", "dir")
        if M in ("MediQ-force", "MediQ-strict"): return ("(MediQ custom format)", "mediq"), ("(MediQ custom format)", "mediq")
        if M == "MEDDxAgent-pb":   return (f"{PC}/ddxplus_htsim/baselines/meddxagent_pubmed_dynamic_gpt54nano_remapped.jsonl", "jsonl"), (f"{LR}/set_b/ddxplus_htsim/baselines/meddxagent", "dir")
        if M == "MEDDxAgent-cw":   return (f"{PC}/ddxplus_htsim/baselines/meddxagent_closedworld_gpt54nano_remapped.jsonl", "jsonl"), (f"{AV}/ddxplus_htsim/baselines/meddxagent_cw", "dir")
        if M == "AC-native":       return (f"{LR}/set_a/ddxplus_htsim/baselines/ac_native", "dir"), (f"{AV}/ddxplus_htsim/baselines/ac_native", "dir")
        if M == "CoD":             return (f"{PC}/ddxplus_htsim/baselines/cod_diagnosisgpt34b_remapped.jsonl", "jsonl"), (f"{LR}/set_b/ddxplus_htsim/baselines/cod_v1", "dir")
        if M == "BMBE":            return (f"{PC}/ddxplus_htsim/mobayes/bmbe_gpt54nano.jsonl", "jsonl"), (f"{AV}/ddxplus_htsim/mobayes/bmbe", "dir")
    if B == "ac":
        if M == "CW":              return (f"{PC}/agentclinic/baselines/cw_gpt54nano.jsonl", "jsonl"), (f"{LR}/set_b/agentclinic/baselines/cw", "dir")
        if M == "CoT":             return (f"{PC}/agentclinic/baselines/cot_gpt54nano.jsonl", "jsonl"), (f"{LR}/set_b/agentclinic/baselines/cot", "dir")
        if M == "DDx-CoT":         return (f"{PC}/agentclinic/baselines/savage_dd_gpt54nano.jsonl", "jsonl"), (f"{LR}/set_b/agentclinic/baselines/savage_dd", "dir")
        if M == "BCoT":            return (f"{PC}/agentclinic/baselines/savage_bay_gpt54nano.jsonl", "jsonl"), (f"{LR}/set_b/agentclinic/baselines/savage_bay", "dir")
        if M == "UoT":             return (f"{LR}/set_a/agentclinic/baselines/uot", "dir"), (f"{LR}/set_b/agentclinic/baselines/uot", "dir")
        if M in ("MediQ-force", "MediQ-strict"): return ("(MediQ custom format)", "mediq"), ("(MediQ custom format)", "mediq")
        if M == "MEDDxAgent-pb":   return (f"{PC}/agentclinic/baselines/meddxagent_pubmed_dynamic_gpt54nano.jsonl", "jsonl"), (f"{LR}/set_b/agentclinic/baselines/meddxagent", "dir")
        if M == "MEDDxAgent-cw":   return (f"{PC}/agentclinic/baselines/meddxagent_closedworld_gpt54nano.jsonl", "jsonl"), (f"{LR}/set_b/agentclinic/baselines/meddxagent_cw", "dir")
        if M == "AC-native":       return (f"{PC}/agentclinic/baselines/ac_native_gpt54nano.jsonl", "jsonl"), (f"{LR}/set_b/agentclinic/baselines/ac_native", "dir")
        if M == "CoD":             return (f"{PC}/agentclinic/baselines/cod_diagnosisgpt34b_v2.jsonl", "jsonl"), (f"{LR}/set_b/agentclinic/baselines/cod", "dir")
        if M == "BMBE":            return (f"{PC}/agentclinic/mobayes/bmbe_gpt54nano.jsonl", "jsonl"), (f"{LR}/set_b/agentclinic/mobayes/bmbe_v5", "dir")
    return None, None


# ─── Per-record token sum ──────────────────────────────────────────────
def _tokens_from_record(r: dict) -> int:
    total = 0
    # (a) Standard token_usage dict (CW, CoT, BMBE, ...)
    tu = r.get("token_usage", {})
    if isinstance(tu, dict):
        for sub in tu.values():
            if not isinstance(sub, dict): continue
            total += int(sub.get("prompt_tokens", 0) or sub.get("input_tokens", 0) or 0)
            total += int(sub.get("completion_tokens", 0) or sub.get("output_tokens", 0) or 0)
    # (b) MediQ raw log: per-call usage under interactive_system
    isys = r.get("interactive_system") or {}
    for entry in isys.get("temp_additional_info") or []:
        if not entry: continue
        u = entry.get("usage") or {}
        total += int(u.get("prompt_tokens", 0) or u.get("input_tokens", 0) or 0)
        total += int(u.get("completion_tokens", 0) or u.get("output_tokens", 0) or 0)
    return total


# ─── MEDDxAgent aux files ──────────────────────────────────────────────
MEDDX_AUX = {
    ("MEDDxAgent-pb", "patientsim", "can"): None,
    ("MEDDxAgent-pb", "htsim",      "can"): "set_a/ddxplus_htsim/_raw/meddxagent_paperbest/20260502_201308/token_usage.json",
    ("MEDDxAgent-pb", "ac",         "can"): "set_a/agentclinic/_raw/meddxagent_paperbest/token_usage.json",
    ("MEDDxAgent-cw", "patientsim", "can"): None,
    ("MEDDxAgent-cw", "htsim",      "can"): "set_a/ddxplus_htsim/_raw/meddxagent_cwabl/20260502_233138/token_usage.json",
    ("MEDDxAgent-cw", "ac",         "can"): "set_a/agentclinic/_raw/meddxagent_cwabl/20260503_001840/token_usage.json",
    ("MEDDxAgent-pb", "patientsim", "v1"):  "set_b/ddxplus_patientsim/baselines/meddxagent_hpc_raw/20260505_123929/token_usage.json",
    ("MEDDxAgent-pb", "htsim",      "v1"):  "set_b/ddxplus_htsim/baselines/meddxagent_hpc_raw/20260505_133301/token_usage.json",
    ("MEDDxAgent-pb", "ac",         "v1"):  "set_b/agentclinic/baselines/meddxagent_hpc_raw/20260505_171151/token_usage.json",
    ("MEDDxAgent-cw", "patientsim", "v1"):  "set_b/ddxplus_patientsim/baselines/meddxagent_cw_raw/20260504_214724/token_usage.json",
    ("MEDDxAgent-cw", "htsim",      "v1"):  "set_b/ddxplus_htsim/baselines/meddxagent_cw_raw/20260504_214724/token_usage.json",
    ("MEDDxAgent-cw", "ac",         "v1"):  "set_b/agentclinic/baselines/meddxagent_cw_raw/20260505_204941/token_usage.json",
}

def _aux_tokens(rel_path: str | None) -> int:
    if not rel_path: return 0
    p = RESULTS / rel_path
    if not p.exists(): return 0
    d = json.loads(p.read_text())
    u = d.get("usage") or {}
    total = 0
    for role in ("doctor", "patient"):
        s = u.get(role) or {}
        total += int(s.get("prompt_tokens", 0) or s.get("input_tokens", 0) or 0)
        total += int(s.get("completion_tokens", 0) or s.get("output_tokens", 0) or 0)
    return total


# ─── UoT cost samples ──────────────────────────────────────────────────
def uot_avg(bkey: str) -> tuple[float | None, str]:
    """Combined UoT tok/case = avg(uot_<bkey>_can, uot_<bkey>_v1).
    Returns (avg, source_description). Falls back to PS canonical sample
    if a cell-specific sample is missing.
    """
    halves = []
    sources = []
    bench_dir = {"patientsim": "ddxplus_patientsim", "htsim": "ddxplus_htsim", "ac": "agentclinic"}
    set_letter = {"can": "a", "v1": "b"}
    for h in ("can", "v1"):
        d = COSTS / "uot_samples" / f"{bench_dir[bkey]}_{set_letter[h]}"
        if not d.exists(): continue
        files = sorted(d.glob("case_*.json"))
        totals = []
        for f in files:
            r = json.loads(f.read_text())
            dt = (r.get("token_usage") or {}).get("doctor", {})
            t = dt.get("total_tokens") or (dt.get("prompt_tokens", 0) + dt.get("completion_tokens", 0))
            if t > 0: totals.append(t)
        if totals:
            halves.append(sum(totals) / len(totals))
            sources.append(f"costs/uot_samples/{bench_dir[bkey]}_{set_letter[h]} (n={len(totals)})")
    if halves:
        return (sum(halves) / len(halves), " + ".join(sources))
    # fallback
    d = COSTS / "uot_samples" / "ddxplus_patientsim_a"
    files = sorted(d.glob("case_*.json"))
    totals = []
    for f in files:
        r = json.loads(f.read_text())
        dt = (r.get("token_usage") or {}).get("doctor", {})
        t = dt.get("total_tokens") or (dt.get("prompt_tokens", 0) + dt.get("completion_tokens", 0))
        if t > 0: totals.append(t)
    if totals:
        return (sum(totals) / len(totals), f"FALLBACK costs/uot_samples/ddxplus_patientsim_a (n={len(totals)})")
    return (None, "no UoT sample available")


# ─── CoD Yi-34B offline pass ───────────────────────────────────────────
def cod_lookup(bkey: str) -> tuple[float | None, str]:
    """Combined CoD tok/case from the v3 offline Yi-34B pass."""
    p = COSTS / "cod_yi34b_token_pass.json"
    if not p.exists():
        return None, "v3 file missing"
    j = json.loads(p.read_text())
    vals = []
    sources = []
    for h in ("can", "v1"):
        e = j.get(f"CoD_{bkey}_{h}")
        if isinstance(e, dict) and e.get("status") == "ok":
            vals.append(e["tok_per_case"])
            sources.append(f"cod_yi34b_token_pass.json:CoD_{bkey}_{h}")
    if vals:
        return (sum(vals) / len(vals), " + ".join(sources))
    return None, "no CoD entry"


# ─── Main per-cell computation ─────────────────────────────────────────
def compute_cell(mkey: str, bkey: str) -> dict:
    """Return {tok_per_case, n_total, sources}."""
    # Special cases: UoT and CoD use offline samples / passes.
    if mkey == "UoT":
        avg, src = uot_avg(bkey)
        return {"tok_per_case": avg, "n_total": "5+5 sample" if avg else "—",
                "sources": src, "method": "UoT cost-sample"}
    if mkey == "CoD":
        avg, src = cod_lookup(bkey)
        return {"tok_per_case": avg, "n_total": "100 (offline)",
                "sources": src, "method": "CoD Yi-34B offline pass"}

    # MediQ uses a custom raw log format (interactive_system.temp_additional_info[].usage)
    # under set_{a,b}/<bench>/baselines/mediq_raw/; build_combined exposes _read_mediq.
    if mkey in ("MediQ-force", "MediQ-strict"):
        if bkey == "patientsim":
            raw_can = bcn._read_mediq("set_a/ddxplus_patientsim/baselines/mediq_raw/results.jsonl") or []
            raw_v1  = bcn._read_mediq("set_b/ddxplus_patientsim/baselines/mediq_raw/results.jsonl") or []
        elif bkey == "htsim":
            # HT-sim: canonical jsonl + v1 mediq raw
            raw_can = bcn.read_jsonl(f"set_a/ddxplus_htsim/baselines/mediq_expert_gpt54nano{'_remapped' if mkey == 'MediQ-force' else ''}.jsonl")
            raw_v1  = bcn._read_mediq("set_b/ddxplus_htsim/baselines/mediq_raw/results.jsonl") or []
        else:  # ac
            raw_can = bcn._read_mediq("set_a/agentclinic/baselines/mediq_raw/results_0_10.jsonl",
                                        "set_a/agentclinic/baselines/mediq_raw/results_10_50.jsonl") or []
            raw_v1  = bcn._read_mediq("set_b/agentclinic/baselines/mediq_raw/results.jsonl") or []
        can_path = ("(MediQ raw)", "mediq")
        v1_path = ("(MediQ raw)", "mediq")
    else:
        # Generic path: load both halves via raw_paths and bcn helpers.
        can_path, v1_path = raw_paths(mkey, bkey)
        def _raw(p):
            if p is None: return []
            rel, kind = p
            if rel.startswith("("): return []
            if kind == "jsonl": return bcn.read_jsonl(rel)
            if kind == "dir":   return bcn.read_dir(rel)
            return []
        raw_can = _raw(can_path)
        raw_v1 = _raw(v1_path)
    n = len(raw_can) + len(raw_v1)
    if n == 0:
        return {"tok_per_case": None, "n_total": 0, "sources": "no records",
                 "method": "in-record token_usage"}

    has_aux = mkey in ("MEDDxAgent-pb", "MEDDxAgent-cw")
    total = 0
    n_with_tokens = 0   # only halves whose tracking actually fired
    src_parts = []

    def _has_tokens(records):
        if not records: return False
        return sum(_tokens_from_record(r) for r in records) > 0

    if _has_tokens(raw_can) or has_aux:
        cs = sum(_tokens_from_record(r) for r in raw_can)
        total += cs
        n_with_tokens += len(raw_can)
        if cs > 0: src_parts.append(f"{can_path[0]} ({len(raw_can)} cases, {cs} tok)")
    if has_aux:
        a = _aux_tokens(MEDDX_AUX.get((mkey, bkey, "can")))
        if a > 0:
            total += a
            src_parts.append(f"aux: {MEDDX_AUX[(mkey, bkey, 'can')]} ({a} tok)")
    if _has_tokens(raw_v1) or has_aux:
        vs = sum(_tokens_from_record(r) for r in raw_v1)
        total += vs
        n_with_tokens += len(raw_v1)
        if vs > 0: src_parts.append(f"{v1_path[0]} ({len(raw_v1)} cases, {vs} tok)")
    if has_aux:
        a = _aux_tokens(MEDDX_AUX.get((mkey, bkey, "v1")))
        if a > 0:
            total += a
            src_parts.append(f"aux: {MEDDX_AUX[(mkey, bkey, 'v1')]} ({a} tok)")

    denom = n_with_tokens if n_with_tokens > 0 else n
    return {"tok_per_case": (total / denom) if denom else None,
            "n_total": f"{n_with_tokens}/{n}" if n_with_tokens != n else n,
            "sources": "; ".join(src_parts) or "tokens absent",
            "method": "in-record token_usage" + (" + MEDDxAgent aux" if has_aux else "")}


# ─── Tab 21 cell readback (for diff mode) ──────────────────────────────
def read_tab21_tokens() -> dict:
    """Parse current tables/tab_ci_all.tex Tab 21 (combined) Tok/case cells.
    Returns {(label, bkey): tok_string}."""
    tex = (REPO_ROOT / "tables" / "tab_ci_all.tex").read_text()
    # Find tab:ci_combined block
    m = re.search(r"\\label\{tab:ci_combined\}.*?\\end\{tabular\}", tex, re.S)
    if not m: return {}
    block = m.group(0)
    out = {}
    for line in block.split("\n"):
        line = line.strip()
        if "\\\\" not in line or "textsubscript" not in line: continue
        mm = re.match(r"(.+?)\s*&\s*(.+?)\s*\\\\$", line)
        if not mm: continue
        label = mm.group(1).strip()
        cells = re.sub(r"\\cellcolor\{[^}]+\}", "", mm.group(2))
        # Tok/case is every 3rd cell (3 metrics × 3 benchmarks)
        cell_list = [c.strip() for c in cells.split("&")]
        for i, (_, bkey) in enumerate(BENCHES):
            tok = cell_list[i * 3 + 2] if len(cell_list) > i * 3 + 2 else "—"
            out[(label, bkey)] = tok
    return out


# ─── CLI ───────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-paths", action="store_true",
                    help="List every cell's data-path mapping.")
    ap.add_argument("--diff", action="store_true",
                    help="Diff computed Tok/case vs current Tab 21.")
    args = ap.parse_args()

    if args.show_paths:
        print(f"RESULTS root: {RESULTS}\n")
        print(f"{'method':<32}{'bench':<11}{'half':<5}{'path':<60}")
        print("-" * 110)
        for label, mkey in METHODS:
            for blabel, bkey in BENCHES:
                can, v1 = raw_paths(mkey, bkey)
                bl = blabel.split()[0]
                if can:
                    print(f"  {label[:30]:<32}{bl:<11}{'can':<5}{can[1]:<5} {can[0]}")
                if v1:
                    print(f"  {label[:30]:<32}{bl:<11}{'v1':<5}{v1[1]:<5} {v1[0]}")
        return

    # Compute and print per-cell
    print(f"{'Method':<32}{'Bench':<11}{'n':<10}{'Tok/case':>10}{'Source / method':<10}")
    print("-" * 105)
    computed: dict = {}
    for label, mkey in METHODS:
        for blabel, bkey in BENCHES:
            r = compute_cell(mkey, bkey)
            tok = r["tok_per_case"]
            tok_s = (f"{tok/1000:.1f}K" if tok and tok >= 1000
                     else f"{int(round(tok))}" if tok else "—")
            print(f"  {label[:30]:<32}{blabel.split()[0]:<11}{str(r['n_total']):<10}{tok_s:>10}  {r['method']}")
            computed[(label, bkey)] = tok_s

    if args.diff:
        print("\n=== Diff vs current tables/tab_ci_all.tex (Table 21) ===")
        tab = read_tab21_tokens()
        diffs = 0
        for (label, mkey), bench_pairs in [((m[0], m[1]), BENCHES) for m in METHODS]:
            for blabel, bkey in bench_pairs:
                ours = computed.get((label, bkey), "—")
                theirs = tab.get((label, bkey), "??")
                if ours != theirs and theirs != "??":
                    print(f"  MISMATCH {label[:30]:<32}{blabel.split()[0]:<11}"
                          f"computed={ours}  tab21={theirs}")
                    diffs += 1
        if diffs == 0:
            print(f"  All {sum(1 for _ in METHODS) * len(BENCHES)} cells match.")
        else:
            print(f"\n  {diffs} mismatches.")


if __name__ == "__main__":
    main()
