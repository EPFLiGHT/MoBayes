#!/usr/bin/env python3
"""Accurate Yi-34B tokenizer pass for CoD/DiagnosisGPT-34B  (v3 = corrected).

Fixes a bug in v2: v2 assumed every doctor turn triggered the 3-pass
diagnostic branch in `cod_cli.DiagnosisChatbot.inference`. Auditing the
saved dialogs shows only ~2% of turns enter that branch (the rest are
single-pass chit-chat / follow-up questions). v3 inspects each doctor
output for the trigger phrase ('Enter the diagnostic process' /
'Analyzing patient symptoms' / Chinese variants) and applies:

  - 3-pass branch: prompt = 3*base + out_t + 200 (retrieved-disease overhead)
  - 1-pass branch: prompt = base                  (single model call)

per inference. base_t still grows with history-concat (CoD's
`<|user|>\\n{q}\\n<|assistant|>\\n{r}\\n` template).

This typically lowers the v2 estimate by 2-3x for AC (where dialogs
are long and the 3-pass branch is rare).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

ROOT = Path("data/benchmark_results")

CELLS = {
    "CoD_patientsim_can": ("set_a/ddxplus_patientsim/baselines/cod_diagnosisgpt34b.jsonl",                 "dialog_history"),
    "CoD_patientsim_v1":  ("set_b/ddxplus_patientsim/baselines/cod/results.jsonl",                     "dialog_history"),
    "CoD_htsim_can":      ("set_a/ddxplus_htsim/baselines/cod_diagnosisgpt34b.jsonl",             "dialog"),
    "CoD_htsim_v1":       ("set_b/ddxplus_htsim/baselines/cod/results.jsonl",                          "dialog"),
    "CoD_ac_can":         ("set_a/agentclinic/baselines/cod_diagnosisgpt34b_v2.jsonl",          "dialog"),
    "CoD_ac_v1":          ("set_b/agentclinic/baselines/cod/results.jsonl",                     "dialog"),
}

# Trigger phrases that put CoD into the 3-pass diagnostic branch
# (cod_cli.DiagnosisChatbot.inference, line 265).
TRIGGER_PHRASES = [
    "Enter the diagnostic process",
    "Analyzing patient symptoms",
    "进入诊断流程",
    "总结病人症状信息",
]
RETRIEVED_DISEASES_OVERHEAD = 200


def _load(path: Path) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _qa_pairs(turns: list) -> list:
    pairs = []
    pending = None
    for t in turns:
        role = (t.get("role") or "").lower()
        c = t.get("content") or ""
        if not c: continue
        if role == "patient":
            pending = c
        elif role == "doctor" and pending is not None:
            pairs.append((pending, c)); pending = None
    return pairs


def _is_3pass(doctor_output: str) -> bool:
    return any(p in doctor_output for p in TRIGGER_PHRASES)


def _per_case_cost(pairs: list, tokenizer) -> tuple[int, int, int]:
    """Returns (prompt_tokens, completion_tokens, n_3pass_turns)."""
    hist = []
    p_tot = c_tot = n3 = 0
    for q, resp in pairs:
        base = ""
        for q_old, r_old in hist:
            base += f"<|user|>\n{q_old}\n<|assistant|>\n{r_old}\n"
        base += f"<|user|>\n{q}\n<|assistant|>\n"
        base_t = len(tokenizer.encode(base, add_special_tokens=False))
        out_t  = len(tokenizer.encode(resp, add_special_tokens=False))
        if _is_3pass(resp):
            p_tot += 3 * base_t + out_t + RETRIEVED_DISEASES_OVERHEAD
            n3 += 1
        else:
            p_tot += base_t
        c_tot += out_t
        hist.append((q, resp))
    return p_tot, c_tot, n3


def main() -> None:
    from transformers import AutoTokenizer
    print("Loading Yi-34B tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B-Chat", use_fast=True)

    results: dict = {}
    for label, (rel, dialog_key) in CELLS.items():
        path = ROOT / rel
        if not path.exists():
            results[label] = {"status": "missing", "path": str(rel)}
            continue
        records = list(_load(path))
        n = len(records)
        ptot = ctot = n3 = total_inf = 0
        for r in records:
            pairs = _qa_pairs(r.get(dialog_key) or [])
            p, c, k = _per_case_cost(pairs, tokenizer)
            ptot += p; ctot += c; n3 += k
            total_inf += len(pairs)
        results[label] = {
            "status": "ok",
            "n_cases": n,
            "prompt_tokens_total": ptot,
            "completion_tokens_total": ctot,
            "tok_per_case": (ptot + ctot) / n if n else 0,
            "total_inferences": total_inf,
            "n_3pass_turns": n3,
            "pct_3pass": 100*n3/total_inf if total_inf else 0,
            "method": "yi34b_tokenizer_offline_v3_branch_aware",
            "source": rel,
            "dialog_key": dialog_key,
        }

    out = Path("data/benchmark_results/costs") / "cod_yi34b_token_pass.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote: {out}\n")
    print(f"{'cell':<22}{'n':>4}{'tok/case':>11}{'inf':>6}{'3pass':>7}{'%3p':>6}")
    print("-" * 60)
    for k, v in results.items():
        if v["status"] == "ok":
            print(f"{k:<22}{v['n_cases']:>4}{v['tok_per_case']:>11.0f}"
                  f"{v['total_inferences']:>6}{v['n_3pass_turns']:>7}{v['pct_3pass']:>5.0f}%")


if __name__ == "__main__":
    main()
