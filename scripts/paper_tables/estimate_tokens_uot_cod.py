"""estimate_tokens_uot_cod.py — Estimate tokens for UoT and CoD by tokenizing
their stored dialogs.

Why: their per-case runners did not capture doctor-side tokens
  - UoT: only patient-simulator usage was logged, the tree-search LLM calls weren't.
  - CoD/DiagnosisGPT-34B: doctor is a self-hosted GPU model with explicit
    `note='per-call usage not tracked'`.
But both runs preserved dialog turns / dialog_history. We tokenize the
text those calls would have processed and produced (questions asked +
patient answers + final diagnosis text) using cl100k_base (close to
gpt-5.4-nano's tokenizer) to get a lower-bound estimate.

Caveat: this captures *visible content* only. UoT's tree-search expansion
prompts (which the LLM sees but aren't recorded in the user-facing
dialog) and CoD's RAG / system-prompt context are NOT included, so the
real token count is HIGHER than what we estimate. We mark these as
'≥ X tok/turn (visible content only)' in the table.

Run:
    python3 scripts/paper_tables/estimate_tokens_uot_cod.py
    # writes data/benchmark_results/estimated_tokens.json; the copy the
    # tables read is data/benchmark_results/costs/estimated_tokens.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import tiktoken

CODEBASE = Path(__file__).resolve().parents[2]
RESULTS = CODEBASE / "data/benchmark_results"
OUT = RESULTS / "estimated_tokens.json"

ENC = tiktoken.get_encoding("cl100k_base")


def tokens_of(text: str) -> int:
    return len(ENC.encode(text or "", disallowed_special=()))


def collect_dialog_text(rec: dict) -> tuple[int, int]:
    """Return (total_token_estimate, n_turns)."""
    # Try multiple shapes
    total = 0
    n_turns = 0

    # bench-style turns list
    turns = rec.get("turns")
    if isinstance(turns, list) and turns:
        for t in turns:
            if isinstance(t, dict):
                total += tokens_of(t.get("content", ""))
                if t.get("role") == "patient":
                    n_turns += 1

    # BMBE/UoT dialog_history list
    if not n_turns:
        dh = rec.get("dialog_history")
        if isinstance(dh, list):
            for t in dh:
                if isinstance(t, dict):
                    total += tokens_of(t.get("content", ""))
                    if t.get("role") == "patient":
                        n_turns += 1

    # CoD raw 'dialog' list (already lighter, doctor diagnoses)
    if not n_turns:
        dlg = rec.get("dialog")
        if isinstance(dlg, list):
            for t in dlg:
                if isinstance(t, dict):
                    total += tokens_of(t.get("content", ""))
                if isinstance(t, dict) and t.get("role") == "patient":
                    n_turns += 1
            # turns int field as fallback
            if not n_turns and isinstance(rec.get("turns"), int):
                n_turns = int(rec["turns"])

    # CoD diagnosis_text (free-text doctor reasoning) — count as completion tokens
    diag = rec.get("diagnosis_text") or rec.get("cod_diagnosis_text") or ""
    total += tokens_of(diag)

    # predicted_set tokens (final ranked diagnoses output)
    ps = rec.get("predicted_set") or []
    if isinstance(ps, list):
        for d in ps[:5]:
            if isinstance(d, str):
                total += tokens_of(d)

    return total, n_turns


def aggregate_dir(rel: str) -> tuple[int, int, int]:
    """Returns (total_tokens, total_turns, n_cases) for a per-case dir."""
    d = RESULTS / rel
    if not d.exists():
        return 0, 0, 0
    cases = sorted(d.glob("case_*.json"))
    tot_tok = tot_turn = 0
    for c in cases:
        rec = json.loads(c.read_text())
        # final-nested or top-level shape
        merged = {**rec}
        if isinstance(rec.get("final"), dict):
            for k, v in rec["final"].items():
                if k not in merged:
                    merged[k] = v
        t, n = collect_dialog_text(merged)
        tot_tok += t
        tot_turn += n
    return tot_tok, tot_turn, len(cases)


def aggregate_jsonl(rel: str) -> tuple[int, int, int]:
    p = RESULTS / rel
    if not p.exists():
        return 0, 0, 0
    tot_tok = tot_turn = 0
    n = 0
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        t, ntr = collect_dialog_text(rec)
        tot_tok += t
        tot_turn += ntr
        n += 1
    return tot_tok, tot_turn, n


# ─── Scan plan: UoT + CoD for canonical & v1 across PatientSim/HT-sim/AC ──
PLAN = [
    # method, benchmark, half, kind, path
    ("UoT", "patientsim", "can", "jsonl", "set_a/ddxplus_patientsim/baselines/uot_gpt54nano.jsonl"),
    ("UoT", "patientsim", "v1",  "dir",   "set_b/ddxplus_patientsim/baselines/uot"),
    ("UoT", "htsim",      "can", "dir",   "judged/set_a/ddxplus_htsim/baselines/uot"),
    ("UoT", "htsim",      "v1",  "dir",   "set_b/ddxplus_htsim/baselines/uot"),
    ("UoT", "ac",         "can", "dir",   "judged/set_a/agentclinic/baselines/uot"),
    ("UoT", "ac",         "v1",  "dir",   "judged/set_b/agentclinic/baselines/uot"),

    ("CoD", "patientsim", "can", "jsonl", "set_a/ddxplus_patientsim/baselines/cod_diagnosisgpt34b.jsonl"),
    ("CoD", "patientsim", "v1",  "jsonl", "set_b/ddxplus_patientsim/baselines/cod/results.jsonl"),
    ("CoD", "htsim",      "can", "jsonl", "set_a/ddxplus_htsim/baselines/cod_diagnosisgpt34b.jsonl"),
    ("CoD", "htsim",      "v1",  "jsonl", "set_b/ddxplus_htsim/baselines/cod/results.jsonl"),
    ("CoD", "ac",         "can", "jsonl", "set_a/agentclinic/baselines/cod_diagnosisgpt34b_v2.jsonl"),
    ("CoD", "ac",         "v1",  "jsonl", "set_b/agentclinic/baselines/cod/results.jsonl"),
]


def main() -> None:
    # Run in 4 batches of ~3 entries each (12 entries / 4 = 3)
    batches = [PLAN[i*3:(i+1)*3] for i in range(math.ceil(len(PLAN) / 3))]

    results = {}
    print(f"Running {len(PLAN)} cells across {len(batches)} batches:")
    for bi, batch in enumerate(batches, 1):
        print(f"\n=== batch {bi}/{len(batches)} ({len(batch)} cells) ===")
        for method, bench, half, kind, path in batch:
            if kind == "dir":
                tok, turns, n = aggregate_dir(path)
            else:
                tok, turns, n = aggregate_jsonl(path)
            key = f"{method}_{bench}_{half}"
            results[key] = {
                "method": method, "benchmark": bench, "half": half, "path": path,
                "n_cases": n, "total_tokens_estimated": tok, "total_turns": turns,
                "tok_per_turn": (tok / turns) if turns else 0,
                "tok_per_case":  (tok / n) if n else 0,
            }
            if n == 0:
                status = "MISSING"
            elif tok == 0:
                status = "NO DIALOG"
            else:
                tpt = tok / turns if turns else tok / n
                status = f"{tok:>9,} tok / {turns:>3} turns / {n} cases  → {tpt:.0f} tok/turn"
            print(f"  {method:<5s} {bench:<11s} {half:<3s}  {status}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
