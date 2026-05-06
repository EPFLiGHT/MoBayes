"""Unified LLM-judge remapper: free-form doctor output -> closed disease list,
with Top-1, Top-3, and abstention judgement in a single concurrent pass.

Reads case_*.json files from --in-dir (raw model outputs) and writes mirrored
case_*.json files into --out-dir (default: data/benchmark_results/judged/<in_dir_name>),
each enriched with these new fields under "final":

  predicted_remapped       : str  ("Unknown" or one of the closed-list diseases)
  predicted_set_remapped   : list (up to 3, mapped from the doctor's differential)
  top1_hit_judged          : bool (predicted_remapped semantically matches gt)
  top3_hit_judged          : bool (any of predicted_set_remapped matches gt)
  abstained_judged         : bool (LLM judged from output text)

Original fields (predicted, top1_hit, abstained, predicted_set, ...) are copied
through unchanged.

Concurrent: ThreadPoolExecutor (default 20 workers) for ~10x speedup vs sequential.
Disease list auto-detected from the input directory name unless --diseases is given:
  ddxplus_*  -> data/kbs/ddxplus/diseases.csv (49 diseases)
  agentclinic_* -> data/kbs/agentclinic_a/diseases.csv (31 diseases)

Usage:
    OPENAI_API_KEY=... PYTHONPATH=. \\
        python3 scripts/analysis/llm_remap_unified.py \\
            --in-dir data/benchmark_results/agentclinic_50_sa_gpt54

    # explicit overrides
    python3 scripts/analysis/llm_remap_unified.py \\
        --in-dir   data/benchmark_results/ddxplus_htsim_50_bmbe_gemma4_31b \\
        --out-dir  data/benchmark_results/judged/set_a/ddxplus_htsim/mobayes/bmbe_gemma4_31b \\
        --diseases data/kbs/ddxplus/diseases.csv \\
        --model    gpt-5.4-nano \\
        --workers  20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from openai import OpenAI


REPO = Path(__file__).resolve().parents[2]


PROMPT = """You are a medical-coding assistant. Given the doctor's diagnostic output below, do three things:

1. TOP-1: pick the disease from the closed list of {n} that semantically matches the doctor's TOP candidate (the rank-1 commit). If none match, output "Unknown".

2. TOP-3: pick up to three diseases from the closed list that semantically match the doctor's RANKED differential (in order). If a candidate cannot be mapped, output "Unknown" for that slot. If the doctor only gave a single candidate, output a list of length 1.

3. ABSTAIN: decide whether the doctor explicitly REFUSED to diagnose. Set "abstained": true ONLY if the doctor's TOP candidate text is literally "ABSTAIN", "Unknown", "I don't know", empty/missing, or otherwise an explicit refusal-to-commit. Set "abstained": false whenever the doctor names ANY disease as their top candidate, EVEN IF that disease does NOT appear in the closed list (off-list = committed-but-wrong, not abstain). In other words, abstain reflects whether the doctor refused to diagnose at all, not whether the diagnosis they gave maps onto the closed list. The differential may still contain real entries even when the top is an explicit refusal; that is normal.

Closed disease list:
{disease_list}

Doctor's diagnostic output:
---
{diag_text}
---

Semantic-match guidance:
- "Chronic Obstructive Pulmonary Disease (COPD)" matches "Chronic obstructive pulmonary disease"
- "CKD" matches "Chronic kidney failure"
- "** Femoropopliteal artery stenosis" matches "Femoropopliteal artery stenosis"
- "Acute acalculous cholecystitis with sepsis" matches "Acalculous cholecystitis"
- Markdown asterisks, parenthetical abbreviations, and minor wording variants are all OK.
- Use "Unknown" only if the candidate clearly does not refer to any listed disease.

Output JSON exactly in this format and nothing else:
{{"top1": "<disease|Unknown>", "top3": ["<d1>", "<d2>", "<d3>"], "abstained": true|false}}
"""


_JSON_RE = re.compile(r"\{.*\}", re.S)


def _parse_json(raw: str) -> dict:
    if not raw:
        return {}
    m = _JSON_RE.search(raw)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _normalise_to_list(diseases: List[str], raw: str) -> str:
    if not raw:
        return "Unknown"
    raw = raw.strip().strip('"').strip("'").strip("*").strip()
    if not raw or raw.lower() in ("unknown", "none", "n/a", "abstain"):
        return "Unknown"
    for d in diseases:
        if d.lower() == raw.lower():
            return d
    # tolerant: substring match if exact missing
    for d in diseases:
        if d.lower() in raw.lower() or raw.lower() in d.lower():
            return d
    return "Unknown"


def extract_pred_text(case: dict) -> str:
    fin = case.get("final", {})
    pred = (fin.get("predicted") or "").strip()
    pred_set = fin.get("predicted_set") or fin.get("predicted_top3") or []
    pred_set = [p[0] if isinstance(p, (list, tuple)) else p for p in pred_set]

    is_abstain = (not pred) or pred.upper().startswith("ABSTAIN") or pred.lower() in ("unknown", "i don't know")

    parts: List[str] = []
    if not is_abstain:
        parts.append(f"Top candidate: {pred}")
    else:
        parts.append("Top candidate: ABSTAIN / no commit")
    if pred_set:
        ranked = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(pred_set[:5]) if d)
        if ranked:
            parts.append(f"Ranked differential:\n{ranked}")
    return "\n".join(parts) if parts else "Top candidate: ABSTAIN / no commit"


def judge_one(client: OpenAI, model: str, diseases: List[str], diag_text: str) -> dict:
    msg = PROMPT.format(
        n=len(diseases),
        disease_list="\n".join(f"- {d}" for d in diseases),
        diag_text=(diag_text or "")[:1200],
    )
    is_reasoning = model.startswith(("gpt-5", "o"))
    kwargs = {"model": model, "messages": [{"role": "user", "content": msg}]}
    if is_reasoning:
        kwargs["max_completion_tokens"] = 200
    else:
        kwargs["temperature"] = 0.0
        kwargs["max_tokens"] = 200
    resp = client.chat.completions.create(**kwargs)
    raw = (resp.choices[0].message.content or "")
    parsed = _parse_json(raw)

    top1 = _normalise_to_list(diseases, str(parsed.get("top1", "Unknown")))
    top3_raw = parsed.get("top3") or []
    if not isinstance(top3_raw, list):
        top3_raw = [str(top3_raw)]
    top3 = [_normalise_to_list(diseases, str(x)) for x in top3_raw[:3]]
    abstained = bool(parsed.get("abstained", False))
    return {"top1": top1, "top3": top3, "abstained": abstained, "_raw": raw}


def process_case(client: OpenAI, model: str, diseases: List[str],
                 in_path: Path, out_path: Path) -> Tuple[bool, bool, bool, str]:
    case = json.loads(in_path.read_text())
    fin = case.setdefault("final", {})
    diag_text = extract_pred_text(case)
    judged = judge_one(client, model, diseases, diag_text)

    fin["predicted_remapped"] = judged["top1"]
    fin["predicted_set_remapped"] = judged["top3"]
    fin["abstained_judged"] = judged["abstained"]

    gt = (case.get("ground_truth") or "").strip().lower()
    norm = lambda s: (s or "").strip().lower()
    fin["top1_hit_judged"] = bool(gt and norm(judged["top1"]) == gt)
    fin["top3_hit_judged"] = bool(gt and any(norm(d) == gt for d in judged["top3"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(case, indent=2, ensure_ascii=False))
    return fin["top1_hit_judged"], fin["top3_hit_judged"], fin["abstained_judged"], judged["top1"]


def auto_diseases_path(in_dir: Path) -> Path:
    name = in_dir.name.lower()
    if "agentclinic" in name:
        return REPO / "data/kbs/agentclinic_a/diseases.csv"
    if "ddxplus" in name:
        return REPO / "data/kbs/ddxplus/diseases.csv"
    raise ValueError(f"Cannot auto-detect diseases CSV for {in_dir.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory of case_*.json files")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write judged case_*.json. "
                         "Default: data/benchmark_results/judged/<in_dir_name>/")
    ap.add_argument("--diseases", default=None,
                    help="CSV (column 'name') OR one-per-line list. "
                         "Default: auto-detected from in-dir name.")
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--force", action="store_true",
                    help="Re-judge even if out_path already exists")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr); sys.exit(1)

    in_dir = Path(args.in_dir).resolve()
    if not in_dir.is_dir():
        print(f"ERROR: in-dir not a directory: {in_dir}", file=sys.stderr); sys.exit(1)

    diseases_path = Path(args.diseases) if args.diseases else auto_diseases_path(in_dir)
    if diseases_path.suffix.lower() == ".csv":
        diseases = pd.read_csv(diseases_path)["name"].tolist()
    else:
        diseases = [l.strip() for l in diseases_path.read_text().splitlines() if l.strip()]
    print(f"[setup] {len(diseases)} closed-set diseases from {diseases_path.name}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO / "data/benchmark_results/judged" / in_dir.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("case_*.json"))
    if not files:
        print(f"ERROR: no case_*.json in {in_dir}", file=sys.stderr); sys.exit(1)
    print(f"[setup] {len(files)} cases | in={in_dir.name} out={out_dir} | model={args.model} workers={args.workers}")

    if not args.force:
        files = [f for f in files if not (out_dir / f.name).exists()]
        if not files:
            print("[skip] all cases already judged; pass --force to re-run.")
            return
        print(f"[setup] {len(files)} pending after skipping already-judged.")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    n_top1 = n_top3 = n_abs = 0
    fails = 0

    def _task(f: Path):
        return process_case(client, args.model, diseases, f, out_dir / f.name)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_task, f): f for f in files}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                t1, t3, ab, mapped = fut.result()
                n_top1 += int(t1)
                n_top3 += int(t3)
                n_abs += int(ab)
                tag = "T1" if t1 else ("T3" if t3 else "  ")
                ab_tag = "AB" if ab else "  "
                print(f"  {f.name}: {mapped[:40]:<40} [{tag}|{ab_tag}]")
            except Exception as e:
                fails += 1
                print(f"  {f.name}: FAIL ({e})")

    n = len(files)
    print()
    print(f"[done] {n} cases | T1={n_top1}/{n} ({n_top1/n:.0%}) "
          f"T3={n_top3}/{n} ({n_top3/n:.0%}) abstain={n_abs}/{n} ({n_abs/n:.0%}) fails={fails}")


if __name__ == "__main__":
    main()
