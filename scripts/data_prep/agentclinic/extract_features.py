"""Build a grounded feature universe for AC-50 from the test cases themselves.

Two LLM passes:
  1. Per-case extraction (gpt-5.4-nano, 50 calls) — pull ATOMIC binary-presence
     symptoms from each Patient_Actor JSON.
  2. Aggressive cross-case dedup (gpt-5.4 full, 1 call) — merge synonyms,
     drop negatives/demographics/labs, target 60-100 canonical features.

Why: the v1 KB was built from a discriminative-feature prompt without
grounding in real case symptoms (311 features, 77% never appear in any
case). v2 features are exactly what AC's PatientAgent will answer YES to.

Usage:
    OPENAI_API_KEY=... PYTHONPATH=. python3 \\
        scripts/data_prep/agentclinic_v2/extract_features.py \\
        --cases data/cases/agentclinic_a/selected_cases.jsonl \\
        --out data/agentclinic_50_v2/features.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import List

from openai import OpenAI


REPO = Path(__file__).resolve().parents[3]


EXTRACT_PROMPT = """You are a clinical knowledge engineer. Given an OSCE patient case, extract a list of ATOMIC binary-presence clinical symptoms / antecedents the patient is reporting.

Rules:
- One concept per item — "fever" not "fever for 3 days with chills".
- Each item is something a patient could plausibly answer YES or NO to.
- Use general clinical phrasing, not patient slang ("dyspnea on exertion" not "out of breath when walking").
- Include positive findings only (what the patient HAS), not denials.
- Cover: presenting symptom, secondary symptoms, key items mentioned in History narrative.
- Drop demographics (age/sex), test results, vital signs, and medication names — these aren't symptoms answerable by patient yes/no.
- Aim for 8-15 atomic items per case. Do NOT pad.

Case (OSCE Patient_Actor):
---
{case_json}
---

Output a JSON array of strings, ONLY the array, nothing else:
["symptom 1", "symptom 2", ...]"""


DEDUP_PROMPT = """You are a clinical knowledge engineer building a diagnostic feature schema.

You will be given a list of {n} candidate symptom phrases. Many are paraphrases of the same clinical concept. Your job: produce a SHORT canonical list (target 60-100 features) by AGGRESSIVELY merging.

Hard rules:

(1) DROP entirely:
  - Anything starting with "no ", "denies ", "without ", "absence of", "lack of" — negatives are not features the patient can answer YES to.
  - Demographics ("67-year-old", "female", "male").
  - Test results, lab values, vital signs ("BP 120/80", "WBC 12k", "MRI showed…").
  - Medication names alone ("acetaminophen", "metoprolol") unless framed as a symptom (e.g. "current antibiotic use" is fine).

(2) MERGE as ONE canonical feature:
  - Synonyms: "fever" + "elevated temperature" + "pyrexia" → "fever"
  - Same concept different wording: "occasional alcohol consumption" + "drinks alcohol occasionally" → "alcohol use"
  - Granular states of same axis: "altered mental status" + "confusion" + "somnolence" + "lethargy" → "altered mental status" (single feature; do not split into 5)
  - Body-region paraphrases: "RUQ pain" + "right upper quadrant abdominal pain" + "discomfort in the right upper quadrant of the abdomen" → "right upper quadrant pain"
  - Tense/timing variants: "ongoing fever" + "fever for 2 days" → "fever"
  - Behavioral cluster: "bizarre behavior" + "inappropriate behavior" + "inappropriate laughter" → "behavioral disturbance"

(3) KEEP DISTINCT:
  - Different anatomical locations: "abdominal pain" vs "chest pain" vs "RUQ pain" — different.
  - Different qualities of same location: "dry cough" vs "productive cough" — different.
  - Symptom vs antecedent: "smoking history" vs "shortness of breath" — different.

Style: lowercase, neutral clinical phrasing, no patient slang, no parenthetical asides.

Input phrases:
---
{phrases}
---

Output a JSON array of canonical feature names, ONLY the array:
["feature 1", "feature 2", ...]"""


_NEG_RX = re.compile(r"^(no |denies |without |absence of |lack of |negative for )", re.I)


def llm(client: OpenAI, model: str, prompt: str, max_tok: int) -> str:
    is_reasoning = model.startswith(("gpt-5", "o"))
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if is_reasoning:
        kwargs["max_completion_tokens"] = max_tok
    else:
        kwargs["temperature"] = 0.0
        kwargs["max_tokens"] = max_tok
    return (client.chat.completions.create(**kwargs).choices[0].message.content or "").strip()


def parse_json_list(text: str) -> List[str]:
    m = re.search(r"\[[\s\S]+\]", text)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
        return [str(i).strip() for i in items if isinstance(i, str) and str(i).strip()]
    except Exception:
        return []


def feature_id(name: str) -> str:
    """Stable id: f_<slug>_<hash6>."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]
    h = hashlib.md5(name.lower().encode()).hexdigest()[:8]
    return f"f_{slug}_{h}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/cases/agentclinic_a/selected_cases.jsonl")
    ap.add_argument("--out", default="data/agentclinic_50_v2/features.csv")
    ap.add_argument("--extract-model", default="gpt-5.4-nano",
                    help="Per-case extraction (50 calls, fast)")
    ap.add_argument("--dedup-model", default="gpt-5.4",
                    help="Cross-case canonicalization (1 call, needs strong model)")
    ap.add_argument("--raw-dump", default="data/agentclinic_50_v2/raw_extracted.json")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr); sys.exit(1)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    cases_path = Path(args.cases) if Path(args.cases).is_absolute() else REPO / args.cases
    out_path = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    raw_path = Path(args.raw_dump) if Path(args.raw_dump).is_absolute() else REPO / args.raw_dump
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    cases = [json.loads(l) for l in cases_path.open()]
    print(f"[setup] {len(cases)} cases | extract={args.extract_model} | dedup={args.dedup_model}",
          flush=True)

    # Pass 1 — per-case extraction
    raw = []
    for i, c in enumerate(cases):
        inner = c.get("OSCE_Examination", c)
        gt = inner.get("Correct_Diagnosis", "")
        case_json = json.dumps(inner.get("Patient_Actor", {}), indent=2)[:2200]
        try:
            symptoms = parse_json_list(
                llm(client, args.extract_model,
                    EXTRACT_PROMPT.format(case_json=case_json), max_tok=600)
            )
        except Exception as e:
            print(f"  [{i+1}/{len(cases)}] ERROR: {e}", flush=True); symptoms = []
        raw.append({"case_idx": i, "ground_truth": gt, "symptoms": symptoms})
        print(f"  [{i+1}/{len(cases)}] gt={gt[:30]:<32} → {len(symptoms)} symptoms", flush=True)
    raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False))
    print(f"[saved] raw → {raw_path}", flush=True)

    # Pre-filter: drop negatives + dedupe identical strings
    distinct = sorted({s for r in raw for s in r["symptoms"] if not _NEG_RX.match(s)})
    print(f"[filter] {sum(len(r['symptoms']) for r in raw)} raw → "
          f"{len(distinct)} distinct (negatives stripped)", flush=True)

    # Pass 2 — aggressive cross-case dedup
    canonical = parse_json_list(
        llm(client, args.dedup_model,
            DEDUP_PROMPT.format(n=len(distinct), phrases="\n".join(f"- {p}" for p in distinct)),
            max_tok=10000)
    )
    if not canonical:
        print("[error] dedup parse failed; falling back to distinct list", file=sys.stderr)
        canonical = distinct
    # Final unique enforce
    seen, final = set(), []
    for c in canonical:
        n = c.lower().strip()
        if n and n not in seen:
            seen.add(n); final.append(c)
    print(f"[dedup] {len(distinct)} → {len(final)} canonical features", flush=True)

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_id", "name", "type", "values", "missing_token"])
        for name in final:
            w.writerow([feature_id(name), name, "binary", '["yes", "no"]', "unknown"])
    print(f"[saved] {out_path} ({len(final)} features)", flush=True)


if __name__ == "__main__":
    main()
