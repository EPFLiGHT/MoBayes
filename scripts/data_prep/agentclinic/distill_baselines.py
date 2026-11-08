"""(Optional) Enrich the KB by distilling winning Q-A pairs from strong open-vocab
baselines into canonical MoBayes features.

Why it helps:
  Open-vocab baselines often ask narrowly-discriminative questions tightly bound
  to the chief complaint (e.g. "pain reproduces on bone palpation",
  "central umbilication", "rapidly enlarging non-tender breast mass"). Their
  question repertoire encodes diagnostic discriminators that a closed-vocabulary
  feature set may lack. Distilling those into named features expands the
  feature universe; the next step (the LLM elicitation in
  `scripts/data_prep/llmkb/generate_distributions.py`) then assigns
  P(feature | disease) over the expanded set.

Pipeline:
  1. Load winning Q-A pairs (from prior extraction step).
  2. Chunk into batches → LLM distills each batch into a bounded list of
     canonical feature names; drop negatives / demographics / lab results.
  3. Dedup the union (LLM, with "preserve specificity" rule).
  4. Diff against the existing feature set → write the NEW features.
  5. Emit features.json + diseases.json for downstream likelihood elicitation.

Usage:
    OPENAI_API_KEY=... PYTHONPATH=. python3 \
      scripts/data_prep/agentclinic/distill_baselines.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI


REPO = Path(__file__).resolve().parents[3]
PAIRS = REPO / "data/kbs/agentclinic_a/raw_extraction/winning_qa_pairs.jsonl"
OUT = REPO / "data/kbs/agentclinic_a"
V2_FEATS = REPO / "data/agentclinic_50_v2/features.csv"
DISEASES = REPO / "data/kbs/agentclinic_a/diseases.csv"


CHUNK_DISTILL_PROMPT = """You are a clinical knowledge engineer building a diagnostic feature schema for a Bayesian agent.

You will receive {n} doctor question + patient answer pairs from a SUCCESSFUL diagnosis dialog (doctor reached the correct diagnosis). The ground-truth disease is "{gt}".

Your job: extract ONLY the CONFIRMED POSITIVE FINDINGS the patient actually has, based on their answer. These are the discriminative features that helped reach the diagnosis.

Hard rules:

(1) ONLY extract features the patient CONFIRMED as present:
  - Question: "Do you have rough scaly lesion?" → Answer: "Yes, it feels rough and scaly" → EXTRACT "rough scaly skin lesion"
  - Question: "Any pain?" → Answer: "No" → DROP (denied)
  - Question: "Lesion size?" → Answer: "About 1-2 cm coin-sized" → EXTRACT "coin-sized skin lesion"
  - Question: "X-ray done?" → Answer: "No, not yet" → DROP (test not done, not a finding)

(2) Multi-part questions: extract each CONFIRMED component separately. If the question asked 3 things and patient confirmed 2, denied 1 → 2 features extracted, not 3.
  - "fever, chills, weight loss?" → "Yes weight loss, no fever" → EXTRACT "unintentional weight loss" only

(3) DROP entirely:
  - Anything denied by patient ("no", "haven't", "don't")
  - Test/lab/imaging RESULTS ("BUN 80") — but keep symptoms they probe ("polyuria", "easy bruising")
  - Demographics alone (age, sex)
  - Whether prior tests were DONE ("have you had an MRI?")
  - Treatment/medication queries

(4) ATOMIC + SPECIFIC clinical phrasing:
  - One concept per feature. Keep specificity: "rapidly enlarging non-tender breast mass" not just "breast mass" if the patient confirmed all attributes.
  - Lowercase except proper nouns. No patient slang.

(5) REPRODUCIBILITY: a downstream LLM will need to answer P(feature=YES | disease) for each feature × 31 diseases. Features must be unambiguous and testable.

Doctor-patient pairs (ground truth disease = "{gt}"):
---
{pairs}
---

Output a JSON array of canonical feature names (the CONFIRMED positive findings only, typically 5-20 per chunk), ONLY the array."""


DEDUP_PROMPT = """You are a clinical knowledge engineer building a diagnostic feature schema.

You will receive {n} candidate symptom / history / risk-factor phrases pooled from many cases. Many are paraphrases of the same clinical concept; produce a canonical list.

Hard rules:

(1) DROP entirely:
  - Negative phrasings ("no fever", "denies cough").
  - Demographic-only items (pure age/sex without clinical content).
  - Test/lab/imaging RESULTS (we want patient-answerable items).
  - Pure feature_type metadata ("medical history clinician notes").

(2) KEEP both general AND specific variants (DO NOT merge across granularity):
  - Keep "skin lesion" AND "rough scaly plaque on sun-exposed area" as separate features.
  - Keep "fever" AND "high-grade fever (>39°C) with rigors" as separate features.
  Rationale: extractor finds the general one; engine asks specific one to discriminate.

(3) MERGE only true duplicates (same concept, same specificity):
  - "fever" + "Fever" + "elevated temperature" → "fever".
  - "RUQ pain" + "right upper quadrant pain" → "right upper quadrant pain".
  - "shortness of breath" + "dyspnea" → "shortness of breath".

(4) Style: clear clinical phrasing, lowercase except proper nouns, no parentheticals describing other features.

Output 80-200 canonical features (avoid shrinking below 80).

Input phrases:
---
{phrases}
---

Output a JSON array of canonical feature names, ONLY the array.
"""


def feature_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]
    h = hashlib.md5(name.lower().encode()).hexdigest()[:8]
    return f"f_{slug}_{h}"


def llm_call(client, model, prompt, max_tok):
    is_reasoning = model.startswith(("gpt-5", "o"))
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if is_reasoning:
        kwargs["max_completion_tokens"] = max_tok
    else:
        kwargs["temperature"] = 0.0
        kwargs["max_tokens"] = max_tok
    return (client.chat.completions.create(**kwargs).choices[0].message.content or "").strip()


def parse_json_list(text):
    m = re.search(r"\[[\s\S]+\]", text)
    if not m: return []
    try:
        return [str(x).strip() for x in json.loads(m.group(0)) if str(x).strip()]
    except Exception:
        return []


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set")
    if not PAIRS.exists():
        sys.exit(f"ERROR: {PAIRS} not found — run the extraction step first")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    pairs = [json.loads(l) for l in open(PAIRS)]
    print(f"[input] {len(pairs)} winning Q-A pairs from {len(set(r['case'] for r in pairs))} cases", flush=True)

    # Group pairs by case (so distill prompt sees coherent dialog) — order by source then case
    by_case = {}
    for r in pairs:
        key = (r["source"], r["case"], r["gt"])
        by_case.setdefault(key, []).append(r)
    print(f"[input] {len(by_case)} (source, case) groups", flush=True)

    # Distill in chunks per group: at most 25 Q-A per LLM call
    distilled = []
    distill_log = []
    CHUNK_SIZE = 25
    n_calls = sum(max(1, (len(p) + CHUNK_SIZE - 1) // CHUNK_SIZE) for p in by_case.values())
    print(f"[distill] making ~{n_calls} LLM calls (gpt-5.4-nano, chunk_size={CHUNK_SIZE})", flush=True)
    done = 0
    for (src, ci, gt), case_pairs in by_case.items():
        for j in range(0, len(case_pairs), CHUNK_SIZE):
            chunk = case_pairs[j : j + CHUNK_SIZE]
            chunk_text = "\n".join(
                f"Q: {r['q'][:400]}\nA: {r['a'][:400]}" for r in chunk
            )
            text = llm_call(
                client, "gpt-5.4-nano",
                CHUNK_DISTILL_PROMPT.format(n=len(chunk), gt=gt, pairs=chunk_text),
                max_tok=1500,
            )
            feats = parse_json_list(text)
            distilled.extend(feats)
            distill_log.append({"source": src, "case": ci, "gt": gt, "chunk_idx": j // CHUNK_SIZE,
                                 "n_pairs": len(chunk), "n_distilled": len(feats), "features": feats})
            done += 1
            if done % 10 == 0 or done == n_calls:
                print(f"  [{done}/{n_calls}] distilled so far: {len(distilled)} raw → "
                      f"{len(set(s.lower().strip() for s in distilled))} unique-lower", flush=True)
    print(f"[distill] done. {len(distilled)} raw → "
          f"{len(set(s.lower().strip() for s in distilled))} unique (case-insensitive)", flush=True)

    # Save raw distilled list
    OUT.mkdir(parents=True, exist_ok=True)
    raw_out = OUT / "raw_extraction"
    raw_out.mkdir(exist_ok=True)
    NEG_RX = re.compile(r"^(no |denies |without |absence of |lack of |negative for )", re.I)
    pool = sorted({s.strip() for s in distilled if s.strip() and not NEG_RX.match(s.strip())})
    with (raw_out / "distilled_raw.json").open("w") as f:
        json.dump({"n_unique": len(pool), "items": pool, "log": distill_log}, f, indent=2)

    # LLM dedup pass
    print(f"[dedup] LLM dedup of {len(pool)} candidates → target 80-200 canonical", flush=True)
    text = llm_call(
        client, "gpt-5.4",
        DEDUP_PROMPT.format(n=len(pool), phrases="\n".join(f"- {p}" for p in pool)),
        max_tok=8000,
    )
    canonical = parse_json_list(text)
    if not canonical:
        sys.exit(f"[error] dedup LLM returned no parseable list. Raw:\n{text[:500]}")
    seen, final = set(), []
    for n in canonical:
        k = n.lower().strip()
        if k and k not in seen:
            seen.add(k); final.append(n)
    print(f"[dedup] {len(pool)} → {len(final)} canonical", flush=True)

    # Diff against v2
    v2_names = [r["name"].strip().lower() for r in csv.DictReader(open(V2_FEATS))]
    v2_set = set(v2_names)
    new_features = [n for n in final if n.lower() not in v2_set]
    overlap = [n for n in final if n.lower() in v2_set]
    print(f"\n[diff vs v2] v2={len(v2_set)} | distilled={len(final)} | "
          f"new={len(new_features)} | overlap={len(overlap)}", flush=True)
    with (raw_out / "diff_vs_v2.json").open("w") as f:
        json.dump({"v2_count": len(v2_set), "distilled_count": len(final),
                    "new_features": new_features, "overlap_with_v2": overlap}, f, indent=2)

    # Combined v5 = v2 ∪ new (so we keep v2 working features and ADD baseline-distilled discriminators)
    v2_full = list(csv.DictReader(open(V2_FEATS)))
    v2_keep = [{"feature_id": r["feature_id"], "feature_type": "symptom",
                 "base_symptom": "", "name": r["name"], "type": "binary"} for r in v2_full]
    new_feats = [{"feature_id": feature_id(n), "feature_type": "symptom",
                   "base_symptom": "", "name": n, "type": "binary"} for n in new_features]
    all_feats = v2_keep + new_feats
    print(f"[v5 KB] features = v2({len(v2_keep)}) + new({len(new_feats)}) = {len(all_feats)}", flush=True)

    (OUT / "raw_input").mkdir(exist_ok=True)
    (OUT / "raw_output").mkdir(exist_ok=True)
    with (OUT / "raw_input/features.json").open("w") as f:
        json.dump({"_meta": {"n": len(all_feats),
                              "source": f"v2({len(v2_keep)}) + baseline-distilled-new({len(new_feats)})"},
                    "features": all_feats}, f, indent=2)

    diseases = [{"disease_id": r["disease_id"], "name": r["name"]}
                for r in csv.DictReader(open(DISEASES))]
    with (OUT / "raw_input/diseases.json").open("w") as f:
        json.dump({"diseases": diseases}, f, indent=2)

    print(f"\n[saved] {OUT}/raw_input/  ({len(diseases)} diseases × {len(all_feats)} features)")
    print(f"[saved] {raw_out}/diff_vs_v2.json — review the new features before likelihood gen")
    print()
    print("Next: generate likelihoods for the v5 KB (this costs ~$0.50)")
    print("  cd scripts/data_prep/llmkb && \\")
    print("  LLMKB_MODEL=gpt-5.4-nano \\")
    print(f"  LLMKB_DISEASES_JSON_PATH={OUT/'raw_input/diseases.json'} \\")
    print(f"  LLMKB_FEATURES_JSON_PATH={OUT/'raw_input/features.json'} \\")
    print(f"  LLMKB_DISTRIBUTIONS_JSON_PATH={OUT/'raw_output/distributions.json'} \\")
    print("  PYTHONPATH=. python3 generate_distributions.py")


if __name__ == "__main__":
    main()
