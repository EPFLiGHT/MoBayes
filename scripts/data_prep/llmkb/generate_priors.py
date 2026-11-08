#!/usr/bin/env python3
"""
Generate population-level priors for LLM-generated KB diseases.

Asks the LLM to estimate prevalence per 100,000 people for each disease,
then normalises to a valid prior distribution. Validates via:
  1. Relative ordering sanity checks (common > rare)
  2. Sum-to-1 constraint
  3. No disease gets zero prior (floor applied)

Usage:
    source .env
    LLMKB_API_KEY="$OPEN_ROUTER" \
    LLMKB_BASE_URL="https://openrouter.ai/api/v1" \
    LLMKB_MODEL="google/gemini-2.5-flash" \
    python3 generate_priors.py \
        --diseases ../../../data/raw/llmkb/input/diseases18.json \
        --output ../../../data/raw/llmkb/output/priors_diseases18.csv
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

from openai import OpenAI
from loaders import load_diseases


PRIOR_FLOOR = 1e-4  # No disease gets less than 0.01%


def make_client() -> OpenAI:
    api_key = os.environ.get("LLMKB_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set LLMKB_API_KEY or OPENAI_API_KEY.")
    kwargs = {"api_key": api_key}
    base_url = os.environ.get("LLMKB_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def build_prevalence_prompt(diseases: List[Dict[str, str]]) -> str:
    disease_list = "\n".join(
        f"- {d['disease_id']}: {d['name']}" for d in diseases
    )
    return f"""You are an epidemiologist estimating disease prevalence for a diagnostic system.

For each disease below, estimate: out of 100,000 people presenting to a general hospital emergency department in a year, how many would be diagnosed with this disease?

This is NOT general population prevalence — it is the proportion among ED patients specifically.
Use your best medical knowledge. Be realistic: common conditions (e.g., gastroenteritis, asthma) should have high counts; rare conditions (e.g., diphtheria, alpha-1 antitrypsin deficiency) should have very low counts.

DISEASES:
{disease_list}

RULES:
1. Return a JSON object mapping disease_id to estimated count per 100,000 ED patients.
2. Counts must be positive integers (minimum 1).
3. The counts should reflect realistic relative differences — e.g., gastroenteritis should be 100x+ more common than diphtheria.
4. Do not make all counts similar — the whole point is to capture prevalence differences.

Return ONLY valid JSON, no explanation. Example format:
{{"d_resp_ar": 5000, "d_resp_a": 3000, "d_acqu_diph": 2}}"""


def validate_prevalence(prevalence: Dict[str, int], diseases: List[Dict[str, str]]) -> bool:
    """Sanity checks on LLM-generated prevalence estimates."""
    disease_ids = {d["disease_id"] for d in diseases}

    # Check all diseases present
    missing = disease_ids - set(prevalence.keys())
    if missing:
        print(f"  WARN: Missing diseases: {missing}")
        return False

    # Check all positive
    for did, count in prevalence.items():
        if count <= 0:
            print(f"  WARN: Non-positive count for {did}: {count}")
            return False

    # Sanity: common diseases should be more prevalent than rare ones
    common = {"d_acqu_gastro", "d_acqu_candi", "d_resp_a", "d_resp_ar"}
    rare = {"d_acqu_diph", "d_resp_a1ad", "d_acqu_chorio", "d_acqu_asper"}

    common_present = common & set(prevalence.keys())
    rare_present = rare & set(prevalence.keys())

    if common_present and rare_present:
        min_common = min(prevalence[d] for d in common_present)
        max_rare = max(prevalence[d] for d in rare_present)
        if min_common <= max_rare:
            print(f"  WARN: Common diseases ({min_common}) not more prevalent than rare ({max_rare})")
            print(f"    Common: {[(d, prevalence[d]) for d in common_present]}")
            print(f"    Rare: {[(d, prevalence[d]) for d in rare_present]}")
            return False

    # Check spread: ratio of max to min should be at least 10x
    counts = list(prevalence.values())
    ratio = max(counts) / max(min(counts), 1)
    if ratio < 10:
        print(f"  WARN: Spread too narrow (max/min = {ratio:.1f}x, want >= 10x)")
        return False

    print(f"  Validation passed: {len(prevalence)} diseases, spread {ratio:.0f}x")
    return True


def prevalence_to_priors(prevalence: Dict[str, int]) -> Dict[str, float]:
    """Normalise counts to probabilities with floor."""
    total = sum(prevalence.values())
    priors = {}
    for did, count in prevalence.items():
        priors[did] = max(count / total, PRIOR_FLOOR)

    # Re-normalise after floor
    total_p = sum(priors.values())
    return {did: p / total_p for did, p in priors.items()}


def main():
    parser = argparse.ArgumentParser(description="Generate LLM-estimated priors")
    parser.add_argument("--diseases", required=True, help="Path to diseases JSON")
    parser.add_argument("--output", required=True, help="Output priors.csv path")
    parser.add_argument("--retries", type=int, default=3, help="Max LLM retries")
    args = parser.parse_args()

    model = os.environ.get("LLMKB_MODEL", "gpt-4o-mini")
    diseases = load_diseases(args.diseases)
    client = make_client()

    print(f"Generating priors for {len(diseases)} diseases using {model}")

    prompt = build_prevalence_prompt(diseases)

    for attempt in range(1, args.retries + 1):
        print(f"\nAttempt {attempt}/{args.retries}...")

        try:
            if os.environ.get("LLMKB_BASE_URL"):
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.choices[0].message.content
            else:
                response = client.responses.create(
                    model=model,
                    input=prompt,
                    temperature=0.1,
                )
                text = response.output_text

            # Strip markdown code fences if present
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            prevalence = json.loads(text)

            # Convert string values to int if needed
            prevalence = {k: int(v) for k, v in prevalence.items()}

            print("\n  Raw prevalence estimates (per 100K ED patients):")
            for did in sorted(prevalence, key=prevalence.get, reverse=True):
                name = next((d["name"] for d in diseases if d["disease_id"] == did), did)
                print(f"    {prevalence[did]:>6d}  {name}")

            if validate_prevalence(prevalence, diseases):
                priors = prevalence_to_priors(prevalence)

                # Write CSV
                os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
                with open(args.output, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["disease_id", "prior"])
                    w.writeheader()
                    for did in sorted(priors.keys()):
                        w.writerow({"disease_id": did, "prior": round(priors[did], 6)})

                print(f"\n  Priors written to {args.output}")
                print("\n  Final priors:")
                for did in sorted(priors, key=priors.get, reverse=True):
                    name = next((d["name"] for d in diseases if d["disease_id"] == did), did)
                    print(f"    {priors[did]:>8.4f}  ({priors[did]*100:>5.2f}%)  {name}")

                return

            print("  Retrying...")

        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nFailed after {args.retries} attempts.")
    sys.exit(1)


if __name__ == "__main__":
    main()
