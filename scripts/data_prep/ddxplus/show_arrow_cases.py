#!/usr/bin/env python3
"""
Print readable sample patient cases from a DDXPlus Hugging Face Arrow shard.

Examples:
    python3 scripts/data_prep/ddxplus/show_arrow_cases.py \
        /path/to/ddxplus-train-00000-of-00002.arrow

    python3 scripts/data_prep/ddxplus/show_arrow_cases.py \
        /path/to/ddxplus-train-00000-of-00002.arrow \
        --num-cases 5 --offset 10

    python3 scripts/data_prep/ddxplus/show_arrow_cases.py \
        /path/to/ddxplus-train-00000-of-00002.arrow \
        --random-sample --seed 7
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_ARROW = Path(
    "data/raw/ddxplus_cache/ddxplus_hf/aai530-group6___ddxplus/default/0.0.0/"
    "2ad986acc1ec62fb4a94171acc43f4fdd5bfde53/ddxplus-train-00000-of-00002.arrow"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show readable sample patient cases from a DDXPlus Arrow shard."
    )
    parser.add_argument(
        "arrow_file",
        nargs="?",
        default=str(DEFAULT_ARROW),
        help="Path to a .arrow file from the DDXPlus HF cache.",
    )
    parser.add_argument(
        "-n",
        "--num-cases",
        type=int,
        default=3,
        help="Number of patient cases to print.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting row index when sampling sequentially.",
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Randomly sample rows instead of using a sequential offset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --random-sample.",
    )
    parser.add_argument(
        "--top-ddx",
        type=int,
        default=5,
        help="How many differential diagnoses to print per case.",
    )
    return parser.parse_args()


def load_dataset(arrow_path: Path):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit(
            "This script requires the 'datasets' package. "
            "Install it with: python3 -m pip install datasets"
        ) from exc

    return Dataset.from_file(str(arrow_path))


def find_metadata_file(start_path: Path, filename: str) -> Optional[Path]:
    for parent in [start_path.parent, *start_path.parents]:
        candidate = parent / filename
        if candidate.exists():
            return candidate
    return None


def load_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    return json.loads(path.read_text())


def parse_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{(":
        return value
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return value


def sex_label(value: str) -> str:
    return {"M": "male", "F": "female"}.get(value, str(value))


def decode_value(raw_value: str, meta: Dict[str, Any]) -> str:
    value_meaning = meta.get("value_meaning") or {}
    if raw_value in value_meaning:
        decoded = value_meaning[raw_value].get("en") or value_meaning[raw_value].get("fr")
        if decoded:
            decoded_text = str(decoded)
            if decoded_text == "Y":
                return "yes"
            if decoded_text == "N":
                return "no"
            return decoded_text
    return raw_value


def decode_evidence_token(token: str, evidence_meta: Dict[str, Dict[str, Any]]) -> Tuple[str, str, bool]:
    evidence_id, raw_value = (token.split("_@_", 1) + [None])[:2]
    meta = evidence_meta.get(evidence_id, {})
    question = meta.get("question_en") or evidence_id
    is_antecedent = bool(meta.get("is_antecedent"))

    if raw_value is None:
        answer = "yes"
    else:
        answer = decode_value(raw_value, meta)

    return question, answer, is_antecedent


def format_evidence_lines(tokens: Sequence[str], evidence_meta: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    antecedents: List[str] = []
    symptoms: List[str] = []
    for token in tokens:
        question, answer, is_antecedent = decode_evidence_token(token, evidence_meta)
        line = f"- {question} -> {answer}"
        if is_antecedent:
            antecedents.append(line)
        else:
            symptoms.append(line)
    return antecedents, symptoms


def format_differential(entries: Any, top_k: int) -> List[str]:
    parsed = parse_literal(entries)
    if not isinstance(parsed, list):
        return []

    lines: List[str] = []
    for item in parsed[:top_k]:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            name, score = item
            try:
                lines.append(f"- {name}: {float(score):.4f}")
            except (TypeError, ValueError):
                lines.append(f"- {name}: {score}")
        else:
            lines.append(f"- {item}")
    return lines


def case_indices(total_rows: int, num_cases: int, offset: int, random_sample: bool, seed: int) -> List[int]:
    if total_rows <= 0:
        return []

    num_cases = max(0, min(num_cases, total_rows))
    if random_sample:
        rng = random.Random(seed)
        return sorted(rng.sample(range(total_rows), num_cases))

    start = max(0, offset)
    end = min(total_rows, start + num_cases)
    return list(range(start, end))


def print_case(index: int, row: Dict[str, Any], evidence_meta: Dict[str, Dict[str, Any]], top_ddx: int) -> None:
    evidences = parse_literal(row.get("EVIDENCES", []))
    if not isinstance(evidences, list):
        evidences = [str(evidences)]

    antecedents, symptoms = format_evidence_lines(evidences, evidence_meta)
    initial_question, initial_answer, _ = decode_evidence_token(
        str(row.get("INITIAL_EVIDENCE", "")),
        evidence_meta,
    )
    differential_lines = format_differential(row.get("DIFFERENTIAL_DIAGNOSIS"), top_ddx)

    print(f"Case #{index}")
    print(f"Patient: {row.get('AGE')} year old {sex_label(str(row.get('SEX', 'unknown')))}")
    print(f"Ground truth pathology: {row.get('PATHOLOGY', 'unknown')}")
    print(f"Initial evidence: {initial_question} -> {initial_answer}")
    print(f"Total evidence items: {len(evidences)}")

    if antecedents:
        print("Antecedents:")
        for line in antecedents:
            print(f"  {line}")

    if symptoms:
        print("Symptoms / findings:")
        for line in symptoms:
            print(f"  {line}")

    if differential_lines:
        print(f"Top {len(differential_lines)} differential diagnoses:")
        for line in differential_lines:
            print(f"  {line}")

    print()


def main() -> int:
    args = parse_args()
    arrow_path = Path(args.arrow_file).expanduser().resolve()
    if not arrow_path.exists():
        print(f"Arrow file not found: {arrow_path}", file=sys.stderr)
        return 1

    dataset = load_dataset(arrow_path)
    evidence_meta = load_json(find_metadata_file(arrow_path, "release_evidences.json"))

    indices = case_indices(
        total_rows=len(dataset),
        num_cases=args.num_cases,
        offset=args.offset,
        random_sample=args.random_sample,
        seed=args.seed,
    )

    if not indices:
        print("No cases selected.", file=sys.stderr)
        return 1

    print(f"Loaded {len(dataset)} rows from {arrow_path}")
    print(f"Showing {len(indices)} case(s): {', '.join(str(i) for i in indices)}")
    print()

    for index in indices:
        print_case(index, dataset[index], evidence_meta, args.top_ddx)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
