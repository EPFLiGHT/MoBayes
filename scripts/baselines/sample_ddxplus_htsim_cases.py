"""Sample 50 stratified DDxPlus cases for the HT-simulator benchmark column.

Strategy:
    - Source: MEDDxAgent's own DDxPlus loader (StreamBench, test split, 1764
      rows, 49 distinct pathologies x 36 cases each).
    - First-occurrence-per-pathology gives full disease coverage (49 cases).
    - Extra slot (#50): second occurrence of the first pathology in test order
      (Acute COPD exacerbation / infection, test idx=1).
    - Deterministic: no shuffle, no seed. Same selection every run.

Output: data/cases/ddxplus_htsim_a/selected_cases.jsonl with StreamBench schema +
patient_id of the form ``htsim_<test_idx>``. Schema is identical to
data/streambench_50/selected_cases.jsonl so existing
baselines/streambench/adapters/* work unchanged.

Run from the meddxagent venv (it has `datasets`, ours doesn't):
    .venv/bin/python \\
        scripts/baselines/sample_ddxplus_htsim_cases.py
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from datasets import load_dataset


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data/cases/ddxplus_htsim_a"
OUT_PATH = OUT_DIR / "selected_cases.jsonl"


def _row_to_case(row: dict, test_idx: int) -> dict:
    case = dict(row)
    case["patient_id"] = f"htsim_{test_idx:04d}"
    case["_test_index"] = test_idx
    case["_source"] = "streambench/ddxplus_kb/test"
    return case


def main() -> None:
    ds = load_dataset("appier-ai-research/StreamBench", "ddxplus_kb")["test"]

    first_occurrence: "OrderedDict[str, int]" = OrderedDict()
    second_occurrence_first: int | None = None
    first_pathology: str | None = None
    for i, row in enumerate(ds):
        path = row["PATHOLOGY"]
        if path not in first_occurrence:
            first_occurrence[path] = i
            if first_pathology is None:
                first_pathology = path
        elif path == first_pathology and second_occurrence_first is None:
            second_occurrence_first = i

    if second_occurrence_first is None:
        raise RuntimeError(
            "No second occurrence found for the first pathology — "
            "test split layout has changed."
        )

    selected_indices = list(first_occurrence.values()) + [second_occurrence_first]
    if len(selected_indices) != 50:
        raise RuntimeError(
            f"Expected 50 indices, got {len(selected_indices)} "
            f"({len(first_occurrence)} pathologies + 1 extra)"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for idx in selected_indices:
            row = ds[idx]
            f.write(json.dumps(_row_to_case(row, idx)) + "\n")

    print(f"[done] wrote {len(selected_indices)} cases → {OUT_PATH}")
    print(f"       distinct pathologies: {len(first_occurrence)}")
    print(f"       extra (#50) is second '{first_pathology}' at test idx {second_occurrence_first}")


if __name__ == "__main__":
    main()
