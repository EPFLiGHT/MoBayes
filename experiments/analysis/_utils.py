"""Shared utilities for analysis scripts."""

import json
import glob
import sys
from typing import List, Dict


def load_jsonl(result_dir: str) -> List[Dict]:
    """Load all JSONL results from a directory."""
    jsonl_files = glob.glob(f"{result_dir}/*.jsonl")
    results = []
    for file in jsonl_files:
        with open(file) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    return results


def load_jsonl_multi(dirs: List[str]) -> List[Dict]:
    """Load JSONL results from multiple directories."""
    all_results = []
    for d in dirs:
        all_results.extend(load_jsonl(d))
    return all_results


def safe_ratio(num: int, den: int) -> float:
    """Safe division returning 0.0 when denominator is zero."""
    return float(num) / float(den) if den else 0.0


def save_table(func, output_path: str, *args, **kwargs):
    """Capture a print-based table function's output to a file.

    Usage:
        save_table(print_table, "output.txt", metrics, title)
    """
    with open(output_path, 'w') as f:
        old_stdout = sys.stdout
        sys.stdout = f
        func(*args, **kwargs)
        sys.stdout = old_stdout
