"""Parser audit (appendix C.17): automatic check of the LLM parser.

Compares every logged parser extraction of the 50 set_b DDxPlus PatientSim
dialogues against the patient profile's value for the queried feature (the
`_ddxplus_meta.feature_map`). No API calls.

The join is exact on `feature_id`. Values are canonicalised to schema space
before comparison:
  binary  "YES"/"no"  -> "yes"/"no"
  numeric "7/10"      -> "7"   (also reported with +/-1 tolerance)
  categ.  kept as-is

Two slices, because a feature missing from the profile has no explicit value:
  A (clean) : only extractions whose feature is in the profile. This is the
              parser-accuracy number.
  B (broad) : all extractions; features missing from the profile count as
              "no"/absent (DDxPlus convention). This also counts simulator
              over-affirmation, so it is an upper bound on channel error.

Usage (from this folder):
    PYTHONPATH=. python3 experiments/parser_audit.py
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from llm_interface.ddxplus_display import is_numeric_scale  # noqa: E402

LOG_DIR = REPO / "data/benchmark_results/set_b/ddxplus_patientsim/mobayes/bmbe"
PROFILES = REPO / "data/cases/ddxplus_patientsim_b/plain/profiles.json"
FEATURES_CSV = REPO / "data/kbs/ddxplus/features.csv"
OUT_JSON = REPO / "experiments/parser_audit_results.json"


def load_schema():
    sch = {}
    for r in csv.DictReader(FEATURES_CSV.open()):
        raw = r.get("values", "[]")
        try:
            vals = json.loads(raw) if raw else []
        except json.JSONDecodeError:
            vals = [x.strip() for x in raw.strip("[]").split(",")]
        sch[r["feature_id"]] = [str(v) for v in vals]
    return sch


def canon(val) -> str:
    """Canonicalise a value (parser output or profile GT) to schema space."""
    if val is None:
        return "unknown"
    s = str(val).strip()
    if not s:
        return "unknown"
    if "/10" in s.replace(" ", ""):          # numeric "7/10" -> "7"
        return s.split("/")[0].strip()
    low = s.lower()
    if low in ("yes", "no", "unknown", "absent", "present"):
        return low
    return s


def profile_gt(feature_map: dict, fid: str):
    """Return canonical GT value if fid is explicitly in feature_map, else None."""
    if fid not in feature_map:
        return None
    desc = feature_map[fid]
    raw = desc.rsplit(": ", 1)[-1] if ": " in desc else desc
    return canon(raw)


def numeric_near(a: str, b: str, tol: int = 1) -> bool:
    try:
        return abs(int(float(a)) - int(float(b))) <= tol
    except (ValueError, TypeError):
        return False


def label(extracted_c: str, gt_c: str, numeric: bool):
    """correct / near (numeric +/-1) / wrong / miss(unknown)."""
    if extracted_c in ("unknown",):
        return "miss"
    if extracted_c == gt_c:
        return "correct"
    if numeric and numeric_near(extracted_c, gt_c):
        return "near"
    return "wrong"


def main():
    schema = load_schema()
    profiles = json.load(PROFILES.open())
    files = sorted(glob.glob(str(LOG_DIR / "case_*.json")))

    # slice -> Counter of labels; also per feature-type and confidence tracking
    A = Counter()   # clean: feature in feature_map
    B = Counter()   # broad: all, default-no for absent binary
    A_by_type = {"binary": Counter(), "numeric": Counter(), "categorical": Counter()}
    conf_of = {"correct": [], "wrong": [], "near": [], "miss": []}
    wrong_examples = []

    def ftype(fid):
        vals = schema.get(fid, [])
        if is_numeric_scale(vals):
            return "numeric"
        if set(v.lower() for v in vals) <= {"yes", "no"}:
            return "binary"
        return "categorical"

    n_extractions = 0
    for f in files:
        c = json.load(f and open(f))
        idx = c["case_idx"]
        fm = profiles[idx]["_ddxplus_meta"]["feature_map"]

        # collect (fid, extracted_value, confidence) from opening + qa
        items = []
        for t in c["turns"]:
            if t.get("kind") == "opening":
                for fid, v in (t.get("extracted_features") or {}).items():
                    items.append((fid, v, None))
            elif t.get("kind") == "qa":
                items.append((t["feature_id"], t.get("extracted_value"),
                              t.get("extraction_confidence")))

        for fid, ev, conf in items:
            n_extractions += 1
            ext_c = canon(ev)
            typ = ftype(fid)
            numeric = (typ == "numeric")

            gt = profile_gt(fm, fid)
            # --- Slice A: only if explicit ground truth ---
            if gt is not None:
                lab = label(ext_c, gt, numeric)
                A[lab] += 1
                A_by_type[typ][lab] += 1
                if conf is not None:
                    conf_of[lab].append(float(conf))
                if lab == "wrong" and len(wrong_examples) < 15:
                    wrong_examples.append(
                        {"case": idx, "fid": fid, "type": typ,
                         "extracted": ext_c, "gt": gt})

            # --- Slice B: broad, default-no for absent binary ---
            gt_b = gt
            if gt_b is None:
                gt_b = "no" if typ == "binary" else "absent"
            B[label(ext_c, gt_b, numeric)] += 1

    def pct(counter):
        tot = sum(counter.values())
        return {k: (counter[k], round(100.0 * counter[k] / tot, 1))
                for k in ("correct", "near", "wrong", "miss")} if tot else {}

    def acc(counter):
        tot = sum(counter.values())
        good = counter["correct"] + counter["near"]
        return round(100.0 * good / tot, 1) if tot else 0.0

    print(f"[parser_audit] {len(files)} cases | {n_extractions} extractions "
          f"| join=feature_id (exact)\n")

    print("=== Slice A (clean: feature in profile feature_map) ===")
    print(f"  n = {sum(A.values())}   accuracy (correct+near) = {acc(A)}%")
    for k, (cnt, p) in pct(A).items():
        print(f"    {k:<8} {cnt:>4}  ({p}%)")
    print("  by feature type (accuracy):")
    for typ, ctr in A_by_type.items():
        if sum(ctr.values()):
            print(f"    {typ:<12} n={sum(ctr.values()):>3}  acc={acc(ctr)}%  "
                  f"(correct {ctr['correct']}, near {ctr['near']}, "
                  f"wrong {ctr['wrong']}, miss {ctr['miss']})")

    print("\n=== Slice B (broad: all extractions, absent -> no/absent) ===")
    print(f"  n = {sum(B.values())}   accuracy = {acc(B)}%")
    for k, (cnt, p) in pct(B).items():
        print(f"    {k:<8} {cnt:>4}  ({p}%)")

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None
    print("\n=== Confidence calibration (Slice A, qa turns only) ===")
    for k in ("correct", "near", "wrong", "miss"):
        print(f"    {k:<8} mean conf = {mean(conf_of[k])}  (n={len(conf_of[k])})")

    if wrong_examples:
        print("\n=== Sample 'wrong' extractions (Slice A) ===")
        for w in wrong_examples[:10]:
            print(f"    case {w['case']:>2} {w['fid']:<14} [{w['type']}] "
                  f"parser={w['extracted']!r} vs gt={w['gt']!r}")

    result = {
        "n_cases": len(files), "n_extractions": n_extractions,
        "slice_A_clean": dict(A), "slice_A_accuracy": acc(A),
        "slice_A_by_type": {t: dict(c) for t, c in A_by_type.items()},
        "slice_B_broad": dict(B), "slice_B_accuracy": acc(B),
        "confidence_means": {k: mean(v) for k, v in conf_of.items()},
        "wrong_examples": wrong_examples,
    }
    with OUT_JSON.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
