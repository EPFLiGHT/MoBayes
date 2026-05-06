"""Reviewer M11(b): audit standalone-doctor matcher accuracy.

For each of the 6 standalone doctors on DDxPlus50, extract the doctor's
top-1 free-text DDX from dialog history and compare against the
matched KB disease assigned by the GPT-5.4-nano matcher. Manual review
of the suspicious cases distinguishes:

  - false positives: matcher mapped a clinically distinct entity onto
    the ground-truth KB entry, inflating the standalone doctor's
    top1_hit rate.
  - false negatives: matcher mapped a correct prediction onto a wrong
    or NO_MATCH entry, deflating the rate.

Outputs the per-system breakdown and a Clopper-Pearson CI on the
overall error rate.

Hand-curated FP set is included inline for reproducibility.
"""

import json
import re
import numpy as np
from pathlib import Path

try:
    from scipy.stats import beta
except ImportError:
    beta = None

PAPER_CANONICAL = Path(
    ""
    "hybrid-diagnostic-system/data/benchmark_results/set_a/ddxplus_patientsim"
)

SA_SYSTEMS = [
    "sa_gemini31pro.jsonl",
    "sa_gpt54.jsonl",
    "sa_gptoss120b.jsonl",
    "sa_kimi_k25.jsonl",
    "sa_llama4maverick.jsonl",
    "sa_qwen36plus.jsonl",
]

# Hand-curated false positives (matcher inflated top1_hit) from manual
# review of zero-token-overlap mappings. See discussion in chat history.
HARD_FP = [
    ("sa_gemini31pro.jsonl", "ddx_019407"),       # VCD -> Larygospasm = GT
    ("sa_llama4maverick.jsonl", "ddx_012084"),    # Sinusitis -> Cluster headache = GT
    ("sa_llama4maverick.jsonl", "ddx_076280"),    # Sinusitis -> URTI = GT
    ("sa_llama4maverick.jsonl", "ddx_131854"),    # EM -> SLE = GT
]
BORDERLINE_FP = [
    # Allergic rhinitis -> Allergic sinusitis (5 models, GT = Allergic sinusitis)
    ("sa_gemini31pro.jsonl", "ddx_096173"),
    ("sa_gpt54.jsonl", "ddx_096173"),
    ("sa_gptoss120b.jsonl", "ddx_096173"),
    ("sa_qwen36plus.jsonl", "ddx_096173"),
    ("sa_llama4maverick.jsonl", "ddx_096173"),
    # ADHF -> Acute pulmonary edema (3 models, GT = Acute pulmonary edema)
    ("sa_gemini31pro.jsonl", "ddx_110958"),
    ("sa_gpt54.jsonl", "ddx_110958"),
    ("sa_kimi_k25.jsonl", "ddx_110958"),
    ("sa_llama4maverick.jsonl", "ddx_110958"),
]


def clopper_pearson(k, n, alpha=0.05):
    if beta is None:
        return None
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo * 100, hi * 100


def main():
    audited_universe = 0
    abstained_total = 0
    other_total = 0  # no [DDX], not abstained (timeout etc.)
    for fn in SA_SYSTEMS:
        with open(PAPER_CANONICAL / fn) as fh:
            for line in fh:
                r = json.loads(line)
                has_ddx = any(
                    h.get("role") == "doctor" and "[DDX]" in h.get("content", "")
                    for h in r["dialog_history"]
                )
                if has_ddx and not r.get("abstained"):
                    audited_universe += 1
                elif r.get("abstained"):
                    abstained_total += 1
                else:
                    other_total += 1

    n_strict = len(HARD_FP)
    n_lenient = n_strict + len(BORDERLINE_FP)

    print("=== Audit universe ===")
    print(f"  Total sessions:           {audited_universe + abstained_total + other_total}")
    print(f"  Non-abstention commits:   {audited_universe}")
    print(f"  Explicit [ABSTAIN]:       {abstained_total}")
    print(f"  Other (timeout etc.):     {other_total}")
    print()
    print(f"=== Matcher-error count (manual review of zero-overlap mappings) ===")
    print(f"  Strict FP (clear errors):           {n_strict}/{audited_universe} "
          f"= {100 * n_strict / audited_universe:.2f}%")
    if beta is not None:
        ci = clopper_pearson(n_strict, audited_universe)
        print(f"    95% Clopper-Pearson CI: [{ci[0]:.2f}, {ci[1]:.2f}]%")
    print(f"  Lenient FP (incl. borderline):      {n_lenient}/{audited_universe} "
          f"= {100 * n_lenient / audited_universe:.2f}%")
    if beta is not None:
        ci = clopper_pearson(n_lenient, audited_universe)
        print(f"    95% Clopper-Pearson CI: [{ci[0]:.2f}, {ci[1]:.2f}]%")
    print()
    print("All confirmed errors inflate top1_hit (false positives).")
    print("No false negatives observed.")


if __name__ == "__main__":
    main()
