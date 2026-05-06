"""Sample 50 cases from the DDxPlus validate split (StreamBench) → data/streambench_50/selected_cases.jsonl."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data/streambench_50"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50,
                    help="Number of cases to sample (default 50)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (must match MEDDxAgent's seed=42 to stay "
                         "in the paper-faithful 500-sample fewshot pool)")
    ap.add_argument("--total-pool-size", type=int, default=500,
                    help="Outer pool size (MEDDxAgent paper-faithful: 500). "
                         "We take args.n cases from this pool's first slice.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] HF dataset: appier-ai-research/StreamBench (ddxplus_kb, validate)")
    from datasets import load_dataset
    dataset = load_dataset("appier-ai-research/StreamBench", "ddxplus_kb")["validate"]
    print(f"[load] {len(dataset)} cases in validate split")

    # Replicate MEDDxAgent's seed-42 random.sample(range(len), 500)
    rng = random.Random(args.seed)
    pool_indices = rng.sample(range(len(dataset)), args.total_pool_size)
    chosen_indices = pool_indices[: args.n]
    print(f"[sample] seed={args.seed}, pool={args.total_pool_size}, "
          f"taking first {args.n} indices from the pool")

    out_path = OUT_DIR / "selected_cases.jsonl"
    with out_path.open("w") as f:
        for i, idx in enumerate(chosen_indices):
            row = dict(dataset[int(idx)])
            row["patient_id"] = f"sb_{i:03d}"
            row["_validate_index"] = int(idx)
            row["_seed"] = args.seed
            f.write(json.dumps(row) + "\n")

    print(f"[done] wrote {len(chosen_indices)} cases → {out_path}")
    # Quick stats
    diagnoses = [json.loads(l)["PATHOLOGY"] for l in open(out_path)]
    from collections import Counter
    cnt = Counter(diagnoses)
    print(f"  unique diagnoses in sample: {len(cnt)}")
    print(f"  top 5: {cnt.most_common(5)}")


if __name__ == "__main__":
    main()
