#!/usr/bin/env python3
"""
Disease Scaling Analysis — How does performance change with KB size?

Creates DDXPlus KB subsets with K *randomly sampled* diseases (multiple seeds)
to isolate the effect of search space size from disease difficulty.

Usage:
    # Step 1: Generate KB subsets and profiles
    PYTHONPATH=. python3 experiments/analysis/scaling_analysis.py --generate

    # Step 2: Run experiments
    bash experiments/ddxplus/07_scaling.sh

    # Step 3: Plot results
    PYTHONPATH=. python3 experiments/analysis/scaling_analysis.py --plot
"""

import argparse
import csv
import json
import os
import random
import sys
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

DDXPLUS_DIR = os.path.join(BASE_DIR, "data", "ddxplus_kb")
SCALING_DIR = os.path.join(BASE_DIR, "data", "scaling_random")
PROFILES_DIR = os.path.join(BASE_DIR, "data", "cases")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "benchmark_results", "scaling_v2")

K_VALUES = [10, 20, 30, 40]
SEEDS = [42, 123, 777]
N_PATIENTS_PER_DISEASE = 1  # N = K × this


def load_all_diseases():
    """Load all leaf disease IDs (excluding d_other and category rows)."""
    diseases = []
    with open(os.path.join(DDXPLUS_DIR, "priors.csv")) as f:
        for r in csv.DictReader(f):
            d = r["disease_id"]
            p = float(r["prior"])
            if p > 0 and d != "d_other":
                diseases.append(d)
    return diseases


def create_kb_subset(k, seed, disease_ids):
    """Create a KB subset with K randomly selected diseases."""
    tag = f"k{k}_s{seed}"
    out_dir = os.path.join(SCALING_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    disease_set = set(disease_ids)

    # diseases.csv — K diseases + parent categories
    parents = set()
    leaf_rows = []
    with open(os.path.join(DDXPLUS_DIR, "diseases.csv")) as f:
        for r in csv.DictReader(f):
            if r["disease_id"] in disease_set:
                leaf_rows.append(r)
                if r.get("parent_id"):
                    parents.add(r["parent_id"])

    parent_rows = []
    with open(os.path.join(DDXPLUS_DIR, "diseases.csv")) as f:
        for r in csv.DictReader(f):
            if r["disease_id"] in parents and r["disease_id"] not in disease_set:
                parent_rows.append(r)

    with open(os.path.join(out_dir, "diseases.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["disease_id", "name", "parent_id", "icd10"])
        w.writeheader()
        w.writerows(parent_rows + leaf_rows)

    # features.csv — copy as-is
    shutil.copy(os.path.join(DDXPLUS_DIR, "features.csv"), out_dir)

    # likelihood_counts.csv — filter to K diseases
    lc_rows = []
    with open(os.path.join(DDXPLUS_DIR, "likelihood_counts.csv")) as f:
        for r in csv.DictReader(f):
            if r["disease_id"] in disease_set:
                lc_rows.append(r)

    with open(os.path.join(out_dir, "likelihood_counts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["disease_id", "feature_id", "value", "count"])
        w.writeheader()
        w.writerows(lc_rows)

    # priors.csv — filter and renormalize
    priors = []
    with open(os.path.join(DDXPLUS_DIR, "priors.csv")) as f:
        for r in csv.DictReader(f):
            if r["disease_id"] in disease_set:
                priors.append((r["disease_id"], float(r["prior"])))

    total = sum(p for _, p in priors)
    with open(os.path.join(out_dir, "priors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["disease_id", "prior"])
        w.writeheader()
        for did, p in priors:
            w.writerow({"disease_id": did, "prior": round(p / total, 6)})

    # Copy auxiliary files if they exist
    for aux in ("demographic_rules.csv", "actions.csv", "action_rules.csv",
                "prevention_map.csv", "disease_action_map.csv"):
        src = os.path.join(DDXPLUS_DIR, aux)
        if os.path.exists(src):
            shutil.copy(src, out_dir)

    return out_dir


def create_profiles(k, seed, disease_ids, kb_dir):
    """Generate benchmark profiles for K-disease subset. N = K × N_PATIENTS_PER_DISEASE."""
    tag = f"scaling_random_k{k}_s{seed}"
    n_target = k * N_PATIENTS_PER_DISEASE

    # Load all patients, filter to K diseases
    patients = []
    with open(os.path.join(DDXPLUS_DIR, "patients.jsonl")) as f:
        for line in f:
            p = json.loads(line)
            gt = (p.get("_ddxplus_meta", {}).get("ground_truth_ids") or [None])[0]
            if gt in disease_ids and gt != "d_other":
                patients.append(p)

    # Stratified sample: N_PATIENTS_PER_DISEASE per disease
    rng = random.Random(seed)

    by_disease = {}
    for p in patients:
        gt = p["_ddxplus_meta"]["ground_truth_ids"][0]
        by_disease.setdefault(gt, []).append(p)

    for d in by_disease:
        rng.shuffle(by_disease[d])

    selected = []
    selected_ids = set()

    # Pick N_PATIENTS_PER_DISEASE from each disease
    for d in sorted(by_disease.keys()):
        for p in by_disease[d][:N_PATIENTS_PER_DISEASE]:
            pid = p.get("hadm_id", "unknown")
            if pid not in selected_ids:
                selected.append(p)
                selected_ids.add(pid)

    # Wrap with plain persona
    from scripts.data_prep.personas import PERSONAS
    from scripts.data_prep.generate_profiles import wrap_ddxplus_profile

    profiles = [wrap_ddxplus_profile(p, PERSONAS["plain"]) for p in selected]

    out_dir = os.path.join(PROFILES_DIR, tag, "plain")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "profiles.json"), "w") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

    print(f"  K={k} seed={seed}: {len(profiles)} profiles, "
          f"{len(by_disease)} diseases with patients → {out_dir}")
    return os.path.join(out_dir, "profiles.json")


def generate():
    """Generate random KB subsets and profiles for all K values × seeds."""
    all_diseases = load_all_diseases()
    print(f"DDXPlus: {len(all_diseases)} diseases total")
    print(f"K values: {K_VALUES}, seeds: {SEEDS}")
    print(f"Patients per disease: {N_PATIENTS_PER_DISEASE} (N = K × {N_PATIENTS_PER_DISEASE})\n")

    manifest = []

    for k in K_VALUES:
        for seed in SEEDS:
            rng = random.Random(seed)
            subset = rng.sample(all_diseases, k)
            disease_set = set(subset)

            kb_dir = create_kb_subset(k, seed, subset)
            profiles_path = create_profiles(k, seed, disease_set, kb_dir)

            manifest.append({
                "k": k,
                "seed": seed,
                "diseases": sorted(subset),
                "kb_dir": kb_dir,
                "profiles": profiles_path,
            })

    # K=49 — full set, single run
    kb_49 = DDXPLUS_DIR
    tag_49 = "scaling_random_k49_s0"
    profiles_49_dir = os.path.join(PROFILES_DIR, tag_49, "plain")
    if not os.path.exists(profiles_49_dir):
        # Generate profiles for K=49
        create_profiles(49, 0, set(all_diseases), kb_49)
    manifest.append({
        "k": 49,
        "seed": 0,
        "diseases": sorted(all_diseases),
        "kb_dir": kb_49,
        "profiles": os.path.join(profiles_49_dir, "profiles.json"),
    })

    # Save manifest
    os.makedirs(SCALING_DIR, exist_ok=True)
    with open(os.path.join(SCALING_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {os.path.join(SCALING_DIR, 'manifest.json')}")
    print(f"Total runs: {len(manifest)} subsets × 2 methods = {len(manifest)*2}")
    print("\nNow run: bash experiments/ddxplus/07_scaling.sh")


def load_results():
    """Load all scaling results from JSONL files."""
    results = {}
    if not os.path.exists(RESULTS_DIR):
        return results

    for entry in os.listdir(RESULTS_DIR):
        entry_dir = os.path.join(RESULTS_DIR, entry)
        if not os.path.isdir(entry_dir):
            continue
        # Parse entry name: bmbe_k10_s42 or sa_k10_s42
        parts = entry.split("_")
        if len(parts) < 3:
            continue
        mode = parts[0]
        k = int(parts[1][1:])
        seed = int(parts[2][1:])

        jsonl_files = [f for f in os.listdir(entry_dir) if f.endswith(".jsonl")]
        if not jsonl_files:
            continue
        latest = max(jsonl_files, key=lambda f: os.path.getmtime(os.path.join(entry_dir, f)))
        records = [json.loads(l) for l in open(os.path.join(entry_dir, latest))]
        if not records:
            continue

        top1 = sum(r["top1_hit"] for r in records) / len(records) * 100
        top3 = sum(r["top3_hit"] for r in records) / len(records) * 100
        n = len(records)

        results.setdefault(mode, {}).setdefault(k, []).append({
            "seed": seed, "top1": top1, "top3": top3, "n": n
        })

    return results


def plot():
    """Plot scaling results with error bars."""
    import numpy as np
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    results = load_results()
    if not results:
        print(f"No results found at {RESULTS_DIR}")
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))

    styles = {
        "bmbe": {"color": "#2166ac", "marker": "o", "ls": "-", "label": "MoBayes + GPT-5.4-nano"},
        "sa":   {"color": "#b2182b", "marker": "s", "ls": "--", "label": "SA GPT-5.4-nano"},
    }

    for mode in ["bmbe", "sa"]:
        if mode not in results:
            continue
        ks = sorted(results[mode].keys())
        means = []
        stds = []
        for k in ks:
            accs = [r["top1"] for r in results[mode][k]]
            means.append(np.mean(accs))
            stds.append(np.std(accs) if len(accs) > 1 else 0)

        s = styles[mode]
        ax.errorbar(ks, means, yerr=stds, fmt=f"{s['marker']}{s['ls']}",
                    color=s["color"], linewidth=2, capsize=4, capthick=1.5,
                    label=s["label"], markersize=7)

    ax.set_xlabel("Number of diseases (K)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_xticks(sorted(set(k for m in results.values() for k in m.keys())))
    ax.legend(frameon=False)
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = os.path.join(BASE_DIR, "figures", "scaling_random.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

    # Print table
    print(f"\n{'K':>4} | {'MoBayes mean±std':>15} | {'SA mean±std':>15} | {'seeds':>5}")
    print("-" * 50)
    all_ks = sorted(set(k for m in results.values() for k in m.keys()))
    for k in all_ks:
        bmbe = results.get("bmbe", {}).get(k, [])
        sa = results.get("sa", {}).get(k, [])
        b_accs = [r["top1"] for r in bmbe]
        s_accs = [r["top1"] for r in sa]
        b_str = f"{np.mean(b_accs):.1f}±{np.std(b_accs):.1f}%" if b_accs else "---"
        s_str = f"{np.mean(s_accs):.1f}±{np.std(s_accs):.1f}%" if s_accs else "---"
        n_seeds = max(len(b_accs), len(s_accs))
        print(f"{k:>4} | {b_str:>15} | {s_str:>15} | {n_seeds:>5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate random KB subsets and profiles")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    args = parser.parse_args()

    if args.generate:
        generate()
    elif args.plot:
        plot()
    else:
        parser.print_help()
