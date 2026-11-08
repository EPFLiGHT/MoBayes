"""
Accuracy-Coverage curve plot for DDXPlus.

DDXPlus (single-disease): Selective Accuracy vs Coverage (Phase-A confidence sweep).
Compares MoBayes (threshold-swept curve) against SA (single point per system).
Used in Appendix (Figure: acc_coverage_ddxplus).

Usage:
    PYTHONPATH=. python3 experiments/analysis/plot_acc_coverage.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
CANONICAL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'benchmark_results', 'set_a')

normalize = lambda s: s.lower().strip().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")


def load_records(path):
    return [json.loads(l) for l in open(path) if l.strip()]


# ── DDXPlus: Selective Accuracy via Phase-A confidence ──

def ddxplus_rc_curve(records, tau_range=np.arange(0.01, 1.0, 0.005)):
    """Sweep Phase-A posterior confidence → (coverage, selective_accuracy)."""
    points = []
    for tau in tau_range:
        sel = [r for r in records if r.get('confidence', 0) >= tau]
        if not sel:
            continue
        acc = sum(r['top1_hit'] for r in sel) / len(sel)
        cov = len(sel) / len(records)
        points.append((cov, acc))
    # τ=0: all patients
    points.append((1.0, sum(r['top1_hit'] for r in records) / len(records)))
    points.sort(key=lambda x: x[0])
    return points


def sa_single_point_accuracy(records):
    """SA (coverage, selective_accuracy) — single point from abstain/diagnose."""
    non_abst = [r for r in records if r.get('stop_reason') != 'ABSTAINED']
    if not non_abst:
        return (0, 0)
    acc = sum(r['top1_hit'] for r in non_abst) / len(non_abst)
    cov = len(non_abst) / len(records)
    return (cov, acc)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    base = CANONICAL_DIR

    ddx_bmbe = [
        ("MoBayes + GPT-5.4-nano", f"{base}/ddxplus_patientsim/mobayes/bmbe_gpt54nano.jsonl", "tab:blue", "-"),
        ("MoBayes + Gemini FL", f"{base}/ddxplus_patientsim/mobayes/bmbe_geminiflashlite.jsonl", "tab:green", "--"),
        ("MoBayes + Llama-4-Scout", f"{base}/ddxplus_patientsim/mobayes/bmbe_llama4scout.jsonl", "tab:orange", "-."),
    ]
    ddx_sa = [
        ("SA GPT-5.4", f"{base}/ddxplus_patientsim/standalone/sa_gpt54.jsonl", "tab:red"),
        ("SA Gemini 3.1 Pro", f"{base}/ddxplus_patientsim/standalone/sa_gemini31pro.jsonl", "tab:purple"),
        ("SA GPT-OSS-120B", f"{base}/ddxplus_patientsim/standalone/sa_gptoss120b.jsonl", "tab:brown"),
        ("SA Llama-4-Maverick", f"{base}/ddxplus_patientsim/standalone/sa_llama4maverick.jsonl", "tab:gray"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for name, path, color, ls in ddx_bmbe:
        records = load_records(path)
        points = ddxplus_rc_curve(records)
        covs, accs = zip(*points)
        ax.plot(covs, [a * 100 for a in accs], color=color, linestyle=ls,
                linewidth=2, label=name)
    for name, path, color in ddx_sa:
        records = load_records(path)
        cov, acc = sa_single_point_accuracy(records)
        ax.scatter([cov], [acc * 100], color=color, marker='X', s=120,
                   zorder=5, label=name)
    ax.set_xlabel("Coverage", fontsize=12)
    ax.set_ylabel("Selective Accuracy (%)", fontsize=12)
    ax.set_title("DDXPlus: Selective Accuracy vs Coverage", fontsize=13)
    ax.legend(fontsize=7, loc='lower left')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(30, 105)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/acc_coverage_ddxplus.pdf", bbox_inches='tight')
    plt.savefig(f"{FIGURES_DIR}/acc_coverage_ddxplus.png", dpi=150, bbox_inches='tight')
    print("Saved DDXPlus plot")
    plt.close()


if __name__ == "__main__":
    main()
