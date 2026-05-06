#!/usr/bin/env python3
"""
Generate paper-quality plots for the MoBayes paper (NeurIPS 2026).

Outputs (saved to figures/):
  stats_vs_scale.pdf       -- model size vs accuracy bubble chart
  persona_radar.pdf         -- persona robustness radar chart

Usage:
  python3 experiments/analysis/generate_paper_plots.py

Dependencies: matplotlib, numpy (standard conda/pip environment)
"""

import json, os, math, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAPER  = os.path.join(BASE, 'figures')
RES    = os.path.join(BASE, 'data', 'benchmark_results', 'set_a')

# ── colour palette ────────────────────────────────────────────────────────────
C_MoBayes  = '#2563eb'   # blue  – MoBayes family
C_MoBayes2 = '#059669'   # green – second MoBayes
C_MoBayes3 = '#7c3aed'   # purple– third MoBayes (Llama-Scout)
C_SA1   = '#dc2626'   # red   – GPT-5.4 SA
C_SA2   = '#b45309'   # amber – Gemini 3.1 Pro SA
C_SA3   = '#6d28d9'   # violet– Llama-4-Maverick SA

GREY    = '#94a3b8'
LIGHT   = '#f1f5f9'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.18,
    'grid.linewidth': 0.5,
})


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Statistics vs Scale
# ═══════════════════════════════════════════════════════════════════════════════
def plot_stats_vs_scale():
    ddx = os.path.join(RES, "ddxplus_patientsim")

    # Model definitions
    # (file, label, color, marker, x_pos, is_bmbe)
    # x_pos: approximate relative scale (log-ish).  Nano~1, Small~2, Medium~3, Large~4
    # x positions: spread out to avoid overlap
    # MoBayes sensors (small LLMs)  |  SA (large LLMs) — separated by vertical gap
    models = [
        # MoBayes sensors
        ('bmbe_gpt54nano.jsonl',      'MoBayes+\nGPT-5.4-nano',      C_MoBayes,  'o', 1.5,  True),
        ('bmbe_geminiflashlite.jsonl','MoBayes+\nGemini FL Lite',     C_MoBayes2, 'o', 2.5,  True),
        # Standalone large models
        ('sa_gpt54.jsonl',            'SA GPT-5.4',                 C_SA1,   's', 6.0,  False),
        ('sa_gemini31pro.jsonl',      'SA Gemini 3.1 Pro',          C_SA2,   's', 8.0,  False),
    ]

    # Reposition: MoBayes left zone (1-3), SA right zone (6-8), big gap in between
    pos_map = {
        'bmbe_gpt54nano.jsonl':       1.5,
        'bmbe_geminiflashlite.jsonl': 2.5,
        'sa_gpt54.jsonl':             6.0,
        'sa_gemini31pro.jsonl':       8.0,
    }

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    # Background shading for each zone
    ax.axvspan(0.3, 4.0,  color='#eff6ff', alpha=0.45, zorder=0)  # blue — MoBayes
    ax.axvspan(5.0, 9.0,  color='#fff1f2', alpha=0.45, zorder=0)  # red  — SA
    # Gap between zones (white / neutral)
    ax.axvspan(4.0, 5.0,  color='white',   alpha=1.0,  zorder=0)

    # Group labels at top
    ax.text(2.0, 91.5, 'MoBayes Sensors', fontsize=9, color='#1e40af',
            fontweight='bold', ha='center')
    ax.text(7.0, 91.5, 'Standalone LLMs', fontsize=9, color='#7f1d1d',
            fontweight='bold', ha='center')

    # Vertical dashed separator
    ax.axvline(x=4.5, color='#94a3b8', linestyle='--', linewidth=1.0,
               alpha=0.6, zorder=1)

    for fname, label, color, marker, _, is_bmbe in models:
        xpos = pos_map[fname]
        path = os.path.join(ddx, fname)
        if not os.path.exists(path):
            continue
        recs = load(path)
        acc   = np.mean([r['top1_hit'] for r in recs]) * 100
        turns = np.mean([r.get('turns', r.get('num_questions', 10)) for r in recs])
        size  = (turns / 22) * 700 + 80

        ax.scatter(xpos, acc, s=size, color=color, alpha=0.88,
                   marker=marker, edgecolors='white', linewidths=1.8,
                   zorder=5 if is_bmbe else 3)

        if is_bmbe:
            ax.annotate(label, (xpos, acc + 5.5), fontsize=8, ha='center',
                        color=color, fontweight='bold', multialignment='center')
        else:
            # stagger SA labels to avoid overlap: GPT-5.4 above, Gemini below
            dy = 5.5 if 'GPT' in label else -8.0
            ax.annotate(label, (xpos, acc + dy), fontsize=8, ha='center',
                        color=color, multialignment='center')

    # Annotation box — in the gap between the two zones
    ax.text(4.5, 75, 'Small sensor\n+ Bayesian engine\n> all frontier SA',
            fontsize=8, color='#1e40af', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', fc='#eff6ff', ec='#bfdbfe', lw=1.2))

    # X axis: no ticks, just group labels via axvspan
    ax.set_xticks([])
    ax.set_xlabel('', fontsize=10)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=10)
    ax.set_ylim(40, 94)
    ax.set_xlim(0.3, 9.0)

    # Legend
    bmbe_patch = mpatches.Patch(color=C_MoBayes,  label='MoBayes + LLM sensor (ours)')
    sa_patch   = mpatches.Patch(color=C_SA1,   label='Standalone LLM doctor')
    size_note  = mpatches.Patch(color='white', label='Bubble size ∝ avg. dialogue turns',
                                 edgecolor=GREY)
    ax.legend(handles=[bmbe_patch, sa_patch, size_note],
              fontsize=8, loc='lower right', framealpha=0.9)

    ax.set_title('Statistics, Not Scale: MoBayes vs Standalone LLMs (DDXPlus)', fontsize=10, pad=10)
    plt.tight_layout()
    out = os.path.join(PAPER, 'stats_vs_scale.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.replace('.pdf','.png'), dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Persona Robustness Radar
# ═══════════════════════════════════════════════════════════════════════════════
def plot_persona_radar():
    abl = os.path.join(RES, "ddxplus_patientsim", 'ablations')

    personas = ['dazed', 'distrustful', 'overanxious', 'verbose_lowrecall']
    labels   = ['Dazed', 'Distrustful', 'Overanxious', 'Verbose /\nLow-Recall']

    bmbe_vals, sa_vals = [], []
    for p in personas:
        bf = os.path.join(abl, f'persona_{p}_bmbe.jsonl')
        sf = os.path.join(abl, f'persona_{p}_sa.jsonl')
        if os.path.exists(bf):
            recs = load(bf)
            bmbe_vals.append(np.mean([r['top1_hit'] for r in recs]))
        else:
            bmbe_vals.append(0)
        if os.path.exists(sf):
            recs = load(sf)
            sa_vals.append(np.mean([r['top1_hit'] for r in recs]))
        else:
            sa_vals.append(0)

    N = len(personas)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    bmbe_plot = bmbe_vals + bmbe_vals[:1]
    sa_plot   = sa_vals   + sa_vals[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))

    # Grid rings
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7.5, color=GREY)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9.5, fontweight='bold')

    # MoBayes polygon
    ax.plot(angles, bmbe_plot, color=C_MoBayes, linewidth=2.5, linestyle='solid', zorder=4)
    ax.fill(angles, bmbe_plot, color=C_MoBayes, alpha=0.15)
    for ang, val in zip(angles[:-1], bmbe_vals):
        ax.annotate(f'{val*100:.0f}%', xy=(ang, val),
                    fontsize=9, fontweight='bold', color=C_MoBayes, ha='center',
                    xytext=(ang, val + 0.07),)

    # SA polygon
    ax.plot(angles, sa_plot, color=C_SA1, linewidth=2.5, linestyle='dashed', zorder=3)
    ax.fill(angles, sa_plot, color=C_SA1, alpha=0.08)
    for ang, val in zip(angles[:-1], sa_vals):
        ax.annotate(f'{val*100:.0f}%', xy=(ang, val),
                    fontsize=9, fontweight='bold', color=C_SA1, ha='center',
                    xytext=(ang, val - 0.13),)

    ax.grid(color=GREY, alpha=0.3)
    ax.set_facecolor('#fafafa')

    # Legend
    bmbe_line = mpatches.Patch(color=C_MoBayes, alpha=0.7, label='MoBayes + GPT-5.4-nano')
    sa_line   = mpatches.Patch(color=C_SA1,  alpha=0.7, label='SA GPT-5.4')
    ax.legend(handles=[bmbe_line, sa_line], loc='upper right',
              bbox_to_anchor=(1.32, 1.12), fontsize=9, framealpha=0.9)

    ax.set_title('Persona Robustness — Top-1 Accuracy\nacross Adversarial Patient Styles',
                 fontsize=10, pad=18)

    plt.tight_layout()
    out = os.path.join(PAPER, 'persona_radar.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.replace('.pdf','.png'), dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(PAPER, exist_ok=True)
    print('Generating plots...')
    plot_stats_vs_scale()
    plot_persona_radar()
    print('Done.')
