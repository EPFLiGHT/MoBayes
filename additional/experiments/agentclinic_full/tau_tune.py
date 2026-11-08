"""Abstention threshold on the logged AgentClinic-MedQA dialogues (appendix C.19).

Replays each case's logged observations through the engine, in order, to get
the final top-1 posterior and whether the top-1/top-3 diagnosis is correct.
Question selection is taken from the log and is not rerun. A case commits iff
its confidence >= tau, otherwise it abstains.

  SA  = correct_committed / committed   (selective accuracy)
  Cov = committed / n                   (coverage)
  DHS = 2*SA*Cov/(SA+Cov)

Reported twice: with opening-statement (intake) observations capped at
confidence 0.8, and with the cap removed (1.0). Bootstrap CIs re-tune tau* on
every resample.

Usage (from this folder):
    PYTHONPATH=. python3 experiments/agentclinic_full/tau_tune.py
"""
from __future__ import annotations
import json, glob, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
KB = REPO / "data/kbs/agentclinic_full_cat"
LOGGED = REPO / "data/benchmark_results/agentclinic_full_cat_bmbe"

sys.path.insert(0, str(REPO))
from world_model.knowledge.data_loader import load_world_model_from_schema, load_disease_names


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower().replace("'", "").replace("’", "")).strip()


def collect(wm, idx2norm, sv, fs, intake_conf):
    """Return list of (confidence, top1_correct, top3_correct)."""
    rows = []
    for f in fs:
        c = json.load(open(f)); gt = norm(c["ground_truth"])
        wm.reset_belief()
        op = c["turns"][0]; ef = op.get("extracted_features") or {}
        if isinstance(ef, dict):
            for k, v in ef.items():
                if k in sv: wm.update(k, v, confidence=intake_conf)
        for t in c["turns"][1:]:
            fid = t.get("feature_id"); v = t.get("extracted_value")
            if fid in sv and v is not None:
                wm.update(fid, v, confidence=t.get("extraction_confidence", 1.0))
        post = wm.posterior(); order = np.argsort(post)[::-1]
        conf = float(post[order[0]])
        top1 = idx2norm[order[0]] == gt
        top3 = gt in [idx2norm[i] for i in order[:3]]
        rows.append((conf, top1, top3))
    return rows


def sweep(rows):
    n = len(rows)
    confs = sorted(set(r[0] for r in rows))
    best = None
    curve = []
    for tau in [0.0] + confs:
        committed = [r for r in rows if r[0] >= tau]
        c = len(committed)
        cov = 100.0 * c / n
        sa = 100.0 * sum(r[1] for r in committed) / c if c else 0.0
        dhs = (2 * sa * cov / (sa + cov)) if (sa + cov) > 0 else 0.0
        curve.append((tau, sa, cov, dhs))
        if best is None or dhs > best[3]:
            best = (tau, sa, cov, dhs)
    return best, curve


def _metrics_at_tau(rows, tau):
    n = len(rows)
    committed = [r for r in rows if r[0] >= tau]
    c = len(committed)
    cov = 100.0 * c / n
    sa = 100.0 * sum(r[1] for r in committed) / c if c else 0.0
    dhs = (2 * sa * cov / (sa + cov)) if (sa + cov) > 0 else 0.0
    return sa, cov, dhs


def bootstrap(rows, B=10000, seed=42):
    """Percentile bootstrap CIs. tau* is re-tuned on each resample (honest CI
    for the tuned DHS). Returns dict metric -> (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(rows)
    arr = np.array([(r[0], r[1], r[2]) for r in rows], dtype=float)
    # point estimates on full sample
    t1p = 100.0 * arr[:, 1].mean()
    t3p = 100.0 * arr[:, 2].mean()
    best, _ = sweep(rows)
    tau_star, sa_p, cov_p, dhs_p = best
    T1 = []; T3 = []; SA = []; COV = []; DHS = []; TAU = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        res = [rows[i] for i in idx]
        T1.append(100.0 * arr[idx, 1].mean())
        T3.append(100.0 * arr[idx, 2].mean())
        b, _ = sweep(res)
        TAU.append(b[0]); SA.append(b[1]); COV.append(b[2]); DHS.append(b[3])

    def ci(pt, samples):
        lo, hi = np.percentile(samples, [2.5, 97.5])
        return (pt, float(lo), float(hi))
    return {
        "Top-1": ci(t1p, T1), "Top-3": ci(t3p, T3),
        "tau*": (tau_star, float(np.percentile(TAU, 2.5)), float(np.percentile(TAU, 97.5))),
        "SA@tau*": ci(sa_p, SA), "Cov@tau*": ci(cov_p, COV), "DHS@tau*": ci(dhs_p, DHS),
    }


def main():
    wm, dis = load_world_model_from_schema(str(KB), prior_mode="uniform")
    id2name = load_disease_names(str(KB))
    idx2norm = {i: norm(id2name.get(dis[i], dis[i])) for i in range(len(dis))}
    sv = {fid: fsx.values for fid, fsx in wm.likelihood_model.schema.items()}
    fs = sorted(glob.glob(str(LOGGED / "case_*.json")))
    n = len(fs)

    for label, ic in [("intake cap 0.8", 0.8), ("intake cap removed (1.0)", 1.0)]:
        rows = collect(wm, idx2norm, sv, fs, ic)
        t1 = 100.0 * sum(r[1] for r in rows) / n
        t3 = 100.0 * sum(r[2] for r in rows) / n
        best, curve = sweep(rows)
        tau, sa, cov, dhs = best
        ci = bootstrap(rows, B=10000)
        print(f"=== {label}  (n={n}) ===")
        print(f"  full-commit: Top-1={t1:.1f}%  Top-3={t3:.1f}%  Cov=100%  DHS(=SA)={t1:.1f}")
        print(f"  tau*={tau:.3f}  ->  SA={sa:.1f}%  Cov={cov:.1f}%  DHS={dhs:.1f}")
        print(f"  --- bootstrap 95% CI (B=10000, tau* re-tuned per resample) ---")
        for k in ("Top-1", "Top-3", "SA@tau*", "Cov@tau*", "DHS@tau*"):
            pt, lo, hi = ci[k]
            print(f"     {k:<10} {pt:5.1f}  [{lo:5.1f}, {hi:5.1f}]")
        print(f"     tau*        {ci['tau*'][0]:.3f}  [{ci['tau*'][1]:.3f}, {ci['tau*'][2]:.3f}]")
        print(f"  curve (tau, SA, Cov, DHS):")
        for tau_, sa_, cov_, dhs_ in curve[:: max(1, len(curve)//8)]:
            print(f"     tau={tau_:.3f}  SA={sa_:5.1f}  Cov={cov_:5.1f}  DHS={dhs_:5.1f}")
        print()


if __name__ == "__main__":
    main()
