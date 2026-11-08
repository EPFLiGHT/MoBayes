"""KB noise and coverage sensitivity (appendix C.16).

Holds the dialogue fixed and perturbs only the knowledge base. The logged
evidence of a clean-KB MoBayes run is replayed through noised copies of the
empirical DDxPlus KB and the posterior is recomputed offline, so question
selection and extraction stay frozen. No API calls.

Data: the 100 DDxPlus PatientSim dialogues (set_a + set_b). The commit
threshold tau is set once on the clean KB (in-sample argmax over a grid, or
fixed with --tau) and held fixed across all noise levels.

Two corruptions of the Dirichlet pseudo-counts:
  sigma : multiplicative log-normal noise, alpha' = alpha * exp(N(0, sigma)).
  rho   : a random fraction rho of (feature, disease) cells is set to a
          uniform distribution over the feature's values.

Usage (from this folder, no API key needed):
    PYTHONPATH=. python3 experiments/kb_noise.py --seeds 20 --tau 0.3
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from world_model.core.likelihood_model import DirichletLikelihoodModel  # noqa: E402
from world_model.core.engine import WorldModel  # noqa: E402
from world_model.knowledge.data_loader import (  # noqa: E402
    load_world_model_from_schema, load_disease_names,
)
from llm_interface.ddxplus_display import is_numeric_scale  # noqa: E402

KB_DIR = REPO / "data/kbs/ddxplus"
SETB_DIR = REPO / "data/benchmark_results/set_b/ddxplus_patientsim/mobayes/bmbe"
SETA_JSONL = REPO / "data/benchmark_results/set_a/ddxplus_patientsim/mobayes/bmbe_gpt54nano.jsonl"
OUT_JSON = REPO / "experiments/kb_noise_results.json"
OUT_PNG = REPO / "experiments/kb_noise_curves.png"

BULK_CONF = 0.8         # config.bulk_extraction_confidence_cap (opening features)
NUMERIC_SIGMA = 1.5     # config.numeric_gaussian_sigma (soft numeric update)

SIGMA_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
RHO_LEVELS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


# --------------------------------------------------------------- clean KB load

class CleanKB:
    """Immutable pieces of the clean empirical KB, shared across noise draws."""

    def __init__(self, data_dir: Path):
        wm, leaves = load_world_model_from_schema(str(data_dir), prior_mode="uniform")
        self.schema = wm.likelihood_model.schema
        self.num_diseases = wm.num_diseases
        self.base_prior = wm.base_prior.copy()
        self.taxonomy = wm.taxonomy
        self.leaf_diseases = leaves
        self.names = load_disease_names(str(data_dir))
        self.alpha0: Dict[str, Dict[str, np.ndarray]] = {
            fid: {v: np.asarray(a, dtype=float) for v, a in vals.items()}
            for fid, vals in wm.likelihood_model.alpha_table.items()
        }

    def name_of(self, leaf_idx: int) -> str:
        did = self.leaf_diseases[leaf_idx]
        return self.names.get(did, did)


# ------------------------------------------------------------ evidence traces
# A "trace" is (ground_truth, logged_pred, logged_conf, evidence) where evidence
# is an ordered list of (feature_id, value, confidence, is_bulk). Both log
# formats (set_b per-case JSON, set_a jsonl dialog_history) reduce to this.

def _trace_from_setb(case: dict) -> Tuple[str, str, float, list]:
    ev = []
    for t in case.get("turns", []):
        if t.get("kind") == "opening":
            for fid, val in (t.get("extracted_features") or {}).items():
                ev.append((fid, val, BULK_CONF, True))
        elif t.get("kind") == "qa":
            ev.append((t["feature_id"], t.get("extracted_value"),
                       float(t.get("extraction_confidence", 1.0)), False))
    fin = case.get("final", {})
    return case.get("ground_truth", ""), fin.get("predicted", ""), fin.get("confidence"), ev


def _trace_from_seta(rec: dict) -> Tuple[str, str, float, list]:
    ev = []
    pending = None
    for s in rec.get("dialog_history", []):
        content = s.get("content", "")
        if s.get("role") == "system" and "Bulk extracted:" in content:
            try:
                d = ast.literal_eval(content.split("Bulk extracted:", 1)[1].strip())
                for fid, val in d.items():
                    ev.append((fid, val, BULK_CONF, True))
            except (ValueError, SyntaxError):
                pass
        if "feature_id" in s:
            pending = s["feature_id"]
        if "extracted_value" in s and pending is not None:
            ev.append((pending, s["extracted_value"],
                       float(s.get("confidence", 1.0)), False))
            pending = None
    return rec.get("ground_truth", ""), rec.get("predicted", ""), rec.get("confidence"), ev


def load_traces() -> Dict[str, List[tuple]]:
    """Return {'set_a': [...], 'set_b': [...]} of evidence traces (n=100 total)."""
    traces = {"set_a": [], "set_b": []}
    for f in sorted(glob.glob(str(SETB_DIR / "case_*.json"))):
        with open(f) as fh:
            traces["set_b"].append(_trace_from_setb(json.load(fh)))
    if SETA_JSONL.exists():
        with open(SETA_JSONL) as fh:
            for line in fh:
                if line.strip():
                    traces["set_a"].append(_trace_from_seta(json.loads(line)))
    return traces


# ---------------------------------------------------------------- noise models

def _copy_alpha(alpha0):
    return {fid: {v: a.copy() for v, a in vals.items()} for fid, vals in alpha0.items()}


def noised_likelihood(kb: CleanKB, kind: str, level: float,
                      rng: np.random.Generator) -> DirichletLikelihoodModel:
    alpha = _copy_alpha(kb.alpha0)
    if level > 0 and kind == "sigma":
        for vals in alpha.values():
            for a in vals.values():
                a *= np.exp(rng.normal(0.0, level, size=a.shape))
                np.maximum(a, 1e-6, out=a)
    elif level > 0 and kind == "rho":
        cells = [(fid, d) for fid in alpha for d in range(kb.num_diseases)]
        k = int(round(level * len(cells)))
        if k > 0:
            for idx in rng.choice(len(cells), size=k, replace=False):
                fid, d = cells[idx]
                for v in alpha[fid]:
                    alpha[fid][v][d] = 1.0
    return DirichletLikelihoodModel(
        alpha_table=alpha, num_diseases=kb.num_diseases, schema=kb.schema)


# ------------------------------------------------------------------- replay

def replay(evidence: list, lm: DirichletLikelihoodModel, kb: CleanKB) -> Tuple[str, float]:
    wm = WorldModel(num_diseases=kb.num_diseases, likelihood_model=lm,
                    taxonomy=kb.taxonomy, base_prior=kb.base_prior,
                    leaf_diseases=kb.leaf_diseases)
    for fid, val, conf, is_bulk in evidence:
        sch = kb.schema.get(fid)
        if sch is None or val is None:
            continue
        if (not is_bulk) and is_numeric_scale(sch.values) and str(val) not in ("unknown", "absent"):
            wm.soft_update(fid, val, sigma=NUMERIC_SIGMA, confidence=conf)
        else:
            wm.update(fid, val, confidence=conf)
    p = wm.posterior()
    return kb.name_of(int(np.argmax(p))), float(np.max(p))


def _norm(s: str) -> str:
    return (s or "").lower().strip()


def replay_all(traces: list, lm: DirichletLikelihoodModel, kb: CleanKB) -> List[Tuple[bool, float]]:
    """Return per-case (top1_correct, confidence) for a given (noised) KB."""
    out = []
    for gt, _, _, ev in traces:
        top, conf = replay(ev, lm, kb)
        out.append((_norm(top) == _norm(gt), conf))
    return out


def dhs_at_tau(results: List[Tuple[bool, float]], tau: float) -> Dict[str, float]:
    n = len(results)
    committed = sum(1 for _, c in results if c >= tau)
    correct = sum(1 for hit, c in results if c >= tau and hit)
    sa = 100.0 * correct / committed if committed else 0.0
    cov = 100.0 * committed / n if n else 0.0
    dhs = (2 * sa * cov / (sa + cov)) if (sa + cov) > 0 else 0.0
    return {"SA": sa, "Cov": cov, "DHS": dhs}


def tune_tau(clean_results: List[Tuple[bool, float]], step: float = 0.01) -> float:
    """Pick tau* maximising clean DHS over the grid (paper's operating point)."""
    best_tau, best_dhs = 0.5, -1.0
    for i in range(int(0.96 / step) + 1):
        tau = round(i * step, 4)
        d = dhs_at_tau(clean_results, tau)["DHS"]
        if d > best_dhs:
            best_dhs, best_tau = d, tau
    return best_tau


# ------------------------------------------------------------------- driver

def sweep(kind, levels, traces, kb, tau, seeds):
    rows = []
    for level in levels:
        if level == 0.0:
            lm = noised_likelihood(kb, kind, 0.0, np.random.default_rng(0))
            m = dhs_at_tau(replay_all(traces, lm, kb), tau)
            rows.append({"level": level, "dhs_mean": m["DHS"], "dhs_lo": m["DHS"],
                         "dhs_hi": m["DHS"], "sa": m["SA"], "cov": m["Cov"], "n_seeds": 1})
            print(f"  {kind}={level:<5} DHS={m['DHS']:5.1f}  (SA={m['SA']:.1f} "
                  f"Cov={m['Cov']:.1f})  [clean ref, tau*={tau}]", flush=True)
            continue
        vals = []
        for s in range(seeds):
            lm = noised_likelihood(kb, kind, level, np.random.default_rng(1000 * s + 7))
            vals.append(dhs_at_tau(replay_all(traces, lm, kb), tau)["DHS"])
        arr = np.array(vals)
        rows.append({"level": level, "dhs_mean": float(arr.mean()),
                     "dhs_lo": float(np.percentile(arr, 2.5)),
                     "dhs_hi": float(np.percentile(arr, 97.5)),
                     "sa": None, "cov": None, "n_seeds": seeds})
        print(f"  {kind}={level:<5} DHS={arr.mean():5.1f}  "
              f"[{np.percentile(arr,2.5):.1f}, {np.percentile(arr,97.5):.1f}]  ({seeds} seeds)",
              flush=True)
    return rows


def fidelity_check(traces: list, kb: CleanKB) -> Dict[str, float]:
    lm = noised_likelihood(kb, "sigma", 0.0, np.random.default_rng(0))
    match, n = 0, 0
    cdiff = []
    for gt, logged_pred, logged_conf, ev in traces:
        top, conf = replay(ev, lm, kb)
        n += 1
        if _norm(top) == _norm(logged_pred):
            match += 1
        if isinstance(logged_conf, (int, float)):
            cdiff.append(abs(conf - float(logged_conf)))
    return {"n": n, "top1_match_rate": 100.0 * match / n if n else 0.0,
            "mean_abs_conf_diff": float(np.mean(cdiff)) if cdiff else float("nan")}


def maybe_plot(sigma_rows, rho_rows, clean_dhs, standalone_dhs, tau):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib not available, skipping PNG")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, rows, xlabel, title in (
        (axes[0], sigma_rows, "likelihood noise sigma", "DHS vs likelihood noise"),
        (axes[1], rho_rows, "KB dropout rho", "DHS vs KB dropout"),
    ):
        xs = [r["level"] for r in rows]
        ys = [r["dhs_mean"] for r in rows]
        ax.plot(xs, ys, "-o", color="#1f77b4", label="MoBayes (replayed)")
        ax.fill_between(xs, [r["dhs_lo"] for r in rows], [r["dhs_hi"] for r in rows],
                        color="#1f77b4", alpha=0.2)
        ax.axhline(clean_dhs, ls="--", color="green", alpha=0.7, label=f"clean KB ({clean_dhs:.0f})")
        if standalone_dhs is not None:
            ax.axhline(standalone_dhs, ls=":", color="red", alpha=0.8,
                       label=f"standalone ({standalone_dhs:.0f})")
        ax.set_xlabel(xlabel); ax.set_ylabel("DHS"); ax.set_title(title)
        ax.set_ylim(0, 100); ax.legend(fontsize=8)
    fig.suptitle(f"KB-noise sensitivity (n=100 DDxPlus PatientSim, tau*={tau})", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"[plot] saved {OUT_PNG}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--standalone-dhs", type=float, default=None)
    ap.add_argument("--tau", type=float, default=None,
                    help="fix tau instead of tuning on clean KB")
    args = ap.parse_args()

    kb = CleanKB(KB_DIR)
    tr = load_traces()
    all_traces = tr["set_a"] + tr["set_b"]
    print(f"[kb_noise] KB={KB_DIR.name} | {kb.num_diseases} diseases | "
          f"n={len(all_traces)} (set_a={len(tr['set_a'])}, set_b={len(tr['set_b'])})\n")

    for name, sub in (("set_a", tr["set_a"]), ("set_b", tr["set_b"]), ("combined", all_traces)):
        if sub:
            f = fidelity_check(sub, kb)
            print(f"[fidelity] {name:<8} sigma=0 top1 match {f['top1_match_rate']:.1f}% "
                  f"(n={f['n']}), mean |dconf| {f['mean_abs_conf_diff']:.3f}")

    clean_results = replay_all(all_traces, noised_likelihood(kb, "sigma", 0.0, np.random.default_rng(0)), kb)
    tau = args.tau if args.tau is not None else tune_tau(clean_results)
    clean_m = dhs_at_tau(clean_results, tau)
    print(f"\n[operating point] tau*={tau}  ->  clean DHS={clean_m['DHS']:.1f} "
          f"(SA={clean_m['SA']:.1f}, Cov={clean_m['Cov']:.1f})")
    print("  (compare to the paper's DDxPlus PatientSim MoBayes cell)\n")

    print("Likelihood-noise sweep:")
    sigma_rows = sweep("sigma", SIGMA_LEVELS, all_traces, kb, tau, args.seeds)
    print("\nKB-dropout sweep:")
    rho_rows = sweep("rho", RHO_LEVELS, all_traces, kb, tau, args.seeds)

    result = {
        "kb": KB_DIR.name, "n_cases": len(all_traces),
        "n_set_a": len(tr["set_a"]), "n_set_b": len(tr["set_b"]),
        "tau_star": tau, "clean_dhs": clean_m["DHS"],
        "clean_sa": clean_m["SA"], "clean_cov": clean_m["Cov"],
        "seeds": args.seeds, "sigma_sweep": sigma_rows, "rho_sweep": rho_rows,
        "standalone_dhs": args.standalone_dhs,
    }
    with OUT_JSON.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[saved] {OUT_JSON}")
    maybe_plot(sigma_rows, rho_rows, clean_m["DHS"], args.standalone_dhs, tau)


if __name__ == "__main__":
    main()
