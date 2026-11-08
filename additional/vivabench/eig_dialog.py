"""VivaBench feasibility study (appendix C.20), no API calls.

Runs an EIG dialogue on a VivaBench KB. Patient answers come from each case's
cached feature vector (vec_cache.json), not from a live examiner. Each turn
picks the highest-EIG unasked feature, looks up its answer in the cache
(present/absent), and applies a weighted soft-evidence update.

Usage (from this folder):
    python3 eig_dialog.py --ci                      # bootstrap CIs, both KBs and modes
    python3 eig_dialog.py <uid> --kb kb_grounded    # single case trace
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from engine import load_kb, eig_over

HERE = Path(__file__).resolve().parent


def _ent(p):
    p = p[p > 1e-20]
    return float(-np.sum(p * np.log2(p)))


def focused_eig(wm, cands, topk=8, w_absent=0.35):
    """EIG restricted to the current top-K leading diseases, absent outcomes
    down-weighted in the simulation."""
    post = wm.posterior()
    idx = np.argsort(post)[::-1][:topk]
    b = post[idx]; s = b.sum(); b = b / s if s > 0 else np.ones(len(idx)) / len(idx)
    cur = _ent(b)
    best, best_e = None, -np.inf
    for fid in cands:
        like = wm.likelihood.get(fid, {})
        exp_h = 0.0
        for v, Lfull in like.items():
            Lv = Lfull[idx]
            pv = float(np.sum(Lv * b))
            if pv <= 1e-9:
                continue
            wv = (w_absent if str(v) == "absent" else 1.0)
            nb = b * np.power(np.clip(Lv, 1e-20, None), wv)
            z = nb.sum()
            exp_h += pv * _ent(nb / z if z > 0 else nb)
        e = cur - exp_h
        if e > best_e:
            best, best_e = fid, e
    return best, best_e


def answer_of(vec, fid):
    """Patient oracle: cache value if recorded, else absent (feature not present)."""
    if fid in vec:
        return str(vec[fid][0]), float(vec[fid][1])
    return "absent", 1.0


def run(uid, kb="kb50", budget=15, w_absent=0.35, tau=0.85, focused=False, topk=8):
    vecs = json.loads((HERE / "vec_cache.json").read_text())
    obj = json.loads((HERE / "diseases50.json").read_text())
    target = obj["case_to_disease"].get(uid)
    if uid not in vecs:
        print(f"no cached vector for {uid}"); return
    vec = vecs[uid]

    wm = load_kb(str(HERE / kb))
    import pandas as pd
    dd = pd.read_csv(HERE / kb / "diseases.csv")
    name_of = dict(zip(dd.disease_id, dd.name))
    fname = {f: wm.schema[f]["name"] for f in wm.schema}

    print("=" * 92)
    print(f"CASE {uid}   KB={kb}   budget={budget}   w_absent={w_absent}   tau={tau}   "
          f"policy={'focused-EIG(top%d)' % topk if focused else 'global-EIG'}")
    print(f"TARGET: {target}  ({name_of.get(target,'?')})")
    print(f"present findings in case: {sum(1 for f in vec if str(vec[f][0])=='present')}"
          f"   |   diseases in KB: {wm.num_diseases}")
    print("=" * 92)

    def top(k=1):
        p = wm.posterior(); order = np.argsort(p)[::-1][:k]
        return [(wm.leaves[i], float(p[i])) for i in order]

    t0 = top(1)[0]
    print(f"[ 0] prior            H={wm.entropy():5.2f}  lead: {name_of.get(t0[0],t0[0])[:34]:34s} p={t0[1]:.3f}")

    for step in range(1, budget + 1):
        cands = wm.unasked_features()
        if focused:
            fid, eig = focused_eig(wm, cands, topk=topk, w_absent=w_absent)
        else:
            fid, eig = eig_over(wm, cands)
        if fid is None or eig < 1e-3:
            print(f"[{step:2d}] EIG exhausted (<1e-3), stop.")
            break
        val, conf = answer_of(vec, fid)
        w = w_absent if val == "absent" else 1.0
        wm.update(fid, val, conf, weight=w)
        mk = {"present": "+", "absent": "-"}.get(val, "?")
        lead, p = top(1)[0]
        hit = " <-- TARGET" if lead == target else ""
        print(f"[{step:2d}] EIG={eig:5.3f} {mk} {fname[fid][:40]:40s} "
              f"H={wm.entropy():5.2f}  lead: {name_of.get(lead,lead)[:28]:28s} p={p:.3f}{hit}")
        if p >= tau and val == "present":
            print(f"     -> confident (p>={tau}), commit.")
            break

    print("-" * 92)
    top5 = top(5)
    ids5 = [d for d, _ in top5]
    for r, (d, p) in enumerate(top5, 1):
        star = " *" if d == target else ""
        print(f"   {r}. {p:5.3f}  {name_of.get(d,d)}{star}")
    print(f"RESULT  top1={target==ids5[0]}  top3={target in ids5[:3]}  top5={target in ids5[:5]}")


# VivaBench phase order and per-phase EIG budget (18 questions, under the
# 20-action cap). One posterior is carried across phases.
PHASE_ORDER = ["history", "physical", "investigation", "imaging"]
PHASE_BUDGET = {"history": 5, "physical": 3, "investigation": 6, "imaging": 4}
_PHASE_CACHE = {}


def _phase_map(kb_dir):
    """feature_id -> phase, from the KB's feature_phase.csv (cached)."""
    key = str(kb_dir)
    if key not in _PHASE_CACHE:
        import pandas as pd
        fp = pd.read_csv(Path(kb_dir) / "feature_phase.csv")
        _PHASE_CACHE[key] = dict(zip(fp.feature_id, fp.phase))
    return _PHASE_CACHE[key]


def _score_case(wm_loader, vec, target, mode, budget=15, w_absent=0.35, topk=8, phase_map=None):
    """Return (hit1,hit3,hit5,maxpost) for one case under a scoring mode."""
    wm = wm_loader()
    ans = lambda f: (str(vec[f][0]), float(vec[f][1])) if f in vec else ("absent", 1.0)
    if mode in ("global", "focused"):
        for _ in range(budget):
            cands = wm.unasked_features()
            fid, eig = (focused_eig(wm, cands, topk, w_absent) if mode == "focused"
                        else eig_over(wm, cands))
            if fid is None or eig < 1e-3:
                break
            v, c = ans(fid)
            wm.update(fid, v, c, weight=(w_absent if v == "absent" else 1.0))
    elif mode in ("phased-global", "phased-focused"):
        # phase-locked: each phase asks only its own features, in order,
        # under a per-phase budget; the posterior is carried across phases.
        sel_focused = mode.endswith("focused")
        for ph in PHASE_ORDER:
            ph_feats = [f for f in wm.schema if phase_map.get(f) == ph]
            for _ in range(PHASE_BUDGET[ph]):
                cands = [f for f in ph_feats if f not in wm.asked]
                if not cands:
                    break
                fid, eig = (focused_eig(wm, cands, topk, w_absent) if sel_focused
                            else eig_over(wm, cands))
                if fid is None or eig < 1e-3:
                    break
                v, c = ans(fid)
                wm.update(fid, v, c, weight=(w_absent if v == "absent" else 1.0))
    elif mode == "explicit":     # skyline: feed every PRESENT finding
        for f, (v, c) in vec.items():
            if str(v) == "present":
                wm.update(f, v, c)
    elif mode == "full":         # skyline: feed EVERY feature (present + absent)
        for f in list(wm.schema):
            v, c = ans(f)
            wm.update(f, v, c, weight=(w_absent if v == "absent" else 1.0))
    post = wm.posterior(); ids = [wm.leaves[i] for i in np.argsort(post)[::-1][:5]]
    return target == ids[0], target in ids[:3], target in ids[:5], float(post.max())


def _dhs_star(records, alpha=0.5):
    """records = [(hit1, maxpost), ...]. Sweep tau, return DHS-optimal
    (tau, SelAcc, Cov, DHS). SelAcc = top1 on committed (maxpost>=tau);
    Cov = commit rate; DHS = harmonic mean of SelAcc, Cov."""
    n = len(records)
    best = (0.0, 0.0, 0.0, 0.0)
    for tau in sorted({r[1] for r in records} | {0.0}):
        comm = [h for h, p in records if p >= tau]
        cov = len(comm) / n
        if not comm or cov == 0:
            continue
        sa = sum(comm) / len(comm)
        if sa == 0 or cov == 0:
            dhs = 0.0
        else:
            dhs = 1.0 / (alpha / sa + (1 - alpha) / cov)
        if dhs > best[3]:
            best = (tau, sa, cov, dhs)
    return best


_TAG = {"global": "EIG-global", "focused": "EIG-focused",
        "phased-global": "phased-global", "phased-focused": "phased-focused",
        "explicit": "skyline-expl", "full": "skyline-full"}


def batch(kbs=("kb50", "kb_grounded"),
          modes=("global", "focused", "phased-global", "phased-focused", "explicit", "full")):
    vecs = json.loads((HERE / "vec_cache.json").read_text())
    c2d = json.loads((HERE / "diseases50.json").read_text())["case_to_disease"]
    uids = [u for u in vecs if u in c2d]
    n = len(uids)
    print(f"LLM-FREE batch  n={n} cases  (patient answers from vec_cache.json)")
    print(f"{'KB':<15}{'mode':<16}{'T-1':>6}{'T-3':>6}{'T-5':>6}"
          f"{'SelAcc':>8}{'Cov':>6}{'DHS':>6}{'tau*':>6}")
    for kb in kbs:
        loader = lambda kb=kb: load_kb(str(HERE / kb))
        pm = _phase_map(HERE / kb)
        for mode in modes:
            h1 = h3 = h5 = 0
            recs = []
            for u in uids:
                a, b, c, mp = _score_case(loader, vecs[u], c2d[u], mode, phase_map=pm)
                h1 += a; h3 += b; h5 += c
                recs.append((a, mp))
            tau, sa, cov, dhs = _dhs_star(recs)
            tag = _TAG.get(mode, mode)
            print(f"{kb:<15}{tag:<16}{h1/n:>5.0%}{h3/n:>6.0%}{h5/n:>6.0%}"
                  f"{sa:>8.0%}{cov:>6.0%}{dhs:>6.0%}{tau:>6.2f}")


def _dhs_at(records, tau, alpha=0.5):
    """SelAcc/Cov/DHS at a FIXED tau. records = [(hit1, maxpost), ...]."""
    n = len(records)
    if n == 0:
        return 0.0, 0.0, 0.0
    comm = [h for h, p in records if p >= tau]
    cov = len(comm) / n
    if not comm:
        return 0.0, 0.0, 0.0
    sa = sum(comm) / len(comm)
    dhs = 1.0 / (alpha / sa + (1 - alpha) / cov) if sa > 0 and cov > 0 else 0.0
    return sa, cov, dhs


def ci(kbs=("kb50", "kb_grounded"), modes=("phased-global", "phased-focused"), B=2000, seed=42):
    """Bootstrap 95% CIs for Top-1 and DHS (paper protocol: B=2000, seed=42,
    percentile, tau* locked once on the full sample). Patient answers LLM-free."""
    vecs = json.loads((HERE / "vec_cache.json").read_text())
    c2d = json.loads((HERE / "diseases50.json").read_text())["case_to_disease"]
    uids = [u for u in vecs if u in c2d]
    n = len(uids)
    rng = np.random.default_rng(seed)
    print(f"LLM-FREE bootstrap CIs  n={n}  B={B}  seed={seed}  percentile [2.5,97.5]")
    print(f"{'KB':<15}{'mode':<16}{'Top-1 [95% CI]':<22}{'DHS [95% CI]':<22}{'tau*':>6}")
    for kb in kbs:
        loader = lambda kb=kb: load_kb(str(HERE / kb))
        pm = _phase_map(HERE / kb)
        for mode in modes:
            recs = []          # (hit1, maxpost) per case
            for u in uids:
                a, _, _, mp = _score_case(loader, vecs[u], c2d[u], mode, phase_map=pm)
                recs.append((a, mp))
            tau, _, _, _ = _dhs_star(recs)
            hits = np.array([r[0] for r in recs], dtype=float)
            t1_pt = hits.mean()
            _, _, dhs_pt = _dhs_at(recs, tau)
            t1s, dhss = [], []
            for _ in range(B):
                idx = rng.integers(0, n, n)
                t1s.append(hits[idx].mean())
                boot = [recs[i] for i in idx]
                dhss.append(_dhs_at(boot, tau)[2])
            t1_lo, t1_hi = np.percentile(t1s, [2.5, 97.5])
            d_lo, d_hi = np.percentile(dhss, [2.5, 97.5])
            tag = _TAG.get(mode, mode)
            t1s_str = f"{t1_pt:.0%} [{t1_lo:.0%},{t1_hi:.0%}]"
            dhs_str = f"{dhs_pt:.0%} [{d_lo:.0%},{d_hi:.0%}]"
            print(f"{kb:<15}{tag:<16}{t1s_str:<22}{dhs_str:<22}{tau:>6.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("uid", nargs="?")
    ap.add_argument("--all", action="store_true", help="LLM-free batch over 50 cases")
    ap.add_argument("--ci", action="store_true", help="bootstrap 95% CIs for T-1 and DHS")
    ap.add_argument("--kb", default="kb50")
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--wabsent", type=float, default=0.35)
    ap.add_argument("--tau", type=float, default=0.85)
    ap.add_argument("--focused", action="store_true")
    ap.add_argument("--topk", type=int, default=8)
    a = ap.parse_args()
    if a.ci:
        ci()
    elif a.all:
        batch()
    else:
        run(a.uid, a.kb, a.budget, a.wabsent, a.tau, a.focused, a.topk)
