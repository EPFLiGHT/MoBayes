"""Pretty terminal viewer for CoD AC-MedQA canonical jsonls.

Two files compared side-by-side:
  ORIG  : data/benchmark_results/set_a/agentclinic/baselines/cod_diagnosisgpt34b.jsonl
  RECOV : data/benchmark_results/set_a/agentclinic/baselines/cod_diagnosisgpt34b_recovered.jsonl

Modes
-----
  python3 scripts/view_cod_dialogs.py
      one-line summary per case + aggregate metrics

  python3 scripts/view_cod_dialogs.py --case N
      full dialog for case N (0-indexed); shows ORIG and RECOV side-by-side

  python3 scripts/view_cod_dialogs.py --filter hit | abstain | ddx | maxturns | changed
      hit       : only cases where ORIG marked top1_hit=True
      abstain   : only cases marked abstained=True in RECOV
      ddx       : only cases where stop_reason=DDX_PROVIDED in ORIG
      maxturns  : only cases where stop_reason=MAX_TURNS in ORIG
      changed   : only cases where the predicted (or abstained) label differs

  python3 scripts/view_cod_dialogs.py --orig <path> --recov <path>
      override default jsonl paths
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

DEFAULT_ORIG = Path(
    ""
    "data/benchmark_results/set_a/agentclinic/baselines/cod_diagnosisgpt34b.jsonl"
)
DEFAULT_RECOV = Path(
    ""
    "data/benchmark_results/set_a/agentclinic/baselines/cod_diagnosisgpt34b_recovered.jsonl"
)

# ANSI colors (no rich/colorama dep)
def _ansi(code, s): return f"\033[{code}m{s}\033[0m"
GREEN = lambda s: _ansi("32", s)
RED   = lambda s: _ansi("31", s)
YEL   = lambda s: _ansi("33", s)
DIM   = lambda s: _ansi("2",  s)
BOLD  = lambda s: _ansi("1",  s)
CYAN  = lambda s: _ansi("36", s)


def hit_glyph(hit):  return GREEN("✓") if hit else RED("✗")
def abst_tag(a):     return YEL("ABST") if a else DIM("    ")


def _trunc(s, n=80):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n-1] + "…"


def metrics(rows):
    n = len(rows)
    abst = sum(1 for r in rows if r.get("abstained"))
    committed = n - abst
    t1 = sum(1 for r in rows if (not r.get("abstained")) and r.get("top1_hit"))
    sa = t1/committed if committed else 0
    cov = committed/n
    dhs = 2*sa*cov/(sa+cov) if sa+cov else 0
    return dict(n=n, T1=round(100*t1/n), SA=round(100*sa),
                Cov=round(100*cov), DHS=round(100*dhs),
                abst=abst, committed=committed)


def banner(label, m, color=BOLD):
    print(color(f"  {label:<8}  T1={m['T1']:>3}  SA={m['SA']:>3}  Cov={m['Cov']:>3}  "
                f"DHS={m['DHS']:>3}  (abst={m['abst']}/{m['n']})"))


def summary(orig, recov):
    print(BOLD("━" * 96))
    print(BOLD(" CoD on AC-MedQA  ──  one-line per case  ──  ORIG vs RECOV "))
    print(BOLD("━" * 96))
    banner("ORIG ", metrics(orig))
    banner("RECOV", metrics(recov))
    print()
    print(DIM(" idx  GT                                  | ORIG hit pred                       "
              " RECOV abst hit pred                  stop_reason"))
    print(DIM(" " + "─" * 95))
    for i, (o, r) in enumerate(zip(orig, recov)):
        gt = (o.get("ground_truth") or "")[:35]
        o_pred = (o.get("predicted") or "")[:30]
        r_pred = (r.get("predicted") or "")[:30]
        o_hit  = bool(o.get("top1_hit"))
        r_hit  = bool(r.get("top1_hit"))
        r_abst = bool(r.get("abstained"))
        sr = (r.get("stop_reason") or o.get("stop_reason") or "")[:18]
        print(f"  {i:>2}  {gt:<36}| {hit_glyph(o_hit)} {o_pred:<32}"
              f" {abst_tag(r_abst)} {hit_glyph(r_hit)} {r_pred:<22} {DIM(sr)}")
    print()


def case_detail(idx, orig, recov):
    if idx < 0 or idx >= len(orig):
        print(RED(f"case index {idx} out of range (0..{len(orig)-1})"))
        return
    o, r = orig[idx], recov[idx]
    gt = o.get("ground_truth")
    print(BOLD("━" * 96))
    print(BOLD(f" Case {idx}   GT = {gt}"))
    print(BOLD("━" * 96))

    def showrow(label, row, color=BOLD):
        print(color(f"\n[{label}]"))
        print(f"  predicted          : {row.get('predicted')!r}")
        print(f"  predicted_set      : {row.get('predicted_set')}")
        print(f"  top1_hit / top3_hit: {hit_glyph(row.get('top1_hit'))} / {hit_glyph(row.get('top3_hit'))}")
        print(f"  abstained          : {YEL('TRUE') if row.get('abstained') else DIM('false')}")
        print(f"  stop_reason        : {row.get('stop_reason')!r}")
        if row.get("posthoc_recomputed"):
            print(f"  posthoc_recomputed : {GREEN('True')}")
        diag = (row.get("diagnosis_text") or "").strip()
        if diag:
            print(f"  diagnosis_text     : {diag}")
        else:
            print(f"  diagnosis_text     : {DIM('(empty)')}")
        conf = row.get("confidence_distribution") or {}
        if conf:
            top3 = sorted(conf.items(), key=lambda kv: -kv[1])[:3]
            print(f"  top-3 confidence   : "
                  + ", ".join(f"{d}({c:.2f})" for d, c in top3))

    showrow("ORIG", o)
    showrow("RECOV", r, color=CYAN)

    dialog = o.get("dialog") or r.get("dialog") or []
    if dialog:
        print(BOLD(f"\n[Dialogue ({len(dialog)} turns)]"))
        for t in dialog:
            role = (t.get("role") or "?").upper()
            cnt = (t.get("content") or "").strip()
            if not cnt:
                continue
            if role == "PATIENT":
                print(f"  {DIM('PAT')}  {_trunc(cnt, 200)}")
            elif role == "DOCTOR":
                print(f"  {GREEN('DOC')}  {_trunc(cnt, 200)}")
            else:
                print(f"  {DIM(role[:3])}  {_trunc(cnt, 200)}")
    print()


def filtered(orig, recov, filt):
    pairs = list(enumerate(zip(orig, recov)))
    if filt == "hit":
        pairs = [p for p in pairs if p[1][0].get("top1_hit")]
    elif filt == "abstain":
        pairs = [p for p in pairs if p[1][1].get("abstained")]
    elif filt == "ddx":
        pairs = [p for p in pairs if p[1][0].get("stop_reason") == "DDX_PROVIDED"]
    elif filt == "maxturns":
        pairs = [p for p in pairs if p[1][0].get("stop_reason") == "MAX_TURNS"]
    elif filt == "changed":
        pairs = [p for p in pairs if (p[1][0].get("predicted") != p[1][1].get("predicted")
                                      or bool(p[1][0].get("abstained")) != bool(p[1][1].get("abstained")))]
    else:
        print(RED(f"unknown filter: {filt!r}"))
        return
    print(BOLD(f"  filter={filt}  ->  {len(pairs)} cases"))
    print()
    if not pairs:
        return
    print(DIM(" idx  GT                                  | ORIG hit pred                       "
              " RECOV abst hit pred                  stop_reason"))
    for i, (o, r) in pairs:
        gt = (o.get("ground_truth") or "")[:35]
        o_pred = (o.get("predicted") or "")[:30]
        r_pred = (r.get("predicted") or "")[:30]
        sr = (r.get("stop_reason") or o.get("stop_reason") or "")[:18]
        print(f"  {i:>2}  {gt:<36}| {hit_glyph(o.get('top1_hit'))} {o_pred:<32}"
              f" {abst_tag(r.get('abstained'))} {hit_glyph(r.get('top1_hit'))} {r_pred:<22} {DIM(sr)}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig",  type=Path, default=DEFAULT_ORIG)
    ap.add_argument("--recov", type=Path, default=DEFAULT_RECOV)
    ap.add_argument("--case",  type=int, default=None,
                    help="show full dialog for this case index")
    ap.add_argument("--filter", choices=["hit", "abstain", "ddx", "maxturns", "changed"],
                    default=None,
                    help="filter the summary to a subset of cases")
    args = ap.parse_args()

    if not args.orig.exists():
        sys.exit(RED(f"missing: {args.orig}"))
    if not args.recov.exists():
        sys.exit(RED(f"missing: {args.recov}"))

    orig  = [json.loads(l) for l in args.orig.open()]
    recov = [json.loads(l) for l in args.recov.open()]
    if len(orig) != len(recov):
        print(YEL(f"warning: row counts differ ({len(orig)} vs {len(recov)})"))

    if args.case is not None:
        case_detail(args.case, orig, recov)
    elif args.filter:
        filtered(orig, recov, args.filter)
    else:
        summary(orig, recov)


if __name__ == "__main__":
    main()
