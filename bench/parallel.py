"""Parallel runner: dispatch (row, column, case_idx) jobs across workers.

Only methods that use the generic per-case loop are eligible — `run_batch`
rows (mediq, meddxagent, cod) are not parallelisable here and must be
launched separately via `python -m bench --row X --column Y`.

Usage:
    PYTHONPATH=. python -m bench.parallel \
        --rows cw,cot,savage_dd,savage_bay,ac_native,bmbe,uot \
        --columns patientsim_v1,htsim_v1 \
        --workers 22 \
        --max-turns 15
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any


def _run_one(row_name: str, col_name: str, case_idx: int, max_turns: int,
              doctor_model: str, patient_model: str, out_suffix: str,
              row_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entry: import inside the process to avoid stale state across workers."""
    from bench import columns, rows
    from bench.runner import run_one_case
    api_key = os.environ.get("OPENAI_API_KEY")
    column = columns.get(col_name)
    row = rows.get(row_name)
    cases = column.load_cases()
    case = cases[case_idx]
    out_dir = column.output_dir(row.NAME, out_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        log = run_one_case(
            case, case_idx,
            column=column, row=row,
            doctor_model=doctor_model, patient_model=patient_model,
            max_turns=max_turns, api_key=api_key, out_dir=out_dir,
            row_kwargs=row_kwargs,
        )
        f = log.get("final", {})
        return {
            "row": row_name, "col": col_name, "idx": case_idx,
            "ok": True,
            "hit": bool(f.get("top1_hit")),
            "abst": bool(f.get("abstained")),
            "elapsed": time.time() - t0,
        }
    except Exception as e:
        return {
            "row": row_name, "col": col_name, "idx": case_idx,
            "ok": False, "error": str(e)[:200],
            "trace": traceback.format_exc()[:500],
            "elapsed": time.time() - t0,
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", required=True, help="comma-separated row names")
    p.add_argument("--columns", required=True, help="comma-separated column names")
    p.add_argument("--workers", type=int, default=22)
    p.add_argument("--max-turns", type=int, default=15)
    p.add_argument("--doctor-model", default="gpt-5.4-nano")
    p.add_argument("--patient-model", default="gpt-5.4-nano")
    p.add_argument("--out-suffix", default="")
    p.add_argument("--row-kwargs", default="{}")
    args = p.parse_args()

    rows = [r.strip() for r in args.rows.split(",") if r.strip()]
    cols = [c.strip() for c in args.columns.split(",") if c.strip()]
    row_kwargs = json.loads(args.row_kwargs)

    # Build job queue: every (row, col, case_idx)
    from bench import columns as cols_pkg
    jobs = []
    for col in cols:
        column = cols_pkg.get(col)
        n = len(column.load_cases())
        for row in rows:
            for idx in range(n):
                jobs.append((row, col, idx))

    print(f"[parallel] {len(jobs)} jobs ({len(rows)} rows x {len(cols)} cols x cases) "
          f"-- workers={args.workers}", flush=True)

    from bench.runner import _write_manifest
    for col in cols:
        column = cols_pkg.get(col)
        for row in rows:
            out_dir = column.output_dir(row, args.out_suffix)
            out_dir.mkdir(parents=True, exist_ok=True)
            _write_manifest(
                out_dir, row=row, column=col, out_dir_name=out_dir.name,
                n_cases=len(column.load_cases()), start_idx=0,
                num_cases=len(column.load_cases()),
                max_turns=args.max_turns,
                doctor_model=args.doctor_model,
                patient_model=args.patient_model,
                doctor_base_url=None, out_suffix=args.out_suffix,
                row_kwargs=row_kwargs, parallel_workers=args.workers,
            )

    n_done = n_hit = n_abst = n_err = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_one, r, c, i, args.max_turns,
                              args.doctor_model, args.patient_model,
                              args.out_suffix, row_kwargs): (r, c, i)
                    for (r, c, i) in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            n_done += 1
            tag = "ERR" if not res["ok"] else \
                   ("HIT" if res["hit"] else ("ABS" if res["abst"] else "miss"))
            if not res["ok"]: n_err += 1
            elif res["hit"]: n_hit += 1
            elif res["abst"]: n_abst += 1
            elapsed_total = time.time() - t0
            print(f"  [{n_done}/{len(jobs)}] {res['row']:>10}/{res['col']:<14} "
                  f"#{res['idx']:>2} {tag:>4} ({res['elapsed']:.0f}s) "
                  f"| HIT={n_hit} ABS={n_abst} ERR={n_err} | total {elapsed_total:.0f}s",
                  flush=True)

    print(f"\n[done] {n_done} jobs in {time.time()-t0:.0f}s "
          f"| HIT={n_hit} ABS={n_abst} ERR={n_err}")


if __name__ == "__main__":
    main()
