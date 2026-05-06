"""CLI entrypoint: python -m bench --row X --column Y."""

from __future__ import annotations

import argparse

from bench.runner import run


def main() -> None:
    p = argparse.ArgumentParser(description="Table 4 cell runner — one command per (row, column).")
    p.add_argument("--row", required=True,
                   help="Baseline name (loads bench.rows.<row>)")
    p.add_argument("--column", required=True,
                   help="Benchmark name (loads bench.columns.<column>)")
    p.add_argument("--num-cases", type=int, default=50)
    p.add_argument("--start-idx", type=int, default=0,
                   help="Skip the first N cases (resume / extend)")
    p.add_argument("--max-turns", type=int, default=15)
    p.add_argument("--doctor-model", default="gpt-5.4-nano")
    p.add_argument("--patient-model", default="gpt-5.4-nano")
    p.add_argument("--out-suffix", default="",
                   help="Optional suffix appended to output dir name")
    p.add_argument("--doctor-base-url", default=None,
                   help="Base URL for doctor LLM (e.g. https://openrouter.ai/api/v1)")
    p.add_argument("--doctor-api-key", default=None,
                   help="API key for doctor LLM (defaults to OPENAI_API_KEY)")
    p.add_argument("--row-kwargs", default="{}",
                   help='JSON dict of row-specific options (e.g. \'{"eig_topk_weight":1.0,'
                        '"eig_focus_conf_threshold":0.3}\' for BMBE focused EIG)')
    args = p.parse_args()
    import json as _json
    row_kwargs = _json.loads(args.row_kwargs) if args.row_kwargs else {}

    run(
        column_name=args.column,
        row_name=args.row,
        num_cases=args.num_cases,
        start_idx=args.start_idx,
        max_turns=args.max_turns,
        doctor_model=args.doctor_model,
        patient_model=args.patient_model,
        out_suffix=args.out_suffix,
        row_kwargs=row_kwargs,
        doctor_base_url=args.doctor_base_url,
        doctor_api_key=args.doctor_api_key,
    )


if __name__ == "__main__":
    main()
