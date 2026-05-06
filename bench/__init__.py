"""bench/ — single entrypoint for every (row, column) cell of Table 4.

Usage:
    python -m bench --row cw --column ac --num-cases 50
    python -m bench --row bmbe --column ac --kb-version v2
    python -m bench --row cod --column patientsim --remote clariden

Layout:
    columns/<name>.py  — case loader + patient adapter + disease universe + output dir
    rows/<name>.py     — doctor wiring (thin wrapper around existing baselines/)
    runner.py          — generic per-case loop (shared by every cell)
    cli.py             — argparse + dispatcher
"""
