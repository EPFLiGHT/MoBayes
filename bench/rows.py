"""Row registry: ``--row <name>`` → the module or object that wires a system
into the bench loop. MoBayes lives in bench/mobayes.py; every baseline keeps
its row next to its code in baselines/<system>/.

A row exposes ``NAME`` and one of (checked in this order by runner.run):
  run_batch(...)     whole column in one call (mediq, meddxagent, cod)
  run_one_case(...)  own per-case loop (bmbe, ac_native)
  make_doctor(...)   doctor for the generic loop in runner.run_one_case
                     (the doctor exposes get_question / parse_ddx / parse_possible)
"""

from importlib import import_module

ROWS = {
    "bmbe":       "bench.mobayes",
    "sa":         "baselines.prompting.rows:SA",
    "cw":         "baselines.prompting.rows:CW",
    "cot":        "baselines.prompting.rows:COT",
    "savage_dd":  "baselines.prompting.rows:SAVAGE_DD",
    "savage_bay": "baselines.prompting.rows:SAVAGE_BAY",
    "ac_native":  "baselines.agentclinic.row",
    "uot":        "baselines.uot.row",
    "mediq":      "baselines.mediq.row",
    "meddxagent": "baselines.meddxagent.row",
    "cod":        "baselines.cod.row",
}


def get(name: str):
    if name not in ROWS:
        raise SystemExit(f"Unknown row {name!r}; choose from {', '.join(ROWS)}")
    module, _, attr = ROWS[name].partition(":")
    mod = import_module(module)
    return getattr(mod, attr) if attr else mod
