# bench/ : single entrypoint for every main-table cell

Replaces ~18 per-baseline runner scripts. Every (row, column) pair in
the paper's baseline-adaptations table is one command:

```bash
PYTHONPATH=. python3 -m bench --row <row> --column <column> [opts]
```

---

## Layout

```
bench/
  __init__.py
  __main__.py
  cli.py             # argparse + dispatcher
  runner.py          # generic per-case loop + 3-tier dispatch
  columns/
    __init__.py      # Protocol + lazy loader
    ac.py            # AgentClinic-MedQA (50 cases, AC PatientAgent, 31 diseases)
    htsim.py         # DDxPlus 50 + MEDDxAgent's LLMPatient (49 + 9 categories)
    patientsim.py    # DDxPlus 50 + our SimplePatientAgent (49 + 9 categories)
  rows/
    __init__.py      # Protocol + lazy loader
    cw.py            cot.py    savage_dd.py    savage_bay.py    # prompt-based
    ac_native.py     # AC's published DoctorAgent (verbatim port)
    bmbe.py          # Bayesian engine + EIG (KB v1/v2 selectable)
    mediq.py         # subprocess to mediQ_benchmark.py + per-column patient
    meddxagent.py    # subprocess to MEDDxAgent driver + per-column bench
    cod.py           # HPC sbatch emit (DiagnosisGPT-34B local model)
```

---

## How dispatch works

`runner.py::run` checks each row module in this order and uses the first
one that exists:

1. `row.run_batch(...)`     — full-column subprocess (mediq, meddxagent, cod)
2. `row.run_one_case(...)`  — custom per-case loop (ac_native, bmbe)
3. `row.make_doctor(...)`   — generic per-case loop (cw, cot, savage_*)

The generic per-case loop in `runner.py::run_one_case` is shared by every
row in tier 3: greeting → patient.respond → doctor.get_question loop with
`[DDX]/[ABSTAIN]` parsing → output JSON.

---

## Quick usage: every main-table cell

| Row × Column | Command |
|---|---|
| **CW × AC** | `python -m bench --row cw --column ac --num-cases 50` |
| **CW × HT-sim** | `$MEDDXAGENT_VENV/bin/python -m bench --row cw --column htsim` |
| **CW × PatientSim** | `python -m bench --row cw --column patientsim --num-cases 50` |
| **CoT × AC** | `python -m bench --row cot --column ac --num-cases 50` |
| **DDx-CoT × AC** | `python -m bench --row savage_dd --column ac --num-cases 50` |
| **Bayes-CoT × AC** | `python -m bench --row savage_bay --column ac --num-cases 50` |
| **AC native × AC** | `python -m bench --row ac_native --column ac --max-turns 20` |
| **AC native × HT-sim** | `$MEDDXAGENT_VENV/bin/python -m bench --row ac_native --column htsim --max-turns 20` |
| **AC native × PatientSim** | `python -m bench --row ac_native --column patientsim --max-turns 20` |
| **BMBE × AC (v2 KB)** | `python -m bench --row bmbe --column ac --max-turns 20` |
| **BMBE × HT-sim** | `$MEDDXAGENT_VENV/bin/python -m bench --row bmbe --column htsim --max-turns 20` |
| **BMBE × PatientSim** | `python -m bench --row bmbe --column patientsim --max-turns 20` |
| **MediQ × AC** | `python -m bench --row mediq --column ac --max-turns 10` |
| **MediQ × HT-sim** | `python -m bench --row mediq --column htsim --max-turns 10` |
| **MediQ × PatientSim** | `python -m bench --row mediq --column patientsim --max-turns 10` |
| **MEDDxAgent CW × AC** | `python -m bench --row meddxagent --column ac` (default variant=paper-best; pass `--row-kwargs '{"variant":"closed-world"}'`) |
| **MEDDxAgent paper-best × HT-sim** | `python -m bench --row meddxagent --column htsim` |
| **CoD × {ac,htsim,patientsim}** | `python -m bench --row cod --column <col>` (emits HPC sbatch; user runs sbatch manually) |

CLI flags: `--num-cases N`, `--start-idx N`, `--max-turns N`, `--doctor-model`,
`--patient-model`, `--out-suffix STR`.

---

## Outputs

Every cell writes one JSON per case to:

```
data/benchmark_results/<column>_50_<row>[_<suffix>]/case_NNN.json
```

`columns/<col>.output_dir(row, suffix)` controls the exact path — for AC
that's `agentclinic_50_<row>` (legacy naming preserved for compatibility
with existing aggregators); for HT-sim and PatientSim:
`ddxplus_htsim_50_<row>` and `ddxplus_patientsim_50_<row>`.

Per-case JSON schema (always at least these keys):

```json
{
  "case_idx": 0,
  "baseline": "cw",
  "column":   "ac",
  "ground_truth":         "Acalculous cholecystitis",
  "patient_actor_hidden": {...},
  "objective_for_doctor": "Age 67 / Female ...",
  "turns":  [{...per-turn...}],
  "final": {
    "predicted":     "Acalculous cholecystitis",
    "predicted_set": ["...", "...", ...],
    "top1_hit":      true,
    "top3_hit":      true,
    "abstained":     false,
    "stop_reason":   "DDX_PROVIDED",
    "turns":         11
  },
  "token_usage": {"doctor": {...}, "patient": {...}}
}
```

---

## Aggregating to table cells

After cases are produced, scoring goes through the unified LLM judge:

```bash
PYTHONPATH=. python3 scripts/analysis/llm_remap_unified.py        # → judged/
PYTHONPATH=. python3 scripts/analysis/recompute_metrics_from_judged.py
```

Judged outputs are mirrored to `data/benchmark_results/judged/<dir>/` so
raw outputs stay untouched. Tables read the `top1_hit_judged`,
`top3_hit_judged`, `abstained_judged` fields, not the strict-match originals.

---

## Prerequisites

| Dep | Where | Required for |
|---|---|---|
| `OPENAI_API_KEY` | env var (or `hybrid-diagnostic-system/.env`) | every row |
| `AGENTCLINIC_REPO=/tmp/agentclinic-tmp` | env var | column=ac, row=ac_native |
| `MEDIQ_REPO=~/Documents/GitHub/mediQ` | env var (default ok) | row=mediq |
| `MEDDXAGENT_REPO=~/Documents/GitHub/meddxagent` | env var (default ok) | row=meddxagent, column=htsim |
| meddxagent venv (Python 3.12) | `~/Documents/GitHub/meddxagent/.venv` | column=htsim (LLMPatient uses 3.10+ syntax), row=meddxagent |
| HPC SSH access | `~/.ssh/hpc-key` (daily refresh) | row=cod |

---

## v1 columns (combined n=100)

`htsim_v1.py` and `patientsim_v1.py` are disjoint 50-patient samples used
to replicate Table 1 on a fresh half. Outputs land under
`data/benchmark_results/set_b/`; aggregated with `set_a/` they give the
combined n=100 reported in the paper. AgentClinic has no v1 (single
canonical sample).

MEDDxAgent paper-best (PubMed RAG + dynamic 5-shot fewshot) requires CUDA
and runs on an HPC node; sbatch wrappers under
`scripts/hpc_meddxagent_pubmed_dynamic_*.sbatch`.

---

## Adding a new row or column

**New row** — drop a file `bench/rows/<name>.py` that exposes `NAME` and any one of:

```python
def make_doctor(disease_universe, doctor_model, api_key, max_turns,
                column_name, **kwargs): ...
def run_one_case(case, case_idx, *, column, doctor_model, patient_model,
                  max_turns, api_key, out_dir, row_kwargs): ...
def run_batch(*, cases, start_idx, column, doctor_model, patient_model,
              max_turns, api_key, out_dir, row_kwargs): ...
```

**New column** — drop `bench/columns/<name>.py` exposing `NAME`,
`load_cases`, `make_patient`, `disease_universe`, `output_dir`,
`matches_gt`, `gt_of`. The Protocol is `bench.columns.Column`.
