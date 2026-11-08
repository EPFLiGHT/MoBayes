# bench/ : single entrypoint for every main-table cell

Every (row, column) pair in the paper's baseline comparison is one command.
A **column** is a benchmark (cases + patient simulator + disease list), a
**row** is a diagnostic system (MoBayes or a baseline):

```bash
PYTHONPATH=. python3 -m bench --row <row> --column <column> [opts]
```

---

## Layout

```
bench/
  cli.py             # argparse → runner.run
  runner.py          # generic per-case loop + 3-tier dispatch
  parallel.py        # run several rows × columns in parallel
  columns/           # benchmarks; each half of the n=100 split is its own column
    patientsim.py    patientsim_v1.py   # DDxPlus + our PatientSim (sets a / b)
    htsim.py         htsim_v1.py        # DDxPlus + MEDDxAgent's LLMPatient (sets a / b)
    ac.py            ac_set2.py         # AgentClinic-MedQA + AC's PatientAgent (sets a / b)
  patients/          # patient simulators shared with baseline runners
    agentclinic.py   # AC PatientAgent wrapper
    htsim.py         # MEDDxAgent LLMPatient wrapper
  rows.py            # row registry: --row name → module (below)
  mobayes.py         # row bmbe: MoBayes (Bayesian engine + EIG)

baselines/           # one folder per prior system, each with its row
  prompting/rows.py  # sa, cw, cot, savage_dd, savage_bay (doctors in prompting/doctors.py)
  agentclinic/row.py # ac_native: AgentClinic's DoctorAgent
  uot/row.py         # uot
  mediq/row.py       # mediq: subprocess to mediQ_benchmark.py
  meddxagent/row.py  # meddxagent: subprocess to meddxagent/run.py
  cod/row.py         # cod: writes a Slurm script (DiagnosisGPT-34B needs a GPU)
```

---

## How dispatch works

`runner.py::run` looks the row up in `rows.py` and uses the first of these
it defines:

1. `row.run_batch(...)`     — full-column subprocess (mediq, meddxagent, cod)
2. `row.run_one_case(...)`  — custom per-case loop (ac_native, bmbe)
3. `row.make_doctor(...)`   — generic per-case loop (sa, cw, cot, savage_*, uot)

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
| **BMBE × AC** | `python -m bench --row bmbe --column ac --max-turns 20` |
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
| `OPENAI_API_KEY` | env var (or a repo-root `.env`) | every row |
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
combined n=100 reported in the paper. The second AgentClinic half is the
`ac_set2` column (cases in `data/cases/agentclinic_b/`).

MEDDxAgent paper-best (PubMed RAG + dynamic 5-shot fewshot) requires CUDA
and runs on an HPC node.

---

## Adding a new row or column

**New row** — add a module (usually `baselines/<system>/row.py`) that exposes
`NAME` and any one of the functions below, and register it in `bench/rows.py`:

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
