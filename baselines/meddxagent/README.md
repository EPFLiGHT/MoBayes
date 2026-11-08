# baselines/meddxagent

MEDDxAgent ([Rose et al., ACL 2025](https://arxiv.org/abs/2502.19175)) as a
baseline (row `meddxagent`) on the PatientSim, HT-sim and AgentClinic-MedQA columns.

## Setup (one-time)

```bash
git clone https://github.com/nec-research/meddxagent ~/Documents/GitHub/meddxagent
cd ~/Documents/GitHub/meddxagent && uv sync
export MEDDXAGENT_REPO=~/Documents/GitHub/meddxagent
```

Patch the vendor repo so doctor / patient OpenAI tokens land in separate
counters. Without this the benchmark still runs, but tokens get conflated:

- `ddxdriver/models/oai_chat.py` — add a `_bmbe_actor` attribute (default
  `"doctor"`) and route `usage_metadata` to `add_doctor` or `add_patient`
  in `baselines.meddxagent.usage_tracker` based on that flag.
- `ddxdriver/patient_agents/llm_patient.py` — in `LLMPatient.__init__`,
  set `self.model._bmbe_actor = "patient"`.

## What's here

`row.py` (row `meddxagent`) calls `run.py --column <col>` in a subprocess with
the MEDDxAgent venv, then converts MEDDxAgent's `results.json` to per-case JSON.

| File | Role |
|---|---|
| `row.py` | Bench row |
| `run.py` | Runner: builds MEDDxAgent's configs for the column and runs `ddxdriver` |
| `benchmarks.py` | ddxdriver bench classes: `MoBayesDDxPlus50` (PatientSim), `DDxPlusHTSim50` (HT-sim), `AgentClinic50` |
| `patients.py` | ddxdriver patients: `MoBayesPatientAgent` (PatientSim), `ACMEDDxAgentPatient` (AgentClinic); HT-sim uses MEDDxAgent's own `LLMPatient` |
| `rag.py` | `ClosedWorldDDxRAG` — optional retriever over the column's KB (`run.py --rag-source closedworld`) |
| `usage_tracker.py` | Token accumulator fed by the patched `OpenAIChat` |

## Running

All three columns go through the unified driver:

```bash
PYTHONPATH=. python -m bench --row meddxagent --column {patientsim,htsim,ac}
```

Variants (`--row-kwargs '{"variant": ...}'`):

- `paper-best` (default): PubMed retrieval + 5 dynamic few-shot examples
  (nearest neighbours by embedding). Needs CUDA.
- `closed-world`: PubMed retrieval and few-shot examples disabled.

In both variants the diagnosis agent's prompt lists the column's closed
disease set ("Diagnosis Options").
