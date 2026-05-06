# baselines/meddxagent

MEDDxAgent ([Rao et al., 2025](https://arxiv.org/abs/2502.19175)) adapters
for the PatientSim, HT-sim, and AgentClinic-MedQA columns.

## Setup (one-time)

```bash
git clone https://github.com/AgentAITech/meddxagent ~/Documents/GitHub/meddxagent
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

| File | Role |
|---|---|
| `benchmark.py`, `patient.py` | PatientSim adapter (our profiles → MEDDxAgent's `Bench` / `PatientAgent`) |
| `benchmark_agentclinic.py`, `patient_agentclinic.py` | AC-MedQA adapter |
| `rag.py` | `ClosedWorldDDxRAG` — local DDxPlus KB retriever (closed-world ablation) |
| `usage_tracker.py` | Module-level token accumulator fed by the patched `OpenAIChat` |

The HT-sim adapter (`htsim_meddxagent_bench.py`) lives next door in
`baselines/streambench/`.

## Running

All three columns go through the unified driver:

```bash
PYTHONPATH=. python -m bench --row meddxagent --column {patientsim,htsim,ac}
```

See `bench/README.md` for the full Row × Column command table, including
the paper-best / closed-world variants and the Clariden sbatch path for
the HT-sim paper-best run.
