# baselines/streambench

Runners and patient-side adapters for the **DDxPlus 50 (MEDDxAgent HT-sim)**
column. Every baseline drives the same 50 cases against MEDDxAgent's
published `LLMPatient`.

## Setup

Shares the MEDDxAgent clone + token-tracking patches with
`baselines/meddxagent/` — see that README. AgentClinic native baseline
additionally needs `git clone https://github.com/SamuelSchmidgall/AgentClinic
/tmp/agentclinic-tmp`. MediQ-Expert needs the mediQ clone described in the
top-level README.

HT-sim runs need a CUDA GPU; Apple Silicon segfaults on
`sentence_transformers + faiss`.

## What's here

| File | Role |
|---|---|
| `htsim_meddxagent_bench.py` | `DDxPlusHTSim50` bench class (our 50-case JSONL → MEDDxAgent bench interface) |
| `sample_cases.py` | Stratified sampler used once to produce `data/cases/ddxplus_htsim_a/selected_cases.jsonl` |
| `adapters/{base,bmbe,mediq,meddxagent,standalone,cod}.py` | Per-baseline patient-side wrappers around `LLMPatient` |

## Running

Every HT-sim cell is one command via the unified driver:

```bash
PYTHONPATH=. python -m bench --row <row> --column htsim
```

See `bench/README.md` for the full Row × Column table.
