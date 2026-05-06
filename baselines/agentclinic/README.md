# baselines/agentclinic

Adapters and runners for the **AgentClinic-MedQA** column (50 cases sampled
by frequency-weighted strategy from `agentclinic_medqa_extended.jsonl`,
yielding a 31-disease closed world). Also hosts the cross-column AC-native
doctor driver used by the HT-sim column.

## Setup

```bash
git clone https://github.com/SamuelSchmidgall/AgentClinic /tmp/agentclinic-tmp
export AGENTCLINIC_REPO=/tmp/agentclinic-tmp
```

## What's here

| File | Role |
|---|---|
| `adapters/` | Per-baseline wrappers around AC's `PatientAgent` (`base`, `bmbe`, `cod`, `mediq`, `meddxagent`, `standalone`) — needed because AC's legacy `query_model` hardcodes a small whitelist of model names |
| `ac_native_doctor.py` | Thin wrapper around AC's published `DoctorAgent` (translates `DIAGNOSIS READY:` ↔ `[DDX]`); reused by the HT-sim column |
| `parsing_prompts.py` | Lenient `[DDX]/[ABSTAIN]` parser shared across `[DDX]`-emitting baselines |
| `run_ac_native.py` | Verbatim port of `agentclinic.main()` per-case loop (final-turn forcing + `MeasurementAgent` + `compare_results` moderator) |
| `posthoc_threshold_sweep.py` | Replays a force-max-turns BMBE run and picks DHS-optimal τ |

## Running

Every AC-MedQA cell goes through the unified driver:

```bash
PYTHONPATH=. python -m bench --row <row> --column ac --num-cases 50
```

See `bench/README.md` for the full Row × Column table.
