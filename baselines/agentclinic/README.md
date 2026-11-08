# baselines/agentclinic

AgentClinic's published `DoctorAgent` (Schmidgall et al., 2024) as a baseline
(row `ac_native`).

| File | Role |
|---|---|
| `doctor.py` | Wraps AC's `DoctorAgent`; maps its free-text `DIAGNOSIS READY:` to the column's closed disease list. Used on the PatientSim and HT-sim columns |

On the AgentClinic columns the row (`baselines/agentclinic/row.py`) runs AC's own
per-case loop instead (`DoctorAgent` + `PatientAgent` + `MeasurementAgent`
+ LLM moderator). AC's `query_model` uses the legacy openai 0.x SDK with a
fixed model list, so both paths keep AC's prompts but call the openai 1.x
client directly.

## Setup

```bash
git clone https://github.com/SamuelSchmidgall/AgentClinic /path/to/agentclinic
export AGENTCLINIC_REPO=/path/to/agentclinic
```

`AGENTCLINIC_REPO` is also needed by the `ac` / `ac_set2` columns, whose
patient is AC's `PatientAgent` (`bench/patients/agentclinic.py`).
