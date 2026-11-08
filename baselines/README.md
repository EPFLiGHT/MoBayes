# baselines/

Prior diagnostic-dialogue systems compared against MoBayes, one folder per
system. Each folder holds the system's `row` (how `python -m bench --row
<name>` drives it) and its code; the registry is `bench/rows.py`.

| Row | System | Folder | External checkout |
|---|---|---|---|
| `sa` | Standalone LLM doctor | `prompting/` (`StandaloneDoctorAgent` is in `scripts/core/benchmark_patientsim.py`) | — |
| `cw` | Closed-world prompt | `prompting/` | — |
| `cot` | Chain-of-Thought | `prompting/` | — |
| `savage_dd`, `savage_bay` | DDx-CoT / Bayesian-CoT (Savage et al., 2024) | `prompting/` | — |
| `ac_native` | AgentClinic's DoctorAgent | `agentclinic/` | AgentClinic (`AGENTCLINIC_REPO`) |
| `uot` | Uncertainty of Thoughts | `uot/` | UoT, cloned by `uot/fetch_uot.sh` |
| `mediq` | MediQ-Expert | `mediq/` | mediQ (`MEDIQ_REPO`) |
| `meddxagent` | MEDDxAgent | `meddxagent/` | meddxagent (`MEDDXAGENT_REPO`) |
| `cod` | Chain-of-Diagnosis / DiagnosisGPT-34B | `cod/` | Chain-of-Diagnosis + GPU (HPC) |

The MoBayes row itself is `bench/mobayes.py`. The patient simulators every
row faces are in `bench/patients/` (AgentClinic, HT-sim) and
`scripts/core/benchmark_patientsim.py` (PatientSim).
