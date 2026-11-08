# baselines/cod

Chain-of-Diagnosis with DiagnosisGPT-34B (Chen et al., 2024) as a baseline
(row `cod`). The model needs a GPU and the ~70 GB checkpoint, so the row
does not run it locally: it writes a Slurm script for the chosen column and
prints the `scp` / `sbatch` commands.

| File | Role |
|---|---|
| `row.py` | Bench row: writes the Slurm script (`python -m bench --row cod --column <col>`) |
| `run.py` | Runner, `--column patientsim / htsim / agentclinic`; `--posthoc-from-jsonl` re-scores a saved AgentClinic run without the model |
| `force_final.py` | AgentClinic post-loop step: map CoD's free-text diagnosis to the closed list, or ask an LLM to commit/abstain when CoD never diagnosed |

## Cluster layout expected by the generated script

| Env var | Default | What |
|---|---|---|
| `HPC_BASE` | `/path/to/hpc/base` | Base dir; holds `cod/Chain-of-Diagnosis` (+ `cod/.venv`), `logs/`, `runs/` |
| `HPC_REPO_DIR` | `$HPC_BASE/MoBayes` | This repository on the cluster (its `.env` is sourced) |
| `HPC_DIAGNOSISGPT_DIR` | `/path/to/diagnosisgpt/model` | DiagnosisGPT-34B checkpoint |
| `HPC_MEDDXAGENT_REPO` | `/path/to/meddxagent` | MEDDxAgent checkout (HT-sim patient) |
| `HPC_AGENTCLINIC_REPO` | `/path/to/agentclinic` | AgentClinic checkout (AC patient) |
| `HPC_HF_CACHE`, `HPC_SLURM_ACCOUNT`, `HPC_SSH_ALIAS` | placeholders | HF cache, Slurm account, SSH host alias |
