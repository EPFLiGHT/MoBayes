"""CoD / DiagnosisGPT-34B row.

DiagnosisGPT-34B (Wang et al. 2024, Yi-34B fine-tune over 9,604-disease
universe). Requires a GPU node + 70 GB checkpoint.

Local: not supported (no GPU + no checkpoint).
Remote (Clariden GH200): emits a parameterised sbatch script next to the
output dir and prints `sbatch` + `scp` instructions; the user submits and
later runs ``--remote-fetch`` to pull back results.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List


NAME = "cod"

# Set HPC_BASE to the cluster directory containing this repo (`bmbe/`),
# the Chain-of-Diagnosis checkout (`cod/`), and the DiagnosisGPT-34B
# checkpoint. HPC_HF_CACHE points to a writable HF cache (often a scratch FS).
_HPC_BASE = Path(os.environ.get("HPC_BASE", "/path/to/hpc/base"))
_HPC_HF_CACHE = os.environ.get("HPC_HF_CACHE", "/path/to/hf_cache")
_HPC_MODEL_DIR = os.environ.get("HPC_DIAGNOSISGPT_DIR", "/path/to/diagnosisgpt/model")
_HPC_AGENTCLINIC_REPO = os.environ.get("HPC_AGENTCLINIC_REPO", "/path/to/agentclinic")
_HPC_SLURM_ACCOUNT = os.environ.get("HPC_SLURM_ACCOUNT", "your_account")
_HPC_SSH_ALIAS = os.environ.get("HPC_SSH_ALIAS", "hpc")


def make_doctor(*args, **kwargs):
    raise NotImplementedError("cod is batch-mode + remote; uses run_batch.")


_SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=cod_{column_name}
#SBATCH --account={slurm_account}
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output={hpc_base}/logs/cod_{column_name}_%j.out
#SBATCH --error={hpc_base}/logs/cod_{column_name}_%j.err

set -e
export PATH="$HOME/.local/bin:$PATH"
cd {hpc_base}
source cod/.venv/bin/activate

set -a
source bmbe/hybrid-diagnostic-system/.env
set +a

export PYTHONPATH="$PWD/cod/Chain-of-Diagnosis:$PWD/bmbe/hybrid-diagnostic-system:$PYTHONPATH"
export AGENTCLINIC_REPO={hpc_agentclinic_repo}
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME={hpc_hf_cache}

mkdir -p {hpc_base}/runs/{run_name}

echo "=== Slurm $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
echo "Start: $(date)"

python {runner} \\
    --cod-repo {hpc_base}/cod/Chain-of-Diagnosis \\
    --model-dir {hpc_model_dir} \\
    --bmbe-root {hpc_base}/bmbe/hybrid-diagnostic-system \\
    {column_args} \\
    --out {hpc_base}/runs/{run_name}/results.jsonl \\
    --num-cases {num_cases} \\
    --max-inquiries 10 \\
    --mapper-model gpt-5.4-nano

echo "End: $(date)"
"""


def _column_args(column_name: str) -> tuple[str, str]:
    """Return (runner_path, --column-specific args) for the cod runner script."""
    bmbe_root = f"{_HPC_BASE}/bmbe/hybrid-diagnostic-system"
    if column_name == "ac":
        return (
            f"{bmbe_root}/scripts/baselines/run_cod_diagnosisgpt_ac.py",
            f"--cases {bmbe_root}/data/cases/agentclinic_a/selected_cases.jsonl "
            f"--diseases {bmbe_root}/data/kbs/agentclinic_a/diseases.csv",
        )
    if column_name == "ac_set2":
        return (
            f"{bmbe_root}/scripts/baselines/run_cod_diagnosisgpt_ac.py",
            f"--cases {bmbe_root}/data/cases/agentclinic_b/selected_cases.jsonl "
            f"--diseases {bmbe_root}/data/kbs/agentclinic_b/diseases.csv",
        )
    if column_name in ("patientsim", "htsim", "patientsim_v1", "htsim_v1"):
        # CoD on DDxPlus uses --profiles + --diseases (49 disease list)
        runner = (f"{bmbe_root}/scripts/baselines/run_cod_diagnosisgpt.py")
        profiles_by_col = {
            "patientsim":    f"{bmbe_root}/data/cases/ddxplus_patientsim_a/plain/profiles.json",
            "patientsim_v1": f"{bmbe_root}/data/cases/ddxplus_patientsim_b/plain/profiles.json",
            "htsim":         f"{bmbe_root}/data/cases/ddxplus_htsim_a/selected_cases.jsonl",
            "htsim_v1":      f"{bmbe_root}/data/cases/ddxplus_htsim_b/selected_cases.jsonl",
        }
        return runner, (f"--profiles {profiles_by_col[column_name]} "
                         f"--diseases {bmbe_root}/data/kbs/ddxplus/diseases.csv")
    raise NotImplementedError(f"cod column={column_name!r}")


def run_batch(*, cases: List[Dict[str, Any]], start_idx: int, column,
              doctor_model: str, patient_model: str, max_turns: int,
              api_key: str, out_dir: Path, row_kwargs: Dict[str, Any]) -> None:
    remote = row_kwargs.get("remote", "clariden")
    if remote != "clariden":
        sys.exit("cod row requires --row-kwargs '{\"remote\":\"clariden\"}' "
                 "(local DiagnosisGPT-34B inference not supported on Mac).")

    runner, column_args = _column_args(column.NAME)
    run_name = f"cod_{column.NAME}_{len(cases)}"
    sbatch_path = out_dir / f"clariden_{run_name}.sbatch"
    out_dir.mkdir(parents=True, exist_ok=True)
    sbatch_path.write_text(_SBATCH_TEMPLATE.format(
        column_name=column.NAME,
        run_name=run_name,
        runner=runner,
        column_args=column_args,
        num_cases=len(cases),
        hpc_base=_HPC_BASE,
        hpc_hf_cache=_HPC_HF_CACHE,
        hpc_model_dir=_HPC_MODEL_DIR,
        hpc_agentclinic_repo=_HPC_AGENTCLINIC_REPO,
        slurm_account=_HPC_SLURM_ACCOUNT,
    ))
    print(f"[cod] sbatch written: {sbatch_path}", flush=True)
    print()
    print(f"  scp {sbatch_path} {_HPC_SSH_ALIAS}:{_HPC_BASE}/")
    print(f"  ssh {_HPC_SSH_ALIAS} 'cd {_HPC_BASE} && "
          f"sbatch {sbatch_path.name}'")
    print()
    print(f"  # then poll, and pull back:")
    print(f"  scp {_HPC_SSH_ALIAS}:{_HPC_BASE}/runs/{run_name}/results.jsonl \\")
    print(f"      {out_dir}/results.jsonl")
    print()
    print("  Run `python -m bench --row cod --column {col} --post` after copy "
          "to convert results.jsonl → per-case JSONs.".format(col=column.NAME))
