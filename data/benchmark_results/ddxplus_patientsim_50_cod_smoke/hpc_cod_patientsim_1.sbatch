#!/bin/bash
#SBATCH --job-name=cod_patientsim
#SBATCH --account=your_account
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/hpc/base/logs/cod_patientsim_%j.out
#SBATCH --error=/path/to/hpc/base/logs/cod_patientsim_%j.err

set -e
export PATH="$HOME/.local/bin:$PATH"
cd /path/to/hpc/base
source cod/.venv/bin/activate

set -a
source bmbe/hybrid-diagnostic-system/.env
set +a

export PYTHONPATH="$PWD/cod/Chain-of-Diagnosis:$PWD/bmbe/hybrid-diagnostic-system:$PYTHONPATH"
export AGENTCLINIC_REPO=/path/to/agentclinic
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/path/to/hf_cache

mkdir -p /path/to/hpc/base/runs/cod_patientsim_1

echo "=== Slurm $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
echo "Start: $(date)"

python /path/to/hpc/base/bmbe/hybrid-diagnostic-system/scripts/baselines/run_cod_diagnosisgpt.py \
    --cod-repo /path/to/hpc/base/cod/Chain-of-Diagnosis \
    --model-dir /path/to/diagnosisgpt/model \
    --bmbe-root /path/to/hpc/base/bmbe/hybrid-diagnostic-system \
    --profiles /path/to/hpc/base/bmbe/hybrid-diagnostic-system/data/cases/ddxplus_patientsim_a/plain/profiles.json --diseases /path/to/hpc/base/bmbe/hybrid-diagnostic-system/data/kbs/ddxplus/diseases.csv \
    --out /path/to/hpc/base/runs/cod_patientsim_1/results.jsonl \
    --num-cases 1 \
    --max-inquiries 10 \
    --mapper-model gpt-5.4-nano

echo "End: $(date)"
