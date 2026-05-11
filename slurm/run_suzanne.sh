#!/bin/bash

#=======================================================================
#SBATCH --job-name=suzanne
#SBATCH -p mesonet
#SBATCH --account=m25115
#=======================================================================
# RESOURCES
#=======================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#=======================================================================
# LOGS
#=======================================================================
#SBATCH --output=slurm/logs/suzanne_%j.log
#SBATCH --error=slurm/logs/suzanne_%j.err
#=======================================================================

# Generic Suzanne training script.
# Usage: sbatch slurm/run_suzanne.sh <config_name> [hydra overrides...]
# Example: sbatch slurm/run_suzanne.sh suzanne_dir conf.model.use_shadow=false

set -e

# Resolve project root
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

CONFIG_NAME="${1:?Usage: sbatch $0 <config_name> [hydra overrides...]}"
shift

echo "================================================"
echo "Suzanne Training"
echo "Config: ${CONFIG_NAME}"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Working dir: $(pwd)"
echo "Overrides: $*"
echo "================================================"

# Load environment
if command -v spack >/dev/null 2>&1; then
  spack load python@3.10 || true
  spack load cuda@11.8.0 || true
fi
source venv-mvscps-py3.10/bin/activate

echo "Python: $(python -V)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

export PL_WEIGHTS_ONLY=0 WANDB_MODE=disabled WANDB_SILENT=true TORCH_CUDA_ARCH_LIST="8.0"

PYTHON_EXIT_CODE=0
python launch.py +conf="${CONFIG_NAME}" "$@" || PYTHON_EXIT_CODE=$?

echo ""
echo "Training completed (exit=${PYTHON_EXIT_CODE})"
