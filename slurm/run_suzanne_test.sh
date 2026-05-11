#!/bin/bash
#SBATCH --job-name=suz_test
#SBATCH -p mesonet
#SBATCH --account=m25115
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/suzanne_test_%j.log
#SBATCH --error=slurm/logs/suzanne_test_%j.err

# Run test phase on a completed training experiment.
# Usage: sbatch --dependency=afterok:JOBID slurm/run_suzanne_test.sh <config_name> <exp_path> <tag_pattern> [extra overrides]
# Example: sbatch --dependency=afterok:12345 slurm/run_suzanne_test.sh suzanne_point exp/suzanne_point 1l_gt_af_maxcos

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_NAME="${1:?Usage: $0 <config_name> <exp_path> <tag_pattern> [overrides...]}"
EXP_PATH="${2:?Missing exp_path}"
TAG_PATTERN="${3:?Missing tag_pattern}"
shift 3

# Find the latest trial directory matching the tag
EXP_NAME=$(grep -oP "name:\s*\K\S+" "configs/conf/${CONFIG_NAME}.yaml" | head -1)
TRIAL_DIR=$(ls -dt "${EXP_PATH}/${EXP_NAME}/"*"${TAG_PATTERN}"* 2>/dev/null | head -1)

if [[ -z "${TRIAL_DIR}" ]]; then
  echo "[ERROR] No trial found matching ${EXP_PATH}/${EXP_NAME}/*${TAG_PATTERN}*"
  exit 1
fi

CKPT="${TRIAL_DIR}/ckpt/last.ckpt"
if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CKPT}"
  exit 1
fi

TRIAL_NAME=$(basename "${TRIAL_DIR}")
echo "================================================"
echo "Suzanne Test Phase"
echo "Config: ${CONFIG_NAME}"
echo "Trial: ${TRIAL_NAME}"
echo "Checkpoint: ${CKPT}"
echo "================================================"

if command -v spack >/dev/null 2>&1; then
  spack load python@3.10 || true
  spack load cuda@11.8.0 || true
fi
source venv-mvscps-py3.10/bin/activate

export PL_WEIGHTS_ONLY=0 WANDB_MODE=disabled WANDB_SILENT=true TORCH_CUDA_ARCH_LIST="8.0"

python launch.py +conf="${CONFIG_NAME}" \
  conf.exp.phase=test \
  conf.exp.exp_path="${EXP_PATH}" \
  conf.exp.trial_name="${TRIAL_NAME}" \
  conf.exp.resume=last.ckpt \
  "$@"

echo "Test phase completed"
