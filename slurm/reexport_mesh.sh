#!/bin/bash
#SBATCH -p mesonet
#SBATCH --account=m25115
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Re-export mesh from an existing checkpoint.
# Usage: EXP_ROOT=... LIGHT_TYPE=... VL_INDEX=... sbatch slurm/reexport_mesh.sh <obj>

set -e
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

OBJ_NAME="${1:?Usage: sbatch $0 <obj_name>}"

if command -v spack >/dev/null 2>&1; then
  spack load python@3.10
  spack load cuda@11.8.0
fi
source venv-mvscps-py3.10/bin/activate

export PL_WEIGHTS_ONLY=0 WANDB_MODE=disabled WANDB_SILENT=true TORCH_CUDA_ARCH_LIST="8.0"

EXP_ROOT="${EXP_ROOT:?Set EXP_ROOT}"
LIGHT_TYPE="${LIGHT_TYPE:-point}"
VL_INDEX="${VL_INDEX:-lucesmv_view_12_light_15}"

# Find the latest trial directory
OBJ_EXP="${EXP_ROOT}/${OBJ_NAME}"
TRIAL_NAME=$(ls -1d "${OBJ_EXP}/@"* 2>/dev/null | sort -r | head -1 | xargs basename)
if [[ -z "$TRIAL_NAME" ]]; then
  echo "[ERROR] No trial found in ${OBJ_EXP}"
  exit 1
fi

echo "Re-exporting mesh: ${OBJ_NAME} from ${EXP_ROOT}, trial=${TRIAL_NAME}"

python launch.py +conf=lucesmv_native \
  conf.dataset.obj_name="${OBJ_NAME}" \
  conf.exp.exp_path="${EXP_ROOT}" \
  conf.exp.phase=predict \
  conf.exp.resume=true \
  conf.exp.trial_name="${TRIAL_NAME}" \
  'conf.dataset.predict_targets=["predict_mesh"]' \
  conf.model.light.light_type="${LIGHT_TYPE}" \
  conf.dataset.train.view_light_index_fname="${VL_INDEX}" \
  conf.dataset.val.view_light_index_fname="${VL_INDEX}" \
  conf.dataset.predict_mesh.view_light_index_fname="${VL_INDEX}" \
  conf.dataset.predict_mesh.gt_mesh_fpath=None
