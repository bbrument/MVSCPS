#!/bin/bash

#=======================================================================
#SBATCH --job-name=lmv-reexp
#SBATCH -p mesonet
#SBATCH --account=m25115
#=======================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#=======================================================================
#SBATCH --output=slurm/logs/lmv-reexp_%j.log
#SBATCH --error=slurm/logs/lmv-reexp_%j.err
#=======================================================================

set -e

# Resolve project root
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

# Parse arguments
OBJ_NAME="${1:?Usage: sbatch $0 <obj_name> <light_type> <trial_name> [mesh_res]}"
LIGHT_TYPE="${2:?Usage: sbatch $0 <obj_name> <light_type> <trial_name> [mesh_res]}"
TRIAL_NAME="${3:?Usage: sbatch $0 <obj_name> <light_type> <trial_name> [mesh_res]}"
MESH_RES="${4:-512}"

if [[ "$LIGHT_TYPE" == "directional" ]]; then
  EXP_SUFFIX="lucesmv_dir"
  METHOD_PREFIX="mvscps-dir"
else
  EXP_SUFFIX="lucesmv_point"
  METHOD_PREFIX="mvscps-point"
fi

CKPT_PATH="/projects/m25115/exp/${EXP_SUFFIX}/${OBJ_NAME}/${TRIAL_NAME}/ckpt/last.ckpt"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "[ERROR] Checkpoint not found: $CKPT_PATH"
  exit 1
fi

echo "================================================"
echo "LUCES-MV Mesh Re-export (${LIGHT_TYPE})"
echo "Object: ${OBJ_NAME}"
echo "Trial: ${TRIAL_NAME}"
echo "Checkpoint: ${CKPT_PATH}"
echo "================================================"

# Load environment
if command -v spack >/dev/null 2>&1; then
  spack load python@3.10 || true
  spack load cuda@11.8.0 || true
fi
source venv-mvscps-py3.10/bin/activate

export PL_WEIGHTS_ONLY=0
export WANDB_MODE=disabled
export WANDB_SILENT=true
export TORCH_CUDA_ARCH_LIST="8.0"

echo "Mesh resolution: ${MESH_RES}"

# Run test phase (mesh export only, no GT eval to avoid ray-mesh hang)
# Disable PyVista rendering (no X11 display on compute nodes)
PYTHON_EXIT=0
python launch.py +conf=lucesmv \
  conf.exp.phase=test \
  conf.dataset.obj_name="${OBJ_NAME}" \
  conf.exp.exp_path=/projects/m25115/exp/${EXP_SUFFIX} \
  conf.exp.trial_name="${TRIAL_NAME}" \
  conf.exp.resume="${CKPT_PATH}" \
  conf.model.light.light_type="${LIGHT_TYPE}" \
  conf.model.geometry.isosurface.resolution="${MESH_RES}" \
  conf.dataset.test.gt_mesh_fpath=None || PYTHON_EXIT=$?

# Check mesh was saved (PyVista visualization may crash without X11, that's OK)
MESH_FILE="/projects/m25115/exp/${EXP_SUFFIX}/${OBJ_NAME}/${TRIAL_NAME}/save/mesh/it20000/it20000_mc${MESH_RES}_world_space.ply"
if [[ -f "$MESH_FILE" ]]; then
  echo "================================================"
  echo "Mesh re-exported: $(du -h "$MESH_FILE" | cut -f1)"
  echo "================================================"
else
  echo "[ERROR] Mesh file not found: $MESH_FILE"
  exit 1
fi

# Copy results to eval
NUM_VIEWS="12"
NUM_LIGHTS="15"
NUM_ITERATIONS="20000"

COPY_EXP_ROOT="/projects/m25115/exp/${EXP_SUFFIX}" \
COPY_EVAL_ROOT="/projects/m25115/eval_3d_datasets/lucesmv/eval" \
COPY_METHOD_PREFIX="${METHOD_PREFIX}" \
"${PROJECT_ROOT}/slurm/copy_results_to_eval.sh" "${OBJ_NAME}" "${NUM_VIEWS}" "${NUM_LIGHTS}" "${NUM_ITERATIONS}" "${MESH_RES}" \
  && echo "[OK] Results copied to eval_3d_datasets" \
  || echo "[WARN] Failed to copy results (non-fatal)"
