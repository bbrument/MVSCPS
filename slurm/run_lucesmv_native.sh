#!/bin/bash

#=======================================================================
#SBATCH --job-name=lmv-native
#SBATCH -p mesonet
#SBATCH --account=m25115
#=======================================================================
# RESOURCES
#=======================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#=======================================================================
# LOGS
#=======================================================================
#SBATCH --output=slurm/logs/lmv-native_%j.log
#SBATCH --error=slurm/logs/lmv-native_%j.err
#=======================================================================

set -e

# Resolve project root
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
  if [[ -n "${SCRIPT_DIR}" && -d "${SCRIPT_DIR}" ]]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  else
    echo "[ERROR] Cannot determine project root directory"
    exit 1
  fi
fi
cd "${PROJECT_ROOT}"

# Parse object name (required first argument)
OBJ_NAME="${1:?Usage: sbatch $0 <obj_name> [extra hydra overrides...]}"
shift

# Validate object name
VALID_OBJECTS="Bowl Buddha Bunny Cup Die Hippo House Owl Queen Squirrel"
if [[ ! " ${VALID_OBJECTS} " =~ " ${OBJ_NAME} " ]]; then
  echo "[ERROR] Invalid object: ${OBJ_NAME}"
  echo "[ERROR] Valid objects: ${VALID_OBJECTS}"
  exit 1
fi

# Determine light type and exp suffix from env or overrides
# LUCESMV_LIGHT_TYPE can be "directional" or "point"
LIGHT_TYPE="${LUCESMV_LIGHT_TYPE:-point}"
if [[ "$LIGHT_TYPE" == "directional" ]]; then
  EXP_SUFFIX="lucesmv_native_dir"
  METHOD_PREFIX="mvscps-native-dir"
else
  EXP_SUFFIX="lucesmv_native_point"
  METHOD_PREFIX="mvscps-native-point"
fi

echo "================================================"
echo "LUCES-MV Native Loader Training (${LIGHT_TYPE} light)"
echo "Object: ${OBJ_NAME}"
echo "Mesh resolution: 1024 (marching cubes)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Working dir: $(pwd)"
echo "================================================"
echo ""

# Load environment
if command -v spack >/dev/null 2>&1; then
  spack load python@3.10 || { echo "[ERROR] Failed to load python@3.10 from spack"; exit 1; }
  spack load cuda@11.8.0 || { echo "[ERROR] Failed to load cuda@11.8.0 from spack"; exit 1; }
else
  echo "[WARN] spack not available, assuming modules are already loaded"
fi

if [[ ! -f venv-mvscps-py3.10/bin/activate ]]; then
  echo "[ERROR] Virtual environment not found: venv-mvscps-py3.10/bin/activate"
  exit 1
fi
source venv-mvscps-py3.10/bin/activate

# Diagnostics
echo "Environment:"
echo "  Python: $(python -V)"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo ""

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU Information:"
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
  echo ""
fi

# Environment variables
export PL_WEIGHTS_ONLY=0
export WANDB_MODE=disabled
export WANDB_SILENT=true
export TORCH_CUDA_ARCH_LIST="8.0"

# Verify native data exists
DATA_DIR="/projects/m25115/LucesMV_processed/${OBJ_NAME}"
if [[ ! -f "${DATA_DIR}/camera_params.json" ]]; then
  echo "[ERROR] Native preprocessed data not found: ${DATA_DIR}/camera_params.json"
  echo "[ERROR] Run preprocessing first: sbatch slurm/preprocess_lucesmv.sh"
  exit 1
fi

echo "Starting training for ${OBJ_NAME} (${LIGHT_TYPE} light, native loader)..."
echo "Data dir: ${DATA_DIR}"
echo "Extra overrides: $*"
echo ""

PYTHON_EXIT_CODE=0
python launch.py +conf=lucesmv_native \
  conf.dataset.obj_name="${OBJ_NAME}" \
  conf.exp.exp_path=/projects/m25115/exp/${EXP_SUFFIX} \
  conf.model.light.light_type="${LIGHT_TYPE}" \
  "$@" || PYTHON_EXIT_CODE=$?

echo ""
if [[ ${PYTHON_EXIT_CODE} -ne 0 ]]; then
  echo "[WARN] Python exited with code ${PYTHON_EXIT_CODE}"
fi
echo "================================================"
echo "Training completed for ${OBJ_NAME} (${LIGHT_TYPE}, native)"
echo "================================================"

# Auto-copy results to eval_3d_datasets
NUM_VIEWS="12"
NUM_LIGHTS="15"
NUM_ITERATIONS="20000"
MESH_RES="1024"
for arg in "$@"; do
  if [[ "$arg" == conf.trainer.max_steps=* ]]; then
    NUM_ITERATIONS="${arg#*=}"
  fi
  if [[ "$arg" == conf.model.geometry.isosurface.resolution=* ]]; then
    MESH_RES="${arg#*=}"
  fi
done

echo ""
echo "================================================"
echo "Copying results to eval_3d_datasets"
echo "  Method: ${METHOD_PREFIX}/nbv-${NUM_VIEWS}/nbl-${NUM_LIGHTS}/nbit-${NUM_ITERATIONS}"
echo "================================================"

COPY_EXP_ROOT="/projects/m25115/exp/${EXP_SUFFIX}" \
COPY_EVAL_ROOT="/projects/m25115/eval_3d_datasets/lucesmv/eval" \
COPY_METHOD_PREFIX="${METHOD_PREFIX}" \
"${PROJECT_ROOT}/slurm/copy_results_to_eval.sh" "${OBJ_NAME}" "${NUM_VIEWS}" "${NUM_LIGHTS}" "${NUM_ITERATIONS}" "${MESH_RES}" \
  && echo "[OK] Results copied to eval_3d_datasets" \
  || echo "[WARN] Failed to copy results to eval_3d_datasets (non-fatal)"
