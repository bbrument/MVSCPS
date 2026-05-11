#!/bin/bash

#=======================================================================
#SBATCH --job-name=dlmv_gt_light
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
#SBATCH --output=slurm/logs/dlmv_gt_light_%j.log
#SBATCH --error=slurm/logs/dlmv_gt_light_%j.err
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
VALID_OBJECTS="bear buddha pot2 cow reading"
if [[ ! " ${VALID_OBJECTS} " =~ " ${OBJ_NAME} " ]]; then
  echo "[ERROR] Invalid object: ${OBJ_NAME}"
  echo "[ERROR] Valid objects: ${VALID_OBJECTS}"
  exit 1
fi

echo "================================================"
echo "DiligentMV GT Fixed Lighting Training"
echo "Object: ${OBJ_NAME}"
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

echo "Starting GT light training for ${OBJ_NAME}..."
echo "Extra overrides: $*"
echo ""

PYTHON_EXIT_CODE=0
python launch.py +conf=diligentmv \
  conf.dataset.obj_name="${OBJ_NAME}" \
  conf.dataset.root_dir=/projects/m25115/DiLiGenT-MV \
  conf.exp.exp_path=/projects/m25115/exp/diligentmv_gt_light \
  conf.dataset.train.view_light_index_fname=view_20_light_1 \
  conf.dataset.val.view_light_index_fname=view_20_light_1 \
  conf.dataset.test.view_light_index_fname=view_20_light_1 \
  conf.model.light.use_gt_light=true \
  conf.model.light.freeze_light=true \
  "$@" || PYTHON_EXIT_CODE=$?

echo ""
if [[ ${PYTHON_EXIT_CODE} -ne 0 ]]; then
  echo "[WARN] Python exited with code ${PYTHON_EXIT_CODE} (likely PyVista visualization crash)"
  echo "[INFO] Continuing to copy mesh results anyway..."
fi
echo "================================================"
echo "GT light training completed for ${OBJ_NAME}"
echo "================================================"

# Auto-copy results to eval_3d_datasets
VIEW_LIGHT_FNAME="view_20_light_1"
for arg in "$@"; do
  if [[ "$arg" == conf.dataset.train.view_light_index_fname=* ]]; then
    VIEW_LIGHT_FNAME="${arg#*=}"
  fi
done

if [[ "$VIEW_LIGHT_FNAME" =~ view_([0-9]+)_light_([0-9]+) ]]; then
  NUM_VIEWS="${BASH_REMATCH[1]}"
  NUM_LIGHTS="${BASH_REMATCH[2]}"
else
  NUM_VIEWS="20"
  NUM_LIGHTS="1"
fi

NUM_ITERATIONS="20000"
for arg in "$@"; do
  if [[ "$arg" == conf.trainer.max_steps=* ]]; then
    NUM_ITERATIONS="${arg#*=}"
  fi
done

MESH_RES="512"
for arg in "$@"; do
  if [[ "$arg" == conf.model.geometry.isosurface.resolution=* ]]; then
    MESH_RES="${arg#*=}"
  fi
done

echo ""
echo "================================================"
echo "Copying results to eval_3d_datasets"
echo "  Parameters: nbv=${NUM_VIEWS}, nbl=${NUM_LIGHTS}, nbit=${NUM_ITERATIONS}, res=${MESH_RES}"
echo "================================================"

COPY_EXP_ROOT="/projects/m25115/exp/diligentmv_gt_light" \
COPY_EVAL_ROOT="/projects/m25115/eval_3d_datasets/dlmv/eval" \
COPY_METHOD_PREFIX="mvscps-gtlight" \
"${PROJECT_ROOT}/slurm/copy_results_to_eval.sh" "${OBJ_NAME}" "${NUM_VIEWS}" "${NUM_LIGHTS}" "${NUM_ITERATIONS}" "${MESH_RES}"
  echo "[OK] Results copied to eval_3d_datasets"
else
  echo "[WARN] Failed to copy results to eval_3d_datasets (non-fatal)"
fi
