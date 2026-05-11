#!/bin/bash
#SBATCH -p mesonet
#SBATCH --account=m25115
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# DiligentMV training with configurable number of lights.
# Usage: DLMV_NUM_LIGHTS=15 sbatch slurm/run_dlmv.sh <obj_name>

set -e
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

OBJ_NAME="${1:?Usage: sbatch $0 <obj_name>}"
shift

VALID_OBJECTS="bear buddha cow pot2 reading"
if [[ ! " ${VALID_OBJECTS} " =~ " ${OBJ_NAME} " ]]; then
  echo "[ERROR] Invalid object: ${OBJ_NAME}. Valid: ${VALID_OBJECTS}"
  exit 1
fi

NUM_LIGHTS="${DLMV_NUM_LIGHTS:-32}"
VL_INDEX="view_20_light_${NUM_LIGHTS}"
EXP_SUFFIX="diligentmv_dir_${NUM_LIGHTS}l"

echo "================================================"
echo "DiligentMV Training (directional, ${NUM_LIGHTS} lights)"
echo "Object: ${OBJ_NAME}"
echo "================================================"

if command -v spack >/dev/null 2>&1; then
  spack load python@3.10
  spack load cuda@11.8.0
fi
source venv-mvscps-py3.10/bin/activate

export PL_WEIGHTS_ONLY=0 WANDB_MODE=disabled WANDB_SILENT=true TORCH_CUDA_ARCH_LIST="8.0"

PYTHON_EXIT_CODE=0
python launch.py +conf=diligentmv \
  conf.dataset.obj_name="${OBJ_NAME}" \
  conf.exp.exp_path=/projects/m25115/exp/${EXP_SUFFIX} \
  conf.exp.test_after_train=false \
  'conf.dataset.predict_targets=["predict_mesh","predict_brdf","predict_relighting"]' \
  conf.dataset.train.view_light_index_fname="${VL_INDEX}" \
  conf.dataset.val.view_light_index_fname="${VL_INDEX}" \
  conf.dataset.test.view_light_index_fname="${VL_INDEX}" \
  conf.dataset.predict_mesh.view_light_index_fname="${VL_INDEX}" \
  conf.dataset.predict_mesh.gt_mesh_fpath=None \
  "$@" || PYTHON_EXIT_CODE=$?

echo "Training completed (exit=$PYTHON_EXIT_CODE)"
