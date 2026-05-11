#!/bin/bash
# Re-export meshes for experiments that completed without predict_mesh.
# Runs the predict phase only (from checkpoint) with mesh export.
#
# Usage: bash slurm/reexport_all_missing.sh [--dry-run]

set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true

OBJECTS="Bowl Buddha Bunny Cup Die Hippo House Owl Queen Squirrel"

# Check which experiments need mesh re-export
TOTAL=0
for suffix in dir_15l dir_1l point_15l point_1l; do
  EXP_ROOT="/projects/m25115/exp/lucesmv_${suffix}"
  [[ ! -d "$EXP_ROOT" ]] && continue

  # Determine light type from suffix
  if [[ "$suffix" == dir_* ]]; then
    LTYPE="directional"
  else
    LTYPE="point"
  fi
  NLIGHTS="${suffix##*_}"
  NLIGHTS="${NLIGHTS%l}"
  VL_INDEX="lucesmv_view_12_light_${NLIGHTS}"

  for OBJ in ${OBJECTS}; do
    OBJ_DIR="${EXP_ROOT}/${OBJ}"
    [[ ! -d "$OBJ_DIR" ]] && continue

    # Find latest trial with a checkpoint
    TRIAL=$(ls -1d "${OBJ_DIR}/@"* 2>/dev/null | sort -r | head -1)
    [[ -z "$TRIAL" ]] && continue

    CKPT="${TRIAL}/ckpt/last.ckpt"
    [[ ! -f "$CKPT" ]] && continue

    # Check if mesh already exists
    if find "${TRIAL}/save/mesh" -name "*world_space.ply" 2>/dev/null | grep -q .; then
      continue  # mesh already exported
    fi

    JOB_NAME="reexp-${suffix}-${OBJ}"
    LOG_PREFIX="slurm/logs/${JOB_NAME}"

    CMD="sbatch --job-name=${JOB_NAME} \
      --output=${LOG_PREFIX}_%j.log \
      --error=${LOG_PREFIX}_%j.err \
      --time=01:00:00 --mem=64G --gres=gpu:1 --cpus-per-task=4 \
      -p mesonet --account=m25115 \
      --wrap='cd $(pwd) && source venv-mvscps-py3.10/bin/activate && \
      spack load python@3.10 && spack load cuda@11.8.0 && \
      export PL_WEIGHTS_ONLY=0 WANDB_MODE=disabled TORCH_CUDA_ARCH_LIST=8.0 && \
      python launch.py +conf=lucesmv_native \
        conf.dataset.obj_name=${OBJ} \
        conf.exp.exp_path=${EXP_ROOT} \
        conf.exp.phase=predict \
        conf.exp.resume=true \
        \"conf.dataset.predict_targets=[predict_mesh]\" \
        conf.model.light.light_type=${LTYPE} \
        conf.dataset.train.view_light_index_fname=${VL_INDEX} \
        conf.dataset.val.view_light_index_fname=${VL_INDEX} \
        conf.dataset.predict_mesh.view_light_index_fname=${VL_INDEX} \
        conf.dataset.predict_mesh.gt_mesh_fpath=None'"

    echo "  [NEED REEXPORT] lucesmv_${suffix}/${OBJ}"
    if ! $DRY_RUN; then
      eval "${CMD}"
    fi
    TOTAL=$((TOTAL + 1))
  done
done

echo ""
echo "Total re-exports needed: ${TOTAL}"
$DRY_RUN && echo "(dry-run — no jobs submitted)"
