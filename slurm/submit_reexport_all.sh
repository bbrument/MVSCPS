#!/bin/bash
# Submit mesh re-export jobs for all LUCES-MV objects (both dir and point)
# Usage: bash slurm/submit_reexport_all.sh [mesh_resolution]

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MESH_RES="${1:-512}"
OBJECTS="Bowl Buddha Bunny Cup Die Hippo House Owl Queen Squirrel"

echo "Submitting mesh re-export jobs (mc${MESH_RES})..."
echo ""

for LIGHT_TYPE in directional point; do
  if [[ "$LIGHT_TYPE" == "directional" ]]; then
    EXP_SUFFIX="lucesmv_dir"
  else
    EXP_SUFFIX="lucesmv_point"
  fi

  for OBJ in $OBJECTS; do
    # Find the trial directory
    OBJ_DIR="/projects/m25115/exp/${EXP_SUFFIX}/${OBJ}"
    TRIAL=$(ls -1d "${OBJ_DIR}/@"* 2>/dev/null | sort -r | head -1)
    if [[ -z "$TRIAL" ]]; then
      echo "[SKIP] No trial found for ${OBJ} (${LIGHT_TYPE})"
      continue
    fi
    TRIAL_NAME=$(basename "$TRIAL")

    CKPT="${TRIAL}/ckpt/last.ckpt"
    if [[ ! -f "$CKPT" ]]; then
      echo "[SKIP] No checkpoint for ${OBJ} (${LIGHT_TYPE})"
      continue
    fi

    JOB_NAME="reexp-${OBJ:0:3}-${LIGHT_TYPE:0:3}"
    JOB_ID=$(sbatch --job-name="$JOB_NAME" --parsable \
      slurm/reexport_mesh_lucesmv.sh "$OBJ" "$LIGHT_TYPE" "$TRIAL_NAME" "$MESH_RES")
    echo "  ${LIGHT_TYPE} ${OBJ}: job ${JOB_ID} (trial ${TRIAL_NAME})"
  done
done

echo ""
echo "All re-export jobs submitted (mc${MESH_RES})."
