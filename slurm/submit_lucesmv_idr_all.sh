#!/bin/bash
# Submit all LUCES-MV IDR experiments: 4 variants × 10 objects = 40 jobs
#
# Variants:
#   - directional 15 lights (lucesmv_dir_15l)
#   - directional 1 light   (lucesmv_dir_1l)
#   - point 15 lights        (lucesmv_point_15l)
#   - point 1 light          (lucesmv_point_1l)
#
# Usage: bash slurm/submit_lucesmv_idr_all.sh [--dry-run]

set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true

OBJECTS="Bowl Buddha Bunny Cup Die Hippo House Owl Queen Squirrel"

# Variants: LIGHT_TYPE NUM_LIGHTS
declare -a VARIANTS=(
  "directional 15"
  "directional 1"
  "point 15"
  "point 1"
)

TOTAL=0
for variant in "${VARIANTS[@]}"; do
  read -r LTYPE NLIGHTS <<< "$variant"
  SUFFIX="${LTYPE:0:3}"  # dir or poi
  [[ "$LTYPE" == "directional" ]] && SUFFIX="dir" || SUFFIX="point"
  JOB_PREFIX="lmv-${SUFFIX}-${NLIGHTS}l"

  echo ""
  echo "================================================"
  echo "Submitting: ${LTYPE} light, ${NLIGHTS} lights"
  echo "  exp: lucesmv_${SUFFIX}_${NLIGHTS}l"
  echo "================================================"

  for OBJ in ${OBJECTS}; do
    JOB_NAME="${JOB_PREFIX}-${OBJ}"
    LOG_PREFIX="slurm/logs/${JOB_NAME}"

    CMD="LUCESMV_LIGHT_TYPE=${LTYPE} LUCESMV_NUM_LIGHTS=${NLIGHTS} sbatch \
      --job-name=${JOB_NAME} \
      --output=${LOG_PREFIX}_%j.log \
      --error=${LOG_PREFIX}_%j.err \
      slurm/run_lucesmv.sh ${OBJ}"

    if $DRY_RUN; then
      echo "  [DRY-RUN] ${CMD}"
    else
      echo "  Submitting ${OBJ}..."
      eval "${CMD}"
    fi
    TOTAL=$((TOTAL + 1))
  done
done

echo ""
echo "================================================"
echo "Total jobs: ${TOTAL}"
$DRY_RUN && echo "(dry-run mode — no jobs submitted)"
echo "================================================"
