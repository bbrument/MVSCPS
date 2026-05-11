#!/bin/bash

# Copy MVSCPS results to eval_3d_datasets for evaluation
# Creates hierarchical directory structure: {method_prefix}/nbv-{views}/nbl-{lights}/nbit-{steps}/mesh.ply
#
# Override defaults with environment variables:
#   COPY_EXP_ROOT    - experiment root (default: /projects/m25115/exp/diligentmv)
#   COPY_EVAL_ROOT   - eval target root (default: /projects/m25115/eval_3d_datasets/dlmv/eval)
#   COPY_METHOD_PREFIX - method name prefix (default: mvscps)

set -e

# Configuration (overridable via env vars)
EXP_ROOT="${COPY_EXP_ROOT:-/projects/m25115/exp/diligentmv}"
EVAL_ROOT="${COPY_EVAL_ROOT:-/projects/m25115/eval_3d_datasets/dlmv/eval}"
METHOD_PREFIX="${COPY_METHOD_PREFIX:-mvscps}"

usage() {
  echo "Usage: $0 <obj_name> <num_views> <num_lights> <num_iterations> [mesh_resolution]"
  echo ""
  echo "Environment variables (optional):"
  echo "  COPY_EXP_ROOT       Experiment root (default: /projects/m25115/exp/diligentmv)"
  echo "  COPY_EVAL_ROOT      Eval target root (default: /projects/m25115/eval_3d_datasets/dlmv/eval)"
  echo "  COPY_METHOD_PREFIX  Method name prefix (default: mvscps)"
  echo ""
  echo "Example (DiligentMV GT light):"
  echo "  COPY_EXP_ROOT=/projects/m25115/exp/diligentmv_gt_light \\"
  echo "  COPY_METHOD_PREFIX=mvscps-gtlight \\"
  echo "  $0 bear 20 1 20000 512"
  exit 1
}

# Parse arguments
[[ $# -lt 4 ]] && usage

OBJ_NAME="$1"
NUM_VIEWS="$2"
NUM_LIGHTS="$3"
NUM_ITERATIONS="$4"
MESH_RES="${5:-512}"

# For eval target, use obj name directly (no mapping needed if already correct)
# Support both mapped (bear→bearPNG) and direct names (Bowl→Bowl)
declare -A OBJ_MAP=(
  ["bear"]="bearPNG"
  ["buddha"]="buddhaPNG"
  ["cow"]="cowPNG"
  ["pot2"]="pot2PNG"
  ["reading"]="readingPNG"
)

EVAL_OBJ="${OBJ_MAP[$OBJ_NAME]:-$OBJ_NAME}"

# Find the latest trial directory for this object that contains the requested mesh
OBJ_EXP_DIR="${EXP_ROOT}/${OBJ_NAME}"
if [[ ! -d "$OBJ_EXP_DIR" ]]; then
  echo "[ERROR] Experiment directory not found: $OBJ_EXP_DIR"
  exit 1
fi

# Search trials from newest to oldest, find one with the requested mesh resolution
LATEST_TRIAL=""
for trial in $(ls -1d "${OBJ_EXP_DIR}/@"* 2>/dev/null | sort -r); do
  if find "${trial}/save/mesh" -name "*mc${MESH_RES}_world_space.ply" 2>/dev/null | grep -q .; then
    LATEST_TRIAL="$trial"
    break
  fi
done

if [[ -z "$LATEST_TRIAL" || ! -d "$LATEST_TRIAL" ]]; then
  echo "[ERROR] No trial with mc${MESH_RES} mesh found in: $OBJ_EXP_DIR"
  exit 1
fi

echo "[INFO] Using trial: $(basename "$LATEST_TRIAL")"

# Find the mesh directory
MESH_DIR="${LATEST_TRIAL}/save/mesh"
if [[ ! -d "$MESH_DIR" ]]; then
  echo "[ERROR] Mesh directory not found: $MESH_DIR"
  exit 1
fi

# Find the latest iteration directory
LATEST_IT_DIR=$(ls -1d "${MESH_DIR}/it"* 2>/dev/null | sort -V | tail -1)
if [[ -z "$LATEST_IT_DIR" || ! -d "$LATEST_IT_DIR" ]]; then
  echo "[ERROR] No iteration directories found in: $MESH_DIR"
  exit 1
fi

echo "[INFO] Using iteration: $(basename "$LATEST_IT_DIR")"

# Find the world_space mesh with the specified resolution
WORLD_MESH=$(ls -1 "${LATEST_IT_DIR}/"*"mc${MESH_RES}_world_space.ply" 2>/dev/null | head -1)
if [[ -z "$WORLD_MESH" || ! -f "$WORLD_MESH" ]]; then
  echo "[ERROR] World space mesh not found with resolution ${MESH_RES}"
  echo "[INFO] Available meshes in ${LATEST_IT_DIR}:"
  ls -1 "${LATEST_IT_DIR}/"*.ply 2>/dev/null || echo "  (none)"
  exit 1
fi

echo "[INFO] Source mesh: $WORLD_MESH"

# Create target directory structure
METHOD_NAME="${METHOD_PREFIX}/nbv-${NUM_VIEWS}/nbl-${NUM_LIGHTS}/nbit-${NUM_ITERATIONS}"
TARGET_DIR="${EVAL_ROOT}/${EVAL_OBJ}/${METHOD_NAME}/results_raw"

mkdir -p "$TARGET_DIR"

# Symlink mesh (saves disk space)
TARGET_MESH="${TARGET_DIR}/mesh.ply"
ln -sf "$(realpath "$WORLD_MESH")" "$TARGET_MESH"

echo "[OK] Linked to: $TARGET_MESH -> $(realpath "$WORLD_MESH")"
echo "[INFO] Method: ${METHOD_NAME}"
