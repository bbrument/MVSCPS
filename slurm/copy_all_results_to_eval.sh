#!/bin/bash

# Copy all MVSCPS results to eval_3d_datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default parameters - override via command line
NUM_VIEWS="${1:-20}"
NUM_LIGHTS="${2:-1}"
NUM_ITERATIONS="${3:-20000}"
MESH_RES="${4:-512}"

echo "================================================"
echo "Copying all MVSCPS results to eval_3d_datasets"
echo "Parameters: nbv=${NUM_VIEWS}, nbl=${NUM_LIGHTS}, nbit=${NUM_ITERATIONS}, res=${MESH_RES}"
echo "================================================"
echo ""

OBJECTS="bear buddha cow pot2 reading"
FAILED=""

for obj in $OBJECTS; do
  echo "--- Processing ${obj} ---"
  if "${SCRIPT_DIR}/copy_results_to_eval.sh" "$obj" "$NUM_VIEWS" "$NUM_LIGHTS" "$NUM_ITERATIONS" "$MESH_RES"; then
    echo ""
  else
    echo "[WARN] Failed to copy ${obj}"
    FAILED="${FAILED} ${obj}"
    echo ""
  fi
done

echo "================================================"
if [[ -z "$FAILED" ]]; then
  echo "All objects copied successfully."
else
  echo "Failed objects:${FAILED}"
fi
echo "================================================"
