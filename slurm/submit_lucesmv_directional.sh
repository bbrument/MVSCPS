#!/bin/bash

# Submit all 10 LUCES-MV training jobs with DIRECTIONAL light model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_DIR="/projects/m25115/eval_3d_datasets/lucesmv/data"
if [[ ! -d "${DATA_DIR}/Bowl/mvps" ]]; then
  echo "[ERROR] IDR data not found: ${DATA_DIR}/Bowl/mvps"
  echo "[ERROR] Run: python data/prepare_data_lucesmv_idr.py first."
  exit 1
fi

mkdir -p slurm/logs

echo "================================================"
echo "Submitting LUCES-MV DIRECTIONAL Light Training"
echo "Data: ${DATA_DIR}"
echo "Exp:  /projects/m25115/exp/lucesmv_dir"
echo "Eval: lucesmv/eval/*/mvscps-dir/nbv-12/nbl-15/nbit-20000"
echo "================================================"
echo ""

for obj in Bowl Buddha Bunny Cup Die Hippo House Owl Queen Squirrel; do
  JOB_ID=$(LUCESMV_LIGHT_TYPE=directional sbatch --parsable \
    --job-name="lmv-dir-${obj}" \
    --output="slurm/logs/lmv-dir-${obj}_%j.log" \
    --error="slurm/logs/lmv-dir-${obj}_%j.err" \
    --export=ALL,LUCESMV_LIGHT_TYPE=directional \
    slurm/run_lucesmv.sh "$obj" "$@")
  echo "Submitted ${obj} (directional): Job ID ${JOB_ID}"
done

echo ""
echo "================================================"
echo "All 10 LUCES-MV directional jobs submitted."
echo "Monitor: squeue -u $USER"
echo "================================================"
