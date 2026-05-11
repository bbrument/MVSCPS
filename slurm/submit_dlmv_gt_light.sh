#!/bin/bash

# Submit all 5 DiligentMV GT fixed lighting training jobs to SLURM

set -e

# Resolve project root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Verify data exists
DATA_DIR="/projects/m25115/DiLiGenT-MV"
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[ERROR] Data directory not found: ${DATA_DIR}"
  echo "[ERROR] Run data/prepare_data_diligentmv.sh first (on login node)."
  exit 1
fi

# Create logs directory if needed
mkdir -p slurm/logs

echo "================================================"
echo "Submitting DiligentMV GT Light Training Jobs"
echo "Data: ${DATA_DIR}"
echo "================================================"
echo ""

for obj in bear buddha pot2 cow reading; do
  JOB_ID=$(sbatch --parsable --job-name="dlmv-gt-${obj}" \
    --output="slurm/logs/dlmv-gt-${obj}_%j.log" \
    --error="slurm/logs/dlmv-gt-${obj}_%j.err" \
    slurm/run_dlmv_gt_light.sh "$obj" "$@")
  echo "Submitted ${obj}: Job ID ${JOB_ID}"
done

echo ""
echo "================================================"
echo "All 5 GT light jobs submitted."
echo "Monitor with: squeue -u $USER"
echo "================================================"
