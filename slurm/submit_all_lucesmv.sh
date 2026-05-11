#!/bin/bash

# Submit all 10 LUCES-MV training jobs to SLURM

set -e

# Resolve project root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Verify data exists
DATA_DIR="/projects/m25115/LucesMV_processed"
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[ERROR] Preprocessed data directory not found: ${DATA_DIR}"
  echo "[ERROR] Run: python data/preprocess_data_lucesmv.py first (on login node)."
  exit 1
fi

# Create logs directory if needed
mkdir -p slurm/logs

echo "================================================"
echo "Submitting LUCES-MV Training Jobs"
echo "Data: ${DATA_DIR}"
echo "================================================"
echo ""

for obj in Bowl Buddha Bunny Cup Die Hippo House Owl Queen Squirrel; do
  JOB_ID=$(sbatch --parsable --job-name="lucesmv-${obj}" \
    --output="slurm/logs/lucesmv-${obj}_%j.log" \
    --error="slurm/logs/lucesmv-${obj}_%j.err" \
    slurm/run_lucesmv.sh "$obj" "$@")
  echo "Submitted ${obj}: Job ID ${JOB_ID}"
done

echo ""
echo "================================================"
echo "All 10 LUCES-MV jobs submitted."
echo "Monitor with: squeue -u $USER"
echo "================================================"
