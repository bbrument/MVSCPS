#!/bin/bash

# Submit LUCES-MV training (native loader) with DIRECTIONAL light for all 10 objects
# Preprocessing is submitted first, training jobs depend on it.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Verify raw data for preprocessing
RAW_DATA="/projects/m25115/LucesMV/calibrated"
if [[ ! -d "${RAW_DATA}/Bowl" ]]; then
  echo "[ERROR] Raw calibrated data not found: ${RAW_DATA}/Bowl"
  exit 1
fi

mkdir -p slurm/logs

# Submit preprocessing job first
echo "================================================"
echo "Submitting LUCES-MV Native Preprocessing"
echo "================================================"

PREPROC_JOB_ID=$(sbatch --parsable \
  --job-name="lmv-preproc" \
  --output="slurm/logs/lmv-preproc_%j.log" \
  --error="slurm/logs/lmv-preproc_%j.err" \
  slurm/preprocess_lucesmv.sh)
echo "Preprocessing job: ${PREPROC_JOB_ID}"

echo ""
echo "================================================"
echo "Submitting LUCES-MV Native DIRECTIONAL Light Training"
echo "  Depends on preprocessing job: ${PREPROC_JOB_ID}"
echo "  Data: /projects/m25115/LucesMV_processed"
echo "  Exp:  /projects/m25115/exp/lucesmv_native_dir"
echo "  Eval: lucesmv/eval/*/mvscps-native-dir/nbv-12/nbl-15/nbit-20000"
echo "  Mesh: mc1024"
echo "================================================"
echo ""

for obj in Bowl Buddha Bunny Cup Die Hippo House Owl Queen Squirrel; do
  JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${PREPROC_JOB_ID} \
    --job-name="lmv-ndir-${obj}" \
    --output="slurm/logs/lmv-ndir-${obj}_%j.log" \
    --error="slurm/logs/lmv-ndir-${obj}_%j.err" \
    --export=ALL,LUCESMV_LIGHT_TYPE=directional \
    slurm/run_lucesmv_native.sh "$obj" "$@")
  echo "Submitted ${obj} (native-directional): Job ID ${JOB_ID}"
done

echo ""
echo "================================================"
echo "All 10 LUCES-MV native directional jobs submitted."
echo "Monitor: squeue -u $USER"
echo "================================================"
