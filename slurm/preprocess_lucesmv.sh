#!/bin/bash

#=======================================================================
#SBATCH --job-name=lmv-preproc
#SBATCH -p mesonet
#SBATCH --account=m25115
#=======================================================================
# RESOURCES (CPU only, no GPU needed)
#=======================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#=======================================================================
# LOGS
#=======================================================================
#SBATCH --output=slurm/logs/lmv-preproc_%j.log
#SBATCH --error=slurm/logs/lmv-preproc_%j.err
#=======================================================================

set -e

# Resolve project root
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

echo "================================================"
echo "LUCES-MV Native Preprocessing"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Working dir: $(pwd)"
echo "================================================"
echo ""

# Load environment
if command -v spack >/dev/null 2>&1; then
  spack load python@3.10 || { echo "[ERROR] Failed to load python@3.10 from spack"; exit 1; }
else
  echo "[WARN] spack not available, assuming modules are already loaded"
fi

if [[ ! -f venv-mvscps-py3.10/bin/activate ]]; then
  echo "[ERROR] Virtual environment not found: venv-mvscps-py3.10/bin/activate"
  exit 1
fi
source venv-mvscps-py3.10/bin/activate

echo "Python: $(python -V)"
echo ""

# Verify raw data exists
RAW_DATA="/projects/m25115/LucesMV/calibrated"
if [[ ! -d "${RAW_DATA}/Bowl" ]]; then
  echo "[ERROR] Raw calibrated data not found: ${RAW_DATA}/Bowl"
  exit 1
fi

TARGET_DIR="/projects/m25115/LucesMV_processed"

echo "Running preprocessing..."
echo "  Raw data: ${RAW_DATA}"
echo "  Target:   ${TARGET_DIR}"
echo ""

python data/preprocess_data_lucesmv.py \
  --data-root "${RAW_DATA}" \
  --target-dir "${TARGET_DIR}"

echo ""
echo "================================================"
echo "Preprocessing complete."
echo "================================================"

# Quick sanity check: print O2W_scale for Bowl
python -c "
import json
with open('${TARGET_DIR}/Bowl/camera_params.json') as f:
    d = json.load(f)
print('Sanity check (Bowl):')
print(f'  O2W_scale: {d[\"O2W_scale\"]:.4f}')
print(f'  O2W_translation: {d[\"O2W_translation\"]}')
"
