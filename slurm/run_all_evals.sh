#!/bin/bash

# Run eval-pipeline on all completed experiments
# Launched as SLURM dependency after all training jobs complete

#SBATCH --job-name=eval-all
#SBATCH -p mesonet
#SBATCH --account=m25115
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/eval-all_%j.log
#SBATCH --error=slurm/logs/eval-all_%j.err

set -e

# Load environment
if command -v spack >/dev/null 2>&1; then
  spack load python@3.10 || true
fi
source /home/babrument/dev/MVSCPS/venv-mvscps-py3.10/bin/activate

EVAL_DIR="/home/babrument/dev/eval_dataset/eval_pipeline"

echo "================================================"
echo "Running all evaluations"
echo "================================================"

# --- Exp 1: DLMV GT light ---
echo ""
echo "=== Exp 1: DLMV GT Light ==="
cd "${EVAL_DIR}"
eval-pipeline -c config/dlmv.yaml run --stages cleanup,evaluate 2>&1 || echo "[WARN] DLMV eval had errors"

# --- Exp 2-5: LUCES-MV (all methods) ---
echo ""
echo "=== Exp 2-5: LUCES-MV (all methods) ==="
eval-pipeline -c config/lucesmv.yaml run --stages cleanup,evaluate 2>&1 || echo "[WARN] LUCES-MV eval had errors"

# --- Aggregate results ---
echo ""
echo "=== Aggregating DLMV ==="
eval-pipeline -c config/dlmv.yaml aggregate 2>&1 || echo "[WARN] DLMV aggregate had errors"

echo ""
echo "=== Aggregating LUCES-MV ==="
eval-pipeline -c config/lucesmv.yaml aggregate 2>&1 || echo "[WARN] LUCES-MV aggregate had errors"

echo ""
echo "================================================"
echo "All evaluations complete!"
echo "================================================"
