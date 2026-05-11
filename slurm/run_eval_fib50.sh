#!/bin/bash
#SBATCH --job-name=eval_fib50
#SBATCH -p mesonet
#SBATCH --account=m25115
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/eval_fib50_%j.log
#SBATCH --error=slurm/logs/eval_fib50_%j.err

# Chains eval_pipeline jobs for all 12 fib50 runs.
# This script copies meshes to eval dir then submits eval-pipeline via SLURM.
#
# Run after all training jobs:
#   sbatch --dependency=afterok:JID1:JID2:... slurm/run_eval_fib50.sh

set -e

if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

if command -v spack >/dev/null 2>&1; then
  spack load python@3.10 || true
fi
source venv-mvscps-py3.10/bin/activate

EVAL_VENV="/home/babrument/dev/MVSCPS/eval_pipeline/venv/bin/activate"
EVAL_CFG="/home/babrument/dev/MVSCPS/eval_pipeline/config/suzanne_fib50.yaml"
EXP_ROOT="exp/suzanne_point_colored_shadows_fib50/suzanne"
EVAL_ROOT="eval/suzanne_point_colored_shadows_fib50/suzanne"

STRATEGIES=(maxgray mingray medgray maxsobel 1l_front 1l_maxvar)
MODELS=(point dir)

echo "================================================"
echo "Copying meshes to eval directory"
echo "================================================"

mkdir -p "${EVAL_ROOT}"

for strat in "${STRATEGIES[@]}"; do
  for model in "${MODELS[@]}"; do
    METHOD="50v50l_learnlight_${model}_neuralbrdf_shadow_${strat}"
    TAG="_${METHOD}"
    # Find the trial dir
    TRIAL_DIR=$(ls -dt "${EXP_ROOT}/suzanne"*"${TAG}"* 2>/dev/null | head -1)
    if [[ -z "${TRIAL_DIR}" ]]; then
      echo "[WARN] No trial found for ${METHOD}"
      continue
    fi
    # Find the mesh (marching cubes 1024)
    MESH=$(find "${TRIAL_DIR}/save/mesh" -name "*mc1024_world_space.ply" 2>/dev/null | sort | tail -1)
    if [[ -z "${MESH}" ]]; then
      MESH=$(find "${TRIAL_DIR}/save/mesh" -name "*.ply" 2>/dev/null | sort | tail -1)
    fi
    if [[ -z "${MESH}" ]]; then
      echo "[WARN] No mesh found for ${METHOD} in ${TRIAL_DIR}"
      continue
    fi
    DEST="${EVAL_ROOT}/${METHOD}/results_raw"
    mkdir -p "${DEST}"
    ln -sf "$(realpath "${MESH}")" "${DEST}/mesh.ply"
    echo "  [OK] ${METHOD} → ${DEST}/mesh.ply"
  done
done

echo ""
echo "================================================"
echo "Submitting eval-pipeline jobs via SLURM"
echo "================================================"

source "${EVAL_VENV}"
cd eval_pipeline
eval-pipeline -c "${EVAL_CFG}" run --stages cleanup,evaluate,visualize
echo ""
echo "eval-pipeline jobs submitted."
