#!/bin/bash

# Dry-run test: submit a single object (bear) with reduced steps

set -e

# Resolve project root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Create logs directory if needed
mkdir -p slurm/logs

echo "Submitting dry-run test job (bear, 100 steps)..."

JOB_ID=$(sbatch --parsable --job-name="diligentmv-test-bear" --time=01:00:00 \
  --output="slurm/logs/diligentmv-test-bear_%j.log" \
  --error="slurm/logs/diligentmv-test-bear_%j.err" \
  slurm/run_diligent_single_light.sh bear \
  conf.trainer.max_steps=100 \
  conf.checkpoint.every_n_train_steps=50 \
  conf.exp.predict_after_train=false \
  conf.exp.test_after_train=false)

echo "Submitted dry-run test: Job ID ${JOB_ID}"
echo "Monitor with: tail -f slurm/logs/diligentmv-test-bear_${JOB_ID}.log"
