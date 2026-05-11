#!/bin/bash

# Pre-flight check: verify all prerequisites before submitting SLURM training jobs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

ERRORS=0

ok()  { echo "  ✓ $1"; }
err() { echo "  ✗ $1"; ERRORS=$((ERRORS + 1)); }

echo "================================================"
echo "DiligentMV Pre-flight Check"
echo "================================================"
echo ""

# 1. Check /projects/m25115/ is accessible and writable
echo "[1/4] Storage access..."
if [[ -d "/projects/m25115/" ]]; then
  ok "/projects/m25115/ exists"
  if [[ -w "/projects/m25115/" ]]; then
    ok "/projects/m25115/ is writable"
  else
    err "/projects/m25115/ is NOT writable"
  fi
else
  err "/projects/m25115/ does not exist or is not accessible"
fi
echo ""

# 2. Check processed data for all 5 objects
echo "[2/4] Processed data..."
OBJECTS="bear buddha pot2 cow reading"
for obj in ${OBJECTS}; do
  IMAGE_DIR="/projects/m25115/DiLiGenT-MV/${obj}/image"
  if [[ -d "${IMAGE_DIR}" ]]; then
    COUNT=$(ls -1 "${IMAGE_DIR}"/*.png 2>/dev/null | wc -l)
    if [[ "${COUNT}" -eq 1920 ]]; then
      ok "${obj}: ${COUNT} images"
    else
      err "${obj}: expected 1920 images, found ${COUNT}"
    fi
  else
    err "${obj}: image directory not found (${IMAGE_DIR})"
  fi
done
echo ""

# 3. Check virtual environment
echo "[3/4] Virtual environment..."
VENV_ACTIVATE="${PROJECT_ROOT}/venv-mvscps-py3.10/bin/activate"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  ok "venv-mvscps-py3.10/bin/activate exists"
else
  err "venv-mvscps-py3.10/bin/activate NOT found"
fi
echo ""

# 4. Check spack modules
echo "[4/4] Spack modules..."
if command -v spack >/dev/null 2>&1; then
  ok "spack command available"
  if spack find python@3.10 >/dev/null 2>&1; then
    ok "python@3.10 module found"
  else
    err "python@3.10 module NOT found in spack"
  fi
  if spack find cuda@11.8.0 >/dev/null 2>&1; then
    ok "cuda@11.8.0 module found"
  else
    err "cuda@11.8.0 module NOT found in spack"
  fi
else
  err "spack command not available"
fi
echo ""

# Summary
echo "================================================"
if [[ ${ERRORS} -eq 0 ]]; then
  echo "All checks passed. Ready to submit jobs."
  echo "  Run: slurm/test_single_object.sh     (dry-run test)"
  echo "  Run: slurm/submit_all_diligent.sh    (full training)"
else
  echo "${ERRORS} check(s) FAILED. Fix issues before submitting jobs."
fi
echo "================================================"

exit ${ERRORS}
