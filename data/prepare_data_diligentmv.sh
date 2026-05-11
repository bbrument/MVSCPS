#!/usr/bin/env bash

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate uv venv ('venv-mvscps-py3.10') if available; otherwise fall back to uv run later
if [[ -f "${ROOT_DIR}/venv-mvscps-py3.10/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${ROOT_DIR}/venv-mvscps-py3.10/bin/activate"
  USING_VENV=1
else
  echo "[WARN] venv-mvscps-py3.10 not found. Will use 'uv run' for Python steps."
  USING_VENV=0
fi

# Ensure gdown is available (install into the active venv if missing)
if ! command -v gdown >/dev/null 2>&1; then
  echo "[INFO] Installing gdown..."
  if [[ "${USING_VENV}" -eq 1 ]]; then
    python -m pip install -q gdown
  else
    uv run python -m pip install -q gdown
  fi
fi

# Download DiLiGenT-MV dataset (Google Drive)
FILE_ID="18dheWmAxCNaBpYoH3usuFeH9vGlhODvx"
ZIP_PATH="/projects/m25115/diligentmv.zip"
OUT_DIR="/projects/m25115/DiLiGenT-MV_origin"

echo "[INFO] Downloading DiLiGenT-MV to ${ZIP_PATH} ..."
gdown "${FILE_ID}" -O "${ZIP_PATH}"

echo "[INFO] Unzipping into ${OUT_DIR} ..."
mkdir -p "${OUT_DIR}"
unzip -o "${ZIP_PATH}" -d "${OUT_DIR}"

# Verify expected structure exists
VERIFY_FILE="${OUT_DIR}/DiLiGenT-MV/mvpmsData/bearPNG/view_01/001.png"
if [[ ! -f "${VERIFY_FILE}" ]]; then
  echo "[ERROR] Expected file not found after extraction: ${VERIFY_FILE}"
  echo "[ERROR] The zip file may be corrupt or have unexpected structure."
  exit 1
fi
echo "[OK] Extraction verified: expected structure found."

run_py() {
  if [[ ${USING_VENV} -eq 1 ]]; then
    python "$@"
  else
    uv run python "$@"
  fi
}

# Run preprocessing
echo "[INFO] Running preprocessing ..."

run_py ${SCRIPT_DIR}/preprocess_data_diligentmv.py \
  --root-dir "${OUT_DIR}/DiLiGenT-MV/mvpmsData" \
  --target-dir "/projects/m25115/DiLiGenT-MV"

# Remove zip only after preprocessing succeeds
rm -f "${ZIP_PATH}"

echo "[DONE] Data prepared at: /projects/m25115/DiLiGenT-MV"