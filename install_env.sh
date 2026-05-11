#!/usr/bin/env bash

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VENV_DIR="${VENV_DIR:-.venv-mvscps}"
REQ="${REQUIREMENTS_FILE:-requirements.txt}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"

export PERSIST_BASE="${PERSIST_BASE:-$HOME/.mvscps-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PERSIST_BASE/uv-cache}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-$PERSIST_BASE/xdg-data}"
mkdir -p "$UV_CACHE_DIR" "$XDG_DATA_HOME"

echo "[INFO] UV_CACHE_DIR = $UV_CACHE_DIR"
echo "[INFO] XDG_DATA_HOME = $XDG_DATA_HOME"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# 0) Load Python from spack if available (for development headers)
if command -v spack >/dev/null 2>&1; then
  echo "[INFO] Loading Python $PYTHON_VERSION from spack for dev headers..."
  spack load python@$PYTHON_VERSION 2>/dev/null && {
    SPACK_PYTHON=$(which python3)
    echo "[INFO] Spack Python: $SPACK_PYTHON"
  } || {
    echo "[WARN] Could not load Python from spack, using system Python"
  }
fi

# 1) ensure uv
mkdir -p "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  ( curl -fsSL https://astral.sh/uv/install.sh \
      | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh ) || true
  hash -r
fi

if ! command -v uv >/dev/null 2>&1; then
  mkdir -p "$PWD/bin"
  ( curl -fsSL https://astral.sh/uv/install.sh \
      | env UV_UNMANAGED_INSTALL="$PWD/bin" sh ) || true
  export PATH="$PWD/bin:$PATH"
  hash -r
fi

uv --version

create_or_fix_venv() {
  if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating venv at $VENV_DIR (python=$PYTHON_VERSION)..."
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
    return
  fi

  # Check if "broken" (Python not executable
  if ! "$VENV_DIR/bin/python" -V >/dev/null 2>&1; then
    echo "[WARN] Existing venv seems broken. Re-creating..."
    rm -rf "$VENV_DIR"
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
  fi
}

create_or_fix_venv

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
echo "[INFO] Activated $VENV_DIR (python: $(python -V 2>/dev/null || echo 'unknown'))"

# 3) CUDA toolchain via spack (if available)
if command -v spack >/dev/null 2>&1; then
  echo "[INFO] Loading CUDA 11.8 via spack..."
  spack load cuda@11.8.0 2>/dev/null || echo "[WARN] Could not load CUDA via spack"
  
  # Get CUDA installation path from spack
  CUDA_HOME=$(spack location -i cuda@11.8.0 2>/dev/null)
  if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ]; then
    export CUDA_HOME
    echo "[INFO] CUDA_HOME set to: $CUDA_HOME"
  fi
fi

# Fallback to /usr/local/cuda if not set by spack
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}"

# Set CUDA architectures for tiny-cuda-nn (common modern GPUs)
# 70: V100, 75: RTX 2080/T4, 80: A100, 86: RTX 3090, 89: RTX 4090, 90: H100
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-70;75;80;86}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6}"
echo "[INFO] TCNN_CUDA_ARCHITECTURES = $TCNN_CUDA_ARCHITECTURES"
echo "[INFO] TORCH_CUDA_ARCH_LIST = $TORCH_CUDA_ARCH_LIST"

# 4) install PyTorch first (cu118 wheels)
echo "[INFO] Installing PyTorch (CUDA 11.8 wheels)..."
uv pip install --index-url "$PYTORCH_INDEX_URL" \
  torch==2.5.1 torchvision==0.20.1

# 5) build tooling for source packages
echo "[INFO] Installing build tools..."
uv pip install -U "setuptools<81" wheel packaging cmake ninja

# 6) install the rest from PyPI (no index override)
if [ -f "$REQ" ]; then
  echo "[INFO] Installing the rest from $REQ..."
  echo "[INFO] Note: tiny-cuda-nn compilation may take several minutes..."
  
  # Install with better error handling
  if ! uv pip install --no-build-isolation -r "$REQ"; then
    echo ""
    echo "[ERROR] Installation failed!"
    echo "[HINT] If tiny-cuda-nn failed to build, check:"
    echo "  1. CUDA_HOME is set correctly: $CUDA_HOME"
    echo "  2. nvcc is in PATH: $(which nvcc 2>/dev/null || echo 'NOT FOUND')"
    echo "  3. TCNN_CUDA_ARCHITECTURES is set: $TCNN_CUDA_ARCHITECTURES"
    echo ""
    echo "You can retry with:"
    echo "  source $VENV_DIR/bin/activate"
    echo "  spack load cuda@11.8.0"
    echo "  export CUDA_HOME=\$(spack location -i cuda@11.8.0)"
    echo "  export TCNN_CUDA_ARCHITECTURES=\"70;75;80;86\""
    echo "  uv pip install --no-build-isolation -r $REQ"
    exit 1
  fi
fi

# ---- 7) quick verify ----
python - <<'PY'
import sys

def ok(m):   print(f"[OK]   {m}")
def warn(m): print(f"[WARN] {m}")

print(f"Python: {sys.version.split()[0]}")

# Torch
try:
    import torch
    ok(f"Torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
except Exception as e:
    warn(f"Torch import failed: {e}")

# VTK / PyVista
try:
    import vtk, pyvista as pv
    from pyvista import _vtk  # ensure internal vtk bridge is available
except Exception as e:
    warn(f"VTK/PyVista import failed: {e}")
else:
    ok(f"VTK {vtk.vtkVersion.GetVTKVersion()} | PyVista {pv.__version__}")
    ok("PyVista import: OK")

# Trimesh + Embree backend
try:
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    eng = m.ray
    mod, cls = type(eng).__module__, type(eng).__name__
    ok(f"Trimesh ray engine: {mod}.{cls}")
    if "ray_triangle" in mod:
        warn("Fallback to ray_triangle (no Embree backend).")
except Exception as e:
    warn(f"Trimesh/Embree check failed: {e}")

# nerfacc: import + check for compiled CUDA extension (best-effort)
try:
    import importlib, os, pathlib, nerfacc
    ok(f"nerfacc {getattr(nerfacc, '__version__', 'unknown')} | path: {os.path.dirname(nerfacc.__file__)}")

    compiled_ok, ext_name = False, None
    for name in ("nerfacc._C", "nerfacc.cuda", "nerfacc._nerfacc"):
        try:
            importlib.import_module(name)
            compiled_ok, ext_name = True, name
            break
        except Exception:
            pass
    if compiled_ok:
        ok(f"nerfacc CUDA extension import: {ext_name}")
    else:
        so_files = list(pathlib.Path(os.path.dirname(nerfacc.__file__)).rglob("*.so"))
        if so_files:
            ok(f"nerfacc compiled .so detected (e.g., {so_files[0].name})")
        else:
            warn("nerfacc compiled extensions not found; build may have failed.")
except Exception as e:
    warn(f"nerfacc import failed: {e}")

# tiny-cuda-nn (tinycudann): instantiate a small network and forward once
try:
    import tinycudann as tcnn, torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = tcnn.Network(
        n_input_dims=3,
        n_output_dims=16,
        network_config={
            "otype": "CutlassMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 16,
            "n_hidden_layers": 2
        }
    ).to(device)
    import torch
    x = torch.rand(128, 3, device=device)
    with torch.no_grad():
        y = net(x)
    ok(f"tinycudann forward OK on {device}: output shape {tuple(y.shape)}")
except Exception as e:
    warn(f"tinycudann test failed: {e}")
PY

echo "[DONE] To activate later: source $VENV_DIR/bin/activate"