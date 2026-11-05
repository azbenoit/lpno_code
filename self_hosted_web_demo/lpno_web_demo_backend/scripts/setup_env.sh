#!/usr/bin/env bash
# Create (or update) the conda environment the inference server runs in.
#
#   scripts/setup_env.sh              # env name "lpfno_demo"
#   LPFNO_ENV=myenv scripts/setup_env.sh
#   LPFNO_CUDA=cpu scripts/setup_env.sh   # CPU-only torch (steady page only is fast enough on CPU)
#
# The install ORDER matters and is the reason this script exists; each step was
# found by installing from scratch on a clean machine:
#   1. torch and torchvision first, from the PyTorch index, at the versions that go
#      together (pyproject.toml pins torch==2.8.*). Without torchvision here, PhysicsNeMo's
#      timm pulls the newest one, which replaces torch; the pin then reinstalls PyPI's
#      default build, whatever LPFNO_CUDA asked for, and the torch_scatter wheel below no
#      longer matches it. The final check refuses that outcome.
#   2. nvidia-physicsnemo-sym ships as an sdist whose build imports torch and always
#      compiles a CUDA kernel, so plain pip fails on a CPU-only torch and on a GPU
#      machine without a CUDA toolkit. lpno/scripts/install_physicsnemo_sym.py installs
#      it on any machine, building the kernel (needed for fp16 training only) when
#      CUDA_HOME is set. This package goes in with --no-build-isolation after it.
#   3. torch_scatter is imported by the vendored model code (ext/lpno)
#      and has no PyPI wheel for CUDA: it comes from the PyG index for the exact
#      torch build, which is why it cannot be a pyproject dependency.
set -euo pipefail

ENV_NAME="${LPFNO_ENV:-lpfno_demo}"
TORCH_VERSION="${LPFNO_TORCH:-2.8.0}"
TORCHVISION_VERSION="${LPFNO_TORCHVISION:-0.23.0}"
CUDA_TAG="${LPFNO_CUDA:-cu128}"          # cu128 | cu126 (Volta GPUs) | cpu
PHYSICSNEMO_VERSION="${LPFNO_PHYSICSNEMO:-1.2.0}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  # conda is usually a shell function; fall back to the standard install location
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    echo "conda not found; install Miniconda first (https://docs.conda.io/en/latest/miniconda.html)" >&2
    exit 1
  fi
fi
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
PY=python

$PY -m pip install --upgrade pip
$PY -m pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
$PY -m pip install "Cython>=3" "numpy>=1.24,<3" setuptools wheel
$PY -m pip install --no-build-isolation "nvidia-physicsnemo==${PHYSICSNEMO_VERSION}"
$PY "${HERE}/../../lpno/scripts/install_physicsnemo_sym.py"
$PY -m pip install --no-build-isolation -e "${HERE}[test]"
$PY -m pip install "hydra-core>=1.3"
$PY -m pip install torch_scatter -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

echo
echo "Verifying imports..."
LPFNO_CUDA_TAG="$CUDA_TAG" $PY - <<'PY'
import importlib
import os
missing = []
for m in ("fastapi", "uvicorn", "pydantic_settings", "numpy", "scipy", "matplotlib", "pyvista", "vtk",
          "yaml", "omegaconf", "torch", "physicsnemo", "physicsnemo.sym", "hydra", "torch_scatter", "wandb", "sklearn"):
    try:
        importlib.import_module(m)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{m}: {type(exc).__name__}: {exc}")
import torch
print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if not torch.__version__.endswith("+" + os.environ["LPFNO_CUDA_TAG"]):
    missing.append(f"torch is {torch.__version__}, but this environment asked for {os.environ['LPFNO_CUDA_TAG']}")
if missing:
    print("MISSING:\n  " + "\n  ".join(missing))
    raise SystemExit(1)
import app.main  # noqa: F401  -- the server itself
print("ok: every runtime import resolves; app.main imports")
PY

echo
echo "Environment '${ENV_NAME}' is ready. Activate it with:  conda activate ${ENV_NAME}"
