#!/bin/bash
# Build the training environment on the cluster (no GPU needed). Submitted by
# reproducibility/rerun/submit.sh build-env, which passes the site values; nothing site-specific here.
#SBATCH --job-name=lpno-build-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
set -euo pipefail
: "${REMOTE_CODE:?}" "${REMOTE_MAMBA:?}" "${REMOTE_ENV:?}"
cd "$REMOTE_CODE"
echo "code at $(cat .deployed_commit 2>/dev/null || echo '?'), env -> $REMOTE_ENV"

TORCH_CUDA="${TORCH_CUDA:-cu126}"   # cu126 keeps Volta (sm_70); the cu128 wheels of torch 2.8 dropped it
export PIP_NO_CACHE_DIR=1           # several GB of wheels; cluster homes have small quotas
if [ ! -x "$REMOTE_ENV/bin/python" ]; then
  "$REMOTE_MAMBA" create -y -p "$REMOTE_ENV" -c conda-forge python=3.10.18 pip
fi
PY="$REMOTE_ENV/bin/python"
# The same pinned recipe as on any other machine (tools/setup_env.py): torch from the index of
# this build, every other package at environment.yml's pins, torch-scatter, physicsnemo-sym.
"$PY" tools/setup_env.py --in-place --torch "$TORCH_CUDA"
"$PY" - <<'EOF'
import torch, torch_scatter, physicsnemo
print("torch", torch.__version__, "CUDA build", torch.version.cuda)
print("architectures", torch.cuda.get_arch_list() or "(none listed on a node without a GPU)")
print("physicsnemo", physicsnemo.__version__, "| torch_scatter ok")
import lpno; print("lpno at", lpno.__file__)
import physicsnemo.sym
from lpno.models import NS_Energy_laser_PINO, NS_Energy_laser_PINO_eval  # noqa: F401
print("training and evaluation entry points import")
EOF
echo "environment ready: $REMOTE_ENV"
