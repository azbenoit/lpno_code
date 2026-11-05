#!/bin/bash
# A short GPU job: the environment imports and torch runs a kernel on the card.
#SBATCH --job-name=lpno-check-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
: "${REMOTE_ENV:?}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$REMOTE_ENV/bin/python" - <<'EOF'
import torch, torch_scatter
print("torch", torch.__version__, "CUDA", torch.version.cuda, "arch list", torch.cuda.get_arch_list())
d = torch.device("cuda")
print("device", torch.cuda.get_device_name(d), "capability", torch.cuda.get_device_capability(d))
x = torch.randn(2048, 2048, device=d)
print("matmul ok", float((x @ x).sum().item()) == float((x @ x).sum().item()))
src = torch.arange(6.0, device=d); idx = torch.tensor([0, 0, 1, 1, 2, 2], device=d)
print("torch_scatter on GPU", torch_scatter.scatter_add(src, idx).tolist())
EOF
