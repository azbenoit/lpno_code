# Backend image of the self-hosted web demo. Build from the REPOSITORY ROOT, so the image can carry the lpno
# package next to the app (Dockerfile.dockerignore keeps the context to those two folders):
#   docker build -f self_hosted_web_demo/lpno_web_demo_backend/docker/Dockerfile -t lpno-demo-backend .
#   docker run --gpus all -p 8000:8000 -v "$PWD/self_hosted_web_demo/lpno_web_demo_backend/data:/app/data" \
#     -v lpno-demo-outputs:/app/outputs lpno-demo-backend
# physicsnemo-sym expects a real CUDA toolkit at build time.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PYVISTA_OFF_SCREEN=true \
    PYVISTA_JUPYTER_BACKEND=static \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        python3 \
        python3-dev \
        python3-pip \
        python-is-python3 \
        libegl1 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY self_hosted_web_demo/lpno_web_demo_backend/pyproject.toml self_hosted_web_demo/lpno_web_demo_backend/README.md ./
COPY self_hosted_web_demo/lpno_web_demo_backend/app ./app
COPY lpno ./ext/lpno

# Same order as scripts/setup_env.sh, for the same reasons (see its header):
# torch pinned first, then PhysicsNeMo + sym without build isolation, then
# torch_scatter from the PyG wheel index for exactly that torch.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install "Cython>=3" "numpy>=1.24,<3" \
    && python -m pip install --no-build-isolation "nvidia-physicsnemo==1.2.0" "nvidia-physicsnemo-sym==2.2.0" \
    && python -m pip install --no-build-isolation . \
    && python -m pip install "hydra-core>=1.3" \
    && python -m pip install torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# The served assets are not in the image: mount the populated data/ directory
# (python scripts/assets.py fetch zenodo) at /app/data, and a writable
# /app/outputs for the neighbour-graph cache.
VOLUME ["/app/data", "/app/outputs"]

EXPOSE 8000

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
