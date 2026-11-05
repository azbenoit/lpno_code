#!/usr/bin/env python3
"""Install nvidia-physicsnemo-sym into the current environment, on any machine.

The package's setup.py always compiles one CUDA extension, the fused update kernel of
its fp16 loss scaler (physicsnemo/sym/amp.py). So `pip install nvidia-physicsnemo-sym`
fails with "CUDA_HOME environment variable is not set" on a GPU machine without a
CUDA toolkit (pip's torch wheels bring the CUDA runtime, not nvcc), and with a
TypeError on any CPU-only torch. lpno needs the package everywhere (the
Hydra entry points, model construction and datasets are built on it) but needs the
kernel only for fp16 training: models/ns_energy_pipeline.py imports it inside a
try/except, and every published config trains in fp32.

This fetches the sdist from PyPI, checks its sha256 against PyPI's record, and
builds the kernel when CUDA_HOME is set and torch is a CUDA build; otherwise it
installs the pure-Python package. Run it with the environment's interpreter, after
torch is installed:

    python lpno/scripts/install_physicsnemo_sym.py
"""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

VERSION = "2.2.0"


def main() -> int:
    meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/nvidia-physicsnemo-sym/{VERSION}/json"))
    sdist = next(u for u in meta["urls"] if u["packagetype"] == "sdist")
    tmp = tempfile.mkdtemp(prefix="physicsnemo_sym_")
    archive = os.path.join(tmp, sdist["filename"])
    urllib.request.urlretrieve(sdist["url"], archive)
    digest = hashlib.sha256(open(archive, "rb").read()).hexdigest()
    if digest != sdist["digests"]["sha256"]:
        print(f"sha256 mismatch for {sdist['filename']}: got {digest}", file=sys.stderr)
        return 1
    with tarfile.open(archive) as t:
        t.extractall(tmp)
    src = os.path.join(tmp, f"nvidia_physicsnemo_sym-{VERSION}")
    setup_py = os.path.join(src, "setup.py")
    text = open(setup_py).read()
    old = "ext_modules=[cuda_extension()],"
    if old not in text:
        print("setup.py has changed upstream; review this script before using it", file=sys.stderr)
        return 1
    text = text.replace("import glob\n", "import glob\nimport os\n", 1).replace(
        old, 'ext_modules=([cuda_extension()] if os.environ.get("CUDA_HOME") and torch.version.cuda else []),')
    open(setup_py, "w").write(text)
    import torch  # noqa: E402  (installed before this script runs, by design)
    with_kernel = bool(os.environ.get("CUDA_HOME") and torch.version.cuda)
    print(f"physicsnemo-sym {VERSION}: {'building' if with_kernel else 'without'} the fp16 scaler kernel")
    # Dependencies first, WITH build isolation: one of them builds from source and needs
    # Cython, which `--no-build-isolation` would require already present (it failed exactly
    # so in a fresh venv, 2026-10-07). Only the package itself needs torch at build time.
    # PIP_CONSTRAINT (e.g. the env's pins, see tools/setup_env.py) applies to both calls.
    from email.parser import Parser

    requires = Parser().parsestr(open(os.path.join(src, "PKG-INFO")).read()).get_all("Requires-Dist") or []
    if requires and subprocess.call([sys.executable, "-m", "pip", "install", *requires]) != 0:
        return 1
    return subprocess.call([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", src])


if __name__ == "__main__":
    sys.exit(main())
