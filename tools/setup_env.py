#!/usr/bin/env python3
"""One command for a working, pinned Python environment for lpno (and aiida-pino).

A plain `pip install -e aiida-pino -e lpno` resolves to whatever is newest on PyPI
(torch 2.14 and nvidia-physicsnemo 1.3 on 2026-10-07, against the 2.8.0 / 1.2.0 the published
models were trained with), never installs nvidia-physicsnemo-sym or torch-scatter, and leaves
`lpno.models` unimportable. This installs, in order:

  1. torch 2.8.0 and torchvision 0.23.0 from the PyTorch index of the chosen build
     (`cpu`; `cu126` for Volta GPUs such as the V100; `cu128` for Ampere and newer);
  2. lpno (and, with --with-aiida, aiida-core 2.8.0 and aiida-pino with its
     console extra), every dependency held to the pins of lpno/environment.yml
     except the three that aiida-core 2.8 holds lower (AIIDA_RELAXES);
  3. torch-scatter from the matching PyG wheel index, and nvidia-physicsnemo-sym through
     lpno/scripts/install_physicsnemo_sym.py;

then imports everything once. Run it with a Python 3.10 interpreter (the pins are for 3.10):

    python3.10 tools/setup_env.py --prefix ~/envs/lpno --torch cpu --with-aiida   # new venv
    python tools/setup_env.py --in-place --torch cu126                             # active env

On a machine with a fragile or slow disk, note that a CUDA build writes 6-8 GB of wheels;
the CPU build with --with-aiida writes about 3 GB (2.7 GB measured).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TORCH = {"torch": "2.8.0", "torchvision": "0.23.0"}
# Pins tied to one CUDA build come with the torch wheels of the chosen index.
CUDA_BUILD = re.compile(r"^(torch|torchvision|triton|nvidia-[a-z0-9-]+-cu12)==")
# --with-aiida: the version that wrote the deposit's provenance archive. It holds jedi<0.19,
# pytz~=2021.1 and importlib-metadata~=6.0; environment.yml was exported from the training
# environment without aiida-core and carries newer ones, so those three pins give way to it.
AIIDA_CORE = "aiida-core==2.8.0"
AIIDA_RELAXES = {"jedi", "pytz", "importlib-metadata"}


def environment_pins(environment_yml: Path) -> list[str]:
    pins, in_pip = [], False
    for raw in environment_yml.read_text().splitlines():
        line = raw.strip()
        if line == "- pip:":
            in_pip = True
            continue
        if in_pip and line.startswith("- "):
            spec = line[2:].split("#", 1)[0].strip()
            if spec and not CUDA_BUILD.match(spec):
                pins.append(spec)
    return pins


def pin_name(spec: str) -> str:
    return re.split(r"[=<>!~ \[;]", spec, maxsplit=1)[0].strip().lower().replace("_", "-")


def run(python: str, *args: str, env: dict | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([python, *args], check=True, env=env)


def load_own_openssl_first(python: str) -> None:
    """Import `ssl` at interpreter start, so the interpreter's own OpenSSL loads first.

    A conda or micromamba Python carries a newer OpenSSL than RHEL 9 (3.6 against 3.0.7 on the
    cluster checked, 2026-10-08). If anything resolves a user name before `ssl` is imported, as
    `verdi presto` does, the system's NSS module loads the system libcrypto.so.3, and `import
    ssl` then fails with "version `OPENSSL_3.3.0' not found". A `.pth` line runs at start-up.
    """
    site = subprocess.run([python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                          check=True, capture_output=True, text=True).stdout.strip()
    Path(site, "00-load-own-openssl.pth").write_text("import ssl\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--prefix", type=Path, help="create (or reuse) a venv here")
    where.add_argument("--in-place", action="store_true", help="install into the running interpreter's environment")
    parser.add_argument("--torch", choices=["cpu", "cu126", "cu128"], default="cpu",
                        help="PyTorch build: cpu, cu126 (Volta, e.g. V100) or cu128 (Ampere and newer)")
    parser.add_argument("--with-aiida", action="store_true", help="also install aiida-pino with its console extra")
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 10):
        print(f"needs Python 3.10 (the pins of environment.yml are for 3.10.18); this is {sys.version.split()[0]}.\n"
              "Get one with conda/micromamba (`micromamba create -p ENV python=3.10`) and run this with it.",
              file=sys.stderr)
        return 2
    if args.in_place:
        python = sys.executable
    else:
        if not (args.prefix / "bin" / "python").exists():
            venv.create(args.prefix, with_pip=True)
        python = str(args.prefix / "bin" / "python")

    run(python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    run(python, "-m", "pip", "install", *(f"{k}=={v}" for k, v in TORCH.items()),
        "--index-url", f"https://download.pytorch.org/whl/{args.torch}")
    pins = environment_pins(REPO / "lpno" / "environment.yml")
    if args.with_aiida:
        pins = [spec for spec in pins if pin_name(spec) not in AIIDA_RELAXES]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(pins) + "\n")
        constraints = handle.name
    env = {**os.environ, "PIP_CONSTRAINT": constraints}
    packages = ["-e", str(REPO / "lpno")]
    if args.with_aiida:
        packages += [AIIDA_CORE, "-e", f"{REPO / 'aiida-pino'}[web]"]
    run(python, "-m", "pip", "install", *packages, env=env)
    run(python, "-m", "pip", "install", "torch-scatter==2.1.2",
        "-f", f"https://data.pyg.org/whl/torch-{TORCH['torch']}+{args.torch}.html", env=env)
    run(python, str(REPO / "lpno" / "scripts" / "install_physicsnemo_sym.py"), env=env)
    load_own_openssl_first(python)

    check = ("import torch, torch_scatter, physicsnemo, physicsnemo.sym, lpno.models.pipeline\n"
             "print('torch', torch.__version__, '| physicsnemo', physicsnemo.__version__, '| lpno imports')\n")
    if args.with_aiida:
        check += "import aiida, aiida_pino.cli\nprint('aiida-core', aiida.__version__, '| aiida-pino CLI imports')\n"
    run(python, "-c", check)
    os.unlink(constraints)
    if not args.in_place:
        print(f"\nready: source {args.prefix}/bin/activate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
