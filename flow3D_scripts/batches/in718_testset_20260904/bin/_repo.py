"""Where the flow3D_scripts checkout is, from wherever this batch directory sits.

The loop ran from ``runs/batches/in718_loop/`` (run outputs are untracked) and is
archived under ``batches/in718_loop/`` (tracked). ``Path(__file__).parents[N]``
differs between the two, so walk up to the checkout that holds the runner instead
of counting levels. ``RUNS`` is where live solves land in either case.
"""
from pathlib import Path


def _repo() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "python" / "run_rectangular_case.py").is_file():
            return p
    raise RuntimeError(f"no flow3D_scripts checkout above {__file__}")


REPO = _repo()
RUNS = REPO / "runs"
MATERIALS = REPO / "materials"
