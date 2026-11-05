"""The pinned `ext/lpno` submodule and the checkpoint plumbing around it.

Shared by the steady adapter (`adapter_pino`) and the time-dependent adapters
(`adapter_randompath`, `adapter_straightpath`); before this module each carried
its own copy of the repo-root lookup, the `sys.path` insert, the graph-cache
directory resolution and the strict checkpoint load.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def submodule_root() -> Path:
    """Where the model code (`lpno`) lives. Three layouts, in order:

    1. ``LPFNO_PINO_SRC`` -- an explicit path to a ``lpno`` checkout;
    2. ``ext/lpno`` -- the pinned git submodule of a standalone clone;
    3. ``../lpno`` -- the sibling checkout, which is what the demo
       sees inside the combined public repository (no submodule there).
    The first that exists wins; with none, the pip-installed package is the
    fallback (see ``ensure_submodule_on_path``).
    """
    import os

    override = os.environ.get("LPFNO_PINO_SRC", "").strip()
    candidates = ([Path(override).expanduser()] if override else []) + [
        repo_root() / "ext" / "lpno",
        repo_root().parent / "lpno",
    ]
    for c in candidates:
        try:
            if (c / "src" / "lpno").is_dir():
                return c
        except OSError:      # unreadable override path: treat as absent
            continue
    return candidates[0]


def submodule_src() -> Path:
    return submodule_root() / "src"


def ensure_submodule_on_path() -> Path:
    """Put the submodule's ``src`` first on ``sys.path`` (idempotent) and return it.

    The adapters import ``lpno.*`` straight from the checkout. A
    deployment without the checkout but with ``lpno`` pip-installed
    is accepted as-is (``src`` is returned unchanged); only when neither exists
    does this raise, so stub mode and the numpy-only preset/GT readers keep
    working without ``ext/``.
    """
    src = submodule_src()
    if not src.exists():
        if importlib.util.find_spec("lpno") is None:
            raise FileNotFoundError(f"Missing submodule src path: {src}")
        return src
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def submodule_commit() -> str | None:
    """HEAD of the pinned submodule, for /healthz; None when git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(submodule_root()), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def graph_cache_dir(settings: Any) -> Path:
    """Absolute, existing directory for the RIGNO/GAOT neighbour-graph cache
    (``settings.path_rigno_graph_cache_dir``, repo-relative when not absolute)."""
    cache_dir = Path(settings.path_rigno_graph_cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = repo_root() / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def extract_state_dict(ckpt: Any) -> Any:
    """The tensors inside whatever ``torch.load`` returned (a bare state dict,
    or one wrapped under a conventional key)."""
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "model", "state_dict", "net", "network", "weights"):
            if key in ckpt:
                return ckpt[key]
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt
        for value in ckpt.values():
            if isinstance(value, dict) and any(hasattr(x, "shape") for x in value.values()):
                return value
    return ckpt


def load_checkpoint_strict(net: Any, checkpoint_path: Path, torch: Any) -> None:
    """Load ``checkpoint_path`` into ``net`` and REFUSE any key mismatch.

    A tolerated mismatch is not a warning, it is a silently wrong model: any
    tensor in ``missing`` keeps its RANDOM INIT and the endpoint serves
    confident nonsense. The live trigger is an architecture default flipping
    under a stored checkpoint -- e.g. use_geoembed_encoder true -> false
    (2026-07-28), after which a pre-flip checkpoint rebuilt from a post-flip
    config loses geo_enc.* and enc_recovery.* and would have loaded "fine".
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = extract_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} does not match the architecture rebuilt "
            f"from its config: missing={missing} unexpected={unexpected}"
        )


def find_hydra_config_path(run_dir: str | Path) -> Path | None:
    """``.hydra/config.yaml`` in the run dir, else its parent or grandparent
    (checkpoints exported from a nested outputs tree)."""
    root = Path(run_dir)
    for candidate in (
        root / ".hydra" / "config.yaml",
        root.parent.parent / ".hydra" / "config.yaml",
        root.parent / ".hydra" / "config.yaml",
    ):
        if candidate.is_file():
            return candidate
    return None
