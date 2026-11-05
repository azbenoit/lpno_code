"""Choose the scatter implementation by PROVING the CUDA op works, not by hasattr.

A CPU-only ``torch-scatter`` build still exposes ``torch_scatter.scatter``, imports
cleanly, reports the correct version and passes ``pip check``. The previous guard --
``hasattr(torch_scatter, 'scatter')`` -- is satisfied by such a build, so GAOT selected
it, announced "Using torch_scatter.scatter for aggregation", and then died on the first
CUDA forward at ``reduce="max"`` with "Not compiled with CUDA support". RIGNO needs only
scatter_add, which falls through to native torch, so it trained perfectly on the same
install: the box looks HALF working, which takes far longer to diagnose than a dead one
because the model, config, data and seat all get blamed first.

This is easy to hit on a fresh machine. ``torch-scatter`` is sdist-only and its
``setup.py`` compiles the CUDA extensions only when ``torch.utils.cpp_extension.CUDA_HOME``
resolves at build time; otherwise it silently takes the CPU branch and succeeds. The tell
on disk is four ``.so`` files in ``site-packages/torch_scatter/`` instead of eight, and a
rebuild that finishes in a second rather than minutes.

Falling back is safe rather than a silent downgrade, and this is measured, not assumed:

* CORRECTNESS. ``scatter_native`` implements max through
  ``scatter_reduce_(reduce="amax", include_self=False)``, EXACT on torch >= 1.12 -- not
  the approximate sum placeholder in its own else-branch, which is dead code on any torch
  this package supports. Confirmed against the real kernel on a 4090: an identical
  50-step GAOT run agrees to all six printed digits of ``mae_T`` at step 0 (the
  deterministic forward pass) between backends.
* SPEED. Deliberately NOT claimed here. This warning used to say "slower", which a
  per-step A/B contradicted: 45.95 ms native against 51.24 ms torch_scatter, ratio 0.90,
  n=2 each, measured by differencing 200- and 1000-step runs so imports and validators
  cancel. That was ONE graph size (latent_tokens [32,20,20], 90x40x40, 4 cases, batch 1)
  and the gap plausibly tracks torch version, since Tensor.scatter_reduce_ has had recent
  work while torch_scatter is a separate extension. A number that specific does not belong
  in a string that outlives the pin, and neither does the unmeasured claim it replaced.

So keeping the run alive with a loud warning beats killing it at step 1.
``use_attention`` defaults True in both ``model_domain.py`` and ``gaot_arch.py`` and every
shipped GAOT config sets it, so there is no configuration that dodges the max path.

Found while commissioning a new compute box, 2026-08-19.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple


def _default_probe(module: Any) -> None:
    """Run the exact op that a CPU-only build cannot: a CUDA segment max."""
    import torch

    module.scatter(
        torch.zeros(4, device="cuda"),
        torch.zeros(4, dtype=torch.long, device="cuda"),
        dim=0,
        dim_size=2,
        reduce="max",
    )


def resolve_scatter(
    torch_scatter_module: Any,
    *,
    cuda_available: bool,
    probe: Optional[Callable[[Any], None]] = None,
) -> Tuple[Callable[..., Any], str]:
    """Return ``(scatter_fn, backend_name)`` where backend is 'torch_scatter' or 'native'.

    The probe only runs when CUDA is actually present, so CPU-only hosts (the website
    stub runtime, CI) are never probed and never warned at.
    """
    import os

    from .scatter_native import scatter_native

    # Deliberate override, so the fallback branch can be exercised on a HEALTHY box.
    # Without it, the only way to watch scatter_native carry a real GAOT run is to
    # rebuild torch-scatter CPU-only on a working machine and restore it afterwards --
    # 20 minutes of degrading a commissioned box to test a code path. This makes it
    # one env var.
    if os.environ.get("PINO_FORCE_NATIVE_SCATTER", "").strip():
        print(
            "PINO_FORCE_NATIVE_SCATTER set: using native PyTorch scatter by request "
            "(numerically exact against the torch_scatter kernel).",
            flush=True,
        )
        return scatter_native, "native"

    if torch_scatter_module is None or not hasattr(torch_scatter_module, "scatter"):
        return scatter_native, "native"

    if not cuda_available:
        # Nothing to prove: the CPU kernels are the ones that will run.
        return torch_scatter_module.scatter, "torch_scatter"

    try:
        (probe or _default_probe)(torch_scatter_module)
    except Exception as exc:  # noqa: BLE001 - any failure here means "do not use it"
        print(
            "WARNING: torch_scatter imports but its CUDA kernels are missing "
            f"({type(exc).__name__}: {exc}). This is a CPU-only torch-scatter build; "
            "GAOT would die on the first forward pass at reduce='max' while RIGNO "
            "trains normally. Falling back to native PyTorch scatter, which is "
            "numerically exact against the torch_scatter kernel. "
            "To fix: rebuild with CUDA_HOME set, nvcc on PATH, and --no-cache-dir "
            "(a cached wheel reinstalls the broken build in about a second; a real "
            "CUDA compile takes minutes).",
            flush=True,
        )
        return scatter_native, "native"

    return torch_scatter_module.scatter, "torch_scatter"
