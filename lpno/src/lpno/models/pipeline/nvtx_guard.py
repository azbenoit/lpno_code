"""Let physicsnemo run on a CPU-only build of torch.

physicsnemo-sym 2.2 marks its work with NVTX ranges: every training step
(`Trainer._train_loop`, `Trainer.compute_gradients`), every graph forward
(`Graph.forward`, so validation and inference too), every constraint forward. On a
torch built without CUDA, `torch.cuda.nvtx` is a stub whose functions raise
"NVTX functions not installed", so a training with `PINO_ALLOW_CPU=1` died at its
first step although nothing else in the pipeline needs a GPU.

NVTX ranges are profiler annotations that nothing reads back, so on such a build
`range_push`/`range_pop` are replaced by no-ops for the rest of the process. On any
torch that has NVTX (every CUDA build, with or without a visible GPU) nothing is
touched. Called from `configure_device` once the run has been allowed onto the CPU.
"""

from __future__ import annotations

_PATCHED = ("range_push", "range_pop")


def _noop(*args, **kwargs) -> int:
    return 0


def nvtx_unavailable() -> bool:
    """True on a torch built without CUDA, whose `torch.cuda.nvtx` functions raise."""
    import torch

    return torch.version.cuda is None and not hasattr(torch._C, "_nvtx")


def make_nvtx_noop_on_cpu_build() -> bool:
    """Turn `torch.cuda.nvtx.range_push`/`range_pop` into no-ops on a CPU-only torch.

    Returns True when it patched (or had already patched), False when torch has NVTX.
    """
    if not nvtx_unavailable():
        return False
    import torch

    nvtx = torch.cuda.nvtx
    if all(getattr(nvtx, name) is _noop for name in _PATCHED):
        return True
    for name in _PATCHED:
        setattr(nvtx, name, _noop)
    print("[device] CPU-only torch build: physicsnemo's NVTX ranges are no-ops in this process",
          flush=True)
    return True
