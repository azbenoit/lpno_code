"""Physical spatial decimation of a FLOW-3D case, for strided derivations.

`PINO_LAZY_STRIDE` decimates *after* inflating the full-resolution arrays, so it
saves GPU memory and no loader time at all. Measured on the 300-case
random-path set (2026-07-28): a run sat at 2.5 s/iteration with the GPU at 0% and eight
dataloader workers pegged at 83-88% CPU, because every `__getitem__` inflates
4.77 GB/case before decimating, and the LRU hit rate over 195 cases / 39,911
frames is 0.653% -- each case is inflated ~102 times instead of once.

Striding into the *stored* arrays cuts the inflate itself (4.77 -> 1.19 GB/case).
The decimation here must match `npz_lazy_store` exactly -- same start-at-0
`[::step]` on each spatial axis -- or a model trained on a derived copy is not
comparable to one trained with the env var. `scripts/build_random_path_split.py`
in `lpno` produced `data/flow3D/random2579_ds_split` with the same
rule, header-verified bit-identical.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

#: Per-axis coordinate vectors, decimated with their own axis' step.
_AXIS_KEYS = ("x", "y", "z")

#: A field carries (nt, nx, ny, nz) and optionally a trailing component axis, so
#: the spatial slice always sits at dimensions 1..3 and never touches time.
_FIELD_MIN_NDIM = 4


def normalize_stride(stride: Any) -> tuple[int, int, int]:
    """Return a validated ``(sx, sy, sz)``.

    Rejects anything unusable rather than falling back to ``(1, 1, 1)``: a
    silent no-op would train at a resolution nobody asked for, and the failure
    is invisible in the metrics.
    """
    if isinstance(stride, (str, bytes)) or not isinstance(stride, Iterable):
        raise ValueError(f"stride must be 3 integers, got {stride!r}")
    try:
        values = tuple(int(v) for v in stride)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stride must be 3 integers, got {stride!r}") from exc
    if len(values) != 3 or any(v < 1 for v in values):
        raise ValueError(f"stride must be 3 integers >= 1, got {stride!r}")
    return values  # type: ignore[return-value]


def stride_arrays(arrays: Mapping[str, np.ndarray], stride: Any) -> dict[str, np.ndarray]:
    """Decimate the spatial axes of one case's arrays.

    Coordinate vectors take their own axis' step; fields take all three on
    dimensions 1..3; everything else (the time vector, per-case scalars such as
    power/speed, manifest-derived channels) passes through untouched.
    """
    sx, sy, sz = normalize_stride(stride)
    if (sx, sy, sz) == (1, 1, 1):
        return dict(arrays)

    steps = {"x": sx, "y": sy, "z": sz}
    out: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if key in _AXIS_KEYS and array.ndim == 1:
            out[key] = array[:: steps[key]]
        elif array.ndim >= _FIELD_MIN_NDIM:
            out[key] = array[:, ::sx, ::sy, ::sz]
        else:
            out[key] = array
    return out


def strided_grid_shape(shape: tuple[int, ...], stride: Any) -> tuple[int, ...]:
    """Shape a field of ``shape`` takes after ``stride``, without loading it."""
    sx, sy, sz = normalize_stride(stride)
    if len(shape) < _FIELD_MIN_NDIM:
        return tuple(shape)
    nt, nx, ny, nz = shape[:4]
    return (nt, len(range(0, nx, sx)), len(range(0, ny, sy)), len(range(0, nz, sz))) + tuple(shape[4:])
