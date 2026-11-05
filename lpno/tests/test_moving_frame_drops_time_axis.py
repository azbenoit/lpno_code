"""A moving frame with avg_window>1 must REMOVE the time axis, not leave T=1.

`load_consolidated_npz`'s contract: "the moving frame trims X and (if
avg_window_size>1) removes the time axis, matching the VTK path." The arch layers
depend on it -- a surviving T=1 reaches a 2D FNO as (B,1,C,X,Z) and trips
`AssertionError: Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for
2D FNO`.

Regressed 2026-08-20 in a540be1 by one character: `[:, -1]` (indexing, drops the
axis) became `[:, -1:]` (slicing, keeps it as size 1). It was invisible for five
days because every real training reads through `NpzDirStore`, which the same commit
notes "was already correct"; the eager path here has exactly one consumer, the CI
GPU smoke, so the regression showed up only as a red pipeline.

That is the reason for this test: the eager path needs a check that does not depend
on a GPU being free.
"""
import numpy as np

from lpno.custom_utils.npz_datasource import _apply_moving_frame


def _var(B=1, T=6, C=1, nX=12, nY=4, nZ=4):
    return {"Temperature": np.arange(B * T * C * nX * nY * nZ, dtype=np.float32)
            .reshape(B, T, C, nX, nY, nZ)}


def _centres(B=1, T=6):
    # beam creeping along x, inside the domain
    return np.stack([np.stack([np.linspace(0.0, 1e-4, T), np.zeros(T)], axis=-1)
                     for _ in range(B)], axis=0)


def _apply(window, **kw):
    return _apply_moving_frame(
        _var(**kw), frame_steps=3, window=window, super_res_factor=1,
        centres=_centres(kw.get("B", 1), kw.get("T", 6)), x0=0.0, dx=1e-5)


def test_averaged_moving_frame_drops_the_time_axis():
    out = _apply(window=2)["Temperature"]
    assert out.ndim == 5, f"expected (B,C,X,Y,Z), got {out.shape}"


def test_unaveraged_moving_frame_keeps_the_time_axis():
    """window=1 is the no-averaging case and must still carry T."""
    out = _apply(window=1)["Temperature"]
    assert out.ndim == 6, f"expected (B,T,C,X,Y,Z), got {out.shape}"


def test_the_retained_frame_is_the_last_one():
    """Dropping the axis must keep the LAST window, not the first."""
    full = _apply(window=2)["Temperature"]
    kept = _apply_moving_frame(
        _var(), frame_steps=3, window=2, super_res_factor=1,
        centres=_centres(), x0=0.0, dx=1e-5)["Temperature"]
    np.testing.assert_array_equal(full, kept)
    assert np.isfinite(full).all()
