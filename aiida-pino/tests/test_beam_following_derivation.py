"""The staged moving frame must follow the beam, not the frame index.

`derive_simulation_arrays` is what actually builds every steady/time-averaged variant --
`materialize_derived_dataset` and the staging path both go through it. It had its own copy
of the index-shift crop (`_crop_shift_moving_frame_single`), which advances the window
exactly one cell per frame and therefore follows the beam only where
``V_scan * dt == dx``.

Measured on the spkh123 train split (96 cases, 2026-08-20): the beam advances 0.051-0.482
cells per frame, median 0.200. Over the 30-frame window the window slides 29 cells while
the beam moves 1.5-14, so the beam drifts 21-27 cells -- 4.2 to 5.4 beam radii -- out of
the window and the average smears the melt pool across that drift.

This was the THIRD copy of the window arithmetic. lpno now owns one
(`models.pipeline.frame_transfer`) and this module calls it.
"""
import numpy as np
import pytest

from aiida_pino.data.pino_dataset_collection import (
    _crop_shift_moving_frame_single,
    _moving_average_last_window_single,
)
frame_transfer = pytest.importorskip(
    "lpno.models.pipeline.frame_transfer",
    reason="needs the sibling lpno checkout installed (README: Companion repositories)",
)
beam_following_slices = frame_transfer.beam_following_slices


def _slow_beam(steps=30, nx=90, cells_per_frame=0.2):
    """A beam creeping at the spkh123 median 0.2 cells/frame, with a one-cell hot spot."""
    x = np.arange(nx, dtype=np.float64)
    beam = 40.0 + cells_per_frame * np.arange(steps)
    data = np.zeros((steps, 1, nx, 2, 2), dtype=np.float32)
    for t in range(steps):
        data[t, 0, int(round(beam[t]))] = 1.0
    centres = np.stack([beam, np.zeros_like(beam)], axis=1)
    return data, centres, x


def test_beam_following_slices_hold_a_slow_beam_still():
    data, centres, x = _slow_beam()
    slices = beam_following_slices(centres, x, steps=30, super_res_factor=1)
    out = _crop_shift_moving_frame_single(data, 30, super_res_factor=1, slices=slices)
    assert out.shape == (30, 1, 90 - 29, 2, 2)
    hot = [int(np.argmax(out[t, 0, :, 0, 0])) for t in range(30)]
    assert max(hot) - min(hot) <= 1, f"beam drifted {max(hot) - min(hot)} cells in the window"


def test_the_average_of_a_followed_beam_keeps_the_peak():
    data, centres, x = _slow_beam()
    slices = beam_following_slices(centres, x, steps=30, super_res_factor=1)
    followed = _moving_average_last_window_single(
        _crop_shift_moving_frame_single(data, 30, super_res_factor=1, slices=slices), 30
    )
    assert followed.max() >= 0.85, f"beam-following average smeared the peak to {followed.max():.3f}"


def test_refuses_to_crop_without_the_beam_track():
    """Silently falling back to the index shift is what produced the smeared datasets."""
    data, _, _ = _slow_beam()
    with pytest.raises(ValueError, match="beam track|slices"):
        _crop_shift_moving_frame_single(data, 30, super_res_factor=1, slices=None)


def test_slices_must_cover_every_kept_frame():
    data, centres, x = _slow_beam()
    slices = beam_following_slices(centres, x, steps=30, super_res_factor=1)[:5]
    with pytest.raises(ValueError, match="slices"):
        _crop_shift_moving_frame_single(data, 30, super_res_factor=1, slices=slices)


def test_output_width_matches_the_legacy_grid():
    """Configs and latent lattices are sized against nx-(steps-1)*sr; only the position changes."""
    data, centres, x = _slow_beam()
    slices = beam_following_slices(centres, x, steps=30, super_res_factor=1)
    out = _crop_shift_moving_frame_single(data, 30, super_res_factor=1, slices=slices)
    assert out.shape[2] == 61


# --- the second copy of the same window arithmetic -------------------------
# 2026-08-21 duplication audit: the beam-following fix (df90f7e) landed in
# `data/pino_dataset_collection.py` and NOT in `utils/dataset_derivation.py`,
# which carries a (B,T,C,X,Y,Z) twin of the same four transforms. The twin has
# no internal callers, but `derive_dataset_arrays` is exported in
# `aiida_pino.utils.__all__`, so it is reachable public API that would silently
# return the index-shift crop this test file exists to reject.

def test_the_six_dimensional_twin_refuses_the_index_shift():
    """The unfixed copy must fail loudly, not return smeared data."""
    import numpy as np
    import pytest

    from aiida_pino.utils.dataset_derivation import _crop_shift_moving_frame

    data = np.zeros((1, 6, 1, 20, 4, 4), dtype=np.float32)
    with pytest.raises(NotImplementedError) as excinfo:
        _crop_shift_moving_frame(data, 3, super_res_factor=1)
    message = str(excinfo.value)
    # must point at the beam track and at the copy that implements it
    assert "beam" in message.lower()
    assert "pino_dataset_collection" in message
