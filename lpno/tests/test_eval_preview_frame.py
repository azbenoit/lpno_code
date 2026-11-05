"""The eval preview must capture a frame that has melt in it.

`lazy_preview_batch0.npz` is the only per-voxel artifact an eval leaves behind --
it is what you open to look at pred-vs-GT. It wrote the FIRST batch, and the
loader is case-ordered with `shuffle=False`, so batch 0 is frame 0 of case 0:
the beam has just switched on, `true_fl` is identically zero and `true_T` is
near-uniform ambient. Measured on the geom40 arms (2026-08-17, model 52874):
every field in the preview was flat, `pred_fl`/`true_fl` both all-zero. Useless
for the one job the file has, on every eval this repo has ever run.

Fix: keep the batch with the most melt instead of the first one. That is one
scalar comparison per batch and one deferred write, and it degrades to "the
first batch" when no batch has any melt (a steady/conduction run), so nothing
that previously produced a preview stops producing one.
"""
from __future__ import annotations

import numpy as np

from lpno.models.pipeline.postprocessing import preferred_preview_batch


def _batch(fl_fraction: float, n=8):
    """A fake truth dict whose `fl` has the requested molten fraction."""
    fl = np.zeros((1, 1, n, n, n), dtype=np.float32)
    n_melt = int(round(fl_fraction * fl.size))
    if n_melt:
        fl.reshape(-1)[:n_melt] = 1.0
    return {"fl": fl, "T": np.full_like(fl, 0.3)}


def test_prefers_the_batch_with_more_melt():
    assert preferred_preview_batch(_batch(0.0), _batch(0.2)) is False   # incumbent 0 loses
    assert preferred_preview_batch(_batch(0.2), _batch(0.0)) is True    # incumbent 0.2 keeps


def test_first_batch_is_kept_when_nothing_has_melt():
    """A conduction-only run must still leave a preview behind."""
    assert preferred_preview_batch(_batch(0.0), _batch(0.0)) is True


def test_ties_keep_the_incumbent():
    """Deterministic: equal melt must not flip the stored batch."""
    assert preferred_preview_batch(_batch(0.1), _batch(0.1)) is True


def test_missing_fl_falls_back_to_temperature_spread():
    """Runs without `fl` (T-only outputs) still pick a frame that has structure."""
    flat = {"T": np.full((1, 1, 4, 4, 4), 0.3, dtype=np.float32)}
    hot = {"T": np.linspace(0.3, 1.0, 64, dtype=np.float32).reshape(1, 1, 4, 4, 4)}
    assert preferred_preview_batch(flat, hot) is False
    assert preferred_preview_batch(hot, flat) is True


def test_none_incumbent_always_loses():
    """The first batch seen is always stored."""
    assert preferred_preview_batch(None, _batch(0.0)) is False
