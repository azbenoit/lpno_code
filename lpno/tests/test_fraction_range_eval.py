"""Volume fractions must be in [0, 1] before the eval masks anything with them.

`postprocessing.py` masks every output key with the PREDICTED alpha -- alpha
included, i.e. alpha masked by itself through a k=20 sigmoid. An out-of-range
alpha saturates its own mask to "keep" and survives into the global metric.
Measured 2026-07-30 on gaot_mat_steady_20k: eval rmse_alpha 1.425 on a [0,1]
quantity and rmse_T 6278 K, versus 0.016 and 47.7 K from that same model's own
in-training validation -- 89x and 132x.

This is a shared-pipeline gap, not a per-family one: no family's final projection
is bounded, so any of them can trip it.
"""
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src" / "lpno" / "models" / "pipeline"


def test_clip_precedes_the_alpha_masking_loop():
    """Order matters: clipping after the mask would leave the mask itself wrong."""
    t = (SRC / "postprocessing.py").read_text()
    clip = t.index('for _frac_key in ("alpha", "fl", "VOF")')
    mask = t.index("if run_cfg.alpha_masking:", clip - 4000)
    assert clip < mask, "fractions must be clipped BEFORE they are used as a mask"


def test_masking_a_saturated_alpha_with_itself_is_a_no_op_without_the_clip():
    """The exact mechanism, reproduced on the real helper."""
    from lpno.models.pipeline.postprocessing import fill_where_mask_smooth_np

    bad = np.array([[50.0, -12.0, 0.7]], dtype=np.float32)
    survived = fill_where_mask_smooth_np(bad, bad, threshold=0.5, d=0.0, k=20, inplace=False)
    assert survived[0, 0] > 10.0, "an out-of-range alpha passes its own mask untouched"

    clipped = np.clip(bad, 0.0, 1.0)
    fixed = fill_where_mask_smooth_np(clipped, clipped, threshold=0.5, d=0.0, k=20, inplace=False)
    assert fixed.max() <= 1.0 and fixed.min() >= 0.0
