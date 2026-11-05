"""`check_segments` must refuse a WRONG split without refusing a merely faint boundary.

The guard exists because the dump is ordered alphabetically and trusting the caller's
order once mis-split a 755-frame dump at 387 when the true segments were 368/387: one
video held a whole case plus 19 frames of the next, and both looked plausible.

But the detector is threshold-based (mean-temperature step > mean + sigma*std), so it
MISSES a boundary between two cases whose fields happen to be similar there. Measured
2026-09-01 on the mb80 render: counts 95/95/34/90/95/95 (sum 504, matching the dump and
the source frame counts exactly at stride 2) were refused because the 34-frame case's
start at 224 produced too small a step. Every OTHER boundary agreed.

So equality is the wrong test. What must never happen is the data showing a boundary the
caller does not have -- that is the mis-split. A caller boundary the detector cannot
confirm is a detector limit, and is reported rather than fatal.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.render_case_video import check_segments


def _dump(segments, step=40.0):
    """A fake true_T whose per-case mean level jumps by `step` between cases."""
    out = []
    for k, n in enumerate(segments):
        base = 100.0 + k * step
        out.append(np.full((n, 1, 4, 4, 4), base, dtype=np.float32)
                   + np.linspace(0, 1, n, dtype=np.float32)[:, None, None, None, None])
    return np.concatenate(out, axis=0)


def test_exact_agreement_passes():
    check_segments(_dump([30, 40, 50]), [30, 40, 50])


def test_a_boundary_in_the_data_that_the_caller_lacks_is_still_fatal():
    """The original mis-split: caller merges two cases the data separates."""
    with pytest.raises(SystemExit) as exc:
        check_segments(_dump([30, 40, 50]), [70, 50])
    assert "boundaries" in str(exc.value)


def test_a_faint_boundary_the_detector_misses_is_tolerated():
    """The mb80 case: one junction has almost no temperature step, the rest agree."""
    # Middle case sits at the SAME level as the one before it, so no step is detectable.
    segs = [np.full((30, 1, 4, 4, 4), 100.0, dtype=np.float32),
            np.full((20, 1, 4, 4, 4), 100.0, dtype=np.float32),   # faint: same level
            np.full((40, 1, 4, 4, 4), 300.0, dtype=np.float32)]
    true_t = np.concatenate(segs, axis=0)
    check_segments(true_t, [30, 20, 40])          # must NOT raise


def test_totals_must_still_match_the_dump():
    with pytest.raises(SystemExit):
        check_segments(_dump([30, 40, 50]), [30, 40, 10])
