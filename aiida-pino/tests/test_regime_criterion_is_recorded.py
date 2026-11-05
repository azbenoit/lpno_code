"""A cached regime label must say which rule produced it.

The rule is `aiida_pino.console.viz.classify_regime`: keyhole iff `aspect_mean > 1.0`
AND `cavity_depth_max >= 2.1 * r0`; otherwise conduction, with `depth_cells < 2`
short-circuiting to LoF. The cached `meltpool_stats` extra carries its criterion, so a
reader of the cache can tell which thresholds produced the label.
"""

from __future__ import annotations

from aiida_pino.console.viz import (
    KEYHOLE_ASPECT_MEAN,
    KEYHOLE_MIN_DEPTH_RADII,
    LOF_DEPTH_CELLS,
    REGIME_CRITERION,
)


def test_the_criterion_string_states_the_actual_thresholds():
    """It must be derived from the constants, not a stale copy of them.

    A threshold changed without the description following it is exactly the
    "verdict without its criterion" failure this exists to prevent.
    """
    assert str(KEYHOLE_ASPECT_MEAN) in REGIME_CRITERION
    assert str(KEYHOLE_MIN_DEPTH_RADII) in REGIME_CRITERION
    assert str(LOF_DEPTH_CELLS) in REGIME_CRITERION


def test_the_criterion_names_its_source():
    assert "classify_regime" in REGIME_CRITERION
