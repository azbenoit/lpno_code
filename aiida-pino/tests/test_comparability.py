"""Comparing two evaluations should be a set difference, not archaeology.

Every axis here cost someone a wrong conclusion in the same week:

* a val-scored number compared against test-scored numbers, published and
  retracted (66.1 K vs 37.8 K);
* two evals of one model at different frame counts, where a 0.892-vs-0.823
  "regime gap" became 0.858 vs 0.850 once the counts matched — an artifact;
* full-resolution GAOT compared against half-resolution RIGNO, unseparated,
  because the truthful stride lives in an extra and not in `training_profile.env`;
* a steady-protocol eval that silently re-derived lab frame and returned the lab
  numbers to every digit.

The point is not to block a comparison. It is that a difference must be VISIBLE
without anyone knowing to look for it.
"""

from __future__ import annotations

from aiida_pino.utils.comparability import compare_axis_sets, max_statistics_comparable


def _axes(**overrides):
    base = {
        "dataset_uuid": "ds-1",
        "split": "test",
        "derivation_kind": "lab",
        "served_grid": "90x40x40",
        "eval_frame_stride": 32,
        "training_stride": "1x1x1",
    }
    base.update(overrides)
    return base


def test_identical_evaluations_report_no_differences():
    report = compare_axis_sets({101: _axes(), 102: _axes()})
    assert report["differing_axes"] == []
    assert report["comparable"] is True


def test_a_split_difference_is_named():
    """The val-vs-test comparison that was published and retracted."""
    report = compare_axis_sets({101: _axes(split="val"), 102: _axes(split="test")})
    assert "split" in report["differing_axes"]
    assert report["comparable"] is False
    assert {"val", "test"} == set(report["values"]["split"].values())


def test_a_derivation_difference_is_named():
    """A steady-protocol eval against a lab one is not the same measurement."""
    report = compare_axis_sets({1: _axes(derivation_kind="steady"), 2: _axes(derivation_kind="lab")})
    assert "derivation_kind" in report["differing_axes"]


def test_a_training_stride_difference_is_named():
    """Full-res against half-res, the comparison that reads as a model result."""
    report = compare_axis_sets({1: _axes(training_stride="1x1x1"), 2: _axes(training_stride="2x2x1")})
    assert "training_stride" in report["differing_axes"]


def test_every_axis_is_checked_not_just_the_first():
    report = compare_axis_sets({1: _axes(split="val", derivation_kind="steady"), 2: _axes()})
    assert set(report["differing_axes"]) >= {"split", "derivation_kind"}


def test_an_unknown_axis_is_reported_as_unknown_not_as_agreement():
    """A missing value must never read as 'these agree'."""
    report = compare_axis_sets({1: _axes(training_stride=None), 2: _axes()})
    assert "training_stride" in report["unknown_axes"]
    assert report["comparable"] is False


def test_max_statistics_refuse_across_different_frame_counts():
    """Peak statistics grow as sqrt(2 ln N); a stride difference fakes a gap."""
    assert max_statistics_comparable({1: _axes(eval_frame_stride=32), 2: _axes(eval_frame_stride=41)}) is False
    assert max_statistics_comparable({1: _axes(), 2: _axes()}) is True


def test_mean_statistics_survive_a_frame_count_difference():
    """Per-cell means are robust to it (23.8 -> 23.9 K over 109 -> 121 frames).

    So a stride difference must not blanket-refuse the comparison — only the
    max statistics. Refusing everything would push people back to eyeballing.
    """
    report = compare_axis_sets({1: _axes(eval_frame_stride=32), 2: _axes(eval_frame_stride=41)})
    assert "eval_frame_stride" in report["differing_axes"]
    assert report["mean_statistics_comparable"] is True
    assert report["max_statistics_comparable"] is False


def test_a_single_evaluation_is_trivially_comparable():
    report = compare_axis_sets({101: _axes()})
    assert report["differing_axes"] == [] and report["comparable"] is True


# --- absent on ONE side is a signal; absent on both is only silence --------
# 2026-08-21, from using this module on a real anomaly:
# a best-MAE model scored fusion IoU 0.407 against 0.766 for a worse-MAE model.
# `compare_axis_sets` correctly reported `differing_axes: []` with an unknown,
# and correctly refused to call it comparable -- but a reader sees "nothing
# differs" and has to notice that `unknown` is carrying the whole verdict.
#
# The two cases are not equally informative and were being collapsed:
#   * absent on BOTH sides  -> both evals predate the field. Uninformative.
#   * absent on ONE side    -> the two were produced by different generations of
#                              the metric code. That is a strong signal, and it
#                              was the actual mechanism here: the control's
#                              metrics carry melt_threshold_K / true_T_max_K /
#                              n_frames as None while the newer eval scored 643
#                              frames.


def test_an_axis_missing_on_one_side_only_is_reported_separately():
    from aiida_pino.utils.comparability import compare_axis_sets

    out = compare_axis_sets({
        1: {"dataset_uuid": "d", "split": "test", "derivation_kind": "k",
            "served_grid": "g", "eval_frame_stride": 1, "training_stride": [1, 1, 1]},
        2: {"dataset_uuid": "d", "split": "test", "derivation_kind": "k",
            "served_grid": "g", "eval_frame_stride": None, "training_stride": [1, 1, 1]},
    })
    assert out["differing_axes"] == []
    assert "eval_frame_stride" in out["unknown_axes"]
    assert out["asymmetric_unknown_axes"] == ["eval_frame_stride"], (
        "an axis recorded on one side and not the other is metric-generation "
        "skew, not mere absence"
    )
    assert out["comparable"] is False


def test_an_axis_missing_on_both_sides_is_not_asymmetric():
    from aiida_pino.utils.comparability import compare_axis_sets

    out = compare_axis_sets({
        1: {"dataset_uuid": "d", "split": None, "derivation_kind": "k",
            "served_grid": "g", "eval_frame_stride": 2, "training_stride": [1, 1, 1]},
        2: {"dataset_uuid": "d", "split": None, "derivation_kind": "k",
            "served_grid": "g", "eval_frame_stride": 2, "training_stride": [1, 1, 1]},
    })
    assert "split" in out["unknown_axes"]
    assert out["asymmetric_unknown_axes"] == [], "absent on both sides is only silence"


def test_asymmetric_unknowns_are_a_subset_of_unknowns():
    """The new field refines the old one; it must never contradict it."""
    from aiida_pino.utils.comparability import compare_axis_sets

    out = compare_axis_sets({
        1: {"dataset_uuid": "d", "split": "test", "eval_frame_stride": 1},
        2: {"dataset_uuid": None, "split": None, "eval_frame_stride": None},
    })
    assert set(out["asymmetric_unknown_axes"]) <= set(out["unknown_axes"])
