"""A steady eval must score every test case.

The auto-subset stride is computed from the collection's stored frame count,
which is the count BEFORE derivation. A steady derivation collapses each case to
one time-averaged frame, so a stride derived from the raw count is applied to a
series that no longer has those frames: on the calibrated straight-track split
(1005 stored frames/case, 13 cases) the stride came out 41 and was applied to a
13-frame series, leaving ONE case scored out of thirteen. The metrics carried no
sign of it beyond `fl.iou_support_frames`.
"""
from aiida_pino.workflows.evaluate_model import auto_eval_stride, served_frames_per_case

STEADY = {"time_as_batch": False, "moving_ref_frame": True, "time_avg_window": 30,
          "moving_frame_steps": 30, "include_time": True}
LAB = {"time_as_batch": True, "moving_ref_frame": False, "time_avg_window": 1,
       "moving_frame_steps": 1, "include_time": True}


def test_steady_derivation_serves_one_frame_per_case():
    """Time-averaging in the beam frame leaves one frame per case, whatever is stored."""
    assert served_frames_per_case(STEADY, stored_per_case=1005) == 1
    assert served_frames_per_case(STEADY, stored_per_case=None) == 1


def test_steady_eval_never_subsets():
    """One frame per case is already below the target, so the stride must be 1."""
    per_case = served_frames_per_case(STEADY, stored_per_case=1005)
    assert auto_eval_stride(per_case, per_case * 13) == 1


def test_the_regression_this_fixes():
    """The exact numbers that scored 1 of 13 cases on the conduction-11ch arms."""
    assert auto_eval_stride(1005, 13065) == 41           # what was computed
    per_case = served_frames_per_case(STEADY, stored_per_case=1005)
    assert auto_eval_stride(per_case, per_case * 13) == 1  # what it should be


def test_non_steady_derivation_keeps_the_stored_count():
    """A lab/time-as-batch eval really does serve every stored frame, so subsetting
    it is correct and must not be disabled by this fix."""
    assert served_frames_per_case(LAB, stored_per_case=1005) == 1005
    assert auto_eval_stride(served_frames_per_case(LAB, stored_per_case=1005), 13065) == 41


def test_missing_profile_falls_back_to_stored():
    """No derivation profile means no claim about the served layout."""
    assert served_frames_per_case(None, stored_per_case=400) == 400
    assert served_frames_per_case({}, stored_per_case=400) == 400
