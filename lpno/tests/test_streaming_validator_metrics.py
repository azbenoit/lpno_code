"""The metrics must live in the validator the TRAINING PATH actually uses.

StreamingGridValidator overrides save_results wholesale (A35: the upstream version
hoards every batch and got runs SIGKILLed at ~45 GB RSS), so it never calls
`_l2_relative_error`. Two earlier attempts patched that hook and the upstream
save_results, passed their unit tests against the wrapper directly, and produced NOTHING
in a live run -- `gaot_cond_disp_tail` reported mae/rmse and no tail deficit or beam-box
at steps 0 and 2000.

This test asserts against the class the training path builds, which is the only check that
would have caught it.
"""
import inspect

from lpno.models.pipeline.model_domain import StreamingGridValidator


def _src():
    return inspect.getsource(StreamingGridValidator.save_results)


def test_the_streaming_validator_is_what_gets_instrumented():
    """Not the upstream GridValidator, and not _l2_relative_error."""
    src = _src()
    assert "tail_deficit_max_" in src
    assert "beambox_mae_" in src


def test_tail_deficit_uses_running_peaks_not_stored_tensors():
    """Streaming is why this class exists; reintroducing whole-split tensors to compute a
    percentile would re-break it. Peaks are accumulable in scalars."""
    src = _src()
    assert "true_max" in src and "pred_max" in src
    assert "torch.cat" not in src, "must stay streaming"


def test_beam_box_is_built_from_invar():
    """beam_dist and z come from the inputs, which only this scope has."""
    src = _src()
    assert '"beam_dist" in invar' in src and '"z" in invar' in src


def test_sign_convention_is_true_minus_pred():
    """Positive = colder than truth. The GAOT baseline is NEGATIVE (over-predicts the
    peak by ~353 K), so a sign flip here would invert every comparison."""
    src = _src()
    assert 's["true_max"] - s["pred_max"]' in src


def test_thermal_headroom_is_not_reported_during_training():
    """It read +837 K on a model running 151 K cold. Surfacing it invites the misread."""
    assert "thermal_headroom" not in _src().replace("thermal_headroom_K is deliberately", "")


def test_beam_box_threshold_matches_the_eval_definition():
    """2 beam radii, inside the metal -- same as postprocessing.beam_box_mask, or the
    training and eval numbers are not comparable."""
    src = _src()
    assert "<= 2.0" in src and '<= 0' in src
