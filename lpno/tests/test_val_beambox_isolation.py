"""How the training-validation beam box relates to checkpoint selection.

Context (2026-08-21): `beam_box_from_input` learned to synthesize a footprint at the origin,
with a fixed default radius, when `beam_dist` is absent. A fair worry:
`best_checkpoint._METRIC_PREFERENCE` puts `beambox_mae_T` ahead of `mae_T` and takes the
first AVAILABLE one, so if that fallback reached validation, a steady run would begin
selecting its weights on a near-beam criterion where every earlier steady run selected on the
whole domain. Validation therefore refused the origin box outright.

RESOLVED 2026-09-01 (on runs 82820/82838 `*_anchorbox_steady_32k`): report it
and let it select. The refusal had produced its own defect -- a beam-following run showed no
beam box for its whole training while its own EVAL reported one, because the eval path passes
`beam_at_origin=bool(run_cfg.moving_ref_frame)` and validation passed nothing. Weights chosen
on the melt region beat weights chosen on the quiescent ~99% substrate, and training now
agrees with eval. The accepted cost: such runs are not checkpoint-comparable to steady arms
trained before this date.

What still must hold, and is pinned below: the origin box follows the CONFIG flag and is
never inferred from the data. A lab-frame run acquiring an origin box is the exact documented
failure this metric has produced three times -- a beam-box number sitting on the substrate
number.
"""
import inspect

from lpno.models.pipeline import validation_beambox
from lpno.models.pipeline.best_checkpoint import _METRIC_PREFERENCE, _pick_metric


def test_steady_without_a_stated_centre_never_emits_beambox():
    """The property, not the proxy: no beam channel of any kind -> no beambox metric.

    Asserting `"beam_dist" in REQUIRED_INPUTS` pinned the IMPLEMENTATION of this guarantee,
    so it failed on a change that kept the guarantee intact (2026-09-01: the gate learned to
    accept a stated `beam_x`/`beam_y` centre). The guarantee is what matters here -- a steady
    run must not start selecting checkpoints on a near-beam criterion -- so pin that.
    """
    validation_beambox._STASH.clear()
    validation_beambox._STASH.update({"z": [object()]})  # z present, no beam channel at all
    assert validation_beambox.beam_box_validation_metrics({}, {}) == {}


def test_a_stated_centre_without_grids_is_not_enough():
    """`beam_x`/`beam_y` alone cannot size a footprint; declining beats guessing."""
    validation_beambox._STASH.clear()
    validation_beambox._STASH.update(
        {"z": [object()], "beam_x": [object()], "beam_y": [object()]}
    )
    assert validation_beambox.beam_box_validation_metrics({}, {}) == {}


def test_origin_box_follows_the_config_flag_and_is_never_inferred():
    """`at_origin` is passed in from `custom.moving_ref_frame`, not sniffed from the data.

    Replaces an assertion that `beam_at_origin` appears nowhere in `_beam_box_from_stash`;
    that guarantee was deliberately lifted (see the module docstring). The guarantee that
    REMAINS is that a lab-frame run cannot acquire one by accident.
    """
    src = inspect.getsource(validation_beambox.beam_box_validation)
    assert "at_origin" in src, "the frame must be passed in explicitly"
    # Default off: nothing acquires an origin box without the flag being set.
    assert validation_beambox._FRAME["at_origin"] is False
    assert not validation_beambox.beam_box_inputs_available(
        lambda k: k in {"z", "x", "y"}, at_origin=False
    ), "a lab-frame run with no beam channel must still decline"
    assert validation_beambox.beam_box_inputs_available(
        lambda k: k in {"z", "x", "y"}, at_origin=True
    ), "a beam-following run needs only the grids: the centre is the origin by construction"


def test_selection_falls_back_to_whole_domain_without_a_beambox_metric():
    """A steady run's losses carry mae_T only, so mae_T is what selects the checkpoint."""
    assert _METRIC_PREFERENCE[0] == "beambox_mae_T", "preference order is deliberate"
    assert _pick_metric({"mae_T": 0.0059, "rmse_T": 0.03}) == ("mae_T", 0.0059)
    assert _pick_metric({"beambox_mae_T": 0.0084, "mae_T": 0.0066})[0] == "beambox_mae_T"
