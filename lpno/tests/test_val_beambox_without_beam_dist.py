"""Training validation must report `beambox_*` on a key set that omits `beam_dist`.

Run 82336 (`FNO mref-tab CROP10 v2`) trained 32k steps with no beam-box column. Its input
keys are the 8-channel set [V_scan,P,H_norm,x,y,z,beam_x,beam_y]: the beam centre is stated
outright, so the box is perfectly determined. `beam_box_from_input` already knew that
(tests/test_beambox_without_beam_dist_via_centre.py, 2026-08-22 "beam box should not
need beam dist, fix it") -- but the fix landed on the EVAL path only, and the two training
gates still required `beam_dist`: `REQUIRED_INPUTS` here and `"beam_dist" in invar` in
`model_domain.save_results`, which is the copy that reaches the CSV.

Same failure mode as the two box-predicate regressions this module already records: a
second copy of a rule that did not get the correction.
"""
import torch

from lpno.models.pipeline import validation_beambox as vb

NX, NY, NZ = 21, 21, 8
SHAPE = (1, 1, NX, NY, NZ)


def _grid(vec, axis):
    """`vec` broadcast over SHAPE along spatial `axis` (0=x, 1=y, 2=z)."""
    view = [1, 1, 1, 1, 1]
    view[2 + axis] = len(vec)
    return torch.as_tensor(vec, dtype=torch.float32).view(*view).expand(*SHAPE).contiguous()


def _stash_8channel():
    """Run 82336's channels: grids and a stated beam centre, no `beam_dist`."""
    vb._STASH.clear()
    vb._STASH["x"] = [_grid(torch.linspace(-1.0, 1.0, NX), 0)]
    vb._STASH["y"] = [_grid(torch.linspace(-1.0, 1.0, NY), 1)]
    vb._STASH["z"] = [_grid(torch.linspace(-1.0, 0.0, NZ), 2)]
    vb._STASH["beam_x"] = [torch.zeros(SHAPE)]
    vb._STASH["beam_y"] = [torch.zeros(SHAPE)]


def test_centre_channels_are_enough_for_validation_metrics():
    _stash_8channel()
    true = {"T": torch.ones(SHAPE)}
    pred = {"T": torch.zeros(SHAPE)}
    out = vb.beam_box_validation_metrics(true, pred)
    assert "beambox_mae_T" in out, "8-channel set states the beam centre; box must be built"
    assert 0 < float(out["beambox_n"]) < float(torch.tensor(SHAPE).prod()), \
        "a box covering everything (or nothing) is the substrate bug, not a beam box"
    assert abs(float(out["beambox_mae_T"]) - 1.0) < 1e-6


def test_box_still_excludes_error_outside_it():
    """Guards the regression this module keeps hitting: a 'beam box' that is the substrate."""
    _stash_8channel()
    x = vb._STASH["x"][0]
    far = x.abs() > 0.9          # cells far from the beam at the origin
    true = {"T": torch.ones(SHAPE)}
    pred = {"T": torch.where(far, torch.full(SHAPE, -50.0), torch.ones(SHAPE))}
    out = vb.beam_box_validation_metrics(true, pred)
    assert abs(float(out["beambox_mae_T"])) < 1e-6, "far-field error must not enter the box"


def test_no_centre_and_no_beam_dist_stays_silent():
    """A config with no trajectory channel at all is legitimate -- skip, do not warn."""
    vb._STASH.clear()
    vb._STASH["z"] = [_grid(torch.linspace(-1.0, 0.0, NZ), 2)]
    assert vb.beam_box_validation_metrics({"T": torch.ones(SHAPE)}, {"T": torch.zeros(SHAPE)}) == {}


def _stash_beam_following():
    """82820/82838 (`*_anchorbox_steady_32k`): moving-reference frame, key set
    [V_scan,P,H_norm,x,y,z,preheat_T0,beam_shape_uniform,focus_factor,beam_radius,
    conduction_field] -- no `beam_dist`, no `beam_x`/`beam_y`. The beam sits at the grid
    origin by construction and `beam_radius` gives the exact size, so the box is fully
    determined with no beam channel at all.
    """
    vb._STASH.clear()
    vb._STASH["x"] = [_grid(torch.linspace(-1.0, 1.0, NX), 0)]
    vb._STASH["y"] = [_grid(torch.linspace(-1.0, 1.0, NY), 1)]
    vb._STASH["z"] = [_grid(torch.linspace(-1.0, 0.0, NZ), 2)]
    vb._STASH["beam_radius"] = [torch.full(SHAPE, 0.25)]


def test_beam_following_frame_needs_no_beam_channel():
    _stash_beam_following()
    vb._FRAME["at_origin"] = True
    try:
        out = vb.beam_box_validation_metrics({"T": torch.ones(SHAPE)}, {"T": torch.zeros(SHAPE)})
    finally:
        vb._FRAME["at_origin"] = False
    assert "beambox_mae_T" in out, "the origin box is determined in a beam-following frame"
    assert 0 < float(out["beambox_n"]) < float(torch.tensor(SHAPE).prod()), \
        "a box covering the whole domain is the substrate bug this metric keeps producing"


def test_same_channels_in_a_lab_frame_still_decline():
    """The flag is the whole safeguard: identical data, no frame claim, no box."""
    _stash_beam_following()
    assert vb._FRAME["at_origin"] is False
    assert vb.beam_box_validation_metrics({"T": torch.ones(SHAPE)}, {"T": torch.zeros(SHAPE)}) == {}
