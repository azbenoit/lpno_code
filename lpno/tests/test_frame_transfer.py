"""The moving frame must FOLLOW THE BEAM, and the three frame treatments must agree.

Why this file exists (measured 2026-08-20 on the spkh123 train split, 96 cases):
`npz_datasource._crop_shift_moving_frame` shifts frame ``t`` by exactly
``super_res_factor * t`` cells. That is a beam-following crop only when
``V_scan * dt == dx``. On spkh123 the beam advances **0.051 .. 0.482 cells per frame
(median 0.200)**, so over the 30-frame steady window the window slides 29 cells while
the beam moves 1.5 .. 14. The beam drifts 21-27 cells -- 4.2 to 5.4 beam radii -- inside
the window that is supposed to hold it fixed, and `time_avg_window=30` then averages the
melt pool across that drift instead of freezing it.

The lazy reader (`NpzDirStore._moving_window`) already crops around the beam's ACTUAL
per-frame cell. Two implementations of one idea, one of them wrong. These tests pin the
shared primitive and the agreement between the paths that use it.
"""
import numpy as np
import pytest

from lpno.models.pipeline.frame_transfer import (
    FrameSpec,
    beam_following_x_slice,
    legacy_window_split,
    to_moving_ref,
    to_steady,
    transfer,
)


# --------------------------------------------------------------------------- #
# the shared primitive
# --------------------------------------------------------------------------- #
def test_window_starts_behind_cells_before_the_beam():
    sl = beam_following_x_slice(beam_cx=50.0, x0=0.0, dx=1.0, nx=90, behind=32, ahead=16)
    assert (sl.start, sl.stop) == (18, 66)


def test_window_width_is_constant_under_clamping():
    for cx in (0.0, 3.0, 44.0, 88.0, 89.0):
        sl = beam_following_x_slice(beam_cx=cx, x0=0.0, dx=1.0, nx=90, behind=32, ahead=16)
        assert sl.stop - sl.start == 48
        assert 0 <= sl.start and sl.stop <= 90


def test_window_clamps_rather_than_running_off_the_domain():
    left = beam_following_x_slice(beam_cx=1.0, x0=0.0, dx=1.0, nx=90, behind=32, ahead=16)
    right = beam_following_x_slice(beam_cx=89.0, x0=0.0, dx=1.0, nx=90, behind=32, ahead=16)
    assert left.start == 0
    assert right.stop == 90


def test_window_respects_a_non_unit_dx_and_origin():
    # 10 um cells starting at -450 um: a beam at -50 um is cell 40.
    sl = beam_following_x_slice(beam_cx=-50e-6, x0=-450e-6, dx=10e-6, nx=90, behind=32, ahead=16)
    assert sl.start == 40 - 32


def test_legacy_window_split_keeps_the_two_to_one_convention():
    # PINO_LAZY_MOVING_REF="32,16" is 2:1 behind:ahead; a width-61 legacy window keeps it.
    assert legacy_window_split(48) == (32, 16)
    behind, ahead = legacy_window_split(61)
    assert behind + ahead == 61
    assert behind > ahead


# --------------------------------------------------------------------------- #
# the bug this file was written for
# --------------------------------------------------------------------------- #
def _slow_beam_case(nt=30, nx=90, cells_per_frame=0.2):
    """A case whose beam creeps at 0.2 cells/frame -- the spkh123 median."""
    beam = 40.0 + cells_per_frame * np.arange(nt)
    field = np.zeros((1, nt, 1, nx, 2, 2), dtype=np.float32)
    for t in range(nt):
        field[0, t, 0, int(round(beam[t])), :, :] = 1.0   # a one-cell hot spot AT the beam
    return field, beam


def test_beam_following_window_holds_a_slow_beam_still():
    """The whole point: a beam slower than one cell per frame must not drift out."""
    field, beam = _slow_beam_case()
    win = to_moving_ref(field, beam_cx=beam, x0=0.0, dx=1.0, behind=32, ahead=16)
    assert win.shape == (1, 30, 1, 48, 2, 2)
    hot = [int(np.argmax(win[0, t, 0, :, 0, 0])) for t in range(30)]
    assert max(hot) - min(hot) <= 1, f"beam drifted {max(hot) - min(hot)} cells inside the window"


def test_legacy_index_shift_would_have_let_it_drift():
    """Pins the defect itself, so nobody reintroduces the one-cell-per-frame shift."""
    from lpno.custom_utils.npz_datasource import _crop_shift_moving_frame

    field, _ = _slow_beam_case()
    shifted = _crop_shift_moving_frame(field, 30, super_res_factor=1)
    hot = [int(np.argmax(shifted[0, t, 0, :, 0, 0])) for t in range(30)]
    assert max(hot) - min(hot) >= 20, (
        "the legacy index shift is supposed to drift here; if this now passes, the "
        "legacy path was fixed and this test should be deleted"
    )


def test_steady_average_of_a_followed_beam_keeps_the_peak():
    """A smeared average loses the peak; a beam-following one keeps it."""
    field, beam = _slow_beam_case()
    followed = to_steady(to_moving_ref(field, beam_cx=beam, x0=0.0, dx=1.0, behind=32, ahead=16),
                         avg_window=30)
    assert followed.shape == (1, 1, 1, 48, 2, 2)
    assert followed.max() > 0.9, f"peak smeared to {followed.max():.3f}"

    from lpno.custom_utils.npz_datasource import _crop_shift_moving_frame
    smeared = _crop_shift_moving_frame(field, 30, super_res_factor=1).mean(axis=1)
    assert smeared.max() < 0.2, "the legacy path is supposed to smear this peak"


# --------------------------------------------------------------------------- #
# transfer between treatments
# --------------------------------------------------------------------------- #
def test_lab_to_steady_equals_lab_to_ref_then_average():
    field, beam = _slow_beam_case()
    lab = FrameSpec("lab")
    ref = FrameSpec("moving_ref", behind=32, ahead=16)
    steady = FrameSpec("steady", behind=32, ahead=16, avg_window=30)
    direct = transfer(field, lab, steady, beam_cx=beam, x0=0.0, dx=1.0)
    staged = to_steady(transfer(field, lab, ref, beam_cx=beam, x0=0.0, dx=1.0), avg_window=30)
    np.testing.assert_allclose(direct, staged, rtol=0, atol=0)


def test_ref_to_steady_averages_an_already_cropped_source():
    field, beam = _slow_beam_case()
    ref = FrameSpec("moving_ref", behind=32, ahead=16)
    steady = FrameSpec("steady", behind=32, ahead=16, avg_window=30)
    cropped = transfer(field, FrameSpec("lab"), ref, beam_cx=beam, x0=0.0, dx=1.0)
    out = transfer(cropped, ref, steady)
    np.testing.assert_allclose(out, to_steady(cropped, avg_window=30))


def test_identity_transfer_is_a_no_op():
    field, beam = _slow_beam_case()
    lab = FrameSpec("lab")
    np.testing.assert_allclose(transfer(field, lab, lab), field)


def test_ref_to_steady_refuses_a_window_mismatch():
    field, beam = _slow_beam_case()
    ref = FrameSpec("moving_ref", behind=32, ahead=16)
    other = FrameSpec("steady", behind=24, ahead=24, avg_window=30)
    cropped = transfer(field, FrameSpec("lab"), ref, beam_cx=beam, x0=0.0, dx=1.0)
    with pytest.raises(ValueError, match="window"):
        transfer(cropped, ref, other)


def test_information_losing_transfers_are_refused():
    field, beam = _slow_beam_case()
    lab = FrameSpec("lab")
    ref = FrameSpec("moving_ref", behind=32, ahead=16)
    steady = FrameSpec("steady", behind=32, ahead=16, avg_window=30)
    cropped = transfer(field, lab, ref, beam_cx=beam, x0=0.0, dx=1.0)
    with pytest.raises(ValueError, match="cannot be un-cropped|lab"):
        transfer(cropped, ref, lab)
    with pytest.raises(ValueError, match="cannot be un-averaged|steady"):
        transfer(to_steady(cropped, 30), steady, ref)


def test_lab_to_ref_needs_the_beam_track():
    field, _ = _slow_beam_case()
    with pytest.raises(ValueError, match="beam_cx"):
        transfer(field, FrameSpec("lab"), FrameSpec("moving_ref", behind=32, ahead=16))


# --------------------------------------------------------------------------- #
# the lazy reader must use the shared primitive, not its own copy
# --------------------------------------------------------------------------- #
def test_lazy_store_crop_matches_the_shared_primitive():
    """NpzDirStore's window and frame_transfer's must not be allowed to drift apart."""
    import inspect
    from lpno.models.pipeline import npz_lazy_store

    src = inspect.getsource(npz_lazy_store)
    assert "beam_following_x_slice" in src, (
        "npz_lazy_store must call the shared beam_following_x_slice, not re-derive the "
        "window arithmetic locally"
    )


# --------------------------------------------------------------------------- #
# the steady derivation itself
# --------------------------------------------------------------------------- #
def test_apply_moving_frame_follows_the_beam_when_given_the_track():
    from lpno.custom_utils.npz_datasource import _apply_moving_frame

    field, beam = _slow_beam_case()
    centres = np.stack([beam, np.zeros_like(beam)], axis=1)[None]      # (B=1, T, 2)
    out = _apply_moving_frame(
        {"Temperature": field}, 30, 30, 1, centres=centres, x0=0.0, dx=1.0
    )["Temperature"]
    # One frame, and the axis is DROPPED rather than left at length 1: the shape is
    # (B, C, X, Y, Z). Written as [:, -1:] when this test was added (a540be1), which
    # kept a T=1 axis and broke every consumer of the eager path -- a 2D FNO gets
    # (B,1,C,X,Z) and asserts "Only 4D tensors accepted". `load_consolidated_npz`
    # documents the removal ("the moving frame trims X and (if avg_window_size>1)
    # removes the time axis"), and no real training caught it because they all read
    # through NpzDirStore. The intent here is unchanged -- one frame survives the
    # 30-frame average -- only the axis bookkeeping is corrected.
    assert out.ndim == 5, f"a 30-frame average must leave one frame, axis dropped: {out.shape}"
    assert out.shape[2] == 90 - 29, "the legacy grid width must be preserved"
    # Not 1.0: the window snaps to whole cells, so a beam creeping 0.2 cells/frame sits one
    # cell over for 3 frames in 30. That residue is sub-cell rounding, not smear -- the
    # legacy index shift scores below 0.2 on the same input (test above).
    assert out.max() >= 0.85, f"beam-following average smeared the peak to {out.max():.3f}"


def test_apply_moving_frame_refuses_to_guess_without_the_track():
    """Silently falling back to the index shift is what produced the smeared datasets."""
    from lpno.custom_utils.npz_datasource import _apply_moving_frame

    field, _ = _slow_beam_case()
    with pytest.raises(ValueError, match="beam track|centres"):
        _apply_moving_frame({"Temperature": field}, 30, 30, 1)


# --------------------------------------------------------------------------- #
# eval-time batch cropping: one eval = one (model, target frame) pair
# --------------------------------------------------------------------------- #
# interior frames need beam cell in [behind, nx-width+behind] = [32, 74] for 32+16 on 90.
def _batch(nb=4, nx=90, ny=6, nz=5, beam_cells=(10, 40, 55, 85)):
    """A (B, C, X, Y, Z) eval batch: B frames, each with its own beam position."""
    x_line = np.linspace(-4.5, 4.5, nx, dtype=np.float32)          # normalised units
    dx = float(x_line[1] - x_line[0])
    T = np.zeros((nb, 1, nx, ny, nz), dtype=np.float32)
    beam = np.zeros((nb, 1, nx, ny, nz), dtype=np.float32)
    for b, cell in enumerate(beam_cells):
        T[b, 0, cell] = 1.0
        beam[b] = x_line[cell]
    x_field = np.broadcast_to(x_line.reshape(1, 1, nx, 1, 1), (nb, 1, nx, ny, nz)).copy()
    return {"T": T}, {"x": x_field, "beam_x": beam}, dx


def test_crop_batch_puts_every_frames_beam_at_the_same_window_cell():
    from lpno.models.pipeline.frame_transfer import crop_batch_to_moving_ref

    arrays, invar, _ = _batch()
    out_a, out_i = crop_batch_to_moving_ref(
        arrays, invar, x_key="x", beam_key="beam_x", behind=32, ahead=16
    )
    assert out_a["T"].shape == (4, 1, 48, 6, 5)
    hot = [int(np.argmax(out_a["T"][b, 0, :, 0, 0])) for b in range(4)]
    # frames 1 and 2 are interior so the beam sits exactly at `behind`; frames 0 and 3
    # are clamped against the domain edges, which is the documented trade.
    assert hot[1] == 32 and hot[2] == 32
    assert hot[0] == 10 and hot[3] == 85 - (90 - 48)


def test_crop_batch_rewrites_x_to_beam_relative():
    from lpno.models.pipeline.frame_transfer import crop_batch_to_moving_ref

    arrays, invar, _ = _batch()
    _, out_i = crop_batch_to_moving_ref(
        arrays, invar, x_key="x", beam_key="beam_x", behind=32, ahead=16
    )
    # interior frames: x == 0 exactly where the beam is
    for b in (1, 2):
        assert abs(float(out_i["x"][b, 0, 32, 0, 0])) < 1e-6


def test_crop_batch_crops_every_array_it_is_given():
    from lpno.models.pipeline.frame_transfer import crop_batch_to_moving_ref

    arrays, invar, _ = _batch()
    invar["beam_dist"] = np.ones_like(invar["x"])
    out_a, out_i = crop_batch_to_moving_ref(
        arrays, invar, x_key="x", beam_key="beam_x", behind=32, ahead=16
    )
    for name, arr in list(out_a.items()) + list(out_i.items()):
        assert arr.shape[-3] == 48, f"{name} kept the lab width"


def test_crop_batch_refuses_without_a_beam_channel():
    from lpno.models.pipeline.frame_transfer import crop_batch_to_moving_ref

    arrays, invar, _ = _batch()
    invar.pop("beam_x")
    with pytest.raises(ValueError, match="beam_x"):
        crop_batch_to_moving_ref(arrays, invar, x_key="x", beam_key="beam_x",
                                 behind=32, ahead=16)


def test_the_transfer_hook_is_in_the_eval_loop_that_actually_runs():
    """The hook must sit in run_lazy_dataset_metrics, not the autoregressive path.

    It first landed in `run_masked_patch_metrics` because a blind replace-first matched the
    identical `fl` derivation there. Nothing failed: the six evals ran green and returned
    numbers byte-identical to the untransferred ones, which is the worst possible outcome --
    a silently inert feature reporting plausible results. Only comparing them against the
    earlier native evals caught it.
    """
    import inspect
    import re

    from lpno.models.pipeline import postprocessing

    src = inspect.getsource(postprocessing).splitlines()
    sites = []
    for i, line in enumerate(src):
        if "crop_batch_to_moving_ref(" in line and "import" not in line:
            for j in range(i, -1, -1):
                if re.match(r"^def ", src[j]):
                    sites.append(src[j].split("(")[0].removeprefix("def "))
                    break
    assert sites == ["run_lazy_dataset_metrics"], (
        f"eval_frame_transfer is wired into {sites}; the standard eval path is "
        "run_lazy_dataset_metrics (it writes lazy_preview_batch0.npz)"
    )


def test_no_unguarded_detach_survives_a_frame_transfer():
    """After a transfer `invar` holds numpy, so nothing downstream may assume torch.

    The preview payload builder did, and every lab->ref eval died with
    `'numpy.ndarray' object has no attribute 'detach'` AFTER the crop had run --
    a failure a config-only smoke test would not reach.
    """
    import inspect
    import re

    from lpno.models.pipeline import postprocessing

    src = inspect.getsource(postprocessing.run_lazy_dataset_metrics).splitlines()
    bad = [
        line.strip()
        for line in src
        if ".detach()" in line and not re.search(r'hasattr\(.*"detach"\)', line)
        and "pred_t.items()" not in line and "true_outvar.items()" not in line
    ]
    assert not bad, f"unguarded .detach() on possibly-numpy inputs: {bad}"
