"""A runtime-cache build that cannot fit must be refused, not run until the disk is full.

2026-08-25. Submission 68844 materialised a gas-cropped variant whose build directory
reached 159 GB for 62 members before stalling with root at 1.3 GB of 1.8 TB. The
collateral is the reason this is a gate and not a warning: training 68720 was killed
mid-checkpoint-write with `u_network.best.pth` truncated to 0 bytes at step 13300 and no
traceback, a transport upload hung, and four queued arms sat in `created` while the daemon
reported 4% worker use -- so the symptom read as worker saturation, not as a full disk.

Measured on the real data: 2.70 GB per member uncompressed against ~23 MB stored, so the
96-member train split needs about 260 GB. It could never have completed.

This refuses rather than waits, which is the OPPOSITE of the FLOW-3D free-space floor, and
deliberately: that floor waits because its pressure is transient (another case is about to
drop its flsgrf). This is structural -- no amount of waiting makes 260 GB fit in 174 GB.
"""

from __future__ import annotations

import pytest

import aiida_pino.utils.runtime_cache as runtime_cache_module

from aiida_pino.utils.runtime_cache import (
    RUNTIME_BUILD_FREE_MARGIN_BYTES,
    estimated_arraydir_bytes,
    refuse_build_that_cannot_fit,
)


class _Member:
    def __init__(self, shapes):
        self._shapes = shapes

    def get_simulation_metadata(self):
        return {"array_shapes": self._shapes}


def _members(n, shape=(353, 1, 90, 40, 40), arrays=17):
    return [_Member({f"f{i}": list(shape) for i in range(arrays)}) for _ in range(n)]


def test_the_estimate_is_exact_arithmetic_not_a_remembered_number():
    """float32, so bytes = prod(shape) * 4 per array, summed.

    Asserted exactly rather than against the 2.70 GB measured on a real member: that
    figure came from one collection's mixed array shapes (some 1-D), so anchoring to it
    would encode the fixture rather than the arithmetic.
    """
    shape = (10, 1, 4, 5, 6)
    expected = 10 * 1 * 4 * 5 * 6 * 4 * 3  # three arrays
    assert estimated_arraydir_bytes([_Member({f"f{i}": list(shape) for i in range(3)})]) == expected


def test_the_estimate_is_the_right_order_for_a_real_time_resolved_case():
    """Sanity against the measured scale: a 353-frame 90x40x40 case is GB, not MB.

    2.70 GB/member was measured on the gas-cropped collection, and 96 of them is the
    ~260 GB that could not fit.
    """
    one = estimated_arraydir_bytes(_members(1))
    assert 1e9 < one < 1e11, one
    assert estimated_arraydir_bytes(_members(96)) == one * 96


def test_a_build_that_cannot_fit_is_refused(tmp_path, monkeypatch):
    """The reported case: ~260 GB wanted, far less free."""
    import shutil


    monkeypatch.setattr(shutil, "disk_usage", lambda _p: type("U", (), {"free": 174 * 1024**3, "total": 4000 * 1024**3})())
    # The gate WAITS for transient pressure now and refuses only what can never fit, so a
    # test that wants the refusal must decline to wait.
    monkeypatch.setattr(runtime_cache_module, "RUNTIME_BUILD_WAIT_S", 0)
    with pytest.raises(RuntimeError, match="refusing to materialise"):
        refuse_build_that_cannot_fit(tmp_path, _members(96), {}, label="train split")


def test_a_build_that_fits_is_allowed(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: type("U", (), {"free": 900 * 1024**3, "total": 4000 * 1024**3})())
    refuse_build_that_cannot_fit(tmp_path, _members(96), {})


def test_the_margin_is_enforced_not_just_the_raw_fit(tmp_path, monkeypatch):
    """Filling the disk to the last byte is what took the unrelated training down."""
    import shutil

    need = estimated_arraydir_bytes(_members(4))
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: type("U", (), {"free": need + 1024**3, "total": 4000 * 1024**3})())
    # The gate WAITS for transient pressure now and only refuses what can never fit, so a
    # test that wants the refusal must say it is not willing to wait. Without this the two
    # refusal cases would each hold the suite for the full 30-minute budget.
    monkeypatch.setattr(runtime_cache_module, "RUNTIME_BUILD_WAIT_S", 0)
    with pytest.raises(RuntimeError):
        refuse_build_that_cannot_fit(tmp_path, _members(4), {})

    monkeypatch.setattr(
        shutil, "disk_usage",
        lambda _p: type("U", (), {"free": need + RUNTIME_BUILD_FREE_MARGIN_BYTES + 1024**3, "total": 4000 * 1024**3})(),
    )
    refuse_build_that_cannot_fit(tmp_path, _members(4), {})


def test_super_resolution_grows_the_estimate():
    """Every other transform shrinks; this one does not, so the bound must account for it."""
    plain = estimated_arraydir_bytes(_members(1), {})
    upsampled = estimated_arraydir_bytes(_members(1), {"super_res_factor": 2})
    assert upsampled == plain * 8


def test_a_member_without_metadata_does_not_block_the_build(tmp_path):
    """Unknown size must not be treated as infinite; the gate is best-effort."""
    class _Broken:
        def get_simulation_metadata(self):
            raise RuntimeError("no metadata")

    assert estimated_arraydir_bytes([_Broken()]) == 0
    refuse_build_that_cannot_fit(tmp_path, [_Broken()], {})


def test_both_build_sites_apply_the_gate():
    """Gating one path leaves the other able to fill the filesystem."""
    from pathlib import Path

    import aiida_pino

    source = (Path(aiida_pino.__file__).parent / "calculations" / "base.py").read_text()
    assert source.count("_refuse_if_it_cannot_fit(") >= 3, (
        "expected the helper plus a call in each of the two build paths"
    )


# --- the estimate must model the derivation, not the raw shapes -------------------
#
# 2026-08-26. Estimating from RAW `array_shapes` over-estimated a `stride: [2,2,1]`
# collection by 4x: mb40_test_8case was refused at "183 GB needed, 200 GB free,
# 21 GB margin" -- 4 GB short -- when the exporter would really have written ~46 GB. The
# same collection now estimates 46 GB and fits. An over-estimate is not the safe direction
# when it blocks real work; it just fails in a way that looks principled.
#
# Order follows `runtime_export.py:141-171`: crop, then frames, then stride, then super-res.


def test_stride_reduces_the_estimate():
    plain = estimated_arraydir_bytes(_members(1), {})
    strided = estimated_arraydir_bytes(_members(1), {"stride": [2, 2, 1]})
    assert strided * 4 == plain


def test_stride_rounds_up_not_down():
    """A stride of 2 over 5 cells keeps 3. Rounding down would under-estimate."""
    m = [_Member({"f": [1, 1, 5, 1, 1]})]
    assert estimated_arraydir_bytes(m, {"stride": [2, 1, 1]}) == 3 * 4


def test_crop_reduces_the_estimate_on_z():
    m = [_Member({"f": [1, 1, 1, 1, 40]})]
    assert estimated_arraydir_bytes(m, {"crop_z_top": 30}) == 10 * 4


def test_time_frames_reduces_the_estimate():
    m = [_Member({"f": [500, 1, 2, 2, 2]})]
    assert estimated_arraydir_bytes(m, {"time_frames": 50}) == 50 * 8 * 4


def test_transforms_compose_in_the_exporter_order():
    """crop then stride: 40 - 30 = 10 cells, strided by 2 -> 5. Not 40/2 - 30."""
    m = [_Member({"f": [1, 1, 1, 1, 40]})]
    assert estimated_arraydir_bytes(m, {"crop_z_top": 30, "stride": [1, 1, 2]}) == 5 * 4


def test_the_steady_beam_frame_route_is_one_window_frame_not_every_raw_frame():
    """2026-10-07: a 295-member steady training set was announced at ~1585 GB because the
    estimate kept every raw frame; the export keeps the last 30, each cut to a window 29
    cells narrower than the domain, and averages them into one."""
    steady = {"moving_ref_frame": True, "moving_frame_steps": 30, "time_avg_window": 30}
    m = [_Member({"Temperature": [798, 1, 90, 60, 55]})]
    assert estimated_arraydir_bytes(m, steady) == 1 * 1 * 61 * 60 * 55 * 4


def test_the_beam_frame_route_without_an_average_keeps_its_window_of_frames():
    moving = {"moving_ref_frame": True, "moving_frame_steps": 30, "time_avg_window": 1}
    m = [_Member({"Velocity": [798, 3, 90, 60, 55]})]
    assert estimated_arraydir_bytes(m, moving) == 30 * 3 * 61 * 60 * 55 * 4


def test_non_canonical_arrays_are_left_alone():
    """Coordinate vectors and scalars are negligible and their axes are not positional."""
    m = [_Member({"time_s": [353], "scalar": [1]})]
    assert estimated_arraydir_bytes(m, {"stride": [2, 2, 1]}) == (353 + 1) * 4


def test_a_strided_collection_that_used_to_be_refused_now_fits(tmp_path, monkeypatch):
    """The mb40 case: 8 members, refused at 183 GB, really ~46 GB."""
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: type("U", (), {"free": 200 * 1024**3, "total": 4000 * 1024**3})())
    members = _members(8, shape=(1200, 1, 150, 100, 40))
    raw = estimated_arraydir_bytes(members, {})
    strided = estimated_arraydir_bytes(members, {"stride": [2, 2, 1]})
    assert strided * 4 == raw
    # The gate WAITS for transient pressure now and only refuses what can never fit, so a
    # test that wants the refusal must say it is not willing to wait. Without this the two
    # refusal cases would each hold the suite for the full 30-minute budget.
    monkeypatch.setattr(runtime_cache_module, "RUNTIME_BUILD_WAIT_S", 0)
    with pytest.raises(RuntimeError):
        refuse_build_that_cannot_fit(tmp_path, members, {}, label="raw")
    refuse_build_that_cannot_fit(tmp_path, members, {"stride": [2, 2, 1]}, label="strided")


def test_a_materialised_build_is_sized_uncompressed(monkeypatch):
    """`materialize: true` writes an UNCOMPRESSED store, so the compressed-scaled estimate is
    wrong by the compression ratio -- 58x measured. On 2026-09-09 an operator was told "about
    1.7 GB will be written" and watched the `.stream-` directory pass 62 GB before killing it.
    """
    from aiida_pino.utils.runtime_cache import estimated_derived_bytes

    members = _members(4)
    monkeypatch.setattr(runtime_cache_module, "_stored_compressed_bytes", lambda m: 10_000)
    compressed = estimated_derived_bytes(members, {})
    uncompressed = estimated_arraydir_bytes(members, {"materialize": True})
    assert compressed < uncompressed, "fixture must make the two estimators differ"
    assert estimated_derived_bytes(members, {"materialize": True}) == uncompressed
