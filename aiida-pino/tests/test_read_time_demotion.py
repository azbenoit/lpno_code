"""Transforms with a read-time twin must not be baked into a staged copy.

2026-08-26, user: "local disk space is precious". Zero-copy staging has been default-on
since 2026-08-13 -- untransformed `npz_file` members become byte-range entries in
`pack_index.json` and staging writes a few KB. But `_export_members` routes a member down
the writer branch whenever `derivation_changes_payload(profile)`, and a `stride` or a
`crop_z_top` in the profile is enough to trigger it. So `mb40_test_8case` (8 members) wrote
**46 GB of uncompressed .npy** whose only reason to exist was a `stride: [2,2,1]` the LOADER
can apply for free at read time.

Both transforms have loader twins -- `PINO_LAZY_STRIDE` and `PINO_LAZY_CROP_Z_TOP` -- and
both are guarded in lpno against composing with an already-baked value. Moving
them out of the profile leaves it a no-op, so the collection stages pack-indexed instead.

Two rules keep this from making things worse, and both are the interesting cases:

- Demote ONLY when it achieves zero copy. If anything else in the profile still rewrites the
  arrays, the copy happens regardless, and removing the stride makes that copy FOUR TIMES
  BIGGER (full resolution instead of strided).
- Never demote for a remote target. Pack-index entries are absolute pack paths on this
  machine, so `allow_pack_index=False` there and the copy is unavoidable -- demoting would
  again only enlarge it.
"""

from __future__ import annotations

import pytest

from aiida_pino.data.pino_dataset_collection import derivation_changes_payload
from aiida_pino.utils.dataset_derivation import demote_read_time_transforms


def test_a_stride_only_profile_becomes_a_no_op_and_moves_to_the_env():
    profile, env, notes = demote_read_time_transforms({"stride": [2, 2, 1]}, {}, read_time_supported=True)
    assert not derivation_changes_payload(profile), "must stage pack-indexed"
    assert env["PINO_LAZY_STRIDE"] == "2,2,1"
    assert notes


def test_a_crop_only_profile_moves_to_the_env():
    profile, env, _ = demote_read_time_transforms({"crop_z_top": 30}, {}, read_time_supported=True)
    assert not derivation_changes_payload(profile)
    assert env["PINO_LAZY_CROP_Z_TOP"] == "30"


def test_stride_and_crop_together():
    profile, env, _ = demote_read_time_transforms(
        {"stride": [2, 2, 1], "crop_z_top": 30}, {}, read_time_supported=True
    )
    assert not derivation_changes_payload(profile)
    assert env["PINO_LAZY_STRIDE"] == "2,2,1"
    assert env["PINO_LAZY_CROP_Z_TOP"] == "30"


def test_a_profile_that_still_rewrites_arrays_is_left_alone():
    """The copy happens anyway; removing the stride would make it 4x bigger.

    `super_res_factor`, not `moving_ref_frame`: only some keys actually rewrite arrays.
    Measured against the predicate itself -- payload-changing: super_res_factor,
    time_frames, stride, crop_z_top. NOT: moving_ref_frame, time_avg_window, which are
    layout keys applied downstream. Picking a layout key here would assert the opposite
    of what it looks like it asserts.
    """
    original = {"stride": [2, 2, 1], "super_res_factor": 2}
    profile, env, notes = demote_read_time_transforms(original, {}, read_time_supported=True)
    assert profile == original
    assert env == {}
    assert not notes


def test_a_target_that_cannot_apply_it_at_read_time_is_left_alone():
    """A checkout that does not know the env var ignores it and trains the full extent.

    Corrects the first version of this rule, which refused for any REMOTE target on the
    reasoning that pack-index cannot travel. True but not the point: the fallback for an
    untransformed member is a hardlink-or-copy of the stored `payload.npz`, still COMPRESSED.
    Measured on one 96-member collection: 13.3 GB of archives copied versus 309 GB of `.npy`
    baked. Demoting wins off-box by 23x, so the question is capability, not locality.
    """
    original = {"stride": [2, 2, 1]}
    profile, env, notes = demote_read_time_transforms(original, {}, read_time_supported=False)
    assert profile == original
    assert env == {}
    assert not notes


def test_a_remote_box_is_opt_in_by_label(monkeypatch):
    """Enabling a box is a deliberate act AFTER its checkout is synced, never a default."""
    import aiida_pino.utils.dataset_derivation as dd

    class _Computer:
        label = "box-a"

        def get_transport_type(self):
            return "core.ssh"

    code = type("Code", (), {"computer": _Computer()})()
    profile_in = {"crop_z_top": 10}

    monkeypatch.delenv("AIIDA_PINO_REMOTE_READ_TIME", raising=False)
    _p, _e, notes = dd.demote_for_code(profile_in, {}, code)
    assert not notes, "an unlisted box must keep the transform baked"

    monkeypatch.setenv("AIIDA_PINO_REMOTE_READ_TIME", "box-a")
    _p, env, notes = dd.demote_for_code(profile_in, {}, code)
    assert notes and env["PINO_LAZY_CROP_Z_TOP"] == "10"

    monkeypatch.setenv("AIIDA_PINO_REMOTE_READ_TIME", "some-other-box")
    _p, _e, notes = dd.demote_for_code(profile_in, {}, code)
    assert not notes, "a DIFFERENT box being synced must not enable this one"


def test_an_env_value_the_caller_already_set_is_not_composed_onto():
    """Demoting on top of an existing env stride must not square the reduction.

    Updated for `45cd077`, which changed how an IDENTICAL env value is handled and left
    this test asserting the reversed contract. The old behaviour bailed out and returned the
    profile with the stride still baked; that made the derivation payload-changing, so the
    collection MATERIALISED and the staged variant then recorded a transform the env var
    also asked for -- 117 GB written in 90 minutes on 2026-08-28, root to 99%, and every run
    that caused it failed anyway.

    The concern this test was written for is unchanged and is what it still checks: the
    reduction must be applied ONCE. The profile goes to identity and the env keeps the single
    `2,2,1`, rather than both carrying it.
    """
    profile, env, notes = demote_read_time_transforms(
        {"stride": [2, 2, 1]}, {"PINO_LAZY_STRIDE": "2,2,1"}, read_time_supported=True
    )
    assert profile["stride"] == [1, 1, 1], "the profile must not also bake the stride"
    assert env["PINO_LAZY_STRIDE"] == "2,2,1", "the env value must be left alone, not doubled"
    # It counts as demoted even though the env was untouched: callers read `notes` to decide
    # whether the profile changed, and it did.
    assert any("PINO_LAZY_STRIDE" in n for n in notes), notes


def test_an_identity_stride_is_not_demoted_but_the_default_crop_is():
    """A no-op stride moves nothing; the DEFAULT gas crop still has to move.

    Rewritten 2026-08-26, when `crop_z_top` became a non-zero default. Before that, a
    `stride: [1,1,1]` profile transformed nothing and there was nothing to demote. Now the
    same profile carries an implicit 10-cell crop, and that crop must ride at read time or
    the collection materialises -- so "no notes" is the wrong expectation, and asserting it
    would have hidden the demotion silently declining.
    """
    profile, env, notes = demote_read_time_transforms({"stride": [1, 1, 1]}, {}, read_time_supported=True)
    assert not derivation_changes_payload(profile), "must still reach pack-indexed"
    assert "PINO_LAZY_STRIDE" not in env, "a no-op stride is not worth an env var"
    assert env["PINO_LAZY_CROP_Z_TOP"] == "10"
    assert notes == ["PINO_LAZY_CROP_Z_TOP=10"]


def test_the_caller_s_dicts_are_not_mutated():
    original, env_in = {"stride": [2, 2, 1]}, {}
    demote_read_time_transforms(original, env_in, read_time_supported=True)
    assert original == {"stride": [2, 2, 1]}
    assert env_in == {}


def test_a_companion_that_reshapes_blocks_the_demotion():
    """A profile whose OTHER transform genuinely rewrites arrays is left alone.

    This replaces a test that asserted a "layout-only companion" still demotes. Measured
    against `derivation_changes_payload`, no such companion exists: the only thing that makes
    an otherwise-default profile payload-changing is the crop itself, and every other key
    named as layout (`time_as_dim`, `moving_frame_steps`, `include_time`) reshapes. So the
    honest property is the opposite one -- demotion declines when it cannot reach zero copy,
    because stripping the crop from a profile that still resamples only makes the copy
    bigger.
    """
    original = {"stride": [2, 2, 1], "time_as_dim": True}
    profile, env, notes = demote_read_time_transforms(original, {}, read_time_supported=True)
    assert not notes
    assert profile == original
    assert env == {}


def test_both_workchains_demote_through_the_same_helper():
    """A rule implemented twice in this repo has diverged every time.

    Updated for the presubmit extraction: the TRAINING workchain no longer calls
    `demote_for_code` inline -- its whole sequence moved into
    `resolve_training_submission`, which it delegates to. The rule this test pins is
    unchanged: BOTH submission paths demote, and through the one helper.
    """
    import inspect
    from pathlib import Path

    import aiida_pino
    from aiida_pino.utils.presubmit import resolve_training_submission
    from aiida_pino.workflows.train_model import TrainModelWorkChain

    assert "demote_for_code(" in inspect.getsource(resolve_training_submission)
    assert "resolve_training_submission" in inspect.getsource(TrainModelWorkChain.validate_inputs)
    root = Path(aiida_pino.__file__).parent / "workflows"
    assert "demote_for_code(" in (root / "evaluate_model.py").read_text()


def test_local_is_decided_by_the_same_test_staging_uses():
    """A second opinion about 'local' would strip a stride from a collection that then
    gets copied at FULL resolution -- strictly worse than baking it."""
    import inspect

    from aiida_pino.utils.dataset_derivation import demote_for_code

    assert "pack_index_allowed_for_transport" in inspect.getsource(demote_for_code)


def test_the_computer_attribute_this_relies_on_actually_exists():
    """`transport_type` is a PROPERTY; there is no `get_transport_type()` on orm.Computer.

    2026-08-26. `demote_for_code` called an invented `get_transport_type()`, the AttributeError
    was swallowed by a broad `except`, and the demotion silently did nothing for EVERY run with
    a computer -- while the unit tests stayed green, because the fake computer in them defined
    the invented method. The mock encoded the wrong assumption, so only real data caught
    it: one run baked 309 GB eleven minutes after the fix was deployed.

    Asserted against the real class so a fake can never hide this again.
    """
    from aiida import orm

    assert hasattr(orm.Computer, "transport_type")
    assert not hasattr(orm.Computer, "get_transport_type")


def test_demotion_fires_for_a_local_aiida_computer():
    """End-to-end against a REAL Computer, not a stand-in."""
    from aiida import orm

    from aiida_pino.utils.dataset_derivation import demote_for_code

    computer = orm.Computer(
        label="pytest-local-demotion",
        hostname="localhost",
        transport_type="core.local",
        scheduler_type="core.direct",
        workdir="/tmp/pytest-demotion",
    )
    code = type("Code", (), {"computer": computer})()
    _p, env, notes = demote_for_code({"crop_z_top": 10}, {}, code)
    assert notes and env["PINO_LAZY_CROP_Z_TOP"] == "10"


def test_an_unreadable_transport_bakes_rather_than_demoting():
    """Unknown capability must fall to the safe side, and never silently to the unsafe one."""
    from aiida_pino.utils.dataset_derivation import demote_for_code

    class _Opaque:
        label = "mystery"
        transport_type = ""

    code = type("Code", (), {"computer": _Opaque()})()
    profile, _env, notes = demote_for_code({"crop_z_top": 10}, {}, code)
    assert not notes
    assert profile["crop_z_top"] == 10
