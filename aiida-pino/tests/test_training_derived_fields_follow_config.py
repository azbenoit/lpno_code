"""A TRAINING must bake the conditioning channels it asks for, even when the submitter
passed an explicit derivation profile.

`51e427f` fixed the EVAL side: `build_derivation_profile_from_training_profile` promotes
buildable `custom.*_input_keys` into `derived_fields`, so a conduction-conditioned eval
reads a materialised variant instead of rebuilding per case into `~/.pino/conduction_cache`.

The training side stayed open, and the reason is sharp: BOTH training funnels
(`presubmit.resolve_training_submission` and `KFoldTrainEvalWorkChain`) call that promoting
builder only when NO `derivation_profile` was passed. An explicit one is taken verbatim --
and the documented submission workflow passes one. So following the documented
workflow silently routed the training to the loader's per-host builder cache, which has no
eviction: ~40-80 GB every couple of days, hand-trimmed three times (98 GB
on 2026-09-08, 81 GB on 2026-09-10), and the likely reason root hit 97 % before the
2026-09-10 /tmp clear that killed 38 eval jobs.

Measured 2026-09-10: ten live processes requested `conduction_field` and baked none of it;
every one named the channel in an OVERRIDE (`config_flow3D_GAOT_pvlhs` does not carry it),
so the promotion had the signal and was simply never consulted.

Union, never replace: the submitter's crop/stride/frame choices are theirs.

**2026-09-11, the same rule reversed to OPT-IN.** The promotion above worked, and it was the
accident: a submitter's identity profile was silently substituted to a derived collection with
the channel baked, which took 61 members off the pack-index path and onto a full copy inside
`do_upload` -- both local GPUs idle for two hours behind a single-core export, and a
`BrokenProcessPool` when two such exports overcommitted RAM. Materialisation should
be hard to do by accident: it is almost never the right solution. `conduction_field` is a READ-TIME path channel the loader builds itself. So the
promotion now applies only when the submitter typed `materialize: true` -- the flag whose name
already says "bake it" -- and the read-time build stays the default.
"""

from __future__ import annotations

import pytest

from aiida_pino.utils.dataset_derivation import (
    validate_derivation_profile,
    with_requested_derived_fields,
)

CONDUCTION_OVERRIDE = {"hydra_overrides": ["++custom.FNO_input_keys=[Temperature,conduction_field]"]}


def _explicit(**kw):
    """What a submitter following the documented workflow passes."""
    return validate_derivation_profile({"crop_z_top": 10, "stride": [2, 2, 1], **kw})


def _opted_in(**kw):
    return _explicit(materialize=True, **kw)


def test_an_explicit_profile_that_opts_in_bakes_the_requested_channel():
    out = with_requested_derived_fields(_opted_in(), CONDUCTION_OVERRIDE)
    assert out["derived_fields"] == ["conduction_field"]


def test_without_the_opt_in_the_loader_builds_it_at_read_time():
    """The default: no materialisation the submitter did not ask for."""
    profile = _explicit()
    out = with_requested_derived_fields(profile, CONDUCTION_OVERRIDE)
    assert out is profile
    assert not out["derived_fields"]


def test_the_submitters_own_choices_are_untouched():
    out = with_requested_derived_fields(_opted_in(), CONDUCTION_OVERRIDE)
    assert out["crop_z_top"] == 10
    assert list(out["stride"]) == [2, 2, 1]


@pytest.mark.requires_lpno
def test_existing_derived_fields_are_unioned_not_replaced():
    out = with_requested_derived_fields(_opted_in(derived_fields=["mat_A"]), CONDUCTION_OVERRIDE)
    assert out["derived_fields"] == ["conduction_field", "mat_A"]


def test_it_is_idempotent():
    once = with_requested_derived_fields(_opted_in(), CONDUCTION_OVERRIDE)
    assert with_requested_derived_fields(once, CONDUCTION_OVERRIDE)["derived_fields"] == ["conduction_field"]


def test_a_training_that_asks_for_nothing_buildable_is_left_alone():
    """Trajectory channels are loader-built and must NOT be promoted."""
    profile = _opted_in()
    out = with_requested_derived_fields(
        profile, {"hydra_overrides": ["++custom.FNO_input_keys=[Temperature,beam_x,path_dose]"]}
    )
    assert out is profile, "no buildable request must not rebuild the profile"
    assert not out["derived_fields"]


def test_the_packaged_config_is_a_fallback_when_nothing_was_overridden():
    """A config carrying the channel by default must promote it too."""
    out = with_requested_derived_fields(
        _opted_in(), {}, hydra_config={"custom": {"FNO_input_keys": ["Temperature", "conduction_field"]}}
    )
    assert out["derived_fields"] == ["conduction_field"]


@pytest.mark.parametrize("module,attr", [
    ("aiida_pino.utils.presubmit", "resolve_training_submission"),
    ("aiida_pino.workflows.kfold_train_eval", "KfoldTrainEvalWorkChain"),
])
def test_both_training_funnels_apply_the_promotion(module, attr):
    """Fixing only the reported funnel would leave the other writing the cache."""
    import importlib
    import inspect

    source = inspect.getsource(getattr(importlib.import_module(module), attr))
    assert "with_requested_derived_fields" in source, f"{attr} must promote requested channels"
