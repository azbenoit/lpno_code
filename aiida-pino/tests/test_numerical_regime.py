"""Which numerical regime produced a number, recorded where a reader will find it.

The user accepted the TF32 deviation, so an Ada box is a different numerical
regime from the existing corpus (3.4e-3 mean relative error on convolutions).
Nothing recorded which regime a result came from.

The trap this module exists to avoid: `cudnn.allow_tf32` defaults to True
EVERYWHERE, and Turing (sm_75) has no TF32 units, so the flag is a no-op on the
primary and honoured on Ada. Recording the flag alone would assert the two boxes
AGREE, when that flag is the entire source of their disagreement. The field that
carries meaning is derived from the capability of the device actually used.
"""

from __future__ import annotations

from aiida_pino.scripts.live_tail import (
    FP32_CONV_REGIME,
    TF32_CONV_REGIME,
    numerical_regime_fields,
)


def test_turing_reports_fp32_even_though_the_flag_is_on():
    """sm_75 has no TF32 units: the flag is requested and not honoured."""
    fields = numerical_regime_fields(cudnn_allow_tf32=True, capability=(7, 5))
    assert fields["tf32_conv_active"] is False
    assert fields["numerical_regime"] == FP32_CONV_REGIME


def test_ada_reports_tf32_with_the_same_flag_value():
    """The identical flag, the opposite meaning. This is the whole point."""
    fields = numerical_regime_fields(cudnn_allow_tf32=True, capability=(8, 9))
    assert fields["tf32_conv_active"] is True
    assert fields["numerical_regime"] == TF32_CONV_REGIME


def test_turning_the_flag_off_on_ada_reports_fp32():
    fields = numerical_regime_fields(cudnn_allow_tf32=False, capability=(8, 9))
    assert fields["tf32_conv_active"] is False
    assert fields["numerical_regime"] == FP32_CONV_REGIME


def test_the_raw_flag_is_kept_beside_the_derived_one():
    """Raw values age better when someone later disputes the derivation."""
    fields = numerical_regime_fields(cudnn_allow_tf32=True, capability=(8, 9), matmul_allow_tf32=False)
    assert fields["cudnn_allow_tf32"] is True
    assert fields["matmul_allow_tf32"] is False
    assert fields["compute_capability"] == "8.9"


def test_a_FALSE_value_survives_capture():
    """`collect_environment` drops falsy values; these must not be dropped.

    `tf32_conv_active: False` is the PRIMARY's value and the one a comparison
    depends on. Dropped, it becomes absence — indistinguishable from a run that
    never captured it, which is the ambiguity this whole field exists to end.
    """
    from aiida_pino.scripts.live_tail import collect_environment

    env = collect_environment()
    numerics = env.get("numerics")
    if numerics is None:
        return  # no torch in this environment; nothing to assert
    assert "tf32_conv_active" in numerics
    assert numerics["numerical_regime"] in (FP32_CONV_REGIME, TF32_CONV_REGIME)


def test_no_capability_means_no_claim():
    """A CPU-only or unreadable device must not be labelled a regime."""
    assert numerical_regime_fields(cudnn_allow_tf32=True, capability=None) == {}


def test_the_regime_values_are_labels_not_a_schema():
    """Nothing may parse these, so they can extend."""
    assert isinstance(FP32_CONV_REGIME, str) and isinstance(TF32_CONV_REGIME, str)
    assert "-" in TF32_CONV_REGIME  # greppable, extensible to e.g. tf32-conv+bf16


def _summary(regime):
    return {"runtime": {"environment": {"hostname": "box", "gpu_name": "GPU", "numerics": regime}}}


def test_the_regime_is_promoted_to_a_queryable_extra():
    """A value inside a Dict is reachable by someone who already suspects a
    question; an extra is reachable by someone who does not."""
    from aiida_pino.parsers.common import runtime_extras_from_summary

    extras = runtime_extras_from_summary(
        _summary({"numerical_regime": "tf32-conv", "tf32_conv_active": True})
    )
    assert extras["numerical_regime"] == "tf32-conv"
    assert extras["numerical_regime_source"] == "captured"
    assert extras["env_hostname"] == "box"


def test_a_run_without_numerics_gets_no_regime_extra():
    """No claim is better than a default claim."""
    from aiida_pino.parsers.common import runtime_extras_from_summary

    extras = runtime_extras_from_summary(_summary(None))
    assert "numerical_regime" not in extras
    assert "numerical_regime_source" not in extras
