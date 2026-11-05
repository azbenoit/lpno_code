"""A materialised derived dataset must be able to carry per-case material channels.

2026-08-25. Every steady arm died at step 0 with
`KeyError: 'mat_k_liq'` (TrainEvalWorkChain -> PinoTrainCalculation, exit 500).
The run was otherwise healthy: the 22-key list reached the hydra command line, both
derivations materialised, both collections staged.

Cause: `BUILDABLE_DERIVED_FIELDS` was `frozenset({"conduction_field"})`, and the
materialiser writes a fixed array set, so a materialised variant physically could not
contain the channels and requesting them was a step-0 KeyError. Same gap the packaged
steady config already documents for trajectory channels (`KeyError: 'beam_x'`).

Material channels need none of `conduction_field`'s lab-frame ordering: they are ONE SCALAR
PER CASE, constant in space and time, so the beam-following crop and the 30-frame average
are identity operations on them. There is no build-then-transform hazard because the value
does not depend on the frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiida_pino.utils.dataset_derivation import (
    BUILDABLE_DERIVED_FIELDS,
    buildable_derived_fields,
    validate_derivation_profile,
)

MATERIAL_SIX = ("mat_k_liq", "mat_H_king", "mat_X_keyhole", "mat_tv_ratio", "mat_theta_v", "mat_theta_m")


@pytest.mark.requires_lpno
def test_the_material_channels_are_requestable():
    allowed = buildable_derived_fields()
    missing = [name for name in MATERIAL_SIX if name not in allowed]
    assert not missing, f"still unrequestable: {missing}"
    for name in MATERIAL_SIX:
        assert validate_derivation_profile({"derived_fields": [name]})["derived_fields"] == [name]


def test_conduction_field_is_still_requestable():
    """The widening must not drop what was already there."""
    assert "conduction_field" in buildable_derived_fields()
    assert "conduction_field" in BUILDABLE_DERIVED_FIELDS


def test_a_typo_is_still_refused_at_submit():
    """The allowlist exists so a typo fails before staging, not after minutes of work."""
    with pytest.raises(ValueError, match="unknown field"):
        validate_derivation_profile({"derived_fields": ["mat_k_liqq"]})


def test_the_list_is_not_a_hardcoded_copy():
    """Restating the names here is how this gap appeared; they come from the datasource.

    If `lpno` gains a channel, it becomes requestable with no edit here. The
    static set is only the fallback for a machine without the sibling package.
    """
    from lpno.custom_utils.npz_datasource import ALL_MATERIAL_CHANNEL_NAMES

    assert frozenset(ALL_MATERIAL_CHANNEL_NAMES) <= buildable_derived_fields()


def test_material_and_path_channels_are_routed_to_different_builders(monkeypatch):
    """A material channel must not be sent to the path builder, which would drop it.

    `_build_path_channels` filters to `PATH_CHANNEL_NAMES` and returns {} for anything
    else, so routing a material channel there yields "the datasource returned no such
    channel" -- a confusing refusal for a channel that is perfectly buildable.
    """
    from aiida_pino.data import pino_dataset_collection as pdc
    import lpno.custom_utils.npz_datasource as ds

    seen: dict[str, list] = {"path": [], "material": []}
    monkeypatch.setattr(ds, "_build_path_channels",
                        lambda names, *a, **k: (seen["path"].append(list(names)) or
                                                {n: np.zeros((1, 1, 1, 1, 1, 1), np.float32) for n in names}))
    monkeypatch.setattr(ds, "_build_material_channels",
                        lambda names, manifest, shape: (seen["material"].append(list(names)) or
                                                        {n: np.full(shape, 1.0, np.float32) for n in names}))

    axes = (np.zeros(2), np.zeros(3), np.zeros(4), np.zeros(5))
    pdc.extra_fields_from_axes(
        ["conduction_field", "mat_k_liq"], {"case_name": "ckc_000"}, *axes, label="ckc_000"
    )
    assert seen["path"] == [["conduction_field"]]
    assert seen["material"] == [["mat_k_liq"]]


def test_a_material_channel_is_broadcast_to_the_canonical_shape(monkeypatch):
    """Constant per case, but shaped like every other field so the transform loop is blind to it."""
    from aiida_pino.data import pino_dataset_collection as pdc
    import lpno.custom_utils.npz_datasource as ds

    monkeypatch.setattr(ds, "_build_material_channels",
                        lambda names, manifest, shape: {n: np.full(shape, 7.0, np.float32) for n in names})
    out = pdc.extra_fields_from_axes(
        ["mat_k_liq"], {"case_name": "ckc_000"},
        np.zeros(6), np.zeros(4), np.zeros(3), np.zeros(5), label="ckc_000",
    )
    assert out["mat_k_liq"].shape == (5, 1, 6, 4, 3)
    assert np.unique(out["mat_k_liq"]).tolist() == [7.0]
