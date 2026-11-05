"""Geometry numbers must be self-describing about their cell scale.

`melt_geometry_stats` derives ONE cell size from the z spacing and applies it to
all three axes. On a `stride=[2,2,1]` dataset dz=10 um but dx=dy=20 um, so every
x/y length is 2x too small and every volume 4x too small (measured 2026-08-04).

The stride is stored in `derivation_profile`, so the information is not lost --
but using it requires knowing the stride, the spacing it implies, AND that the
metric is z-isotropic. That three-step chain is exactly what makes a silent wrong
comparison likely, and a written note elsewhere does not help because it is not in the
data a reader queries. So the scale is recorded beside the numbers, the same way
`temperature_scale_k` already is.
"""

from __future__ import annotations

from aiida_pino.console.model_metrics import geometry_scale


def test_strided_eval_is_flagged_not_comparable():
    g = geometry_scale({"stride": [2, 2, 1]})
    assert g["axes_isotropic"] is False
    assert g["comparable_to_isotropic_evals"] is False
    assert g["length_correction_xy"] == 2.0
    assert g["volume_correction"] == 4.0


def test_full_resolution_eval_is_marked_comparable():
    g = geometry_scale({"stride": [1, 1, 1]})
    assert g["axes_isotropic"] is True
    assert g["comparable_to_isotropic_evals"] is True


def test_unknown_stride_does_not_claim_comparability():
    """Absence of a stride must read as 'unknown', never as 'fine'."""
    for profile in ({}, None, {"stride": None}, {"stride": [2, 2]}):
        g = geometry_scale(profile)
        assert g["axes_isotropic"] is None
        assert "comparable_to_isotropic_evals" not in g


def test_note_is_always_present():
    for profile in ({"stride": [1, 1, 1]}, {"stride": [2, 2, 1]}, {}):
        assert "z spacing" in geometry_scale(profile)["note"]
