"""Which alloy a simulation is, as a query rather than a per-node dig.

Measured 2026-08-10 before the fix: **0 of 3,967** simulations carried any
material extra, so "which material is this?" meant opening
`simulation_metadata.flow3d_case_manifest.material.properties` node by node --
and the melt-pool temperature fallback, which needs a solidus and refuses to
guess one, could not be applied in bulk.
"""

from __future__ import annotations

from aiida_pino.data.pino_simulation import material_facts


def _metadata(material):
    return {"flow3d_case_manifest": {"material": material}}


def test_a_recorded_material_becomes_scalars():
    facts = material_facts(_metadata({
        "name": "Titanium Alloy Ti Al V IMI 318 Mills data",
        "file": "materials/ti_imi318_mills.json",
        "properties": {"solidus_temperature_k": 1873.15, "liquidus_temperature_k": 1923.15},
    }))
    assert facts["material_name"].startswith("Titanium Alloy")
    assert facts["solidus_k"] == 1873.15
    assert facts["liquidus_k"] == 1923.15
    assert facts["material_source"] == "manifest"


def test_a_missing_material_is_findable_not_silent():
    """The 84 `resampled_straight_path_data` cases have no material block at all.

    Their alloy is inferable from the family, but inference is not provenance:
    a guessed solidus would score one alloy with another's melting point, and
    `melt_metrics.melt_mask` refuses a guess for exactly that reason. So the
    gap is RECORDED as a gap -- `extras.material_source == "unrecorded"` finds
    all 85 in one filter.
    """
    facts = material_facts({"flow3d_case_manifest": {"derived": {}, "input": {}, "note": "x"}})
    assert facts == {"material_source": "unrecorded"}
    assert "solidus_k" not in facts, "never invent a melting point"

    assert material_facts({}) == {"material_source": "unrecorded"}
    assert material_facts({"flow3d_case_manifest": {"material": {}}}) == {
        "material_source": "unrecorded"
    }


def test_solidus_is_read_across_manifest_vintages():
    for key in ("solidus_temperature_k", "solidus_k", "T_solidus", "TS1"):
        facts = material_facts(_metadata({"name": "x", "properties": {key: 1600.0}}))
        assert facts["solidus_k"] == 1600.0, key


def test_a_material_without_properties_still_names_itself():
    facts = material_facts(_metadata({"flow3d_label": "TI64"}))
    assert facts["material_name"] == "TI64"
    assert facts["material_source"] == "manifest"
    assert "solidus_k" not in facts


def test_stamped_by_the_queryable_extras_path():
    from aiida_pino.data.pino_simulation import queryable_flow3d_extras

    extras = queryable_flow3d_extras(_metadata({
        "name": "Inconel 718", "properties": {"solidus_temperature_k": 1533.0},
    }))
    assert extras["material_name"] == "Inconel 718"
    assert extras["solidus_k"] == 1533.0
    assert extras["material_source"] == "manifest"
