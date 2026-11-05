"""A material fingerprint that matches must mean the materials are the same.

Round 7 finding: the first version hashed seven hand-picked prepin scalars, and
**446 of 1115 runs shared a fingerprint with a materially different run**. The
pair that settled it, both from batch `t2_nist195_transient_20260806`:

    33447  t2n195_ctrl     vaporization_temperature_k 3520  4315 cycles
    33511  t2n195_tv3425   vaporization_temperature_k 3425  4442 cycles

A control and its arm, 95 K apart, 2.9% different in solver work, declared
identical -- because `SCALAR_KEYS` never named `tv1`. Hand-picking cannot be
complete; the authority is `run_summary.manifest.material`, which carries all
15 properties plus the optical block.
"""

from __future__ import annotations

from aiida_pino.utils.material_identity import material_extras, material_identity

PREPIN = """
    rhof=4420,
    cv1=546,
#sigma(1)
1524.38  1.58287
1740.89  1.48781
#end sigma(1)
"""

CTRL = {
    "name": "ti_imi318_mills",
    "properties": {
        "solid_density_kg_m3": 4420.0,
        "surface_tension_n_m": 1.58287,
        "vaporization_temperature_k": 3520.0,
    },
    "optical": {"absorptivity": 0.35},
}
ARM = {**CTRL, "properties": {**CTRL["properties"], "vaporization_temperature_k": 3425.0}}


def test_a_property_the_scalar_list_never_named_still_changes_the_fingerprint():
    """The exact defect: tv is not in SCALAR_KEYS, and the prepin text is identical."""
    ctrl = material_extras(PREPIN, CTRL)
    arm = material_extras(PREPIN, ARM)
    assert ctrl["material_fingerprint"] != arm["material_fingerprint"]
    # And the differing value itself is queryable, not just hashed away.
    assert ctrl["material_vaporization_temperature_k"] == 3520.0
    assert arm["material_vaporization_temperature_k"] == 3425.0


def test_the_same_material_still_matches():
    """A fingerprint nobody can match is as useless as one everybody matches."""
    assert (
        material_extras(PREPIN, CTRL)["material_fingerprint"]
        == material_extras(PREPIN, dict(CTRL))["material_fingerprint"]
    )


def test_the_optical_block_counts_too():
    other = {**CTRL, "optical": {"absorptivity": 0.42}}
    assert material_extras(PREPIN, CTRL)["material_fingerprint"] != material_extras(PREPIN, other)["material_fingerprint"]


def test_a_changed_table_still_changes_the_fingerprint():
    """The prepin half must not be lost now that the manifest half exists."""
    other = PREPIN.replace("1.48781", "1.20000")
    assert material_extras(PREPIN, CTRL)["material_fingerprint"] != material_extras(other, CTRL)["material_fingerprint"]
    assert material_identity(PREPIN)["tables"]["sigma"]["rows"] == 2


def test_no_material_information_yields_nothing():
    assert material_extras("", None) == {}
    assert material_extras("nothing useful here", {}) == {}


def test_stamping_never_fails_a_parse(aiida_profile_clean):
    """This runs in the daemon's parser; a raise there loses the run's artifacts.

    Round 7: `stamp_from_artifacts` guarded its two repository reads but not
    `material_extras` or the extras writes, and the parser path had a measured
    0-for-2 success rate so nothing would have noticed.
    """
    from aiida import orm

    from aiida_pino.utils.material_identity import stamp_from_artifacts

    folder = orm.FolderData()
    folder.base.repository.put_object_from_bytes(b"fake prepin\n", "prepin.case")
    folder.store()
    node = orm.Dict(dict={}).store()

    # Unparseable content: no crash, nothing invented.
    assert stamp_from_artifacts(node, folder) == {}

    # A folder with no prepin at all, and no folder at all.
    assert stamp_from_artifacts(node, orm.FolderData().store()) == {}
    assert stamp_from_artifacts(node, None) == {}

    # A material block that is the wrong shape entirely must not raise either.
    assert stamp_from_artifacts(node, folder, {"properties": "not a dict"}) == {}


def test_the_prepin_alone_still_works():
    """Older runs have no manifest material block; they must still fingerprint."""
    extras = material_extras(PREPIN, None)
    assert extras["material_fingerprint"]
    assert extras["material_rhof"] == 4420.0
    assert extras["material_sigma_rows"] == 2
