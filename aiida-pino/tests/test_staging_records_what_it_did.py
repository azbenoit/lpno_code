"""A collection's recorded shapes, and a staging's manifest, describe what is SERVED -- not
what was requested.

Two records lied the same way. `build_collection_metadata_from_simulations` computed
`array_shapes` by RUNNING the derivation on member 0 for every role, so a base collection
stamping the default `crop_z_top: 10` recorded z=30 while its members serve z=40 -- 706
such records were corrected by extras on 2026-09-09. And the staged `manifest.json` carried
that same `derivation_profile`, which the loader's `_variant_crop_z_top` read as the crop
BAKED into the arrays on disk. On a zero-copy staging nothing is on disk to have been
cropped, so the read-time crop `PINO_LAZY_CROP_Z_TOP=10` was refused as "cropping again",
and the refusal's own remedy -- drop the env crop -- would train on z=40 while every
record said 30 (2026-09-11).

Both fixed at the source: a non-derived collection records the raw extent, and every
staged manifest carries a `staging` block saying whether it is zero-copy and how many
z-cells were actually baked -- measured from what was written, not copied from the request.
"""

from __future__ import annotations

import json

from aiida_pino.utils import build_collection_metadata_from_simulations
from aiida_pino.utils.compose import compose_collection, member_inputs

from tests.test_runtime_export import _npz_file_sim

CROPPED = {"crop_z_top": 2, "stride": [1, 1, 1]}


def _raw_z(sim) -> int:
    """The member's own recorded extent (validated against real arrays 10/10 on 2026-09-09)."""
    shapes = sim.get_array_shapes() or {}
    return int(next(v for v in shapes.values() if len(v) >= 3)[-1])


def test_a_base_collection_records_the_raw_extent_even_with_a_crop_stamped(aiida_profile_clean, tmp_path):
    sims = [_npz_file_sim(tmp_path / f"b{i}", f"base_{i}", seed=i) for i in range(2)]
    metadata = build_collection_metadata_from_simulations(
        sims, collection_name="base", dataset_role="base", composition_kind="composed",
        extra_metadata={"derivation_profile": dict(CROPPED)},
    )
    node, _ = compose_collection(sims, metadata, **member_inputs(sims))
    stored = node.get_dataset_metadata()
    assert stored["spatial_shape"][-1] == _raw_z(sims[0]), "nothing applies a base collection's crop"
    assert stored["derivation_profile"]["crop_z_top"] == 2, "the request itself is still recorded"


def test_a_derived_collection_records_the_cropped_extent(aiida_profile_clean, tmp_path):
    sims = [_npz_file_sim(tmp_path / f"d{i}", f"derived_{i}", seed=i) for i in range(2)]
    metadata = build_collection_metadata_from_simulations(
        sims, collection_name="derived", dataset_role="derived", composition_kind="composed",
        extra_metadata={"derivation_profile": dict(CROPPED)},
    )
    node, _ = compose_collection(sims, metadata, **member_inputs(sims))
    assert node.get_dataset_metadata()["spatial_shape"][-1] == _raw_z(sims[0]) - 2


def test_a_zero_copy_staging_says_so_and_bakes_nothing(aiida_profile_clean, tmp_path):
    sims = [_npz_file_sim(tmp_path / f"z{i}", f"zc_{i}", seed=i) for i in range(2)]
    metadata = build_collection_metadata_from_simulations(
        sims, collection_name="zc", dataset_role="base", composition_kind="composed",
        extra_metadata={"derivation_profile": dict(CROPPED)},
    )
    node, _ = compose_collection(sims, metadata, **member_inputs(sims))
    out = tmp_path / "staged_zc"
    node.export_runtime_arraydir(out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["staging"] == {"zero_copy": True, "baked_crop_z_top": 0}


def test_a_materialised_cropped_staging_reports_the_measured_bake(aiida_profile_clean, tmp_path):
    sims = [_npz_file_sim(tmp_path / f"m{i}", f"mat_{i}", seed=i) for i in range(2)]
    metadata = build_collection_metadata_from_simulations(
        sims, collection_name="mat", dataset_role="derived", composition_kind="composed",
        extra_metadata={"derivation_profile": dict(CROPPED)},
    )
    node, _ = compose_collection(sims, metadata, **member_inputs(sims))
    out = tmp_path / "staged_mat"
    node.export_runtime_arraydir(out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["staging"]["zero_copy"] is False
    assert manifest["staging"]["baked_crop_z_top"] == 2, "raw z minus the z actually written"
