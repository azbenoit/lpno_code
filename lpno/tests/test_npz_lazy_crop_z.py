"""Gas-side z-crop applied at READ time, so a cropped dataset needs no materialising.

Requested by the Aiida session 2026-08-25. `crop_z_top` in aiida-pino's derivation
profile makes a derivation "transforming", which flips the collection off the
pack-index path (zero copy, stored object referenced in place) onto materialisation,
which writes UNCOMPRESSED arrays: 2.70 GB per member against ~23 MB stored, ~413 GB
for the 96-member train split against 167 GB free. An ungated build reached 159 GB
for 62 members before stalling with root at 1.3 GB, killing training 68720
mid-checkpoint-write.

Read-time is the only way the feature works on a full split: the dataset stays
pack-indexed at zero copy.

The failure worth catching is NOT the shape -- it is a TARGET that did not get
cropped with its input, which trains fine against a differently-shaped ground truth
and gives a wrong number. Every field goes through one slicing expression in
`_case_uncached`, so inputs and targets are cropped by construction rather than by
discipline; these tests pin that.
"""
from __future__ import annotations

import numpy as np
import json
import pytest

from lpno.models.pipeline.npz_lazy_store import NpzDirStore
from lpno.custom_utils import npz_datasource as nds

from test_npz_lazy_store import _write_cases

FIELDS = ["Temperature", "Velocity", "Fraction Of Fluid", "Pressure"]


def _store(tmp_path, **kw):
    return NpzDirStore(tmp_path, requested_fields=FIELDS, **kw)


def test_crop_removes_only_the_top_of_z(tmp_path):
    _write_cases(tmp_path, n=1)
    full = _store(tmp_path).read_case(0)
    cropped = _store(tmp_path, crop_z_top=2).read_case(0)

    for key in full:
        a, b = full[key], cropped[key]
        assert a.shape[:-1] == b.shape[:-1], f"{key}: only z may change"
        assert b.shape[-1] == a.shape[-1] - 2, f"{key}: z not cropped by 2"
        # the retained cells must be the SAME values, from the bottom
        np.testing.assert_array_equal(b, a[..., : a.shape[-1] - 2], err_msg=key)


def test_inputs_and_targets_crop_to_the_same_extent(tmp_path):
    """The failure that trains fine and reports a wrong number."""
    _write_cases(tmp_path, n=1)
    raw = _store(tmp_path, crop_z_top=3).read_case(0)
    extents = {k: v.shape[-1] for k, v in raw.items()}
    assert len(set(extents.values())) == 1, f"z extents disagree across fields: {extents}"


def test_reported_grid_matches_what_is_served(tmp_path):
    """spatial_shape() must not promise cells the reader will not return."""
    _write_cases(tmp_path, n=1)
    s = _store(tmp_path, crop_z_top=2)
    assert s.spatial_shape()[-1] == s.read_case(0)["Temperature"].shape[-1]


def test_crop_composes_with_stride(tmp_path):
    """Crop is in RAW cells, then the stride applies -- matching the profile's units."""
    _write_cases(tmp_path, n=1)
    nz_raw = _store(tmp_path).read_case(0)["Temperature"].shape[-1]
    s = _store(tmp_path, crop_z_top=2, spatial_stride=(1, 1, 2))
    expected = len(range(0, nz_raw - 2, 2))
    assert s.read_case(0)["Temperature"].shape[-1] == expected
    assert s.spatial_shape()[-1] == expected


def test_zero_crop_is_byte_identical(tmp_path):
    _write_cases(tmp_path, n=1)
    a = _store(tmp_path).read_case(0)["Temperature"]
    b = _store(tmp_path, crop_z_top=0).read_case(0)["Temperature"]
    np.testing.assert_array_equal(a, b)


def test_crop_larger_than_the_grid_is_refused(tmp_path):
    _write_cases(tmp_path, n=1)
    nz = _store(tmp_path).read_case(0)["Temperature"].shape[-1]
    with pytest.raises(ValueError, match="crop_z_top"):
        _store(tmp_path, crop_z_top=nz)


def test_negative_crop_is_refused(tmp_path):
    _write_cases(tmp_path, n=1)
    with pytest.raises(ValueError, match="crop_z_top"):
        _store(tmp_path, crop_z_top=-1)


def test_no_runtime_cache_directory_is_created(tmp_path):
    """The point of the feature: still pack-indexed, nothing materialised."""
    _write_cases(tmp_path, n=1)
    before = {p.name for p in tmp_path.iterdir()}
    _store(tmp_path, crop_z_top=2).read_case(0)
    assert {p.name for p in tmp_path.iterdir()} == before, "read-time crop wrote something"


def test_env_var_applies_the_crop(tmp_path, monkeypatch, capsys):
    """`PINO_LAZY_CROP_Z_TOP` is the submit-time surface, matching PINO_LAZY_STRIDE."""
    from types import SimpleNamespace

    from lpno.models.pipeline import npz_lazy_store as mod

    _write_cases(tmp_path, n=1)
    monkeypatch.setenv("PINO_LAZY_CROP_Z_TOP", "2")
    run_cfg = SimpleNamespace(vtk_fields=FIELDS, lazy_npz=True)
    store = mod.NpzDirStore(tmp_path, requested_fields=FIELDS,
                            crop_z_top=int(mod.os.environ["PINO_LAZY_CROP_Z_TOP"]))
    full = mod.NpzDirStore(tmp_path, requested_fields=FIELDS)
    assert store.spatial_shape()[-1] == full.spatial_shape()[-1] - 2


def test_the_grid_line_announces_the_crop(tmp_path, capsys):
    """The effective grid must be ground truth in the log, not inferred from config.

    A 2026-08-19 run read as "the window did not apply" because the announcement
    printed the pre-window grid; the same trap applies to a crop.
    """
    _write_cases(tmp_path, n=1)
    _store(tmp_path, crop_z_top=2).spatial_shape()
    assert "crop_z_top 2" in capsys.readouterr().out


def test_eager_path_says_it_ignores_the_crop(tmp_path, monkeypatch, capsys):
    """The duplicated-builder trap: one read path honours it, the other must say so.

    Flagged by the Aiida session -- a fix landing at one read path while a sibling
    serves the old behaviour has shipped in this codebase more than once. The eager
    `load_consolidated_npz` does not apply the crop, so a run using it would train
    on the full z extent while its profile claims a crop, and the only symptom is a
    number.
    """
    from lpno.custom_utils.npz_datasource import load_consolidated_npz

    _write_cases(tmp_path, n=1)
    monkeypatch.setenv("PINO_LAZY_CROP_Z_TOP", "2")
    try:
        load_consolidated_npz(tmp_path, requested_fields=["Temperature"])
    except Exception:
        pass  # the load itself may need more setup; the warning fires before it
    out = capsys.readouterr().out
    assert "PINO_LAZY_CROP_Z_TOP" in out and "EAGER" in out, out[:300]


def _write_manifest(root, **derivation):
    import json
    (root / "manifest.json").write_text(json.dumps(
        {"dataset_metadata": {"derivation_profile": derivation}}))


def test_double_crop_is_refused(tmp_path, monkeypatch):
    """A baked crop plus a read-time crop subtracts twice, silently.

    Raised by the Aiida session before wiring the profile through: the stride axis
    already guards this (`_variant_stride`), the crop axis did not. Baked-cropped
    variants already exist (pk 68783, 96 members; 69438), so the moment a profile's
    `crop_z_top` also becomes an env var, every one of them is a double-crop:
    40 -> 10 on disk, then -30 again at read.

    Same failure class as a target that did not crop with its input -- it trains and
    reports a wrong number -- arriving by a different route.
    """
    from lpno.models.pipeline import npz_lazy_store as mod

    _write_cases(tmp_path, n=1)
    _write_manifest(tmp_path, crop_z_top=3)
    monkeypatch.setenv("PINO_LAZY_CROP_Z_TOP", "3")
    with pytest.raises(ValueError, match="already cropped|crop_z_top"):
        mod._resolve_read_time_crop(tmp_path, "3")


def test_baked_crop_alone_is_fine(tmp_path):
    """The variant carries the crop and no env var: nothing to compose, no refusal."""
    from lpno.models.pipeline import npz_lazy_store as mod

    _write_cases(tmp_path, n=1)
    _write_manifest(tmp_path, crop_z_top=3)
    assert mod._resolve_read_time_crop(tmp_path, "") == 0


def test_env_crop_on_an_uncropped_variant_is_fine(tmp_path):
    from lpno.models.pipeline import npz_lazy_store as mod

    _write_cases(tmp_path, n=1)
    _write_manifest(tmp_path, stride=[1, 1, 1])
    assert mod._resolve_read_time_crop(tmp_path, "2") == 2


def test_unreadable_manifest_does_not_block_a_run(tmp_path):
    """Same tolerance as _variant_stride: a bad manifest must not stop a working run."""
    from lpno.models.pipeline import npz_lazy_store as mod

    _write_cases(tmp_path, n=1)
    (tmp_path / "manifest.json").write_text("{not json")
    assert mod._resolve_read_time_crop(tmp_path, "2") == 2


def _staged(tmp_path, manifest: dict, *, pack_index: bool):
    root = tmp_path / "staged"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest))
    if pack_index:
        (root / "pack_index.json").write_text("{}")
    return root


def test_a_zero_copy_staging_has_nothing_baked_whatever_its_profile_says(tmp_path):
    """The manifest's `derivation_profile` is the REQUEST. On a zero-copy staging no array
    was written, so nothing was cropped, and the read-time crop is the only way the crop
    happens at all. cj102455 (2026-09-11) was refused for "cropping again" here, and the
    refusal's remedy would have trained uncropped data under a record claiming crop 10.
    """
    from lpno.models.pipeline.npz_lazy_store import _variant_crop_z_top

    root = _staged(tmp_path, {"dataset_metadata": {"derivation_profile": {"crop_z_top": 10}}}, pack_index=True)
    assert _variant_crop_z_top(root) == 0


def test_the_staging_block_is_the_measured_bake_when_present(tmp_path):
    from lpno.models.pipeline.npz_lazy_store import _variant_crop_z_top

    root = _staged(tmp_path, {"staging": {"zero_copy": False, "baked_crop_z_top": 3},
                              "dataset_metadata": {"derivation_profile": {"crop_z_top": 10}}}, pack_index=False)
    assert _variant_crop_z_top(root) == 3, "what was written, not what was asked"


def test_an_old_manifest_without_the_block_still_reads_the_profile(tmp_path):
    """Materialised stagings written before the block exist on disk; keep them readable."""
    from lpno.models.pipeline.npz_lazy_store import _variant_crop_z_top

    root = _staged(tmp_path, {"dataset_metadata": {"derivation_profile": {"crop_z_top": 10}}}, pack_index=False)
    assert _variant_crop_z_top(root) == 10
