"""The grid an eval actually served must reach the node, not only the log.

On the nine tabled lab evals every node records `derivation_profile.crop_z_top = 0`,
including the three that demonstrably served a 30-deep domain, and the only field that
differs between the groups -- `PINO_LAZY_CROP_Z_TOP` -- reads `''` on exactly those three,
which is how the loader spells NO crop. So the one discriminating value on the node asserts
the opposite of the truth, and neither stride nor crop is auditable from provenance
(2026-09-08). `[grid] serving` in the log was the only
correct record, and a log is not queryable.

This is a capture, not a computation: `NpzDirStore` already has these numbers when it prints
that line.
"""
import numpy as np

from test_npz_lazy_store import _manifest, _write_cases   # noqa: F401  (shared fixtures)

from lpno.models.pipeline.npz_lazy_store import NpzDirStore


def _store(tmp_path, **kw):
    return NpzDirStore(tmp_path, requested_fields=["Temperature"], **kw)


def test_served_grid_is_none_before_the_store_resolves_a_grid(tmp_path):
    _write_cases(tmp_path)
    assert _store(tmp_path).served_grid() is None, "nothing resolved yet"


def test_served_grid_reports_the_plain_case(tmp_path):
    nx, ny, nz, nt = _write_cases(tmp_path)
    s = _store(tmp_path)
    s.spatial_shape()   # what the pipeline calls at setup; populates the stamp
    g = s.served_grid()
    assert (g["nx"], g["ny"], g["nz"]) == (nx, ny, nz)
    assert g["raw"] == [nx, ny, nz]
    assert g["stride"] == [1, 1, 1] and g["crop_z_top"] == 0 and g["moving_window"] is None


def test_served_grid_reports_a_crop_the_profile_would_call_zero(tmp_path):
    """The exact case that was unauditable: a cropped depth against an uncropped raw one."""
    nx, ny, nz, nt = _write_cases(tmp_path)
    s = _store(tmp_path, crop_z_top=2)
    s.spatial_shape()   # what the pipeline calls at setup; populates the stamp
    g = s.served_grid()
    assert g["crop_z_top"] == 2
    assert g["nz"] == nz - 2 and g["raw"][2] == nz, "served depth differs from raw depth"


def test_served_grid_reports_stride(tmp_path):
    nx, ny, nz, nt = _write_cases(tmp_path)
    s = _store(tmp_path, spatial_stride=(2, 2, 1))
    s.spatial_shape()   # what the pipeline calls at setup; populates the stamp
    g = s.served_grid()
    assert g["stride"] == [2, 2, 1]
    assert g["nx"] < nx and g["raw"][0] == nx
