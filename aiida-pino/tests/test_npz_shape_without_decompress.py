"""Reading a single field must not decompress the whole payload.

`canonicalize_flow3d_case_arrays` classified members with
`is_metadata_array(name, np.shape(archive[name]))`, and indexing an NpzFile
DECOMPRESSES that member. So the classification pass materialised every array in
the payload just to read shapes -- on every call, even when `only=[one_field]`.

`get_array(name)` calls that function once per field, and one
`compute_case_metrics` needs ~8 arrays (x, y, z, time_s, Fraction Of Fluid,
Melt Region, Temperature, ...). The whole payload was therefore decompressed
about eight times per case. Measured on the core-dataset backfill: 3193 s/case,
against ~4 min for the same work through the console.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiida_pino.utils.flow3d_npz import member_shape


def _case_npz(tmp_path, compressed=True):
    path = tmp_path / "payload.npz"
    payload = {
        "x": np.linspace(0, 1e-3, 6).astype(np.float32),
        "y": np.linspace(0, 5e-4, 5).astype(np.float32),
        "z": np.linspace(-3e-4, 1e-4, 4).astype(np.float32),
        "time_s": np.linspace(0, 1e-3, 3).astype(np.float64),
        "Temperature": np.zeros((3, 6, 5, 4), np.float32),
        "Melt Region": np.zeros((3, 6, 5, 4), np.float32),
        "frame_stride": np.array(2),          # rank-0 scalar => metadata
    }
    (np.savez_compressed if compressed else np.savez)(path, **payload)
    return path, payload


@pytest.mark.parametrize("compressed", [True, False])
def test_member_shape_matches_the_real_shape(tmp_path, compressed):
    path, payload = _case_npz(tmp_path, compressed)
    archive = np.load(path)
    for name, array in payload.items():
        assert member_shape(archive, name) == array.shape, name


def test_member_shape_reads_the_header_not_the_data(tmp_path, monkeypatch):
    """The point of the helper: it must not go through __getitem__."""
    path, _ = _case_npz(tmp_path)
    archive = np.load(path)

    def _boom(self, key):  # pragma: no cover - fires only on regression
        raise AssertionError(f"decompressed {key!r} just to read its shape")

    monkeypatch.setattr(type(archive), "__getitem__", _boom)
    assert member_shape(archive, "Temperature") == (3, 6, 5, 4)


def test_falls_back_for_a_plain_mapping():
    """Ingest paths pass dict-likes; those have no zip member to peek at."""
    payload = {"Temperature": np.zeros((2, 3, 4, 5), np.float32)}

    class _Mapping(dict):
        files = ["Temperature"]

    assert member_shape(_Mapping(payload), "Temperature") == (2, 3, 4, 5)
