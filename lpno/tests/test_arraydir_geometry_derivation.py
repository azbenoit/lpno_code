"""A staged arraydir runtime must define `run_cfg`'s grid, not the config's hardcode.

The arraydir loader (`load_runtime_store`) serves arrays at native shape and never sees
`custom.geometry`, yet config loading only derived the domain from data for consolidated
NPZ. The three anchorbox steady arms (m82910 FNO, m83376 RIGNO, m83672 GAOT, 2026-09-01)
therefore trained on 61x60x55 tensors while `run_cfg` carried the inherited
`resolution: [90, 40, 30]`: validator xy slices were cut at z-index 27 of 55 instead of
near the surface, the FNO Nyquist cap was computed for 40x30, and bounds/dx described a
box the data did not have. The tensors themselves were never squeezed -- which is why it
was silent.

aiida-pino records the served grid in the runtime manifest (`dataset_metadata.grid_params`,
`spatial_shape`); for a moving-frame variant that grid is post-window (x already reduced
by steps-1), so the config-time x shift must not be applied on top.
"""
import json

import numpy as np
import pytest

from lpno.custom_utils.npz_datasource import derive_geometry_from_data
from lpno.models.pipeline import config_loading
from lpno.models.pipeline.lazy_runtime import (
    RUNTIME_FORMAT_V2,
    runtime_grid_params,
    runtime_window_is_baked,
)

GRID = {"resolution": [61, 60, 55], "x_range": [-4.45e-4, 4.45e-4],
        "y_range": [-2.95e-4, 2.95e-4], "z_range": [-4.45e-4, 9.5e-5]}
HARDCODE = {"resolution": [90, 40, 30], "x_range": [-4.5e-4, 4.5e-4],
            "y_range": [-2e-4, 2e-4], "z_range": [-3e-4, 0.0]}


def _runtime(root, *, moving=True, steps=30, with_grid=True):
    member = root / "members" / "simulation_0000"
    member.mkdir(parents=True)
    arr = np.zeros((1, 4, 3, 2), np.float32)   # tiny; the manifest carries the grid
    np.save(member / "Temperature.npy", arr)
    (member / "manifest.json").write_text(json.dumps({
        "format": RUNTIME_FORMAT_V2, "source_kind": "simulation",
        "simulation_metadata": {"case_name": "c0"},
        "array_entries": [{"original_name": "Temperature", "filename": "Temperature.npy",
                           "shape": list(arr.shape)}],
    }))
    meta = {"moving_ref_frame": moving, "moving_frame_steps": steps,
            "derivation_profile": {"moving_ref_frame": moving, "moving_frame_steps": steps}}
    if with_grid:
        meta["grid_params"] = GRID
        meta["spatial_shape"] = GRID["resolution"]
    (root / "manifest.json").write_text(json.dumps({
        "format": RUNTIME_FORMAT_V2, "source_kind": "dataset_collection",
        "dataset_metadata": meta,
        "members": [{"relative_path": "members/simulation_0000"}],
    }))
    return root


def test_runtime_manifest_grid_is_read(tmp_path):
    rt = _runtime(tmp_path)
    assert runtime_grid_params(rt)["resolution"] == (61, 60, 55)
    assert derive_geometry_from_data(rt)["resolution"] == (61, 60, 55)


def test_hardcoded_config_resolution_is_replaced_for_arraydir(tmp_path):
    rt = _runtime(tmp_path)
    grid, baked = config_loading._grid_params_for_data(dict(HARDCODE), str(rt), "flow3D")
    assert grid["resolution"] == (61, 60, 55), "the served grid, not the inherited [90, 40, 30]"
    assert grid["z_range"] == pytest.approx((-4.45e-4, 9.5e-5))
    assert baked is True, "steps30 variant: the x-window is already in the served extent"


def test_window_not_baked_for_lab_frame_variant(tmp_path):
    rt = _runtime(tmp_path, moving=False, steps=1)
    _, baked = config_loading._grid_params_for_data(dict(HARDCODE), str(rt), "flow3D")
    assert baked is False
    assert runtime_window_is_baked(rt) is False


def test_runtime_without_grid_refuses_rather_than_keeping_the_hardcode(tmp_path):
    rt = _runtime(tmp_path, with_grid=False)
    with pytest.raises(ValueError, match="re-stage|Re-stage"):
        derive_geometry_from_data(rt)


def test_non_runtime_directory_keeps_the_config_box(tmp_path):
    grid, baked = config_loading._grid_params_for_data(dict(HARDCODE), str(tmp_path), "flow3D")
    assert grid["resolution"] == [90, 40, 30] and baked is False
