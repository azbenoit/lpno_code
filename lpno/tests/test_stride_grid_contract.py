"""The effective training resolution must be stated once and enforced everywhere.

Two ways a run can silently train or score on the wrong grid, both of which hit
this campaign:

1. DOUBLE STRIDING. aiida-pino can bake a stride into the dataset VARIANT (the
   staged NPZ are already coarse, e.g. `var_1_s2x2x1_time_noref` holding
   45x20x40), and `PINO_LAZY_STRIDE` strides again at read time. Set both and the
   effective grid is the PRODUCT -- 22x10x40 from a 90x40x40 dataset -- which no
   log line, config or manifest states on its own.

2. TRAIN/EVAL RESOLUTION MISMATCH. On 2026-08-19 both spkh123 arms (55755 GAOT,
   55761 RIGNO) trained on the strided variant and were then scored on the FULL
   grid: `[grid] serving 45x20x40` in the train calc against
   `[grid] serving 90x40x40` in every eval calc. Nothing objected. The models
   were never asked to predict the grid they were graded on, and a graph network
   (RIGNO rebuilds its graph from the coordinates) is hit far harder by that than
   a token-based one, so the measured architecture gap was partly an artefact.

The contract: the grid the training store served is stamped next to the
checkpoint, and any later store that serves a different grid must say so out
loud. A deliberate resolution change (super-resolution study) stays possible via
`super_res_factor` or an explicit env acknowledgement.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from lpno.custom_utils import npz_datasource as nds
from lpno.models.pipeline.npz_lazy_store import (
    GRID_STAMP_NAME,
    maybe_build_npz_store,
)

from test_npz_lazy_store import _write_cases


def _run_cfg(tmp_path, network_dir, *, eval_only=False, super_res=1):
    from types import SimpleNamespace
    return SimpleNamespace(
        cfg=OmegaConf.create({"network_dir": str(network_dir)}),
        data_directory_path=str(tmp_path),
        vtk_fields=["Temperature", "Pressure"],
        keys=SimpleNamespace(str_fno_input_keys=["V_scan", "P", "x", "y", "z"]),
        moving_ref_frame=False,
        time_as_batch=True,
        time_as_dim=False,
        include_time=True,
        lazy_npz=True,
        eval_only=eval_only,
        super_res_factor=super_res,
    )


def _variant_manifest(data_dir: Path, stride):
    """Write the staged-dataset manifest aiida-pino leaves beside the members."""
    (data_dir / "manifest.json").write_text(json.dumps({
        "dataset_identifier": "test__var_1_time_noref",
        "dataset_metadata": {"derivation_profile": {"stride": list(stride)}},
    }))


# --- 1. double striding -------------------------------------------------------

def test_variant_stride_and_env_stride_together_are_refused(tmp_path, monkeypatch):
    """The exact silent-4x-coarsening case: both mechanisms armed at once."""
    data = tmp_path / "data"; data.mkdir()
    _write_cases(data, nx=10, ny=8, nz=6)
    _variant_manifest(data, (2, 2, 1))
    monkeypatch.setenv("PINO_LAZY_STRIDE", "2,2,1")
    with pytest.raises(ValueError, match="already strided"):
        maybe_build_npz_store(data, _run_cfg(data, tmp_path / "net"))


def test_variant_stride_alone_is_fine(tmp_path, monkeypatch):
    data = tmp_path / "data"; data.mkdir()
    _write_cases(data, nx=10, ny=8, nz=6)
    _variant_manifest(data, (2, 2, 1))
    monkeypatch.delenv("PINO_LAZY_STRIDE", raising=False)
    store = maybe_build_npz_store(data, _run_cfg(data, tmp_path / "net"))
    assert store.spatial_shape() == (10, 8, 6)  # files are already coarse


def test_env_stride_alone_is_fine(tmp_path, monkeypatch):
    data = tmp_path / "data"; data.mkdir()
    _write_cases(data, nx=10, ny=8, nz=6)
    _variant_manifest(data, (1, 1, 1))
    monkeypatch.setenv("PINO_LAZY_STRIDE", "2,2,1")
    store = maybe_build_npz_store(data, _run_cfg(data, tmp_path / "net"))
    assert store.spatial_shape() == (5, 4, 6)


def test_absent_manifest_does_not_block(tmp_path, monkeypatch):
    """Back-compat: datasets staged before the manifest carried a stride."""
    data = tmp_path / "data"; data.mkdir()
    _write_cases(data, nx=10, ny=8, nz=6)
    monkeypatch.setenv("PINO_LAZY_STRIDE", "2,2,1")
    store = maybe_build_npz_store(data, _run_cfg(data, tmp_path / "net"))
    assert store.spatial_shape() == (5, 4, 6)


# --- 2. train/eval resolution contract ----------------------------------------

def test_training_stamps_the_served_grid(tmp_path, monkeypatch):
    data = tmp_path / "data"; data.mkdir()
    net = tmp_path / "net"; net.mkdir()
    _write_cases(data, nx=10, ny=8, nz=6)
    monkeypatch.delenv("PINO_LAZY_STRIDE", raising=False)
    store = maybe_build_npz_store(data, _run_cfg(data, net))
    assert store.spatial_shape() == (10, 8, 6)
    stamp = json.loads((net / GRID_STAMP_NAME).read_text())
    assert tuple(stamp["grid"]) == (10, 8, 6)


def test_eval_on_a_different_grid_is_reported_not_refused(tmp_path, monkeypatch, capsys):
    """Standing policy: a strided-trained model IS scored at full resolution.

    Changed from a refusal 2026-08-19. Full resolution is the resolution a result is
    claimed at, so blocking that blocks the intended workflow; the shift is a number
    to report (measured: RIGNO 0.18%, GAOT +16% MAE), not an error. The defect the
    contract exists for -- training at half resolution UNINTENTIONALLY -- is caught
    by the double-stride refusal and the train-vs-validation check instead.
    """
    net = tmp_path / "net"; net.mkdir()
    (net / GRID_STAMP_NAME).write_text(json.dumps({"grid": [5, 4, 6]}))
    data = tmp_path / "data"; data.mkdir()
    _write_cases(data, nx=10, ny=8, nz=6)          # full grid at eval
    monkeypatch.delenv("PINO_LAZY_STRIDE", raising=False)
    monkeypatch.delenv("PINO_ALLOW_GRID_SHIFT", raising=False)
    store = maybe_build_npz_store(data, _run_cfg(data, net, eval_only=True))
    assert store.spatial_shape() == (10, 8, 6)
    out = capsys.readouterr().out
    assert "trained on 5x4x6" in out and "scoring on 10x8x6" in out, out


def test_eval_on_the_same_grid_passes(tmp_path, monkeypatch):
    net = tmp_path / "net"; net.mkdir()
    (net / GRID_STAMP_NAME).write_text(json.dumps({"grid": [10, 8, 6]}))
    data = tmp_path / "data"; data.mkdir()
    _write_cases(data, nx=10, ny=8, nz=6)
    monkeypatch.delenv("PINO_LAZY_STRIDE", raising=False)
    store = maybe_build_npz_store(data, _run_cfg(data, net, eval_only=True))
    assert store.spatial_shape() == (10, 8, 6)


def test_super_resolution_study_is_still_allowed(tmp_path, monkeypatch):
    """A resolution change is the POINT of a super-res eval; do not block it."""
    net = tmp_path / "net"; net.mkdir()
    (net / GRID_STAMP_NAME).write_text(json.dumps({"grid": [5, 4, 6]}))
    data = tmp_path / "data"; data.mkdir()
    _write_cases(data, nx=10, ny=8, nz=6)
    monkeypatch.delenv("PINO_LAZY_STRIDE", raising=False)
    store = maybe_build_npz_store(data, _run_cfg(data, net, eval_only=True, super_res=2))
    assert store.spatial_shape() == (10, 8, 6)


def test_explicit_env_acknowledgement_is_allowed(tmp_path, monkeypatch):
    net = tmp_path / "net"; net.mkdir()
    (net / GRID_STAMP_NAME).write_text(json.dumps({"grid": [5, 4, 6]}))
    data = tmp_path / "data"; data.mkdir()
    _write_cases(data, nx=10, ny=8, nz=6)
    monkeypatch.delenv("PINO_LAZY_STRIDE", raising=False)
    monkeypatch.setenv("PINO_ALLOW_GRID_SHIFT", "1")
    store = maybe_build_npz_store(data, _run_cfg(data, net, eval_only=True))
    assert store.spatial_shape() == (10, 8, 6)


def test_eval_without_a_stamp_is_not_blocked(tmp_path, monkeypatch):
    """Models trained before this contract existed must stay evaluable."""
    net = tmp_path / "net"; net.mkdir()
    data = tmp_path / "data"; data.mkdir()
    _write_cases(data, nx=10, ny=8, nz=6)
    monkeypatch.delenv("PINO_LAZY_STRIDE", raising=False)
    store = maybe_build_npz_store(data, _run_cfg(data, net, eval_only=True))
    assert store.spatial_shape() == (10, 8, 6)


def test_grid_announce_reports_the_moving_ref_window(tmp_path, monkeypatch, capsys):
    """The announce must match spatial_shape(), window included.

    It printed the strided grid only, so a moving-ref run whose store served
    48x40x40 announced "90x40x40" -- which reads as "the window did not apply".
    On 2026-08-19 that nearly inverted the reading of two arms; only the
    trained_grid.json stamp contradicted it.
    """
    from lpno.models.pipeline.npz_lazy_store import NpzDirStore
    from lpno.custom_utils import npz_datasource as nds

    data = tmp_path / "d"; data.mkdir()
    _write_cases(data, nx=10, ny=8, nz=6)
    store = NpzDirStore(
        data,
        requested_fields=["Temperature"],
        path_channels=list(nds.PATH_CHANNEL_NAMES),
        material_channels=list(nds.MATERIAL_CHANNEL_NAMES),
        moving_window=(3, 2),
    )
    assert store.spatial_shape() == (5, 8, 6)
    line = [l for l in capsys.readouterr().out.splitlines() if "[grid]" in l]
    assert line, "no grid announce"
    assert "serving 5x8x6" in line[0], line[0]
    assert "window 3+2" in line[0], line[0]


# --- 3. mixed-grid datasets are exempt from the shape half of the contract ----

def test_mixed_grid_dataset_does_not_trip_the_train_val_check(tmp_path, monkeypatch):
    """A dataset whose members legitimately differ in shape must still train.

    `ti64_geom40` varies the DOMAIN EXTENT by design: 40 cases from 90x41x40 to
    187x125x40 at a fixed 10 um cell. Its train store's first case and its val
    store's first case therefore serve different node counts with no striding
    involved and nothing wrong. The contract compared `store.spatial_shape()`,
    which is the FIRST case's shape, so it read that as a train/val resolution
    mismatch and refused the run (2026-08-19: all three e8+depth arms and a
    train-split eval died with "already served 45x21x40 but now serves
    59x32x40"). RIGNO gained per-shape graphs in 52da00e precisely so mixed-grid
    training works; the contract must not undo that.

    What still has to be enforced on such a dataset is the STRIDE, since
    double-striding is invisible in a per-case shape either way.
    """
    # `_write_cases` reuses mlhs_{i:03d}, so a second call OVERWRITES rather than
    # adds. Build the larger case elsewhere and move it in under a later name,
    # otherwise train and val end up with an identical first case and the test
    # passes without exercising anything.
    train = tmp_path / "train"; train.mkdir()
    _write_cases(train, n=2, nx=10, ny=8, nz=6)          # first case 10x8x6
    big = tmp_path / "big"; big.mkdir()
    _write_cases(big, n=1, nx=14, ny=12, nz=6)
    for f in sorted(big.iterdir()):
        if f.suffix == ".npz":
            f.rename(train / "mlhs_009.npz")             # sorts last: train stays 10x8x6
    val = tmp_path / "val"; val.mkdir()
    _write_cases(val, n=1, nx=14, ny=12, nz=6)           # val's first case is 14x12x6

    net = tmp_path / "net"; net.mkdir()
    cfg = _run_cfg(tmp_path, net)
    maybe_build_npz_store(train, cfg, force=True)      # stamps
    maybe_build_npz_store(val, cfg, force=True)        # must NOT raise


def test_uniform_grid_train_val_mismatch_is_still_refused(tmp_path):
    """The guard the above must not weaken: two UNIFORM stores that disagree."""
    train = tmp_path / "train"; train.mkdir()
    _write_cases(train, n=2, nx=10, ny=8, nz=6)
    val = tmp_path / "val"; val.mkdir()
    _write_cases(val, n=2, nx=14, ny=12, nz=6)
    net = tmp_path / "net"; net.mkdir()
    cfg = _run_cfg(tmp_path, net)
    maybe_build_npz_store(train, cfg, force=True)
    with pytest.raises(ValueError, match="same resolution"):
        maybe_build_npz_store(val, cfg, force=True)
