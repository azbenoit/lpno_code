"""Regression tests for the safe bugs found by the public-cut audit (2026-09-14)."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf


def test_custom_lazy_npz_reaches_the_loader(monkeypatch):
    """The docstring promised `custom.lazy_npz`; `getattr(run_cfg, ...)` never saw a Hydra
    key because RunConfig declares no such field. Now routed through `custom_flag`."""
    from lpno.models.pipeline.npz_lazy_store import lazy_npz_enabled

    monkeypatch.delenv("PINO_LAZY_NPZ", raising=False)
    cfg = SimpleNamespace(cfg=OmegaConf.create({"custom": {"lazy_npz": True}}))
    assert lazy_npz_enabled(cfg) is True
    off = SimpleNamespace(cfg=OmegaConf.create({"custom": {}}))
    assert lazy_npz_enabled(off) is False
    monkeypatch.setenv("PINO_LAZY_NPZ", "1")
    assert lazy_npz_enabled(off) is True, "the env var still wins"


def test_eager_loader_reads_a_pack_index_staging(tmp_path):
    """`load_consolidated_npz` did `np.load(case)` on `_case_files()` output, which yields
    PackSlicePath objects on a pack-index staging; their `__fspath__` raises by design.
    Two sibling call sites already wrapped with `_open_case_safely`; this one did not."""
    import io

    from lpno.custom_utils import npz_datasource as nds
    from test_pack_slice import _pack_with_index   # shared fixture: one pack + index, no real .npz

    nx, ny, nz, nt = 6, 5, 4, 3
    payload = {
        "x": np.linspace(-1e-3, 1e-3, nx).astype("f4"), "y": np.linspace(-5e-4, 5e-4, ny).astype("f4"),
        "z": np.linspace(-3e-4, 1e-4, nz).astype("f4"), "time_s": np.linspace(0, 2e-3, nt),
        "power_w": np.float64(150.0), "scan_speed_m_per_s": np.float64(0.8), "case_name": np.array("c1"),
        "Temperature": np.random.default_rng(0).random((nt, nx, ny, nz)).astype("f4"),
        "Fraction Of Fluid": np.ones((nt, nx, ny, nz), "f4"),
    }
    real = tmp_path / "real"; real.mkdir()
    np.savez(real / "c1.npz", **payload)
    staged = _pack_with_index(tmp_path / "staged", {"c1": (real / "c1.npz").read_bytes()})
    assert not list(staged.glob("*.npz")), "index-only staging: nothing but the pack and its index"
    kw = dict(requested_fields=["Temperature"], time_axis=True, avg_window_size=1,
              moving_frame_steps=None, moving_ref_frame=False)
    got = nds.load_consolidated_npz(staged, **kw)
    want = nds.load_consolidated_npz(real, **kw)
    assert np.array_equal(got["Temperature"], want["Temperature"])


def test_spool_workers_launch_with_this_interpreter(monkeypatch, tmp_path):
    """`_start_all` spawned per-GPU workers with a bare "python", contradicting
    `build_command` in the same file, which pins sys.executable and says why."""
    from lpno import gpu_spool as gs

    seen = []

    class _P:
        def __init__(self, argv, *a, **k):
            seen.append(list(argv))

        def wait(self):
            return 0

    monkeypatch.setattr(gs.subprocess, "Popen", _P)
    assert gs._start_all(tmp_path, [0, 1], False, 5.0) == 0
    assert len(seen) == 2 and all(argv[0] == sys.executable for argv in seen), seen
