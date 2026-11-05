"""Regression tests: direct VTK directory loading is deprecated (BUG-01/BUG-02).

Consolidated NPZ (or an arraydir runtime / aiida dataset) is the supported input
path. The legacy VTK parser has known silent-corruption modes (subdir P/V parse,
decimal time suffixes), so `load_and_preprocess_data` must refuse a plain VTK
directory unless the caller explicitly opts in with PINO_ALLOW_VTK=1.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from lpno.models.pipeline import preprocessing_datasets as pp


class _SentinelDataSource:
    """Stub that proves the gate opened without parsing anything."""

    def parse_directory(self, *args, **kwargs):
        raise RuntimeError("legacy VTK datasource invoked")


def _vtk_run_cfg():
    return SimpleNamespace(
        data_directory_path="",
        include_time=False,
        time_avg_window=1,
        super_res_factor=1,
        vtk_fields=["Temperature"],
        grid_params={},
        software="flow3D",
    )


def _vtk_dir(tmp_path: Path) -> Path:
    case = tmp_path / "legacy_vtk_case"
    case.mkdir()
    (case / "timestep_000.vtk").write_text("# vtk stub\n", encoding="utf-8")
    return case


def test_vtk_directory_is_refused_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("PINO_ALLOW_VTK", raising=False)
    monkeypatch.setattr(pp, "VTKOpenFOAMDataSource", _SentinelDataSource)
    with pytest.raises(ValueError) as excinfo:
        pp.load_and_preprocess_data(_vtk_run_cfg(), input_dir=str(_vtk_dir(tmp_path)))
    message = str(excinfo.value)
    assert "deprecated" in message
    assert "NPZ" in message
    assert "PINO_ALLOW_VTK" in message


def test_vtk_directory_loads_with_explicit_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("PINO_ALLOW_VTK", "1")
    monkeypatch.setattr(pp, "VTKOpenFOAMDataSource", _SentinelDataSource)
    # Reaching the sentinel proves the gate opened and the legacy path still works.
    with pytest.raises(RuntimeError, match="legacy VTK datasource invoked"):
        pp.load_and_preprocess_data(_vtk_run_cfg(), input_dir=str(_vtk_dir(tmp_path)))
