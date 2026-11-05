"""An OPTIONAL default path must not hard-require the data root to be resolvable.

`param_dir` is only read for piecewise-tabulated c(T) materials
(`material_params.c == "c"`); every random-path and straight-path config here uses
polynomial c(T) and never touches it. `config_loading` says so in a comment and passes
`require_exists=False` for exactly that reason -- but `_resolve_default_data_path`
called `_resolve_default_data_root()` BEFORE consulting `require_exists`, so the
optional path still hard-required the data ROOT.

Invisible on the primary workstation, where the repo's own `data/` sits adjacent and
resolution always succeeds. Fatal on a machine that receives its datasets through AiiDA
staging instead of holding the 147 GB `data/` tree -- i.e. on any compute box. Reported
2026-08-19 from a compute box, where a `config_flow3D_GAOT_pvlhs` run with
`custom.data_dir` and `custom.val_data_dir` both passed explicitly still died at config
load, before step 0, on a 16 KB material-parameter path it would never have read.

The contract: when the root cannot be resolved and the path is optional, DEFER -- and let
the read site raise if anything ever actually reads it.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lpno.models.pipeline import config_loading as cl
from lpno.models.pipeline.preprocessing_datasets import (
    _define_material_properties,
)


@pytest.fixture()
def no_data_root(monkeypatch):
    """A machine with neither the env var nor an adjacent repo `data/`."""
    monkeypatch.delenv(cl._DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(cl, "_discover_repo_data_root", lambda: None)


def test_optional_path_defers_when_root_is_unresolvable(no_data_root):
    """The bug: this raised, killing runs that never read the path."""
    resolved = cl._resolve_default_data_path(
        Path("abaqus_data/Ti64_params"), label="parameter data", require_exists=False,
    )
    assert resolved == cl._UNRESOLVED_DATA_PATH


def test_required_path_still_raises_when_root_is_unresolvable(no_data_root):
    """Deferring must not weaken the check for paths that are actually needed."""
    with pytest.raises(ValueError, match=cl._DATA_ROOT_ENV_VAR):
        cl._resolve_default_data_path(
            Path("training_data"), label="training data", require_exists=True,
        )


def test_optional_path_still_resolves_when_the_root_exists(monkeypatch, tmp_path):
    """Unchanged behaviour wherever the data root is present."""
    monkeypatch.setenv(cl._DATA_ROOT_ENV_VAR, str(tmp_path))
    resolved = cl._resolve_default_data_path(
        Path("abaqus_data/Ti64_params"), label="parameter data", require_exists=False,
    )
    assert resolved == str(tmp_path / "abaqus_data/Ti64_params")


# --- the read site is where an unresolved path must finally be refused ---------

def _run_cfg(param_dir, c_spec):
    return SimpleNamespace(
        param_dir=param_dir,
        material_params={"c": c_spec},
        laser_params={},
        pde_weight=0.1,
    )


def test_polynomial_material_trains_with_an_unresolved_param_dir():
    """The whole point: these configs never read it, so they must not be blocked."""
    var = {"T": None, "P": [[120.0]], "V_scan": [[0.4]]}
    run_cfg = _run_cfg(cl._UNRESOLVED_DATA_PATH, "poly")
    _define_material_properties(var, run_cfg)
    assert "c" not in var                                   # c(T) branch skipped
    assert run_cfg.laser_params["P"] == pytest.approx(120.0)  # and it ran to the end


def test_tabulated_material_raises_actionably_on_an_unresolved_param_dir():
    """Deferred, not dropped: a config that DOES read it gets a usable error."""
    with pytest.raises(FileNotFoundError) as excinfo:
        _define_material_properties(
            {"T": None, "P": [[120.0]], "V_scan": [[0.4]]},
            _run_cfg(cl._UNRESOLVED_DATA_PATH, "c"),
        )
    message = str(excinfo.value)
    assert "custom.param_dir" in message
    assert cl._DATA_ROOT_ENV_VAR in message
