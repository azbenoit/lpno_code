"""Omitting `flow3d_scripts_root` must not land on the stalest copy in the tree.

Reported 2026-08-06: leave it out of the execution profile and the adapter fell
through to the vendored `external/flow3D_scripts` submodule, pinned behind
master, which *ignores* `recoil_model` and `void_vapor_pressure_pa` instead of
rejecting them. The first sign was exit 410 on every child, minutes after a
submit that looked clean. It cost one batch in full.

Measured on the submitting machine: the sibling checkout is at `a755a0b` with **11**
occurrences of `recoil_model`; the vendored submodule is at `d4896fe` with
**0**. 1175 of 1178 `Flow3DRunCalculation`s pin the sibling; the 3 that do not
are the 3 console submits that left it unset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiida_pino.scripts.flow3d_run_adapter import RUNNER_RELATIVE, resolve_runner_root
from aiida_pino.utils.flow3d_config import default_runner_root


def _fake_checkout(root: Path) -> Path:
    (root / RUNNER_RELATIVE.parent).mkdir(parents=True, exist_ok=True)
    (root / RUNNER_RELATIVE).write_text("# runner\n")
    return root


def test_the_sibling_checkout_wins_over_the_vendored_submodule(tmp_path, monkeypatch):
    repo = tmp_path / "aiida-pino"
    sibling = _fake_checkout(tmp_path / "flow3D_scripts")
    submodule = _fake_checkout(repo / "external" / "flow3D_scripts")
    monkeypatch.setattr(
        "aiida_pino.scripts.flow3d_run_adapter._default_repo_root", lambda: repo
    )
    monkeypatch.delenv("F3D_SCRIPTS_ROOT", raising=False)

    resolved = resolve_runner_root({})
    assert resolved == sibling.resolve(), f"expected the sibling checkout, got {resolved}"
    assert resolved != submodule.resolve()


def test_the_submodule_is_still_the_last_resort(tmp_path, monkeypatch):
    """Only reachable when there is no sibling checkout at all."""
    repo = tmp_path / "aiida-pino"
    submodule = _fake_checkout(repo / "external" / "flow3D_scripts")
    monkeypatch.setattr(
        "aiida_pino.scripts.flow3d_run_adapter._default_repo_root", lambda: repo
    )
    monkeypatch.delenv("F3D_SCRIPTS_ROOT", raising=False)
    assert resolve_runner_root({}) == submodule.resolve()


def test_an_explicit_root_without_a_runner_is_still_a_hard_error(tmp_path, monkeypatch):
    """A guess may fall through; something the caller named may not."""
    monkeypatch.delenv("F3D_SCRIPTS_ROOT", raising=False)
    with pytest.raises(ValueError, match="does not exist"):
        resolve_runner_root({"flow3d_scripts_root": str(tmp_path / "nope")})


def test_an_explicit_root_beats_the_sibling(tmp_path, monkeypatch):
    repo = tmp_path / "aiida-pino"
    _fake_checkout(tmp_path / "flow3D_scripts")
    chosen = _fake_checkout(tmp_path / "chosen")
    monkeypatch.setattr(
        "aiida_pino.scripts.flow3d_run_adapter._default_repo_root", lambda: repo
    )
    monkeypatch.delenv("F3D_SCRIPTS_ROOT", raising=False)
    assert resolve_runner_root({"flow3d_scripts_root": str(chosen)}) == chosen.resolve()


def test_the_console_default_names_a_real_runner():
    """`default_runner_root` is what makes submit-time validation always run."""
    root = default_runner_root()
    if not root:
        pytest.skip("no sibling flow3D_scripts checkout on this machine")
    assert (Path(root) / RUNNER_RELATIVE).is_file()
    # It must be a *current* runner, not a copy that predates the calibration
    # keys — that is the whole point of preferring it.
    assert "recoil_model" in (Path(root) / RUNNER_RELATIVE).read_text()
