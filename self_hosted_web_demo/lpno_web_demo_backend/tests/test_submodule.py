"""ensure_submodule_on_path tolerates a missing ext/ checkout when the package
is importable from site-packages (stub mode, presets, GT readers), and still
refuses when it is importable from nowhere."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app import submodule


def test_missing_src_is_fine_when_package_is_installed(tmp_path: Path, monkeypatch) -> None:
    pkg = tmp_path / "site" / "lpno"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    monkeypatch.delitem(sys.modules, "lpno", raising=False)
    missing = tmp_path / "no-such-ext" / "src"
    monkeypatch.setattr(submodule, "submodule_src", lambda: missing)

    assert submodule.ensure_submodule_on_path() == missing
    assert str(missing) not in sys.path


def test_missing_src_raises_when_package_is_nowhere(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "no-such-ext" / "src"
    monkeypatch.setattr(submodule, "submodule_src", lambda: missing)
    monkeypatch.setattr(submodule.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(FileNotFoundError):
        submodule.ensure_submodule_on_path()
