"""Regression test: the default optimizer dependency must be declared.

Every shipped config sets ``optimizer: lion`` (``lion_pytorch.Lion`` registered in
``models/__init__.py``), so a clean install must pull in ``lion-pytorch`` or the
documented Quick Start dies with ModuleNotFoundError before the first step.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_lion_pytorch_declared_in_pyproject():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "lion-pytorch" in pyproject, (
        "configs default to optimizer=lion but pyproject.toml does not declare lion-pytorch"
    )


def test_lion_pytorch_declared_in_environment_yml():
    environment = (ROOT / "environment.yml").read_text(encoding="utf-8")
    assert "lion-pytorch" in environment, (
        "configs default to optimizer=lion but environment.yml does not install lion-pytorch"
    )
