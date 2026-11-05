"""A trained model must carry the hydra config it was trained with.

Hydra writes `config.yaml` into its own run dir (`outputs/<name>_<stamp>/.hydra/`),
never into `network_dir`, so nothing ever retrieved it: `PinoModelData` held only
`*.pth`. The consequences were live, not theoretical:

- eval fell back to package defaults with
  "Eval defaults not loaded from model .hydra config; no config.yaml found",
  computed nothing, and exited 502 with `metrics.json == {}` (two runs,
  2026-07-22) -- the model had loaded fine, so the failure looked like bad data;
- serving the same checkpoint on the website needed the config reconstructed by
  hand from a sibling model dir, because the training work dir had been deleted.

The adapter now copies the run's `.hydra` into `network_dir` before the summary
is written, so the config travels with the checkpoints into the model node.
`train_summary.contains_hydra` reports what actually landed.
"""
from __future__ import annotations

from pathlib import Path

from aiida_pino.scripts.train_adapter import archive_hydra_config


def _make_run(root: Path, name: str, *, config: str = "custom:\n  name: r\n") -> Path:
    run = root / "outputs" / name / ".hydra"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(config, encoding="utf-8")
    (run / "overrides.yaml").write_text("- custom.name=r\n", encoding="utf-8")
    return run


def test_config_is_copied_into_the_network_dir(tmp_path):
    _make_run(tmp_path, "rigno_a_0722_1200")
    out = tmp_path / "training_output"
    out.mkdir()
    copied = archive_hydra_config(out, cwd=tmp_path)
    assert copied is True
    assert (out / ".hydra" / "config.yaml").is_file()
    assert "custom" in (out / ".hydra" / "config.yaml").read_text()


def test_overrides_travel_too(tmp_path):
    """overrides.yaml records what was changed from the packaged defaults."""
    _make_run(tmp_path, "rigno_a_0722_1200")
    out = tmp_path / "training_output"
    out.mkdir()
    archive_hydra_config(out, cwd=tmp_path)
    assert (out / ".hydra" / "overrides.yaml").is_file()


def test_newest_run_dir_wins(tmp_path):
    """A warm start leaves an older run dir behind; take the run just finished."""
    old = _make_run(tmp_path, "rigno_a_0722_1200", config="custom:\n  name: OLD\n")
    new = _make_run(tmp_path, "rigno_a_0722_1800", config="custom:\n  name: NEW\n")
    import os
    import time

    os.utime(old / "config.yaml", (time.time() - 9000, time.time() - 9000))
    os.utime(new / "config.yaml", (time.time(), time.time()))
    out = tmp_path / "training_output"
    out.mkdir()
    archive_hydra_config(out, cwd=tmp_path)
    assert "NEW" in (out / ".hydra" / "config.yaml").read_text()


def test_missing_config_is_reported_not_raised(tmp_path):
    """A trainer that wrote no hydra dir must not fail the whole run."""
    out = tmp_path / "training_output"
    out.mkdir()
    assert archive_hydra_config(out, cwd=tmp_path) is False
    assert not (out / ".hydra").exists()


def test_existing_network_dir_config_is_kept(tmp_path):
    """If the trainer already put one there, do not clobber it."""
    out = tmp_path / "training_output"
    (out / ".hydra").mkdir(parents=True)
    (out / ".hydra" / "config.yaml").write_text("custom:\n  name: ALREADY\n", encoding="utf-8")
    _make_run(tmp_path, "rigno_a_0722_1200", config="custom:\n  name: OTHER\n")
    assert archive_hydra_config(out, cwd=tmp_path) is True
    assert "ALREADY" in (out / ".hydra" / "config.yaml").read_text()
