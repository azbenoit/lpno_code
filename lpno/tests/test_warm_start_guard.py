"""A warm start that cannot load its donor must RAISE, not train from scratch.

physicsnemo's `sym/trainer.py` (upstream, read-only here) wraps the
`initialization_network_dir` load in a bare `except`: on a state-dict mismatch it
logs one red "Fail loading model:" line, sets step back to 0, and **carries on
with randomly initialised weights**. The run then trains to completion and
produces a curve that looks entirely normal.

Measured 2026-08-24 on arm 65831: a donor built when
`config_flow3D_RIGNO_randompath_onecycle` defaulted to `num_graph_variants: 1`,
loaded into a run that resolved the config after the default moved to 8. Step-0
beam-box MAE was 1592 K where a working warm start starts at 42-55 K. 6000 steps
of GPU spent answering nothing.

This guard runs BEFORE training, so the failure costs seconds rather than hours.
"""
from __future__ import annotations

import pathlib

import pytest

from lpno.models.ns_energy_pipeline import assert_init_checkpoint_loadable


class _Model:
    """Stands in for a physicsnemo model: a checkpoint name plus a load()."""

    def __init__(self, checkpoint_filename="u_network.0.pth", fail=False):
        self.checkpoint_filename = checkpoint_filename
        self.fail = fail
        self.loaded_from = None

    def load(self, directory, map_location=None):
        if self.fail:
            raise RuntimeError("size mismatch for proc_steps.8.edge_mlp.layers.0.weight")
        self.loaded_from = directory


def test_no_warm_start_requested_is_a_no_op():
    """An ordinary from-scratch run must not be affected by this guard."""
    model = _Model()
    for empty in (None, "", "  "):
        assert_init_checkpoint_loadable(model, empty)
    assert model.loaded_from is None


def test_loads_when_the_donor_matches(tmp_path):
    (tmp_path / "u_network.0.pth").write_bytes(b"weights")
    model = _Model()
    assert_init_checkpoint_loadable(model, tmp_path)
    assert model.loaded_from == str(tmp_path)


def test_raises_when_the_donor_checkpoint_is_absent(tmp_path):
    """The exact case that produced 1592 K: physicsnemo warns and continues."""
    model = _Model()
    with pytest.raises(FileNotFoundError, match="u_network.0.pth"):
        assert_init_checkpoint_loadable(model, tmp_path)


def test_raises_on_a_shape_mismatch(tmp_path):
    """The load fails, so training would silently start from random weights."""
    (tmp_path / "u_network.0.pth").write_bytes(b"weights")
    model = _Model(fail=True)
    with pytest.raises(RuntimeError) as exc:
        assert_init_checkpoint_loadable(model, tmp_path)
    msg = str(exc.value)
    # The message must name the CAUSE, not just the symptom: an architecture key
    # that drifted is what puts a donor and a run out of shape.
    assert "architecture" in msg.lower()
    assert "size mismatch" in msg  # the underlying error is preserved, not swallowed


def test_raises_when_the_directory_does_not_exist(tmp_path):
    model = _Model()
    with pytest.raises(FileNotFoundError):
        assert_init_checkpoint_loadable(model, tmp_path / "nope")


def test_missing_checkpoint_filename_is_an_error():
    """Guarding with no checkpoint name would silently check nothing."""
    model = _Model(checkpoint_filename=None)
    with pytest.raises(RuntimeError, match="checkpoint filename"):
        assert_init_checkpoint_loadable(model, pathlib.Path("."))
