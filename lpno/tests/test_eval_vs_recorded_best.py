"""An eval far worse than the model's own recorded best is probably a bug.

Model 68163 recorded `mae_T 0.012049` in its `best_checkpoint.json` and scored
558 K whole-domain -- 15x worse -- with exit 0, a populated metrics.json and
nothing flagged anywhere. It read as a completed eval of a bad model.

This compares the eval's headline metric against what the model demonstrably
reached and says so when the gap is large. Unlike `use_best_checkpoint`, which
fixes ONE mechanism (scoring last weights instead of best), this catches the class:
era mismatch, wrong geometry, wrong dataset, a truncated checkpoint -- anything
that lands an eval far from what the training achieved.

WARN, not fail, and with a loose factor. The two numbers are different quantities:
the recorded one is a VALIDATION metric on val frames, the eval is a test-split
full-frame score. They differ legitimately and not always in the same direction --
68163's rescued weights recorded ~0.012 in validation and scored BETTER on the test
split. A tight threshold would fire on healthy runs; a failure would block a
legitimate result over a comparison of two unlike things.
"""
import json

import pytest

from lpno.models.pipeline.postprocessing import _write_metric_file


def _recorded(tmp_path, metric=0.012049, key="mae_T", step=26000):
    # The WRITER's schema (best_checkpoint.py): "metric" is the NAME, "value" the number.
    # These fixtures were first written to the reader's misreading of it, so the
    # positive test passed against a guard that never fired on a real sidecar.
    (tmp_path / "best_checkpoint.json").write_text(json.dumps(
        {"step": step, "metric": key, "value": metric, "weights": "u_network.best.pth"}))
    return tmp_path


def test_a_far_worse_eval_warns_and_calls_it_a_probable_bug(tmp_path, capsys):
    net = tmp_path / "net"; net.mkdir(); _recorded(net)
    out = tmp_path / "eval"; out.mkdir()
    # 68163's real numbers: recorded 0.012049, scored ~0.186 normalized (15x)
    _write_metric_file(out / "model_errors.txt", {"T": {"abs_err_mean": 0.186}},
                       network_dir=net)
    msg = capsys.readouterr().out
    assert "0.012049" in msg and "0.186" in msg, "name BOTH numbers -- the gap is the tell"
    assert "bug" in msg.lower(), "must say a bug is likely, not merely note a difference"
    assert "26000" in msg, "name the step the best came from"


def test_a_comparable_eval_is_silent(tmp_path, capsys):
    net = tmp_path / "net"; net.mkdir(); _recorded(net)
    out = tmp_path / "eval"; out.mkdir()
    _write_metric_file(out / "model_errors.txt", {"T": {"abs_err_mean": 0.0135}},
                       network_dir=net)
    assert capsys.readouterr().out == "", "normal val-vs-test spread must not cry wolf"


def test_a_better_eval_is_silent(tmp_path, capsys):
    """68163's rescued weights scored BETTER on test than in validation."""
    net = tmp_path / "net"; net.mkdir(); _recorded(net)
    out = tmp_path / "eval"; out.mkdir()
    _write_metric_file(out / "model_errors.txt", {"T": {"abs_err_mean": 0.0084}},
                       network_dir=net)
    assert capsys.readouterr().out == ""


def test_no_recorded_best_is_silent(tmp_path, capsys):
    """Pre-keeper models archive none; their evals must stay quiet."""
    net = tmp_path / "net"; net.mkdir()
    out = tmp_path / "eval"; out.mkdir()
    _write_metric_file(out / "model_errors.txt", {"T": {"abs_err_mean": 9.9}},
                       network_dir=net)
    assert capsys.readouterr().out == ""


def test_absent_network_dir_is_silent(tmp_path, capsys):
    out = tmp_path / "eval"; out.mkdir()
    _write_metric_file(out / "model_errors.txt", {"T": {"abs_err_mean": 9.9}})
    assert capsys.readouterr().out == ""


def test_a_broken_sidecar_does_not_break_the_eval(tmp_path, capsys):
    net = tmp_path / "net"; net.mkdir()
    (net / "best_checkpoint.json").write_text("{not json")
    out = tmp_path / "eval"; out.mkdir()
    _write_metric_file(out / "model_errors.txt", {"T": {"abs_err_mean": 9.9}},
                       network_dir=net)  # must not raise
    assert (out / "model_errors.txt").is_file(), "metrics must still be written"


def test_the_wiring_uses_the_field_run_cfg_actually_has(tmp_path, capsys):
    """`RunConfig` carries `network_base_dir`, not `network_dir`.

    A `getattr(run_cfg, "network_dir", None)` would resolve to None on every real
    eval and the warning would never fire -- committed, green, and inert. Pin the
    field name so the wiring cannot quietly stop working.
    """
    from types import SimpleNamespace

    from lpno.models.pipeline import postprocessing as pp

    net = tmp_path / "net"; net.mkdir(); _recorded(net)
    seen = {}
    real = pp._warn_if_far_from_recorded_best

    def spy(errors, network_dir):
        seen["dir"] = network_dir
        return real(errors, network_dir)

    pp._warn_if_far_from_recorded_best = spy
    try:
        out = tmp_path / "eval"; out.mkdir()
        run_cfg = SimpleNamespace(network_base_dir=str(net))
        pp._write_metric_file(out / "model_errors.txt", {"T": {"abs_err_mean": 0.186}},
                              network_dir=getattr(run_cfg, "network_base_dir", None))
    finally:
        pp._warn_if_far_from_recorded_best = real
    assert seen.get("dir") == str(net), "the wired attribute must resolve to a real dir"
    assert "PROBABLY A BUG" in capsys.readouterr().out
