"""A `time_avg_window` the run cannot use must say so, not read as a parameter.

`time_as_batch` and `time_as_dim` both force `avg_window_size = 1`
(`preprocessing_datasets.py:603-613`), so a config carrying `time_avg_window: 30`
alongside either is declaring something that never takes effect. Several configs
do exactly that, config_flow3D_RIGNO_randompath{,_onecycle} among the shipped ones.

Found 2026-08-26 the expensive way. A run meant to replicate the demo model
`rigno_straight_tb` was going to be re-launched on the eager path specifically to
make its `time_avg_window: 30` take effect -- a 309 GB materialisation to reproduce
a value that cannot fire, since the demo is itself `time_as_batch`. The declaration
and the consumer were different facts and nothing said so.

Announce, do not change: the forcing is correct and long-standing. What was missing
is that the config reads as though 30 means something.
"""
from types import SimpleNamespace

import pytest


def _run_cfg(**kw):
    base = dict(time_as_batch=False, time_as_dim=False, include_time=True,
                time_avg_window=30, moving_ref_frame=False, moving_frame_steps=1,
                use_2d=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _announce(run_cfg, capsys):
    from lpno.models.pipeline.preprocessing_datasets import (
        _warn_if_time_avg_window_is_inert,
    )
    _warn_if_time_avg_window_is_inert(run_cfg)
    return capsys.readouterr().out


def test_time_as_batch_makes_the_window_inert_and_says_so(capsys):
    out = _announce(_run_cfg(time_as_batch=True), capsys)
    assert "time_avg_window" in out and "30" in out
    assert "time_as_batch" in out, "name WHICH setting disables it"
    assert "no effect" in out.lower() or "ignored" in out.lower()


def test_time_as_dim_likewise(capsys):
    assert "time_as_dim" in _announce(_run_cfg(time_as_dim=True), capsys)


def test_a_window_that_does_apply_is_silent(capsys):
    assert _announce(_run_cfg(), capsys) == ""


def test_window_of_one_is_silent(capsys):
    """Nothing is being lost, so nothing to say."""
    assert _announce(_run_cfg(time_as_batch=True, time_avg_window=1), capsys) == ""
