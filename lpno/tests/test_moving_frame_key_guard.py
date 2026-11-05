"""Trajectory channels in the moving reference frame: WARN, and keep them.

Regression origin (2026-08-05): the first varspeed GAOT submission listed `beam_x` and
friends in `custom.FNO_input_keys` while inheriting `custom.moving_ref_frame=true` from
the straight-path config it was based on. The run staged its dataset, built a 600k-node
GAOT graph, and only then died with a bare `KeyError: 'beam_x'` far from the setting
that caused it. This file used to pin a hard refusal of that combination.

CONTRACT CHANGED 2026-08-19, deliberately, on a measurement:

* Decision -- the steady and moving-ref arms must inherit the SAME input keys as their
  lab-frame partners even where a channel goes redundant, so the three frame treatments
  stay comparable and warm-startable from one another. A key set that differs by frame
  makes a cross-frame comparison impossible and silently reshapes the lifting layer.
* Measurement -- the docstring the refusal rested on ("the datasource never builds
  them") is stale for the consolidated-NPZ path, which every current dataset uses.
  Verified by running it: under `moving_ref_frame=True` with `avg_window_size=30` and
  `moving_frame_steps=30`, `load_consolidated_npz` returns beam_x, beam_y, beam_dist,
  path_dose and path_dose_recent at the SAME shape as Temperature. They are built at
  `npz_datasource.py:1230` and then cropped and shifted with every other key by
  `_apply_moving_frame` at :1249, so they stay consistent with the field data.
  `tests/test_path_channels.py::test_moving_frame_keeps_every_trajectory_channel`
  pins that end to end.

What survives is the warning -- and its TEXT was corrected 2026-08-21, because it named
only two paths where there are three. It told the reader to ask "does this run read
VTK?", which returns NO for a staged/derived NPZ member, the path that actually fails
(`lazy_runtime.py:68-70`: trajectory channels "need a per-frame beam path a staged
derived member does not carry, and must fail by name"). The real condition is whether
the dataset's members carry a per-frame beam path, not the storage format.
"""
import pytest

from lpno.models.pipeline.config_loading import (
    _reject_lab_frame_keys_in_moving_frame,
)


def test_trajectory_keys_in_moving_frame_warn_but_are_kept(capsys):
    """The combination that used to be refused is now allowed, and announced."""
    _reject_lab_frame_keys_in_moving_frame(
        ['x', 'y', 'z', 'beam_x', 'beam_y', 'path_dose'], moving_ref_frame=True)
    out = capsys.readouterr().out
    assert 'WARNING' in out
    for channel in ('beam_x', 'beam_y', 'path_dose'):
        assert channel in out, channel


def test_the_warning_names_the_channels_and_the_case_that_still_breaks(capsys):
    """A warning nobody can act on is noise: name the channel and the real risk."""
    _reject_lab_frame_keys_in_moving_frame(['beam_x'], moving_ref_frame=True)
    out = capsys.readouterr().out
    assert 'beam_x' in out
    assert 'VTK' in out, 'must name the datasource where these are genuinely absent'


def test_lab_frame_keys_are_silent_in_the_lab_frame(capsys):
    """The random-path recipe: moving_ref_frame=false, nothing to say."""
    _reject_lab_frame_keys_in_moving_frame(
        ['x', 'beam_x', 'beam_dist', 'path_dose', 'path_dose_recent'],
        moving_ref_frame=False)
    assert capsys.readouterr().out == ''


def test_moving_frame_without_trajectory_keys_is_silent(capsys):
    """No trajectory channels requested: nothing to warn about."""
    _reject_lab_frame_keys_in_moving_frame(
        ['x', 'y', 'z', 'P', 'V_scan', 'H_norm'], moving_ref_frame=True)
    assert capsys.readouterr().out == ''


def test_the_tau_bank_channels_are_covered_too(capsys):
    """The bank is part of PATH_CHANNEL_NAMES, so it must reach the same path."""
    _reject_lab_frame_keys_in_moving_frame(
        ['path_dose_tau30us'], moving_ref_frame=True)
    assert 'path_dose_tau30us' in capsys.readouterr().out


def test_warning_names_the_staged_lazy_path_that_actually_fails(capsys):
    """Reported 2026-08-21: the warning named a path that was not in use.

    `fno_spkh123_steady_32k_added14ch` (AiiDA 63295/63296/63298) died on ds55011
    with `KeyError: 'beam_dist'` after printing this warning. The warning's stated
    test is "does this run read VTK?" -- and the run reads a FLOW-3D NPZ dataset,
    so anyone applying that criterion answers no and proceeds into the failure.

    There are THREE paths, not two:
      load_consolidated_npz              BUILDS the channels
      legacy VTK / OpenFOAM              does not build them
      NpzDirStore / staged derived member does not, BY DESIGN
        (`lazy_runtime.py:68-70`: they "need a per-frame beam path a staged
         derived member does not carry, and must fail by name")

    The third is the one that ran. The KeyError is correct behaviour; only the
    diagnostic was wrong.
    """
    _reject_lab_frame_keys_in_moving_frame(['beam_dist'], moving_ref_frame=True)
    out = capsys.readouterr().out
    assert 'staged' in out.lower() or 'NpzDirStore' in out, (
        'must name the lazy/staged member path -- the one that actually fails')


def test_warning_states_the_real_condition_not_the_storage_format(capsys):
    """The trigger is whether members carry a per-frame beam path, not the format.

    ds55011 is `var_1_time_noref`, whose manifests have none. The sibling arm
    `fno_spkh123_mref_32k_orig6ch` died on the same root cause with a different
    message entirely -- `ValueError: [NpzDirStore] moving_window needs per-frame
    beam centres; the case manifest has no laser path` -- which the old text did
    not mention at all. A reader who knows only "NPZ vs VTK" cannot predict
    either failure.
    """
    _reject_lab_frame_keys_in_moving_frame(['beam_x'], moving_ref_frame=True)
    out = capsys.readouterr().out.lower()
    assert 'beam path' in out or 'per-frame' in out, (
        'must name the per-frame beam path as the condition')
    assert 'laser path' in out or 'no laser path' in out or 'beam centres' in out, (
        'must mention the sibling moving_window failure so both are predictable')
