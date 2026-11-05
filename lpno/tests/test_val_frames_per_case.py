"""custom.val_frames_per_case: temporal subsample of the in-training validator.

The pvlhs wave-1 arms (val_max_cases=0, 12 cases x ~500 frames) spent 10.8 h of an
11.6 h run validating -- 17 passes x 38 min -- while training itself took 47 min.
The case-count cap alone cannot bound that; this is its frames-per-case twin.
"""
import types

import numpy as np

from lpno.models.pipeline.preprocessing_datasets import (
    _build_standard_entries,
    _effective_val_frames_per_case,
)


class _Store:
    def __init__(self, frames_by_case):
        self._frames = frames_by_case

    def time_steps(self, case_idx):
        return self._frames[case_idx]


def _cfg(**kw):
    base = dict(time_as_batch=True, time_as_dim=False, val_frames_per_case=10, eval_only=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_cap_keeps_n_evenly_spaced_frames_with_endpoints():
    store = _Store({0: 500})
    entries = _build_standard_entries(store, [0], _cfg(), frames_per_case=10)
    times = [t for _c, t in entries]
    assert len(times) == 10
    assert times[0] == 0 and times[-1] == 499  # endpoints always scored
    gaps = np.diff(times)
    assert gaps.max() - gaps.min() <= 1  # evenly spaced

def test_zero_or_short_case_keeps_every_frame():
    store = _Store({0: 7})
    assert len(_build_standard_entries(store, [0], _cfg(), frames_per_case=0)) == 7
    assert len(_build_standard_entries(store, [0], _cfg(), frames_per_case=10)) == 7
    assert len(_build_standard_entries(store, [0], _cfg(), frames_per_case=7)) == 7

def test_train_entries_never_capped():
    """Only the val call site passes frames_per_case; the default must be full."""
    store = _Store({0: 500})
    assert len(_build_standard_entries(store, [0], _cfg())) == 500

def test_eval_only_scores_the_whole_split():
    assert _effective_val_frames_per_case(_cfg(eval_only=True)) == 0
    assert _effective_val_frames_per_case(_cfg(eval_only=True, val_frames_per_case=10)) == 0

def test_training_run_keeps_the_cap():
    assert _effective_val_frames_per_case(_cfg()) == 10
    assert _effective_val_frames_per_case(_cfg(val_frames_per_case=0)) == 0

def test_default_is_bounded_in_run_config():
    """The default must be the safe one: a config that says nothing gets 10
    frames per case, and switching it off is the explicit act."""
    from lpno.models.pipeline.dataclasses import RunConfig
    import dataclasses
    field = {f.name: f for f in dataclasses.fields(RunConfig)}["val_frames_per_case"]
    assert field.default == 10

def test_multi_case_mixed_lengths():
    store = _Store({3: 400, 7: 4})
    entries = _build_standard_entries(store, [3, 7], _cfg(), frames_per_case=10)
    by_case = {}
    for c, t in entries:
        by_case.setdefault(c, []).append(t)
    assert len(by_case[3]) == 10
    assert by_case[7] == [0, 1, 2, 3]  # shorter than the cap: untouched
