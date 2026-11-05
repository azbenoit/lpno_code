"""Lazy-store autoregressive rollout: per-sample window assembly.

The eager NPZ path stacks every case into one array, so it needs a uniform
timestep count and holds the whole split in RAM. The ti64 random-path sets
(AiiDA collection 2579: 300 cases, nt 14-400, ~40k train frames) are neither,
so AR rollout on them has to stream frame-by-frame from the lazy store.

These tests pin the lazy assembly to the eager builders' exact key/value
contract, so `UnrolledGridConstraint` cannot tell the two paths apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from lpno.models.pipeline.preprocessing_datasets import (
    _ar_window_sample,
    _build_ar_window_entries,
    _build_autoregressive_pairs,
    _build_autoregressive_window_pairs,
)

CONTEXT = ["P", "V_scan", "x"]
STATE = ["T", "alpha"]
EXTRA = ["fl"]
SUFFIX = "_prev"


def _temporal_var(n_time: int = 8) -> dict[str, np.ndarray]:
    """One case, distinct value per (key, frame) so misplaced frames show up."""
    shape = (1, n_time, 1, 2, 2, 2)
    out = {}
    for i, key in enumerate(CONTEXT + STATE + EXTRA):
        base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        out[key] = base + 100.0 * i
    return out


def _frames(var: dict[str, np.ndarray], t0: int, count: int) -> list[dict[str, np.ndarray]]:
    """Per-frame dicts as the lazy store + preprocess would hand them over."""
    return [{key: arr[0, t0 + j] for key, arr in var.items()} for j in range(count)]


@pytest.mark.parametrize("k", [2, 3, 4])
def test_window_sample_matches_eager_window_pairs(k: int) -> None:
    var = _temporal_var()
    ds_input, ds_output = _build_autoregressive_window_pairs(
        var=var,
        context_keys=CONTEXT,
        state_keys=STATE,
        state_prev_suffix=SUFFIX,
        unroll_steps=k,
        extra_target_keys=EXTRA,
    )
    n_windows = var["T"].shape[1] - k

    for t0 in range(n_windows):
        invar, outvar = _ar_window_sample(
            _frames(var, t0, k + 1),
            context_keys=CONTEXT,
            state_keys=STATE,
            state_prev_suffix=SUFFIX,
            unroll_steps=k,
            extra_target_keys=EXTRA,
        )
        assert set(invar) == set(ds_input), f"invar key mismatch at window {t0}"
        assert set(outvar) == set(ds_output), f"outvar key mismatch at window {t0}"
        for key, arr in ds_input.items():
            np.testing.assert_allclose(invar[key], arr[t0], err_msg=f"{key} @ window {t0}")
        for key, arr in ds_output.items():
            np.testing.assert_allclose(outvar[key], arr[t0], err_msg=f"{key} @ window {t0}")


def test_one_step_sample_matches_eager_pairs() -> None:
    """k=1 degenerates to the one-step contract the validator is built on."""
    var = _temporal_var()
    ds_input, ds_output = _build_autoregressive_pairs(
        var=var,
        context_keys=CONTEXT,
        state_keys=STATE,
        state_prev_suffix=SUFFIX,
        extra_target_keys=EXTRA,
    )

    for t0 in range(var["T"].shape[1] - 1):
        invar, outvar = _ar_window_sample(
            _frames(var, t0, 2),
            context_keys=CONTEXT,
            state_keys=STATE,
            state_prev_suffix=SUFFIX,
            unroll_steps=1,
            extra_target_keys=EXTRA,
        )
        assert set(invar) == set(ds_input)
        assert set(outvar) == set(ds_output)
        for key, arr in ds_input.items():
            np.testing.assert_allclose(invar[key], arr[t0], err_msg=f"{key} @ pair {t0}")
        for key, arr in ds_output.items():
            np.testing.assert_allclose(outvar[key], arr[t0], err_msg=f"{key} @ pair {t0}")


def test_window_sample_rejects_short_frame_run() -> None:
    var = _temporal_var()
    with pytest.raises(ValueError, match="requires 4 frames"):
        _ar_window_sample(
            _frames(var, 0, 3),
            context_keys=CONTEXT,
            state_keys=STATE,
            state_prev_suffix=SUFFIX,
            unroll_steps=3,
            extra_target_keys=EXTRA,
        )


def test_entries_cover_variable_length_cases() -> None:
    """The whole point of the lazy path: cases need not share a timestep count."""

    class _Store:
        def time_steps(self, case_idx: int) -> int:
            return {0: 14, 1: 400, 2: 21}[case_idx]

    entries = _build_ar_window_entries(_Store(), [0, 1, 2], unroll_steps=3)

    assert len(entries) == (14 - 3) + (400 - 3) + (21 - 3)
    assert entries[0] == (0, 0)
    assert max(t for c, t in entries if c == 0) == 10  # last window reads frames 10..13
    assert all(t >= 0 for _, t in entries)


def test_entries_skip_cases_too_short_to_unroll() -> None:
    class _Store:
        def time_steps(self, case_idx: int) -> int:
            return {0: 3, 1: 9}[case_idx]

    entries = _build_ar_window_entries(_Store(), [0, 1], unroll_steps=4)

    assert [c for c, _ in entries] == [1] * 5  # case 0 (nt=3) cannot form a 4-window


def test_noise_perturbs_seed_prev_only() -> None:
    var = _temporal_var()
    frames = _frames(var, 0, 4)
    kw = dict(
        context_keys=CONTEXT,
        state_keys=STATE,
        state_prev_suffix=SUFFIX,
        unroll_steps=3,
        extra_target_keys=EXTRA,
    )
    clean_in, clean_out = _ar_window_sample(frames, **kw)
    noisy_in, noisy_out = _ar_window_sample(
        frames, prev_noise_std=1.0, rng=np.random.default_rng(0), **kw
    )

    for key in STATE:
        assert not np.allclose(noisy_in[f"{key}{SUFFIX}"], clean_in[f"{key}{SUFFIX}"])
    for key, arr in clean_out.items():
        np.testing.assert_allclose(noisy_out[key], arr, err_msg=f"target {key} must stay clean")
    for key in CONTEXT:
        np.testing.assert_allclose(noisy_in[key], clean_in[key])
        np.testing.assert_allclose(noisy_in[f"{key}__ar1"], clean_in[f"{key}__ar1"])
