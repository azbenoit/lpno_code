"""`time_frames`: a time subsample the derivation profile can actually express.

Before this, a dataset holding 20 of 60 frames carried a derivation profile
BYTE-IDENTICAL to its full-resolution source, so the transformation existed only
in prose and the dataset had to be marked unreconstructable
(`resampled_straight_path_data`, measured 2026-08-10).
"""

from __future__ import annotations

import numpy as np
import pytest

from aiida_pino.data.pino_dataset_collection import _subsample_time, frame_indices
from aiida_pino.utils.dataset_derivation import (
    identity_derivation_profile,
    validate_derivation_profile,
)


def test_the_real_case_20_of_60():
    """The measured pattern: ends kept, steps of 3 and 4 -- not uniform.

    Keeping the true endpoints matters more than uniform spacing: the last
    frame is the converged state every metric is read from.
    """
    keep = frame_indices(60, 20)
    assert len(keep) == 20
    assert keep[0] == 0 and keep[-1] == 59
    assert sorted(set(np.diff(keep))) == [3, 4]


def test_no_subsample_keeps_everything():
    assert frame_indices(60, None) == list(range(60))
    assert frame_indices(60, 0) == list(range(60))
    assert frame_indices(20, 60) == list(range(20)), "asking for more than exist keeps all"


def test_arrays_and_the_time_axis_stay_in_step():
    arrays = {
        "time_s": np.arange(60, dtype=float),
        "Temperature": np.arange(60 * 2 * 2 * 2, dtype=np.float32).reshape(60, 2, 2, 2),
        "x": np.linspace(0, 1, 2),  # an axis, not a frame series
    }
    out = _subsample_time(arrays, 20)
    assert out["time_s"].shape == (20,)
    assert out["Temperature"].shape == (20, 2, 2, 2)
    assert out["x"].shape == (2,), "spatial axes must not be touched"
    # the kept frames are the SAME frames in both
    assert np.allclose(out["time_s"], frame_indices(60, 20))
    assert np.allclose(out["Temperature"][:, 0, 0, 0],
                       arrays["Temperature"][frame_indices(60, 20), 0, 0, 0])


def test_the_profile_accepts_and_validates_it():
    profile = validate_derivation_profile({"time_frames": 20})
    assert profile["time_frames"] == 20
    assert validate_derivation_profile({})["time_frames"] is None

    with pytest.raises(ValueError, match="time_frames must be >= 2"):
        validate_derivation_profile({"time_frames": 1})


def test_a_default_time_frames_cannot_fork_a_dataset():
    """Same rule as stride: present-but-default must not change identity.

    A default `stride: [1, 1, 1]` once minted a live twin of the random-path test
    split -- the dataset every published evaluation of it points at.
    """
    assert "time_frames" not in identity_derivation_profile({"time_frames": None})
    assert identity_derivation_profile({}) == identity_derivation_profile({"time_frames": None})
    assert identity_derivation_profile({"time_frames": 20}) != identity_derivation_profile({})
