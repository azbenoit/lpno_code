"""Spatial striding is available to RIGNO runs, but never automatic.

RIGNO's cost scales with grid points, and striding keeps a full-resolution run
inside the 24 GB card (22.2 -> 14.7 GB measured on the 300-case random-path dataset). It is applied
at read time by ``lpno``' NpzDirStore, so no second copy of the
dataset is written.

``RIGNO_DEFAULT_SPATIAL_STRIDE`` is empty on purpose: several splits on disk are
*already* 75x50x40 (`ti64_random150_ds_*`), and an automatic stride would
silently downsample those a second time. A caller asks for it explicitly via the
training profile's ``env``.
"""
from __future__ import annotations

import pytest
from aiida_pino.utils.profiles import (
    RIGNO_DEFAULT_SPATIAL_STRIDE,
    load_training_profile_defaults,
    merge_training_profile,
)

RIGNO_CONFIG = "config_flow3D_RIGNO_randompath"
NON_RIGNO_CONFIG = "config_NS_energy_PINO_flow3D"
STRIDE_KEY = "PINO_LAZY_STRIDE"


def _base(config_name: str, **extra):
    return {
        "config_name": config_name,
        "run_name": "stride_test",
        "hydra_overrides": [],
        "cuda_device": "0",
        "wandb_offline": True,
        **extra,
    }


def test_nothing_is_downsampled_unless_asked():
    """No automatic stride: an already-coarse dataset must not be halved again."""
    assert RIGNO_DEFAULT_SPATIAL_STRIDE == ""
    profile = load_training_profile_defaults(RIGNO_CONFIG)
    assert STRIDE_KEY not in (profile.get("env") or {})


def test_non_rigno_config_also_carries_no_stride():
    profile = load_training_profile_defaults(NON_RIGNO_CONFIG)
    assert STRIDE_KEY not in (profile.get("env") or {})


def test_explicit_stride_is_carried_through():
    """The feature still works: asking for it puts it on the job."""
    merged = merge_training_profile(_base(RIGNO_CONFIG, env={STRIDE_KEY: "2,2,1"}))
    assert merged["env"][STRIDE_KEY] == "2,2,1"


def test_several_env_entries_all_survive():
    """env merges per key, like `autoregressive` -- it does not replace wholesale.

    Regression for the bug this file found: `merged.update(...)` used to drop
    every packaged default the moment a caller passed one variable of its own.
    """
    merged = merge_training_profile(
        _base(
            RIGNO_CONFIG,
            env={
                STRIDE_KEY: "2,2,1",
                "PINO_LOADER_WORKERS": "8",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
        )
    )
    assert merged["env"][STRIDE_KEY] == "2,2,1"
    assert merged["env"]["PINO_LOADER_WORKERS"] == "8"
    assert merged["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


@pytest.mark.requires_lpno
def test_env_defaults_survive_a_partial_override(monkeypatch):
    """A packaged default and a caller entry coexist, whatever the default is."""
    import aiida_pino.utils.profiles as p

    monkeypatch.setattr(p, "RIGNO_DEFAULT_SPATIAL_STRIDE", "2,2,1")
    defaults = p.load_training_profile_defaults(RIGNO_CONFIG)
    assert defaults["env"][STRIDE_KEY] == "2,2,1"
    merged = p.merge_training_profile(_base(RIGNO_CONFIG, env={"PINO_LOADER_WORKERS": "8"}))
    assert merged["env"]["PINO_LOADER_WORKERS"] == "8"
    assert merged["env"][STRIDE_KEY] == "2,2,1", "packaged default was clobbered"


def test_merged_rigno_profile_still_validates():
    merged = merge_training_profile(_base(RIGNO_CONFIG))
    assert merged["config_name"] == RIGNO_CONFIG
    assert isinstance(merged["env"], dict)
