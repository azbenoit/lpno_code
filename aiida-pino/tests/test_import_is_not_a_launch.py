"""`import-model` must not be subject to launch-only checks.

2026-08-23, found while verifying the `import-model --training-source` fix end to end: a
dry-run import printed

    [profile] WARNING cuda_device='0' pins this run to ONE gpu

about a run that does not exist. An import registers a model from the artifacts of a
training that already ran -- it starts nothing, takes no GPU and no host RAM.

The warning is only the visible half. `_check_host_ram_against_loader_settings` RAISES, so
an import could be REFUSED over the loader settings of a finished training, which is
precisely the failure `validate_training_profile`'s `for_launch` docstring records for the
frozen profile of one random-path model.
"""

from __future__ import annotations

import pytest

from aiida_pino.utils.profiles import load_training_profile_defaults, validate_training_profile


def _profile(**kw):
    base = load_training_profile_defaults("config_NS_energy_PINO_flow3D")
    base.update({"run_name": "x", "split_mode": "standard"})
    base.update(kw)
    return base


def test_an_import_does_not_warn_about_a_pinned_gpu(capsys):
    """The config-derived card is not a choice about a run that never launches."""
    validate_training_profile(_profile(cuda_device="0"), for_import=True)
    assert "pins this run" not in capsys.readouterr().out


def test_a_launch_still_warns(capsys):
    """The warning must survive for the case it was written for."""
    validate_training_profile(_profile(cuda_device="0"), for_import=False, for_launch=True)
    assert "pins this run" in capsys.readouterr().out


def test_an_import_is_not_refused_over_loader_settings_of_a_finished_run():
    """The half that is not cosmetic: the host-RAM check raises."""
    tripping = _profile(
        host_ram_mb=1024,
        env={"PINO_LOADER_WORKERS": "8", "PINO_LAZY_STRIDE": "1,1,1"},
    )
    # A launch is entitled to refuse it...
    with pytest.raises(ValueError):
        validate_training_profile(dict(tripping), for_import=False, for_launch=True)
    # ...but an import of an already-finished training must not be.
    validate_training_profile(dict(tripping), for_import=True)
