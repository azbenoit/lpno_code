"""A read-time CROP reduces the cached case exactly as a stride does.

2026-08-26, measured. The guard's "full resolution"
test inspected only `PINO_LAZY_STRIDE`, so arms serving 90x40x30 via
`PINO_LAZY_CROP_Z_TOP=10` were charged the full-resolution floor of 40960 MiB:

    run A    (cropped, workers=2)  booked 40960  actual RSS  8188 MiB
    AR arm B (cropped, workers=2)  booked 40960  actual RSS  9673 MiB

The highest `host_ram_peak_mb` ever recorded on ANY RIGNO run is 16641. Nothing has
approached 40 GB.

It is not merely an over-booking. Host RAM is booked machine-wide against a usable pool of
~88542 MiB, so 40960 per arm admits TWO concurrent runs on a box with four 24 GB cards --
both machines sat half-idle with jobs queued while VRAM looked like the constraint.

What is deliberately NOT relaxed: the grid-aware floor. `full_res_host_ram_floor_mb`
scales with a declared `graph_nodes`, and a GAOT graph build genuinely peaks around 57 GB.
A cropped run builds a SMALLER graph, so a profile that declares its node count still gets
the floor its own grid implies. Only the flat anchor -- measured on the uncropped
150x100x40 grid -- stops applying to a grid that is not that one.
"""

from __future__ import annotations

import pytest

from aiida_pino.utils.profiles import validate_training_profile


# The same BASE the sibling guard test uses, so this file cannot drift from the shape a
# real profile must have.
BASE = {
    'config_name': 'config_flow3D_GAOT_varspeed_multitau',
    'run_name': 'x',
    'cuda_device': 'auto',
    'wandb_offline': True,
    'hydra_overrides': [],
}




def _profile(**over):
    """BASE with two loader workers and a 20480 booking, plus any override."""
    prof = dict(BASE)
    prof["host_ram_mb"] = 20480
    env = {"PINO_LOADER_WORKERS": "2"}
    env.update(over.pop("env", {}))
    prof["env"] = env
    prof.update(over)
    return prof


def test_a_cropped_run_is_not_charged_the_full_resolution_floor():
    """The reported case: 20480 booked, ~8-9.7 GiB actually used."""
    out = validate_training_profile(_profile(env={"PINO_LAZY_CROP_Z_TOP": "10"}))
    assert out["host_ram_mb"] == 20480


def test_an_uncropped_unstrided_run_still_is():
    """The guard must keep protecting the configuration it was calibrated on."""
    with pytest.raises(ValueError, match="too small"):
        validate_training_profile(_profile())


def test_a_strided_run_is_still_accepted():
    out = validate_training_profile(_profile(env={"PINO_LAZY_STRIDE": "2,2,1"}))
    assert out["host_ram_mb"] == 20480


def test_a_no_op_crop_does_not_count_as_a_reduction():
    """`PINO_LAZY_CROP_Z_TOP=0` serves the full grid; treating it as reduced would let a
    full-resolution run through on a quarter of the RAM it needs."""
    with pytest.raises(ValueError, match="too small"):
        validate_training_profile(_profile(env={"PINO_LAZY_CROP_Z_TOP": "0"}))


def test_a_declared_graph_node_count_still_sets_a_floor_even_when_cropped():
    """The GAOT protection is grid-aware and must survive: a cropped run that declares a
    large node count is still asked for the RAM that node count implies."""
    with pytest.raises(ValueError, match="too small"):
        validate_training_profile(
            _profile(graph_nodes=600000, env={"PINO_LAZY_CROP_Z_TOP": "10"})
        )


def test_the_refusal_names_the_crop_as_well_as_the_stride():
    """A message naming only PINO_LAZY_STRIDE sent the reporter looking at the wrong knob."""
    with pytest.raises(ValueError) as excinfo:
        validate_training_profile(_profile())
    assert "PINO_LAZY_CROP_Z_TOP" in str(excinfo.value)
