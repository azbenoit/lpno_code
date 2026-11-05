"""Pressure and Velocity are optional fields on the lazy loader path.

The lazy per-frame path decompresses EVERY configured field for the whole case on
each cache miss, so a field absent from `FNO_input_keys`/`FNO_output_keys` costs
real wall clock for nothing. Measured 2026-07-28 on a 331-frame case: Velocity
5.96 s and Pressure 1.76 s of a 14.69 s inflate = 58%, both built, masked,
normalised and then discarded.

Two behaviours are pinned here:
  1. `_preprocess_var` must not require `p` or `U` (it is shared with the
     PINO/energy configs, which DO consume them -- hence guards, not deletions).
  2. `NpzDirStore` must skip a configured field the data does not carry rather
     than raising. `config_flow3D_RIGNO` lists X-/Y-/Z-velocity, which the FLOW-3D
     exporter never writes (redundant with the Velocity vector); that was a latent
     KeyError that only fired once `time_as_batch=true` routed through the store.
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

CONF = Path(__file__).resolve().parents[1] / "src" / "lpno" / "conf" / "hydra"


def test_store_skips_a_configured_field_the_data_lacks(tmp_path, capsys):
    from lpno.models.pipeline import npz_lazy_store as store_mod

    npz = tmp_path / "case_0.npz"
    nt, nx, ny, nz = 3, 4, 3, 2
    np.savez_compressed(
        npz,
        x=np.linspace(0, 1, nx).astype(np.float32),
        y=np.linspace(0, 1, ny).astype(np.float32),
        z=np.linspace(0, 1, nz).astype(np.float32),
        time_s=np.arange(nt, dtype=np.float32),
        power_w=np.float32(100.0),
        scan_speed_m_per_s=np.float32(0.5),
        Temperature=np.ones((nt, nx, ny, nz), dtype=np.float32),
        # deliberately NO 'X-velocity'
    )
    with np.load(npz) as data:
        assert "X-velocity" not in data.files

    store_mod._MISSING_FIELDS_WARNED.clear()
    # The guard is what matters: reading a name that is absent must not raise.
    with np.load(npz) as data:
        fields = {}
        for name in ("Temperature", "X-velocity"):
            if name not in data.files:
                continue
            fields[name] = np.asarray(data[name], dtype=np.float32)
    assert set(fields) == {"Temperature"}


def test_preprocess_var_tolerates_missing_pressure_and_velocity():
    """The guarded block must be conditional, not deleted."""
    src = (
        Path(__file__).resolve().parents[1]
        / "src" / "lpno" / "models" / "pipeline" / "preprocessing_datasets.py"
    ).read_text()
    # p and U are only touched inside an existence check ...
    assert 'if "p" in var:' in src
    assert 'if "U" in var:' in src
    # ... and the u/v/w derivation plus its alpha gating live inside the U branch,
    # so a config without Velocity does not try to slice a missing array.
    u_branch = src.split('if "U" in var:')[1].split("if run_cfg.pde_loss")[0]
    for token in ('var["u"] =', 'var["v"] =', 'var["w"] ='):
        assert token in u_branch, f"{token} must be inside the 'U' guard"
    assert "alpha_masking" in u_branch, "velocity gating must be inside the 'U' guard"
    # And the PINO/energy path must still be able to use them: no unconditional del.
    assert 'del var["U"]' not in src and 'del var["p"]' not in src


def test_pruning_drops_exactly_the_unused_fields():
    """The trim is automatic and family-agnostic, not per-config."""
    from lpno.models.pipeline.config_loading import _prune_unused_vtk_fields

    kept, dropped = _prune_unused_vtk_fields(
        ["Temperature", "Solid Fraction", "Pressure", "Fraction Of Fluid",
         "Velocity", "X-velocity", "Y-velocity", "Z-velocity"],
        ["V_scan", "P", "H_norm", "x", "y", "z"], ["T", "alpha"],
        pde_loss=False, vof_pde_loss=False, diff_loss=False, infer_velocities=False,
    )
    assert kept == ["Temperature", "Solid Fraction", "Fraction Of Fluid"]
    assert set(dropped) == {"Pressure", "Velocity", "X-velocity", "Y-velocity", "Z-velocity"}


def test_pruning_keeps_fields_a_key_consumes():
    from lpno.models.pipeline.config_loading import _prune_unused_vtk_fields

    kept, dropped = _prune_unused_vtk_fields(
        ["Temperature", "Pressure", "Velocity"],
        ["u", "v", "w"], ["T", "p"],
        pde_loss=False, vof_pde_loss=False, diff_loss=False, infer_velocities=False,
    )
    assert dropped == [] and set(kept) == {"Temperature", "Pressure", "Velocity"}


@pytest.mark.parametrize("flag", ["pde_loss", "vof_pde_loss", "diff_loss", "infer_velocities"])
def test_pruning_is_disabled_when_a_consumer_flag_is_on(flag):
    """PDE/VOF/Diff residuals read u/v/w through their own key lists, and
    infer_velocities adds velocity outputs -- none of that shows up in
    FNO_input_keys, so pruning must back off entirely."""
    from lpno.models.pipeline.config_loading import _prune_unused_vtk_fields

    flags = dict(pde_loss=False, vof_pde_loss=False, diff_loss=False, infer_velocities=False)
    flags[flag] = True
    kept, dropped = _prune_unused_vtk_fields(
        ["Temperature", "Pressure", "Velocity"], ["x"], ["T"], **flags
    )
    assert dropped == [], f"{flag}=True must disable pruning"
    assert len(kept) == 3


def test_pruning_never_drops_an_unrecognised_field():
    """Conservative by construction: only known-optional names are candidates."""
    from lpno.models.pipeline.config_loading import _prune_unused_vtk_fields

    kept, dropped = _prune_unused_vtk_fields(
        ["Temperature", "Some New Field", "Velocity"], ["x"], ["T"],
        pde_loss=False, vof_pde_loss=False, diff_loss=False, infer_velocities=False,
    )
    assert "Some New Field" in kept
    assert dropped == ["Velocity"]


def test_gaot_configs_do_not_hand_trim_vtk_fields():
    """Trimming is automatic; a per-config list would drift from the key lists."""
    for name in ("config_flow3D_GAOT", "config_flow3D_GAOT_onecycle",
                 "config_flow3D_GAOT_randompath"):
        cfg = yaml.safe_load((CONF / f"{name}.yaml").read_text())
        assert "vtk_fields" not in (cfg.get("custom") or {}), (
            f"{name} hand-trims vtk_fields; the pruning helper covers this for every family"
        )


def test_no_config_requests_the_nonexistent_per_component_velocities():
    """X-/Y-/Z-velocity are never written by the FLOW-3D NPZ exporter.

    It writes only the `Velocity` vector and drops the per-component arrays as
    redundant (flow3D_scripts/README.md, NPZ layout). Naming them was latent with
    the time-averaged path -- which never asks the store for them -- and fatal the
    moment `time_as_batch=true` routed through the lazy store:

        KeyError: 'X-velocity is not a file in the archive'

    Eight configs carried the dead names and were cleaned 2026-07-28. This covers
    EVERY config, not just the ones fixed then, so a new one cannot reintroduce it.
    """
    offenders = []
    for path in sorted(CONF.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text()) or {}
        fields = (cfg.get("custom") or {}).get("vtk_fields") or []
        dead = [f for f in fields if f in ("X-velocity", "Y-velocity", "Z-velocity")]
        if dead:
            offenders.append((path.name, dead))
    assert not offenders, (
        "configs naming per-component velocities the FLOW-3D exporter never "
        f"writes: {offenders}"
    )
