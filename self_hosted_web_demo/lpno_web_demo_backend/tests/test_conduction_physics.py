"""The conduction channel must be built the way each served model trained on it.

The uniform-drawing re-run models (custom.conduction_source_sigma_factor=0.5,
custom.power_weighted_conduction=true) were about to be served while
_path_channels passed neither setting to the lpno builder, which then built the
original channel (source sigma = beam radius, unscaled) for every model. These
pin the meta.json fields, their use in the builder, and the load-time check
against the checkpoint's own config.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.adapter_straightpath import StraightPathAdapter
from app.config import Settings
from app.model_physics import ModelPhysics, physics_config_mismatches, physics_from_meta


@pytest.fixture(scope="module")
def adapter() -> StraightPathAdapter:
    return StraightPathAdapter(Settings(use_stub_runtime=True), stub=True)


def _conduction(adapter: StraightPathAdapter, physics: ModelPhysics, p_watts: float = 150.0) -> np.ndarray:
    case = adapter.resolve(p_watts=p_watts, v_scan_mps=0.6, physics=physics)
    return np.asarray(adapter._path_channels(case, keys=["conduction_field"])["conduction_field"])


def test_meta_fields_parse_and_default_to_the_original_channel() -> None:
    assert physics_from_meta({}).power_weighted_conduction is False
    assert physics_from_meta({}).conduction_sigma_factor == 1.0
    ph = physics_from_meta({"physics": {"power_weighted_conduction": True, "conduction_source_sigma_factor": 0.5}})
    assert ph.power_weighted_conduction is True and ph.conduction_sigma_factor == 0.5


def test_the_builder_receives_the_models_conduction_settings(adapter: StraightPathAdapter) -> None:
    """The channel differs between factor 0.5 and 1.0, and equals the lpno builder called directly."""
    wide = _conduction(adapter, ModelPhysics(power_weighted_dose=True))
    narrow = _conduction(adapter, ModelPhysics(power_weighted_dose=True, conduction_sigma_factor=0.5))
    assert wide.max() > 0 and narrow.max() > 0
    assert not np.allclose(wide, narrow, rtol=1e-3, atol=1e-6)

    ph = ModelPhysics(power_weighted_dose=True, power_weighted_conduction=True, conduction_sigma_factor=0.5)
    case = adapter.resolve(p_watts=150.0, v_scan_mps=0.6, physics=ph)
    served = np.asarray(adapter._path_channels(case, keys=["conduction_field"])["conduction_field"])
    nds = adapter._npz_datasource()
    x, y, z = adapter.grid()
    direct = nds._build_path_channels(["conduction_field"], case.manifest, x, y, z, case.times_fine,
                                      power_weighted=True, power_weighted_conduction=True,
                                      conduction_sigma_factor=0.5)["conduction_field"][0][case.frame_idx]
    np.testing.assert_array_equal(served, np.asarray(direct))


def test_power_weighted_conduction_scales_with_power(adapter: StraightPathAdapter) -> None:
    ph = ModelPhysics(power_weighted_dose=True, power_weighted_conduction=True, conduction_sigma_factor=0.5)
    c100, c200 = _conduction(adapter, ph, 100.0), _conduction(adapter, ph, 200.0)
    assert c200.max() == pytest.approx(2.0 * c100.max(), rel=1e-5)
    plain = ModelPhysics(power_weighted_dose=True, conduction_sigma_factor=0.5)
    assert _conduction(adapter, plain, 200.0).max() == pytest.approx(_conduction(adapter, plain, 100.0).max(), rel=1e-5)


def test_meta_and_config_must_agree() -> None:
    ph = ModelPhysics(power_weighted_dose=True, power_weighted_conduction=True, conduction_sigma_factor=0.5)
    cfg = {"power_weighted_dose": True, "power_weighted_conduction": True, "conduction_source_sigma_factor": 0.5}
    assert physics_config_mismatches(ph, cfg) == []
    stale = ModelPhysics(power_weighted_dose=True)          # a meta.json that forgot the conduction settings
    lines = physics_config_mismatches(stale, cfg)
    assert len(lines) == 2 and any("conduction_source_sigma_factor" in s for s in lines)
    # keys the config does not state mean the pipeline defaults
    assert physics_config_mismatches(ModelPhysics(power_weighted_dose=True), {"power_weighted_dose": True}) == []
    assert physics_config_mismatches(ModelPhysics(), {"power_weighted_conduction": None}) == []


def test_served_grid_shape_is_the_page_grid_without_a_model_grid(adapter: StraightPathAdapter) -> None:
    """The FNO repair compares trained_grid.json against this shape; the straight page serves 90x40x40."""
    x, y, z = adapter.grid()
    assert adapter.served_grid_shape("fno_cal") == (x.size, y.size, z.size)
    assert adapter.served_grid_shape("fno_cal") == (90, 40, 40)
