"""The two-alloy label and the material-aware H_norm on staged (arraydir) members.

2026-10-12: every two-alloy model trained on process parameters alone, so a Ti-6Al-4V
case and an Inconel 718 case at the same laser settings presented identical inputs.
`mat_alloy_flag` names the alloy; `h_norm_include_material` puts King's material
factor into H_norm. These pin that the label reads the manifest correctly, that the
H_norm material values are served without becoming inputs, and that the denominator
can read them from a staged member (it could not: scalar_value had no material path).
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from lpno.custom_utils import npz_datasource as nds
from lpno.custom_utils.utils import h_norm_raw, king_material_factor
from lpno.models.pipeline.lazy_runtime import (
    RUNTIME_FORMAT_V2,
    ArrayDirCollectionStore,
    ArrayDirSimulationStore,
    compute_h_norm_denom,
)

TI64 = {
    "name": "ti64_v3_calibrated",
    "properties": {"density_kg_m3": 4420.0, "specific_heat_j_kg_k": 546.0,
                   "thermal_conductivity_w_m_k": 7.0, "latent_heat_j_kg": 345000.0,
                   "liquidus_temperature_k": 1923.15},
    "optical": {"normal_absorptivity": 0.35},
}
IN718 = {
    "name": "Inconel 718",
    "properties": {"density_kg_m3": 8190.0, "specific_heat_j_kg_k": 435.0,
                   "thermal_conductivity_w_m_k": 8.9, "latent_heat_j_kg": 210000.0,
                   "liquidus_temperature_k": 1608.15},
    "optical": {"normal_absorptivity": 0.35},
}


@pytest.mark.parametrize("material, flag", [
    (IN718, 1.0), (TI64, 0.0),
    ({"flow3d_label": "IN718"}, 1.0), ({"file": "in_718_v2.json"}, 1.0),
    ({"name": "AlSi10Mg"}, 0.0), ({"name": "SS316L"}, 0.0),
])
def test_flag_reads_the_material_name(material, flag):
    assert nds._alloy_flag({"material": material}) == flag


def test_flag_refuses_an_unnamed_material():
    with pytest.raises(KeyError):
        nds._alloy_flag({"material": {"properties": {}}})


def test_flag_is_a_material_channel_outside_the_manifest_six():
    assert "mat_alloy_flag" in nds.ALL_MATERIAL_CHANNEL_NAMES
    assert "mat_alloy_flag" not in nds.MATERIAL_CHANNEL_NAMES
    out = nds._build_material_channels(["mat_alloy_flag"], {"material": IN718}, (2, 3))
    assert out["mat_alloy_flag"].shape == (2, 3) and np.all(out["mat_alloy_flag"] == 1.0)


def test_h_norm_material_channels_are_served_but_not_declared():
    keys = ["P", "V_scan", "H_norm", "mat_alloy_flag"]
    off = SimpleNamespace(h_norm_include_material=False)
    on = SimpleNamespace(h_norm_include_material=True)
    assert nds.served_material_channels(off, keys) == ["mat_alloy_flag"]
    served = nds.served_material_channels(on, keys)
    assert served[0] == "mat_alloy_flag" and set(served[1:]) == set(nds.MATERIAL_CHANNEL_NAMES)


def _write_member(root, material, power, speed):
    root.mkdir(parents=True, exist_ok=True)
    arrays = {"Temperature": np.random.rand(1, 4, 3, 2).astype(np.float32),
              "P": np.full((1, 4, 3, 2), power, np.float32),
              "V_scan": np.full((1, 4, 3, 2), speed, np.float32)}
    for name, arr in arrays.items():
        np.save(root / f"{name}.npy", arr)
    manifest = {
        "format": RUNTIME_FORMAT_V2,
        "source_kind": "simulation",
        "simulation_metadata": {"case_name": root.name,
                                "flow3d_case_manifest": {"input": {}, "material": material}},
        "array_entries": [{"original_name": n, "filename": f"{n}.npy", "shape": list(a.shape)}
                          for n, a in arrays.items()],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_keyless_read_serves_the_label(tmp_path):
    _write_member(tmp_path / "a", IN718, 200.0, 1.0)
    _write_member(tmp_path / "b", TI64, 200.0, 1.0)
    assert np.all(ArrayDirSimulationStore(tmp_path / "a").read_case(0)["mat_alloy_flag"] == 1.0)
    assert np.all(ArrayDirSimulationStore(tmp_path / "b").read_case(0)["mat_alloy_flag"] == 0.0)
    keyed = ArrayDirSimulationStore(tmp_path / "a").read_case(0, keys=["Temperature", "mat_alloy_flag"])
    assert set(keyed) == {"Temperature", "mat_alloy_flag"}


def test_denominator_reads_material_from_staged_members(tmp_path):
    """compute_h_norm_denom with include_material must equal the numerator's max."""
    cases = [("a", IN718, 200.0, 1.0), ("b", TI64, 150.0, 0.5), ("c", TI64, 250.0, 1.2)]
    for name, material, power, speed in cases:
        _write_member(tmp_path / name, material, power, speed)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format": RUNTIME_FORMAT_V2, "source_kind": "dataset_collection",
        "members": [{"relative_path": n} for n, *_ in cases]}))
    store = ArrayDirCollectionStore(tmp_path)

    def numerator(material, power, speed):
        p = material["properties"]
        factor = king_material_factor(rho=p["density_kg_m3"], cp=p["specific_heat_j_kg_k"],
                                      k=p["thermal_conductivity_w_m_k"], latent=p["latent_heat_j_kg"],
                                      T_l=p["liquidus_temperature_k"], A=0.35)
        return float(h_norm_raw(power, speed, material_factor=factor))

    expected = max(numerator(m, p, v) for _, m, p, v in cases)
    got = compute_h_norm_denom(store, range(len(cases)), time_as_batch=False, include_material=True)
    assert np.isclose(got, expected, rtol=1e-6)
    blind = compute_h_norm_denom(store, range(len(cases)), time_as_batch=False)
    assert not np.isclose(got, blind)
