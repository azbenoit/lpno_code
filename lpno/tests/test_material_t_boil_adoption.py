"""The mask fill follows the DATA's material file, not a hardcoded constant.

`T_BOIL_K = 3123.0` is the Mills Ti64 vaporization temperature frozen into code. The
calibrated material declares 3520 K, and on calibrated data the 3123 fill sits below real
metal peaks -- the measured frozen-tail-deficit cap. The fix that generalizes to other
materials: read `material.properties.vaporization_temperature_k` from the dataset's own
case manifests at load time.

Precedence: explicit `custom.T_boil_k` > the dataset's material file > legacy 3123.
Multi-material sets (disagreeing manifests) keep the legacy default with a loud warning,
so the material-variation campaign's behaviour is unchanged until a per-case fill exists.
"""
import json

import numpy as np
import pytest

from lpno.custom_utils.utils import boiling_temperature, t_boil_from_manifest
from lpno.models.pipeline.preprocessing_datasets import adopt_material_t_boil


def test_reads_the_flow3d_case_manifest():
    man = {"material": {"properties": {"vaporization_temperature_k": 3520.0}}}
    assert t_boil_from_manifest(man) == 3520.0


def test_reads_the_arraydir_member_wrapper():
    man = {"simulation_metadata": {"flow3d_case_manifest": {
        "material": {"properties": {"vaporization_temperature_k": 3123.15}}}}}
    assert t_boil_from_manifest(man) == 3123.15


def test_absent_material_returns_none():
    assert t_boil_from_manifest({}) is None
    assert t_boil_from_manifest({"material": {"properties": {}}}) is None
    assert t_boil_from_manifest(None) is None


def _dir_with_manifests(tmp_path, values):
    for i, v in enumerate(values):
        man = {"material": {"properties": {"vaporization_temperature_k": v},
                            "name": f"mat{v}"}}
        (tmp_path / f"case{i}.manifest.json").write_text(json.dumps(man))
    return tmp_path


def test_single_material_dataset_sets_the_fill(tmp_path):
    d = _dir_with_manifests(tmp_path, [3520.0, 3520.0, 3520.0])
    mp = {}
    adopt_material_t_boil(mp, d)
    assert mp["T_boil"] == 3520.0
    assert boiling_temperature(mp) == 3520.0


def test_explicit_override_wins(tmp_path):
    d = _dir_with_manifests(tmp_path, [3520.0])
    mp = {"T_boil": 4000.0}
    adopt_material_t_boil(mp, d)
    assert mp["T_boil"] == 4000.0


def test_disagreeing_materials_keep_the_legacy_default(tmp_path, capsys):
    d = _dir_with_manifests(tmp_path, [3520.0, 3123.15])
    mp = {}
    adopt_material_t_boil(mp, d)
    assert "T_boil" not in mp
    assert boiling_temperature(mp) == 3123.0
    assert "multi-material" in capsys.readouterr().out.lower()


def test_no_manifests_keep_the_legacy_default(tmp_path):
    mp = {}
    adopt_material_t_boil(mp, tmp_path)
    assert "T_boil" not in mp


def test_member_layout_is_found(tmp_path):
    m = tmp_path / "members" / "simulation_0000"
    m.mkdir(parents=True)
    (m / "manifest.json").write_text(json.dumps(
        {"simulation_metadata": {"flow3d_case_manifest": {
            "material": {"properties": {"vaporization_temperature_k": 3520.0}}}}}))
    mp = {}
    adopt_material_t_boil(mp, tmp_path)
    assert mp["T_boil"] == 3520.0
