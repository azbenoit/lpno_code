"""The steady folders carry the H_norm denominator twice: meta.json physics
(the registry's copy, what /models reports) and flow3D_params.yaml ref_scales.H
(what adapter_pino actually feeds the model). Nothing else pins them together."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from app.config import Settings

FOLDERS = sorted(Path(Settings().steady_models_dir).glob("*/meta.json"))


@pytest.mark.parametrize("meta_path", FOLDERS, ids=[p.parent.name for p in FOLDERS])
def test_meta_h_norm_denom_equals_served_ref_scale_h(meta_path: Path) -> None:
    yaml_path = meta_path.parent / "flow3D_params.yaml"
    if not yaml_path.is_file():
        pytest.skip("no flow3D_params.yaml beside this meta.json")
    meta = json.loads(meta_path.read_text())
    served = yaml.safe_load(yaml_path.read_text())["ref_scales"]["H"]
    assert float(meta["physics"]["h_norm_denom"]) == pytest.approx(float(served), rel=0, abs=1e-9)
