"""The overlay conditioning channels, and their refusal to fail quietly.

A conditioning channel that silently broadcasts the same constant to every case
trains happily and answers a different question -- it becomes a bias term, and
nothing in the loss curve shows it. These tests pin the loud failures.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from lpno.custom_utils import npz_datasource as nds


MANIFEST = {
    "case_name": "ckc_007",
    "material": {
        "properties": {
            "density_kg_m3": 4420.0,
            "specific_heat_j_kg_k": 546.0,
            "thermal_conductivity_w_m_k": 22.0,
            "latent_heat_j_kg": 345000.0,
            "liquidus_temperature_k": 1923.0,
        },
        "optical": {"normal_absorptivity": 0.35},
    },
}

OVERLAY = {
    "ckc_007": {
        "k_liquidus_w_m_k": 31.5,
        "h_king": 6.25,
        "x_keyhole": 3.10,
        "tv_ratio": 2.016,
        "theta_v": 1.42,
        "theta_m": 2.86,
    }
}


class MaterialOverlayChannels(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.overlay_path = Path(self._tmp.name) / "overlay.json"
        self.overlay_path.write_text(json.dumps(OVERLAY), encoding="utf-8")
        nds._load_material_overlay.cache_clear()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(nds._load_material_overlay.cache_clear)

    def _with_env(self, value: str | None) -> None:
        import os

        old = os.environ.get(nds.MATERIAL_OVERLAY_ENV)
        if value is None:
            os.environ.pop(nds.MATERIAL_OVERLAY_ENV, None)
        else:
            os.environ[nds.MATERIAL_OVERLAY_ENV] = value

        def restore() -> None:
            if old is None:
                os.environ.pop(nds.MATERIAL_OVERLAY_ENV, None)
            else:
                os.environ[nds.MATERIAL_OVERLAY_ENV] = old

        self.addCleanup(restore)

    def test_all_material_channels_are_declared(self) -> None:
        # 6 manifest + 2 melt-window (mat_Ts/mat_Tv, added 2026-08-26 so the melt
        # window and the alpha-mask fill can follow the material) + 6 overlay + the
        # alloy label (mat_alloy_flag, 2026-10-12).
        self.assertEqual(len(nds.ALL_MATERIAL_CHANNEL_NAMES), 15)
        for name in ("mat_k_liq", "mat_H_king", "mat_X_keyhole",
                     "mat_tv_ratio", "mat_theta_v", "mat_theta_m"):
            self.assertIn(name, nds.ALL_MATERIAL_CHANNEL_NAMES)
        for name in ("mat_Ts", "mat_Tv"):
            self.assertIn(name, nds.ALL_MATERIAL_CHANNEL_NAMES)
            self.assertNotIn(name, nds.MATERIAL_CHANNEL_NAMES)  # never implied

    def test_overlay_channels_stay_out_of_the_manifest_backed_set(self) -> None:
        # MATERIAL_CHANNEL_NAMES doubles as "build all of these" for callers that
        # have no overlay. An overlay channel leaking into it turns every such
        # caller into a hard failure -- which is exactly what happened first try.
        self.assertEqual(len(nds.MATERIAL_CHANNEL_NAMES), 6)
        for name in nds.MATERIAL_OVERLAY_CHANNEL_NAMES:
            self.assertNotIn(name, nds.MATERIAL_CHANNEL_NAMES)

    def test_requesting_every_manifest_channel_needs_no_overlay(self) -> None:
        self._with_env(None)
        out = nds._build_material_channels(nds.MATERIAL_CHANNEL_NAMES, MANIFEST, (2, 2))
        self.assertEqual(set(out), set(nds.MATERIAL_CHANNEL_NAMES))

    def test_manifest_channels_still_work_without_an_overlay(self) -> None:
        # The six shipped channels must not acquire a dependency on the overlay.
        self._with_env(None)
        out = nds._build_material_channels(["mat_rho", "mat_A"], MANIFEST, (2, 2))
        self.assertEqual(set(out), {"mat_rho", "mat_A"})
        self.assertTrue(np.allclose(out["mat_rho"], 4420.0))
        self.assertTrue(np.allclose(out["mat_A"], 0.35))

    def test_overlay_channels_carry_their_per_case_value(self) -> None:
        self._with_env(str(self.overlay_path))
        out = nds._build_material_channels(
            ["mat_k_liq", "mat_H_king", "mat_X_keyhole", "mat_theta_v"],
            MANIFEST, (3, 3))
        self.assertTrue(np.allclose(out["mat_k_liq"], 31.5))
        self.assertTrue(np.allclose(out["mat_H_king"], 6.25))
        self.assertTrue(np.allclose(out["mat_X_keyhole"], 3.10))
        self.assertTrue(np.allclose(out["mat_theta_v"], 1.42))
        for arr in out.values():
            self.assertEqual(arr.shape, (3, 3))
            self.assertEqual(arr.dtype, np.float32)

    def test_missing_env_var_raises_rather_than_defaulting(self) -> None:
        self._with_env(None)
        with self.assertRaises(KeyError) as ctx:
            nds._build_material_channels(["mat_k_liq"], MANIFEST, (2, 2))
        self.assertIn(nds.MATERIAL_OVERLAY_ENV, str(ctx.exception))

    def test_case_absent_from_overlay_raises(self) -> None:
        self._with_env(str(self.overlay_path))
        other = dict(MANIFEST, case_name="ckc_999")
        with self.assertRaises(KeyError) as ctx:
            nds._build_material_channels(["mat_H_king"], other, (2, 2))
        self.assertIn("ckc_999", str(ctx.exception))

    def test_manifest_without_case_name_raises(self) -> None:
        self._with_env(str(self.overlay_path))
        anon = {"material": MANIFEST["material"]}
        with self.assertRaises(KeyError) as ctx:
            nds._build_material_channels(["mat_theta_m"], anon, (2, 2))
        self.assertIn("case_name", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
