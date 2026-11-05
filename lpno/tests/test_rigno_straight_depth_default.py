"""The straight-path RIGNO config trains at depth 8, the random-path one at 18.

Depth 18 was made the RIGNO default on 2026-08-23 on RANDOM-PATH evidence -- the
d18_g8 recipe was the best RIGNO on the 1601 random-path set (model 62929,
substrate 10.91 K), and the config comment says so. On the calibrated STRAIGHT
path the measurement runs the other way: matched seeds with one-cycle and
num_graph_variants=8 on both sides give depth 8 a beam-box MAE of
105.90 +/- 1.85 K against depth 18's 164.47 +/- 93.92 K on ds55017, at an
indistinguishable substrate MAE (34.46 vs 34.80 K). The seed spread is the
decisive part: two depth-8 seeds agree within 2.6 K where three depth-18 seeds
span 87-269 K.

These tests pin BOTH values so neither can drift into the other's territory
unnoticed -- a straight-path config that silently inherits 18 again would
reproduce the instability, and a random-path config dropped to 8 would discard
the evidence it was chosen on.
"""
from __future__ import annotations

from pathlib import Path

import yaml

CONF = Path(__file__).resolve().parents[1] / "src" / "lpno" / "conf" / "hydra"


def _arch_rigno(name: str) -> dict:
    doc = yaml.safe_load((CONF / f"{name}.yaml").read_text())
    return doc["arch"]["rigno"]


def test_straight_path_config_trains_at_depth_8() -> None:
    rigno = _arch_rigno("config_flow3D_RIGNO_straight_onecycle")
    assert rigno["processor_steps"] == 8, (
        "straight-path RIGNO must train at depth 8; depth 18 was chosen on "
        "random-path evidence and is markedly less stable on the straight path"
    )


def test_straight_path_config_keeps_the_graph_variant_count() -> None:
    # The eight pre-existing depth-8 lab arms ran num_graph_variants=1 with the
    # expLR schedule, which confounded depth with two other knobs. The measured
    # depth-8 result is at ngv=8 with one-cycle, so the config must pin ngv.
    rigno = _arch_rigno("config_flow3D_RIGNO_straight_onecycle")
    assert rigno["num_graph_variants"] == 8


def test_random_path_config_still_trains_at_depth_18() -> None:
    rigno = _arch_rigno("config_flow3D_RIGNO_randompath_onecycle")
    assert rigno["processor_steps"] == 18, (
        "random-path RIGNO keeps depth 18: that is the recipe it was measured "
        "best on (1601, model 62929). The straight-path change must not leak here"
    )
