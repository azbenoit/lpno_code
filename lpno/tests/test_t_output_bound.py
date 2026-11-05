"""The long-horizon GAOT configs must bound T, and the bound must be physical.

Root cause (verified in code 2026-07-30): the alpha mask node sits in the TRAINING
forward graph and computes `T*s + t_fill*(1-s)` with `s = sigmoid(20*(alpha-0.5))`, so
`d(loss)/d(raw T) = s`, which at `alpha ~ 0` is `sigmoid(-10) = 4.5e-5`. Raw T in the
gas is therefore effectively unsupervised and drifts without bound over a long run,
since the final projection is a plain linear layer with no range constraint. alpha IoU
is 0.99909, so ~0.09% of cells are misclassified; where a gas cell is predicted
`alpha = 1.0` the gate opens and the drifted value passes through at full weight.

Measured, eval 23531 on `gaot_mat_steady_20k`: T `pred_max` 21804.6 non-dimensional =
65.4 million K, at a cell with predicted alpha 1.0, z = +0.15 (above the surface) and
truth 1.041 (the gas fill); `pred_min` -0.202 = -607 K. T MAE 17.75 K, T RMSE 6279 K.
"""
from pathlib import Path

import yaml

CONF = Path(__file__).resolve().parents[1] / "src" / "lpno" / "conf" / "hydra"
REF_T = 3000.0
# Long-horizon runs need it too: gaot_cond_32k_ctrl (model 26734) trained cleanly to step
# 12000, spiked at 14000 and diverged monotonically from 16000 -- 1.4e6 K to 3.8e8 K by
# step 28000, alpha pinned at 0.74988 (the metal fraction, i.e. collapsed to a constant).
# The 20k runs finish before that window, which is why the bound was first scoped only to
# the material configs. The bound is a training-stability guard, not only a metric guard.
LONG_HORIZON_CONFIGS = ("config_flow3D_GAOT_randompath", "config_flow3D_GAOT_onecycle")


def _bounds(name):
    path = CONF / f"{name}.yaml"
    cfg = yaml.safe_load(path.read_text())
    return ((cfg.get("arch") or {}).get("gaot") or {}).get("output_bounds") or {}


def test_long_horizon_configs_bound_T_for_stability():
    """Divergence at step ~16000 is unrecoverable without a ceiling on T."""
    for name in LONG_HORIZON_CONFIGS:
        b = _bounds(name)
        assert "T" in b, (
            f"{name} does not bound T; gaot_cond_32k_ctrl diverged to 3.8e8 K from "
            "step 16000 with exactly this configuration"
        )
        lo, hi = b["T"]
        assert lo == 0.0 and hi * REF_T >= 3123.0, f"{name}: {lo}..{hi}"
