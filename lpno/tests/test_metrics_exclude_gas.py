"""The gas gap above the material must not be counted in the headline metrics.

With alpha masking the gas region is filled with the SAME constant in prediction and
truth, so it contributes ~zero error while making up ~25% of the domain. That dilutes
every mean by ~1/0.75. Measured 2026-07-30 on split 21285, substrate/global MAE ratio:

    field   GAOT cond150        RIGNO ti64ds3
    T       9.74 -> 12.95 K     6.81 -> 8.94 K     (1.33x, 1.31x)
    alpha   1.30x               1.26x
    fl      1.33x               1.33x

So every absolute figure reported before this was ~30% flattering. RMSE inflates less
(1.04-1.15) because it is dominated by large errors, which live in the metal anyway.
"""
import numpy as np

from lpno.models.pipeline.dataclasses import RunConfig
from lpno.models.pipeline.postprocessing import z_region_indices


def test_run_config_carries_the_flag_for_provenance():
    """It must be a real field, not an ad-hoc attribute, so it reaches AiiDA."""
    assert "metrics_exclude_gas" in RunConfig.__dataclass_fields__


def test_headline_keys_are_never_redefined_in_place():
    """`abs_err_mean` and `rmse` must stay WHOLE-DOMAIN regardless of the flag.

    All 120 stored evals carry whole-domain values under those keys, and the aiida-pino
    console ranks models on `T.abs_err_mean` (console/model_metrics.py), so writing
    substrate-only values under the same name silently mixes two populations in one
    leaderboard. An earlier version of this feature did exactly that. The gas-excluded
    figure is `substrate_abs_err_mean` / `substrate_rmse`, which every eval has always
    emitted -- the flag is a recorded preference, not a redefinition.
    """
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src" / "lpno" / "models" / "pipeline" / "postprocessing.py"
    ).read_text()
    assert 'payload["abs_err_mean"] = float(ra_sub' not in src
    assert 'payload["rmse"] = float(np.sqrt(ra_sub' not in src
    assert 'payload["metrics_exclude_gas"] = True' in src


def test_substrate_keys_are_what_carry_the_gas_excluded_number():
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src" / "lpno" / "models" / "pipeline" / "postprocessing.py"
    ).read_text()
    assert 'f"{tag}_abs_err_mean"' in src and 'f"{tag}_rmse"' in src


def test_substrate_mask_excludes_only_the_gas_slices():
    """z <= 0 is the material; z > 0 is the gas gap above it."""
    z = np.zeros((1, 1, 1, 5), dtype=np.float32)
    z[..., :] = np.array([-3.0, -2.0, -1.0, 0.0, 1.5])
    substrate, surf = z_region_indices(z)
    assert substrate.tolist() == [True, True, True, True, False]
    assert surf == 3  # the slice nearest 0 from below


def test_no_gas_gap_makes_it_a_no_op():
    """Datasets without an air gap must be unaffected -- the exclusion is
    dataset-specific by construction, not by configuration."""
    z = np.zeros((1, 1, 1, 4), dtype=np.float32)
    z[..., :] = np.array([-3.0, -2.0, -1.0, 0.0])
    substrate, _ = z_region_indices(z)
    assert substrate.all(), "with no z>0 slices the substrate mask covers everything"


def test_all_gas_returns_no_substrate():
    z = np.ones((1, 1, 1, 3), dtype=np.float32)
    substrate, surf = z_region_indices(z)
    assert substrate is None and surf is None


def test_dilution_is_visible_on_every_eval():
    """substrate_fraction makes the ~25% free region explicit without redefining keys."""
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src" / "lpno" / "models" / "pipeline" / "postprocessing.py"
    ).read_text()
    assert 'payload["substrate_fraction"]' in src
    assert 'payload["metrics_exclude_gas"] = True' in src
