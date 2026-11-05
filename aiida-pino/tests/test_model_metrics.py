"""Unit conversion and best-metric badges for model evaluations.

These are the numbers a person reads off the console and believes, so the rules
that matter are: a Kelvin value is only ever shown when the model's own scale
says so, ratios are never multiplied by it, and a badge moves the moment a
better evaluation exists.
"""

from __future__ import annotations

from aiida_pino.console import model_metrics as mm

FLAT = {
    "T.abs_err_mean": 0.0042426158326571,
    "T.rmse": 0.013575382722641,
    "T.rel_rmse": 0.03599676557313,
    "T.substrate_rmse": 0.01304958737024,
    "alpha.iou_mean": 0.99929071303234,
    "alpha.iou_std": 0.00077932178926285,
    "fl.iou_mean": 0.7533074379114,
    "fl.iou_std": 0.20872794583646,
    "geometry.melt_depth_um_true": 53.0,
    "geometry.melt_depth_um_pred": 51.5,
    "geometry.melt_depth_um_abs_err": 1.5,
}


def test_kelvin_factor_comes_from_the_models_own_archived_config() -> None:
    factor, why = mm.kelvin_factor({"arch": {"ref_scales": {"T": 3000, "L": 1e-4}}})
    assert factor == 3000.0
    assert "3000" in why


def test_kelvin_factor_refuses_to_guess() -> None:
    for config, reason in [(None, "no archived"), ({}, "no ref_scales"), ({"ref_scales": {"T": "nope"}}, "unusable")]:
        factor, why = mm.kelvin_factor(config)
        assert factor is None, f"guessed a scale from {config!r}"
        assert why


def test_temperature_errors_convert_but_ratios_do_not() -> None:
    kelvin = mm.to_kelvin(FLAT, 3000.0)
    assert round(kelvin["T.abs_err_mean"], 1) == 12.7
    assert round(kelvin["T.rmse"], 1) == 40.7
    assert round(kelvin["T.substrate_rmse"], 1) == 39.1
    assert "T.rel_rmse" not in kelvin, "a dimensionless ratio was scaled into fake Kelvin"


def test_no_factor_means_no_kelvin_at_all() -> None:
    assert mm.to_kelvin(FLAT, None) == {}


def test_iou_is_reported_as_percent_error() -> None:
    pct = mm.iou_error_pct(FLAT)
    assert round(pct["alpha.iou_err_pct"], 2) == 0.07
    assert round(pct["fl.iou_err_pct"], 1) == 24.7
    assert round(pct["fl.iou_std_pp"], 1) == 20.9


def test_geometry_rows_carry_relative_error() -> None:
    rows = mm.geometry_rows(FLAT)
    depth = next(r for r in rows if r["region"] == "melt" and r["quantity"] == "depth")
    assert depth["true"] == 53.0 and depth["pred"] == 51.5
    assert round(depth["rel_pct"], 1) == 2.8
    assert all(r["region"] != "keyhole" for r in rows), "invented keyhole rows from absent metrics"


def _row(eval_pk, model, dataset, mae, rmse=None, display=None):
    return {"eval_pk": eval_pk, "model_uuid": model, "dataset_key": dataset,
            "dataset_display": display or dataset, "t_mae_k": mae, "t_rmse_k": rmse}


def test_best_badges_are_per_dataset() -> None:
    ranked = mm.rank_evaluations([
        _row(1, "m1", "ds-a", 12.0), _row(2, "m2", "ds-a", 9.0),
        _row(3, "m3", "ds-b", 40.0),
    ])
    assert ranked[2]["badges"] == ["best T MAE (ds-a)"]
    assert ranked[1]["badges"] == []
    assert ranked[3]["badges"] == ["best T MAE (ds-b)"], "a different dataset has its own winner"
    assert ranked[1]["ranks"]["t_mae_k"] == {"rank": 2, "of": 2, "best": 9.0, "delta": 3.0}


def test_a_better_evaluation_takes_the_badge_over() -> None:
    before = mm.rank_evaluations([_row(1, "m1", "ds", 12.0)])
    assert before[1]["badges"] == ["best T MAE (ds)"]
    after = mm.rank_evaluations([_row(1, "m1", "ds", 12.0), _row(2, "m2", "ds", 8.0)])
    assert after[2]["badges"] == ["best T MAE (ds)"]
    assert after[1]["badges"] == [], "the superseded model kept its badge"


def test_models_without_a_known_kelvin_scale_do_not_win() -> None:
    ranked = mm.rank_evaluations([_row(1, "m1", "ds", None), _row(2, "m2", "ds", 30.0)])
    assert ranked[2]["badges"] == ["best T MAE (ds)"]
    assert ranked[1]["badges"] == [] and ranked[1]["ranks"] == {}


def test_ties_do_not_flap() -> None:
    ranked = mm.rank_evaluations([_row(7, "m1", "ds", 10.0), _row(3, "m2", "ds", 10.0)])
    assert ranked[3]["badges"] == ["best T MAE (ds)"], "the older evaluation should keep a tie"
    assert ranked[7]["badges"] == []


def test_both_metrics_get_their_own_winner() -> None:
    ranked = mm.rank_evaluations([
        _row(1, "m1", "ds", 10.0, 60.0), _row(2, "m2", "ds", 12.0, 40.0),
    ])
    assert ranked[1]["badges"] == ["best T MAE (ds)"]
    assert ranked[2]["badges"] == ["best T RMSE (ds)"]
