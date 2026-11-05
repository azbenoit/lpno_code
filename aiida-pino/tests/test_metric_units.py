"""An evaluation's numbers carry their own unit.

Metrics are stored normalised (T divided by `ref_scales["T"]`), so `T.rmse =
0.0056427` is 16.93 K only if you know the factor 3000. Review fix 6b: until
now that factor lived solely in the *model's* archived `.hydra/config.yaml`,
missing for 19 of 122 models including the declared 1601 baseline — a headline
result whose unit lived in another node's config file.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from aiida_pino.console import model_metrics

METRICS = {"T": {"rmse": 0.0056427, "abs_err_mean": 0.004}, "alpha": {"iou_mean": 0.9}}


class _FakeRepo:
    def __init__(self, content):
        self.content = content

    def get_object_content(self, name):
        if self.content is None or name != ".hydra/config.yaml":
            raise FileNotFoundError(name)
        return self.content


class _FakeModel:
    """Just enough model to read an archived Hydra config off."""

    def __init__(self, content):
        self.base = type("B", (), {"repository": _FakeRepo(content)})()


def test_units_are_built_from_the_models_archived_config():
    model = _FakeModel("model:\n  loader:\n    ref_scales:\n      T: 3000.0\n")
    units = model_metrics.build_metric_units(model, METRICS)
    assert units["temperature_scale_k"] == 3000.0
    assert "ref_scales.T=3000" in units["temperature_scale_source"]
    # the published random-path baseline: 0.0056427 * 3000 = 16.93 K
    assert units["kelvin"]["T.rmse"] == pytest.approx(16.9281)
    # dimensionless metrics are not "converted" into nonsense
    assert "alpha.iou_mean" not in units["kelvin"]


def test_a_model_without_an_archived_config_yields_no_kelvin():
    """19 of 122 models have no config. They must report normalised numbers,
    never a plausible-looking 3000x guess."""
    units = model_metrics.build_metric_units(_FakeModel(None), METRICS)
    assert units["temperature_scale_k"] is None
    assert units["kelvin"] == {}
    assert units["temperature_scale_source"]  # says why, for the UI to show


def test_named_config_is_a_labelled_second_tier():
    """Models that archive no config fall back to the *shipped* config of the
    name they trained under — the only way the random-path RIGNO baseline gets a
    unit at all. It must never masquerade as the model's own record."""
    factor, source = model_metrics.named_config_scale("config_flow3D_RIGNO_randompath")
    if factor is not None:  # lpno present, as in this environment
        assert factor == 3000.0
        assert "shipped config" in source and "not from a copy" in source
    # an unknown or empty name resolves to nothing rather than to a default
    assert model_metrics.named_config_scale("no_such_config_anywhere") == (None, "")
    assert model_metrics.named_config_scale("") == (None, "")


def test_recorded_factor_prefers_attributes_then_extras():
    attrs = {"metric_units": {"temperature_scale_k": 3000.0, "temperature_scale_source": "record"}}
    extras = {"metric_units": {"temperature_scale_k": 1500.0, "temperature_scale_source": "backfill"}}
    assert model_metrics.recorded_kelvin_factor(attrs, extras) == (3000.0, "record")
    assert model_metrics.recorded_kelvin_factor({}, extras) == (1500.0, "backfill")
    # nothing recorded, or an unusable value, falls through to the caller
    assert model_metrics.recorded_kelvin_factor({}, {}) == (None, "")
    assert model_metrics.recorded_kelvin_factor({}, {"metric_units": {"temperature_scale_k": 0}}) == (None, "")
    assert model_metrics.recorded_kelvin_factor({}, {"metric_units": "nonsense"}) == (None, "")


def test_evaluation_metadata_carries_the_unit(aiida_profile_clean, pino_model_factory, pino_dataset_factory):
    """The end state that matters: the unit is on the evaluation record itself,
    so reading a metric never requires opening another node."""
    from aiida import orm

    from aiida_pino.utils import build_evaluation_metadata

    model = pino_model_factory()
    dataset = pino_dataset_factory("units-ds")
    metadata = build_evaluation_metadata(
        model,
        dataset,
        {},
        orm.Dict(dict=METRICS),
        orm.Dict(dict={"status": "ok"}),
        {},
    )
    assert "metric_units" in metadata
    units = metadata["metric_units"]
    assert set(units) >= {"temperature_scale_k", "temperature_scale_source", "kelvin"}
    # the factory's model may or may not archive a config; either way the block
    # is present and self-consistent
    if units["temperature_scale_k"]:
        assert units["kelvin"]["T.rmse"] == pytest.approx(METRICS["T"]["rmse"] * units["temperature_scale_k"])
    else:
        assert units["kelvin"] == {}


def test_a_nested_archived_config_is_found():
    """A run whose artifacts were archived under the run directory keeps its
    config at `<run>/.hydra/config.yaml`. Reading only the root path made model
    9716 — the declared 1601 baseline — look like it archived nothing, so its
    unit was resolved from a repo file and stamped "this model archives no
    config of its own", which was false and went into the database.
    """
    class _NestedRepo:
        def list_object_names(self, path=None):
            return ["rigno_kh2579_lowlr_step18000"]

        def get_object_content(self, name):
            if name == "rigno_kh2579_lowlr_step18000/.hydra/config.yaml":
                return "model:\n  loader:\n    ref_scales:\n      T: 3000.0\n"
            raise FileNotFoundError(name)

    model = type("M", (), {"base": type("B", (), {"repository": _NestedRepo()})()})()
    assert model_metrics.model_hydra_config(model)["model"]["loader"]["ref_scales"]["T"] == 3000.0
    units = model_metrics.build_metric_units(model, METRICS)
    assert units["temperature_scale_k"] == 3000.0
    assert "archived Hydra config" in units["temperature_scale_source"]
    assert "shipped config" not in units["temperature_scale_source"]


@pytest.mark.requires_lpno
def test_the_fallback_says_why_the_models_own_config_could_not_answer():
    """A model that archives a config *without* `ref_scales` is not a model that
    archives no config. Hardcoding the latter put a false sentence on two
    evaluations, in the one field a reviewer checks against the repository."""
    has_config_but_no_scales = _FakeModel("model:\n  loader:\n    lr: 0.001\n")
    units = model_metrics.build_metric_units(has_config_but_no_scales, METRICS)
    source = units["temperature_scale_source"]
    if units["temperature_scale_k"]:  # a shipped config answered
        assert "no ref_scales" in source or "has no `ref_scales" in source
        assert "archives no config of its own" not in source

    # and the genuinely config-less case still says exactly that
    _, bare = model_metrics.named_config_scale(
        "config_flow3D_RIGNO_randompath", because="no archived Hydra config on this model"
    )
    assert "no archived Hydra config on this model" in bare
