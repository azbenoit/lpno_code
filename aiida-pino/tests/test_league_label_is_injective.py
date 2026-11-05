"""An evaluation row must name exactly one dataset.

Measured on the authors' database 2026-08-06: 14 of 152 evaluations — 9 % of the record —
print `Variable length random paths_test_split | var_1_time_noref | flow3d`
while sitting on three different nodes:

    node A    6 evals  stride [2, 2, 1]     <- half resolution
    node B    7 evals  stride None          <- full
    node C    1 eval   stride [1, 1, 1]

Downsampling smooths the temperature field, so the half-resolution scores come
out systematically lower. Three rows printed the same dataset string for
13.138 K, 13.478 K (both on node A) and 14.609 K (on node B); ranking them by the
printed Kelvin gives the wrong order. The badge path already qualified its label
through `_league_label`; the row a reader actually looks at did not.
"""

from __future__ import annotations

import pytest

from aiida_pino.console.config import Settings
from aiida_pino.console.service import AiidaConsoleService


@pytest.fixture
def svc():
    return AiidaConsoleService(Settings())


def test_the_suffix_names_the_node_and_flags_a_strided_one(aiida_profile_clean, pino_dataset_factory):
    full = pino_dataset_factory("league-full")
    assert svc_suffix(full) == f" (#{full.pk})"

    strided = pino_dataset_factory("league-strided")
    metadata = dict(strided.base.attributes.get("dataset_metadata") or {})
    metadata["derivation_profile"] = {**(metadata.get("derivation_profile") or {}), "stride": [2, 2, 1]}
    strided.backend_entity.set_attribute("dataset_metadata", metadata)
    assert svc_suffix(strided) == f" (#{strided.pk}, stride 2x2x1)"


def svc_suffix(dataset) -> str:
    return AiidaConsoleService(Settings())._league_suffix(dataset.uuid)


def test_an_unknown_uuid_adds_nothing(svc):
    assert svc._league_suffix("00000000-0000-0000-0000-000000000000") == ""
    assert svc._league_suffix(None) == ""


def test_two_datasets_sharing_a_display_name_get_different_row_labels(
    aiida_profile_clean, pino_dataset_factory
):
    """The defect, reduced: same printed name, different data."""
    a = pino_dataset_factory("same-name")
    b = pino_dataset_factory("same-name-too")
    b.label = a.label  # what the live pair does by accident
    service = AiidaConsoleService(Settings())

    class _Eval:
        label = "eval"

    metadata = {"dataset_source_label": a.label, "evaluation_name": "eval"}
    left = service._qualified_eval_dataset(_Eval(), metadata, a.uuid)
    right = service._qualified_eval_dataset(_Eval(), metadata, b.uuid)
    assert left != right, "two different datasets must not print the same row label"
    assert f"#{a.pk}" in left and f"#{b.pk}" in right


def test_a_dataset_with_no_name_stays_none(svc):
    """Appending to a missing display name must not invent one (or crash)."""
    assert svc._qualified_eval_dataset(type("E", (), {"label": None})(), {}, None) is None
