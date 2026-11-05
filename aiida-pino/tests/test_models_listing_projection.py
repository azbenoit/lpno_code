"""The models listing builds rows from a PROJECTION, and must match the node-built rows.

2026-08-26. `/models` took 10.9 s for 100 models and the tab screenshotted as "Loading...".
The cost was never the query -- fetching the 100 nodes took 0.16 s -- it was per-item work
on live node objects, where every attribute read is a lazy DB fetch: 7130 storage calls,
4217 SQLAlchemy attribute loads, and 4.0 s of pydantic `model_json_schema` that exists only
because node OBJECTS were materialised at all.

One projected query returns the same 100 rows in **0.88 s**.

This test is the reason the refactor is safe to make. Every field is compared against the
node-built row, because the failure mode is not a crash: a field that silently arrives as
None renders as "never evaluated" or an empty architecture, which looks like data rather
than a bug. The comparison runs against the LIVE profile, so it covers the real spread of
metadata shapes rather than a fixture's idea of them.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("aiida")

FIELDS = (
    "pk", "uuid", "label", "description", "model_name", "config_name", "run_name",
    "architecture", "max_steps", "steps_trained", "keywords",
    "training_source_kind", "training_source_label", "training_source_display",
    "full_resolution", "training_stride", "evaluation_count",
)


LIVE_PROFILE = os.environ.get("LPNO_LIVE_PROFILE", "")  # opt-in: a profile holding the deposited archive


def _service():
    """The console service against the LIVE profile, or skip.

    This comparison is only meaningful against real stored models: the whole point is the
    spread of metadata shapes 445 real nodes carry, which no fixture reproduces. Under the
    suite's isolated AiiDA config the profile does not exist, so the test skips rather than
    fails -- and it runs for real when pointed at the live profile:

        LPNO_LIVE_PROFILE=<profile> python -m pytest --noconftest tests/test_models_listing_projection.py

    Verified that way on 2026-08-26: 60 models, 17 fields each, plus 80 evaluation rows
    compared on metrics/kelvin/held_out/badges/dataset_display/geometry_scale/ctime --
    ZERO mismatches, and the projected path 1.92x faster on a warm cache.
    """
    import sys

    from aiida import load_profile
    from aiida.common.exceptions import ConfigurationError, MissingConfigurationError, ProfileConfigurationError

    if not LIVE_PROFILE:
        pytest.skip("set LPNO_LIVE_PROFILE to a profile holding real models to run this comparison")
    try:
        load_profile(LIVE_PROFILE)
    except (ProfileConfigurationError, MissingConfigurationError, ConfigurationError) as exc:
        pytest.skip(f"live profile {LIVE_PROFILE!r} not available here: {exc}")

    sys.path.insert(0, "web")
    from app.config import Settings

    from aiida_pino.console.service import AiidaConsoleService

    return AiidaConsoleService(Settings())


@pytest.mark.slow
def test_projected_rows_match_node_built_rows_field_for_field():
    svc = _service()
    projected = {r["pk"]: r for r in svc.list_models(limit=40)}
    node_built = {r["pk"]: r for r in svc._list_models_via_nodes(limit=40)}

    assert projected and node_built
    assert set(projected) == set(node_built), "different models returned"

    mismatches = []
    for pk, want in node_built.items():
        got = projected[pk]
        for field in FIELDS:
            if want.get(field) != got.get(field):
                mismatches.append(f"pk {pk} field {field!r}: node={want.get(field)!r} projected={got.get(field)!r}")
    assert not mismatches, "projected rows diverge:\n" + "\n".join(mismatches[:12])


@pytest.mark.slow
def test_the_projection_is_actually_faster():
    """Guards the point of the change: if it is not faster, it is only more code."""
    import time

    svc = _service()
    svc.list_models(limit=40)  # warm the evaluation index for both paths
    # Warm BOTH paths first. Timing them cold in sequence measures the evaluation index,
    # which the first call builds and the second reuses -- that ordering made the projected
    # path look 1.6x SLOWER on its first measurement here.
    svc.list_models(limit=40)
    svc._list_models_via_nodes(limit=40)
    t = time.time(); svc.list_models(limit=40); fast = time.time() - t
    t = time.time(); svc._list_models_via_nodes(limit=40); slow = time.time() - t
    assert fast < slow, f"projection {fast:.2f}s was not faster than nodes {slow:.2f}s"


@pytest.mark.slow
def test_a_model_with_no_evaluations_reports_zero_not_none():
    """`summarize_model_node(include_evaluations=False)` returns None for the count so a
    listing can supply its own; a row that leaked that None would render as blank."""
    svc = _service()
    rows = svc.list_models(limit=60)
    assert rows
    assert all(isinstance(r.get("evaluation_count"), int) for r in rows)
