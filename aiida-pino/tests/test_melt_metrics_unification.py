"""The console scores melt-pool geometry with the SHARED definition.

Per the melt-metrics unification (2026-08-10):
`console/viz.py` reimplemented melt-pool
metrics in numpy and had already drifted -- with `Melt Region` absent it skipped
every melt metric silently, while `melt_metrics.melt_mask` falls back to
T > solidus. `ti64_varspeed150` (150 cases) is the concrete instance.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiida_pino.console import viz


def test_stats_version_invalidates_older_stats():
    """The version gate IS the invalidation mechanism; v8 recomputes v6/v7."""
    assert viz.STATS_VERSION == 8


def test_sim_fields_is_lazy(monkeypatch):
    """Loads a field on first access, not up front: the whole point of the Mapping.

    A real dict would pull every field of every case and throw away the
    streaming the viewer depends on (a 331-frame case is expensive).
    """
    loaded: list[str] = []

    class _Sim:
        uuid = "u-1"

        class base:
            class attributes:
                @staticmethod
                def get(key, default=None):
                    return {"Temperature": "T", "Melt Region": "MR"} if key == "array_aliases" else default

    monkeypatch.setattr(viz, "_field_stack_cached",
                        lambda uuid, alias: loaded.append(alias) or np.zeros((2, 2, 2, 2)))
    monkeypatch.setattr(viz, "_axes_cached",
                        lambda uuid: (np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)))

    fields = viz._SimFields(_Sim())
    assert loaded == [], "constructing the mapping must load nothing"
    assert set(fields) == {"Temperature", "Melt Region", "x", "y", "z", "time_s"}
    assert len(fields) == 6
    assert loaded == [], "iterating keys must load nothing"

    _ = fields["Temperature"]
    assert loaded == ["Temperature"], "only the accessed field is loaded"

    # melt_metrics probes with .get(); a missing field must answer None, not raise.
    assert fields.get("Heat Absorption Rate") is None
    with pytest.raises(KeyError):
        fields["Heat Absorption Rate"]


def test_solidus_comes_from_the_manifest_never_guessed():
    """melt_mask demands solidus explicitly: scoring one alloy with another's
    melting point is a silent error, so an absent value must stay absent."""

    def _sim(metadata, extras=None):
        store = dict(extras or {})

        class _Sim:
            class base:
                class attributes:
                    @staticmethod
                    def get(key, default=None):
                        return metadata if key == "simulation_metadata" else default

                class extras:
                    @staticmethod
                    def get(key, default=None):
                        return store.get(key, default)

        return _Sim()

    manifest = {"flow3d_case_manifest": {"material": {"properties": {"solidus_k": 1873.15}}}}
    assert viz._solidus_k(_sim(manifest)) == pytest.approx(1873.15)
    assert viz._solidus_k(_sim({})) is None

    # The stamped extra wins: for a case whose own manifest records no material,
    # it carries the value established from a same-case sibling.
    assert viz._solidus_k(_sim({}, {"solidus_k": 1873.15})) == pytest.approx(1873.15)
    # A nonsense stamp is ignored rather than propagated.
    assert viz._solidus_k(_sim({}, {"solidus_k": 0})) is None


def test_a_case_that_cannot_be_scored_says_why(monkeypatch):
    """Never a silent absence: an unscored case carries its reason."""
    monkeypatch.setattr(viz, "_melt_metrics_module", lambda: (None, "sibling not configured"))

    class _Sim:
        uuid = "u-2"

        class base:
            class attributes:
                @staticmethod
                def get(key, default=None):
                    return default

    fusion, why = viz.fusion_geometry_from_melt_metrics(_Sim())
    assert fusion == {}
    assert why == "sibling not configured"
