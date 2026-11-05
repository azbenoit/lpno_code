"""One resolver for "what grid did this run actually see" — both sources, composed.

The effective resolution of a run is `dataset.derivation_profile.stride`
(decimation baked into the served arrays) COMPOSED WITH `env.PINO_LAZY_STRIDE`
(a loader stride applied on top). Reading one source alone has now produced
three separate wrong answers:

- the leaderboard printed one label for three test-split nodes (2026-08-06, F3);
- a backfill read only one source and reported "0 models evaluated at two
  resolutions" when the true answer is 1 (`rigno_kh2579_lowlr_s18000`);
- on 2026-08-09 a correctly-configured queued run was killed because
  `attrs.get('derivation')` — the wrong key — returned None and was read as
  "the stride was dropped".

Composition is multiplication, not precedence: staging a pre-strided [2,2,1]
copy while `PINO_LAZY_STRIDE=2,2,1` is still set serves 38x25x40, the recorded
double-stride bug. A resolver that lets env "win over" the dataset stride calls
that case [2,2,1] and hides exactly the failure it exists to expose.
"""

from __future__ import annotations

import pytest

from aiida_pino.utils.profiles import effective_stride


def _attrs(stride=None, with_dp=True):
    if not with_dp:
        return {"compatibility_signature": "abc"}
    dp = {"include_time": True, "time_avg_window": 30}
    if stride is not None:
        dp["stride"] = stride
    return {"derivation_profile": dp, "compatibility_signature": "abc"}


def _profile(env_stride=None):
    env = {"PINO_LAZY_CACHE": "1"}
    if env_stride is not None:
        env["PINO_LAZY_STRIDE"] = env_stride
    return {"env": env}


def test_env_stride_alone():
    """Eval 9722: base dataset, loader stride 2,2,1."""
    assert effective_stride(_attrs(with_dp=False), _profile("2,2,1")) == (2, 2, 1)


def test_no_stride_anywhere_is_full_resolution():
    """Eval 12985: base dataset, no env stride."""
    assert effective_stride(_attrs(with_dp=False), _profile()) == (1, 1, 1)


def test_dataset_stride_alone():
    """Eval 28046: derived [2,2,1] dataset, no env stride."""
    assert effective_stride(_attrs([2, 2, 1]), _profile()) == (2, 2, 1)


def test_noop_dataset_stride_with_env_stride():
    """Eval 23925: derived [1,1,1] dataset plus env 2,2,1."""
    assert effective_stride(_attrs([1, 1, 1]), _profile("2,2,1")) == (2, 2, 1)


def test_composition_is_multiplication_not_precedence():
    """The double-stride bug: pre-strided copy + env stride = strided twice.

    An env-wins rule reports (2, 2, 1) here and hides the misconfiguration.
    """
    assert effective_stride(_attrs([2, 2, 1]), _profile("2,2,1")) == (4, 4, 1)


def test_malformed_env_stride_raises_not_degrades():
    """A value the loader itself would reject must be loud, never [1,1,1]."""
    with pytest.raises(ValueError):
        effective_stride(_attrs(with_dp=False), _profile("2,2"))
    with pytest.raises(ValueError):
        effective_stride(_attrs(with_dp=False), _profile("a,b,c"))


def test_malformed_dataset_stride_raises_not_degrades():
    with pytest.raises(ValueError):
        effective_stride(_attrs([2, 2]), _profile())


def test_schema_drift_refuses_to_guess():
    """The exact 2026-08-09 misread: a stride-bearing sibling key, wrong name.

    If the attributes carry something that looks like the derivation payload
    under an unexpected name, absence of `derivation_profile` is NOT evidence
    of "no stride" — refuse rather than silently answer full resolution.
    """
    attrs = {"derivation": {"stride": [2, 2, 1]}, "compatibility_signature": "abc"}
    with pytest.raises(KeyError):
        effective_stride(attrs, _profile())


def test_missing_env_block_is_fine():
    assert effective_stride(_attrs(with_dp=False), {}) == (1, 1, 1)
    assert effective_stride(_attrs(with_dp=False), {"env": None}) == (1, 1, 1)
