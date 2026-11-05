"""Power-weighted path_dose (opt-in) for power-varying datasets.

Default path_dose is per-case dwell-normalized [0,1] and power-blind (a
constant-per-case power cancels in the normalization). power_weighted scales it
by power_w/_P_REF_W AFTER normalization so the channel amplitude encodes power.
"""
import numpy as np
import pytest

from lpno.custom_utils.npz_datasource import _build_path_channels


def _mani(power):
    return {
        "input": {"laser": {"beam_radius_m": 5e-5, "scan_speed_m_per_s": 0.5, "power_w": power}},
        "derived": {"laser_path_xy_m": [[-5e-4, 0.0], [5e-4, 0.0]], "laser_off_time_s": 1e9},
    }


def _grid():
    x = np.linspace(-7.5e-4, 7.5e-4, 20).astype("f4")
    y = np.linspace(-5e-4, 5e-4, 15).astype("f4")
    z = np.linspace(0, 3e-4, 8).astype("f4")
    t = np.linspace(0, 2e-3, 10)
    return x, y, z, t


def _dose(power, weighted):
    x, y, z, t = _grid()
    return _build_path_channels(["path_dose"], _mani(power), x, y, z, t, power_weighted=weighted)["path_dose"]


def test_off_is_power_blind_and_default():
    # default (off) is identical regardless of power — the pre-existing behavior.
    a = _build_path_channels(["path_dose"], _mani(150.0), *_grid(), )["path_dose"]  # default off
    b = _dose(50.0, False)
    assert np.allclose(a, b)


def test_weighted_scales_by_power_over_ref():
    off = _dose(150.0, False)
    m = off > 1e-6
    assert np.allclose((_dose(150.0, True))[m] / off[m], 150.0 / 100.0, atol=1e-4)
    assert np.allclose((_dose(70.0, True))[m] / off[m], 70.0 / 100.0, atol=1e-4)


def test_every_dose_channel_is_power_weighted_not_just_path_dose():
    """A dose channel measures deposited energy, so ALL of them must carry power.

    Found 2026-08-14: `power_weighted` only scaled `path_dose`, leaving
    `path_dose_recent` and the multi-tau bank power-blind even with the flag ON.
    On a power-varying set that makes a 68 W and a 500 W case with the same track
    indistinguishable in every memory channel but one.
    """
    x, y, z, t = _grid()
    keys = ["path_dose", "path_dose_recent"]
    off = _build_path_channels(keys, _mani(250.0), x, y, z, t, power_weighted=False)
    on = _build_path_channels(keys, _mani(250.0), x, y, z, t, power_weighted=True)
    for k in keys:
        m = off[k] > 1e-6
        assert m.any(), f"{k} is all-zero, test is vacuous"
        assert np.allclose(on[k][m] / off[k][m], 250.0 / 100.0, atol=1e-4), (
            f"{k} ignored power_weighted"
        )


def test_config_key_reaches_run_cfg():
    """`custom.power_weighted_dose: true` in a yaml must actually switch it on.

    Found 2026-08-14: `power_weighted_dose_enabled` reads
    `getattr(run_cfg, "power_weighted_dose", False)`, but `config_loading` never
    populated the field and `RunConfig` never declared it -- so the documented
    config key was dead and only the env var worked. A silently-ignored config key
    is worse than no key: the yaml reads as if the channel is power-aware.
    """
    from dataclasses import fields

    from lpno.models.pipeline.dataclasses import RunConfig
    from lpno.models.pipeline.npz_lazy_store import power_weighted_dose_enabled

    assert "power_weighted_dose" in {f.name for f in fields(RunConfig)}, (
        "RunConfig must declare power_weighted_dose or the config key is dead"
    )

    class _Cfg:
        power_weighted_dose = True

    assert power_weighted_dose_enabled(_Cfg()) is True


def test_geometry_channels_stay_power_blind():
    """beam_dist is the FOOTPRINT (a normalized spatial kernel), not an energy.

    Power belongs in the dose family and in the `P` channel; scaling the geometry
    would conflate "where the beam is" with "how hard it drives".
    """
    x, y, z, t = _grid()
    off = _build_path_channels(["beam_dist"], _mani(250.0), x, y, z, t, power_weighted=False)
    on = _build_path_channels(["beam_dist"], _mani(250.0), x, y, z, t, power_weighted=True)
    assert np.allclose(off["beam_dist"], on["beam_dist"])


def test_recency_dose_is_causal_and_decays():
    # Causal exp-decay dose (lever 1): at a region the beam passed early, cumulative
    # dose only grows, but the recency dose must PEAK then DECAY once the beam leaves.
    x, y, z, t = _grid()
    ch = _build_path_channels(["path_dose", "path_dose_recent"], _mani(100.0), x, y, z, t)
    cum = ch["path_dose"][0, :, 0, :, :, 0]
    rec = ch["path_dose_recent"][0, :, 0, :, :, 0]
    ix, iy = np.unravel_index(cum[1].argmax(), cum[1].shape)  # start-region voxel
    assert cum[-1, ix, iy] >= cum[1, ix, iy] - 1e-6          # cumulative never decreases
    assert rec[-1, ix, iy] < rec[:, ix, iy].max() - 1e-6      # recency decays after beam leaves
    assert rec.max() < 10.0                                    # O(1) normalized


# --------------------------------------------------------------------------- #
# conduction_field: the same rule, its own flag
# --------------------------------------------------------------------------- #
def _cond(power, *, dose=False, conduction=False, cache_dir=None, case="c"):
    """conduction_field for one case. Its builder caches to disk keyed WITHOUT power,
    so every call here gets its own cache dir unless a shared one is passed on purpose."""
    import os

    x, y, z, t = _grid()
    m = _mani(power)
    m["case_name"] = case
    m["derived"]["surface_z_m"] = 0.0
    old = os.environ.get("PINO_CONDUCTION_CACHE")
    if cache_dir is not None:
        os.environ["PINO_CONDUCTION_CACHE"] = str(cache_dir)
    try:
        return _build_path_channels(
            ["conduction_field"], m, x, y, z, t,
            power_weighted=dose, power_weighted_conduction=conduction,
        )["conduction_field"]
    finally:
        if cache_dir is not None:
            if old is None:
                os.environ.pop("PINO_CONDUCTION_CACHE", None)
            else:
                os.environ["PINO_CONDUCTION_CACHE"] = old


def test_conduction_is_power_blind_by_default(tmp_path):
    """The reported defect: two cases differing only in power gave a bit-identical
    channel (pks 24940/25824, 100 vs 140 W, maxabs 0.000e+00)."""
    a = _cond(100.0, cache_dir=tmp_path / "a")
    b = _cond(300.0, cache_dir=tmp_path / "b")
    assert np.array_equal(a, b), "default must stay power-blind"


def test_conduction_responds_to_its_own_flag(tmp_path):
    base = _cond(250.0, cache_dir=tmp_path / "off")
    scaled = _cond(250.0, conduction=True, cache_dir=tmp_path / "on")
    assert np.allclose(scaled, base * 2.5, rtol=1e-5, atol=1e-8)
    assert not np.array_equal(scaled, base)


def test_power_weighted_dose_alone_leaves_conduction_UNCHANGED(tmp_path):
    """The regression that protects the manuscript's tabled models: every one of them
    sets power_weighted_dose=True, so that flag must not start rescaling conduction."""
    base = _cond(250.0, cache_dir=tmp_path / "neither")
    dose_only = _cond(250.0, dose=True, cache_dir=tmp_path / "dose")
    assert np.array_equal(base, dose_only), "power_weighted_dose must not touch conduction"


def test_the_scale_is_applied_after_the_cache_so_a_hit_is_not_mis_scaled(tmp_path):
    """The cache key carries no power term. Scaling inside the cached artifact would
    serve the wrong power on every hit -- the 2026-08-13 multibeam collision class.
    One shared cache dir, same case: the unweighted read must survive a weighted one."""
    shared = tmp_path / "shared"
    first = _cond(250.0, conduction=True, cache_dir=shared)      # writes the cache
    second = _cond(250.0, conduction=False, cache_dir=shared)    # reads it back
    assert np.allclose(first, second * 2.5, rtol=1e-5, atol=1e-8)


def test_conduction_config_key_reaches_run_cfg():
    """`custom.power_weighted_conduction: true` must actually switch it on -- the same
    dead-yaml-key trap `power_weighted_dose` hit on 2026-08-14."""
    from dataclasses import fields

    from lpno.models.pipeline.dataclasses import RunConfig
    from lpno.models.pipeline.npz_lazy_store import (
        power_weighted_conduction_enabled,
        power_weighted_dose_enabled,
    )

    assert "power_weighted_conduction" in {f.name for f in fields(RunConfig)}, (
        "RunConfig must declare power_weighted_conduction or the config key is dead"
    )

    class _Cfg:
        power_weighted_conduction = True
        power_weighted_dose = False

    assert power_weighted_conduction_enabled(_Cfg()) is True
    assert power_weighted_dose_enabled(_Cfg()) is False, "the two flags are independent"


# --------------------------------------------------------------------------- #
# The EAGER path must honour the same flags as the lazy store
# --------------------------------------------------------------------------- #
# Until 2026-09-08 `load_consolidated_npz` took no config and called
# `_build_path_channels` with six positional arguments, so both flags stayed False:
# `custom.power_weighted_dose: true` was silently inert on that path while the lazy
# store honoured it. One config, two different input channels, decided by which
# loader ran -- the class of divergence that invalidates a comparison without warning.
def _eager_case(tmp_path, power, case="e0"):
    """One consolidated NPZ plus its sidecar manifest, the eager loader's input."""
    import json

    x, y, z, t = _grid()
    nt, nx, ny, nz = t.size, x.size, y.size, z.size
    d = tmp_path / f"{case}"
    d.mkdir(parents=True, exist_ok=True)
    fields = {
        "x": x, "y": y, "z": z, "time_s": t.astype("f8"),
        "power_w": np.float64(power), "scan_speed_m_per_s": np.float64(0.5),
        "case_name": np.array(case),
    }
    rng = np.random.default_rng(0)
    for name in ("Temperature", "Fraction Of Fluid", "Pressure", "Solid Fraction"):
        fields[name] = rng.random((nt, nx, ny, nz)).astype("f4")
    fields["Velocity"] = rng.random((nt, nx, ny, nz, 3)).astype("f4")
    np.savez(d / f"{case}.npz", **fields)
    m = _mani(power)
    m["case_name"] = case
    m["derived"]["surface_z_m"] = 0.0
    (d / f"{case}.manifest.json").write_text(json.dumps(m))
    return d


def _eager(tmp_path, channel, *, power=250.0, dose=False, conduction=False, tag="t"):
    import os

    from lpno.custom_utils.npz_datasource import load_consolidated_npz

    d = _eager_case(tmp_path / tag, power)
    os.environ["PINO_CONDUCTION_CACHE"] = str(tmp_path / tag / "cache")
    try:
        var = load_consolidated_npz(
            d, requested_fields=["Temperature"], time_axis=True,
            avg_window_size=1, moving_frame_steps=None, moving_ref_frame=False,
            path_channels=[channel],
            power_weighted=dose, power_weighted_conduction=conduction,
        )
    finally:
        os.environ.pop("PINO_CONDUCTION_CACHE", None)
    return var[channel]


def test_eager_dose_honours_power_weighted_dose(tmp_path):
    off = _eager(tmp_path, "path_dose", dose=False, tag="a")
    on = _eager(tmp_path, "path_dose", dose=True, tag="b")
    assert not np.allclose(off, on), "eager path ignored power_weighted_dose"
    assert np.allclose(on, off * 2.5, rtol=1e-5, atol=1e-8)


def test_eager_conduction_honours_its_own_flag(tmp_path):
    off = _eager(tmp_path, "conduction_field", conduction=False, tag="c")
    on = _eager(tmp_path, "conduction_field", conduction=True, tag="d")
    assert np.allclose(on, off * 2.5, rtol=1e-5, atol=1e-8)


def test_eager_dose_flag_alone_leaves_conduction_UNCHANGED(tmp_path):
    """The donor-protecting invariant, mirrored from the lazy side."""
    base = _eager(tmp_path, "conduction_field", dose=False, tag="e")
    dose_only = _eager(tmp_path, "conduction_field", dose=True, tag="f")
    assert np.array_equal(base, dose_only), "power_weighted_dose must not touch conduction"


def test_the_production_caller_passes_both_resolvers():
    """The two-hop thread: signature accepts them AND load_and_preprocess_data forwards
    them from the same resolvers the lazy store uses. A kwarg nothing passes is inert."""
    import inspect

    from lpno.custom_utils.npz_datasource import load_consolidated_npz
    from lpno.models.pipeline import preprocessing_datasets as PD

    params = inspect.signature(load_consolidated_npz).parameters
    assert {"power_weighted", "power_weighted_conduction"} <= set(params)
    assert all(params[k].default is False for k in ("power_weighted", "power_weighted_conduction"))
    src = inspect.getsource(PD.load_and_preprocess_data)
    assert "power_weighted=power_weighted_dose_enabled(run_cfg)" in src
    assert "power_weighted_conduction=power_weighted_conduction_enabled(run_cfg)" in src


# --------------------------------------------------------------------------- #
# A requested channel must be built, or refused loudly
# --------------------------------------------------------------------------- #
def test_tau_only_request_builds_the_bank(tmp_path):
    """Reported 2026-09-08 (random paths): the tau bank is built INSIDE a block gated on
    `local_keys | family_keys`, and none of the twelve tau names was in either, so a
    request naming only tau channels returned {} -- no error, no channel, a model trained
    on a silently truncated input. Live configs escaped only by also asking for
    `beam_dist`; the tau-only ablation is the case that breaks."""
    import os

    from lpno.custom_utils.npz_datasource import PATH_CHANNEL_NAMES

    x, y, z, t = _grid()
    m = _mani(200.0)
    m["case_name"] = "tau"
    m["derived"]["surface_z_m"] = 0.0
    tau = [k for k in PATH_CHANNEL_NAMES if "tau" in k]
    assert len(tau) == 12, "bare + _sigma + _sigma_exp"
    os.environ["PINO_CONDUCTION_CACHE"] = str(tmp_path / "cache")
    try:
        out = _build_path_channels(tau, m, x, y, z, t)
    finally:
        os.environ.pop("PINO_CONDUCTION_CACHE", None)
    assert set(out) >= set(tau), f"not built: {sorted(set(tau) - set(out))}"
    for k in tau:
        assert np.isfinite(out[k]).all() and np.any(out[k] != 0), f"{k} built but empty"


def test_a_requested_channel_that_is_not_built_raises(tmp_path):
    """The general guard: the tau gap was silent only because nothing compared the
    request against the result. Simulate the next such gap by shrinking the gate."""
    import os

    from lpno.custom_utils import npz_datasource as nds

    x, y, z, t = _grid()
    m = _mani(200.0)
    m["case_name"] = "guard"
    m["derived"]["surface_z_m"] = 0.0
    os.environ["PINO_CONDUCTION_CACHE"] = str(tmp_path / "cache")
    real = nds._TAU_BANK_NAMES
    try:
        nds._TAU_BANK_NAMES = ()          # re-open the exact hole that was fixed
        with pytest.raises(KeyError, match="requested but not built"):
            nds._build_path_channels(["path_dose_tau30us"], m, x, y, z, t)
    finally:
        nds._TAU_BANK_NAMES = real
        os.environ.pop("PINO_CONDUCTION_CACHE", None)


def test_a_full_request_returns_every_channel(tmp_path):
    """The guard refuses only genuine misses: all of PATH_CHANNEL_NAMES round-trips."""
    import os

    from lpno.custom_utils.npz_datasource import PATH_CHANNEL_NAMES

    x, y, z, t = _grid()
    m = _mani(200.0)
    m["case_name"] = "full"
    m["derived"]["surface_z_m"] = 0.0
    os.environ["PINO_CONDUCTION_CACHE"] = str(tmp_path / "cache")
    try:
        out = _build_path_channels(list(PATH_CHANNEL_NAMES), m, x, y, z, t)
    finally:
        os.environ.pop("PINO_CONDUCTION_CACHE", None)
    assert set(out) >= set(PATH_CHANNEL_NAMES)
