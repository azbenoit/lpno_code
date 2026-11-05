# pyright: reportAttributeAccessIssue=false
# pyright: reportUnboundVariable=false
# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportOptionalSubscript=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportOptionalMemberAccess=false
"""Data loading, normalization, split logic, and dataset assembly utilities."""

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.model_selection import KFold

try:
    from physicsnemo.sym.dataset import DictGridDataset, DictPointwiseDataset
except ImportError:
    from physicsnemo.sym.dataset import DictGridDataset

    DictPointwiseDataset = DictGridDataset
from ...custom_utils.dataset_loading import get_c_piecewise_func
from ...custom_utils.npz_datasource import (
    ALL_MATERIAL_CHANNEL_NAMES,
    PATH_CHANNEL_NAMES,
    is_consolidated_npz,
    load_consolidated_npz,
    served_material_channels,
)
from ...custom_utils.openFoam_datasource import VTKOpenFOAMDataSource
from ...custom_utils.utils import (
    boiling_temperature,
    fill_where_mask_smooth_np,
    make_interface_sharp,
    smooth_liquid_fraction_numpy,
    split_dict,
    t_boil_from_manifest,
    h_norm_raw,
    king_material_factor,
)
from .lazy_runtime import (
    ArrayDirCollectionStore,
    ArrayDirDatasetStore,
    ArrayDirSimulationStore,
    ArrayDirUnionStore,
    LazySampleDataset,
    MaskedTemporalPatchDataset,
    RuntimeStore,
    apply_random_mask,
    compute_h_norm_denom,
    is_arraydir_runtime,
    is_npz_file_collection,
    load_runtime_store,
)
from .h_norm_record import write_training_record as write_h_norm_record
from .input_fields_record import write_input_fields_record
from .npz_lazy_store import (
    NpzDirStore,
    maybe_build_npz_store,
    conduction_source_sigma_factor,
    power_weighted_conduction_enabled,
    power_weighted_dose_enabled,
)

# Every store the lazy training path recognises (arraydir runtimes + the lazy
# consolidated-NPZ store). Used by the store-vs-eager-dict dispatch below.
_RUNTIME_STORE_TYPES = (
    ArrayDirDatasetStore,
    ArrayDirUnionStore,
    ArrayDirSimulationStore,
    ArrayDirCollectionStore,
    NpzDirStore,
)

from .dataclasses import DatasetBundle, KeyConfig, RunConfig, SplitSpec


_ALPHA_MASKING_NOTICES: set[str] = set()


def _print_alpha_masking_notice_once(key: str) -> None:
    """Print alpha-masking behavior once per process instead of once per batch."""
    if key in _ALPHA_MASKING_NOTICES:
        return
    _ALPHA_MASKING_NOTICES.add(key)
    print(f"Alpha masking enabled for {key}; outside-metal values use boiling-temperature fill.")


def _rename_field(var: dict[str, np.ndarray], old: str, new: str) -> None:
    """Rename a parsed field in-place when the source backend uses legacy names."""

    if old in var:
        var[new] = var.pop(old)


def _define_material_properties(var: dict[str, np.ndarray], run_cfg: RunConfig) -> None:
    """Inject runtime material/laser properties derived from loaded sample tensors."""

    c_spec = run_cfg.material_params.get("c", "c")
    if c_spec == "c":
        if not run_cfg.param_dir:
            # config_loading deferred this path because no data root was resolvable.
            # Deferred, not dropped: this config DOES read it, so refuse here with the
            # two ways to fix it rather than with a confusing relative-path error.
            raise FileNotFoundError(
                "material_params.c='c' (piecewise tabulated) needs specific_heat.txt, "
                "but no parameter directory could be resolved on this machine. Set "
                "custom.param_dir to a directory containing specific_heat.txt, or set "
                "PINO_SIMULATIONS_DATA_ROOT to the repo data directory."
            )
        sh_path = Path(run_cfg.param_dir) / "specific_heat.txt"
        if not sh_path.exists():
            raise FileNotFoundError(
                f"material_params.c='c' (piecewise tabulated) requires '{sh_path}', "
                f"but it does not exist. Set custom.param_dir to a directory "
                f"containing specific_heat.txt."
            )
        c_fit, _, _ = get_c_piecewise_func(str(sh_path))
        var["c"] = c_fit(var["T"])
    else:
        print("Info: material_params.c is not 'c'; skipping temperature-dependent c(T).")

    run_cfg.material_params["weight"] = run_cfg.pde_weight
    run_cfg.laser_params["P"] = float(np.ravel(var["P"])[0])
    run_cfg.material_params["V_scan"] = float(np.ravel(var["V_scan"])[0])


def _require_physical_laser_parameters(var: dict[str, np.ndarray]) -> None:
    """Refuse a non-positive laser power or scan speed before forming H_norm.

    A datasource that could not read P/V_scan substituted 0.0, and
    H_norm = P/sqrt(|V_scan|) then evaluated 0/0 = NaN on a scan speed no welding
    case has. This is the choke point every datasource passes through, so the
    check lives here rather than in each loader. All 8176 consolidated NPZ cases
    under data/flow3D carry non-zero values.
    """

    # A negative V_scan is a legitimate reverse-direction scan (H_norm takes its
    # magnitude); a zero one is not a scan. Power has no such reading: it must be
    # positive.
    checks: list[tuple[str, float]] = []
    if "P" in var:
        checks.append(("P", float(np.min(np.asarray(var["P"])))))
    if "V_scan" in var:
        checks.append(("V_scan", float(np.min(np.abs(np.asarray(var["V_scan"]))))))
    for key, smallest in checks:
        if smallest <= 0.0:
            raise ValueError(
                f"'{key}' contains {smallest} -- physically invalid for a laser welding "
                f"case, and it makes H_norm = P/sqrt(|V_scan|) undefined. Check the "
                f"dataset's power_w / scan_speed_m_per_s (or the filename tokens the "
                f"VTK datasource parses)."
            )


def _warn_if_time_avg_window_is_inert(run_cfg) -> None:
    """Say so when `time_avg_window` is declared but cannot take effect.

    `time_as_batch` and `time_as_dim` both force `avg_window_size = 1` below, so a
    config carrying `time_avg_window: 30` alongside either declares something that
    never fires. Several configs do (RIGNO_randompath and RIGNO_randompath_onecycle
    among the shipped ones), and the lazy store does no
    averaging at all, so on that path it is inert regardless.

    Announce, do not change: the forcing is correct and long-standing. What cost
    something was that the config READ like a parameter -- a run was about to be
    re-launched on the eager path, at a 309 GB materialisation, specifically to make
    a 30 take effect that its own `time_as_batch` had already killed.
    """
    window = int(getattr(run_cfg, "time_avg_window", 1) or 1)
    if window <= 1:
        return
    for flag in ("time_as_batch", "time_as_dim"):
        if bool(getattr(run_cfg, flag, False)):
            print(
                f"NOTE: custom.time_avg_window={window} has NO EFFECT here -- "
                f"custom.{flag}=true forces the averaging window to 1, so this run "
                f"trains on per-frame data. The value is inert, not applied; drop it "
                f"or unset {flag} if averaging was intended.",
                flush=True,
            )
            return


def _slice_middle_y(var: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Slice y at mid-plane for grid tensors with spatial y dimension."""

    sliced: dict[str, np.ndarray] = {}
    for name, arr in var.items():
        if not isinstance(arr, np.ndarray):
            sliced[name] = arr
            continue
        if arr.ndim == 6:
            y_mid = arr.shape[4] // 2
            sliced[name] = arr[:, :, :, :, y_mid, :]
        elif arr.ndim == 5:
            y_mid = arr.shape[3] // 2
            sliced[name] = arr[:, :, :, y_mid, :]
        else:
            sliced[name] = arr
    return sliced


def _time_axis_to_dim(var: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert temporal tensors (B,T,C,X,Z) to spatialized form (B,C,X,T,Z)."""

    converted: dict[str, np.ndarray] = {}
    for name, arr in var.items():
        if isinstance(arr, np.ndarray) and arr.ndim == 5:
            converted[name] = np.transpose(arr, (0, 2, 3, 1, 4))
        else:
            converted[name] = arr
    return converted


def _flatten_time_axis(var: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Flatten temporal tensors from (B,T,C,...) to (B*T,C,...) per key."""

    temporal_keys = [name for name, arr in var.items() if isinstance(arr, np.ndarray) and arr.ndim in (5, 6)]
    if not temporal_keys:
        return var

    ref_key = temporal_keys[0]
    ref_shape = var[ref_key].shape
    ref_b, ref_t = ref_shape[0], ref_shape[1]
    flattened: dict[str, np.ndarray] = {}

    for name, arr in var.items():
        if isinstance(arr, np.ndarray) and arr.ndim in (5, 6):
            if arr.shape[0] != ref_b or arr.shape[1] != ref_t:
                raise ValueError(
                    f"Temporal tensor '{name}' has inconsistent leading dimensions {arr.shape[:2]} "
                    f"(expected {(ref_b, ref_t)} from '{ref_key}')."
                )
            flattened[name] = arr.reshape(ref_b * ref_t, *arr.shape[2:])
        else:
            flattened[name] = arr
    return flattened


def _flatten_spatial_pointwise(var: dict[str, np.ndarray], keys: Sequence[str]) -> dict[str, np.ndarray]:
    """Flatten structured tensors from ``(B, C, ...)`` to pointwise ``(N, C)`` rows."""

    flattened: dict[str, np.ndarray] = {}
    for key in keys:
        arr = np.asarray(var[key])
        if arr.ndim < 3:
            raise ValueError(
                f"Pointwise flattening expects structured tensors shaped like (B, C, ...); "
                f"key '{key}' has shape {arr.shape}."
            )
        moved = np.moveaxis(arr, 1, -1)
        flattened[key] = moved.reshape(-1, arr.shape[1])
    return flattened


def _fill_temperature(var, run_cfg):
    """Alpha-mask fill: the material's OWN vaporisation temperature when known.

    One fill per dataset is wrong on a multi-material split, and the resolver
    already says so and keeps the legacy 3520 K anyway. On ck300 that fills gas
    with Ti64's boiling point over materials that vaporise at 2068 K, which both
    mislabels the gas layer and sharpens the "hot means gas" signal that leaves
    predicted melt empty inside metal.
    """
    tv = var.get("mat_Tv")
    if tv is None:
        _fill_notice(False)
        return boiling_temperature(run_cfg.material_params)
    _fill_notice(True)
    return tv


_FILL_NOTED = set()


def _fill_notice(per_case: bool) -> None:
    if per_case in _FILL_NOTED:
        return
    _FILL_NOTED.add(per_case)
    print(f"[t_boil] mask fill: {'PER-CASE from mat_Tv' if per_case else 'single value from run_cfg'}")


_MELT_WINDOW_NOTED = set()


def _melt_window_notice(per_case: bool) -> None:
    """Say once which melt window `fl` was built with — the metric depends on it."""
    if per_case in _MELT_WINDOW_NOTED:
        return
    _MELT_WINDOW_NOTED.add(per_case)
    if per_case:
        print("[fl] melt window: PER-CASE from mat_Ts/mat_Tl channels")
    else:
        print("[fl] melt window: run_cfg.material_params (no mat_Ts/mat_Tl channels "
              "requested; correct only if every case shares this material)")


_MATERIAL_KEY_LIST_ATTRS = (
    "str_fno_input_keys", "str_fno_output_keys",
    "str_energy_input_keys", "str_energy_output_keys",
    "str_vof_input_keys", "str_vof_output_keys",
    "str_diff_input_keys", "str_diff_output_keys",
)


def _declared_material_channels(run_cfg) -> set[str] | None:
    """Material channels the config actually declares, or None if undiscoverable.

    "Present in var" does NOT mean "the model uses it". The lazy arraydir stores
    serve the manifest six on every KEYLESS read (lazy_runtime: `wanted =
    material_requested or MATERIAL_CHANNEL_NAMES`), and every production read is
    keyless -- so a single-material-era config that never asked for a material
    channel still gets six of them in `var`, and scale-checking everything served
    refused it (evals 75405/75410, m57990/m58734).

    None means the key lists are unavailable; the caller then checks everything,
    so the 68893 protection cannot be lost by an unexpected run_cfg shape.
    """
    keys = getattr(run_cfg, "keys", None)
    if keys is None:
        return None
    declared: set[str] = set()
    found = False
    for attr in _MATERIAL_KEY_LIST_ATTRS:
        value = getattr(keys, attr, None)
        if value is None:
            continue
        found = True
        declared.update(str(k) for k in value)
    if not found:
        return None
    return declared & set(ALL_MATERIAL_CHANNEL_NAMES)


def _preprocess_var(
    var: dict[str, np.ndarray],
    run_cfg: RunConfig,
    h_norm_mode: str,
    set_runtime_params: bool,
    preserve_time_axis: bool = False,
    h_norm_denom: float | None = None,
) -> dict[str, np.ndarray]:
    """Apply backend harmonization, masking, normalization, and derived channels."""

    time_as_batch = bool(getattr(run_cfg, "time_as_batch", False))
    include_time = bool(getattr(run_cfg, "include_time", False))
    fno_input_keys = []
    if hasattr(run_cfg, "keys") and getattr(run_cfg.keys, "str_fno_input_keys", None) is not None:
        fno_input_keys = list(run_cfg.keys.str_fno_input_keys)
    require_time_channel = include_time and (time_as_batch or "t_hat" in fno_input_keys)
    if require_time_channel and "time_s" not in var:
        raise ValueError("Time-conditioned preprocessing requires 'time_s' in loaded tensors, but it is missing.")
    if require_time_channel and np.isnan(var["time_s"]).any():
        raise ValueError("Found NaN values in 'time_s' while temporal conditioning is enabled.")

    if run_cfg.software == "flow3D":
        # Flow3D exports use verbose field names; map them to pipeline-standard keys.
        _rename_field(var, "Temperature", "T")
        _rename_field(var, "Velocity", "U")
        _rename_field(var, "Fraction Of Fluid", "alpha.metal")
        _rename_field(var, "Pressure", "p")

    if set_runtime_params:
        _define_material_properties(var, run_cfg)

    if run_cfg.sharp_alpha:
        var["alpha"] = make_interface_sharp(var["alpha.metal"])
    else:
        var["alpha"] = var["alpha.metal"]

    # PER-CASE melt window when the material channels are present. `fl` IS the melt
    # region, so a fixed Ti64 pair mislabels every material with a different
    # liquidus -- and ck300 spans 868-1898 K, with Ti64 at the very top. Falls back
    # to run_cfg so single-material runs are bit-identical to before.
    _Ts, _Tl = var.get("mat_Ts"), var.get("mat_Tl")
    if _Ts is None or _Tl is None:
        _Ts, _Tl = run_cfg.material_params["T_s"], run_cfg.material_params["T_l"]
        _melt_window_notice(False)
    else:
        _melt_window_notice(True)
    var["fl"] = smooth_liquid_fraction_numpy(var["T"], _Ts, _Tl)

    mask_channel_axis = 2 if preserve_time_axis else 1
    if run_cfg.alpha_masking:
        for key in list(var.keys()):
            if key == "T":
                _print_alpha_masking_notice_once(key)
                var[key] = fill_where_mask_smooth_np(
                    var[key],
                    var["alpha"],
                    threshold=0.5,
                    d=_fill_temperature(var, run_cfg),
                    channel_axis=mask_channel_axis,
                    inplace=False,
                )
            elif key not in ("x", "y", "z", "P", "V_scan", "Deposition", "time_s", "t_hat",
                             *PATH_CHANNEL_NAMES, *ALL_MATERIAL_CHANNEL_NAMES):
                var[key] = fill_where_mask_smooth_np(
                    var[key], var["alpha"], threshold=0.5, d=0.0, channel_axis=mask_channel_axis, inplace=False
                )

    _require_physical_laser_parameters(var)
    # H* ~ A P / (rho (cp dT + L_f) sqrt(pi D v r0^3)) -- the same closed form
    # `_h_norm_local` implements and the batch generator's h_norm_ti64 uses. The r0^3 term
    # was dropped while every case shared a radius; ti64_preheat60_fixed59 has 44 distinct
    # radii, where its absence made two cases differing only in beam radius produce the
    # SAME H_norm. Unconditional because H_norm is divided by its own max over the split
    # (BUG-04's shared denominator), so a CONSTANT radius cancels exactly and every
    # constant-radius arm -- all of spkh123 -- is bit-identical to before.
    # BUG-04: the H_norm denom must be identical at train and val/eval, otherwise
    # the same physical (P, V_scan) normalizes to a different network input across
    # splits. The authoritative train pass (set_runtime_params=True) fits the
    # data-driven denom once and stashes it on the runtime params; the later
    # val/eval pass (set_runtime_params=False) reuses it instead of recomputing a
    # per-split max. ref_scale and an explicit h_norm_denom already share a denom.
    laser_params = getattr(run_cfg, "laser_params", None)
    stashed = laser_params.get("h_norm_denom") if isinstance(laser_params, dict) else None
    # The r0^3 term cancels against a fitted max only when BOTH sides are computed in the
    # SAME pass. Across the stash boundary it does not: a model trained before the term
    # existed stashed a radius-BLIND denominator, and dividing a radius-CARRYING numerator
    # by it leaves 1/sqrt(r0^3) -- 2.83e6 at r0 = 50 um -- silently in the network input.
    # Measured 2026-08-21 re-evaluating model 57990: in_H_norm 1.28e6, T MAE 9.6e7 K.
    # So the stash records the convention it was fitted under, and a stash WITHOUT the
    # marker is legacy (radius-blind) by definition -- the only reading that is safe for
    # every model trained before this key existed. Pinned by
    # tests/test_h_norm_stash_convention.py.
    stash_radius_aware = (
        bool(laser_params.get("h_norm_denom_radius_aware"))
        if isinstance(laser_params, dict)
        else False
    )
    reusing_legacy_stash = (
        not set_runtime_params and stashed is not None and h_norm_denom is None
        and h_norm_mode != "ref_scale" and not stash_radius_aware
    )
    # OPT-IN, default OFF (2026-08-21). King's normalized enthalpy is P/sqrt(V*r0^3), and
    # carrying r0 here is right for a dataset whose beam radius VARIES. Making it
    # unconditional broke three things at once, none of them loudly:
    #   1. every model trained before the term existed, because BUG-04 stashes the
    #      denominator at train time and reuses it at eval -- so a radius-carrying
    #      numerator met a radius-blind stashed denominator and left 1/sqrt(r0^3)
    #      (2.83e6 at r0 = 50 um) in the network input;
    #   2. `lazy_runtime.compute_h_norm_denom`, whose radius read goes through
    #      `store.scalar_value` -- but beam_radius is a _CONSTANT_PATH_CHANNELS member,
    #      SYNTHESISED from the case manifest inside read_case and never stored as an
    #      array. get_array raises, a bare `except Exception` swallows it, and the
    #      denominator silently goes radius-blind while the numerator does not;
    #   3. the explicit `custom.h_norm_denom` override, which was chosen under the
    #      radius-blind convention by every run that set it.
    # The information is not lost: `beam_radius` is its own input channel (48c5c33), so
    # the network can form any function of (P, V, r0) regardless. On a constant-radius
    # dataset -- all of spkh123 -- the term is a pure no-op that only creates ways to
    # disagree. Switch it on deliberately, per dataset, once the denominator can see the
    # radius on every store type. Pinned by tests/test_h_norm_stash_convention.py.
    include_radius = bool(getattr(run_cfg, "h_norm_include_radius", False))
    if include_radius:
        # The King term cancels ONLY when the same convention builds both sides. Three
        # denominators cannot carry it, and a missing channel silently drops it from the
        # numerator instead. Each disagreement is worth 1/sqrt(r0^3) ~ 2.8e6 and returns a
        # plausible number, never an error -- so refuse instead of guessing. Audit
        # 2026-08-21; pinned by tests/test_h_norm_radius_consistency.py.
        if h_norm_mode == "ref_scale":
            raise ValueError(
                "h_norm_include_radius=True is incompatible with h_norm_mode='ref_scale': "
                "the denominator is the constant ref_scales['H'], which carries no r0, so "
                "the r0^3 term would survive into H_norm. Use the fitted (max) denominator."
            )
        # Key on the CONFIG field, not the function argument. `_prepare_runtime_reference`
        # and the lazy sample builders pass the pipeline's OWN fitted denominator through
        # this same parameter, and that one IS radius-aware (it comes from
        # _effective_h_norm_denom with include_radius set). Only a user-supplied
        # custom.h_norm_denom carries the radius-blind convention. Guarding on the
        # parameter refused the legitimate internal path and killed run 61121, the very
        # training this flag exists to enable. A denominator an EVAL took from the model's
        # training record (or refitted on its training split) was fitted radius-aware and
        # is marked so (`h_norm_record`); only a user-supplied value is refused.
        if (getattr(run_cfg, "h_norm_denom", None) is not None
                and not bool(getattr(run_cfg, "h_norm_denom_radius_aware", False))):
            raise ValueError(
                "h_norm_include_radius=True is incompatible with an explicit "
                "custom.h_norm_denom: that value was chosen under the radius-BLIND "
                "convention, so dividing a radius-carrying numerator by it leaves "
                "1/sqrt(r0^3) in the network input. Drop one of the two options."
            )
        if "beam_radius" not in var:
            raise ValueError(
                "h_norm_include_radius=True but this sample carries no 'beam_radius'. "
                "The numerator would silently go radius-blind while the denominator does "
                "not. Add beam_radius to custom.FNO_input_keys, or turn the flag off."
            )
    # beam_radius is still raw metres here; the ref_scales["L"] division happens below.
    beam_radius = var.get("beam_radius") if (include_radius and not reusing_legacy_stash) else None
    # Material half of King's number, opt-in. MUST match `compute_h_norm_denom`'s
    # `include_material`: the two are divided by each other, so a mismatch is silent and
    # catastrophic (2026-08-20 fed the network 2.5e6 instead of 0.886 exactly this way).
    _mat_factor = None
    if getattr(run_cfg, "h_norm_include_material", False):
        _need = ("mat_rho", "mat_cp", "mat_k", "mat_latent", "mat_Tl", "mat_A")
        _miss = [k for k in _need if k not in var]
        if _miss:
            raise ValueError(
                f"h_norm_include_material=True needs {_need} served with the sample; missing {_miss}. "
                "Refusing rather than silently falling back to a material-blind numerator, "
                "which the denominator would not match.")
        _mat_factor = king_material_factor(
            rho=var["mat_rho"], cp=var["mat_cp"], k=var["mat_k"],
            latent=var["mat_latent"], T_l=var["mat_Tl"], A=var["mat_A"])
    var["H_norm"] = h_norm_raw(var["P"], var["V_scan"], beam_radius, material_factor=_mat_factor)
    if h_norm_mode == "ref_scale":
        denom = float(run_cfg.ref_scales.get("H", 5.6))
    elif h_norm_denom is not None:
        denom = float(h_norm_denom)
    elif not set_runtime_params and stashed is not None:
        denom = float(stashed)
    else:
        # Standard train pass: fit the denom on this (train) split's data.
        denom = float(np.max(var["H_norm"]))
    if np.isclose(denom, 0.0):
        denom = 1.0
    if set_runtime_params and h_norm_mode != "ref_scale" and h_norm_denom is None and isinstance(laser_params, dict):
        laser_params["h_norm_denom"] = denom
        # Record the convention so the val/eval pass rebuilds the numerator to match.
        laser_params["h_norm_denom_radius_aware"] = beam_radius is not None
        # Same reason: a stash fitted material-AWARE must not be reused by a
        # material-blind numerator, or the ratio carries the material factor.
        laser_params["h_norm_denom_material_aware"] = _mat_factor is not None
    var["H_norm"] = var["H_norm"] / denom

    ref = run_cfg.ref_scales
    v_scan_scale = float(ref["V"])
    v_melt_scale = float(ref.get("V_melt", ref["V"]))
    var["x"] = var["x"] / ref["L"]
    var["y"] = var["y"] / ref["L"]
    var["z"] = var["z"] / ref["L"]
    # Beam-centre channels are lab-frame positions (metres); scale like x/y so the
    # network sees them on the same non-dimensional footing as the coordinates.
    for _bk in ("beam_x", "beam_y"):
        if _bk in var:
            var[_bk] = var[_bk] / ref["L"]
    # beam_dist (footprint, peak 1) and path_dose (fraction of dwell, <=1) are
    # already O(1); leave them unscaled.
    # beam_radius is a LENGTH in metres (35-70 um here): scale like x/y/z or it arrives
    # as ~3.5e-5 and is numerically invisible beside O(1) channels.
    if "beam_radius" in var:
        var["beam_radius"] = var["beam_radius"] / ref["L"]
    # Kelvin-valued conditioning channels MUST be scaled like T: preheat_T0 spans
    # 293-773 and soak_field 273-773, i.e. 25-140x every other channel, and ~273 of
    # that is a constant offset carrying no information. Unscaled, one channel
    # dominates the lifting layer's pre-activation by two orders of magnitude.
    for _tk in ("preheat_T0", "soak_field"):
        if _tk in var:
            var[_tk] = var[_tk] / ref["T"]
    # Material channels are raw SI properties spanning orders of magnitude; divide
    # each by its ref scale so the network sees O(1) values (like x/V_scan/P/T).
    # A MISSING scale is refused, not defaulted to 1.0. Silently dividing by 1.0 is
    # how fno_ck300_lab_32k_mat12 (68893) trained 32k steps and exited 0 while being
    # physically useless: config_flow3D_FNO_pvlhs inherits ref_scales from
    # config_NS_energy_PINO_flow3D, which defines no mat_* at all, so mat_latent
    # (~3e5) and mat_rho (~4.4e3) entered the lifting layer beside O(1) channels and
    # the model never reached melt (peak 1211 K vs an 1898 K threshold). Same reason
    # `_overlay_value` raises on a miss: a silent constant trains happily and answers
    # a different question.
    _declared = _declared_material_channels(run_cfg)
    _missing = [
        k for k in ALL_MATERIAL_CHANNEL_NAMES
        if k in var and k not in ref and (_declared is None or k in _declared)
    ]
    if _missing:
        raise ValueError(
            f"material channels {_missing} have no custom.ref_scales entry, so they "
            f"would enter the network as raw SI values. Add a scale per channel "
            f"(one `custom.ref_scales.<channel>` entry per requested mat_* channel)."
        )
    for _mk in ALL_MATERIAL_CHANNEL_NAMES:
        # `_mk in ref` is not redundant with the guard above: a channel that is
        # served but undeclared is deliberately left unguarded, and it is also
        # left raw rather than KeyError-ing here. Input assembly indexes var by
        # the declared keys, so an unscaled undeclared channel never reaches the
        # network.
        if _mk in var and _mk in ref:
            var[_mk] = var[_mk] / float(ref[_mk])
    if "time_s" in var:
        var["t_hat"] = var["time_s"] * (ref["V"] / ref["L"])
    var["V_scan"] = var["V_scan"] / v_scan_scale
    var["P"] = var["P"] / ref["P"]
    var["T"] = var["T"] / ref["T"]

    # Pressure and Velocity are OPTIONAL. A config whose input/output keys use
    # neither can leave them out of `custom.vtk_fields`, which is the single
    # biggest loader saving on the lazy path: they are 58% of the per-case zlib
    # inflate (Velocity 5.96 s + Pressure 1.76 s of 14.69 s, measured 2026-07-28)
    # and are otherwise built, masked, normalised and then discarded. Guarded
    # rather than removed because this function is shared with the PINO/energy
    # configs, which do consume them.
    if "p" in var:
        var["p"] = var["p"] / (ref["Rho"] * ref["V"] ** 2)
    if "U" in var:
        var["U"] = var["U"] / v_melt_scale

        if preserve_time_axis:
            var["u"] = var["U"][:, :, 0:1, ...]
            var["v"] = var["U"][:, :, 1:2, ...]
            var["w"] = var["U"][:, :, 2:3, ...]
        else:
            var["u"] = var["U"][:, 0:1, ...]
            var["v"] = var["U"][:, 1:2, ...]
            var["w"] = var["U"][:, 2:3, ...]

        # Mirror inference-side velocity masking: apply liquid-fraction gating to u/v/w.
        if run_cfg.alpha_masking:
            for vel_key in ("u", "v", "w"):
                var[vel_key] = fill_where_mask_smooth_np(
                    var[vel_key],
                    var["fl"],
                    threshold=0.5,
                    d=0.0,
                    channel_axis=mask_channel_axis,
                    k=20,
                    inplace=False,
                )

    if run_cfg.pde_loss:
        var["energy"] = np.zeros_like(var["T"])
    if run_cfg.vof_pde_loss:
        var["VOF"] = np.zeros_like(var["T"])

    return var


def adopt_beam_radius(laser_params: dict, input_dir) -> None:
    """Stash the dataset's beam radius on the runtime laser params, for the beam box.

    The beam box is a determined geometric region -- radius and centre fix it -- but the
    only route to the radius used to be the `beam_dist` INPUT CHANNEL, which the steady
    arms do not carry. The metric was therefore skipped silently on every steady run and
    the manuscript's architecture table reached a reviewer with no beam-box column.

    The radius is a property of the DATA, not of the network's input list, so it is read
    here alongside the mask fill (`adopt_material_t_boil`) rather than being smuggled in
    as a channel the model would then also be fed.

    Precedence mirrors the fill: an explicit `laser_params["beam_radius_m"]` wins; a
    unanimous manifest value is adopted; disagreeing manifests (a varying-radius set such
    as ti64_preheat60_fixed59, 44 distinct radii) are left ALONE, because one radius per
    dataset is wrong there and the per-case `beam_radius` channel is the right source.
    """
    if not isinstance(laser_params, dict) or laser_params.get("beam_radius_m") is not None:
        return
    root = Path(str(input_dir)) if input_dir else None
    if root is None or not root.is_dir():
        return
    import json as _json

    manifests = sorted(root.glob("*.manifest.json")) or sorted(root.glob("members/*/manifest.json"))
    values = set()
    for mp_ in manifests:
        try:
            payload = _json.loads(mp_.read_text())
        except Exception:  # noqa: BLE001 - one unreadable manifest must not kill a load
            continue
        inner = (payload.get("simulation_metadata") or {}).get("flow3d_case_manifest") or payload
        value = (((inner.get("input") or {}).get("laser") or {}).get("beam_radius_m"))
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            values.add(round(value, 12))
    if len(values) == 1:
        laser_params["beam_radius_m"] = float(next(iter(values)))


def adopt_material_t_boil(material_params: dict, input_dir) -> None:
    """Let the mask fill follow the DATA's material file (its vaporization temperature).

    The hardcoded T_BOIL_K default is the Mills Ti64 value frozen into code; the
    calibrated Ti64 material declares 3520 K, so on calibrated data the fill sat 400 K
    below the material's own boiling point and capped masked-metal T below real peaks
    (the measured frozen tail_deficit_max_T). Reading the manifest generalizes the fill
    to any single-material dataset -- Mills sets resolve to their own file's value,
    which equals the legacy constant, so old-physics behaviour is unchanged.

    Precedence: an explicit `custom.T_boil_k` (already present as material_params
    ["T_boil"]) wins; a unanimous manifest value is adopted; disagreeing manifests (a
    multi-material set) keep the legacy default LOUDLY, because one fill per dataset is
    wrong there either way and silently picking a side would change that campaign's task.
    """
    if not isinstance(material_params, dict) or "T_boil" in material_params:
        return
    root = Path(str(input_dir)) if input_dir else None
    if root is None or not root.is_dir():
        return
    import json as _json

    manifests = sorted(root.glob("*.manifest.json")) or sorted(root.glob("members/*/manifest.json"))
    values = set()
    for mp_ in manifests:
        try:
            v = t_boil_from_manifest(_json.loads(mp_.read_text()))
        except Exception:  # noqa: BLE001 - a single unreadable manifest must not kill a load
            continue
        if v is not None:
            values.add(round(float(v), 2))
    if not values:
        return
    if len(values) > 1:
        from ...custom_utils.utils import boiling_temperature as _bt

        print(
            f"[t_boil] multi-material dataset (vaporization temperatures {sorted(values)} K): "
            f"keeping the legacy {_bt({})} K mask fill. One fill per dataset is wrong here; "
            "set custom.T_boil_k to choose explicitly, or custom.per_case_t_boil=true to fill "
            "each case from its own mat_Tv.",
            flush=True,
        )
        return
    value = values.pop()
    material_params["T_boil"] = float(value)
    print(f"[t_boil] mask fill adopted from the dataset's material file: {value} K "
          f"(override with custom.T_boil_k)", flush=True)


def adopt_metric_melt_window(run_cfg, input_dir) -> None:
    """Record the DATA's own melt window on `run_cfg`, for metrics only.

    `melt_threshold_K`, `thermal_headroom_K` and `can_reach_melt` threshold at the
    midpoint `(T_s+T_l)/2` taken from `material_params` -- the CONFIG's material. Evaluate a
    Ti64-trained model on IN718 and all three are measured against 1898.0 K, Ti64's midpoint,
    where IN718's is 1565.65: 332 K out, so the metrics answer a question about a material
    the data does not contain (self-improve session, 2026-09-04, eval on ds 87432).

    Deliberately written to `run_cfg.metric_melt_window` and NOT into `material_params`:
    that dict also drives `fl`, which is a TARGET, and moving a target silently would change
    the task the campaign is scored on. Fixing a measurement does not need that, and must not
    do it. `custom.melt_window_from_manifest=false` opts out; an explicit
    `custom.melt_window_k: [T_s, T_l]` wins outright.

    A multi-material set gets no dataset-level value -- there is no single right one -- and
    the per-case `mat_Ts`/`mat_Tl` channels take over where they are served.
    """
    if getattr(run_cfg, "metric_melt_window", None) is not None:
        return
    from ...custom_utils.utils import custom_flag

    explicit = custom_flag(run_cfg, "melt_window_k", None)
    if explicit is not None:
        try:
            t_s, t_l = (float(v) for v in explicit)
            run_cfg.metric_melt_window = (t_s, t_l)
            print(f"[melt-window] metrics threshold from custom.melt_window_k: "
                  f"{t_s}-{t_l} K", flush=True)
            return
        except (TypeError, ValueError):
            pass
    if not bool(custom_flag(run_cfg, "melt_window_from_manifest", True)):
        return
    root = Path(str(input_dir)) if input_dir else None
    if root is None or not root.is_dir():
        return
    import json as _json

    from ...custom_utils.utils import melt_window_from_manifest

    manifests = sorted(root.glob("*.manifest.json")) or sorted(root.glob("members/*/manifest.json"))
    windows = set()
    for mp_ in manifests:
        try:
            w = melt_window_from_manifest(_json.loads(mp_.read_text()))
        except Exception:  # noqa: BLE001 - one unreadable manifest must not kill a load
            continue
        if w is not None:
            windows.add((round(w[0], 2), round(w[1], 2)))
    if len(windows) != 1:
        if len(windows) > 1:
            print(
                f"[melt-window] multi-material dataset ({len(windows)} distinct solidus/liquidus "
                "pairs): no dataset-level metric threshold. Per-case mat_Ts/mat_Tl are used "
                "where served; otherwise the config's window stands and the melt metrics "
                "describe the CONFIG's material, not each case's.",
                flush=True,
            )
        return
    t_s, t_l = windows.pop()
    cfg_s = float((getattr(run_cfg, "material_params", None) or {}).get("T_s", 0.0))
    cfg_l = float((getattr(run_cfg, "material_params", None) or {}).get("T_l", 0.0))
    run_cfg.metric_melt_window = (t_s, t_l)
    if abs(0.5 * (t_s + t_l) - 0.5 * (cfg_s + cfg_l)) > 1.0:
        print(
            f"[melt-window] the DATA's melt window is {t_s}-{t_l} K (midpoint "
            f"{0.5 * (t_s + t_l):.2f} K); the config declares {cfg_s}-{cfg_l} K (midpoint "
            f"{0.5 * (cfg_s + cfg_l):.2f} K). Melt metrics now use the data's. `fl` still "
            "uses the config window -- it is a target, unchanged here.",
            flush=True,
        )


def load_and_preprocess_data(
    run_cfg: RunConfig,
    input_dir: Optional[str] = None,
    h_norm_mode: str = "max",
    set_runtime_params: bool = True,
) -> dict[str, np.ndarray] | RuntimeStore:
    """Load VTK directory data and return normalized/augmented field tensors."""

    resolved_input_dir = input_dir or run_cfg.data_directory_path
    adopt_material_t_boil(getattr(run_cfg, "material_params", None), resolved_input_dir)
    adopt_metric_melt_window(run_cfg, resolved_input_dir)
    adopt_beam_radius(getattr(run_cfg, "laser_params", None), resolved_input_dir)
    if is_arraydir_runtime(resolved_input_dir):
        # An aiida-pino collection of npz_file-mode members carries its cases as flat
        # `<case>.npz` at the root: read those, not the v1 member manifests the
        # arraydir store cannot parse (and which carry no laser-path channels).
        if is_npz_file_collection(resolved_input_dir):
            npz_store = maybe_build_npz_store(resolved_input_dir, run_cfg, force=True)
            if npz_store is not None:
                return npz_store
        return load_runtime_store(resolved_input_dir)
    npz_store = maybe_build_npz_store(resolved_input_dir, run_cfg)
    if npz_store is not None:
        return npz_store

    time_as_batch = bool(getattr(run_cfg, "time_as_batch", False))
    time_as_dim = bool(getattr(run_cfg, "time_as_dim", False))
    use_2d = bool(getattr(run_cfg, "use_2d", False))
    moving_ref_frame = bool(getattr(run_cfg, "moving_ref_frame", True))
    moving_frame_steps = int(getattr(run_cfg, "moving_frame_steps", run_cfg.time_avg_window))
    effective_moving_frame_steps = moving_frame_steps if moving_ref_frame else 1
    if time_as_batch and not run_cfg.include_time:
        raise ValueError("time_as_batch=True requires include_time=True.")
    if time_as_dim and not run_cfg.include_time:
        raise ValueError("time_as_dim=True requires include_time=True.")

    _warn_if_time_avg_window_is_inert(run_cfg)
    if time_as_dim:
        time_axis = True
        # Keep moving-frame x-shift/crop active, but disable temporal averaging.
        avg_window_size = 1
    elif time_as_batch:
        time_axis = True
        # time_as_batch keeps the temporal axis; disable temporal averaging.
        avg_window_size = 1
    else:
        time_axis = run_cfg.include_time
        avg_window_size = run_cfg.time_avg_window

    input_keys = []
    if hasattr(run_cfg, "keys") and getattr(run_cfg.keys, "str_fno_input_keys", None) is not None:
        input_keys = list(run_cfg.keys.str_fno_input_keys)
    path_channels = [k for k in input_keys if k in PATH_CHANNEL_NAMES]
    material_channels = served_material_channels(run_cfg, input_keys)
    from ...custom_utils.utils import custom_flag as _custom_flag

    if bool(_custom_flag(run_cfg, "per_case_t_boil", False)) and "mat_Tv" not in material_channels:
        material_channels.append("mat_Tv")   # auxiliary: read for the fill, not a model input

    if is_consolidated_npz(resolved_input_dir):
        # Consolidated NPZ already stores a uniform grid; load it natively (no
        # crop/resample) so its own rectangular geometry flows through.
        var = load_consolidated_npz(
            resolved_input_dir,
            requested_fields=run_cfg.vtk_fields,
            super_res_factor=run_cfg.super_res_factor,
            time_axis=time_axis,
            avg_window_size=avg_window_size,
            moving_frame_steps=effective_moving_frame_steps,
            moving_ref_frame=moving_ref_frame,
            path_channels=path_channels,
            material_channels=material_channels,
            # Same two resolvers the lazy store uses, so one config cannot mean two
            # different sets of dose channels depending on which loader ran.
            power_weighted=power_weighted_dose_enabled(run_cfg),
            power_weighted_conduction=power_weighted_conduction_enabled(run_cfg),
            conduction_sigma_factor=conduction_source_sigma_factor(run_cfg),
        )
    else:
        # BUG-01/BUG-02: direct VTK directory loading is deprecated. The legacy
        # parser silently corrupts subdir-layout params (P=V=0) and decimal-time
        # trajectories (T collapses to 1). Consolidated NPZ (or an arraydir
        # runtime / aiida dataset) is the supported input path.
        if os.environ.get("PINO_ALLOW_VTK", "0") != "1":
            # A path that is gone is not a format problem: say which it is, or
            # the message below sends you converting datasets that were never
            # at fault. Only replaces the message we were about to raise, so
            # the PINO_ALLOW_VTK escape hatch keeps working unchanged.
            vanished = _describe_missing_dataset_dir(resolved_input_dir)
            if vanished is not None:
                raise ValueError(vanished)
            raise ValueError(
                f"Direct VTK directory loading is deprecated: '{resolved_input_dir}' is "
                "neither a consolidated-NPZ dataset nor an arraydir runtime. Convert the "
                "cases to consolidated NPZ (or ingest via aiida-pino raw_*_to_dataset) and "
                "point the config at that dataset. The legacy VTK parser has known "
                "silent-corruption modes (per-subdirectory cases parse as P = V_scan = 0, "
                "and decimal time suffixes collapse a trajectory to single frames); set "
                "PINO_ALLOW_VTK=1 to load VTK anyway."
            )
        data_source = VTKOpenFOAMDataSource()
        var = data_source.parse_directory(
            run_cfg.grid_params,
            input_dir=input_dir or run_cfg.data_directory_path,
            requested_fields=run_cfg.vtk_fields,
            super_res_factor=run_cfg.super_res_factor,
            debug_params=False,
            time_axis=time_axis,
            avg_window_size=avg_window_size,
            moving_frame_steps=effective_moving_frame_steps,
            moving_ref_frame=moving_ref_frame,
            flip_axes=run_cfg.software == "openFoam",
            allow_grid_interpolation=getattr(run_cfg, "allow_grid_interpolation", False),
        )
    if use_2d:
        var = _slice_middle_y(var)
    preserve_time_axis = bool(getattr(run_cfg, "autoregressive_enabled", False)) and time_as_batch
    if time_as_dim:
        var = _time_axis_to_dim(var)
    elif time_as_batch and not preserve_time_axis:
        var = _flatten_time_axis(var)
    return _preprocess_var(
        var,
        run_cfg,
        h_norm_mode=h_norm_mode,
        set_runtime_params=set_runtime_params,
        preserve_time_axis=preserve_time_axis,
        h_norm_denom=getattr(run_cfg, "h_norm_denom", None),
    )


def print_dict_shapes(var: dict[str, np.ndarray] | RuntimeStore) -> None:
    """Print tensor shapes for quick dataset sanity checks."""

    if isinstance(var, _RUNTIME_STORE_TYPES):
        print(f"Runtime store: {var.__class__.__name__}")
        for key in var.list_array_names():
            print(f"{key}: {var.array_shape(key)}")
        return
    for key, value in var.items():
        print(f"{key}: {value.shape}")


def sample_count(var: dict[str, np.ndarray] | RuntimeStore) -> int:
    """Return sample count for eager dictionaries or lazy runtime stores."""
    if isinstance(var, _RUNTIME_STORE_TYPES):
        return var.sample_count()
    return int(next(iter(var.values())).shape[0])


def _effective_h_norm_denom(run_cfg, store, case_indices) -> float:
    """Explicit ``custom.h_norm_denom`` wins; else fit the max over the train split.

    An explicit denom keeps H_norm consistent across runs whose train sets
    differ — growing active-learning datasets, candidate-grid evals — where the
    per-split fitted max would rescale the same physical (P, V) between runs.
    """
    explicit = getattr(run_cfg, "h_norm_denom", None)
    if explicit:
        return float(explicit)
    return compute_h_norm_denom(
        store,
        case_indices,
        time_as_batch=run_cfg.time_as_batch,
        include_radius=bool(getattr(run_cfg, "h_norm_include_radius", False)),
        include_material=bool(getattr(run_cfg, "h_norm_include_material", False)),
    )


def _prepare_runtime_reference(
    store: RuntimeStore,
    run_cfg: RunConfig,
    case_indices: Sequence[int],
    *,
    h_norm_denom: float,
) -> None:
    """Initialize runtime material/laser params from one representative sample."""
    if not case_indices:
        return
    case_idx = int(case_indices[0])
    time_idx = 0 if run_cfg.time_as_batch and store.has_time_axis() else None
    reference = store.read_case(case_idx, time_idx=time_idx)
    if run_cfg.use_2d and not run_cfg.time_as_dim:
        reference = _slice_middle_y(reference)
    if run_cfg.time_as_dim:
        reference = _time_axis_to_dim(_slice_middle_y(reference))
    _preprocess_var(
        reference,
        run_cfg,
        h_norm_mode="max",
        set_runtime_params=True,
        preserve_time_axis=False,
        h_norm_denom=h_norm_denom,
    )


def _build_standard_entries(
    store: RuntimeStore,
    case_indices: Sequence[int],
    run_cfg: RunConfig,
    frames_per_case: int = 0,
) -> list[tuple[int, int | None]]:
    """Expand case indices into logical dataset entries.

    `frames_per_case > 0` keeps that many evenly spaced frames per case (always
    including the first and last) instead of every stored frame. Used only for
    the in-training validation split -- see _effective_val_frames_per_case.
    """
    entries: list[tuple[int, int | None]] = []
    if run_cfg.time_as_batch and not run_cfg.time_as_dim:
        for case_idx in case_indices:
            n_frames = store.time_steps(case_idx=int(case_idx))
            if 0 < frames_per_case < n_frames:
                times = np.unique(np.linspace(0, n_frames - 1, frames_per_case).round().astype(int))
            else:
                times = range(n_frames)
            for time_idx in times:
                entries.append((int(case_idx), int(time_idx)))
        return entries
    return [(int(case_idx), None) for case_idx in case_indices]


# --- Temporal-consistency (causality lever 2) ride-along config (env-gated) ------ #
_TC_CANDIDATE_KEYS = ("t_hat", "beam_x", "beam_y", "beam_dist", "path_dose", "path_dose_recent")


def tc_weight() -> float:
    """Temporal-consistency loss weight (env PINO_TC_WEIGHT); 0 = off."""
    try:
        return max(0.0, float(os.environ.get("PINO_TC_WEIGHT", "0")))
    except ValueError:
        return 0.0


def _tc_delta() -> int:
    try:
        return max(1, int(os.environ.get("PINO_TC_DELTA", "1")))
    except ValueError:
        return 1


def tc_advance_keys(str_input_keys) -> list:
    """Time-varying inputs advanced to t+delta for the second forward pass."""
    return [k for k in _TC_CANDIDATE_KEYS if k in str_input_keys]


def tc_ride_along_keys(str_input_keys) -> list:
    """Extra invar keys the sample must declare when temporal consistency is on."""
    return [f"{k}__adv" for k in tc_advance_keys(str_input_keys)] + ["T__adv_tgt"]


def _read_processed_frame(
    *,
    store: RuntimeStore,
    run_cfg: RunConfig,
    h_norm_denom: float,
    case_idx: int,
    time_idx: int | None,
) -> dict[str, np.ndarray]:
    """Read one frame from the store and run the standard per-frame preprocess."""

    raw = store.read_case(case_idx, time_idx=time_idx)
    if run_cfg.use_2d and not run_cfg.time_as_dim:
        raw = _slice_middle_y(raw)
    if run_cfg.time_as_dim:
        raw = _time_axis_to_dim(_slice_middle_y(raw))
    return _preprocess_var(
        raw,
        run_cfg,
        h_norm_mode="max",
        set_runtime_params=False,
        preserve_time_axis=False,
        h_norm_denom=h_norm_denom,
    )


def _build_standard_sample_builder(
    *,
    store: RuntimeStore,
    run_cfg: RunConfig,
    key_cfg: KeyConfig,
    h_norm_denom: float,
):
    """Create a per-sample builder for standard lazy training/eval."""
    from ...custom_utils.utils import custom_flag as _custom_flag

    # Eval-only: score the melt-pool metrics with each case's own melt window. Every
    # two-alloy training and eval before 2026-10-13 used run_cfg's Ti-6Al-4V pair for
    # IN718 cases too, because keyless reads never serve mat_Ts. The window channels
    # ride beside the inputs (never declared, so the network does not see them) and
    # reach both fl derivations: the target in _preprocess_var and the prediction in
    # postprocessing._apply_prediction_postprocessing. Training keeps the run_cfg pair,
    # whose fl target must match the graph's fl node.
    per_case_window = bool(_custom_flag(run_cfg, "eval_per_case_melt_window", False))

    def _with_melt_window(raw, case_idx):
        reference = next(v for k, v in raw.items() if k not in ("Velocity", "U"))
        for name in ("mat_Ts", "mat_Tl"):
            raw[name] = np.full(np.shape(reference), store.scalar_value(name, int(case_idx)),
                                dtype=np.float32)
        return raw

    def _builder(case_idx: int, time_idx: int | None):
        raw = store.read_case(case_idx, time_idx=time_idx)
        if per_case_window:
            raw = _with_melt_window(raw, case_idx)
        if run_cfg.use_2d and not run_cfg.time_as_dim:
            raw = _slice_middle_y(raw)
        if run_cfg.time_as_dim:
            raw = _time_axis_to_dim(_slice_middle_y(raw))
        processed = _preprocess_var(
            raw,
            run_cfg,
            h_norm_mode="max",
            set_runtime_params=False,
            preserve_time_axis=False,
            h_norm_denom=h_norm_denom,
        )
        invar = {key: processed[key] for key in key_cfg.str_fno_input_keys}
        if per_case_window:
            invar.update({k: processed[k] for k in ("mat_Ts", "mat_Tl")})
        outvar = {key: processed[key] for key in key_cfg.str_fno_output_keys if key != "T_dot"}
        outvar["fl"] = processed["fl"]
        if "T_dot" in key_cfg.str_fno_output_keys and time_idx is not None:
            # Causality lever 3: auxiliary dT/dt target from the GT finite difference
            # to the neighbour frame (cache hit — the whole case is already loaded).
            # Backward difference at the last frame. Both T and t_hat are normalized
            # by _preprocess_var, so the ratio is in scaled units, consistent per sample.
            nt = store.time_steps(case_idx=case_idx)
            j = time_idx + 1 if time_idx + 1 < nt else time_idx - 1
            nb_raw = store.read_case(case_idx, time_idx=j)
            if run_cfg.use_2d and not run_cfg.time_as_dim:
                nb_raw = _slice_middle_y(nb_raw)
            if run_cfg.time_as_dim:
                nb_raw = _time_axis_to_dim(_slice_middle_y(nb_raw))
            nb = _preprocess_var(
                nb_raw, run_cfg, h_norm_mode="max", set_runtime_params=False,
                preserve_time_axis=False, h_norm_denom=h_norm_denom,
            )
            dt = nb["t_hat"] - processed["t_hat"]  # nonzero (distinct frame times)
            outvar["T_dot"] = (nb["T"] - processed["T"]) / dt
        if tc_weight() > 0.0 and time_idx is not None:
            # Causality lever 2: ride-along time-advanced inputs at t+delta (a cache
            # hit) + the GT T(t+delta), consumed by TemporalConsistencyGridConstraint
            # for a second, non-autoregressive forward pass.
            nt = store.time_steps(case_idx=case_idx)
            j = time_idx + _tc_delta()
            j = j if j < nt else max(0, time_idx - _tc_delta())
            adv_raw = store.read_case(case_idx, time_idx=j)
            if run_cfg.use_2d and not run_cfg.time_as_dim:
                adv_raw = _slice_middle_y(adv_raw)
            if run_cfg.time_as_dim:
                adv_raw = _time_axis_to_dim(_slice_middle_y(adv_raw))
            adv = _preprocess_var(
                adv_raw, run_cfg, h_norm_mode="max", set_runtime_params=False,
                preserve_time_axis=False, h_norm_denom=h_norm_denom,
            )
            for k in tc_advance_keys(key_cfg.str_fno_input_keys):
                invar[f"{k}__adv"] = adv[k]
            invar["T__adv_tgt"] = adv["T"]
        return invar, outvar

    return _builder


def _random_spatial_slices(
    spatial_shape: Sequence[int],
    patch_size: Sequence[int],
    rng: np.random.Generator,
) -> tuple[slice, ...]:
    """Return random spatial slices bounded by the available runtime volume."""
    slices: list[slice] = []
    for dim, requested in zip(spatial_shape, patch_size):
        extent = min(int(requested), int(dim))
        start = 0 if dim <= extent else int(rng.integers(0, dim - extent + 1))
        slices.append(slice(start, start + extent))
    return tuple(slices)


def _build_masked_patch_builder(
    *,
    store: RuntimeStore,
    run_cfg: RunConfig,
    h_norm_denom: float,
):
    """Create a patch-sampling builder for masked next-step autoregressive training."""

    if run_cfg.use_2d:
        raise ValueError("Masked temporal patch autoregressive mode currently expects 3D runtime data.")

    def _builder(*, rng: np.random.Generator, case_indices: Sequence[int]):
        eligible_case_indices = [
            int(case_idx) for case_idx in case_indices if store.time_steps(case_idx=int(case_idx)) >= 2
        ]
        if not eligible_case_indices:
            raise ValueError("Masked autoregressive training requires at least one simulation with 2 timesteps.")
        case_idx = int(eligible_case_indices[int(rng.integers(0, len(eligible_case_indices)))])
        max_time = store.time_steps(case_idx=case_idx)
        if max_time < 2:
            raise ValueError("Masked autoregressive training requires at least 2 timesteps.")
        time_idx = int(rng.integers(0, max_time - 1))
        spatial_slices = _random_spatial_slices(store.spatial_shape(), run_cfg.autoregressive_patch_size, rng)

        current_raw = store.read_case(case_idx, time_idx=time_idx + 1, spatial_slices=spatial_slices)
        previous_raw = store.read_case(case_idx, time_idx=time_idx, spatial_slices=spatial_slices)
        current = _preprocess_var(
            current_raw,
            run_cfg,
            h_norm_mode="max",
            set_runtime_params=False,
            preserve_time_axis=False,
            h_norm_denom=h_norm_denom,
        )
        previous = _preprocess_var(
            previous_raw,
            run_cfg,
            h_norm_mode="max",
            set_runtime_params=False,
            preserve_time_axis=False,
            h_norm_denom=h_norm_denom,
        )

        invar = {key: current[key] for key in run_cfg.autoregressive_context_keys}
        for state_key in run_cfg.autoregressive_state_keys:
            prev_key = f"{state_key}{run_cfg.autoregressive_state_prev_suffix}"
            mask_key = f"{prev_key}_mask"
            masked_prev, mask = apply_random_mask(
                np.asarray(previous[state_key][0], dtype=np.float32),
                rng=rng,
                mask_mode=run_cfg.autoregressive_mask_mode,
                mask_ratio=run_cfg.autoregressive_mask_ratio,
            )
            invar[prev_key] = masked_prev[np.newaxis, ...]
            invar[mask_key] = mask[np.newaxis, ...]

        outvar = {key: current[key] for key in run_cfg.autoregressive_state_keys}
        outvar["fl"] = current["fl"]
        return invar, outvar

    return _builder


def _subsample_val_cases(indices: list[int], max_cases: int, seed: int) -> list[int]:
    """Seeded, order-preserving subset of validation case indices.

    ``max_cases <= 0`` (or >= len) keeps everything. A full-split validation
    pass on the 52-case kh2579 val set costs ~43 min cold at batch 1 and
    recurs every ``rec_validation_freq`` steps; ~5 cases track the same
    l2_relative_error signal at a fraction of the cost.
    """
    if max_cases <= 0 or max_cases >= len(indices):
        return list(indices)
    rng = np.random.default_rng([int(seed), len(indices), int(max_cases)])
    keep = rng.choice(len(indices), size=int(max_cases), replace=False)
    return [indices[int(i)] for i in sorted(keep)]


def _describe_missing_dataset_dir(path) -> str | None:
    """Explain a staged dataset dir that is absent or empty, else None.

    Separated from the format check so a *vanished* input is never reported as
    a *wrong-format* one. aiida-pino resolves its dataset symlinks at upload
    time but the trainer reads them at execution time, so a cache cleared while
    the job waits for a GPU seat leaves a dangling link -- and the VTK
    deprecation message below then blames the dataset format instead.
    """
    from pathlib import Path

    path = Path(path)
    if path.is_symlink() and not path.exists():
        return (
            f"Staged dataset path '{path}' is a dangling symlink to "
            f"'{os.readlink(path)}'. The link is created when the job is uploaded but "
            "read when it runs, so an aiida-pino runtime cache entry deleted in "
            "between (e.g. while the job waited for a GPU seat) strands it. "
            "Resubmit the job: the cache is re-exported on the next submit."
        )
    if not path.exists():
        return f"Staged dataset path '{path}' does not exist."
    if not path.is_dir():
        return f"Staged dataset path '{path}' is not a directory."
    if not any(path.iterdir()):
        return (
            f"Staged dataset directory '{path}' is empty. An interrupted or "
            "partially-cleared aiida-pino runtime cache export looks like this; "
            "resubmit the job to re-export it."
        )
    return None


def _effective_val_max_cases(run_cfg) -> int:
    """Validation cases to keep: the cap, or 0 (all) on a scoring run.

    `custom.val_max_cases` bounds the *in-training* validator, which reruns
    every `rec_validation_freq`. NS_Energy_laser_PINO_eval reuses the same
    `run_training_pipeline` -> `build_datasets`, so honouring the cap there
    would score a subset and report it as the split metric.
    """
    if bool(getattr(run_cfg, "eval_only", False)):
        return 0
    return int(getattr(run_cfg, "val_max_cases", 0))


def _effective_val_frames_per_case(run_cfg) -> int:
    """Frames per validation case to keep: the cap, or 0 (all) on a scoring run.

    Same contract as _effective_val_max_cases, on the time axis: the in-training
    validator reruns every rec_validation_freq and its per-frame metrics are
    heavily autocorrelated within a case, so a handful of spread frames answer
    the health question. A scoring run (NS_Energy_laser_PINO_eval) must see the
    full split; its own subsetting is PINO_EVAL_FRAME_STRIDE, which stamps the
    stride into the stored metrics.
    """
    if bool(getattr(run_cfg, "eval_only", False)):
        return 0
    return int(getattr(run_cfg, "val_frames_per_case", 0))


def _build_lazy_datasets(
    *,
    var: RuntimeStore,
    key_cfg: KeyConfig,
    split: SplitSpec,
    run_cfg: RunConfig,
    var_val_source: RuntimeStore | None = None,
) -> DatasetBundle:
    """Build lazy map-style datasets from a staged arraydir runtime."""
    train_store = var
    val_store = var if var_val_source is None else var_val_source
    train_case_indices = [int(idx) for idx in split.train_idx.tolist()]
    val_case_indices = [int(idx) for idx in split.val_idx.tolist()]
    val_max = _effective_val_max_cases(run_cfg)
    if 0 < val_max < len(val_case_indices):
        n_all = len(val_case_indices)
        val_case_indices = _subsample_val_cases(
            val_case_indices, val_max, int(getattr(run_cfg, "seed", 42)))
        print(f"[val] validating on {len(val_case_indices)}/{n_all} cases "
              f"(custom.val_max_cases={val_max}, seeded subset)")

    if run_cfg.time_as_dim and not train_store.has_uniform_timesteps(train_case_indices):
        raise ValueError("time_as_dim requires all selected training simulations to have the same timestep count.")
    if run_cfg.time_as_dim and not val_store.has_uniform_timesteps(val_case_indices):
        raise ValueError("time_as_dim requires all selected validation simulations to have the same timestep count.")

    train_h_norm = _effective_h_norm_denom(run_cfg, train_store, train_case_indices)
    # BUG-04: val/eval reuse the train denom rather than fitting a separate
    # per-split max, so identical (P, V_scan) normalizes identically across
    # splits (mirrors the eager path's stash/reuse in _preprocess_var).
    val_h_norm = train_h_norm
    # And the record that carries it to a later evaluation (no-op at eval, where
    # `train_store` is the scored split and network_dir the model being scored).
    write_h_norm_record(run_cfg, train_h_norm, n_cases=len(train_case_indices))
    write_input_fields_record(run_cfg)
    if bool(getattr(run_cfg, "eval_only", False)) and getattr(run_cfg, "h_norm_denom", None) is None:
        run_cfg.h_norm_denom_scored_fit = float(train_h_norm)  # the fallback, for the metrics
    _prepare_runtime_reference(train_store, run_cfg, train_case_indices, h_norm_denom=train_h_norm)

    if run_cfg.autoregressive_enabled and run_cfg.autoregressive_mode == "rollout":
        # Streamed twin of the eager pair path: one k-window per sample, so
        # variable-length cases need no truncation and the split never lands in
        # RAM whole. Train uses k-windows (multi-step loss), validation stays
        # one-step so the in-training validator keeps its per-frame meaning.
        unroll_steps = max(1, int(getattr(run_cfg, "autoregressive_unroll_steps", 1)))
        bundle_steps = max(1, int(getattr(run_cfg, "autoregressive_bundle_steps", 1)))
        prev_noise_std = float(getattr(run_cfg, "autoregressive_prev_noise_std", 0.0))
        window_steps = bundle_steps if bundle_steps > 1 else unroll_steps
        train_entries = _build_ar_window_entries(train_store, train_case_indices, window_steps)
        val_entries = _build_ar_window_entries(val_store, val_case_indices, 1)
        if not train_entries:
            raise ValueError(
                f"No training case has the {window_steps + 1} consecutive frames "
                f"autoregressive unroll_steps/bundle_steps={window_steps} needs."
            )
        outvar_keys = list(dict.fromkeys(list(run_cfg.autoregressive_state_keys) + ["fl"]))
        # Bundled future frames are extra TRAIN outvar keys; validation stays
        # one-step (base keys only) so the in-training validator keeps its
        # per-frame meaning.
        train_outvar_keys = list(outvar_keys)
        if bundle_steps > 1:
            train_outvar_keys += [
                f"{key}_bnd{i}"
                for key in run_cfg.autoregressive_state_keys
                for i in range(1, bundle_steps)
            ]
        data_ds = LazySampleDataset(
            case_coherent=bool(getattr(run_cfg, "case_coherent_sampling", False)),
            seed=int(getattr(run_cfg, "seed", 42)),
            store=train_store,
            entries=train_entries,
            sample_builder=_build_ar_window_sample_builder(
                store=train_store,
                run_cfg=run_cfg,
                key_cfg=key_cfg,
                h_norm_denom=train_h_norm,
                unroll_steps=unroll_steps,
                prev_noise_std=prev_noise_std,
                bundle_steps=bundle_steps,
            ),
            invar_keys=_ar_window_invar_keys(run_cfg, unroll_steps, bundle_steps),
            outvar_keys=train_outvar_keys,
        )
        val_data_ds = LazySampleDataset(
            store=val_store,
            entries=val_entries,
            sample_builder=_build_ar_window_sample_builder(
                store=val_store,
                run_cfg=run_cfg,
                key_cfg=key_cfg,
                h_norm_denom=val_h_norm,
                unroll_steps=1,
                prev_noise_std=0.0,  # clean validation targets
            ),
            invar_keys=_ar_window_invar_keys(run_cfg, 1),
            outvar_keys=outvar_keys,
        )
        print(
            f"[AR] lazy rollout: {len(train_entries)} train windows "
            f"(k={unroll_steps}, bundle={bundle_steps}) over "
            f"{len(train_case_indices)} cases, {len(val_entries)} val pairs over {len(val_case_indices)} cases"
        )

        def _val_frame_reader(*, case_idx: int, time_idx: int):
            return _read_processed_frame(
                store=val_store, run_cfg=run_cfg, h_norm_denom=val_h_norm,
                case_idx=case_idx, time_idx=time_idx)

        return DatasetBundle(
            val_frame_reader=_val_frame_reader,
            val_case_indices=[int(c) for c in val_case_indices],
            val_case_lengths={int(c): int(val_store.time_steps(case_idx=int(c)))
                              for c in val_case_indices},
            var_train=None,
            var_val=None,
            ds_input=None,
            ds_output=None,
            ds_input_val=None,
            ds_output_val=None,
            data_ds=data_ds,
            val_data_ds=val_data_ds,
        )

    if run_cfg.autoregressive_enabled:
        if run_cfg.autoregressive_mode != "masked_patch":
            raise ValueError(
                f"Unknown autoregressive.mode='{run_cfg.autoregressive_mode}'; "
                "lazy arraydir stores support 'rollout' and 'masked_patch'."
            )
        patch_builder_train = _build_masked_patch_builder(store=train_store, run_cfg=run_cfg, h_norm_denom=train_h_norm)
        patch_builder_val = _build_masked_patch_builder(store=val_store, run_cfg=run_cfg, h_norm_denom=val_h_norm)
        data_ds = MaskedTemporalPatchDataset(
            store=train_store,
            case_indices=train_case_indices,
            build_patch_sample=patch_builder_train,
            length=run_cfg.autoregressive_train_patches_per_epoch,
            seed=run_cfg.seed + split.fold * 10000,
            invar_keys=_aux_invar_keys(run_cfg, key_cfg.str_fno_input_keys),
            outvar_keys=list(key_cfg.str_fno_output_keys) + ["fl"],
        )
        val_data_ds = MaskedTemporalPatchDataset(
            store=val_store,
            case_indices=val_case_indices,
            build_patch_sample=patch_builder_val,
            length=run_cfg.autoregressive_val_patches_per_epoch,
            seed=run_cfg.seed + split.fold * 10000 + 5000,
            invar_keys=_aux_invar_keys(run_cfg, key_cfg.str_fno_input_keys),
            outvar_keys=list(key_cfg.str_fno_output_keys) + ["fl"],
        )
        return DatasetBundle(
            var_train=None,
            var_val=None,
            ds_input=None,
            ds_output=None,
            ds_input_val=None,
            ds_output_val=None,
            data_ds=data_ds,
            val_data_ds=val_data_ds,
        )

    train_entries = _build_standard_entries(train_store, train_case_indices, run_cfg)
    val_frames = _effective_val_frames_per_case(run_cfg)
    val_entries = _build_standard_entries(val_store, val_case_indices, run_cfg, frames_per_case=val_frames)
    if val_frames > 0 and run_cfg.time_as_batch and not run_cfg.time_as_dim:
        print(f"[val] validating on {len(val_entries)} frames over {len(val_case_indices)} cases "
              f"(custom.val_frames_per_case={val_frames}, evenly spaced)")
    train_builder = _build_standard_sample_builder(
        store=train_store,
        run_cfg=run_cfg,
        key_cfg=key_cfg,
        h_norm_denom=train_h_norm,
    )
    val_builder = _build_standard_sample_builder(
        store=val_store,
        run_cfg=run_cfg,
        key_cfg=key_cfg,
        h_norm_denom=val_h_norm,
    )
    # temporal-consistency (lever 2) declares extra ride-along invar keys the builder emits
    _std_invar = _aux_invar_keys(run_cfg, key_cfg.str_fno_input_keys)
    if tc_weight() > 0.0:
        _std_invar = _std_invar + tc_ride_along_keys(key_cfg.str_fno_input_keys)
    data_ds = LazySampleDataset(
        store=train_store,
        entries=train_entries,
        sample_builder=train_builder,
        invar_keys=_std_invar,
        outvar_keys=list(key_cfg.str_fno_output_keys) + ["fl"],
        case_coherent=bool(getattr(run_cfg, "case_coherent_sampling", False)),
        seed=int(getattr(run_cfg, "seed", 42)),
    )
    val_data_ds = LazySampleDataset(
        store=val_store,
        entries=val_entries,
        sample_builder=val_builder,
        invar_keys=_std_invar,
        outvar_keys=list(key_cfg.str_fno_output_keys) + ["fl"],
    )
    return DatasetBundle(
        var_train=None,
        var_val=None,
        ds_input=None,
        ds_output=None,
        ds_input_val=None,
        ds_output_val=None,
        data_ds=data_ds,
        val_data_ds=val_data_ds,
    )


class NoisyPrevGridDataset(DictGridDataset):
    """Grid dataset that redraws Gaussian noise on the ``*_prev`` inputs on every
    ``__getitem__`` (per-batch pushforward noise, mitigation #2).

    Static build-time noise perturbs each sample once per run; resampling on
    access shows the model a fresh error draw each epoch, closer to the changing
    state it feeds back at rollout. Only the listed prev keys are perturbed;
    context and targets stay clean.
    """

    def __init__(self, invar, outvar, prev_keys, noise_std, lambda_weighting=None, seed=0):
        super().__init__(invar, outvar, lambda_weighting)
        import torch

        self._prev_keys = [key for key in prev_keys if key in self.invar]
        self._noise_std = float(noise_std)
        self._gen = torch.Generator().manual_seed(int(seed))

    def __getitem__(self, idx):
        import torch

        invar, outvar, lambda_weighting = super().__getitem__(idx)
        if self._noise_std > 0.0:
            for key in self._prev_keys:
                tensor = invar[key]
                noise = torch.randn(tensor.shape, generator=self._gen, dtype=tensor.dtype)
                invar[key] = tensor + noise * self._noise_std
        return invar, outvar, lambda_weighting


def _build_autoregressive_pairs(
    var: dict[str, np.ndarray],
    context_keys: list[str],
    state_keys: list[str],
    state_prev_suffix: str,
    extra_target_keys: list[str] | None = None,
    prev_noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Create one-step training pairs using next-step context and previous state.

    Each pair predicts targets at timestep ``t+1`` from autoregressive state at
    timestep ``t`` together with exogenous/context inputs defined on timestep
    ``t+1``. ``prev_noise_std`` adds Gaussian noise to the previous-state inputs
    only (standard rollout-stability trick); targets and context stay clean.
    """

    extra_target_keys = list(extra_target_keys or [])
    required_keys = list(dict.fromkeys(context_keys + state_keys + extra_target_keys))
    missing_keys = [key for key in required_keys if key not in var]
    if missing_keys:
        raise KeyError(f"Autoregressive pair construction missing keys: {missing_keys}")

    if not state_prev_suffix:
        raise ValueError("state_prev_suffix must be a non-empty string.")

    ref_shape: tuple[int, int] | None = None
    for key in required_keys:
        arr = np.asarray(var[key])
        if arr.ndim not in (5, 6):
            raise ValueError(
                f"Autoregressive mode expects temporal grid tensors with ndim in (5, 6); key '{key}' has shape {arr.shape}."
            )
        if ref_shape is None:
            ref_shape = (arr.shape[0], arr.shape[1])
        elif ref_shape != (arr.shape[0], arr.shape[1]):
            raise ValueError(
                f"Temporal shape mismatch for key '{key}': {(arr.shape[0], arr.shape[1])} != {ref_shape}."
            )

    assert ref_shape is not None
    n_batch, n_time = ref_shape
    if n_time < 2:
        raise ValueError(f"Autoregressive mode requires at least 2 timesteps; got {n_time}.")

    def _flatten_steps(arr: np.ndarray, step_slice: slice) -> np.ndarray:
        stepped = arr[:, step_slice, ...]
        return stepped.reshape(n_batch * (n_time - 1), *arr.shape[2:])

    ds_input: dict[str, np.ndarray] = {}
    for key in context_keys:
        ds_input[key] = _flatten_steps(np.asarray(var[key]), slice(1, None))

    if prev_noise_std < 0.0:
        raise ValueError(f"prev_noise_std must be >= 0; got {prev_noise_std}.")
    if prev_noise_std > 0.0 and rng is None:
        rng = np.random.default_rng()

    for key in state_keys:
        prev_key = f"{key}{state_prev_suffix}"
        if prev_key in ds_input:
            raise ValueError(
                f"Autoregressive input collision for key '{prev_key}'. "
                "Choose a different state_prev_suffix or adjust FNO_input_keys."
            )
        prev_frames = _flatten_steps(np.asarray(var[key]), slice(0, -1))
        if prev_noise_std > 0.0:
            prev_frames = prev_frames + rng.normal(0.0, prev_noise_std, size=prev_frames.shape).astype(
                prev_frames.dtype, copy=False
            )
        ds_input[prev_key] = prev_frames

    ds_output: dict[str, np.ndarray] = {}
    for key in state_keys:
        ds_output[key] = _flatten_steps(np.asarray(var[key]), slice(1, None))
    for key in extra_target_keys:
        ds_output[key] = _flatten_steps(np.asarray(var[key]), slice(1, None))

    return ds_input, ds_output


def _build_autoregressive_window_pairs(
    var: dict[str, np.ndarray],
    context_keys: list[str],
    state_keys: list[str],
    state_prev_suffix: str,
    unroll_steps: int,
    extra_target_keys: list[str] | None = None,
    prev_noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """k-step window pairs for the multi-step unrolled loss (mitigation #4).

    For each window starting at trajectory index ``t0`` (``n_time - k`` windows
    per case), emits per-step keys so every tensor stays a single frame
    ``(N, C, X, Y, Z)`` and the physicsnemo Graph ignores the extras:

      context ``c``:            base = GT[t0+1],  ``c__ar{i}`` = GT[t0+1+i]
      seed prev ``{s}_prev``:   GT[t0]                       (only the seed)
      target ``s`` (+extras):   base = GT[t0+1],  ``s__ar{i}`` = GT[t0+1+i]

    with ``i = 1..k-1``. ``UnrolledGridConstraint`` seeds step 0 from the GT prev,
    then feeds its own prediction into the ``*_prev`` slot for steps ``i>=1`` and
    scores each step against the matching target. ``k=1`` is one-step training and
    should use ``_build_autoregressive_pairs`` instead. Noise perturbs only the
    seed prev (train-only), matching the one-step builder.
    """

    if unroll_steps < 2:
        raise ValueError(f"window pairs require unroll_steps >= 2; got {unroll_steps}.")
    extra_target_keys = list(extra_target_keys or [])
    required_keys = list(dict.fromkeys(context_keys + state_keys + extra_target_keys))
    missing_keys = [key for key in required_keys if key not in var]
    if missing_keys:
        raise KeyError(f"Autoregressive window construction missing keys: {missing_keys}")
    if not state_prev_suffix:
        raise ValueError("state_prev_suffix must be a non-empty string.")
    if prev_noise_std < 0.0:
        raise ValueError(f"prev_noise_std must be >= 0; got {prev_noise_std}.")

    ref_shape: tuple[int, int] | None = None
    for key in required_keys:
        arr = np.asarray(var[key])
        if arr.ndim not in (5, 6):
            raise ValueError(
                f"Autoregressive mode expects temporal grid tensors with ndim in (5, 6); "
                f"key '{key}' has shape {arr.shape}."
            )
        if ref_shape is None:
            ref_shape = (arr.shape[0], arr.shape[1])
        elif ref_shape != (arr.shape[0], arr.shape[1]):
            raise ValueError(f"Temporal shape mismatch for key '{key}': got {arr.shape[:2]} vs {ref_shape}.")

    assert ref_shape is not None
    n_batch, n_time = ref_shape
    n_windows = n_time - unroll_steps
    if n_windows < 1:
        raise ValueError(
            f"unroll_steps={unroll_steps} needs at least {unroll_steps + 1} timesteps; got {n_time}."
        )
    if prev_noise_std > 0.0 and rng is None:
        rng = np.random.default_rng()

    def _win(arr: np.ndarray, start: int) -> np.ndarray:
        # frames [start, start + n_windows) across all cases -> (n_batch*n_windows, C, ...)
        stepped = arr[:, start : start + n_windows, ...]
        return stepped.reshape(n_batch * n_windows, *arr.shape[2:])

    # Only the base (step-0) outvar keys drive the physicsnemo Graph; per-step
    # future context and future targets ride along as extra ds_input keys (the
    # Graph ignores unused inputs), which UnrolledGridConstraint slices per step.
    ds_input: dict[str, np.ndarray] = {}
    for key in context_keys:
        arr = np.asarray(var[key])
        ds_input[key] = _win(arr, 1)  # step 0 context = GT[t0+1]
        for i in range(1, unroll_steps):
            ds_input[f"{key}__ar{i}"] = _win(arr, 1 + i)

    for key in state_keys:
        prev_key = f"{key}{state_prev_suffix}"
        if prev_key in ds_input:
            raise ValueError(
                f"Autoregressive input collision for key '{prev_key}'. "
                "Choose a different state_prev_suffix or adjust FNO_input_keys."
            )
        seed = _win(np.asarray(var[key]), 0)  # seed prev = GT[t0]
        if prev_noise_std > 0.0:
            seed = seed + rng.normal(0.0, prev_noise_std, size=seed.shape).astype(seed.dtype, copy=False)
        ds_input[prev_key] = seed

    target_keys = list(dict.fromkeys(list(state_keys) + extra_target_keys))
    ds_output: dict[str, np.ndarray] = {}
    for key in target_keys:
        arr = np.asarray(var[key])
        ds_output[key] = _win(arr, 1)  # step 0 target = GT[t0+1]
        for i in range(1, unroll_steps):
            ds_input[f"{key}__tgt_ar{i}"] = _win(arr, 1 + i)  # future targets ride in ds_input

    return ds_input, ds_output


def _ar_window_sample(
    frames: Sequence[dict[str, np.ndarray]],
    *,
    context_keys: Sequence[str],
    state_keys: Sequence[str],
    state_prev_suffix: str,
    unroll_steps: int,
    extra_target_keys: Sequence[str] | None = None,
    prev_noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    bundle_steps: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Assemble ONE k-window AR sample from ``unroll_steps + 1`` consecutive frames.

    The per-sample twin of ``_build_autoregressive_window_pairs``: same keys,
    same frame-to-slot mapping, one window at a time so the lazy store can
    stream them instead of stacking every case in RAM. ``frames[j]`` is the
    preprocessed frame at ``t0 + j``. ``unroll_steps=1`` degenerates to the
    one-step layout of ``_build_autoregressive_pairs`` (no ``__ar`` extras),
    which is what the in-training validator consumes.

    ``bundle_steps=B > 1`` (temporal bundling, upstream RIGNO stepping.py):
    one forward predicts frames t0+1..t0+B, so the invar keeps the one-step
    layout (step-0 context + seed prev, no ``__ar``/``__tgt_ar`` extras) and
    the outvar carries the future frames as ``{key}_bnd{i}`` for the state keys
    (extra targets like ``fl`` stay step-0 only). Needs ``unroll_steps=1`` and
    ``bundle_steps + 1`` frames.
    """

    if unroll_steps < 1:
        raise ValueError(f"unroll_steps must be >= 1; got {unroll_steps}.")
    if bundle_steps < 1:
        raise ValueError(f"bundle_steps must be >= 1; got {bundle_steps}.")
    if bundle_steps > 1 and unroll_steps != 1:
        raise ValueError(
            f"bundle_steps={bundle_steps} requires unroll_steps=1 (bundling and "
            f"unrolled feedback are mutually exclusive); got unroll_steps={unroll_steps}."
        )
    n_expected = (bundle_steps if bundle_steps > 1 else unroll_steps) + 1
    if len(frames) != n_expected:
        raise ValueError(
            f"unroll_steps={unroll_steps}/bundle_steps={bundle_steps} requires "
            f"{n_expected} frames; got {len(frames)}."
        )
    if not state_prev_suffix:
        raise ValueError("state_prev_suffix must be a non-empty string.")
    if prev_noise_std < 0.0:
        raise ValueError(f"prev_noise_std must be >= 0; got {prev_noise_std}.")
    if prev_noise_std > 0.0 and rng is None:
        rng = np.random.default_rng()

    invar: dict[str, np.ndarray] = {}
    for key in context_keys:
        invar[key] = frames[1][key]  # step 0 context = GT[t0+1]
        for i in range(1, unroll_steps):
            invar[f"{key}__ar{i}"] = frames[1 + i][key]

    for key in state_keys:
        prev_key = f"{key}{state_prev_suffix}"
        if prev_key in invar:
            raise ValueError(
                f"Autoregressive input collision for key '{prev_key}'. "
                "Choose a different state_prev_suffix or adjust FNO_input_keys."
            )
        seed = frames[0][key]  # seed prev = GT[t0]
        if prev_noise_std > 0.0:
            assert rng is not None
            seed = seed + rng.normal(0.0, prev_noise_std, size=seed.shape).astype(seed.dtype, copy=False)
        invar[prev_key] = seed

    outvar: dict[str, np.ndarray] = {}
    for key in dict.fromkeys(list(state_keys) + list(extra_target_keys or [])):
        outvar[key] = frames[1][key]  # step 0 target = GT[t0+1]
        for i in range(1, unroll_steps):
            invar[f"{key}__tgt_ar{i}"] = frames[1 + i][key]  # future targets ride in invar

    if bundle_steps > 1:
        # Bundled future frames are real outvar keys: the Graph requests them,
        # the widened model head produces them, the stock constraint trains them.
        for key in state_keys:
            for i in range(1, bundle_steps):
                outvar[f"{key}_bnd{i}"] = frames[1 + i][key]
        # ...and each bundled frame gets ITS OWN exogenous context. Measured on
        # kh2579: the laser moves 0.57 beam radii across a 4-frame bundle on a
        # random path, so frames t+2.. are not determined by the step-0 context
        # (first bundle4 run: flat train loss, rising val error). The scan path
        # is planned and known at inference, so this is forcing, not leakage.
        for key in context_keys:
            for i in range(1, bundle_steps):
                invar[f"{key}__ar{i}"] = frames[1 + i][key]

    return invar, outvar


def _build_ar_window_entries(
    store: RuntimeStore,
    case_indices: Sequence[int],
    unroll_steps: int,
) -> list[tuple[int, int]]:
    """(case, t0) entries for every k-window, honouring each case's own length.

    Cases shorter than ``unroll_steps + 1`` frames contribute nothing rather
    than truncating the split to the shortest case (which is what the eager
    stacking path would force).
    """

    entries: list[tuple[int, int]] = []
    for case_idx in case_indices:
        n_windows = int(store.time_steps(case_idx=int(case_idx))) - int(unroll_steps)
        entries.extend((int(case_idx), t0) for t0 in range(max(0, n_windows)))
    return entries


def _build_ar_window_sample_builder(
    *,
    store: RuntimeStore,
    run_cfg: RunConfig,
    key_cfg: KeyConfig,
    h_norm_denom: float,
    unroll_steps: int,
    prev_noise_std: float = 0.0,
    bundle_steps: int = 1,
):
    """Per-sample builder that streams the k+1 frames of one AR window."""

    context_keys = list(getattr(run_cfg, "autoregressive_context_keys", []))
    state_keys = list(getattr(run_cfg, "autoregressive_state_keys", []))
    state_prev_suffix = str(getattr(run_cfg, "autoregressive_state_prev_suffix", "_prev"))
    if not context_keys or not state_keys:
        raise ValueError("Autoregressive mode requires non-empty context and state key sets.")
    seed = int(getattr(run_cfg, "seed", 0))

    def _builder(case_idx: int, time_idx: int | None):
        if time_idx is None:
            raise ValueError("Autoregressive rollout samples need a frame index.")
        frames = [
            _read_processed_frame(
                store=store,
                run_cfg=run_cfg,
                h_norm_denom=h_norm_denom,
                case_idx=case_idx,
                time_idx=time_idx + j,
            )
            for j in range((bundle_steps if bundle_steps > 1 else unroll_steps) + 1)
        ]
        # Deterministic per-sample stream so a resumed/reshuffled epoch sees the
        # same perturbation for the same window.
        rng = np.random.default_rng([seed, int(case_idx), int(time_idx)]) if prev_noise_std > 0.0 else None
        return _ar_window_sample(
            frames,
            context_keys=context_keys,
            state_keys=state_keys,
            state_prev_suffix=state_prev_suffix,
            unroll_steps=unroll_steps,
            extra_target_keys=["fl"],
            prev_noise_std=prev_noise_std,
            rng=rng,
            bundle_steps=bundle_steps,
        )

    return _builder


def _aux_invar_keys(run_cfg, base) -> list[str]:
    """The dataset's read keys: the model's inputs plus any AUXILIARY channel a graph node
    or a metric needs that the model must not see. `mat_Tv` rides along for the per-case
    mask fill (`custom.per_case_t_boil`); it never enters `str_fno_input_keys`, and input
    assembly indexes by those, so the network is blind to it. Serving it is what lets the
    training MaskNode and the eval fills read the same per-case value."""
    from ...custom_utils.utils import custom_flag

    keys = list(base)
    if bool(custom_flag(run_cfg, "per_case_t_boil", False)) and "mat_Tv" not in keys:
        keys.append("mat_Tv")
    return keys


def _ar_window_invar_keys(run_cfg: RunConfig, unroll_steps: int,
                          bundle_steps: int = 1) -> list[str]:
    """Every invar key `_ar_window_sample` emits, in dataset-declaration order."""

    context_keys = list(getattr(run_cfg, "autoregressive_context_keys", []))
    state_keys = list(getattr(run_cfg, "autoregressive_state_keys", []))
    suffix = str(getattr(run_cfg, "autoregressive_state_prev_suffix", "_prev"))
    keys = list(context_keys)
    for key in context_keys:
        keys.extend(f"{key}__ar{i}" for i in range(1, unroll_steps))
    keys.extend(f"{key}{suffix}" for key in state_keys)
    for key in dict.fromkeys(list(state_keys) + ["fl"]):
        keys.extend(f"{key}__tgt_ar{i}" for i in range(1, unroll_steps))
    if bundle_steps > 1:
        # Per-bundled-frame exogenous context (see _ar_window_sample).
        for key in context_keys:
            keys.extend(f"{key}__ar{i}" for i in range(1, bundle_steps))
    keys = _aux_invar_keys(run_cfg, keys)
    return keys


def get_splits(
    mode: str,
    n_samples: int,
    k_folds: int = 8,
    fold_index: int = -1,
    seed: int = 42,
    val_n_samples: Optional[int] = None,
) -> list[SplitSpec]:
    """Return split specs for standard mode or sklearn-based k-fold mode."""

    if mode == "standard":
        val_size = n_samples if val_n_samples is None else val_n_samples
        return [
            SplitSpec(
                fold=0,
                train_idx=np.arange(n_samples),
                val_idx=np.arange(val_size),
            )
        ]

    if mode == "kfold":
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        all_idx = np.arange(n_samples)
        splits = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(all_idx)):
            if fold_index not in (-1, fold):
                continue
            splits.append(SplitSpec(fold=fold, train_idx=train_idx, val_idx=val_idx))
        return splits

    raise ValueError(f"Unsupported split mode: {mode}")


def build_datasets(
    var: dict[str, np.ndarray] | RuntimeStore,
    key_cfg: KeyConfig,
    split: SplitSpec,
    pde_loss: bool,
    vof_pde_loss: bool,
    diff_loss: bool,
    var_val_source: Optional[dict[str, np.ndarray] | RuntimeStore] = None,
    run_cfg: Optional[RunConfig] = None,
) -> DatasetBundle:
    """Create all DictGridDataset objects needed by active training branches."""
    if isinstance(var, _RUNTIME_STORE_TYPES):
        if run_cfg is None:
            raise ValueError("Lazy runtime dataset construction requires run_cfg.")
        if run_cfg.data_layout != "grid":
            raise ValueError("Lazy arraydir runtime currently supports grid-model training/evaluation only.")
        if pde_loss or vof_pde_loss or diff_loss:
            raise ValueError("Lazy arraydir runtime currently supports data-only training/evaluation only.")
        lazy_val_source = (
            var_val_source
            if isinstance(var_val_source, _RUNTIME_STORE_TYPES)
            else None
        )
        return _build_lazy_datasets(
            var=var,
            key_cfg=key_cfg,
            split=split,
            run_cfg=run_cfg,
            var_val_source=lazy_val_source,
        )

    var_train = split_dict(var, split.train_idx)
    if var_val_source is None:
        var_val = split_dict(var, split.val_idx)
    else:
        # Standard mode can provide a dedicated validation source directory.
        var_val = split_dict(var_val_source, split.val_idx)

    autoregressive_enabled = bool(getattr(run_cfg, "autoregressive_enabled", False)) if run_cfg is not None else False
    if autoregressive_enabled:
        eager_ar_mode = str(getattr(run_cfg, "autoregressive_mode", "rollout")).strip().lower()
        if eager_ar_mode != "rollout":
            raise ValueError(
                f"autoregressive.mode='{eager_ar_mode}' requires lazy arraydir stores; the eager VTK/NPZ "
                "dict data path only supports autoregressive.mode='rollout'."
            )
        if pde_loss or vof_pde_loss or diff_loss:
            raise ValueError(
                "Autoregressive mode currently supports data-only training; disable pde_loss/VOF_pde_loss/Diff_loss."
            )
        context_keys = list(getattr(run_cfg, "autoregressive_context_keys", []))
        state_keys = list(getattr(run_cfg, "autoregressive_state_keys", []))
        state_prev_suffix = str(getattr(run_cfg, "autoregressive_state_prev_suffix", "_prev"))
        if not context_keys or not state_keys:
            raise ValueError("Autoregressive mode requires non-empty context and state key sets.")

        prev_noise_std = float(getattr(run_cfg, "autoregressive_prev_noise_std", 0.0))
        prev_noise_resample = bool(getattr(run_cfg, "autoregressive_prev_noise_resample", False))
        grid_layout = not (run_cfg is not None and run_cfg.data_layout == "pointwise")
        # Per-batch resampling (grid only) builds clean pairs and adds fresh noise
        # in the dataset wrapper; otherwise keep the static build-time noise.
        build_noise = 0.0 if (prev_noise_resample and grid_layout) else prev_noise_std
        unroll_steps = int(getattr(run_cfg, "autoregressive_unroll_steps", 1))
        if int(getattr(run_cfg, "autoregressive_bundle_steps", 1)) > 1:
            raise ValueError(
                "autoregressive.bundle_steps>1 is only implemented for the lazy "
                "dataset path (PINO_LAZY_NPZ); the eager NPZ path does not build "
                "bundled outvar keys."
            )
        if unroll_steps > 1 and not grid_layout:
            raise ValueError(
                "autoregressive.unroll_steps>1 (multi-step loss) is grid-only; "
                "DeepONet/pointwise unroll is not implemented."
            )
        train_rng = np.random.default_rng(int(getattr(run_cfg, "seed", 0)) + split.fold)
        # Multi-step (k>1) training uses windowed pairs; validation stays one-step
        # so the in-training validator keeps its per-frame meaning.
        if unroll_steps > 1:
            ds_input, ds_output = _build_autoregressive_window_pairs(
                var=var_train,
                context_keys=context_keys,
                state_keys=state_keys,
                state_prev_suffix=state_prev_suffix,
                unroll_steps=unroll_steps,
                extra_target_keys=["fl"],
                prev_noise_std=build_noise,
                rng=train_rng,
            )
        else:
            ds_input, ds_output = _build_autoregressive_pairs(
                var=var_train,
                context_keys=context_keys,
                state_keys=state_keys,
                state_prev_suffix=state_prev_suffix,
                extra_target_keys=["fl"],
                prev_noise_std=build_noise,
                rng=train_rng,
            )
        ds_input_val, ds_output_val = _build_autoregressive_pairs(
            var=var_val,
            context_keys=context_keys,
            state_keys=state_keys,
            state_prev_suffix=state_prev_suffix,
            extra_target_keys=["fl"],
        )
        print(
            f"Autoregressive dataset input keys: {ds_input.keys()}, "
            f"state outputs: {state_keys}, extra outputs: ['fl']"
        )
        if run_cfg is not None and run_cfg.data_layout == "pointwise":
            input_keys = list(ds_input.keys())
            output_keys = list(ds_output.keys())
            ds_input = _flatten_spatial_pointwise(ds_input, input_keys)
            ds_output = _flatten_spatial_pointwise(ds_output, output_keys)
            ds_input_val = _flatten_spatial_pointwise(ds_input_val, input_keys)
            ds_output_val = _flatten_spatial_pointwise(ds_output_val, output_keys)
            data_ds = DictPointwiseDataset(ds_input, ds_output)
            val_data_ds = DictPointwiseDataset(ds_input_val, ds_output_val)
        else:
            if prev_noise_resample and prev_noise_std > 0.0:
                prev_keys = [f"{key}{state_prev_suffix}" for key in state_keys]
                data_ds = NoisyPrevGridDataset(
                    ds_input,
                    ds_output,
                    prev_keys=prev_keys,
                    noise_std=prev_noise_std,
                    seed=int(getattr(run_cfg, "seed", 0)) + split.fold,
                )
            else:
                data_ds = DictGridDataset(ds_input, ds_output)
            val_data_ds = DictGridDataset(ds_input_val, ds_output_val)
    else:
        if run_cfg is not None and run_cfg.data_layout == "pointwise":
            pointwise_output_keys = list(key_cfg.str_fno_output_keys)
            if "fl" in var_train and "fl" not in pointwise_output_keys:
                pointwise_output_keys.append("fl")
            ds_input = _flatten_spatial_pointwise(var_train, key_cfg.str_fno_input_keys)
            ds_output = _flatten_spatial_pointwise(var_train, pointwise_output_keys)
            ds_input_val = _flatten_spatial_pointwise(var_val, key_cfg.str_fno_input_keys)
            ds_output_val = _flatten_spatial_pointwise(var_val, pointwise_output_keys)
            print(f"Pointwise dataset input keys: {ds_input.keys()}, output keys: {pointwise_output_keys}")
            data_ds = DictPointwiseDataset(ds_input, ds_output)
            val_data_ds = DictPointwiseDataset(ds_input_val, ds_output_val)
        else:
            ds_input = {key: var_train[key] for key in key_cfg.str_fno_input_keys}
            ds_output = {key: var_train[key] for key in key_cfg.str_fno_output_keys}
            ds_output["fl"] = var_train["fl"]

            ds_input_val = {key: var_val[key] for key in key_cfg.str_fno_input_keys}
            ds_output_val = {key: var_val[key] for key in key_cfg.str_fno_output_keys}
            ds_output_val["fl"] = var_val["fl"]

            print(f"Dataset input keys: {ds_input.keys()}, output keys: {key_cfg.str_fno_output_keys}")
            data_ds = DictGridDataset(ds_input, ds_output)
            val_data_ds = DictGridDataset(ds_input_val, ds_output_val)

    energy_ds = None
    val_energy_ds = None
    if pde_loss:
        print("Creating PDE dataset for energy equation")
        ds_input_energy = {key: var_train[key] for key in key_cfg.str_energy_input_keys}
        ds_output_energy = {key: var_train[key] for key in key_cfg.str_energy_output_keys}
        energy_ds = DictGridDataset(ds_input_energy, ds_output_energy)

        ds_input_energy_val = {key: var_val[key] for key in key_cfg.str_energy_input_keys}
        ds_output_energy_val = {key: var_val[key] for key in key_cfg.str_energy_output_keys}
        val_energy_ds = DictGridDataset(ds_input_energy_val, ds_output_energy_val)

    vof_ds = None
    val_vof_ds = None
    if vof_pde_loss:
        print("Creating PDE dataset for VOF equation")
        ds_input_vof = {key: var_train[key] for key in key_cfg.str_vof_input_keys}
        ds_output_vof = {key: var_train[key] for key in key_cfg.str_vof_output_keys}
        vof_ds = DictGridDataset(ds_input_vof, ds_output_vof)

        ds_input_vof_val = {key: var_val[key] for key in key_cfg.str_vof_input_keys}
        ds_output_vof_val = {key: var_val[key] for key in key_cfg.str_vof_output_keys}
        val_vof_ds = DictGridDataset(ds_input_vof_val, ds_output_vof_val)

    diff_ds = None
    val_diff_ds = None
    if diff_loss:
        print("Creating PDE dataset for Diffusion terms")
        ds_input_diff = {key: var_train[key] for key in key_cfg.str_diff_input_keys}
        ds_output_diff = {key: var_train[key] for key in key_cfg.str_diff_output_keys}
        diff_ds = DictGridDataset(ds_input_diff, ds_output_diff)

        ds_input_diff_val = {key: var_val[key] for key in key_cfg.str_diff_input_keys}
        ds_output_diff_val = {key: var_val[key] for key in key_cfg.str_diff_output_keys}
        val_diff_ds = DictGridDataset(ds_input_diff_val, ds_output_diff_val)

    return DatasetBundle(
        var_train=var_train,
        var_val=var_val,
        ds_input=ds_input,
        ds_output=ds_output,
        ds_input_val=ds_input_val,
        ds_output_val=ds_output_val,
        data_ds=data_ds,
        val_data_ds=val_data_ds,
        energy_ds=energy_ds,
        val_energy_ds=val_energy_ds,
        vof_ds=vof_ds,
        val_vof_ds=val_vof_ds,
        diff_ds=diff_ds,
        val_diff_ds=val_diff_ds,
    )
