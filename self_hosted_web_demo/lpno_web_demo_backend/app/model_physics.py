"""Per-model laser physics, parsed from a model folder's meta.json "physics" block.

Separate from `model_registry` (which caches it per folder) and from
`adapter_randompath` (which consumes it) so neither has to import the other for
the dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Fixed physics of the legacy random-path sweep (matches the training/test
# sweep generation); the ModelPhysics defaults, so a model without a physics
# block behaves exactly as the historical fixed-physics models did.
P_WATTS = 150.0
V_SCAN_M_PER_S = 0.8
MAX_PATH_LENGTH_M = 1.5e-3


@dataclass(frozen=True, slots=True)
class ModelPhysics:
    """Per-model laser-physics capabilities, from meta.json's "physics" block.

    The defaults reproduce the historical fixed-physics behaviour (rp3 & friends:
    P=150 W, V=0.8 m/s, H_norm=1.0, fixed 2 ms / 20-frame grid, plain dose), so a
    model without the block behaves exactly as before this change.
    """

    p_default: float = P_WATTS
    v_default: float = V_SCAN_M_PER_S
    p_range: tuple[float, float] | None = None  # None = fixed (control greyed out)
    v_range: tuple[float, float] | None = None
    h_norm_denom: float | None = None           # None = H_norm 1.0 (single-P/V split)
    power_weighted_dose: bool = False
    # conduction_field as the model trained on it (custom.power_weighted_conduction,
    # custom.conduction_source_sigma_factor): the defaults are the original channel
    # (source sigma = beam radius, unscaled); the uniform-drawing re-run models use
    # sigma = r0/2 scaled by P/P_ref. load_model checks both against the checkpoint config.
    power_weighted_conduction: bool = False
    conduction_sigma_factor: float = 1.0
    duration_scaled_times: bool = False          # scale the time grid to length/speed
    dt_s: float = 1.0e-5                         # training frame step for scaled grids
    # None = no arc-length trim (variable-length models: the path length IS a free
    # variable they were trained over, so trimming it would defeat the point).
    # A float caps the drawn path at that arc length (the legacy 1.5 mm behaviour).
    max_path_length_m: float | None = MAX_PATH_LENGTH_M
    start_anywhere: bool = False
    # Subdirectory of path_data_dir holding THIS model's ground-truth preset cases
    # (its own test split). None/"" = the legacy shared root. A model must only be
    # offered presets it was trained for: the old fixed-physics cases are not this
    # model's GT and vice versa.
    presets_dir: str | None = None

    @property
    def variable(self) -> bool:
        return self.p_range is not None or self.v_range is not None


def physics_config_mismatches(physics: ModelPhysics, custom: Any) -> list[str]:
    """Differences between a meta.json physics block and the checkpoint's own `custom` config.

    The channel settings below change the VALUES of the inputs, so a meta.json that disagrees with
    the config feeds the network inputs it never trained on and the server still returns plausible
    PNGs. Only keys the config states are compared; an absent key means the pipeline default
    (False / 1.0), which is also the ModelPhysics default. Returns readable `key: meta vs config` lines.
    """
    def get(key: str, default: Any) -> Any:
        try:
            value = custom.get(key, default) if hasattr(custom, "get") else getattr(custom, key, default)
        except Exception:
            value = default
        return default if value is None else value

    checks = (
        ("power_weighted_dose", physics.power_weighted_dose, bool(get("power_weighted_dose", False))),
        ("power_weighted_conduction", physics.power_weighted_conduction,
         bool(get("power_weighted_conduction", False))),
        ("conduction_source_sigma_factor", physics.conduction_sigma_factor,
         float(get("conduction_source_sigma_factor", 1.0))),
    )
    out = []
    for key, meta_value, cfg_value in checks:
        same = abs(float(meta_value) - float(cfg_value)) < 1e-9 if isinstance(cfg_value, float) \
            else bool(meta_value) == bool(cfg_value)
        if not same:
            out.append(f"{key}: meta.json {meta_value!r} vs config {cfg_value!r}")
    return out


def physics_from_meta(meta: dict[str, Any] | None) -> ModelPhysics:
    """Parse a meta.json "physics" block; absent/malformed -> fixed defaults."""
    ph = (meta or {}).get("physics")
    if not isinstance(ph, dict):
        return ModelPhysics()

    def _rng(block: Any) -> tuple[tuple[float, float] | None, float | None]:
        if not isinstance(block, dict):
            return None, None
        lo, hi = block.get("min"), block.get("max")
        rng = (float(lo), float(hi)) if lo is not None and hi is not None else None
        d = block.get("default")
        return rng, (float(d) if d is not None else None)

    p_rng, p_def = _rng(ph.get("p_watts"))
    v_rng, v_def = _rng(ph.get("v_scan_mps"))
    return ModelPhysics(
        p_default=p_def if p_def is not None else P_WATTS,
        v_default=v_def if v_def is not None else V_SCAN_M_PER_S,
        p_range=p_rng,
        v_range=v_rng,
        h_norm_denom=float(ph["h_norm_denom"]) if ph.get("h_norm_denom") else None,
        power_weighted_dose=bool(ph.get("power_weighted_dose", False)),
        power_weighted_conduction=bool(ph.get("power_weighted_conduction", False)),
        conduction_sigma_factor=float(ph.get("conduction_source_sigma_factor", 1.0)),
        duration_scaled_times=bool(ph.get("duration_scaled_times", False)),
        dt_s=float(ph.get("dt_s", 1.0e-5)),
        # explicit null -> no trim; absent -> the legacy cap
        max_path_length_m=(
            None if ("max_path_length_m" in ph and ph["max_path_length_m"] is None)
            else float(ph.get("max_path_length_m", MAX_PATH_LENGTH_M))
        ),
        start_anywhere=bool(ph.get("start_anywhere", False)),
        presets_dir=(str(ph["presets_dir"]) or None) if ph.get("presets_dir") else None,
    )
