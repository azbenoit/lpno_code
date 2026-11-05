"""Physical units and leaderboard badges for model evaluations.

Evaluations store temperature metrics in the trainer's normalised units: the
loader divides T by ``ref_scales["T"]`` (3000 K in every shipped
``config_flow3D_*`` config), so a stored ``T.rmse`` of 0.0136 means 40.7 K. The
scale is a pure factor with no offset, so MAE and RMSE both convert by
multiplication.

The factor is read from the model's own archived Hydra config
(``.hydra/config.yaml`` in the model node's repository) rather than assumed: a
model trained on a different normalisation would otherwise be reported in
confidently wrong Kelvin. When it cannot be established, the caller shows
normalised numbers and says so — never a guessed Kelvin value.

Badges ("best T MAE" / "best T RMSE") are computed per evaluation dataset, in
Kelvin, on every read. Nothing is stored, so a model that is beaten loses its
badge on the next render with no migration, no backfill and no stale tag — and
because the badges never touch ``extras.keywords`` a machine can never corrupt
(or be corrupted by) the user's own keywords.
"""

from __future__ import annotations

import math
from pathlib import Path as _Path
from typing import Any

# T metrics are errors on a divided-by-ref_scales["T"] field: multiply to get K.
KELVIN_KEYS = (
    "T.abs_err_mean", "T.abs_err_std", "T.rmse",
    "T.substrate_abs_err_mean", "T.substrate_rmse",
    "T.surface_abs_err_mean", "T.surface_rmse",
)
# Ratios are already dimensionless — multiplying them would be nonsense.
DIMENSIONLESS_PREFIXES = ("rel_", "l2_")


def kelvin_factor(hydra_config: dict[str, Any] | None) -> tuple[float | None, str]:
    """(factor, provenance) from a model's archived Hydra config.

    Returns (None, why) when the temperature scale cannot be established, so the
    caller can fall back to normalised units instead of inventing Kelvin.
    """
    if not isinstance(hydra_config, dict):
        return None, "no archived Hydra config on this model"
    scales = _find_key(hydra_config, "ref_scales")
    if not isinstance(scales, dict) or "T" not in scales:
        return None, "archived Hydra config has no ref_scales.T"
    try:
        value = float(scales["T"])
    except (TypeError, ValueError):
        return None, f"unusable ref_scales.T={scales['T']!r}"
    if not math.isfinite(value) or value <= 0:
        return None, f"unusable ref_scales.T={scales['T']!r}"
    return value, f"ref_scales.T={value:g} from the model's archived Hydra config"


def model_hydra_config(model) -> dict[str, Any] | None:
    """A model's archived `.hydra/config.yaml`, or None when it has none.

    Lives here rather than in the console so the evaluation workflow and the
    renderer read the scale the same way — the workflow now records it *on the
    evaluation*, and this is the fallback for the 103 evaluations made before
    it did.
    """
    import yaml

    for path in _hydra_config_paths(model):
        try:
            raw = model.base.repository.get_object_content(path)
            parsed = yaml.safe_load(raw)
        except Exception:  # noqa: BLE001 — unreadable or not YAML; try the next
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _hydra_config_paths(model) -> list[str]:
    """Where a model's archived Hydra config can be, root first.

    Not only `.hydra/config.yaml`: a training run whose artifacts were archived
    under the run directory keeps it at `<run_name>/.hydra/config.yaml`. Reading
    only the root path made three models look like they archived nothing — model
    **9716, the declared 1601 baseline, among them** — so the unit of the
    headline metric was resolved from a repo config file and stamped with the
    sentence "this model archives no config of its own", which was false. One of
    the three was imported after that code shipped, so this was not a
    historical quirk.
    """
    paths = [".hydra/config.yaml"]
    try:
        for name in model.base.repository.list_object_names():
            if name.endswith(".yaml") or "." in name:
                continue  # a file at the root, not a run directory
            paths.append(f"{name}/.hydra/config.yaml")
    except Exception:  # noqa: BLE001
        pass
    return paths


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """{"T": {"rmse": ...}} -> {"T.rmse": ...}; only finite numbers survive."""
    flat: dict[str, float] = {}
    for field, block in (metrics or {}).items():
        if not isinstance(block, dict):
            continue
        for name, value in block.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                flat[f"{field}.{name}"] = float(value)
    return flat


def _named_scale_with_inheritance(root, name: str, seen: set) -> tuple[float | None, str]:
    """`ref_scales.T` from a shipped config, following its Hydra `defaults:` chain.

    Reading only the named file finds nothing for a config that inherits its scales, which
    is most of them: `ref_scales` lives under `custom:` in 22 of the 61 shipped configs, and
    the rest compose it in. Measured on the live monitor 2026-08-27: six running trainings
    had a beam-box MAE and NO Kelvin, so the console showed `0.381 norm.` where the repo
    rule is Kelvin. `config_flow3D_RIGNO_pvlhs` and `config_flow3D_FNO_randompath` both
    resolve T=3000 one level up, through `config_flow3D_RIGNO_randompath_onecycle` and
    `config_NS_energy_PINO_flow3D` respectively.

    Only BARE sibling names are followed. A `defaults:` entry containing `/` or `@` is a
    package-directed include from another group (`/arch/fno_cfg@arch.fno`) and is not a
    config of this name in this directory; `_self_` is the file's own position in the
    composition order, not a parent.

    Later entries win in Hydra, so the chain is walked in reverse and the first scale found
    that way is the one composition would have applied.
    """
    import yaml

    if name in seen or len(seen) > 8:
        return None, ""
    seen.add(name)
    path = root / f"{name}.yaml"
    if not path.is_file():
        return None, ""
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — bad YAML in one ancestor must not kill the lookup
        return None, ""
    if not isinstance(parsed, dict):
        return None, ""
    value, _ = kelvin_factor(parsed)
    if value is not None:
        return value, name
    for parent in reversed(parsed.get("defaults") or []):
        if not isinstance(parent, str) or parent == "_self_" or "/" in parent or "@" in parent:
            continue
        value, found_in = _named_scale_with_inheritance(root, parent, seen)
        if value is not None:
            return value, found_in
    return None, ""


def named_config_scale(config_name: str, *, because: str = "") -> tuple[float | None, str]:
    """`ref_scales.T` from the *shipped* config of that name — a weaker source.

    Second tier, used only when a model archives no config of its own. It is
    weaker because the file could have changed since the model trained, so the
    provenance string says plainly where the number came from rather than
    passing it off as the model's own record. Checked before adding: `T: 3000`
    has been the value in `config_flow3D_RIGNO_randompath.yaml` for that file's
    entire history (`git log -L`), which is what makes the fallback defensible
    for the random-path RIGNO baseline — the one
    result whose unit was otherwise unrecoverable.
    """
    name = str(config_name or "").strip()
    if not name:
        return None, ""
    try:
        import lpno

        root = _Path(lpno.__file__).parent / "conf" / "hydra"
        value, found_in = _named_scale_with_inheritance(root, name, set())
    except Exception:  # noqa: BLE001 — lpno absent, unreadable, or bad YAML
        return None, ""
    if value is None:
        return None, ""
    # The scale may have come from an ANCESTOR of the named config, so say which file it was
    # actually read from rather than implying the named one carried it.
    if found_in != name:
        name = f"{found_in}` (inherited by `{name}"
    # `because` is the *measured* reason the model's own config could not
    # answer, propagated from `kelvin_factor`. Hardcoding "this model archives
    # no config of its own" made the stored sentence false for models that DO
    # archive a config which simply has no `ref_scales` key -- 2 of the 4
    # evaluations carrying this sentence were wrong that way, and the provenance
    # sentence is precisely the field a reviewer checks against the repository.
    reason = (because or "this model archives no config of its own").rstrip(".")
    return value, (
        f"ref_scales.T={value:g} from the shipped config `{name}` — {reason}, so the scale "
        "is read from the named config, not from a copy frozen at training time"
    )


def build_metric_units(model, metrics: dict[str, Any]) -> dict[str, Any]:
    """The unit record an evaluation carries, so its numbers are self-describing.

    Stored metrics are normalised: T is divided by `ref_scales["T"]` at load
    time, so `T.rmse = 0.0136` is 40.7 K only if you know the factor. Until now
    that factor lived solely in the *model's* archived Hydra config — missing for
    19 of 122 models, including the declared 1601 baseline — which made the
    headline result's unit depend on a file in another node.

    `temperature_scale_k` is None when the scale cannot be established, and
    `kelvin` is then empty: a number without its unit is reported as
    normalised, never as a guessed Kelvin.
    """
    factor, source = kelvin_factor(model_hydra_config(model))
    if factor is None:
        try:
            config_name = str((model.get_model_metadata() or {}).get("config_name") or "")
        except Exception:  # noqa: BLE001
            config_name = ""
        factor, named_source = named_config_scale(config_name, because=source)
        if factor is not None:
            source = named_source
    flat = flatten_metrics(metrics)
    return {
        "temperature_scale_k": factor,
        "temperature_scale_source": source,
        "kelvin": to_kelvin(flat, factor),
    }


def build_metric_units_with_geometry(model, metrics, derivation_profile=None):
    """`build_metric_units` plus the geometry cell-scale record."""
    out = build_metric_units(model, metrics)
    out["geometry_scale"] = geometry_scale(derivation_profile)
    return out


#: Geometry metrics (`melt_*_um`, `*_msd_um`, `melt_volume_um3`) are computed with
#: ONE cell size taken from the z spacing and applied to all three axes
#: (`melt_geometry_stats`). That is correct only on an isotropic grid. A dataset
#: derived with `stride=[2,2,1]` has dz=10 um but dx=dy=20 um, so on those evals
#: every x/y length is 2x too small and every volume 4x too small (measured
#: 2026-08-04). The stride is stored in `derivation_profile`, but reading a number
#: correctly then requires knowing the stride, the spacing it implies, AND that the
#: metric is z-isotropic -- a three-step chain nobody follows. So the scale is
#: recorded next to the numbers instead, exactly as `temperature_scale_k` is.
GEOMETRY_ISOTROPIC_NOTE = (
    "geometry lengths use one cell size from the z spacing on all axes; "
    "on an anisotropic grid x/y lengths and volumes are under-scaled"
)


def geometry_scale(derivation_profile: dict[str, Any] | None) -> dict[str, Any]:
    """Cell-spacing record so geometry numbers are self-describing.

    `axes_isotropic` False means the stored `*_um` / `*_um3` geometry values are
    under-scaled on x and y and MUST NOT be compared against an isotropic
    (full-resolution) evaluation without rescaling.
    """
    stride = None
    if isinstance(derivation_profile, dict):
        stride = derivation_profile.get("stride")
    if not isinstance(stride, (list, tuple)) or len(stride) != 3:
        return {"stride": None, "axes_isotropic": None, "note": GEOMETRY_ISOTROPIC_NOTE}
    sx, sy, sz = (int(v) for v in stride)
    isotropic = sx == sy == sz
    out = {
        "stride": [sx, sy, sz],
        "axes_isotropic": isotropic,
        "note": GEOMETRY_ISOTROPIC_NOTE,
    }
    if not isotropic:
        # What a consumer must multiply the stored value by to get the true one.
        out["length_correction_xy"] = sx / sz if sz else None
        out["volume_correction"] = (sx * sy) / (sz * sz) if sz else None
        out["comparable_to_isotropic_evals"] = False
    else:
        out["comparable_to_isotropic_evals"] = True
    return out


def recorded_kelvin_factor(evaluation_metadata: dict[str, Any] | None, extras: dict[str, Any] | None):
    """(factor, source) recorded on the evaluation itself, or (None, "").

    Attributes first (the creating process's own record), then extras (what the
    backfill added to evaluations that predate it). Callers fall back to reading
    the model's config when this returns None.
    """
    for block in ((evaluation_metadata or {}).get("metric_units"), (extras or {}).get("metric_units")):
        if not isinstance(block, dict):
            continue
        value = block.get("temperature_scale_k")
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            return float(value), str(block.get("temperature_scale_source") or "recorded on the evaluation")
    return None, ""


def _find_key(obj: Any, key: str, depth: int = 0) -> Any:
    if depth > 6 or not isinstance(obj, dict):
        return None
    if key in obj:
        return obj[key]
    for value in obj.values():
        found = _find_key(value, key, depth + 1)
        if found is not None:
            return found
    return None


def to_kelvin(flat_metrics: dict[str, float], factor: float | None) -> dict[str, float]:
    """Temperature metrics in Kelvin. Empty when the factor is unknown."""
    if not factor:
        return {}
    return {key: value * factor for key, value in flat_metrics.items() if key in KELVIN_KEYS}


def iou_error_pct(flat_metrics: dict[str, float]) -> dict[str, float]:
    """IoU as a percentage *error*: 100 * (1 - IoU), and its spread in points."""
    out: dict[str, float] = {}
    for field in ("alpha", "fl"):
        mean = flat_metrics.get(f"{field}.iou_mean")
        if mean is not None:
            out[f"{field}.iou_err_pct"] = (1.0 - mean) * 100.0
        std = flat_metrics.get(f"{field}.iou_std")
        if std is not None:
            out[f"{field}.iou_std_pp"] = std * 100.0
    return out


def geometry_rows(flat_metrics: dict[str, float]) -> list[dict[str, Any]]:
    """Melt-pool and keyhole geometry as true/pred/abs-err rows, if measured."""
    rows: list[dict[str, Any]] = []
    for region in ("melt", "keyhole"):
        for quantity, unit in (("depth", "µm"), ("width", "µm"), ("length", "µm"), ("volume", "µm³")):
            suffix = "um3" if quantity == "volume" else "um"
            base = f"geometry.{region}_{quantity}_{suffix}"
            true, pred = flat_metrics.get(f"{base}_true"), flat_metrics.get(f"{base}_pred")
            err = flat_metrics.get(f"{base}_abs_err")
            if true is None and pred is None:
                continue
            rows.append({
                "region": region, "quantity": quantity, "unit": unit,
                "true": true, "pred": pred, "abs_err": err,
                # Percent of the true value: an absolute µm error means little
                # without the scale it is measured against.
                "rel_pct": (abs(err) / abs(true) * 100.0) if err is not None and true else None,
            })
    return rows


def rank_evaluations(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Badges and ranks per evaluation, grouped by evaluation dataset.

    `rows` need `eval_pk`, `model_uuid`, `dataset_key`, `dataset_display` and
    Kelvin `t_mae_k` / `t_rmse_k`. A model whose temperature scale is unknown
    has None there and is left out of that metric's ranking entirely — it must
    not win a Kelvin leaderboard on numbers that may not be Kelvin.
    """
    out: dict[int, dict[str, Any]] = {int(r["eval_pk"]): {"badges": [], "ranks": {}} for r in rows}
    for r in rows:
        # Not a style flag: this text is the only thing standing between a
        # training-fit number and a reader comparing it against held-out scores.
        if r.get("held_out") is False:
            n = r.get("train_overlap_n")
            out[int(r["eval_pk"])]["warning"] = (
                f"TRAINED ON {n} OF THE SCORED CASES -- training-fit, not a held-out result"
                if isinstance(n, int) else
                "SCORED ON TRAINING DATA -- training-fit, not a held-out result"
            )
    by_dataset: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(row.get("dataset_key"), []).append(row)

    for dataset_key, members in by_dataset.items():
        if dataset_key is None:
            continue
        for metric, label in (("t_mae_k", "best T MAE"), ("t_rmse_k", "best T RMSE")):
            scored = [r for r in members if isinstance(r.get(metric), (int, float)) and math.isfinite(r[metric])]
            if not scored:
                continue
            # Each model competes with its own best evaluation on this dataset.
            best_per_model: dict[Any, dict[str, Any]] = {}
            for row in scored:
                key = row.get("model_uuid") or row["eval_pk"]
                current = best_per_model.get(key)
                if current is None or row[metric] < current[metric] or (
                    row[metric] == current[metric] and int(row["eval_pk"]) < int(current["eval_pk"])
                ):
                    best_per_model[key] = row
            # Ties: the older evaluation keeps the badge, so it does not flap.
            ordered = sorted(best_per_model.values(), key=lambda r: (r[metric], int(r["eval_pk"])))
            for position, row in enumerate(ordered, start=1):
                entry = out[int(row["eval_pk"])]
                entry["ranks"][metric] = {
                    "rank": position,
                    "of": len(ordered),
                    "best": ordered[0][metric],
                    "delta": row[metric] - ordered[0][metric],
                }
            winner = ordered[0]
            display = winner.get("dataset_display") or str(dataset_key)
            out[int(winner["eval_pk"])]["badges"].append(f"{label} ({display})")
    return out
