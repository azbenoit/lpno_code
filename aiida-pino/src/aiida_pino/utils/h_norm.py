"""Supply the H_norm denominator a model TRAINED with to its evaluation.

lpno normalizes ``H_norm = (P / sqrt|V|) / denom`` with `denom` the maximum over
the training split, and since 2026-10-30 records that value as ``h_norm_denom.json`` beside
the weights (`lpno.models.pipeline.h_norm_record`). Models trained before then
carry no record, and their standalone evaluations refitted the denominator on the scored
split: every published single-alloy evaluation scored its model with H_norm scaled by
train max / scored max (0.918 to 1.445 on the test splits, up to 3.99 on one-case figure
evaluations).

The provenance graph still knows the training split. `backfill_h_norm_record` rebuilds the
record from the model's training-source simulations (their stored laser power and scan
speed, and material constants or beam radius when the model's archived config set
`h_norm_include_material` / `h_norm_include_radius`), using lpno' own
`h_norm_raw` so the two sides cannot drift apart. `PinoEvalCalculation` writes it into the
staged model folder, where the evaluation reads it as it would a training record.

One documented difference: the arraydir runtime serves P and V_scan as float32 arrays, the
NPZ store and the simulation node as float64, so a backfilled denominator can differ from
the training's own fit by float32 rounding (below 1e-7 relative, measured against the
stored previews of the published evaluations on 2026-10-30).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

#: Twin of `lpno.models.pipeline.h_norm_record.H_NORM_RECORD_NAME`
#: (tests/test_eval_supplies_training_h_norm.py pins the two equal).
H_NORM_RECORD_NAME = "h_norm_denom.json"
RECORD_FORMAT = 1
SOURCE_BACKFILL = "training split max, supplied by aiida-pino from the model's training source"

_CONFIG_PATHS = (".hydra/config.yaml", "model/.hydra/config.yaml")


def _case_manifest(attributes: dict) -> dict:
    """The case manifest as the staged stores see it (`ArrayDirSimulationStore._material_manifest`)."""
    metadata = attributes.get("simulation_metadata") or {}
    manifest = dict(metadata.get("flow3d_case_manifest") or {})
    if "material" not in manifest and isinstance(metadata.get("flow3d_material"), dict):
        manifest["material"] = metadata["flow3d_material"]
    return manifest


def case_h_norm_raw(attributes: dict, *, include_radius: bool = False, include_material: bool = False) -> float | None:
    """The H_norm numerator of one stored simulation, as `compute_h_norm_denom` reads it.

    None for a stationary case (V_scan 0), which the denominator fit skips too.
    """
    from lpno.custom_utils.utils import h_norm_raw, king_material_factor

    power = float(attributes["laser_power"])
    speed = abs(float(attributes["V_scan"]))
    if speed <= 0.0:
        return None
    radius = None
    manifest = _case_manifest(attributes) if (include_radius or include_material) else {}
    if include_radius:
        from lpno.custom_utils.npz_datasource import build_constant_path_channels

        built = build_constant_path_channels(["beam_radius"], manifest, (1,)) if manifest else {}
        if "beam_radius" in built:
            radius = float(built["beam_radius"].reshape(-1)[0])
        elif attributes.get("beam_radius") is not None:
            radius = float(attributes["beam_radius"])
        else:
            raise ValueError("h_norm_include_radius is set but the simulation carries no beam radius")
    factor = None
    if include_material:
        from lpno.custom_utils.npz_datasource import MATERIAL_CHANNEL_NAMES, _build_material_channels

        channels = _build_material_channels(MATERIAL_CHANNEL_NAMES, manifest, (1,))
        value = {name: float(channels[name].reshape(-1)[0]) for name in MATERIAL_CHANNEL_NAMES}
        factor = float(king_material_factor(rho=value["mat_rho"], cp=value["mat_cp"], k=value["mat_k"],
                                            latent=value["mat_latent"], T_l=value["mat_Tl"], A=value["mat_A"]))
    return float(h_norm_raw(power, speed, radius, material_factor=factor))


def training_split_record(member_attributes: Iterable[dict], *, include_radius: bool = False,
                          include_material: bool = False, source_uuid: str | None = None) -> dict:
    """The record lpno would have written: the maximum over the training split."""
    values = [case_h_norm_raw(attrs, include_radius=include_radius, include_material=include_material)
              for attrs in member_attributes]
    values = [v for v in values if v is not None]
    if not values:
        raise ValueError("the training split has no moving case to fit the H_norm denominator on")
    return {
        "format": RECORD_FORMAT,
        "h_norm_denom": max(values),
        "radius_aware": bool(include_radius),
        "material_aware": bool(include_material),
        "source": SOURCE_BACKFILL,
        "n_cases": len(values),
        "training_source_uuid": source_uuid,
    }


def archived_model_config(model) -> dict | None:
    """The model's archived Hydra config (``.hydra/`` or, for an imported model, ``model/.hydra/``)."""
    import yaml

    for path in _CONFIG_PATHS:
        try:
            return yaml.safe_load(model.base.repository.get_object_content(path)) or {}
        except (FileNotFoundError, OSError):
            continue
    return None


def _model_has_record(model) -> bool:
    names = {str(p) for p in model.base.repository.list_object_names()}
    if H_NORM_RECORD_NAME in names:
        return True
    try:
        return H_NORM_RECORD_NAME in {str(p) for p in model.base.repository.list_object_names("model")}
    except (FileNotFoundError, OSError, NotADirectoryError):
        return False


def training_member_uuids(model) -> tuple[list[str], str | None]:
    """The model's training simulations: its recorded training-source summary, else the source node."""
    from aiida import orm

    metadata = model.get_model_metadata() if hasattr(model, "get_model_metadata") else {}
    summary = metadata.get("training_source_summary") or {}
    source_uuid = metadata.get("training_source_uuid") or model.base.extras.get("train_dataset_uuid")
    members = [m.get("uuid") for m in (summary.get("members") or []) if isinstance(m, dict) and m.get("uuid")]
    if members:
        return members, source_uuid
    if not source_uuid:
        return [], None
    source = orm.load_node(source_uuid)
    if hasattr(source, "get_member_simulation_uuids"):
        return list(source.get_member_simulation_uuids()), source_uuid
    if hasattr(source, "iter_member_datasets"):
        uuids: list[str] = []
        for member in source.iter_member_datasets():
            if hasattr(member, "get_member_simulation_uuids"):
                uuids.extend(member.get_member_simulation_uuids())
        return uuids, source_uuid
    return [], source_uuid


def backfill_h_norm_record(model, target_dir: str | Path) -> tuple[Path | None, str]:
    """Write the training denominator into ``target_dir`` for a model that carries none.

    Returns (path, why) or (None, why not). Writes nothing when the model already carries a
    record or trained with an explicit ``custom.h_norm_denom`` (its evaluation inherits that
    from the archived config).
    """
    from aiida import orm

    if _model_has_record(model):
        return None, "the model carries its own training record"
    config = archived_model_config(model)
    if config is None:
        return None, "the model archived no .hydra/config.yaml, so its H_norm convention is unknown"
    custom: dict[str, Any] = config.get("custom") or {}
    if custom.get("h_norm_denom") is not None:
        return None, "the model trained with an explicit custom.h_norm_denom, which its evaluation inherits"
    if str(custom.get("split_mode", "standard")).lower() != "standard":
        return None, "k-fold models normalize H_norm by a reference scale, not a split maximum"
    uuids, source_uuid = training_member_uuids(model)
    if not uuids:
        return None, "the model records no training simulations"
    attributes = []
    for uuid in uuids:
        attributes.append(orm.load_node(uuid).base.attributes.all)
    record = training_split_record(
        attributes,
        include_radius=bool(custom.get("h_norm_include_radius", False)),
        include_material=bool(custom.get("h_norm_include_material", False)),
        source_uuid=source_uuid,
    )
    target = Path(target_dir) / H_NORM_RECORD_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return target, f"{record['h_norm_denom']:.10g} over {record['n_cases']} training simulations"
