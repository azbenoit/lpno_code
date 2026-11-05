"""The H_norm denominator a model trained with: recorded beside its weights, replayed at eval.

``H_norm = (P / sqrt|V|) / denom`` (King's numerator, `custom_utils.utils.h_norm_raw`), and
`denom` is the maximum of the numerator over the TRAINING split (BUG-04), unless
``custom.h_norm_denom`` fixes it. Until 2026-10-30 nothing carried the fitted value past the
training run. A standalone evaluation (`NS_Energy_laser_PINO_eval`, ``custom.data_dir`` = the
split being scored) fitted it again on the scored split, so every scored model saw H_norm
scaled by train max / scored max: 0.918 to 1.445 on the published single-alloy test splits,
and up to 3.99 on the single-case evaluations behind the prediction figures, where the scored
case's own H_norm is 1.0 by construction (found 2026-10-12 and 2026-10-30).

Training now writes `h_norm_denom.json` into ``network_dir``, beside ``u_network.0.pth`` and
``trained_grid.json`` (aiida-pino keeps the ``.json`` files of a training output in the model
node). At evaluation the denominator resolves, first match wins:

1. an explicit ``custom.h_norm_denom`` (command line, or inherited from the model's archived
   config, which is how the two-alloy models replay theirs);
2. the model's record;
3. for a model trained before the record existed, a fit over the training split named by the
   model's archived ``.hydra/config.yaml`` (``custom.data_dir``), when that directory is
   readable here;
4. otherwise the scored split, as before, under a banner saying the number is not comparable.

aiida-pino supplies (2) for older models from the provenance graph: the eval calculation
writes the record into the staged model folder (`aiida_pino.utils.h_norm`). Every evaluation
stores the denominator it used in an ``[h_norm]`` block of ``model_errors.txt``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: File name of the record; aiida-pino writes the same name (`aiida_pino.utils.h_norm`).
H_NORM_RECORD_NAME = "h_norm_denom.json"
RECORD_FORMAT = 1

SOURCE_EXPLICIT = "explicit custom.h_norm_denom"
SOURCE_RECORD = "training record"
SOURCE_RECOMPUTED = "training split, recomputed"
SOURCE_SCORED_SPLIT = "scored split"


@dataclass(frozen=True)
class HNormResolution:
    """The denominator an evaluation uses, and where it came from.

    `denom` None means the scored split (source SOURCE_SCORED_SPLIT): the pipeline then
    fits it exactly as before. `radius_aware` marks a denominator fitted with the r0 term,
    which may meet a radius-carrying numerator (see `_preprocess_var`).
    """

    denom: float | None
    source: str
    radius_aware: bool = False
    detail: str = ""


def _network_dir(run_cfg: Any) -> Path | None:
    cfg = getattr(run_cfg, "cfg", None)
    value = None
    if cfg is not None:
        try:
            from omegaconf import OmegaConf

            value = OmegaConf.select(cfg, "network_dir", default=None)
        except Exception:  # noqa: BLE001 - a plain namespace in tests
            value = getattr(cfg, "network_dir", None)
    if not value:
        value = getattr(run_cfg, "network_base_dir", None)
    if not value or str(value) in ("PATH", "."):
        return None
    return Path(str(value))


def write_training_record(run_cfg: Any, denom: float | None, *, n_cases: int | None = None,
                          network_dir: str | Path | None = None) -> Path | None:
    """Write the denominator a training run uses into its ``network_dir``.

    No-op at evaluation, where ``network_dir`` is the model being scored, and when no
    denominator is known. A read-only ``network_dir`` must not fail a healthy run, so an
    OSError only warns.
    """
    if bool(getattr(run_cfg, "eval_only", False)) or denom is None:
        return None
    network_dir = Path(str(network_dir)) if network_dir else _network_dir(run_cfg)
    if network_dir is None or str(network_dir) in ("PATH", "."):
        return None
    explicit = getattr(run_cfg, "h_norm_denom", None)
    payload = {
        "format": RECORD_FORMAT,
        "h_norm_denom": float(denom),
        "radius_aware": bool(getattr(run_cfg, "h_norm_include_radius", False)) and not explicit,
        "material_aware": bool(getattr(run_cfg, "h_norm_include_material", False)),
        "source": SOURCE_EXPLICIT if explicit else "training split max",
        "n_cases": None if n_cases is None else int(n_cases),
    }
    path = network_dir / H_NORM_RECORD_NAME
    try:
        network_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    except OSError as exc:
        print(f"[H_norm] WARNING: could not write {path}: {exc}. An evaluation of this model "
              "will have to recompute its training denominator.", flush=True)
        return None
    print(f"[H_norm] training denominator {float(denom):.10g} ({payload['source']}) "
          f"recorded in {path}", flush=True)
    return path


def find_record(network_dir: str | Path | None) -> tuple[Path | None, dict | None]:
    """The model's record: in ``network_dir``, its grandparent or its parent.

    The same three places `config_loading._find_eval_hydra_config_path` searches for the
    archived config, since an imported model keeps its files one level down (``model/``).
    """
    if not network_dir:
        return None, None
    run_dir = Path(str(network_dir))
    for candidate in (run_dir, run_dir.parent.parent, run_dir.parent):
        path = candidate / H_NORM_RECORD_NAME
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"unreadable H_norm record {path}: {exc}") from exc
            if not isinstance(payload, dict) or payload.get("h_norm_denom") is None:
                raise ValueError(f"H_norm record {path} carries no h_norm_denom")
            return path, payload
    return None, None


def _open_store(data_dir: Path):
    """A lazy store over ``data_dir`` for scalar reads only, or None for eager-only formats.

    Built directly rather than through `load_and_preprocess_data`, which would adopt the
    TRAINING data's melt window, fill temperature and beam radius into the eval's run_cfg,
    and would load an eager-format split whole.
    """
    from ...custom_utils.npz_datasource import is_consolidated_npz
    from .lazy_runtime import is_arraydir_runtime, is_npz_file_collection, load_runtime_store
    from .npz_lazy_store import NpzDirStore

    if is_arraydir_runtime(data_dir):
        if is_npz_file_collection(data_dir):
            return NpzDirStore(data_dir)
        return load_runtime_store(data_dir)
    if is_consolidated_npz(data_dir):
        return NpzDirStore(data_dir)
    return None


def recompute_training_denominator(data_dir: str | Path, *, time_as_batch: bool, include_radius: bool,
                                   include_material: bool) -> tuple[float | None, str]:
    """`compute_h_norm_denom` over every case of a training split, as training fitted it.

    Returns (denominator, "") or (None, reason). Training fits over all cases of
    ``custom.data_dir`` (`get_splits` standard mode), so this does too.
    """
    from .lazy_runtime import compute_h_norm_denom

    path = Path(str(data_dir))
    if not path.is_dir():
        return None, f"training data '{path}' is not readable here"
    try:
        store = _open_store(path)
    except Exception as exc:  # noqa: BLE001 - any failure is a "cannot", reported loudly
        return None, f"training data '{path}' could not be opened: {exc}"
    if store is None:
        return None, f"training data '{path}' is not a lazy store this check can read"
    try:
        n = int(store.sample_count())
        denom = compute_h_norm_denom(store, range(n), time_as_batch=time_as_batch,
                                     include_radius=include_radius, include_material=include_material)
    except Exception as exc:  # noqa: BLE001
        return None, f"fitting the denominator on '{path}' failed: {exc}"
    return float(denom), f"{n} cases of '{path}'"


def _archived_training_data_dir(network_dir: Path | None) -> str | None:
    """``custom.data_dir`` of the model's archived config: the split it trained on."""
    if network_dir is None:
        return None
    from omegaconf import OmegaConf

    for candidate in (network_dir / ".hydra" / "config.yaml",
                      network_dir.parent.parent / ".hydra" / "config.yaml",
                      network_dir.parent / ".hydra" / "config.yaml"):
        if candidate.is_file():
            try:
                value = OmegaConf.select(OmegaConf.load(candidate), "custom.data_dir", default=None)
            except Exception:  # noqa: BLE001
                return None
            return str(value) if value else None
    return None


def _banner(lines: list[str]) -> None:
    rule = "!" * 100
    print("\n".join([rule, *(f"!! {line}" for line in lines), rule]), flush=True)


def resolve_eval_denominator(*, network_dir: str | Path | None, explicit: float | None,
                             include_radius: bool, include_material: bool,
                             time_as_batch: bool) -> HNormResolution:
    """Resolve the H_norm denominator of an evaluation (order in the module docstring)."""
    run_dir = Path(str(network_dir)) if network_dir and str(network_dir) not in ("PATH", ".") else None
    record_path, record = find_record(run_dir)

    if explicit is not None:
        if record is not None and abs(float(record["h_norm_denom"]) - float(explicit)) > 1e-9 * abs(float(explicit)):
            _banner([f"custom.h_norm_denom={float(explicit):.10g} overrides the denominator this model "
                     f"trained with, {float(record['h_norm_denom']):.10g} ({record_path}).",
                     "Every H_norm input is scaled by their ratio, "
                     f"{float(record['h_norm_denom']) / float(explicit):.6g}. Drop the override "
                     "unless this is deliberate."])
        return HNormResolution(float(explicit), SOURCE_EXPLICIT)

    if record is not None:
        if bool(record.get("material_aware", False)) != bool(include_material):
            raise ValueError(
                f"H_norm record {record_path} was fitted with material_aware="
                f"{record.get('material_aware')}, but this evaluation runs "
                f"h_norm_include_material={include_material}. The numerator and denominator would "
                "use different conventions, which mis-scales every H_norm input silently.")
        if bool(record.get("radius_aware", False)) and not include_radius:
            raise ValueError(
                f"H_norm record {record_path} was fitted radius-aware, but this evaluation runs "
                "h_norm_include_radius=false, so the r0 term would not cancel.")
        denom = float(record["h_norm_denom"])
        print(f"[H_norm] eval denominator {denom:.10g}: the model's {SOURCE_RECORD} ({record_path})",
              flush=True)
        return HNormResolution(denom, SOURCE_RECORD, radius_aware=bool(record.get("radius_aware", False)),
                               detail=str(record_path))

    data_dir = _archived_training_data_dir(run_dir)
    reason = "the model's archived config names no custom.data_dir"
    if data_dir:
        denom, reason = recompute_training_denominator(
            data_dir, time_as_batch=time_as_batch, include_radius=include_radius,
            include_material=include_material)
        if denom is not None:
            print(f"[H_norm] eval denominator {denom:.10g}: no {H_NORM_RECORD_NAME} in the model, "
                  f"recomputed on its training split ({reason})", flush=True)
            return HNormResolution(denom, SOURCE_RECOMPUTED, radius_aware=bool(include_radius), detail=reason)

    _banner([f"NO TRAINING NORMALIZATION FOR H_norm: the model carries no {H_NORM_RECORD_NAME} and "
             f"its training split cannot be read ({reason}).",
             "The denominator is refitted on the SCORED split, so H_norm reaches the network scaled by "
             "(training max / scored max), which is 1.0 only by coincidence.",
             "Metrics from this evaluation are not comparable with the model's training. Pass "
             "++custom.h_norm_denom=<training max>, or evaluate through aiida-pino, which supplies it."])
    return HNormResolution(None, SOURCE_SCORED_SPLIT, detail=reason)


def metric_block(run_cfg: Any) -> dict[str, dict[str, float]]:
    """The ``[h_norm]`` block an evaluation stores beside its metrics, or {} outside eval.

    ``denom`` is the denominator the network saw (NaN when it was fitted on the scored split
    and is not known here), ``scored_split`` is 1.0 when that fit was on the scored split.
    """
    source = getattr(run_cfg, "h_norm_denom_source", None)
    if not source or not bool(getattr(run_cfg, "eval_only", False)):
        return {}
    denom = getattr(run_cfg, "h_norm_denom", None)
    if denom is None:  # the scored-split fit: lazy path, else the eager stash
        denom = getattr(run_cfg, "h_norm_denom_scored_fit", None)
    if denom is None:
        denom = (getattr(run_cfg, "laser_params", None) or {}).get("h_norm_denom")
    return {"h_norm": {
        "denom": float(denom) if denom is not None else float("nan"),
        "scored_split": 1.0 if source == SOURCE_SCORED_SPLIT else 0.0,
    }}
