"""Record the input-field settings a training run actually used, beside its weights.

`input_fields.json` sits in ``network_dir`` with ``u_network.0.pth``, ``h_norm_denom.json`` and ``trained_grid.json``
(aiida-pino keeps every ``.json`` of a training output in the model node). It holds the EFFECTIVE values of the
settings that change the values of the physics-derived input channels, resolved the way the data pipeline resolves
them (npz_lazy_store's helpers: config, then environment overrides such as PINO_POWER_WEIGHTED_CONDUCTION, then the
code default):

  - conduction_source_sigma_factor: the conduction_field source width as a multiple of the beam radius (1.0 is the
    original channel, sigma = r0; 0.5 matches the solver's Gaussian beam);
  - power_weighted_conduction / power_weighted_dose: whether those channels are scaled by P / P_ref.

Before this record, a model whose stored config left a key out depended on the code default for its meaning, and an
environment override left no trace in the config at all. The record makes the input fields a model was trained on
readable from the model alone, whatever later defaults become.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .h_norm_record import _network_dir

INPUT_FIELDS_RECORD_NAME = "input_fields.json"
RECORD_FORMAT = 1


def input_field_settings(run_cfg: Any) -> dict:
    """The effective input-field settings of `run_cfg`, as the data pipeline applies them."""
    from .npz_lazy_store import (conduction_source_sigma_factor, power_weighted_conduction_enabled,
                                 power_weighted_dose_enabled)

    keys = [str(k) for k in (getattr(run_cfg, "str_fno_input_keys", None) or [])]
    return {
        "format": RECORD_FORMAT,
        "input_keys": keys,
        "conduction_field": "conduction_field" in keys,
        "conduction_source_sigma_factor": float(conduction_source_sigma_factor(run_cfg)),
        "power_weighted_conduction": bool(power_weighted_conduction_enabled(run_cfg)),
        "power_weighted_dose": bool(power_weighted_dose_enabled(run_cfg)),
    }


def write_input_fields_record(run_cfg: Any, *, network_dir: str | Path | None = None) -> Path | None:
    """Write `input_fields.json` into a training run's ``network_dir``. No-op at evaluation (``network_dir`` is then
    the model being scored); a read-only ``network_dir`` only warns."""
    if bool(getattr(run_cfg, "eval_only", False)):
        return None
    network_dir = Path(str(network_dir)) if network_dir else _network_dir(run_cfg)
    if network_dir is None or str(network_dir) in ("PATH", "."):
        return None
    payload = input_field_settings(run_cfg)
    path = network_dir / INPUT_FIELDS_RECORD_NAME
    try:
        network_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    except OSError as exc:
        print(f"[input-fields] WARNING: could not write {path}: {exc}", flush=True)
        return None
    print(f"[input-fields] conduction sigma factor {payload['conduction_source_sigma_factor']:g}, "
          f"power-weighted conduction {payload['power_weighted_conduction']}, power-weighted dose "
          f"{payload['power_weighted_dose']} recorded in {path}", flush=True)
    return path
