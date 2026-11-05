"""Material identity as queryable values, not an absolute path and a row count.

`run_config.material.file` is a path string, and `case_manifest.json` records
only `table_rows: {surface_tension_n_m: 23}` — a count. So "which σ arm is this"
could not be answered by a query, and two runs differing only in their surface
tension table looked identical.

The values are not lost: the rendered `prepin.*` is retained in the
`run_artifacts` FolderData and carries the full `#sigma(1)` table. This reads
them back out and turns them into extras — a fingerprint that separates arms,
plus the scalars and the table ranges that say *how* they differ. Reported as a
surfacing gap rather than lost provenance, and that is exactly what it was.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

#: `    rhof=4420,` — a scalar assignment in a prepin namelist.
_SCALAR = re.compile(r"^\s*([a-z][a-z0-9_]*)\s*=\s*([-+0-9.eE]+)\s*,?\s*$", re.M)

#: `#sigma(1)` ... rows of `T value` ... `#end sigma(1)` (or the next `#` block).
_TABLE = re.compile(r"^#(?!end\b)([a-z][a-z0-9_]*)\(\d+\)\s*$(.*?)(?=^#)", re.M | re.S)

#: Scalars worth lifting: density, specific heat, conductivity, and the
#: calibration knobs the campaign moved.
SCALAR_KEYS = ("rhof", "cv1", "thc1", "tsdrg", "pv0", "tvexp", "cbmengy", "ereflct")


def _rows(block: str) -> list[tuple[float, float]]:
    out = []
    for line in block.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            out.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return out


def material_identity(prepin_text: str) -> dict[str, Any]:
    """Scalars, table summaries and a fingerprint, from a rendered prepin."""
    text = prepin_text or ""
    scalars = {}
    for key, value in _SCALAR.findall(text):
        if key in SCALAR_KEYS and key not in scalars:
            try:
                scalars[key] = float(value)
            except ValueError:
                continue

    tables: dict[str, dict[str, Any]] = {}
    for name, block in _TABLE.findall(text):
        rows = _rows(block)
        if not rows:
            continue
        values = [value for _t, value in rows]
        digest = hashlib.sha1(
            ";".join(f"{t:.6g},{v:.6g}" for t, v in rows).encode("utf-8")
        ).hexdigest()[:12]
        tables[name] = {
            "rows": len(rows),
            "hash": digest,
            "min": min(values),
            "max": max(values),
            "t_min": min(t for t, _v in rows),
            "t_max": max(t for t, _v in rows),
        }

    fingerprint = hashlib.sha1(
        repr(sorted(scalars.items())).encode("utf-8")
        + repr(sorted((k, v["hash"]) for k, v in tables.items())).encode("utf-8")
    ).hexdigest()[:12]
    return {"scalars": scalars, "tables": tables, "fingerprint": fingerprint}


def material_extras(prepin_text: str, material_block: dict | None = None) -> dict[str, Any]:
    """Flat, queryable form of `material_identity`.

    `material_block` is `run_summary.manifest.material`, and it is the
    authoritative half: 15 structured properties plus the optical block, as
    opposed to the seven scalars `SCALAR_KEYS` happened to name. Hand-picking
    was wrong and measurably so — 446 of 1115 runs shared a fingerprint with a
    materially different run, including t2n195_ctrl and
    t2n195_tv3425, whose only difference is
    `vaporization_temperature_k` 3520 -> 3425 and which the solver ran for 4315
    vs 4442 cycles. A control and its arm, declared identical.

    The prepin still supplies the tabulated properties, which the manifest
    reduces to a row count.
    """
    identity = material_identity(prepin_text)
    properties = dict((material_block or {}).get("properties") or {})
    optical = dict((material_block or {}).get("optical") or {})
    if not identity["scalars"] and not identity["tables"] and not properties:
        return {}

    extras: dict[str, Any] = {}
    for key, value in identity["scalars"].items():
        extras[f"material_{key}"] = value
    for name, summary in identity["tables"].items():
        extras[f"material_{name}_hash"] = summary["hash"]
        extras[f"material_{name}_rows"] = summary["rows"]
        extras[f"material_{name}_min"] = summary["min"]
        extras[f"material_{name}_max"] = summary["max"]

    name = str((material_block or {}).get("name") or "").strip()
    if name:
        extras["material_name"] = name
    for key, value in properties.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            extras[f"material_{key}"] = float(value)

    # Every property, every optical value, every table's contents -- so a
    # fingerprint match means the materials really are the same.
    extras["material_fingerprint"] = hashlib.sha1(
        json.dumps(
            {
                "properties": properties,
                "optical": optical,
                "scalars": identity["scalars"],
                "tables": {k: v["hash"] for k, v in identity["tables"].items()},
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return extras


def stamp_from_artifacts(node, folder, material_block: dict | None = None) -> dict[str, Any]:
    """Read the retained prepin off a run's artifacts and stamp the extras."""
    if folder is None:
        return {}
    try:
        names = [n for n in folder.base.repository.list_object_names() if n.startswith("prepin.")]
    except Exception:  # noqa: BLE001
        return {}
    if not names:
        return {}
    # The whole tail is guarded, not just the reads. This runs inside the
    # daemon's parser for every Flow3DRunCalculation, and an unhandled exception
    # there fails the parse and loses the run's artifacts -- a catastrophic
    # trade for a metadata nicety. Measured safe on 38 recent prepins (0 raised,
    # slowest 4.3 ms), but "happens to be safe today" is not a guarantee.
    try:
        text = folder.base.repository.get_object_content(names[0])
        extras = material_extras(text, material_block)
        for key, value in extras.items():
            if node.base.extras.get(key, None) is None:
                node.base.extras.set(key, value)
        return extras
    except Exception as exc:  # noqa: BLE001 - never fail a parse over material metadata
        print(f"[material] identity not recorded for {getattr(node, 'pk', '?')}: {exc}")
        return {}
