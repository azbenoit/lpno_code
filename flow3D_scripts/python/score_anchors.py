#!/usr/bin/env python3
"""Score exported runs against the published anchors in `validation/anchors/`.

The join between a run and its measured target used to live in throwaway scripts,
so the numbers in a write-up were not regenerable by a reader. This is that join,
tracked: it reads an anchor file, finds the runs whose conditions match, and prints
one row per condition plus a summary that always reports the **mean signed error**
alongside the pass count -- a one-sign offset across a ladder is a finding even
when every point is inside the band.

    python3 python/score_anchors.py validation/anchors/galiegue2025_ti64_targets.json \\
        --runs runs --sim-kind fusion

Matching is on power, speed AND spot radius. Spot size is the dominant variable at
fixed power, so matching on power and speed alone silently compares different
experiments (`--r0-tol` relaxes it if a run's radius is only nominally equal).

Metric definitions and the acceptance band come from `melt_metrics.py`; the band's
derivation from the sources' own scatter is in `validation/ACCEPTANCE.md`. Fusion
and vapour-depression depths are different physical surfaces: `--sim-kind` states
which one the runs are being read as, and a mismatch against the anchor's
`depth_kind` raises rather than silently comparing.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import melt_metrics as mm  # noqa: E402


#: AiiDA profile consulted for runs whose local export was reclaimed.
AIIDA_PROFILE_ENV = "F3D_AIIDA_PROFILE"
AIIDA_PROFILE_DEFAULT = None  # the configured default profile


@dataclass(frozen=True)
class Case:
    """One scoreable run, whether or not its directory still exists.

    `case_dir` is None for a run that lives only in AiiDA. Reclaiming disk has taken
    whole run trees, not just their `npz/` -- the `ti64_anchorbox_lhs150_20260830`
    batch went entirely on 2026-09-04, taking the manifests of the anchor-condition
    replicates with it -- so "the manifest survives" is not a safe assumption and the
    conditions have to be able to come from the node too.
    """
    name: str
    manifest: Mapping[str, Any]
    case_dir: Path | None = None
    uuid: str | None = None


@functools.lru_cache(maxsize=1)
def _aiida_manifests() -> dict[str, tuple[str, dict[str, Any], str]]:
    """``{case_name: (uuid, case_manifest, source_dir)}`` for simulations AiiDA can describe.

    The node stores the whole FLOW-3D case manifest under
    `simulation_metadata.flow3d_case_manifest`, so a run whose directory is gone can
    still supply its power, speed, spot radius and material.

    A name that resolves to more than one node is DROPPED rather than guessed: 67 of
    them exist, every pair with a different content hash, because they are re-solves
    spanning physics eras. Scoring the wrong era against a measurement is worse than
    scoring nothing.
    """
    try:
        from aiida import load_profile
        from aiida.orm import Node, QueryBuilder

        load_profile(os.environ.get(AIIDA_PROFILE_ENV, AIIDA_PROFILE_DEFAULT))
        rows = QueryBuilder().append(
            Node, filters={"node_type": {"like": "%PinoSimulationData%"}},
            project=["extras.case_name", "uuid",
                     "attributes.simulation_metadata.flow3d_case_manifest",
                     "attributes.simulation_metadata.source_uri",
                     "attributes.simulation_metadata.source_npz_realpath"]).all()
    except Exception:  # noqa: BLE001 - not installed, no profile, DB down: all fine
        return {}
    seen: dict[str, list[tuple[str, Any, str]]] = {}
    for name, uuid, manifest, uri, realpath in rows:
        if name and isinstance(manifest, dict):
            seen.setdefault(str(name), []).append(
                (str(uuid), manifest, _source_dir(uri, realpath)))
    return {name: entries[0] for name, entries in seen.items() if len(entries) == 1}


def _source_dir(uri: Any, realpath: Any) -> str:
    """The run DIRECTORY a node was ingested from, or "" when it records none.

    `source_uri` is `file:///...run_dir`; `source_npz_realpath` is
    `.../run_dir/npz/<case>.npz`, two levels deeper. All 5428 PinoSimulationData nodes
    carry at least one (measured 2026-09-09), so this is a scoping key and not a guess.
    """
    for raw in (uri, realpath):
        if not isinstance(raw, str) or not raw:
            continue
        path = Path(raw[7:] if raw.startswith("file://") else raw)
        # Two recorded shapes, measured on the live store: a run DIRECTORY
        # (`file:///...runs/abx_121_lhs`) and, for the ti64_random_keyhole150-era
        # imports, the NPZ FILE itself. Normalise both to the directory, stepping out
        # of a `npz/` subdirectory when there is one.
        if path.suffix == ".npz":
            path = path.parent
            if path.name == "npz":
                path = path.parent
        return str(path)
    return ""


@functools.lru_cache(maxsize=1)
def _aiida_index() -> dict[str, list[str]]:
    """``{case_name: [uuid, ...]}`` for ingested FLOW-3D simulations; ``{}`` if absent.

    Keyed on ``extras.case_name`` rather than the node label: 1565 of 2223 named
    simulations do not contain their own case name in their label, so label matching
    silently misses most of them.

    The import is deliberately inside the function. This module is imported by
    `tests/test_score_anchors.py` at module scope and the documented suite runs under
    the plain `python3`, which has no `aiida` installed -- a top-level import would
    fail all 55 of those tests at collection. Offline is a supported mode: this is a
    validation tool a reader must be able to run without a database, so every failure
    to reach AiiDA degrades to "no index" rather than raising.
    """
    try:
        from aiida import load_profile
        from aiida.orm import Node, QueryBuilder

        load_profile(os.environ.get(AIIDA_PROFILE_ENV, AIIDA_PROFILE_DEFAULT))
        rows = QueryBuilder().append(
            Node, filters={"node_type": {"like": "%PinoSimulationData%"}},
            project=["extras.case_name", "uuid"]).all()
    except Exception:  # noqa: BLE001 - not installed, no profile, DB down: all fine
        return {}
    index: dict[str, list[str]] = {}
    for name, uuid in rows:
        if name:
            index.setdefault(str(name), []).append(str(uuid))
    return index


@contextlib.contextmanager
def _load_aiida_payload(uuid: str) -> Iterator[Any]:
    """Open a simulation's stored ``payload.npz`` as the compact export it is.

    `npz_file` storage keeps the original per-case NPZ verbatim, so this mapping is
    byte-identical to what `np.load` gives for a local export and `melt_metrics`
    needs no conversion.

    A context manager because `repository.open` is one, and `np.load` is LAZY: the
    returned `NpzFile` decompresses a member on first access, so the underlying
    handle has to stay open for as long as the caller reads from it.

    Never `node.get_array()` here. That routes through the canonicaliser, which
    returns `(T, 1, nx, ny, nz)` with x/y/z broadcast to full grids, while
    `melt_metrics` indexes 1-D coordinate vectors -- it would read a plausible,
    wrong geometry rather than fail.
    """
    import numpy as np
    from aiida import load_profile, orm

    load_profile(os.environ.get(AIIDA_PROFILE_ENV, AIIDA_PROFILE_DEFAULT))
    node = orm.load_node(uuid=uuid)
    with node.base.repository.open("payload.npz", "rb") as handle:
        with np.load(handle) as data:
            yield data


@contextlib.contextmanager
def open_case_arrays(case: "Case | Path", *, aiida: bool = True) -> Iterator[Any]:
    """This case's exported arrays: the local `npz/` first, then AiiDA. None if neither.

    Disk wins because it is the same bytes without a database round trip, and the
    AiiDA index is not even built when a local export is present.

    **An ambiguous case name yields nothing rather than a guess.** 67 reclaimed case
    names resolve to more than one node and every one of those pairs has a different
    content hash -- they are re-solves spanning physics eras. Taking whichever the
    query returned last would silently score a different era's simulation against the
    same measurement.

    Accepts a bare `Path` as well, for callers that only have a directory.
    """
    import numpy as np

    if isinstance(case, Path):
        case = Case(name=case.name, manifest={}, case_dir=case)
    if case.case_dir is not None:
        local = sorted(case.case_dir.glob("npz/*.npz"))
        if local:
            with np.load(local[0]) as data:
                yield data
            return
    if not aiida:
        yield None
        return
    uuid = case.uuid
    if uuid is None:
        uuids = _aiida_index().get(case.name) or []
        if len(uuids) != 1:
            yield None
            return
        uuid = uuids[0]
    try:
        with _load_aiida_payload(uuid) as data:
            yield data
    except Exception:  # noqa: BLE001 - an unreadable node must not abort the sweep
        yield None


@dataclass(frozen=True)
class Summary:
    n: int
    n_within_band: int
    mean_signed_error_pct: float
    systematic: bool


#: Anchor files this tool deliberately does not read, and why. Kept explicit rather
#: than silently skipped: a scorer that quietly ignores an anchor looks like an
#: anchor with no matching runs. Enforced by tests/test_score_anchors.py.
UNSUPPORTED_SCHEMAS = {
    "amb718_targets.json":
        "keyed by string tuples '(power_W, speed_mm_s, spot_um)' rather than a list "
        "of points, and the reproducibility notebook indexes it by those keys -- "
        "converging its schema would break that consumer",
    "wang_cocr_targets.json":
        "`conditions` holds two named sweeps (`power_series`, `speed_series`), each "
        "with its own `_fixed` block giving the parameter held constant plus a "
        "`points` list -- so the conditions are only complete once `_fixed` is merged "
        "into each point, which this reader does not do yet",
}


def _first(d: Mapping[str, Any], *names: str) -> Any:
    """First present key among alternative spellings.

    The anchor files were written at different times and disagree: `power_w` vs
    `power_W`, `speed_m_s` vs `speed_m_per_s`.
    """
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return None


#: Substrings that identify an alloy family, checked in order. Used on BOTH the
#: anchor's group/material text and the run's material name and folder, so a run is
#: never scored against a different alloy's measurement. gan_points.json alone holds
#: Ti64, SS316 and Al6061, and matching on (power, speed, spot) let an SS316 run
#: match a Gan point on 2026-07-30.
_ALLOY_TOKENS = (
    ("ti64", ("ti64", "ti6al4v", "ti-6al-4v", "imi318", "imi 318", "ti6,", "ti6 ",
              "titanium alloy ti al v")),
    ("ss316", ("ss316", "316l", "316 l", "steel 316")),
    ("ss304", ("ss304", "304l", "304 l", "steel 304")),
    ("al6061", ("al6061", "6061")),
    ("in718", ("in718", "inconel718", "inconel 718", "718")),
    ("in625", ("in625", "inconel625", "inconel 625", "625")),
    ("cocr", ("cocr", "co-cr", "cocrmo")),
    ("pure_copper", ("pure_copper", "copper")),
)


def alloy_key(text: str | None) -> str | None:
    """Canonical alloy family for a group label, material name or material path.

    Returns None when nothing matches, so an unlabelled side never blocks a match --
    a wrong block is as damaging as a wrong match, and this is a guard not a filter.
    """
    if not text:
        return None
    low = str(text).lower()
    for key, tokens in _ALLOY_TOKENS:
        if any(tok in low for tok in tokens):
            return key
    return None


# Anchor files whose sources' licences do not permit redistribution (validation/anchors/README.md).
NOT_DISTRIBUTED = {
    "galiegue2025_ti64_targets.json": "Galiegue et al. 2025 (Sci. Rep. 15, 43243) is CC BY-NC-ND 4.0",
    "hawk304l_targets.json": "Hawk et al. 2022 (Weld. World 66, 895-906) is a subscription article",
}


def load_anchor(path: Path) -> dict[str, Any]:
    """Read an anchor file, merging a `<stem>.meta.json` sidecar if one exists.

    Some anchor files are bare JSON lists (`gan_points.json`) and so cannot carry
    `depth_kind` or provenance themselves, and existing consumers read them as
    lists -- wrapping them would break those. The sidecar keeps the declaration in
    tracked data rather than leaving it to whoever runs the scorer.
    """
    if not Path(path).is_file() and Path(path).name in NOT_DISTRIBUTED:
        raise FileNotFoundError(
            f"{Path(path).name} is not distributed with this repository: "
            f"{NOT_DISTRIBUTED[Path(path).name]}. Build it from the publication as "
            f"validation/anchors/README.md describes and place it at {path}.")
    raw = json.loads(Path(path).read_text())
    anchor: dict[str, Any] = {"points": raw} if isinstance(raw, list) else dict(raw)
    side = Path(path).with_suffix("").with_suffix(".meta.json") \
        if Path(path).suffixes[-2:] == [".meta", ".json"] else \
        Path(str(Path(path).with_suffix("")) + ".meta.json")
    if side.is_file():
        meta = json.loads(side.read_text())
        for k, v in meta.items():
            anchor.setdefault(k, v)
    return anchor


def normalize_points(anchor: Mapping[str, Any], *,
                     series: str | None = None) -> list[dict[str, Any]]:
    """Flatten an anchor file to a uniform list of measured points.

    Drops points with no measured depth: several anchor tables carry 0.0 for
    conditions that produced no melt, and a zero target cannot be scored against.

    Handles two container shapes: a flat list of points, and a mapping of named
    series to lists of rows (one series per material variant). For the latter each
    point carries its series in `label`, because variants at identical process
    conditions can differ by 2x and must not be pooled. `series=` selects one.

    **Stationary spot welds are represented as speed 0.** A dwell case sets
    `laser.dwell_time_s` rather than `scan_speed_m_per_s`, so its run already
    reports speed 0 and the match works. This is only unambiguous while the dwell
    time is constant across the anchor's conditions -- it is (10 ms for every Hawk
    row) -- otherwise dwell would have to be matched on as well.
    """
    beam = anchor.get("beam") or {}
    default_kind = anchor.get("depth_kind")
    beam_r0 = _first(beam, "r0_m", "beam_radius_m")
    if beam_r0 is None and _first(beam, "r0_um", "spot_value_um") is not None:
        beam_r0 = float(_first(beam, "r0_um", "spot_value_um")) * 1e-6
    if beam_r0 is None and _first(beam, "spot_diameter_um") is not None:
        beam_r0 = float(_first(beam, "spot_diameter_um")) * 1e-6 / 2.0

    container = anchor.get("points") or anchor.get("conditions") or []
    raw: list[Mapping[str, Any]] = []
    if isinstance(container, Mapping):
        for name, rows in container.items():
            if not isinstance(rows, list):
                continue                       # `_source`, `_note` and friends
            if series is not None and name != series:
                continue
            for r in rows:
                if isinstance(r, Mapping):
                    raw.append({**r, "_series": name})
    else:
        raw = list(container)

    out: list[dict[str, Any]] = []
    for p in raw:
        depth = _first(p, "depth_um")
        if depth is None or float(depth) <= 0.0:
            continue
        r0 = _first(p, "r0_m", "r0")
        if r0 is None and _first(p, "r0_um") is not None:
            r0 = float(_first(p, "r0_um")) * 1e-6
        if r0 is None:
            r0 = beam_r0
        speed = _first(p, "speed_m_s", "speed_m_per_s", "V")
        if speed is None:
            speed = 0.0        # stationary spot weld: no scanning, see the docstring
        out.append({
            "power_w": float(_first(p, "power_w", "power_W", "P")),
            "speed_m_s": float(speed),
            "r0_m": float(r0) if r0 is not None else None,
            "depth_um": float(depth),
            "width_um": (float(_first(p, "width_um")) if _first(p, "width_um") else None),
            "depth_kind": p.get("depth_kind") or default_kind,
            "label": p.get("_series") or p.get("set") or p.get("label"),
            "alloy": alloy_key(p.get("material") or p.get("group") or anchor.get("material")),
        })
    return out


class AmbiguousAnchor(ValueError):
    """Raised when one run's conditions match several measured variants.

    Hawk reports four 304L variants at identical power, dwell and spot whose depths
    differ by up to 2x, so "the first match" is an arbitrary choice that changes the
    verdict. The caller must name the variant instead.
    """


def filter_by_r0(points: Sequence[Mapping[str, Any]], r0_um: float | None,
                 tol_um: float = 2.0) -> list[dict[str, Any]]:
    """Keep only points at one beam radius.

    An anchor can hold a fixed-spot power ladder and a defocus spot-size series at
    once; a summary over both mixes two different experiments. Pinning the radius is
    what turns "the Galiegue anchor" into "the manuscript ladder".
    """
    if r0_um is None:
        return [dict(p) for p in points]
    return [dict(p) for p in points
            if p.get("r0_m") is not None and abs(p["r0_m"] * 1e6 - r0_um) <= tol_um]


def match_points(run: Mapping[str, float], points: Sequence[Mapping[str, Any]], *,
                 p_rtol: float = 0.01, v_tol: float = 5e-4,
                 r0_tol: float = 2e-6) -> list[dict[str, Any]]:
    """Every measured point at this run's conditions.

    Never falls back to the nearest point: a "closest match" silently scores a run
    against an experiment it did not reproduce.

    ``p_rtol`` is RELATIVE (default 1%), because these papers quote power to about
    that and an absolute watt tolerance is the wrong shape: 0.5 W is 0.1% at 456 W
    but 5% at 10 W. A CoCr case built at 120.0 W was silently dropped against a
    121 W measurement under the old absolute 0.5 W guard.
    """
    out = []
    for p in points:
        if abs(p["power_w"] - run["power_w"]) > p_rtol * max(p["power_w"], 1.0):
            continue
        if abs(p["speed_m_s"] - run["speed_m_s"]) > v_tol:
            continue
        if p["r0_m"] is not None and run.get("r0_m") is not None:
            if abs(p["r0_m"] - run["r0_m"]) > r0_tol:
                continue
        if p.get("alloy") and run.get("alloy") and p["alloy"] != run["alloy"]:
            continue
        out.append(dict(p))
    return out


def match_point(run: Mapping[str, float], points: Sequence[Mapping[str, Any]],
                **kw: Any) -> dict[str, Any] | None:
    """The single measured point at this run's conditions, or None.

    Raises AmbiguousAnchor rather than choosing among several variants.
    """
    got = match_points(run, points, **kw)
    if len(got) > 1:
        labels = ", ".join(str(p.get("label")) for p in got)
        raise AmbiguousAnchor(
            f"run at P={run['power_w']:g} V={run['speed_m_s']:g} matches "
            f"{len(got)} measured variants ({labels}); name one with series=")
    return got[0] if got else None


def score_row(*, sim_depth_um: float, sim_width_um: float,
              point: Mapping[str, Any], sim_kind: str,
              band: float = mm.DEFAULT_BAND) -> dict[str, Any]:
    """Signed errors for both observables, refusing a cross-kind depth comparison."""
    d = mm.score_point(sim_depth_um=sim_depth_um, measured_depth_um=point["depth_um"],
                       sim_kind=sim_kind, measured_kind=point["depth_kind"], band=band)
    row: dict[str, Any] = {
        "sim_depth_um": sim_depth_um, "measured_depth_um": point["depth_um"],
        "depth_err_pct": d.error_pct, "depth_in_band": d.within_band,
        "width_err_pct": None, "width_in_band": None,
        "depth_kind": point["depth_kind"], "label": point.get("label"),
    }
    if point.get("width_um"):
        w = mm.score_point(sim_depth_um=sim_width_um, measured_depth_um=point["width_um"],
                           sim_kind=sim_kind, measured_kind=point["depth_kind"], band=band)
        row["width_err_pct"] = w.error_pct
        row["width_in_band"] = w.within_band
        row["measured_width_um"] = point["width_um"]
        row["sim_width_um"] = sim_width_um
    return row


def summarize(rows: Sequence[Mapping[str, Any]], observable: str) -> Summary:
    """Pass count AND mean signed error; `systematic` flags a one-sign offset."""
    key, band_key = f"{observable}_err_pct", f"{observable}_in_band"
    errs = [r[key] for r in rows if r.get(key) is not None]
    within = sum(1 for r in rows if r.get(band_key))
    L = mm.ladder_summary(errs)
    return Summary(len(errs), within, L.mean_signed_error_pct, L.systematic)


def recipe_fingerprint(case_dir: Path) -> str:
    """The prepin parameters that identify the physics recipe.

    Pooling runs from different recipes into one summary is what produced the
    2026-07-30 retractions: a "hybrid" arm (`aplume=0.2` with a pinned
    `tv1=3123.15`) was scored and reported as if it were the shipped recipe. The
    fingerprint is printed per row and a mixed set is called out.

    `aplume` and `tv1` alone are NOT enough: the standing recipe and the rejected
    measured-absorptance arm share both and differ only in `ereflct`/`iereflct`
    (0.2169 with the tabular eps(T) table vs 0.2017 scalar), so a summary pooled a
    rejected arm into the standing-recipe ladder until 2026-07-30. Those two are the
    same pair that exposed the straight-path train/test physics split.
    """
    import re

    for f in sorted(case_dir.glob("prepin.*")):
        try:
            txt = f.read_text(errors="ignore")
        except OSError:
            continue
        vals = []
        for key in ("aplume", "tv1", "ereflct", "iereflct"):
            m = re.search(rf"\b{key}\s*=\s*([-0-9.eE+]+)", txt)
            if m:
                vals.append(f"{key}={m.group(1)}")
        if vals:
            return ",".join(vals)
    return "unknown"


def recipes_present(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Distinct recipe fingerprints among scored rows."""
    return sorted({str(r.get("recipe", "unknown")) for r in rows})


def duplicate_conditions(rows: Sequence[Mapping[str, Any]]) -> dict[tuple, list[str]]:
    """Conditions matched by more than one run, keyed by (power, speed, r0, variant).

    Resolution variants and reruns are legitimate; pooling them into one summary is
    not, because it counts N runs at one condition as N independent points.
    """
    seen: dict[tuple, list[str]] = {}
    for r in rows:
        key = (round(float(r["power_w"]), 3), round(float(r["speed_m_s"]), 6),
               round(float(r["r0_m"]), 9), r.get("label"))
        seen.setdefault(key, []).append(str(r.get("case")))
    return {k: v for k, v in seen.items() if len(v) > 1}


def conditions_from_manifest(doc: Mapping[str, Any]) -> dict[str, float] | None:
    """Power, speed and spot radius from a case-manifest document.

    Split out from `run_conditions` so the same reading serves a manifest on disk and
    the copy AiiDA stores on the simulation node (`simulation_metadata.
    flow3d_case_manifest`). A reclaimed run has no directory left to read, but the
    node kept the whole document.
    """
    try:
        laser = doc["input"]["laser"]
    except (KeyError, TypeError):
        return None
    mat = doc.get("material") or {}
    name = mat.get("name") if isinstance(mat, dict) else mat
    path = (doc.get("input", {}).get("material") or {})
    path = path.get("file") if isinstance(path, dict) else path
    if "power_w" not in laser or "beam_radius_m" not in laser:
        # Independent beams carry their own power and path, leaving no top-level
        # scalar, so the run has no single condition to match an anchor point
        # against. Skip it. Summing to a total would false-match a 2 x 120 W case
        # to a 240 W single-track measurement.
        return None
    return {"power_w": float(laser["power_w"]),
            "speed_m_s": float(laser.get("scan_speed_m_per_s") or 0.0),
            "r0_m": float(laser["beam_radius_m"]),
            "alloy": alloy_key(name) or alloy_key(path)}


def run_conditions(case_dir: Path) -> dict[str, float] | None:
    """Power, speed and spot radius of a completed run, from its own manifest."""
    f = case_dir / "case_manifest.json"
    if not f.is_file():
        return None
    try:
        doc = json.loads(f.read_text())
    except ValueError:
        return None
    return conditions_from_manifest(doc)


def _laser_off_from_manifest(doc: Mapping[str, Any]) -> float | None:
    try:
        return doc["derived"]["laser_off_time_s"]
    except Exception:  # noqa: BLE001 - absent or old manifest; fall back to the heuristic
        return None


def laser_off_s(case_dir: Path) -> float | None:
    f = case_dir / "case_manifest.json"
    try:
        return _laser_off_from_manifest(json.loads(f.read_text()))
    except Exception:  # noqa: BLE001 - absent or old manifest; fall back to the heuristic
        return None


def iter_cases(runs: Path, *, aiida: bool = True) -> Iterator[Case]:
    """Every scoreable run: manifests on disk first, then AiiDA-only survivors.

    Enumerating by `npz/*.npz` made the scorer blind to a whole population -- a case
    that solved and then had its export reclaimed looked exactly like one that never
    ran. Enumerating by manifest fixes the common case, but not the harder one: disk
    reclamation has removed entire run trees, so for those the DATABASE is the only
    record that the case ever existed, and the node carries its manifest.

    Disk wins on a name collision: it is the same simulation, and reading it needs no
    database round trip.
    """
    seen: set[str] = set()
    for manifest_path in sorted(runs.glob("**/case_manifest.json")):
        case_dir = manifest_path.parent
        if case_dir.name in seen:
            continue
        try:
            doc = json.loads(manifest_path.read_text())
        except ValueError:
            continue
        seen.add(case_dir.name)
        yield Case(name=case_dir.name, manifest=doc, case_dir=case_dir)
    if not aiida:
        return
    # SCOPED to `runs`, on the path the node records being ingested from. Without this
    # the fallback yielded every case in the database regardless of the directory it was
    # handed: a temp tree holding one run scored four unrelated `galiegue_*` cases, and a
    # recipe comparison scoped to one batch would quietly average in the rest of the store
    # (self-improve, 2026-09-09). `runs_root` has to keep meaning what it says.
    #
    # A no-op for whole-tree scoring, which is how the campaign runs it -- every node's
    # recorded path is under `runs/` -- so no scored result moves. It changes SUBSET
    # scoring, which is the defect.
    root = runs.resolve()
    skipped_unscoped = 0
    skipped_outside = 0
    for name, (uuid, manifest, source_dir) in sorted(_aiida_manifests().items()):
        if name in seen:
            continue
        if not source_dir:
            # No recorded path means it cannot be placed in or out of this tree.
            # Excluding is the honest scoped answer; including is the bug above.
            skipped_unscoped += 1
            continue
        try:
            Path(source_dir).resolve().relative_to(root)
        except ValueError:
            skipped_outside += 1
            continue
        yield Case(name=name, manifest=manifest, case_dir=None, uuid=uuid)
    if skipped_outside:
        # Announced, never silent: the scoping is the fix, but a caller who expected
        # those cases must SEE that they were dropped rather than read a quietly
        # smaller table. 509 of these live in ../lpno/data (the
        # ti64_random_keyhole150 import) and are real runs outside this tree.
        print(f"[score-anchors] {skipped_outside} AiiDA case(s) lie outside {root} and were "
              f"not scored. Point --runs at their tree, or score the parent directory, to "
              f"include them.", flush=True)
    if skipped_unscoped:
        print(f"[score-anchors] {skipped_unscoped} AiiDA case(s) record no source path and "
              f"were left out of a scoring scoped to {root}", flush=True)


def score_anchor_file(anchor: Path, runs: Path, *, sim_kind: str = "fusion",
                      band: float = mm.DEFAULT_BAND, p_rtol: float = 0.01,
                      r0_tol: float = 2e-6, series: str | None = None,
                      recipe: str | None = None, r0_um: float | None = None,
                      aiida: bool = True) -> tuple[list[dict[str, Any]], int]:
    """Score every run under ``runs`` that matches a point in ``anchor``.

    Returns (rows, ambiguous_count). Kept separate from the CLI so a scoring plan
    can drive it, because the correct flags differ per anchor -- sim-kind, recipe
    fingerprint and material variant are all anchor-specific, and remembering them
    by hand is how the wrong ones get used.

    ``aiida=False`` restricts scoring to exports still on disk.
    """

    points = filter_by_r0(normalize_points(load_anchor(anchor), series=series), r0_um)
    if not points:
        return [], 0
    rows: list[dict[str, Any]] = []
    ambiguous = 0
    for case in iter_cases(runs, aiida=aiida):
        cond = conditions_from_manifest(case.manifest)
        if cond is None:
            continue
        try:
            point = match_point(cond, points, r0_tol=r0_tol, p_rtol=p_rtol)
        except AmbiguousAnchor:
            ambiguous += 1
            continue
        if point is None:
            continue
        fingerprint = recipe_fingerprint(case.case_dir) if case.case_dir else ""
        if recipe and recipe not in fingerprint:
            # A case with no directory has no `prepin.*` to fingerprint, so a recipe
            # filter cannot be satisfied for it. Excluding is the honest outcome:
            # the alternative is asserting a recipe that was never read.
            continue
        # Conditions are matched BEFORE the arrays are opened: most of the tree
        # matches nothing, and this way neither a local `np.load` nor an AiiDA
        # round trip is spent on it.
        with open_case_arrays(case, aiida=aiida) as data:
            if data is None:
                continue
            try:
                valid = mm.domain_validity(data)
                off = laser_off_s(case.case_dir) if case.case_dir else \
                    _laser_off_from_manifest(case.manifest)
                g = (mm.fusion_geometry_auto(data) if sim_kind == "fusion"
                     else mm.depression(data, laser_off_s=off))
                cell_um = mm.cell_size_um(data)
            except Exception:  # noqa: BLE001 - a broken export must not abort the sweep
                continue
        row = score_row(sim_depth_um=g.depth_um, sim_width_um=g.width_um,
                        point=point, sim_kind=sim_kind, band=band)
        row.update(case=case.name, valid=valid.ok, floor_k=valid.floor_k,
                   power_w=cond["power_w"], speed_m_s=cond["speed_m_s"],
                   r0_m=cond["r0_m"], recipe=fingerprint,
                   cell_um=cell_um, source="disk" if case.case_dir else "aiida")
        rows.append(row)
    return rows, ambiguous


def run_plan(plan_path: Path, runs: Path, *, band: float = mm.DEFAULT_BAND,
             aiida: bool = True) -> int:
    """Score every anchor in a tracked plan and print one table per alloy.

    The plan records WHY each anchor needs its flags, so the reasoning travels with
    the numbers instead of living in a shell history.
    """
    plan = json.loads(Path(plan_path).read_text())
    print(f"  scoring plan: {plan_path}   runs: {runs}\n")
    for entry in plan["anchors"]:
        anchor = Path(plan_path).parent / entry["anchor"]
        if not anchor.is_file():
            print(f"  {entry['anchor']}: MISSING -- skipped\n")
            continue
        rows, amb = score_anchor_file(
            anchor, runs, sim_kind=entry.get("sim_kind", "fusion"),
            series=entry.get("series"), recipe=entry.get("recipe"),
            r0_um=entry.get("r0_um"), band=band, aiida=aiida)
        label = entry.get("label", entry["anchor"])
        print(f"  === {label}  ({entry.get('sim_kind','fusion')};"
              f" {entry.get('note','')})")
        valid = [r for r in rows if r["valid"]]
        if not rows:
            print(f"      no matching runs"
                  f"{f' ({amb} ambiguous)' if amb else ''}\n")
            continue
        for r in sorted(valid, key=lambda r: (r["power_w"], r["speed_m_s"])):
            w = (f"{r['width_err_pct']:+5.0f}%" if r["width_err_pct"] is not None else "   -- ")
            print(f"      {r['case']:26s} P={r['power_w']:5.0f} V={r['speed_m_s']:.3f} "
                  f"depth {r['depth_err_pct']:+5.0f}%  width {w}")
        if len(valid) != len(rows):
            print(f"      ({len(rows)-len(valid)} domain-invalid excluded)")
        recs = recipes_present(rows)
        if len(recs) > 1:
            print(f"      WARNING: {len(recs)} physics recipes pooled -- this summary "
                  f"describes nothing. Tighten the plan's `recipe` for this entry.")
        dups = duplicate_conditions(rows)
        if dups:
            extra = sum(len(v) - 1 for v in dups.values())
            print(f"      WARNING: {len(dups)} condition(s) matched by >1 run "
                  f"({extra} extra row(s)) -- counted as independent points but they "
                  f"are resolution variants or reruns:")
            for k, cases in sorted(dups.items()):
                print(f"          P={k[0]:g} V={k[1]:g} r0={k[2]*1e6:.0f}um: {', '.join(sorted(cases))}")
        for obs in ("depth", "width"):
            su = summarize(valid, obs)
            if su.n:
                print(f"      {obs:6s}: {su.n_within_band}/{su.n} in band, mean signed "
                      f"{su.mean_signed_error_pct:+.0f}%, systematic={su.systematic}")
        print()
    return 0


def main() -> int:
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("anchor", type=Path, nargs="?")
    ap.add_argument("--plan", type=Path, default=None,
                    help="a tracked scoring plan (validation/scoring_plan.json): score "
                         "every anchor with the flags it needs, in one command")
    ap.add_argument("--runs", type=Path, default=Path("runs"),
                    help="directory searched (recursively) for <case>/case_manifest.json; "
                         "arrays come from <case>/npz/ when present, else from AiiDA")
    ap.add_argument("--no-aiida", action="store_true",
                    help="score only exports still on disk. Without it, a case whose NPZ "
                         "was reclaimed is read from its stored payload in AiiDA, which "
                         "is what most solved runs look like once disk has been freed")
    ap.add_argument("--sim-kind", choices=mm.VALID_DEPTH_KINDS, default="fusion",
                    help="which surface the runs are read as (default fusion)")
    ap.add_argument("--band", type=float, default=mm.DEFAULT_BAND)
    ap.add_argument("--p-rtol", type=float, default=0.01,
                    help="relative power match tolerance (default 0.01 = 1%%)")
    ap.add_argument("--r0-tol", type=float, default=2e-6)
    ap.add_argument("--series", default=None,
                    help="for anchors with several material variants at identical "
                         "process conditions, score only this one (variants can differ "
                         "by 2x, so pooling them is meaningless)")
    ap.add_argument("--recipe", default=None,
                    help="only score runs whose prepin fingerprint contains this "
                         "substring, e.g. 'tv1=3520'. Without it, runs from different "
                         "recipes may be pooled and the summary will say so.")
    args = ap.parse_args()
    if args.plan:
        return run_plan(args.plan, args.runs, band=args.band, aiida=not args.no_aiida)
    if args.anchor is None:
        ap.error("give an anchor file or --plan")

    points = normalize_points(load_anchor(args.anchor), series=args.series)
    if not points:
        print(f"no usable measured points in {args.anchor}")
        return 1

    rows = []
    ambiguous = 0
    for case in iter_cases(args.runs, aiida=not args.no_aiida):
        cond = conditions_from_manifest(case.manifest)
        if cond is None:
            continue
        try:
            point = match_point(cond, points, r0_tol=args.r0_tol, p_rtol=args.p_rtol)
        except AmbiguousAnchor as exc:
            print(f"  {case.name}: {exc}")
            ambiguous += 1
            continue
        if point is None:
            continue
        with open_case_arrays(case, aiida=not args.no_aiida) as data:
            if data is None:
                continue
            try:
                valid = mm.domain_validity(data)
                if args.sim_kind == "fusion":
                    g = mm.fusion_geometry_auto(data)
                else:
                    off = laser_off_s(case.case_dir) if case.case_dir else \
                        _laser_off_from_manifest(case.manifest)
                    g = mm.depression(data, laser_off_s=off)
                cell_um = mm.cell_size_um(data)
            except Exception as exc:  # noqa: BLE001 - a broken export must not abort the sweep
                print(f"  {case.name}: unreadable ({type(exc).__name__})")
                continue
        row = score_row(sim_depth_um=g.depth_um, sim_width_um=g.width_um,
                        point=point, sim_kind=args.sim_kind, band=args.band)
        fingerprint = recipe_fingerprint(case.case_dir) if case.case_dir else ""
        if args.recipe and args.recipe not in fingerprint:
            continue
        row.update(case=case.name, valid=valid.ok, floor_k=valid.floor_k,
                   power_w=cond["power_w"], speed_m_s=cond["speed_m_s"], r0_m=cond["r0_m"],
                   recipe=fingerprint, cell_um=cell_um,
                   source="disk" if case.case_dir else "aiida")
        rows.append(row)

    if not rows:
        if ambiguous:
            print(f"  {ambiguous} run(s) matched several measured variants; "
                  f"re-run with --series to name one.")
        else:
            print("no runs matched any measured condition")
        return 1

    print(f"  anchor: {args.anchor.name}   runs read as: {args.sim_kind}   band: +-{args.band:.0%}")
    print("  case                      P     V     r0    depth ours/meas      width ours/meas     valid  cell  recipe")
    for r in sorted(rows, key=lambda r: (r["power_w"], r["speed_m_s"])):
        w = (f"{r.get('sim_width_um', 0):5.0f}/{r.get('measured_width_um', 0):<6.1f}"
             f"({r['width_err_pct']:+4.0f}%)" if r["width_err_pct"] is not None else "      --        ")
        print(f"  {r['case']:24s} {r['power_w']:5.0f} {r['speed_m_s']:.3f} {r['r0_m']*1e6:5.0f} "
              f"{r['sim_depth_um']:5.0f}/{r['measured_depth_um']:<6.1f}({r['depth_err_pct']:+4.0f}%) "
              f"{w}  {'ok' if r['valid'] else 'DOMAIN-INVALID'}  {r['cell_um']:.0f}um  {r['recipe']}")
    recipes = recipes_present(rows)
    if len(recipes) > 1:
        print(f"  WARNING: {len(recipes)} different physics recipes matched "
              f"({', '.join(recipes)}). A summary pooled over recipes describes "
              f"nothing -- re-run with --recipe to isolate one.")
    dups = duplicate_conditions(rows)
    if dups:
        n_extra = sum(len(v) - 1 for v in dups.values())
        print(f"  WARNING: {len(dups)} measured condition(s) matched by more than one run "
              f"({n_extra} extra row(s)). The summary counts them as independent points, "
              f"which they are not -- they are resolution variants or reruns:")
        for k, cases in sorted(dups.items()):
            print(f"    P={k[0]:g} V={k[1]:g} r0={k[2]*1e6:.0f}um: {', '.join(sorted(cases))}")
    valid_rows = [r for r in rows if r["valid"]]
    if len(valid_rows) != len(rows):
        print(f"  ({len(rows) - len(valid_rows)} domain-invalid row(s) excluded from the summary)")
    for obs in ("depth", "width"):
        s = summarize(valid_rows, obs)
        if s.n:
            print(f"  {obs:6s}: {s.n_within_band}/{s.n} in band, mean signed "
                  f"{s.mean_signed_error_pct:+.0f}%, systematic={s.systematic}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
