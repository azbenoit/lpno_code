"""Native AiiDA simulation node for per-simulation PINO tensors."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterator

import numpy as np
from numpy.lib import format as npformat

from aiida import orm

from .pino_dataset import (
    build_dataset_label,
    safe_attribute_get,
    sanitize_array_names,
    validate_source_type,
)


def is_stationary_case(metadata: dict[str, Any] | None) -> bool:
    """True when this case is a spot/dwell weld: the laser does not translate.

    A spot weld genuinely has `V_scan = 0` — that zero is the measurement, not a
    dropped value — so the "positive scan speed" guards must accept it *only*
    here. Read from the normalized `laser_stationary` field when the adapter set
    one, else from the case manifest's `derived.laser_stationary`, so
    simulations imported before that field existed still resolve correctly.
    """
    metadata = metadata or {}
    explicit = metadata.get("laser_stationary")
    if explicit is not None:
        return bool(explicit)
    manifest = metadata.get("flow3d_case_manifest") or {}
    return bool((manifest.get("derived") or {}).get("laser_stationary"))


def check_laser_scalars(metadata: dict[str, Any], *, where: str) -> None:
    """Laser power must be positive; scan speed positive unless this is a spot weld.

    The point of the guard is to catch a *dropped* value silently becoming a
    zero conditioning channel. For a stationary case zero is the physics, so
    refusing it would make spot welds unstorable — which it did until
    2026-08-04.
    """
    stationary = is_stationary_case(metadata)
    for key in ("laser_power", "V_scan"):
        value = metadata.get(key)
        allows_zero = stationary and key == "V_scan"
        if allows_zero and value is None:
            metadata[key] = value = 0.0
        ok = value is not None and math.isfinite(float(value))
        ok = ok and (float(value) >= 0.0 if allows_zero else float(value) > 0.0)
        if not ok:
            hint = (
                " (this case is not marked stationary; a dwell/spot run needs "
                "`derived.laser_stationary: true` in its manifest for V_scan=0 to be accepted)"
                if key == "V_scan" and not stationary and value in (0, 0.0)
                else ""
            )
            raise ValueError(
                f"{where} resolves `{key}` to {value!r}{hint}. A finite laser power and scan "
                "speed are needed to rebuild the P/V_scan channels; refusing to store "
                "silent-zero/NaN conditioning."
            )


def payload_content_hash(node) -> str:
    """A hash of what a stored simulation actually contains, not where it came from.

    Free: the disk-objectstore is content-addressed with sha256, so every
    repository object's key *is* the hash of its bytes. `payload.npz` in
    npz_file mode; in arrays mode, a hash over the per-array keys in name order,
    which is content-derived just the same.

    Why it exists: `build_simulation_fingerprint` hashes `source_uri` and
    `relative_paths`, so two imports of the same case from two directories get
    different fingerprints and both are stored. Twelve such pairs were found, one
    of them with both nodes inside the same live collection.
    Returns "" when the node has no repository objects (an evicted node).
    """
    objects = (node.base.repository.metadata or {}).get("o") or {}
    payload = objects.get("payload.npz")
    if isinstance(payload, dict) and payload.get("k"):
        return str(payload["k"])
    keys = sorted(
        str(entry["k"]) for entry in objects.values() if isinstance(entry, dict) and entry.get("k")
    )
    if not keys:
        return ""
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def build_simulation_fingerprint(metadata: dict[str, Any]) -> str:
    """Build a stable fingerprint for one simulation."""
    payload = {
        "source_type": metadata.get("source_type", ""),
        "source_uri": metadata.get("source_uri", ""),
        "source_folder_name": metadata.get("source_folder_name", ""),
        "source_group": metadata.get("source_group", ""),
        "relative_paths": list(metadata.get("relative_paths", [])),
        "grid_params": metadata.get("grid_params", {}),
        "requested_fields": list(metadata.get("requested_fields", [])),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


#: Solidus lives under several spellings across manifest vintages.
_SOLIDUS_KEYS = ("solidus_temperature_k", "solidus_k", "T_solidus", "TS1")


def material_facts(source) -> dict:
    """Which alloy a simulation is, as queryable scalars.

    "Which material is this?" had no answer without opening
    `simulation_metadata.flow3d_case_manifest.material.properties` node by node:
    measured 2026-08-10, **0 of 3,967** simulations carried any material extra.
    That also made the melt-pool temperature fallback unusable in bulk, since
    `melt_metrics.melt_mask` needs a solidus and refuses to guess one.

    Returns `material_source` on every simulation, including
    `"unrecorded"` when the manifest carries no material block at all -- which
    is a provenance gap worth finding, not a value worth inventing. The alloy is
    NEVER inferred from a sibling case or a dataset name here: writing a guessed
    solidus would score one alloy with another's melting point, silently.
    """
    manifest = case_manifest(source)
    material = manifest.get("material")
    if not isinstance(material, dict) or not material:
        return {"material_source": "unrecorded"}
    props = material.get("properties") if isinstance(material.get("properties"), dict) else {}
    facts: dict = {"material_source": "manifest"}
    name = material.get("name") or material.get("flow3d_label")
    if name:
        facts["material_name"] = str(name)
    if material.get("file"):
        facts["material_file"] = str(material["file"])
    for key in _SOLIDUS_KEYS:
        value = props.get(key)
        if isinstance(value, (int, float)) and value > 0:
            facts["solidus_k"] = float(value)
            break
    liquidus = props.get("liquidus_temperature_k")
    if isinstance(liquidus, (int, float)) and liquidus > 0:
        facts["liquidus_k"] = float(liquidus)
    return facts


def case_manifest(source) -> dict:
    """The FLOW-3D case manifest of anything that plausibly holds one, as a dict.

    Consumers grew the same defensive double-parse because the manifest's shape
    depends on where it was read from. Measured 2026-08-09 so the accessor can
    say what is actually true:

    - On a stored simulation node, `simulation_metadata.flow3d_case_manifest`
      is a dict on 3,942 of 3,942 -- never a JSON string.
    - Inside it, `input` is always a dict (the parsed run config), while
      `source_config` is a path STRING by design: it names the file `input` was
      parsed from. Do not json.loads it; read `input` instead.
    - A `.manifest.json` sidecar read from disk arrives as text, and that is
      the only place the string form exists.

    Accepts a PinoSimulationData node, a metadata dict, a manifest dict, a JSON
    string, or a path to a manifest file. Returns {} for anything empty.
    """
    import json as _json
    from pathlib import Path as _Path

    if source is None:
        return {}
    if hasattr(source, "get_simulation_metadata"):
        source = source.get_simulation_metadata() or {}
    if isinstance(source, (str, _Path)):
        text = str(source)
        if not text.lstrip().startswith("{"):
            text = _Path(text).read_text(encoding="utf-8")
        source = _json.loads(text)
    if not isinstance(source, dict):
        return {}
    # a metadata dict wraps the manifest; a manifest is returned as-is
    manifest = source.get("flow3d_case_manifest", source)
    if isinstance(manifest, str):
        manifest = _json.loads(manifest)
    return manifest if isinstance(manifest, dict) else {}


def queryable_flow3d_extras(metadata: dict) -> dict:
    """Case name and mesh shape as flat extras, for FLOW-3D simulations.

    Every one of these is already stored — three levels down, inside
    `simulation_metadata.flow3d_case_manifest`, where no practical query
    reaches them. Measured on the authors' database 2026-08-06: 3692 simulations carry that
    manifest, and **no** query could return (cell size, cells) across
    simulations without loading every node.

    `case_name` matters most. The node label is not reliably the case name —
    1565 of 2223 named simulations do not contain their own case name in their
    label — so `filters={'label': {'like': ...}}` is guesswork, and looking a
    case up by the name printed on disk had no entry point at all.
    """
    metadata = metadata or {}
    manifest = metadata.get("flow3d_case_manifest") or {}
    extras: dict = {}

    case_name = str(manifest.get("case_name") or metadata.get("case_name") or "").strip()
    if case_name:
        extras["case_name"] = case_name

    mesh = (manifest.get("input") or {}).get("mesh") or {}
    cell_size = mesh.get("cell_size_m")
    if isinstance(cell_size, (int, float)) and cell_size > 0:
        extras["cell_size_m"] = float(cell_size)

    # `nominal_mesh_cells` is what the runner derived from the bounds and the
    # cell size, i.e. the mesh actually requested. `grid_params.resolution` is
    # the shape of the EXPORTED arrays, which a stride or a crop can change, so
    # the two disagree and only the first answers "how big was the solve".
    cells = manifest.get("derived", {}).get("nominal_mesh_cells")
    if isinstance(cells, dict):
        axes = [cells.get(axis) for axis in ("x", "y", "z")]
        if all(isinstance(value, (int, float)) and value > 0 for value in axes):
            # ROUND, never int(). The runner writes this as a raw float division
            # (`extent / cell_size`), so 0.0009/1e-5 lands at 89.99999999999999
            # and `int()` floors it to 89 — cell_count 178000 instead of 180000,
            # 1-3% low. 1881 of 1896 on-disk manifests truncate on at least one
            # axis this way, and 60 simulations ingested on 2026-08-06 carried a
            # count that disagreed with their own `grid_params.resolution`.
            # Only 13 of 5688 axis values are genuinely fractional (assumptions
            # A10, a mesh the solver re-sized); the rest sit within 1e-14 of a
            # whole number. The earlier backfill was right only by accident: it
            # read the value back after AiiDA had rounded it to 14 significant
            # figures, so it saw 90.0.
            rounded = [int(round(value)) for value in axes]
            extras["mesh_cells"] = rounded
            extras["cell_count"] = rounded[0] * rounded[1] * rounded[2]

    cores = (manifest.get("parallel") or {}).get("cores")
    if isinstance(cores, int) and cores > 0:
        extras["cores"] = cores

    # Solve wall clock and cycle count. `mtime - ctime` on the node measures
    # queue and monitor latency, not the solve, so it is not a substitute. The
    # runner writes this `run` block into the manifest *after* the solve, and
    # 1543 of the 3692 stored manifests carry it — unreachable at three levels
    # down, which is why the honest answer lived on disk.
    run = manifest.get("run") or {}
    elapsed = run.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)) and elapsed > 0:
        extras["wall_seconds"] = float(elapsed)
        extras["wall_seconds_source"] = "case_manifest.run"
    # The runner writes `cycles` as a STRING ("10163"), so a numeric check drops
    # every one of them; coerce rather than skip.
    try:
        cycles = int(str(run.get("cycles")).strip())
    except (TypeError, ValueError):
        cycles = 0
    if cycles > 0:
        extras["cycles"] = cycles
    cpu_seconds = run.get("cpu_seconds")
    if isinstance(cpu_seconds, (int, float)) and cpu_seconds > 0:
        extras["cpu_seconds"] = float(cpu_seconds)
    # The one line flow3D_scripts treats as "did this solve actually finish".
    completion = str(run.get("completion") or "").strip()
    if completion:
        extras["completion"] = completion
        extras["normal_completion"] = "normal completion" in completion.lower()
    # Which alloy, as scalars. `material_source` is always set, so a simulation
    # with no material block is FINDABLE as a provenance gap
    # (`extras.material_source == "unrecorded"`) rather than merely silent.
    extras.update(material_facts(metadata))
    return extras


def rebuild_parametric_beam_paths(case_manifest: dict, case_name: str = "") -> dict:
    """Fill `derived.beams[i].path_xy_m` for beams defined PARAMETRICALLY, exactly.

    The prepin-extraction merge (`merge_multibeam_paths`) recovers per-beam paths from the
    velocity TABLES, which the mbc40 family carries. The mbs40 family does not: its beams
    are stored as a definition -- `{"type": "random", "seed": 11, "length_m": ...,
    "start_point_m": [...]}` in `input.laser.beams[i].path` -- so the merge has nothing to
    merge and emits no `path_xy_m`. Downstream, `_multibeam_specs` refuses ("2 beams but
    per-beam paths for only 0"), and a delete-and-restage cannot help: the regenerated
    sidecars are identical. 31 of 61 members of one multi-beam
    train view were affected.

    The path is RECOVERED, not invented, by calling the generator that produced the case --
    `laser_path.build_path` from the flow3D_scripts checkout, seeded identically. Verified
    on sim 69183: both beams reproduce the recorded `num_samples` (19, 22) and
    `path_length_m` to nine decimals, and the recorded `start_m`.

    Synthesizing a straight line from `start_m` + `scan_speed_m_per_s` + `duration_s` --
    the shape first proposed -- would be WRONG for this family: `path_type` is `random`,
    a seeded walk. It would have produced plausible geometry that silently mistrains,
    which is the failure mode this repo has paid for repeatedly.

    Best-effort: a beam already carrying a path is untouched, and anything unresolvable
    (no seed, no builder, an unknown `path_type` the builder rejects) is left alone for
    the downstream refusal to catch loudly rather than papered over here.
    """
    import copy

    derived = (case_manifest or {}).get("derived") or {}
    beams = derived.get("beams") or []
    specs = (((case_manifest or {}).get("input") or {}).get("laser") or {}).get("beams") or []
    if not beams or len(specs) != len(beams):
        return case_manifest
    if all(beam.get("path_xy_m") for beam in beams):
        return case_manifest

    try:
        build_path = _load_build_path()
    except Exception:  # noqa: BLE001 - no checkout, no reconstruction; refuse downstream
        return case_manifest
    if build_path is None:
        return case_manifest

    geometry = ((case_manifest.get("input") or {}).get("geometry") or {})
    laser = ((case_manifest.get("input") or {}).get("laser") or {})
    try:
        length = float(geometry["length_m"])
        width = float(geometry["width_m"])
        radius = float(laser["beam_radius_m"])
    except (KeyError, TypeError, ValueError):
        return case_manifest
    bounds = (-length / 2.0, length / 2.0, -width / 2.0, width / 2.0)

    merged = copy.deepcopy(case_manifest)
    out_beams = (merged.get("derived") or {}).get("beams") or []
    for beam, spec in zip(out_beams, specs):
        if beam.get("path_xy_m"):
            continue
        path_spec = (spec or {}).get("path")
        if not isinstance(path_spec, dict):
            continue
        try:
            built = build_path(dict(path_spec), radius=radius, bounds=bounds)
        except Exception:  # noqa: BLE001 - unknown path_type or bad spec: leave it
            continue
        samples = [[float(x), float(y)] for x, y in built.samples]
        recorded = beam.get("num_samples")
        if isinstance(recorded, int) and recorded != len(samples):
            # The rebuild disagrees with what the run recorded, so it is NOT the same
            # path. Refuse it rather than serve a different geometry than was simulated.
            continue
        beam["path_xy_m"] = samples
        speed = beam.get("scan_speed_m_per_s") or spec.get("scan_speed_m_per_s")
        if speed:
            # Constant scan speed along the polyline: the time at each sample is its
            # arc length divided by the speed, which is how the generator lays out the
            # velocity table it writes for FLOW-3D.
            times, travelled = [0.0], 0.0
            for (x0, y0), (x1, y1) in zip(samples, samples[1:]):
                travelled += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                times.append(travelled / float(speed))
            beam["path_times_s"] = times
    return merged


def _load_build_path():
    """`laser_path.build_path` from the flow3D_scripts checkout, or None.

    The generator is the single source of truth for path geometry; importing it is what
    keeps this a RECONSTRUCTION rather than a second implementation that can drift from
    the code that produced the data.
    """
    import importlib.util
    import sys
    from pathlib import Path as _Path

    from aiida_pino.utils.flow3d_config import default_runner_root

    root = default_runner_root()
    if not root:
        return None
    module_path = _Path(root) / "python" / "laser_path.py"
    if not module_path.is_file():
        return None
    cached = sys.modules.get("_aiida_pino_laser_path")
    if cached is None:
        spec = importlib.util.spec_from_file_location("_aiida_pino_laser_path", module_path)
        cached = importlib.util.module_from_spec(spec)
        sys.modules["_aiida_pino_laser_path"] = cached
        spec.loader.exec_module(cached)
    return getattr(cached, "build_path", None)


def merge_multibeam_paths(case_manifest: dict, beam_paths, case_name: str) -> dict:
    """Merge prepin-extracted per-beam paths into a case manifest's `derived.beams`.

    The mbc40 ingest thinned the beam tables, so the paths were re-extracted from the
    rendered prepins and stored as a mutable `multibeam_paths` extra (attributes are
    sealed). Pairing is POSITIONAL: `derived.beams[i].index` is 1-based while the
    extractor recorded 0-based, and an index-keyed join silently gave beam 0 beam 1's
    path and beam 1 none at all (2026-08-12). A length mismatch raises rather than
    merging part of the list.

    Beams left WITHOUT a path are then reconstructed from their generating seed, here and
    not in the callers. There are THREE sidecar writers -- this one's caller in
    `_export_runtime_npz_case`, `_write_runtime_strided_npz_manifest`, and the derived
    worker -- and putting the reconstruction in `merged_case_manifest` instead reached only
    the transform route: the pack-index (non-transforming) staging kept writing pathless
    sidecars for all 31 mbs40 cases, so a rebuilt cache still failed at the first loader
    read. Every writer already routes through THIS function,
    which makes it the one seam that cannot be missed -- the same duplicated-builder lesson
    the forced adapter twins carry.
    """
    import copy as _copy

    if not beam_paths:
        # Nothing extracted from the prepin (the parametric families have no velocity
        # table to extract from), but the beams may still be reconstructable.
        return rebuild_parametric_beam_paths(case_manifest, case_name)

    merged = _copy.deepcopy(case_manifest)
    beams = ((merged.get("derived") or {}).get("beams")) or []
    if len(beams) != len(beam_paths):
        raise ValueError(
            f"multibeam_paths has {len(beam_paths)} beams but the case manifest has "
            f"{len(beams)} for {case_name}; refusing to guess the pairing."
        )
    for beam, src in zip(beams, beam_paths):
        for key in ("path_xy_m", "path_times_s", "turn_off_time_s"):
            if key in src:
                beam[key] = src[key]
    # Extraction is authoritative where it applies; this only fills beams it left empty.
    return rebuild_parametric_beam_paths(merged, case_name)


class DataEvictedError(RuntimeError):
    """Raised when array data is requested from a metadata-only simulation.

    The node still holds all its metadata and provenance; only the arrays were
    deleted to reclaim disk. Re-import from source to restore them.
    """


class PinoSimulationData(orm.ArrayData):
    """Store one native PINO simulation as named arrays plus per-simulation metadata.

    Two storage modes exist:

    - ``arrays`` (``from_arrays``): every canonical array is its own ``.npy``
      repository object. This is the historical mode and stays the default.
    - ``npz_file`` (``from_npz_file``): the original compact FLOW-3D per-case
      NPZ is stored verbatim as the single repository object ``payload.npz``
      (compact exports compress 20-50x; unpacking them to raw ``.npy`` would
      inflate the repository by that factor). Reads canonicalize on the fly
      through ``aiida_pino.utils.flow3d_npz`` so both modes return
      byte-identical arrays.
    """

    FORMAT_VERSION = 1
    RUNTIME_MANIFEST_VERSION = "aiida_arraydir_v2"
    NPZ_RUNTIME_MANIFEST_VERSION = "aiida_npz_file_v1"
    DEFAULT_SOURCE_TYPE = "flow3d"
    TIME_VALUES_STORAGE_NAME = "_time_values_s"
    PAYLOAD_FILENAME = "payload.npz"
    STORAGE_MODE_ARRAYS = "arrays"
    STORAGE_MODE_NPZ_FILE = "npz_file"
    DATA_EVICTED_EXTRA = "data_evicted"
    # Escape hatch for the store()-level arrays-mode guard (tests, deliberate
    # legacy imports). Anything else must store compressed npz_file payloads.
    ARRAYS_STORE_ENV = "AIIDA_PINO_ALLOW_ARRAYS_STORAGE"

    def store(self, *args, **kwargs):  # noqa: D102 - contract documented on the guard
        if not self.is_stored and self.get_storage_mode() == self.STORAGE_MODE_ARRAYS:
            if os.environ.get(self.ARRAYS_STORE_ENV, "") != "1":
                raise ValueError(
                    "Refusing to store an arrays-mode simulation: uncompressed per-array "
                    "storage is ~20x the compact npz_file payload and once grew to ~754 GiB. "
                    "Store through from_npz_file, or set "
                    f"{self.ARRAYS_STORE_ENV}=1 to store legacy arrays deliberately and "
                    "convert them afterwards with `aiida-pino compress`."
                )
        return super().store(*args, **kwargs)

    # ---------------------------------------------------------- data eviction
    def is_data_evicted(self) -> bool:
        """True when the array data was deleted but the metadata was kept."""
        try:
            return bool(self.base.extras.get(self.DATA_EVICTED_EXTRA, False))
        except Exception:
            return False

    def data_size_bytes(self) -> int:
        """Reclaimable bytes of this simulation's stored arrays (metadata-cheap)."""
        total = 0
        for name in self.base.repository.list_object_names():
            try:
                total += int(self.base.repository.get_object_size(name))
            except Exception:  # noqa: BLE001 - a directory has no size that matters
                pass
        return total

    def _require_data(self) -> None:
        if self.is_data_evicted():
            info = self.base.extras.get(self.DATA_EVICTED_EXTRA, {}) or {}
            reason = f" ({info['reason']})" if isinstance(info, dict) and info.get("reason") else ""
            raise DataEvictedError(
                f"Simulation {self.pk} is metadata-only: its array data was evicted{reason}. "
                "Put it back with `aiida-pino restore-payloads --deposit <dir>` (a published deposit) "
                "or re-import it from source."
            )

    def evict_data(self, *, reason: str | None = None) -> int:
        """Delete the array data, keeping every attribute, link and extra.

        Metadata (all node attributes), provenance (all links) and identity
        (pk/uuid) are untouched — a model trained on a dataset that includes
        this simulation still resolves its training source. Only the repository
        bytes go, and reads raise :class:`DataEvictedError` afterwards.

        AiiDA seals a stored node's repository, so this reaches under the ORM to
        erase it and re-persist the ``repository_metadata`` column. It is a
        deliberate one-way operation; the freed bytes are only reclaimed from
        disk by ``verdi storage maintain --full``. Returns the byte count that
        became reclaimable (0 if already evicted).
        """
        if self.is_data_evicted():
            return 0
        freed = self.data_size_bytes()
        # Drop only THIS node's reference to its objects, never the objects
        # themselves: the object store is content-addressed and deduplicated, so
        # deleting the backing bytes would corrupt any other node that shares
        # them. `reset()` empties the virtual hierarchy without touching the
        # backend; the now-unreferenced bytes are reclaimed, refcount-safely, by
        # `verdi storage maintain --full`. AiiDA seals a stored node's
        # repository, so re-persist the metadata column directly.
        self.base.repository._repository.reset()
        self.base.repository._update_repository_metadata()
        backend = self.backend_entity
        backend.repository_metadata = self.base.repository.metadata
        backend.store()
        payload: dict[str, Any] = {"bytes_freed": int(freed)}
        if reason:
            payload["reason"] = str(reason)
        self.base.extras.set(self.DATA_EVICTED_EXTRA, payload)
        return int(freed)

    def restore_data(self, source, *, expected_sha256: str | None = None) -> dict[str, Any]:
        """Put the arrays of an evicted simulation back, keeping its identity.

        The inverse of :meth:`evict_data`. The published deposit ships its simulation nodes
        metadata-only and their payloads as per-family tarballs. `from_npz_file` would store
        a NEW node (new UUID) that no deposited collection, model or evaluation references,
        so nothing downstream could use it; this writes the payload back into THIS node and
        rebuilds its stored per-frame times from the payload's own `time_s`, exactly as
        `from_npz_file` stored them. Every collection, derivation and re-run that names the
        node then reads it as before the eviction.

        ``source`` is the payload file or a readable, seekable binary file object (a tar
        member opened with `tarfile.extractfile` works). The object store is
        content-addressed with sha256, so ``expected_sha256`` (the deposit's
        `payload_manifest.csv`) is checked against the stored key exactly, and nothing is
        persisted on a mismatch. Returns ``{"restored": bool, "bytes": int, "sha256": str}``.
        """
        if not self.is_data_evicted():
            return {"restored": False, "bytes": 0, "sha256": payload_content_hash(self)}

        def _times(handle) -> np.ndarray:
            with np.load(handle, allow_pickle=False) as archive:
                return np.asarray(archive["time_s"], dtype=np.float64).astype(np.float32)

        if isinstance(source, (str, os.PathLike)):
            times = _times(source)
        else:
            start = source.tell()
            times = _times(source)
            source.seek(start)
        expected_frames = self.get_num_timesteps()
        if expected_frames > 1 and times.size != expected_frames:
            raise ValueError(
                f"payload has {times.size} frames but simulation {self.uuid} records {expected_frames}; "
                "this is not its payload"
            )

        repository = self.base.repository
        repository._repository.reset()
        if isinstance(source, (str, os.PathLike)):
            with open(source, "rb") as handle:
                repository._repository.put_object_from_filelike(handle, self.PAYLOAD_FILENAME)
        else:
            repository._repository.put_object_from_filelike(source, self.PAYLOAD_FILENAME)
        key = str(repository._repository.get_object(self.PAYLOAD_FILENAME).key)
        if expected_sha256 and key != str(expected_sha256):
            repository._repository.reset()   # nothing persisted: the node stays evicted
            raise ValueError(f"payload for simulation {self.uuid} has sha256 {key}, expected {expected_sha256}")
        time_buffer = io.BytesIO()
        npformat.write_array(time_buffer, times, allow_pickle=False)
        time_buffer.seek(0)
        repository._repository.put_object_from_filelike(time_buffer, f"{self.TIME_VALUES_STORAGE_NAME}.npy")
        # A stored node's repository is sealed at the ORM; persist the metadata column
        # directly, as evict_data and compress_to_npz_file do.
        repository._update_repository_metadata()
        backend = self.backend_entity
        backend.repository_metadata = repository.metadata
        backend.store()
        self._cached_arrays = {}
        size = int(repository.get_object_size(self.PAYLOAD_FILENAME))
        self.base.extras.delete(self.DATA_EVICTED_EXTRA)
        self.base.extras.set("data_restored", {"sha256": key, "bytes": size})
        return {"restored": True, "bytes": size, "sha256": key}

    # ------------------------------------------------- arrays -> npz_file
    def compress_to_npz_file(self) -> dict[str, int]:
        """Convert a legacy arrays-mode simulation to compressed ``npz_file`` mode.

        De-broadcasts the stored canonical arrays back to the compact FLOW-3D
        schema (1-D axes + scalars + per-field frames), verifies that
        re-canonicalizing the rebuilt payload reproduces every stored array
        byte-for-byte, then swaps the repository objects in place. Identity
        (pk/uuid), attributes, links and extras all survive; ``get_array``
        returns identical data before and after. Raises ``ValueError`` and
        leaves the node untouched if any stored array is not an exact
        canonicalization image (never guess data away). Returns
        ``{"bytes_before", "bytes_after"}`` (both 0 when already converted).
        """
        import tempfile
        import zipfile

        from aiida_pino.utils.flow3d_npz import (
            CHANNEL_NAMES,
            canonicalize_flow3d_case_arrays,
        )

        self._require_data()
        if self.get_storage_mode() == self.STORAGE_MODE_NPZ_FILE:
            return {"bytes_before": 0, "bytes_after": 0}

        aliases = self.get_array_aliases()
        fields = [str(f) for f in safe_attribute_get(self, "requested_fields", [])]
        missing = [c for c in CHANNEL_NAMES if c not in aliases]
        extra = sorted(set(aliases) - set(CHANNEL_NAMES) - set(fields))
        if missing or extra:
            raise ValueError(
                f"Simulation {self.pk} is not a canonical run-dir import: "
                f"missing channel arrays {missing}, unexpected arrays {extra}. "
                "Only sims with the x/y/z/P/V_scan/time_s channels plus their "
                "requested fields can be de-broadcast."
            )
        # Bound, not inline: the writes below read these back, and
        # `check_laser_scalars` normalizes a stationary case's absent V_scan to 0.0
        # in place -- passing the dict as a literal discarded both.
        laser_scalars = {
            "laser_power": safe_attribute_get(self, "laser_power"),
            "V_scan": safe_attribute_get(self, "V_scan"),
            "laser_stationary": safe_attribute_get(self, "laser_stationary"),
            "flow3d_case_manifest": (self.get_simulation_metadata() or {}).get("flow3d_case_manifest"),
        }
        check_laser_scalars(laser_scalars, where=f"Simulation {self.pk}")

        bytes_before = self.data_size_bytes()
        metadata = self.get_simulation_metadata()
        case_name = str(metadata.get("case_name") or self.get_simulation_identifier())

        def _stored(name: str) -> np.ndarray:
            # arrays mode: plain ArrayData read of one `.npy` object. Evict it
            # from ArrayData's instance cache right away — the cache keeps every
            # array alive for the node's lifetime, which for a batch conversion
            # of GiB-scale sims is an OOM, not a cache (seen live 2026-07-27:
            # ~123 GiB RSS after ~100 sims).
            alias = aliases[name]
            array = np.asarray(orm.ArrayData.get_array(self, alias))
            self._cached_arrays.pop(alias, None)
            return array

        # Extract the compact pieces; exactness is proven by the verify pass.
        x = np.ascontiguousarray(_stored("x")[0, 0, :, 0, 0], dtype=np.float32)
        y = np.ascontiguousarray(_stored("y")[0, 0, 0, :, 0], dtype=np.float32)
        z = np.ascontiguousarray(_stored("z")[0, 0, 0, 0, :], dtype=np.float32)
        times = np.ascontiguousarray(_stored("time_s")[:, 0, 0, 0, 0], dtype=np.float64)

        with tempfile.TemporaryDirectory() as workdir:
            compact_path = Path(workdir) / self.PAYLOAD_FILENAME
            # Stream one member at a time: the largest field of the fattest sim
            # is ~2.7 GiB, the full canonical set ~18 GiB — never hold them all.
            with zipfile.ZipFile(compact_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:

                def _write(name: str, array: np.ndarray) -> None:
                    with archive.open(f"{name}.npy", "w", force_zip64=True) as handle:
                        npformat.write_array(handle, np.ascontiguousarray(array), allow_pickle=False)

                _write("x", x)
                _write("y", y)
                _write("z", z)
                _write("time_s", times)
                _write("power_w", np.float64(laser_scalars["laser_power"]))
                _write("scan_speed_m_per_s", np.float64(laser_scalars["V_scan"]))
                _write("case_name", np.str_(case_name))
                for field in fields:
                    stored = _stored(field)
                    if stored.ndim != 5:
                        raise ValueError(
                            f"Simulation {self.pk} field `{field}` has shape "
                            f"{list(stored.shape)}, expected (T, C, nx, ny, nz)."
                        )
                    if stored.shape[1] == 1:
                        _write(field, stored[:, 0].astype(np.float32, copy=False))
                    else:  # vector field, canonical (T, C, ...) -> compact (T, ..., C)
                        _write(field, np.moveaxis(stored, 1, -1).astype(np.float32, copy=False))
                    del stored

            # Verify: the rebuilt payload must canonicalize back to the stored
            # bytes, array for array, before anything is swapped.
            with np.load(compact_path, allow_pickle=False) as rebuilt:
                for name in (*CHANNEL_NAMES, *fields):
                    case = canonicalize_flow3d_case_arrays(
                        rebuilt,
                        case_name=case_name,
                        laser_power=float(laser_scalars["laser_power"]),
                        v_scan=float(laser_scalars["V_scan"]),
                        selected_fields=fields,
                        only=[name],
                    )
                    if not np.array_equal(case.arrays[name], _stored(name)):
                        raise ValueError(
                            f"Simulation {self.pk} array `{name}` is not an exact "
                            "canonicalization image of the rebuilt compact payload; "
                            "refusing to convert."
                        )
                    del case

            time_values = np.asarray(
                orm.ArrayData.get_array(self, self.TIME_VALUES_STORAGE_NAME), dtype=np.float32
            )
            # Swap the repository content in one backend persist (same pattern
            # as evict_data — a stored node's repository is sealed at the ORM).
            repository = self.base.repository
            repository._repository.reset()
            with compact_path.open("rb") as handle:
                repository._repository.put_object_from_filelike(handle, self.PAYLOAD_FILENAME)
            time_buffer = io.BytesIO()
            npformat.write_array(time_buffer, time_values, allow_pickle=False)
            time_buffer.seek(0)
            repository._repository.put_object_from_filelike(
                time_buffer, f"{self.TIME_VALUES_STORAGE_NAME}.npy"
            )
            repository._update_repository_metadata()
            backend = self.backend_entity
            backend.repository_metadata = repository.metadata
            backend.store()
            bytes_after = int(compact_path.stat().st_size) + int(time_buffer.getbuffer().nbytes)

        # The frontend blocks attribute writes on stored nodes; the backend
        # setter flushes immediately and keeps the mode truthful for queries.
        self._cached_arrays = {}
        backend.set_attribute("storage_mode", self.STORAGE_MODE_NPZ_FILE)
        self.base.extras.set("storage_mode", self.STORAGE_MODE_NPZ_FILE)
        self.base.extras.set(
            "compressed_from_arrays",
            {"bytes_before": int(bytes_before), "bytes_after": int(bytes_after)},
        )
        return {"bytes_before": int(bytes_before), "bytes_after": int(bytes_after)}

    @classmethod
    def _normalize_metadata(
        cls,
        normalized: dict[str, Any],
        *,
        aliases: dict[str, str],
        array_shapes: dict[str, list[int]],
        requested_fields_default: list[str],
        time_values_array: np.ndarray | None,
    ) -> None:
        """Apply the shared metadata defaults/derivations in place."""
        normalized.setdefault("format_version", cls.FORMAT_VERSION)
        normalized.setdefault("source_type", cls.DEFAULT_SOURCE_TYPE)
        validate_source_type(str(normalized["source_type"]))
        normalized.setdefault("source_origin_kind", "import")
        normalized.setdefault("source_folder_name", "simulation")
        normalized.setdefault("source_group", normalized.get("group_name", normalized["source_folder_name"]))
        normalized.setdefault("requested_fields", requested_fields_default)
        normalized.setdefault("source_uri", "")
        normalized.setdefault("relative_paths", [])
        # Optional human-facing annotations. `description` and `batch_keyword`
        # normalize None -> "" so the query surface is always a string; the
        # `usable_for_training` review flag defaults True (freshly imported runs
        # are usable until a reviewer flips its extra). None of these feed the
        # fingerprint, so annotating a simulation never changes its identity.
        normalized["description"] = str(normalized.get("description") or "")
        normalized["batch_keyword"] = str(normalized.get("batch_keyword") or "")
        usable_for_training = normalized.get("usable_for_training")
        normalized["usable_for_training"] = True if usable_for_training is None else bool(usable_for_training)
        # Resolve the spot/dwell flag once, by the same rule the guards use. The
        # storage path read `normalized.get("laser_stationary", False)`, which has
        # no manifest fallback, so a case accepted via `flow3d_case_manifest`
        # stored `laser_stationary=False` next to `V_scan=0.0` -- a moving laser
        # at zero speed, which is the combination the guard exists to reject.
        normalized["laser_stationary"] = is_stationary_case(normalized)
        normalized["array_aliases"] = aliases
        normalized["array_shapes"] = array_shapes
        first_shape = next(iter(array_shapes.values()), [])
        if first_shape and len(first_shape) >= 4:
            normalized.setdefault("spatial_shape", list(first_shape[-3:]))
        else:
            normalized.setdefault("spatial_shape", [])

        if time_values_array is not None:
            normalized["num_timesteps"] = int(time_values_array.shape[0])
            normalized.setdefault("effective_end_time_s", float(time_values_array[-1]))
            if int(time_values_array.shape[0]) > 1:
                deltas = np.diff(time_values_array.astype(np.float64))
                if np.allclose(deltas, deltas[0]):
                    normalized.setdefault("delta_t_s", float(deltas[0]))
                else:
                    normalized.setdefault("delta_t_s", None)
            else:
                normalized.setdefault("delta_t_s", None)
        else:
            shape = next(iter(array_shapes.values()), [])
            inferred_steps = int(shape[0]) if shape and len(shape) >= 5 else 1
            normalized.setdefault("num_timesteps", inferred_steps)
            normalized.setdefault("delta_t_s", None)
            normalized.setdefault("effective_end_time_s", None)

        normalized.setdefault("simulation_fingerprint", build_simulation_fingerprint(normalized))
        normalized.setdefault(
            "simulation_identifier",
            f"{normalized['source_folder_name']}::{normalized['source_group']}::{normalized['simulation_fingerprint'][:8]}",
        )

    @classmethod
    def _apply_node_metadata(
        cls,
        node: "PinoSimulationData",
        normalized: dict[str, Any],
        *,
        aliases: dict[str, str],
        storage_mode: str,
        source: dict | None,
    ) -> None:
        """Write the shared attribute/extras contract onto the node."""
        node.base.attributes.set("format_version", normalized["format_version"])
        node.base.attributes.set("storage_mode", storage_mode)
        node.base.attributes.set("source_type", normalized["source_type"])
        node.base.attributes.set("source_origin_kind", normalized["source_origin_kind"])
        node.base.attributes.set("source_folder_name", normalized["source_folder_name"])
        node.base.attributes.set("source_group", normalized["source_group"])
        node.base.attributes.set("source_uri", normalized["source_uri"])
        node.base.attributes.set("requested_fields", normalized["requested_fields"])
        node.base.attributes.set("relative_paths", normalized["relative_paths"])
        node.base.attributes.set("simulation_identifier", normalized["simulation_identifier"])
        node.base.attributes.set("simulation_fingerprint", normalized["simulation_fingerprint"])
        node.base.attributes.set("num_timesteps", normalized["num_timesteps"])
        node.base.attributes.set("spatial_shape", normalized["spatial_shape"])
        node.base.attributes.set("array_aliases", aliases)
        node.base.attributes.set("array_shapes", normalized["array_shapes"])
        node.base.attributes.set("delta_t_s", normalized.get("delta_t_s"))
        node.base.attributes.set("effective_end_time_s", normalized.get("effective_end_time_s"))
        node.base.attributes.set("V_scan", normalized.get("V_scan"))
        node.base.attributes.set("laser_power", normalized.get("laser_power"))
        node.base.attributes.set("beam_radius", normalized.get("beam_radius"))
        node.base.attributes.set("beam_blocks", normalized.get("beam_blocks"))
        # A spot/dwell weld: V_scan = 0 is its measurement. Promoted so it is
        # queryable and so the guards can read it without opening the manifest.
        node.base.attributes.set("laser_stationary", bool(normalized.get("laser_stationary", False)))
        node.base.attributes.set("description", normalized["description"])
        node.base.attributes.set("batch_keyword", normalized["batch_keyword"])
        node.base.attributes.set("usable_for_training", normalized["usable_for_training"])
        node.base.attributes.set("simulation_metadata", normalized)
        node.label = build_dataset_label(
            normalized["source_folder_name"],
            normalized["source_group"],
            normalized["source_type"],
        )
        if source is not None:
            node.source = source
        node.base.extras.set("aiida_pino_kind", "simulation")
        node.base.extras.set("storage_mode", storage_mode)
        node.base.extras.set("source_type", normalized["source_type"])
        node.base.extras.set("source_folder_name", normalized["source_folder_name"])
        node.base.extras.set("source_group", normalized["source_group"])
        node.base.extras.set("simulation_identifier", normalized["simulation_identifier"])
        node.base.extras.set("simulation_fingerprint", normalized["simulation_fingerprint"])
        node.base.extras.set("num_timesteps", normalized["num_timesteps"])
        node.base.extras.set("delta_t_s", normalized.get("delta_t_s"))
        node.base.extras.set("effective_end_time_s", normalized.get("effective_end_time_s"))
        node.base.extras.set("V_scan", normalized.get("V_scan"))
        node.base.extras.set("laser_power", normalized.get("laser_power"))
        node.base.extras.set("beam_radius", normalized.get("beam_radius"))
        node.base.extras.set("beam_blocks", normalized.get("beam_blocks"))
        node.base.extras.set("laser_stationary", bool(normalized.get("laser_stationary", False)))
        # Case name and mesh shape, lifted out of the nested FLOW-3D manifest so
        # a query can reach them (see `queryable_flow3d_extras`).
        for key, value in queryable_flow3d_extras(normalized).items():
            node.base.extras.set(key, value)
        node.base.extras.set("description", normalized["description"])
        node.base.extras.set("batch_keyword", normalized["batch_keyword"])
        node.base.extras.set("usable_for_training", normalized["usable_for_training"])
        # Shared annotation contract: mutable review extras with fixed defaults.
        node.base.extras.set("split", "unassigned")
        node.base.extras.set("keywords", [])

    @classmethod
    def from_arrays(
        cls,
        arrays: dict[str, np.ndarray],
        *,
        metadata: dict[str, Any],
        source: dict | None = None,
    ) -> "PinoSimulationData":
        """Create a simulation node from per-field arrays and metadata."""
        original_names = list(arrays)
        aliases = sanitize_array_names(original_names)

        node = cls()
        for original_name, array in arrays.items():
            node.set_array(aliases[original_name], np.asarray(array))

        normalized = dict(metadata)
        time_values = normalized.pop("time_values_s", None)
        normalized.setdefault("source_uri", source.get("uri") if source else "")
        array_shapes = {
            original_name: list(np.asarray(array).shape) for original_name, array in arrays.items()
        }

        time_values_array = None
        if time_values is not None:
            time_values_array = np.asarray(time_values, dtype=np.float32)
            if time_values_array.ndim != 1:
                raise ValueError("time_values_s must be one-dimensional when provided.")
            if time_values_array.shape[0] == 0:
                # An empty list carries no time information; infer from array shapes.
                time_values_array = None

        if time_values_array is not None:
            for array_name, shape in array_shapes.items():
                if len(shape) >= 5 and int(shape[0]) != int(time_values_array.shape[0]):
                    raise ValueError(
                        f"time_values_s has {int(time_values_array.shape[0])} entries but array "
                        f"`{array_name}` has {int(shape[0])} time steps."
                    )
            node.set_array(cls.TIME_VALUES_STORAGE_NAME, time_values_array)

        cls._normalize_metadata(
            normalized,
            aliases=aliases,
            array_shapes=array_shapes,
            requested_fields_default=original_names,
            time_values_array=time_values_array,
        )
        cls._apply_node_metadata(
            node, normalized, aliases=aliases, storage_mode=cls.STORAGE_MODE_ARRAYS, source=source
        )
        return node

    @classmethod
    def from_npz_file(
        cls,
        path: str | os.PathLike,
        *,
        metadata: dict[str, Any],
        source: dict | None = None,
    ) -> "PinoSimulationData":
        """Create a simulation node that stores a compact FLOW-3D NPZ verbatim.

        ``path`` is resolved through ``os.path.realpath`` and the file is stored
        unchanged as the single repository object ``payload.npz``. Shapes and
        dtypes come from header-only reads; only the small axis arrays are
        decompressed at creation time. Frame subsampling is not supported in
        this mode: every stored frame is kept.

        ``metadata`` is the case description stored as ``simulation_metadata``:
        at least ``case_name``, ``laser_power`` (W) and ``V_scan`` (m/s), plus
        ``requested_fields`` and the case manifest when known. This makes a NEW
        node. To put a published deposit's payload back into the deposited node
        (same UUID, so its collections, models and evaluations still name it),
        use :meth:`restore_data` or ``aiida-pino restore-payloads``.
        """
        from aiida_pino.utils.flow3d_npz import (
            canonical_array_names,
            canonical_shapes_from_headers,
            grid_params_from_axes,
            read_npz_headers,
            require_monotonic,
        )

        real_path = Path(os.path.realpath(os.fspath(path)))
        if not real_path.is_file():
            raise ValueError(f"FLOW-3D NPZ payload does not exist or is not a file: {real_path}")

        normalized = dict(metadata)
        normalized.pop("time_values_s", None)
        case_name = str(normalized.get("case_name") or "")
        requested = normalized.get("requested_fields") or None
        if requested is not None:
            requested = [str(field) for field in requested]

        headers = read_npz_headers(str(real_path))
        array_shapes, fields = canonical_shapes_from_headers(
            headers, case_name=case_name or real_path.stem, selected_fields=requested
        )

        with np.load(real_path, allow_pickle=False) as archive:
            x = np.asarray(archive["x"], dtype=np.float32)
            y = np.asarray(archive["y"], dtype=np.float32)
            z = np.asarray(archive["z"], dtype=np.float32)
            times = np.asarray(archive["time_s"], dtype=np.float64)
            for axis, name in ((x, "x"), (y, "y"), (z, "z"), (times, "time_s")):
                require_monotonic(axis, name, case_name or real_path.stem)
            if not case_name and "case_name" in archive.files:
                case_name = str(archive["case_name"])
            for scalar_key, metadata_key in (("power_w", "laser_power"), ("scan_speed_m_per_s", "V_scan")):
                if normalized.get(metadata_key) is None and scalar_key in archive.files:
                    value = float(np.asarray(archive[scalar_key]))
                    if math.isfinite(value):
                        normalized[metadata_key] = value
        case_name = case_name or real_path.stem

        check_laser_scalars(normalized, where=f"FLOW-3D case `{case_name}`")

        normalized["case_name"] = case_name
        normalized["requested_fields"] = fields
        normalized.setdefault("grid_params", grid_params_from_axes(x, y, z))
        normalized.setdefault("source_uri", f"file://{real_path}")
        normalized.setdefault("relative_paths", [str(real_path)])
        normalized["source_npz_realpath"] = str(real_path)
        # Identity comes from the canonical (realpath) source file, deliberately
        # excluding folder-layout metadata: a symlink view of the same NPZ (for
        # example inside a train/ split folder) must dedupe onto its canonical
        # source. Annotations stay excluded from identity.
        normalized.setdefault(
            "simulation_fingerprint",
            build_simulation_fingerprint(
                {
                    "source_type": str(normalized.get("source_type", cls.DEFAULT_SOURCE_TYPE)),
                    "source_uri": f"file://{real_path}",
                    "source_folder_name": "",
                    "source_group": "",
                    "relative_paths": [str(real_path)],
                    "grid_params": normalized.get("grid_params", {}),
                    "requested_fields": fields,
                }
            ),
        )

        original_names = canonical_array_names(fields)
        aliases = sanitize_array_names(original_names)

        node = cls()
        node.base.repository.put_object_from_file(str(real_path), cls.PAYLOAD_FILENAME)
        time_values_array = times.astype(np.float32)
        node.set_array(cls.TIME_VALUES_STORAGE_NAME, time_values_array)

        cls._normalize_metadata(
            normalized,
            aliases=aliases,
            array_shapes=array_shapes,
            requested_fields_default=fields,
            time_values_array=time_values_array,
        )
        cls._apply_node_metadata(
            node, normalized, aliases=aliases, storage_mode=cls.STORAGE_MODE_NPZ_FILE, source=source
        )
        return node

    def get_storage_mode(self) -> str:
        """Return the array storage mode (``arrays`` or ``npz_file``)."""
        return str(safe_attribute_get(self, "storage_mode", self.STORAGE_MODE_ARRAYS))

    @contextmanager
    def _open_payload_archive(self):
        """Open the stored ``payload.npz`` as a numpy archive."""
        self._require_data()
        # ponytail: reads the whole compressed payload into RAM per call; the
        # compact NPZs are 20-50x smaller than their arrays, so this stays small.
        with self.base.repository.open(self.PAYLOAD_FILENAME, mode="rb") as handle:
            buffer = io.BytesIO(handle.read())
        with np.load(buffer, allow_pickle=False) as archive:
            yield archive

    def _canonical_arrays_from_payload(self, only: list[str] | None = None) -> dict[str, np.ndarray]:
        """Canonicalize arrays from the stored payload (npz_file mode)."""
        from aiida_pino.utils.flow3d_npz import canonicalize_flow3d_case_arrays

        metadata = self.get_simulation_metadata()
        case_name = str(metadata.get("case_name") or self.get_simulation_identifier())
        with self._open_payload_archive() as archive:
            case = canonicalize_flow3d_case_arrays(
                archive,
                case_name=case_name,
                laser_power=float(safe_attribute_get(self, "laser_power")),
                v_scan=float(safe_attribute_get(self, "V_scan")),
                selected_fields=list(safe_attribute_get(self, "requested_fields", [])) or None,
                only=only,
            )
        return case.arrays

    def get_array(self, name: str | None = None) -> np.ndarray:
        """Return one array; npz_file-mode nodes canonicalize from ``payload.npz``."""
        self._require_data()
        if self.get_storage_mode() != self.STORAGE_MODE_NPZ_FILE or name == self.TIME_VALUES_STORAGE_NAME:
            return super().get_array(name)
        if name is None:
            raise ValueError("npz_file-mode simulations store multiple arrays; request one by name.")
        aliases = self.get_array_aliases()
        stored_to_original = {stored: original for original, stored in aliases.items()}
        if name in stored_to_original:
            original = stored_to_original[name]
        elif name in aliases:
            original = name
        else:
            raise KeyError(f"Simulation has no array `{name}`. Known arrays: {sorted(stored_to_original)}.")
        return self._canonical_arrays_from_payload(only=[original])[original]

    def get_simulation_metadata(self) -> dict[str, Any]:
        """Return the simulation metadata payload."""
        metadata = dict(safe_attribute_get(self, "simulation_metadata", {}))
        for key in (
            "delta_t_s",
            "effective_end_time_s",
            "V_scan",
            "laser_power",
            "beam_radius",
            "beam_blocks",
            "laser_stationary",
            # Read back from extras so a post-store edit (a reviewer flipping
            # `usable_for_training`, retagging `batch_keyword`) is reflected.
            "description",
            "batch_keyword",
            "usable_for_training",
        ):
            try:
                value = self.base.extras.get(key, None)
            except Exception:
                value = None
            if value is not None:
                metadata[key] = value
        return metadata

    def get_array_aliases(self) -> dict[str, str]:
        """Return the original->stored field mapping."""
        return dict(safe_attribute_get(self, "array_aliases", {}))

    def list_original_array_names(self) -> list[str]:
        """Return stored field names."""
        return list(self.get_array_aliases())

    def get_array_shapes(self) -> dict[str, list[int]]:
        """Return stored field shapes."""
        return dict(safe_attribute_get(self, "array_shapes", {}))

    def get_simulation_identifier(self) -> str:
        """Return the stable simulation identifier."""
        return str(safe_attribute_get(self, "simulation_identifier", ""))

    def get_simulation_fingerprint(self) -> str:
        """Return the stable simulation fingerprint."""
        return str(safe_attribute_get(self, "simulation_fingerprint", ""))

    def get_num_timesteps(self) -> int:
        """Return the stored timestep count."""
        return int(safe_attribute_get(self, "num_timesteps", 1))

    def get_time_values_s(self) -> np.ndarray | None:
        """Return stored time values when present; raises :class:`DataEvictedError` on an evicted node."""
        try:
            return np.asarray(self.get_array(self.TIME_VALUES_STORAGE_NAME), dtype=np.float32)
        except DataEvictedError:
            raise
        except Exception:
            metadata = self.get_simulation_metadata()
            values = metadata.get("time_values_s")
            if values is None:
                return None
            return np.asarray(values, dtype=np.float32)

    def get_array_by_original_name(self, name: str) -> np.ndarray:
        """Load one field array by its original name."""
        aliases = self.get_array_aliases()
        stored_name = aliases.get(name, name)
        return self.get_array(stored_name)

    def iter_original_arrays(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield the arrays using their original field names."""
        aliases = self.get_array_aliases()
        for original_name, stored_name in aliases.items():
            yield original_name, self.get_array(stored_name)

    def _read_repository_npy_header(self, filename: str) -> tuple[list[int], str]:
        """Read shape and dtype from a stored ``.npy`` array without materializing it."""
        with self.base.repository.open(filename, mode="rb") as handle:
            version = npformat.read_magic(handle)
            if version == (1, 0):
                shape, _, dtype = npformat.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, _, dtype = npformat.read_array_header_2_0(handle)
            else:
                shape, _, dtype = npformat.read_array_header_2_0(handle)
        return list(shape), np.dtype(dtype).name

    def _copy_repository_object_to_path(self, filename: str, destination: Path) -> None:
        """Copy a repository object to ``destination`` using streamed IO.

        Written to a temp name and atomically renamed: opening an EXISTING
        destination "wb" truncates the inode in place, and a concurrent reader
        holding that file open sees mixed old/new bytes -- the `BadZipFile: Bad
        CRC-32` that killed two varspeed runs at step ~8000 (2026-08-13/14)
        while every file verified clean afterwards. `os.replace` swaps the
        directory entry instead; an open reader keeps the old inode intact.
        """
        tmp = destination.with_name(destination.name + ".tmp-write")
        with self.base.repository.open(filename, mode="rb") as handle:
            with tmp.open("wb") as target:
                shutil.copyfileobj(handle, target, length=1024 * 1024 * 16)
        os.replace(tmp, destination)

    def _try_link_repository_object(self, filename: str, destination: Path) -> bool:
        """Try to hardlink a repository object into ``destination`` when locally possible."""
        try:
            repository = self.base.repository._repository
            backend = repository.backend
            container = getattr(backend, "_container", None)
            if container is None or not hasattr(container, "get_object_meta"):
                return False
            obj = self.base.repository.get_object(filename)
            meta = container.get_object_meta(obj.key)
            if getattr(getattr(meta, "type", None), "value", None) != "loose":
                return False
            loose_path = Path(container._get_loose_path_from_hashkey(obj.key))
            if not loose_path.is_file():
                return False
            if os.stat(loose_path).st_dev != os.stat(destination.parent).st_dev:
                return False
            destination.hardlink_to(loose_path)
            return True
        except Exception:
            return False

    def iter_runtime_array_entries(self) -> Iterator[dict[str, Any]]:
        """Yield manifest entries for all stored field arrays."""
        aliases = self.get_array_aliases()
        if self.get_storage_mode() == self.STORAGE_MODE_NPZ_FILE:
            # No per-array `.npy` repository objects exist in this mode; the
            # canonical shapes come from the attributes and everything is
            # float32 after canonicalization. `filename` is what a staged
            # export of the array would be called.
            shapes = self.get_array_shapes()
            for original_name, stored_name in aliases.items():
                yield {
                    "original_name": original_name,
                    "stored_name": stored_name,
                    "filename": f"{stored_name}.npy",
                    "shape": list(shapes.get(original_name, [])),
                    "dtype": "float32",
                }
            return
        for original_name, stored_name in aliases.items():
            filename = f"{stored_name}.npy"
            shape, dtype = self._read_repository_npy_header(filename)
            yield {
                "original_name": original_name,
                "stored_name": stored_name,
                "filename": filename,
                "shape": shape,
                "dtype": dtype,
            }

    def _export_runtime_npz_case(self, target_path: Path, pack_index: list | None = None) -> Path:
        """Stage an npz_file-mode simulation as a flat consolidated per-case NPZ.

        The stored ``payload.npz`` is hardlinked (falling back to a copy across
        devices) as ``<case_name>.npz`` into the *collection root* when staged
        through ``PinoDatasetCollectionData.export_runtime_arraydir`` (layout
        ``<root>/members/simulation_NNNN``), so the staged dataset directory is
        a flat folder of per-case NPZ files that the lpno
        consolidated-NPZ datasource reads natively. A direct per-simulation
        export stages the file into ``target_path`` itself. The FLOW-3D case
        manifest, when known, is written beside it as the
        ``<case_name>.manifest.json`` sidecar the datasource uses for path and
        material channels.
        """
        target_path.mkdir(parents=True, exist_ok=True)
        metadata = self.get_simulation_metadata()
        case_name = str(
            metadata.get("case_name")
            or safe_attribute_get(self, "source_group", "")
            or self.get_simulation_fingerprint()
        )
        if target_path.parent.name == "members":
            flat_root = target_path.parent.parent
        else:
            flat_root = target_path
        flat_root.mkdir(parents=True, exist_ok=True)

        npz_destination = flat_root / f"{case_name}.npz"
        if npz_destination.exists():
            if flat_root == target_path:
                npz_destination.unlink()
            else:
                # The collection export starts from an empty root, so an
                # existing file is a case_name collision between two members.
                raise ValueError(
                    f"Duplicate case name `{case_name}` while staging npz_file simulations; "
                    "collection members must have unique case names."
                )
        indexed = False
        if pack_index is not None:
            # Zero-copy: the payload is an uncompressed contiguous byte range of
            # a pack file (A-PACKSLICE), so the staged "file" can be an index
            # entry the pino loader opens as a bounded window over the pack.
            # Falls through to a real copy when the payload is not locatable.
            from aiida_pino.utils.runtime_export import payload_locator

            locator = payload_locator(self)
            if locator is not None:
                obj = self.base.repository.get_object(self.PAYLOAD_FILENAME)
                pack_index.append(
                    {
                        "filename": npz_destination.name,
                        "pack_path": locator["path"],
                        "offset": locator["offset"],
                        "size": locator["size"],
                        "hashkey": obj.key,
                        "simulation_pk": self.pk,
                        "simulation_uuid": str(self.uuid),
                    }
                )
                indexed = True
        if not indexed:
            if not self._try_link_repository_object(self.PAYLOAD_FILENAME, npz_destination):
                self._copy_repository_object_to_path(self.PAYLOAD_FILENAME, npz_destination)

        case_manifest = metadata.get("flow3d_case_manifest")
        sidecar_filename = None
        if isinstance(case_manifest, dict) and case_manifest:
            # Multi-beam cases: the ingested manifest thinned the beam tables
            # (`beam_table_thinned`), so the per-beam paths were re-extracted
            # from the rendered prepin (velocity-table integration) and attached
            # as a MUTABLE extra -- attributes are sealed on stored nodes. Merge
            # them into the sidecar here so the training datasource can build
            # per-beam conditioning channels. Matched by beam index.
            case_manifest = merge_multibeam_paths(
                case_manifest, self.base.extras.get("multibeam_paths", None), case_name
            )
            sidecar_filename = f"{case_name}.manifest.json"
            (flat_root / sidecar_filename).write_text(
                json.dumps(case_manifest, indent=2, sort_keys=True), encoding="utf-8"
            )

        manifest = {
            "format": self.NPZ_RUNTIME_MANIFEST_VERSION,
            "source_kind": "simulation",
            "storage_mode": self.STORAGE_MODE_NPZ_FILE,
            "simulation_uuid": self.uuid,
            "simulation_pk": self.pk,
            "simulation_identifier": self.get_simulation_identifier(),
            "simulation_fingerprint": self.get_simulation_fingerprint(),
            "source_type": safe_attribute_get(self, "source_type"),
            "case_name": case_name,
            "npz_file": os.path.relpath(npz_destination, target_path),
            "npz_via_pack_index": indexed,
            "case_manifest_file": (
                os.path.relpath(flat_root / sidecar_filename, target_path) if sidecar_filename else None
            ),
            "array_aliases": self.get_array_aliases(),
            "simulation_metadata": metadata,
        }
        manifest_path = target_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest_path

    def export_runtime_arraydir(self, target_dir: str | Path, *, pack_index: list | None = None) -> Path:
        """Export a per-simulation runtime directory with manifest plus per-array shards."""
        self._require_data()
        target_path = Path(target_dir)
        if self.get_storage_mode() == self.STORAGE_MODE_NPZ_FILE:
            return self._export_runtime_npz_case(target_path, pack_index=pack_index)
        target_path.mkdir(parents=True, exist_ok=True)
        metadata = self.get_simulation_metadata()
        array_entries: list[dict[str, Any]] = []

        for entry in self.iter_runtime_array_entries():
            destination = target_path / entry["filename"]
            if destination.exists():
                destination.unlink()
            if not self._try_link_repository_object(entry["filename"], destination):
                self._copy_repository_object_to_path(entry["filename"], destination)
            array_entries.append(entry)

        time_values = self.get_time_values_s()
        time_values_filename = None
        if time_values is not None:
            time_values_filename = "time_values_s.npy"
            np.save(target_path / time_values_filename, np.asarray(time_values, dtype=np.float32))

        manifest = {
            "format": self.RUNTIME_MANIFEST_VERSION,
            "source_kind": "simulation",
            "simulation_uuid": self.uuid,
            "simulation_pk": self.pk,
            "simulation_identifier": self.get_simulation_identifier(),
            "simulation_fingerprint": self.get_simulation_fingerprint(),
            "source_type": safe_attribute_get(self, "source_type"),
            "array_aliases": self.get_array_aliases(),
            "array_entries": array_entries,
            "time_values_filename": time_values_filename,
            "simulation_metadata": metadata,
        }
        manifest_path = target_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest_path
