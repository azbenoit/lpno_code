"""Native AiiDA dataset node for PINO/OpenFOAM VTK tensors."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterator

import numpy as np
from numpy.lib import format as npformat

from aiida import orm


def safe_attribute_get(node: orm.Node, key: str, default=None):
    """Return an AiiDA node attribute while tolerating missing keys on older backends."""
    try:
        value = node.base.attributes.get(key)
    except (AttributeError, KeyError):
        return default
    return default if value is None else value


def sanitize_array_name(name: str) -> str:
    """Convert an arbitrary field name into a valid `ArrayData` array name."""
    sanitized = re.sub(r"[^0-9A-Za-z_]", "_", name.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "array"


def sanitize_array_names(names: list[str]) -> dict[str, str]:
    """Create a collision-free original->stored array-name mapping."""
    mapping: dict[str, str] = {}
    used: set[str] = set()

    for original in names:
        candidate = sanitize_array_name(original)
        if candidate not in used:
            mapping[original] = candidate
            used.add(candidate)
            continue

        index = 2
        while f"{candidate}_{index}" in used:
            index += 1
        unique = f"{candidate}_{index}"
        mapping[original] = unique
        used.add(unique)

    return mapping


def build_dataset_label(folder_name: str, cache_basename: str, source_type: str) -> str:
    """Build a human-readable label for a dataset node."""
    cache_stem = Path(cache_basename).stem
    return f"{folder_name} | {cache_stem} | {source_type}"


def _stable_short_hash(payload: dict) -> str:
    """Return a short stable hash for dataset-family identity."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:10]


def build_family_identifier(
    folder_name: str,
    source_type: str,
    *,
    grid_params: dict | None = None,
    requested_fields: list[str] | None = None,
    source_uri: str = "",
) -> str:
    """Build a stable family identifier shared by a canonical base and its derivatives."""
    stem = sanitize_array_name(folder_name)
    source = sanitize_array_name(source_type)
    fingerprint = _stable_short_hash(
        {
            "folder_name": folder_name,
            "source_type": source_type,
            "grid_params": grid_params or {},
            "requested_fields": requested_fields or [],
            "source_uri": source_uri,
        }
    )
    return f"{stem}__{source}__{fingerprint}"


def build_dataset_identifier(
    folder_name: str,
    cache_basename: str,
    source_type: str,
    *,
    family_identifier: str | None = None,
) -> str:
    """Build a stable identifier for a dataset node."""
    variant_token = sanitize_array_name(Path(cache_basename).stem)
    family = family_identifier or build_family_identifier(folder_name, source_type)
    return f"{family}__{variant_token}"


def datasource_cache_basename_from_metadata(metadata: dict) -> str:
    """Return the cache basename expected by `openFoam_datasource.parse_directory`."""
    super_res_factor = int(metadata.get("super_res_factor", 1) or 1)
    time_axis_mode = metadata.get("time_axis_mode", {}) or {}
    derivation_profile = metadata.get("derivation_profile", {}) or {}
    include_time = bool(time_axis_mode.get("include_time", metadata.get("time_axis", False)))
    time_as_batch = bool(time_axis_mode.get("time_as_batch", derivation_profile.get("time_as_batch", False)))
    time_as_dim = bool(time_axis_mode.get("time_as_dim", derivation_profile.get("time_as_dim", False)))
    moving_ref_frame = bool(metadata.get("moving_ref_frame", derivation_profile.get("moving_ref_frame", False)))
    moving_frame_steps = int(metadata.get("moving_frame_steps", derivation_profile.get("moving_frame_steps", 1)) or 1)
    avg_window_size = int(metadata.get("avg_window_size", derivation_profile.get("time_avg_window", 1)) or 1)

    if not include_time:
        return f"var_{super_res_factor}.npz"
    if time_as_batch or time_as_dim or not moving_ref_frame:
        return f"var_{super_res_factor}_time_noref.npz"
    return f"var_{super_res_factor}_time_steps{moving_frame_steps}_avg{avg_window_size}.npz"


SUPPORTED_SOURCE_TYPES = ("flow3d", "openfoam", "openfoam_raw")


def validate_source_type(source_type: str) -> str:
    """Validate a dataset/simulation `source_type` against the supported set."""
    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(
            f"Unsupported source_type `{source_type}`; supported source types: "
            f"{', '.join(SUPPORTED_SOURCE_TYPES)}."
        )
    return source_type


def is_canonical_metadata(metadata: dict) -> bool:
    """Return whether the metadata describes the canonical base-dataset layout."""
    time_axis_mode = metadata.get("time_axis_mode", {})
    return (
        bool(metadata.get("time_axis"))
        and bool(time_axis_mode.get("include_time", False))
        # canonical is time_as_batch: time_as_dim needs uniform frame counts and
        # cannot represent ragged data. Kept in step with
        # DEFAULT_DERIVATION_PROFILE -- the two must never disagree.
        and bool(time_axis_mode.get("time_as_batch", False))
        and not bool(time_axis_mode.get("time_as_dim", False))
        and not bool(metadata.get("moving_ref_frame", False))
        and int(metadata.get("moving_frame_steps", 1)) == 1
        and int(metadata.get("avg_window_size", 1)) == 1
        and int(metadata.get("super_res_factor", 1)) == 1
    )


class PinoDatasetData(orm.ArrayData):
    """Store a parsed PINO dataset as named arrays plus queryable metadata."""

    FORMAT_VERSION = 1
    RUNTIME_MANIFEST_VERSION = "aiida_arraydir_v1"
    DEFAULT_SOURCE_TYPE = "flow3d"
    SUPPORTED_SOURCE_TYPES = SUPPORTED_SOURCE_TYPES
    DEFAULT_DATASET_ROLE = "base"
    CANONICAL_LAYOUT = "time_batch_noavg_noref"
    DATASET_ROLES = ("base", "derived")
    DEFAULT_BASE_CACHE_BASENAME = "var_1_time_noref.npz"

    @classmethod
    def from_arrays(
        cls,
        arrays: dict[str, np.ndarray],
        *,
        metadata: dict,
        source: dict | None = None,
    ) -> "PinoDatasetData":
        """Create a dataset node from original-name arrays and metadata."""
        original_names = list(arrays)
        aliases = sanitize_array_names(original_names)

        node = cls()
        for original_name, array in arrays.items():
            node.set_array(aliases[original_name], np.asarray(array))

        normalized = dict(metadata)
        normalized.setdefault("format_version", cls.FORMAT_VERSION)
        normalized.setdefault("source_type", cls.DEFAULT_SOURCE_TYPE)
        validate_source_type(str(normalized["source_type"]))
        normalized.setdefault("dataset_role", cls.DEFAULT_DATASET_ROLE)
        normalized.setdefault("source_folder_name", "dataset")
        normalized.setdefault("requested_fields", [])
        normalized.setdefault("input_dir", "")
        normalized.setdefault("source_uri", source.get("uri") if source else "")
        normalized.setdefault("cache_basename", datasource_cache_basename_from_metadata(normalized))
        normalized.setdefault(
            "family_identifier",
            build_family_identifier(
                normalized["source_folder_name"],
                normalized["source_type"],
                grid_params=normalized.get("grid_params"),
                requested_fields=list(normalized.get("requested_fields", [])),
                source_uri=str(normalized.get("source_uri", "")),
            ),
        )
        normalized.setdefault("canonical_layout", cls.CANONICAL_LAYOUT if is_canonical_metadata(normalized) else "")
        normalized.setdefault("base_dataset_uuid", "")
        normalized.setdefault("base_dataset_pk", None)
        normalized.setdefault("parent_dataset_uuid", "")
        normalized.setdefault("parent_dataset_pk", None)
        normalized.setdefault("derivation_profile", {})
        normalized.setdefault("derivation_hash", "")
        normalized.setdefault("reconstructable", normalized["dataset_role"] == "derived")
        normalized["array_aliases"] = aliases
        normalized["array_shapes"] = {
            original_name: list(np.asarray(array).shape) for original_name, array in arrays.items()
        }
        normalized["dataset_identifier"] = normalized.get(
            "dataset_identifier",
            build_dataset_identifier(
                normalized["source_folder_name"],
                normalized["cache_basename"],
                normalized["source_type"],
                family_identifier=normalized["family_identifier"],
            ),
        )

        node.base.attributes.set("format_version", normalized["format_version"])
        node.base.attributes.set("source_type", normalized["source_type"])
        node.base.attributes.set("dataset_role", normalized["dataset_role"])
        node.base.attributes.set("canonical_layout", normalized["canonical_layout"])
        node.base.attributes.set("family_identifier", normalized["family_identifier"])
        node.base.attributes.set("cache_basename", normalized["cache_basename"])
        node.base.attributes.set("source_folder_name", normalized["source_folder_name"])
        node.base.attributes.set("dataset_identifier", normalized["dataset_identifier"])
        node.base.attributes.set("requested_fields", normalized["requested_fields"])
        node.base.attributes.set("input_dir", normalized["input_dir"])
        node.base.attributes.set("source_uri", normalized["source_uri"])
        node.base.attributes.set("base_dataset_uuid", normalized["base_dataset_uuid"])
        node.base.attributes.set("base_dataset_pk", normalized["base_dataset_pk"])
        node.base.attributes.set("parent_dataset_uuid", normalized["parent_dataset_uuid"])
        node.base.attributes.set("parent_dataset_pk", normalized["parent_dataset_pk"])
        node.base.attributes.set("derivation_profile", normalized["derivation_profile"])
        node.base.attributes.set("derivation_hash", normalized["derivation_hash"])
        node.base.attributes.set("reconstructable", normalized["reconstructable"])
        node.base.attributes.set("array_aliases", aliases)
        node.base.attributes.set("array_shapes", normalized["array_shapes"])
        node.base.attributes.set("dataset_metadata", normalized)
        node.label = build_dataset_label(
            normalized["source_folder_name"],
            normalized["cache_basename"],
            normalized["source_type"],
        )
        if source is not None:
            node.source = source
        node.base.extras.set("aiida_pino_kind", "dataset")
        node.base.extras.set("source_type", normalized["source_type"])
        node.base.extras.set("dataset_identifier", normalized["dataset_identifier"])
        node.base.extras.set("dataset_role", normalized["dataset_role"])
        node.base.extras.set("family_identifier", normalized["family_identifier"])
        node.base.extras.set("canonical_layout", normalized["canonical_layout"])
        node.base.extras.set("reconstructable", normalized["reconstructable"])
        node.base.extras.set("derivation_hash", normalized["derivation_hash"])
        if normalized["base_dataset_uuid"]:
            node.base.extras.set("base_dataset_uuid", normalized["base_dataset_uuid"])
        if normalized["parent_dataset_uuid"]:
            node.base.extras.set("parent_dataset_uuid", normalized["parent_dataset_uuid"])
        if normalized.get("source_folder_uuid"):
            node.base.extras.set("source_folder_uuid", normalized["source_folder_uuid"])
        if normalized.get("source_folder_pk") is not None:
            node.base.extras.set("source_folder_pk", normalized["source_folder_pk"])
        return node

    @classmethod
    def from_npz_file(
        cls,
        npz_path: str | Path,
        *,
        metadata: dict,
        source: dict | None = None,
    ) -> "PinoDatasetData":
        """Create a dataset node directly from a compressed NPZ payload."""
        path = Path(npz_path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        return cls.from_arrays(arrays, metadata=metadata, source=source)

    def get_dataset_metadata(self) -> dict:
        """Return the full dataset metadata payload."""
        return dict(safe_attribute_get(self, "dataset_metadata", {}))

    def get_array_aliases(self) -> dict[str, str]:
        """Return the original->stored array-name mapping."""
        return dict(safe_attribute_get(self, "array_aliases", {}))

    def list_original_array_names(self) -> list[str]:
        """Return the original field names stored in the dataset."""
        return list(self.get_array_aliases())

    def get_array_shapes(self) -> dict[str, list[int]]:
        """Return original-name array shapes."""
        return dict(safe_attribute_get(self, "array_shapes", {}))

    def get_dataset_role(self) -> str:
        """Return the dataset role."""
        return str(safe_attribute_get(self, "dataset_role", self.DEFAULT_DATASET_ROLE))

    def get_family_identifier(self) -> str:
        """Return the dataset family identifier."""
        return str(safe_attribute_get(self, "family_identifier", ""))

    def is_base_dataset(self) -> bool:
        """Return whether this node is a base dataset."""
        return self.get_dataset_role() == "base"

    def is_derived_dataset(self) -> bool:
        """Return whether this node is a derived dataset."""
        return self.get_dataset_role() == "derived"

    def get_array_by_original_name(self, name: str) -> np.ndarray:
        """Load an array by its original field name."""
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
        """Copy a repository object to ``destination`` using streamed IO."""
        with self.base.repository.open(filename, mode="rb") as handle:
            with destination.open("wb") as target:
                shutil.copyfileobj(handle, target, length=1024 * 1024 * 16)

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
        """Yield manifest entries for all stored arrays."""
        aliases = self.get_array_aliases()
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

    def export_runtime_arraydir(self, target_dir: str | Path) -> Path:
        """Export a native runtime directory with manifest plus per-array ``.npy`` shards."""
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        metadata = self.get_dataset_metadata()
        array_entries: list[dict[str, Any]] = []

        for entry in self.iter_runtime_array_entries():
            destination = target_path / entry["filename"]
            if destination.exists():
                destination.unlink()
            if not self._try_link_repository_object(entry["filename"], destination):
                self._copy_repository_object_to_path(entry["filename"], destination)
            array_entries.append(entry)

        manifest = {
            "format": self.RUNTIME_MANIFEST_VERSION,
            "source_kind": "dataset",
            "dataset_uuid": self.uuid,
            "dataset_pk": self.pk,
            "dataset_identifier": safe_attribute_get(self, "dataset_identifier"),
            "family_identifier": safe_attribute_get(self, "family_identifier"),
            "dataset_role": safe_attribute_get(self, "dataset_role"),
            "canonical_layout": safe_attribute_get(self, "canonical_layout"),
            "source_type": safe_attribute_get(self, "source_type"),
            "array_aliases": self.get_array_aliases(),
            "array_entries": array_entries,
            "dataset_metadata": metadata,
        }
        manifest_path = target_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest_path

    def export_npz_dataset(self, target_dir: str | Path) -> Path:
        """Export the dataset to a temporary NPZ directory for the current PINO runtime."""
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        metadata = self.get_dataset_metadata()
        cache_basename = str(metadata.get("cache_basename", "var_1.npz"))
        npz_path = target_path / cache_basename
        arrays = {name: array for name, array in self.iter_original_arrays()}
        np.savez_compressed(npz_path, **arrays)
        runtime_basename = datasource_cache_basename_from_metadata(metadata)
        cache_files = [npz_path.name]
        if runtime_basename != npz_path.name:
            runtime_path = target_path / runtime_basename
            try:
                runtime_path.hardlink_to(npz_path)
            except OSError:
                shutil.copy2(npz_path, runtime_path)
            cache_files.append(runtime_path.name)

        manifest = {
            "cache_files": cache_files,
            "primary_cache_file": npz_path.name,
            "runtime_cache_file": runtime_basename,
            "dataset_identifier": safe_attribute_get(self, "dataset_identifier"),
            "family_identifier": safe_attribute_get(self, "family_identifier"),
            "dataset_role": safe_attribute_get(self, "dataset_role"),
            "source_type": safe_attribute_get(self, "source_type"),
            "array_shapes": safe_attribute_get(self, "array_shapes"),
            "dataset_metadata": metadata,
        }
        (target_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return npz_path


PINO_DATASET_INPUT_TYPES = (PinoDatasetData, orm.FolderData)
