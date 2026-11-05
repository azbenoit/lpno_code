"""Lazy runtime readers and dataset builders for staged array-directory datasets."""

from __future__ import annotations

from bisect import bisect_right
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from physicsnemo.sym.dataset.dataset import Dataset


RUNTIME_FORMAT_V1 = "aiida_arraydir_v1"
RUNTIME_FORMAT_V2 = "aiida_arraydir_v2"
# aiida-pino stages a collection whose members are stored as whole case NPZs by
# hardlinking each `<case>.npz` flat into the collection root and tagging the member
# manifest with this format. The root manifest still says arraydir_v2, so the format
# to dispatch on lives one level down, in the members.
RUNTIME_FORMAT_NPZ_FILE = "aiida_npz_file_v1"
_UNMASKED_INPUT_KEYS = {"x", "y", "z", "P", "V_scan", "time_s", "t_hat", "H_norm"}


def is_arraydir_runtime(input_dir: str | Path) -> bool:
    """Return whether ``input_dir`` is a staged native arraydir runtime."""
    manifest_path = Path(input_dir) / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("format") in {RUNTIME_FORMAT_V1, RUNTIME_FORMAT_V2}


def _runtime_dataset_metadata(input_dir: str | Path) -> dict:
    """`dataset_metadata` of a staged runtime manifest (member `simulation_metadata`
    for a single-simulation runtime), or {} when there is none to read."""
    manifest_path = Path(input_dir) / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if payload.get("source_kind") == "simulation":
        return dict(payload.get("simulation_metadata") or {})
    return dict(payload.get("dataset_metadata") or {})


def runtime_grid_params(input_dir: str | Path) -> dict | None:
    """The rectangular grid a staged arraydir runtime actually SERVES, from its manifest.

    aiida-pino records it at export (`dataset_metadata.grid_params`: ranges plus
    `resolution`, and `spatial_shape`), and it is the post-derivation grid -- for a
    moving-frame variant the x extent is already the window (61 from a raw 90 at 30
    steps). Nothing in lpno read it: the arraydir branch of the loader
    serves arrays at their native shape without ever consulting `custom.geometry`, so
    a config's hardcoded `resolution: [90, 40, 30]` survived into `run_cfg` while the
    tensors were 61x60x55 (anchorbox steady arms m82910/m83376/m83672, 2026-09-01),
    which put every reader of `n{x,y,z}_effective` -- validator slice indices, the
    FNO Nyquist cap, bounds/dx -- on a grid the data did not have.

    None when the manifest carries no usable grid; the caller then keeps its config.
    """
    meta = _runtime_dataset_metadata(input_dir)
    grid = meta.get("grid_params")
    if not isinstance(grid, dict):
        return None
    try:
        res = grid.get("resolution") or meta.get("spatial_shape")
        out = {
            "x_range": tuple(float(v) for v in grid["x_range"]),
            "y_range": tuple(float(v) for v in grid["y_range"]),
            "z_range": tuple(float(v) for v in grid["z_range"]),
            "resolution": tuple(int(v) for v in res),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if len(out["resolution"]) != 3:
        return None
    return out


def runtime_window_is_baked(input_dir: str | Path) -> bool:
    """Whether the staged runtime already holds the beam-following x-window.

    A derived variant with `moving_ref_frame` and `moving_frame_steps > 1` was exported
    with the window applied, so its served x extent is already `nx - (steps - 1)`. The
    config-time shift that reproduces that reduction for raw data must then NOT be
    applied a second time -- the twin of `_variant_stride`/`_variant_crop_z_top` in
    `npz_lazy_store`, and for the same reason.
    """
    meta = _runtime_dataset_metadata(input_dir)
    profile = meta.get("derivation_profile") if isinstance(meta.get("derivation_profile"), dict) else {}
    moving = meta.get("moving_ref_frame", profile.get("moving_ref_frame"))
    steps = meta.get("moving_frame_steps", profile.get("moving_frame_steps"))
    try:
        return bool(moving) and int(steps or 1) > 1
    except (TypeError, ValueError):
        return False


def is_npz_file_collection(input_dir: str | Path) -> bool:
    """Return whether a staged collection's members are whole case NPZs.

    Such a collection carries its cases as flat `<case>.npz` at the root, which is
    the layout `NpzDirStore` reads -- and that store is also the only one that
    synthesizes the laser-path channels, so the loader routes these there instead
    of to `ArrayDirCollectionStore` (which cannot parse the v1 member manifest).
    """
    root = Path(input_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if payload.get("source_kind") != "dataset_collection":
        return False
    members = payload.get("members") or []
    if not members:
        return False
    # Members are staged uniformly, so the first one settles the format.
    member_manifest = root / str(members[0].get("manifest_file", ""))
    try:
        member_payload = json.loads(member_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return member_payload.get("format") == RUNTIME_FORMAT_NPZ_FILE


#: Path channels that are constant per case and therefore reconstructible from the case
#: manifest alone, with no beam kinematics. The trajectory channels (beam_x, beam_dist,
#: path_dose, ...) are NOT here: they need a per-frame beam path a staged derived member
#: does not carry, and must fail by name rather than be approximated.
_CONSTANT_PATH_CHANNELS = ("preheat_T0", "beam_shape_uniform", "focus_factor",
                           "beam_radius")
#: kept identical to npz_datasource.CONSTANT_PATH_CHANNELS; pinned by test.


class ArrayDirDatasetStore:
    """Lazy reader for one staged dataset manifest."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format") != RUNTIME_FORMAT_V1:
            raise ValueError(f"Unsupported runtime format in {self.root}: {self.manifest.get('format')}")
        if self.manifest.get("source_kind") != "dataset":
            raise ValueError(f"{self.root} is not a dataset runtime.")
        self.metadata = dict(self.manifest.get("dataset_metadata", {}))
        self.array_aliases = dict(self.manifest.get("array_aliases", {}))
        self.array_entries = {
            entry["original_name"]: dict(entry)
            for entry in self.manifest.get("array_entries", [])
        }
        if not self.array_entries:
            raise ValueError(f"No array entries found in {self.root}/manifest.json")
        self._cache: dict[str, np.ndarray] = {}

    def list_array_names(self) -> list[str]:
        return list(self.array_entries)

    def array_shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.array_entries[name]["shape"])

    def sample_count(self) -> int:
        first = next(iter(self.array_entries))
        return int(self.array_shape(first)[0])

    def has_time_axis(self) -> bool:
        return bool(self.metadata.get("time_axis")) or bool(
            (self.metadata.get("time_axis_mode") or {}).get("include_time", False)
        )

    def time_steps(self, case_idx: int | None = None) -> int:
        del case_idx
        first = next(iter(self.array_entries))
        shape = self.array_shape(first)
        if self.has_time_axis() and len(shape) >= 2:
            return int(shape[1])
        return 1

    def has_uniform_timesteps(self, case_indices: Iterable[int] | None = None) -> bool:
        del case_indices
        return True

    def spatial_shape(self) -> tuple[int, ...]:
        first = next(iter(self.array_entries))
        shape = self.array_shape(first)
        if self.has_time_axis():
            return tuple(shape[3:])
        return tuple(shape[2:])

    def _array_path(self, name: str) -> Path:
        return self.root / self.array_entries[name]["filename"]

    def get_array(self, name: str) -> np.ndarray:
        if name not in self._cache:
            self._cache[name] = np.load(self._array_path(name), mmap_mode="r", allow_pickle=False)
        return self._cache[name]

    def scalar_value(self, name: str, case_idx: int, time_idx: int | None = None) -> float:
        arr = self.get_array(name)
        if self.has_time_axis():
            t_idx = 0 if time_idx is None else int(time_idx)
            tail = (0,) * max(0, arr.ndim - 2)
            return float(arr[(case_idx, t_idx) + tail])
        tail = (0,) * max(0, arr.ndim - 1)
        return float(arr[(case_idx,) + tail])

    def read_case(
        self,
        case_idx: int,
        *,
        time_idx: int | None = None,
        spatial_slices: tuple[slice, ...] | None = None,
        keys: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Read one case or one case/timestep into small in-memory numpy arrays."""
        selected = list(keys or self.list_array_names())
        payload: dict[str, np.ndarray] = {}
        for name in selected:
            arr = self.get_array(name)
            if self.has_time_axis():
                if time_idx is None:
                    sample = np.asarray(arr[case_idx : case_idx + 1], dtype=np.float32)
                    if spatial_slices is not None:
                        sample = sample[(slice(None), slice(None), slice(None)) + spatial_slices]
                else:
                    sample = np.asarray(arr[case_idx, time_idx], dtype=np.float32)[np.newaxis, ...]
                    if spatial_slices is not None:
                        sample = sample[(slice(None), slice(None)) + spatial_slices]
            else:
                sample = np.asarray(arr[case_idx], dtype=np.float32)[np.newaxis, ...]
                if spatial_slices is not None:
                    sample = sample[(slice(None), slice(None)) + spatial_slices]
            payload[name] = sample
        return payload


class ArrayDirUnionStore:
    """Lazy reader for a staged dataset-union manifest."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format") != RUNTIME_FORMAT_V1:
            raise ValueError(f"Unsupported runtime format in {self.root}: {self.manifest.get('format')}")
        if self.manifest.get("source_kind") != "dataset_union":
            raise ValueError(f"{self.root} is not a dataset-union runtime.")
        member_entries = list(self.manifest.get("members", []))
        if not member_entries:
            raise ValueError(f"No members found in union runtime {self.root}")
        self.members: list[ArrayDirDatasetStore] = []
        self.offsets: list[int] = []
        total = 0
        for member in member_entries:
            member_root = self.root / member["relative_path"]
            self.members.append(ArrayDirDatasetStore(member_root))
            self.offsets.append(total)
            total += int(member["sample_count"])
        self.total_samples = total
        self.metadata = dict(self.members[0].metadata)

    def list_array_names(self) -> list[str]:
        return self.members[0].list_array_names()

    def array_shape(self, name: str) -> tuple[int, ...]:
        """Return the logical union shape without concatenating member arrays."""
        member_shape = self.members[0].array_shape(name)
        return (self.total_samples, *tuple(member_shape[1:]))

    def sample_count(self) -> int:
        return self.total_samples

    def has_time_axis(self) -> bool:
        return self.members[0].has_time_axis()

    def time_steps(self, case_idx: int | None = None) -> int:
        if case_idx is None:
            return self.members[0].time_steps()
        member, local_idx = self._resolve_case(case_idx)
        return member.time_steps(case_idx=local_idx)

    def has_uniform_timesteps(self, case_indices: Iterable[int] | None = None) -> bool:
        indices = list(case_indices) if case_indices is not None else list(range(self.total_samples))
        if not indices:
            return True
        counts = {self.time_steps(case_idx=index) for index in indices}
        return len(counts) <= 1

    def spatial_shape(self) -> tuple[int, ...]:
        return self.members[0].spatial_shape()

    def _resolve_case(self, case_idx: int) -> tuple[ArrayDirDatasetStore, int]:
        if case_idx < 0 or case_idx >= self.total_samples:
            raise IndexError(case_idx)
        member_idx = bisect_right(self.offsets, case_idx) - 1
        if member_idx < 0:
            member_idx = 0
        local_idx = case_idx - self.offsets[member_idx]
        return self.members[member_idx], local_idx

    def scalar_value(self, name: str, case_idx: int, time_idx: int | None = None) -> float:
        member, local_idx = self._resolve_case(case_idx)
        return member.scalar_value(name, local_idx, time_idx=time_idx)

    def read_case(
        self,
        case_idx: int,
        *,
        time_idx: int | None = None,
        spatial_slices: tuple[slice, ...] | None = None,
        keys: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        member, local_idx = self._resolve_case(case_idx)
        return member.read_case(local_idx, time_idx=time_idx, spatial_slices=spatial_slices, keys=keys)


class ArrayDirSimulationStore:
    """Lazy reader for one staged simulation manifest."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format") != RUNTIME_FORMAT_V2:
            raise ValueError(f"Unsupported runtime format in {self.root}: {self.manifest.get('format')}")
        if self.manifest.get("source_kind") != "simulation":
            raise ValueError(f"{self.root} is not a simulation runtime.")
        self.metadata = dict(self.manifest.get("simulation_metadata", {}))
        self.array_aliases = dict(self.manifest.get("array_aliases", {}))
        self.array_entries = {
            entry["original_name"]: dict(entry)
            for entry in self.manifest.get("array_entries", [])
        }
        if not self.array_entries:
            raise ValueError(f"No array entries found in {self.root}/manifest.json")
        time_filename = self.manifest.get("time_values_filename")
        self.time_values = None
        if time_filename:
            self.time_values = np.load(self.root / time_filename, mmap_mode="r", allow_pickle=False)
        self._cache: dict[str, np.ndarray] = {}

    def list_array_names(self) -> list[str]:
        return list(self.array_entries)

    def array_shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.array_entries[name]["shape"])

    def sample_count(self) -> int:
        return 1

    def has_time_axis(self) -> bool:
        if self.time_values is not None:
            return True
        first = next(iter(self.array_entries))
        return len(self.array_shape(first)) >= 5

    def time_steps(self, case_idx: int | None = None) -> int:
        del case_idx
        if self.time_values is not None:
            return int(self.time_values.shape[0])
        first = next(iter(self.array_entries))
        shape = self.array_shape(first)
        if len(shape) >= 5:
            return int(shape[0])
        return 1

    def has_uniform_timesteps(self, case_indices: Iterable[int] | None = None) -> bool:
        del case_indices
        return True

    def spatial_shape(self) -> tuple[int, ...]:
        first = next(iter(self.array_entries))
        shape = self.array_shape(first)
        return tuple(shape[-3:])

    def _array_path(self, name: str) -> Path:
        return self.root / self.array_entries[name]["filename"]

    def get_array(self, name: str) -> np.ndarray:
        if name not in self._cache:
            self._cache[name] = np.load(self._array_path(name), mmap_mode="r", allow_pickle=False)
        return self._cache[name]

    def _constant_channel_scalar(self, name: str) -> float | None:
        """Value of a case-CONSTANT path channel, synthesised from the case manifest.

        These channels are never stored as arrays -- read_case builds them on demand --
        so every SCALAR read path used to raise KeyError on them. compute_h_norm_denom
        needs beam_radius as a cheap scalar, and that KeyError (swallowed by a bare
        except) is what left a radius-blind denominator against a radius-carrying
        numerator on 2026-08-21. Returns None when the name is not such a channel or the
        member carries no case manifest, so the caller still raises for a genuinely
        unknown name. Pinned by tests/test_constant_channel_scalar.py.
        """
        if name not in _CONSTANT_PATH_CHANNELS:
            return None
        case_manifest = self.metadata.get("flow3d_case_manifest")
        if not isinstance(case_manifest, dict):
            return None
        from lpno.custom_utils.npz_datasource import (
            build_constant_path_channels,
        )

        built = build_constant_path_channels([name], case_manifest, (1, 1, 1, 1, 1))
        if name not in built:
            return None
        return float(np.asarray(built[name]).reshape(-1)[0])

    def scalar_value(self, name: str, case_idx: int, time_idx: int | None = None) -> float:
        if case_idx != 0:
            raise IndexError(case_idx)
        if name not in self.array_entries:
            # Stored arrays win; only fall back for the synthesised constant channels.
            constant = self._constant_channel_scalar(name)
            if constant is not None:
                return constant
            # Material channels too: compute_h_norm_denom reads the six through here when
            # `h_norm_include_material` is set, and get_array cannot serve a channel that
            # is synthesised from the manifest.
            material = self._material_channel_scalar(name)
            if material is not None:
                return material
        arr = self.get_array(name)
        if self.has_time_axis() and arr.ndim >= 5:
            t_idx = 0 if time_idx is None else int(time_idx)
            tail = (0,) * max(0, arr.ndim - 1)
            return float(arr[(t_idx,) + tail])
        tail = (0,) * max(0, arr.ndim)
        return float(arr[tail])

    def _material_channel_scalar(self, name: str) -> float | None:
        """A material channel's per-case value from the staged manifest, or None."""
        from lpno.custom_utils.npz_datasource import (
            ALL_MATERIAL_CHANNEL_NAMES,
            _build_material_channels,
        )

        if name not in ALL_MATERIAL_CHANNEL_NAMES:
            return None
        manifest = self._material_manifest()
        if manifest is None:
            return None
        return float(_build_material_channels([name], manifest, (1,))[name].reshape(-1)[0])

    def _material_manifest(self) -> dict | None:
        """FLOW-3D case manifest carried in the staged member metadata, if any.

        AiiDA-staged members store the full case manifest under
        ``simulation_metadata.flow3d_case_manifest`` (with the material payload
        either inline or as the sibling ``flow3d_material``); it is the same
        dict the sidecar-manifest path feeds ``_build_material_channels``.
        """
        metadata = self.metadata
        manifest = metadata.get("flow3d_case_manifest")
        if not isinstance(manifest, dict):
            return None
        if "material" not in manifest and isinstance(metadata.get("flow3d_material"), dict):
            manifest = {**manifest, "material": metadata["flow3d_material"]}
        return manifest if "material" in manifest else None

    def read_case(
        self,
        case_idx: int,
        *,
        time_idx: int | None = None,
        spatial_slices: tuple[slice, ...] | None = None,
        keys: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        if case_idx != 0:
            raise IndexError(case_idx)
        from lpno.custom_utils.npz_datasource import (
            MATERIAL_CHANNEL_NAMES,
            MATERIAL_LABEL_CHANNEL_NAMES,
            _build_material_channels,
        )

        # The manifest six plus the alloy label, both built from the manifest alone. A
        # keyless read serves both; the label reaches the network only when a config
        # declares it as an input.
        served = MATERIAL_CHANNEL_NAMES + MATERIAL_LABEL_CHANNEL_NAMES

        # A keyless read is the training sample builder's call. It must include the
        # constants the manifest can supply, or they are never requested and the builder
        # dies later with a bare KeyError (run 58791: exit 300 on preheat_T0 AFTER this
        # store had been taught to build it). Only the constants -- a keyless read must
        # not invent trajectory channels it cannot reconstruct.
        if keys is None:
            selected = list(self.list_array_names())
            if isinstance(self.metadata.get("flow3d_case_manifest"), dict):
                selected += [c for c in _CONSTANT_PATH_CHANNELS if c not in selected]
        else:
            selected = list(keys)
        # Material channels are synthesized constants, not stored arrays —
        # split them out so the array loop below never KeyErrors on them.
        material_requested = [name for name in selected if name in served]
        selected = [name for name in selected if name not in served]
        # Case-CONSTANT path channels are the same shape of thing: synthesised from the
        # manifest, needing no per-frame kinematics. `materialize_derived_dataset` drops
        # every path channel, which on the preheat axis is fatal -- without preheat_T0 two
        # cases at the same (P, V) and different initial temperature present identical
        # inputs, so the axis is unlearnable. The trajectory channels genuinely cannot be
        # rebuilt here (no beam path per frame) and are left to fail by name below.
        constant_requested = [n for n in selected if n in _CONSTANT_PATH_CHANNELS]
        selected = [n for n in selected if n not in _CONSTANT_PATH_CHANNELS]
        payload: dict[str, np.ndarray] = {}
        for name in selected:
            arr = self.get_array(name)
            if self.has_time_axis() and arr.ndim >= 5:
                if time_idx is None:
                    sample = np.asarray(arr, dtype=np.float32)[np.newaxis, ...]
                    if spatial_slices is not None:
                        sample = sample[(slice(None), slice(None), slice(None)) + spatial_slices]
                else:
                    sample = np.asarray(arr[int(time_idx)], dtype=np.float32)[np.newaxis, ...]
                    if spatial_slices is not None:
                        sample = sample[(slice(None), slice(None)) + spatial_slices]
            else:
                sample = np.asarray(arr, dtype=np.float32)[np.newaxis, ...]
                if spatial_slices is not None:
                    sample = sample[(slice(None), slice(None)) + spatial_slices]
            payload[name] = sample

        if constant_requested:
            case_manifest = self.metadata.get("flow3d_case_manifest")
            if not isinstance(case_manifest, dict):
                raise KeyError(
                    f"Constant path channels {constant_requested} requested but the staged "
                    f"member at {self.root} carries no flow3d_case_manifest. Refusing rather "
                    "than defaulting: a silent 293.15 K would train a preheat sweep on one "
                    "temperature."
                )
            reference = next(
                (v for k, v in payload.items() if k not in ("Velocity", "U")),
                next(iter(payload.values()), None),
            )
            if reference is None:
                raise KeyError(
                    f"Constant path channels {constant_requested} need at least one stored "
                    f"array to take their shape from; none was requested."
                )
            from lpno.custom_utils.npz_datasource import (
                build_constant_path_channels,
            )

            built = build_constant_path_channels(
                constant_requested, case_manifest, reference.shape)
            for name in constant_requested:
                if name not in built:
                    raise KeyError(
                        f"Constant path channel {name!r} could not be built from the "
                        f"manifest of {self.root}."
                    )
                payload[name] = built[name]

        manifest = self._material_manifest()
        if manifest is not None and payload:
            # Constant fields shaped like a scalar sample (Velocity carries an
            # extra component axis, so prefer any other reference field).
            reference = next(
                (value for key, value in payload.items() if key not in ("Velocity", "U")),
                next(iter(payload.values())),
            )
            shape = reference.shape
            if material_requested:
                payload.update(_build_material_channels(material_requested, manifest, shape))
            else:
                payload.update(_build_material_channels(list(MATERIAL_CHANNEL_NAMES), manifest, shape))
                try:
                    payload.update(_build_material_channels(
                        list(MATERIAL_LABEL_CHANNEL_NAMES), manifest, shape))
                except KeyError:
                    # A material block without a name gets no label. Runs that never
                    # declare the label are unaffected; one that does fails at input
                    # assembly, by name.
                    pass
        elif material_requested:
            raise KeyError(
                f"Material channels {material_requested} requested but the staged member "
                f"at {self.root} carries no flow3d_case_manifest material payload."
            )
        return payload


class ArrayDirCollectionStore:
    """Lazy reader for a staged dataset-collection manifest."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format") != RUNTIME_FORMAT_V2:
            raise ValueError(f"Unsupported runtime format in {self.root}: {self.manifest.get('format')}")
        if self.manifest.get("source_kind") != "dataset_collection":
            raise ValueError(f"{self.root} is not a dataset-collection runtime.")
        member_entries = list(self.manifest.get("members", []))
        if not member_entries:
            raise ValueError(f"No members found in collection runtime {self.root}")
        self.members = [ArrayDirSimulationStore(self.root / member["relative_path"]) for member in member_entries]
        self.total_samples = len(self.members)
        self.metadata = dict(self.manifest.get("dataset_metadata", {}))

    def list_array_names(self) -> list[str]:
        return self.members[0].list_array_names()

    def array_shape(self, name: str) -> tuple[int, ...]:
        """Return the logical collection shape without materializing the collection."""
        return (self.total_samples, *self.members[0].array_shape(name))

    def sample_count(self) -> int:
        return self.total_samples

    def has_time_axis(self) -> bool:
        return self.members[0].has_time_axis()

    def time_steps(self, case_idx: int | None = None) -> int:
        if case_idx is None:
            return max(member.time_steps() for member in self.members)
        if case_idx < 0 or case_idx >= self.total_samples:
            raise IndexError(case_idx)
        return self.members[case_idx].time_steps()

    def has_uniform_timesteps(self, case_indices: Iterable[int] | None = None) -> bool:
        indices = list(case_indices) if case_indices is not None else list(range(self.total_samples))
        if not indices:
            return True
        counts = {self.time_steps(case_idx=index) for index in indices}
        return len(counts) <= 1

    def spatial_shape(self) -> tuple[int, ...]:
        return self.members[0].spatial_shape()

    def _resolve_case(self, case_idx: int) -> ArrayDirSimulationStore:
        if case_idx < 0 or case_idx >= self.total_samples:
            raise IndexError(case_idx)
        return self.members[case_idx]

    def scalar_value(self, name: str, case_idx: int, time_idx: int | None = None) -> float:
        member = self._resolve_case(case_idx)
        return member.scalar_value(name, 0, time_idx=time_idx)

    def read_case(
        self,
        case_idx: int,
        *,
        time_idx: int | None = None,
        spatial_slices: tuple[slice, ...] | None = None,
        keys: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        member = self._resolve_case(case_idx)
        return member.read_case(0, time_idx=time_idx, spatial_slices=spatial_slices, keys=keys)


RuntimeStore = ArrayDirDatasetStore | ArrayDirUnionStore | ArrayDirSimulationStore | ArrayDirCollectionStore


def load_runtime_store(input_dir: str | Path) -> RuntimeStore:
    """Load a staged dataset, union, simulation, or collection runtime manifest."""
    manifest = json.loads((Path(input_dir) / "manifest.json").read_text(encoding="utf-8"))
    source_kind = manifest.get("source_kind")
    if source_kind == "dataset":
        return ArrayDirDatasetStore(input_dir)
    if source_kind == "dataset_union":
        return ArrayDirUnionStore(input_dir)
    if source_kind == "simulation":
        return ArrayDirSimulationStore(input_dir)
    if source_kind == "dataset_collection":
        return ArrayDirCollectionStore(input_dir)
    raise ValueError(f"Unsupported arraydir source kind: {source_kind}")


from ...custom_utils.utils import h_norm_raw, king_material_factor


def _radius(store: RuntimeStore, case_idx: int, time_idx) -> float:
    """Beam radius for one case. Raises rather than defaulting: a silent None here is
    exactly the asymmetry that mis-scaled every H_norm on 2026-08-21."""
    try:
        return store.scalar_value("beam_radius", int(case_idx), time_idx=time_idx)
    except KeyError as exc:  # a store type that cannot synthesise the channel
        raise ValueError(
            f"h_norm_include_radius=True but {type(store).__name__} cannot serve "
            f"'beam_radius' as a scalar, so the denominator would silently drop the r0 "
            f"term the numerator carries. Teach that store the constant path channels, "
            f"or turn the flag off."
        ) from exc


def compute_h_norm_denom(
    store: RuntimeStore,
    case_indices: Iterable[int],
    *,
    time_as_batch: bool,
    include_radius: bool = False,
    include_material: bool = False,
) -> float:
    """Compute the split-level H_norm denominator from lightweight scalar reads only.

    `include_radius` must match the numerator's `custom.h_norm_include_radius`; the two
    are divided by each other, so a mismatch is silent and catastrophic. The same holds
    for `include_material` / `custom.h_norm_include_material`.

    The material read goes through `store.scalar_value`, which RAISES for a channel it
    cannot serve. That is deliberate: a bare `except` here would hand back a
    material-blind denominator while the numerator carried the material term, which is
    exactly the asymmetry that put in_H_norm = 1.28e6 into the 2026-08-21 evals.
    """
    max_value = 0.0
    for case_idx in case_indices:
        if time_as_batch and store.has_time_axis():
            time_indices = range(store.time_steps(case_idx=int(case_idx)))
        else:
            time_indices = (None,)
        for time_idx in time_indices:
            power = store.scalar_value("P", int(case_idx), time_idx=time_idx)
            v_scan = abs(store.scalar_value("V_scan", int(case_idx), time_idx=time_idx))
            if v_scan <= 0.0:
                continue
            # Same formula as the numerator, via the shared helper -- and the numerator
            # is radius-blind unless `custom.h_norm_include_radius` is set, so this is
            # too. Do NOT reinstate a `store.scalar_value("beam_radius", ...)` read here
            # without checking every store type: beam_radius is a _CONSTANT_PATH_CHANNELS
            # member, synthesised from the case manifest inside read_case and never
            # stored as an array, so get_array raises and a bare `except` turns that into
            # a silently radius-blind denominator while the numerator keeps the radius.
            # That asymmetry is what put in_H_norm = 1.28e6 into the 2026-08-21 evals.
            factor = None
            if include_material:
                factor = float(king_material_factor(
                    rho=store.scalar_value("mat_rho", int(case_idx), time_idx=time_idx),
                    cp=store.scalar_value("mat_cp", int(case_idx), time_idx=time_idx),
                    k=store.scalar_value("mat_k", int(case_idx), time_idx=time_idx),
                    latent=store.scalar_value("mat_latent", int(case_idx), time_idx=time_idx),
                    T_l=store.scalar_value("mat_Tl", int(case_idx), time_idx=time_idx),
                    A=store.scalar_value("mat_A", int(case_idx), time_idx=time_idx),
                ))
            denom = float(h_norm_raw(
                power, v_scan,
                None if not include_radius else _radius(store, case_idx, time_idx),
                material_factor=factor,
            ))
            if denom > max_value:
                max_value = denom
    return max_value if max_value > 0.0 else 1.0


def default_lambda_weighting(outvar: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return unit lambda-weighting arrays matching ``outvar``."""
    return {key: np.ones_like(value, dtype=np.float32) for key, value in outvar.items()}


def strip_sample_axis(var: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remove the singleton leading sample axis from a per-item payload."""
    stripped: dict[str, np.ndarray] = {}
    for key, value in var.items():
        array = np.array(value, dtype=np.float32, copy=True)
        if array.ndim > 0 and array.shape[0] == 1:
            stripped[key] = array[0]
        else:
            stripped[key] = array
    return stripped


class LazySampleDataset(Dataset):
    """Map-style dataset that lazily reads full samples from an arraydir runtime.

    ``case_coherent=True`` turns on two-level shuffling: cases are visited in a
    shuffled order and frames are shuffled *within* each case, so the store's LRU is
    hit for all but the first frame of every case block. This is the standard access
    pattern for compressed sharded datasets, and it exists because the alternative was
    measured to be the training bottleneck: a global frame shuffle over 120 varspeed
    cases with a 1-case cache decompresses a whole ~600 MB case (5.9 s, 1.9 GB raw) for
    every 19 MB frame it serves -- a ~99% miss rate that starved the GPU to 14 s/iter
    under co-tenancy and OOM-killed loader workers at 4 workers (each worker privately
    re-decompressing cases). With coherence the decompression is paid once per ~390
    frames per worker.

    Mechanics: the upstream loader (physicsnemo's ``get_dataloader``) drives a
    ``RandomSampler`` we cannot replace, so in coherent mode the sampler's index is
    used only as a tick and samples are served from the dataset's own per-worker
    stream. Every entry is still served exactly once per epoch (the stream is a
    permutation), workers shard by case block via ``worker_init_fn``, and the order
    reshuffles every epoch from ``seed + epoch``. Statistically this is the two-level
    shuffle used by any sharded-data pipeline; it changes sample ORDER, never sample
    distribution. Default OFF so existing runs keep byte-identical behaviour.
    """

    auto_collation = False

    def __init__(
        self,
        *,
        store: RuntimeStore,
        entries: Sequence[tuple[int, int | None]],
        sample_builder,
        invar_keys: Sequence[str],
        outvar_keys: Sequence[str],
        case_coherent: bool = False,
        seed: int = 42,
    ):
        self.store = store
        self.entries = list(entries)
        self.sample_builder = sample_builder
        self._invar_keys = list(invar_keys)
        self._outvar_keys = list(outvar_keys)
        self.case_coherent = bool(case_coherent)
        self.seed = int(seed)
        # per-process stream state (each fork worker gets its own copy)
        self._worker_id = 0
        self._num_workers = 1
        self._stream: list[int] = []
        self._cursor = 0
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def invar_keys(self):
        return list(self._invar_keys)

    @property
    def outvar_keys(self):
        return list(self._outvar_keys)

    # ------------------------------------------------------------------ #
    # case-coherent streaming
    # ------------------------------------------------------------------ #
    def worker_init_fn(self, iworker):
        """Chain the base seeding, then bind this worker's shard and thread budget."""
        super().worker_init_fn(iworker)
        import torch as _torch

        # Loader workers doing tensor work each spawn a full intra-op thread pool;
        # N jobs x W workers x T threads oversubscribes the box and was a measured
        # contributor to the 14 s/iter collapse. physicsnemo's own get_dataloader
        # comment says exactly this. One thread per worker is correct here: the
        # work is decompression + elementwise fills, not large matmuls.
        _torch.set_num_threads(1)
        info = _torch.utils.data.get_worker_info()
        if info is not None:
            self._worker_id = int(info.id)
            self._num_workers = int(info.num_workers)
        self._rebuild_stream()

    def _rebuild_stream(self) -> None:
        """Shuffled cases, shuffled frames within each case, sharded by worker.

        Sharding is BY CASE, not by index, so two workers never hold the same case
        decompressed at once -- that per-worker duplication is what made 4 workers
        OOM under a global shuffle.
        """
        import numpy as _np

        rng = _np.random.default_rng(self.seed + 7919 * self._epoch)
        by_case: dict[int, list[int]] = {}
        for entry_idx, (case_idx, _t) in enumerate(self.entries):
            by_case.setdefault(case_idx, []).append(entry_idx)
        case_order = list(by_case)
        rng.shuffle(case_order)
        stream: list[int] = []
        for pos, case_idx in enumerate(case_order):
            if pos % self._num_workers != self._worker_id:
                continue
            frame_indices = by_case[case_idx]
            rng.shuffle(frame_indices)
            stream.extend(frame_indices)
        self._stream = stream
        self._cursor = 0

    def _next_coherent_index(self) -> int:
        if not self._stream:
            self._rebuild_stream()
        if self._cursor >= len(self._stream):
            self._epoch += 1
            self._rebuild_stream()
        entry_idx = self._stream[self._cursor]
        self._cursor += 1
        return entry_idx

    def __getitem__(self, idx):
        if self.case_coherent:
            # The sampler's idx is only a tick; the stream guarantees each entry is
            # served once per epoch in a case-blocked order.
            idx = self._next_coherent_index()
        case_idx, time_idx = self.entries[idx]
        invar, outvar = self.sample_builder(case_idx=case_idx, time_idx=time_idx)
        invar = strip_sample_axis(invar)
        outvar = strip_sample_axis(outvar)
        lambda_weighting = default_lambda_weighting(outvar)
        return (
            Dataset._to_tensor_dict(invar),
            Dataset._to_tensor_dict(outvar),
            Dataset._to_tensor_dict(lambda_weighting),
        )


class MaskedTemporalPatchDataset(Dataset):
    """Map-style masked next-step patch dataset for autoregressive training."""

    auto_collation = False

    def __init__(
        self,
        *,
        store: RuntimeStore,
        case_indices: Sequence[int],
        build_patch_sample,
        length: int,
        seed: int,
        invar_keys: Sequence[str],
        outvar_keys: Sequence[str],
    ):
        self.store = store
        self.case_indices = list(case_indices)
        self.build_patch_sample = build_patch_sample
        self.length = int(length)
        self.seed = int(seed)
        self._invar_keys = list(invar_keys)
        self._outvar_keys = list(outvar_keys)
        if not self.case_indices:
            raise ValueError("MaskedTemporalPatchDataset requires at least one case index.")

    def __len__(self) -> int:
        return self.length

    @property
    def invar_keys(self):
        return list(self._invar_keys)

    @property
    def outvar_keys(self):
        return list(self._outvar_keys)

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + int(idx))
        invar, outvar = self.build_patch_sample(rng=rng, case_indices=self.case_indices)
        invar = strip_sample_axis(invar)
        outvar = strip_sample_axis(outvar)
        lambda_weighting = default_lambda_weighting(outvar)
        return (
            Dataset._to_tensor_dict(invar),
            Dataset._to_tensor_dict(outvar),
            Dataset._to_tensor_dict(lambda_weighting),
        )


def apply_random_mask(
    array: np.ndarray,
    *,
    rng: np.random.Generator,
    mask_mode: str,
    mask_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask a fraction of voxels or one block in a patch tensor."""
    mask_ratio = float(np.clip(mask_ratio, 0.0, 1.0))
    mask = np.zeros_like(array, dtype=np.float32)
    if mask_ratio <= 0.0:
        return array, mask

    spatial_shape = array.shape[1:]
    if mask_mode == "voxel":
        mask = (rng.random(size=array.shape) < mask_ratio).astype(np.float32)
    elif mask_mode == "block":
        block_shape = [max(1, int(round(dim * mask_ratio ** (1.0 / max(1, len(spatial_shape)))))) for dim in spatial_shape]
        starts = [
            int(rng.integers(0, max(1, dim - block + 1)))
            for dim, block in zip(spatial_shape, block_shape)
        ]
        slices = tuple(slice(start, min(dim, start + block)) for start, dim, block in zip(starts, spatial_shape, block_shape))
        mask[(slice(None),) + slices] = 1.0
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")

    masked = np.asarray(array, dtype=np.float32).copy()
    masked[mask.astype(bool)] = 0.0
    return masked, mask
