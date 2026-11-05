"""Lazy per-case store over a directory of consolidated FLOW-3D NPZ files.

The eager ``load_consolidated_npz`` path stacks *every* case into one in-memory
array, so a large split (e.g. the 600-case material_lhs_300 sweep at 100 frames)
does not fit in RAM. This store gives the same fields to the lazy training path
(``_build_lazy_datasets`` -> ``LazySampleDataset``) while holding only a few
cases in memory at once, so the full dataset at full time resolution trains
without shrinking cases or dropping frames.

It conforms to the ``RuntimeStore`` interface the lazy builders use
(``read_case``/``scalar_value``/``time_steps``/``spatial_shape``/...), reusing the
channel builders in ``custom_utils.npz_datasource`` so the tensors are identical
to the eager path. The heavy per-case data (decompressed fields + the cumulative
path channels, which need the whole trajectory) is LRU-cached; the cheap
per-frame broadcast channels (coords, laser/material scalars, beam centre) are
built on demand for one timestep, so a single sample is only ~10 MB.

Opt in with the env var ``PINO_LAZY_NPZ=1`` (or a ``custom.lazy_npz`` config
flag); otherwise consolidated NPZ still loads eagerly.

ponytail: reads the existing compressed NPZ (a cache miss decompresses one case).
If cache-miss decompression dominates wall-clock, stage the fields as mmap-able
.npy once and index frames directly -- not needed until it measurably bottlenecks.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from ...custom_utils.npz_datasource import _open_case_safely as _open_case
from .frame_transfer import beam_following_x_slice
from ...custom_utils.npz_datasource import (
    ALL_MATERIAL_CHANNEL_NAMES,
    PATH_CHANNEL_NAMES,
    _beam_centres_over_time,
    _build_material_channels,
    _build_path_channels,
    _case_files,
    _load_manifest,
    is_consolidated_npz,
    served_material_channels,
)

# Path channels that are constant in space (just the per-frame beam centre); the
# rest (footprint, cumulative dose) are full spatial fields we precompute+cache.
_SCALAR_PATH_CHANNELS = ("beam_x", "beam_y")
_FIELD_PATH_CHANNELS = tuple(k for k in PATH_CHANNEL_NAMES if k not in _SCALAR_PATH_CHANNELS)

_DEFAULT_FIELDS = ("Temperature", "Velocity", "Fraction Of Fluid", "Pressure")


def lazy_npz_enabled(run_cfg: Any) -> bool:
    """Whether consolidated NPZ should load through this lazy store."""
    env = os.environ.get("PINO_LAZY_NPZ", "")
    if env.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    # `custom.lazy_npz` was promised by this docstring and dead in practice: RunConfig has
    # no such field, so `getattr(run_cfg, ...)` never saw a Hydra `custom.*` key. Same
    # defect class as melt_window_k (2026-09-04); same remedy. Nothing in either repo sets
    # the key, so no running configuration changes.
    from ...custom_utils.utils import custom_flag

    return bool(custom_flag(run_cfg, "lazy_npz", False))


#: Cores kept free for the OS, interactive logins, a concurrent solver and CI.
_RESERVED_CORES = 4


def dataloader_workers(run_cfg: Any) -> int:
    """How many worker processes decompress cases in parallel.

    `SupervisedGridConstraint` defaults to 0, which decompresses one case per
    step on the main process while the GPU waits -- 13.56 s/case on the
    full-resolution 2579 data against ~1.65 s of compute. Workers overlap that
    with training and, unlike a case-grouped sampler, leave global shuffling
    intact. Each worker keeps its own LRU, so prefer a small `PINO_LAZY_CACHE`
    when raising this. Env `PINO_LOADER_WORKERS` wins over the config, matching
    the other switches here; 0 keeps the old serial behaviour.
    """
    raw = os.environ.get("PINO_LOADER_WORKERS", "").strip()
    if not raw:
        raw = str(getattr(run_cfg, "loader_workers", 0))
    try:
        workers = int(raw)
    except (TypeError, ValueError):
        return 0
    if workers <= 0:
        return 0
    ceiling = max(1, (os.cpu_count() or 1) - _RESERVED_CORES)
    return min(workers, ceiling)


def power_weighted_dose_enabled(run_cfg: Any) -> bool:
    """Whether path_dose is scaled by power (env PINO_POWER_WEIGHTED_DOSE=1 or config).

    Off by default: only power-varying datasets need it (a fixed-power set gains
    nothing and it would rescale an input channel existing runs learned). Mirrors
    ``lazy_npz_enabled`` — env var wins, else the run_cfg flag if present.
    """
    env = os.environ.get("PINO_POWER_WEIGHTED_DOSE", "")
    if env.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return bool(getattr(run_cfg, "power_weighted_dose", False))


def power_weighted_conduction_enabled(run_cfg: Any) -> bool:
    """Whether conduction_field is scaled by power (env PINO_POWER_WEIGHTED_CONDUCTION=1
    or `custom.power_weighted_conduction`).

    Deliberately NOT keyed off `power_weighted_dose`: the manuscript's tabled models all
    set that flag, and widening it would change what they are fed at eval time.
    """
    env = os.environ.get("PINO_POWER_WEIGHTED_CONDUCTION", "")
    if env.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return bool(getattr(run_cfg, "power_weighted_conduction", False))


def conduction_source_sigma_factor(run_cfg: Any) -> float:
    """Standard deviation of the conduction_field source as a multiple of the beam radius
    (`custom.conduction_source_sigma_factor`; 1.0, the original channel, when unset; 0.5 matches the
    solver's Gaussian beam, whose 1/e^2 radius is the configured beam radius)."""
    value = getattr(run_cfg, "conduction_source_sigma_factor", None)
    return 1.0 if value is None else float(value)


# Fields named in the config but absent from the data; warned once each.
_MISSING_FIELDS_WARNED: set[str] = set()
_BAKED_ANNOUNCED: set[str] = set()


def _zip_member_memmaps(path, names):
    """Memory-map the named float32 members of an UNCOMPRESSED on-disk npz.

    Returns {name: np.memmap} for every requested member that is eligible:
    stored with compress_type=0 (a materialised variant, aiida-pino 81791e8),
    C-order, float32. Anything else -- DEFLATE member, other dtype, Fortran
    order, unreadable file -- is simply absent from the result, and the caller
    falls back to the eager whole-case load for that member. Never raises.

    Why: a DEFLATE member must be inflated in full to serve one frame, which is
    the measured 7.9 s/step with the GPU at 0 % on full-res coherence-OFF arms.
    An uncompressed member is a plain .npy at a byte offset inside the zip, so
    the inner array can be mapped and sliced per frame; the OS page cache then
    holds hot frames SHARED across loader workers and processes, instead of
    each worker holding whole decompressed cases in a private LRU.
    """
    import zipfile as _zf
    from numpy.lib import format as _npf

    want = {f"{n}.npy": n for n in names}
    out: Dict[str, np.ndarray] = {}
    try:
        with _zf.ZipFile(path) as z, open(path, "rb") as f:
            for info in z.infolist():
                name = want.get(info.filename)
                if name is None or info.compress_type != _zf.ZIP_STORED:
                    continue
                # data begins after the LOCAL header, whose name/extra lengths can
                # differ from the central directory's -- read them from the file.
                f.seek(info.header_offset)
                lh = f.read(30)
                if len(lh) < 30 or lh[:4] != b"PK\x03\x04":
                    continue
                nlen = int.from_bytes(lh[26:28], "little")
                elen = int.from_bytes(lh[28:30], "little")
                f.seek(info.header_offset + 30 + nlen + elen)
                try:
                    version = _npf.read_magic(f)
                    shape, fortran, dtype = _npf._read_array_header(f, version)  # noqa: SLF001
                except Exception:
                    continue
                if fortran or dtype != np.float32:
                    continue
                out[name] = np.memmap(path, dtype=dtype, mode="r",
                                      offset=f.tell(), shape=shape, order="C")
    except Exception:
        return {}
    return out


_BAKED_META_KEY = "path_channel_meta"


def _baked_channels_usable(meta, *, power_weighted_dose: bool,
                           power_weighted_conduction: bool,
                           conduction_source_sigma_factor: float = 1.0) -> bool:
    """Whether baked path channels in a case NPZ may be served to this run.

    Baked channels turn a ~1.9 GB per-miss rebuild into a mmap read, but
    `power_weighted_dose` / `power_weighted_conduction` change the VALUES, so a
    bake made under different flags must never be served -- that is silently
    wrong numbers rather than a slow run. On any mismatch, or unreadable meta,
    return False and let the caller build from the manifest: correct, just slow.

    ``meta is None`` means the NPZ carries channels from a producer that predates
    this guard (build_steady_split's co-moving beam_dist); that path keeps its
    existing behaviour.
    """
    if meta is None:
        return True
    if not isinstance(meta, dict):
        return False
    # A bake from before the source-width option recorded no factor and was built at 1.0.
    return (bool(meta.get("power_weighted_dose")) == bool(power_weighted_dose)
            and bool(meta.get("power_weighted_conduction")) == bool(power_weighted_conduction)
            and float(meta.get("conduction_source_sigma_factor", 1.0)) == float(conduction_source_sigma_factor))


class _CaseData:
    """Heavy per-case payload held in the LRU: raw fields + cumulative path fields."""

    __slots__ = ("x", "y", "z", "t", "power", "speed", "fields", "centres", "path_fields")

    def __init__(self, x, y, z, t, power, speed, fields, centres, path_fields):
        self.x, self.y, self.z, self.t = x, y, z, t
        self.power, self.speed = power, speed
        self.fields = fields              # {raw_name: (nt, [C,] X, Y, Z)}
        self.centres = centres            # (nt, 2) beam centre, or None
        self.path_fields = path_fields    # {name: (1, nt, 1, X, Y, Z)} for beam_dist/path_dose


GRID_STAMP_NAME = "trained_grid.json"


def _variant_stride(root: str | Path) -> tuple[int, int, int]:
    """The stride aiida-pino baked into the staged dataset VARIANT, if any.

    A variant like ``var_1_s2x2x1_time_noref`` holds ALREADY-COARSE arrays on
    disk (45x20x40 from a 90x40x40 dataset) and records that in the staged
    ``manifest.json``. Read-time striding on top of it multiplies, which is the
    one composition nothing else in the stack states.
    """
    manifest = Path(root) / "manifest.json"
    if not manifest.is_file():
        return (1, 1, 1)
    try:
        meta = (json.loads(manifest.read_text()) or {}).get("dataset_metadata") or {}
        stride = (meta.get("derivation_profile") or {}).get("stride")
        return _validate_stride(stride) if stride else (1, 1, 1)
    except (ValueError, OSError, TypeError, AttributeError):
        # An unreadable manifest must not block a run that was fine before.
        return (1, 1, 1)


def _variant_crop_z_top(root: str | Path) -> int:
    """The gas-side z-crop aiida-pino baked into the staged VARIANT, if any.

    The crop twin of `_variant_stride`, and for the same reason: a materialised
    cropped variant holds ALREADY-SHORTENED arrays on disk and records the crop in
    its staged `manifest.json`. Cropping again at read subtracts twice.
    """
    manifest = Path(root) / "manifest.json"
    if not manifest.is_file():
        return 0
    # A zero-copy staging is a pack index over raw payloads: no array was written, so
    # nothing was cropped, whatever the profile requested. Measured from the directory,
    # so it holds for stagings written before the `staging` block existed. cj102455
    # (2026-09-11) was refused here for "cropping again" a crop that had not happened,
    # and this function's own remedy would have trained z=40 under a record saying 30.
    if (Path(root) / "pack_index.json").is_file():
        return 0
    try:
        payload = json.loads(manifest.read_text()) or {}
        staging = payload.get("staging") or {}
        if "baked_crop_z_top" in staging:
            # What aiida-pino MEASURED it wrote (raw z minus written z), never the request.
            return int(staging["baked_crop_z_top"] or 0) if not staging.get("zero_copy") else 0
        meta = payload.get("dataset_metadata") or {}
        baked = (meta.get("derivation_profile") or {}).get("crop_z_top")
        return int(baked) if baked else 0
    except (ValueError, OSError, TypeError, AttributeError):
        # An unreadable manifest must not block a run that was fine before.
        return 0


def _resolve_read_time_crop(input_dir: str | Path, raw_crop: str) -> int:
    """Read-time crop in cells, refusing to compose with a crop already baked in.

    Baked-cropped variants exist (pk 68783, 96 members; 69438), so once a profile's
    `crop_z_top` also reaches the loader as an env var, every one of them is a
    double-crop: 40 -> 10 on disk, then the same subtraction again at read. That
    trains and reports a wrong number rather than failing, which is why this refuses
    instead of picking one.
    """
    raw_crop = (raw_crop or "").strip()
    if not raw_crop:
        return 0
    try:
        crop = int(raw_crop)
    except ValueError:
        raise ValueError(
            f"[NpzDirStore] PINO_LAZY_CROP_Z_TOP must be an integer number of "
            f"cells, got '{raw_crop}'.") from None
    baked = _variant_crop_z_top(input_dir)
    if baked and crop:
        raise ValueError(
            f"[NpzDirStore] the dataset staged at '{input_dir}' is already cropped "
            f"crop_z_top={baked} by its derivation profile, and "
            f"PINO_LAZY_CROP_Z_TOP='{raw_crop}' would crop it again -- effective "
            f"{baked + crop} cells removed. Use ONE mechanism: either stage an "
            f"uncropped variant and keep PINO_LAZY_CROP_Z_TOP, or keep the cropped "
            f"variant and unset PINO_LAZY_CROP_Z_TOP."
        )
    return crop


def _network_dir(run_cfg: Any) -> Optional[Path]:
    """``cfg.network_dir``: the output dir when training, the model dir at eval."""
    cfg = getattr(run_cfg, "cfg", None)
    if cfg is None:
        return None
    try:
        from omegaconf import OmegaConf

        value = OmegaConf.select(cfg, "network_dir", default=None)
    except Exception:
        value = getattr(cfg, "network_dir", None)
    if not value or str(value) in ("PATH", "."):
        return None
    return Path(str(value))


def _stamped_grid(stamp: Path) -> Optional[tuple[int, int, int]]:
    try:
        grid = json.loads(stamp.read_text())["grid"]
        return tuple(int(v) for v in grid)  # type: ignore[return-value]
    except (ValueError, OSError, KeyError, TypeError):
        return None


def _store_grid_is_uniform(store) -> bool:
    """True when every case in the store serves the same grid.

    A mixed-extent dataset (`ti64_geom40`: 40 domains from 90x41x40 to
    187x125x40 at a fixed 10 um cell) has per-case shapes BY DESIGN, so
    `spatial_shape()` -- the first case's shape -- is not a property of the run
    and cannot be compared across stores. Reads only the cached per-case
    metadata (x/y/z lengths), never a field, so this costs no decompression.
    """
    try:
        n = int(store.sample_count())
    except Exception:
        return True
    if n <= 1:
        return True
    try:
        first = None
        for i in range(n):
            m = store._meta(i)
            shape = (int(m["nx"]), int(m["ny"]), int(m["nz"]))
            if first is None:
                first = shape
            elif shape != first:
                return False
        return True
    except Exception:
        # Unreadable metadata must not turn a contract check into a crash; the
        # conservative answer keeps the pre-existing behaviour.
        return True


def _stamped_uniform(stamp) -> bool:
    """`grid_uniform` from the stamp; True when absent, so older stamps behave
    exactly as before."""
    try:
        return bool(json.loads(stamp.read_text()).get("grid_uniform", True))
    except Exception:
        return True


def _enforce_grid_contract(store: "NpzDirStore", run_cfg: Any) -> None:
    """Stamp the grid training served, and refuse to score a model off it.

    On 2026-08-19 both spkh123 arms trained on a strided variant (45x20x40) and
    were scored on the full grid (90x40x40) -- the train calc and the eval calc
    each printed their own ``[grid] serving`` line and neither compared them. A
    graph network rebuilds its connectivity from the coordinates, so a 4x node
    count at inference is a structural shift, and the measured architecture gap
    was partly that artefact.

    Deliberate resolution changes stay possible: ``super_res_factor != 1`` is a
    super-resolution study by definition, and ``PINO_ALLOW_GRID_SHIFT=1`` is the
    explicit escape hatch. Models trained before the stamp existed are never
    blocked.
    """
    network_dir = _network_dir(run_cfg)
    if network_dir is None:
        return
    served = tuple(int(v) for v in store.spatial_shape()[:3])
    stamp = network_dir / GRID_STAMP_NAME

    if bool(getattr(run_cfg, "eval_only", False)):
        if int(getattr(run_cfg, "super_res_factor", 1) or 1) != 1:
            return
        if os.environ.get("PINO_ALLOW_GRID_SHIFT", "").strip():
            return
        trained = _stamped_grid(stamp) if stamp.is_file() else None
        if trained is not None and trained != served:
            # REPORT, do not refuse. Standing policy (2026-08-19): a model
            # trained at a reduced resolution is still scored at FULL resolution,
            # because full resolution is the resolution the result is claimed at.
            # Refusing that would block the intended workflow, and the transfer cost
            # is measured and small for RIGNO (0.030 K, 0.18%) though not for GAOT
            # (+16% MAE), so it is a number to report, not an error.
            #
            # The defect this contract was built for was never the eval side: it was
            # training at half resolution UNINTENTIONALLY. That is caught by the
            # double-stride refusal above and by the train-vs-validation check below.
            print(
                f"[grid] NOTE: trained on {trained[0]}x{trained[1]}x{trained[2]}, "
                f"scoring on {served[0]}x{served[1]}x{served[2]} -- this measures "
                f"resolution transfer as well as accuracy. Compare only against other "
                f"evals with the same served grid.",
                flush=True,
            )
        return

    # Training: stamp once. A second store (validation) must agree with the first.
    uniform = _store_grid_is_uniform(store)
    if stamp.is_file():
        previous = _stamped_grid(stamp)
        stamped_uniform = _stamped_uniform(stamp)
        if previous is not None and previous != served and uniform and stamped_uniform:
            raise ValueError(
                f"[NpzDirStore] this run already served "
                f"{previous[0]}x{previous[1]}x{previous[2]} but now serves "
                f"{served[0]}x{served[1]}x{served[2]}. Train and validation data "
                f"must be the same resolution."
            )
        if previous is not None and previous != served:
            print(
                f"[grid] mixed-extent dataset: train served "
                f"{previous[0]}x{previous[1]}x{previous[2]}, this store serves "
                f"{served[0]}x{served[1]}x{served[2]}. Per-case grids differ by "
                f"design, so the train/val shape check does not apply; the stride "
                f"contract still does.",
                flush=True,
            )
        return
    try:
        network_dir.mkdir(parents=True, exist_ok=True)
        stamp.write_text(json.dumps({
            "grid": list(served),
            "variant_stride": list(_variant_stride(store.root)),
            "grid_uniform": bool(uniform),
        }, indent=1))
    except OSError:
        pass  # a read-only network_dir must not fail a healthy run


def _validate_stride(stride: Any) -> tuple[int, int, int]:
    """Normalize a per-axis spatial stride, rejecting anything that would mislead."""
    try:
        values = tuple(int(v) for v in stride)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"[NpzDirStore] spatial_stride must be 3 integers, got {stride!r}.") from exc
    if len(values) != 3 or any(v < 1 for v in values):
        raise ValueError(f"[NpzDirStore] spatial_stride must be 3 integers >= 1, got {stride!r}.")
    return values  # type: ignore[return-value]


def _strided_len(n: int, step: int) -> int:
    return len(range(0, int(n), int(step)))


class NpzDirStore:
    """Lazy reader over ``<dir>/<case>.npz`` (+ ``<case>.manifest.json``) files."""

    def __init__(
        self,
        input_dir: str | Path,
        *,
        requested_fields: Optional[Iterable[str]] = None,
        path_channels: Optional[Iterable[str]] = None,
        material_channels: Optional[Iterable[str]] = None,
        cache_size: int = 3,
        power_weighted_dose: bool = False,
        power_weighted_conduction: bool = False,
        conduction_source_sigma_factor: float = 1.0,
        spatial_stride: tuple[int, int, int] = (1, 1, 1),
        moving_window: tuple[int, int] | None = None,
        crop_z_top: int = 0,
    ):
        self.root = Path(input_dir)
        self._power_weighted_dose = bool(power_weighted_dose)
        self._power_weighted_conduction = bool(power_weighted_conduction)
        self._conduction_source_sigma_factor = float(conduction_source_sigma_factor)
        self._stride = _validate_stride(spatial_stride)
        # Gas-side z-crop in RAW cells, dropped from the TOP of z before the
        # stride applies -- the read-time twin of aiida-pino's `crop_z_top`
        # derivation field. Read-time because a crop makes a derivation
        # "transforming", which flips the collection off the pack-index path
        # onto materialisation: 2.70 GB per member against ~23 MB stored, so
        # the 96-member train split wants ~413 GB and does not fit. Applied in
        # the single slicing expression every field goes through, so inputs
        # and targets crop together by construction.
        self._crop_z_top = int(crop_z_top)
        if self._crop_z_top < 0:
            raise ValueError(
                f"[NpzDirStore] crop_z_top must be >= 0, got {crop_z_top!r}.")
        # Per-frame laser-tracking x-window (behind, ahead) in cells: the lazy
        # twin of `_apply_moving_frame`, minus its one-cell-per-frame assumption
        # -- the window follows the beam's ACTUAL per-frame cell from the cached
        # manifest centres. Clamped inside the domain so every sample has one
        # constant shape (what batch>1 needs). See test_npz_lazy_moving_ref.py.
        self._moving_window: tuple[int, int] | None = None
        if moving_window is not None:
            behind, ahead = (int(v) for v in moving_window)
            if behind < 0 or ahead < 0 or behind + ahead < 2:
                raise ValueError(
                    f"[NpzDirStore] moving_window must be two non-negative cells "
                    f"with width >= 2, got {moving_window!r}.")
            if self._stride != (1, 1, 1):
                raise ValueError(
                    "[NpzDirStore] moving_window cannot be combined with a spatial "
                    "stride; pick one resolution mechanism.")
            self._moving_window = (behind, ahead)
        self.cases = list(_case_files(self.root))
        if not self.cases:
            raise ValueError(f"[NpzDirStore] no case NPZ under '{self.root}'.")
        self.field_names = list(requested_fields) if requested_fields is not None else list(_DEFAULT_FIELDS)
        self.path_channels = [k for k in (path_channels or []) if k in PATH_CHANNEL_NAMES]
        self.material_channels = [k for k in (material_channels or []) if k in ALL_MATERIAL_CHANNEL_NAMES]
        self._field_path_channels = [k for k in self.path_channels if k in _FIELD_PATH_CHANNELS]
        self._scalar_path_channels = [k for k in self.path_channels if k in _SCALAR_PATH_CHANNELS]
        self._cache_size = int(cache_size)
        self._cache: "OrderedDict[int, _CaseData]" = OrderedDict()
        self._meta_cache: Dict[int, Dict[str, Any]] = {}
        self._dims: Optional[tuple[int, int, int]] = None  # (nx, ny, nz)
        self._grid_announced = False
        self._served_grid: dict | None = None
        if self._crop_z_top:
            raw_nz = int(np.asarray(np.load(_open_case(self.cases[0]))["z"]).size)
            if self._crop_z_top >= raw_nz:
                raise ValueError(
                    f"[NpzDirStore] crop_z_top={self._crop_z_top} removes the whole "
                    f"grid (raw nz={raw_nz}).")
        if self._moving_window is not None:
            width = sum(self._moving_window)
            nx0 = self._meta(0)["nx"]
            if width > nx0:
                raise ValueError(
                    f"[NpzDirStore] moving_window width {width} exceeds the grid "
                    f"(nx={nx0}).")

    # ---- lightweight metadata (no field decompression) --------------------- #
    def _meta(self, case_idx: int) -> Dict[str, Any]:
        if case_idx not in self._meta_cache:
            with np.load(_open_case(self.cases[case_idx])) as d:
                raw = (np.asarray(d["x"]).size, np.asarray(d["y"]).size, np.asarray(d["z"]).size)
                self._meta_cache[case_idx] = {
                    "nt": int(np.asarray(d["time_s"]).size),
                    # strided lengths: every consumer (spatial_shape, array_shape,
                    # the AR patch sampler) must see the grid _case actually serves
                    "nx": _strided_len(raw[0], self._stride[0]),
                    "ny": _strided_len(raw[1], self._stride[1]),
                    "nz": _strided_len(max(raw[2] - self._crop_z_top, 0), self._stride[2]),
                    "power": float(np.asarray(d["power_w"])),
                    "speed": float(np.asarray(d["scan_speed_m_per_s"])),
                }
            # Announce the grid this store actually serves, once. The effective
            # resolution otherwise lives only as the composition of two AiiDA
            # attributes (dataset derivation stride x env PINO_LAZY_STRIDE), and
            # misreading one of them killed a correctly-configured queued run on
            # 2026-08-09. The log line is ground truth, next to the checkpoint.
            if not self._grid_announced:
                m = self._meta_cache[case_idx]
                # Report what spatial_shape() returns, INCLUDING the moving-ref
                # window. Printing the strided grid alone said "90x40x40" for a run
                # whose store served 48x40x40, which on 2026-08-19 read as "the
                # window did not apply" until trained_grid.json contradicted it.
                nx = (sum(self._moving_window) if self._moving_window is not None
                      else m["nx"])
                window = (f", window {self._moving_window[0]}+{self._moving_window[1]}"
                          if self._moving_window is not None else "")
                crop = f", crop_z_top {self._crop_z_top}" if self._crop_z_top else ""
                print(
                    f"[grid] serving {nx}x{m['ny']}x{m['nz']} "
                    f"(raw {raw[0]}x{raw[1]}x{raw[2]}, "
                    f"stride {self._stride[0]}x{self._stride[1]}x{self._stride[2]}"
                    f"{crop}{window})",
                    flush=True,
                )
                self._grid_announced = True
                # The same facts as the line above, kept for the metrics payload. The
                # printed line has been the ONLY record of what an eval actually served,
                # and a log is not queryable: on the nine tabled lab evals every node says
                # `derivation_profile.crop_z_top = 0` including the three that demonstrably
                # served 30-deep, and the one field that differs, `PINO_LAZY_CROP_Z_TOP`,
                # reads '' on exactly those three -- which is how the loader spells NO crop.
                # So the only discriminating value on the node asserts the opposite of the
                # truth, and neither stride nor crop is auditable from provenance
                # (2026-09-08). A capture, not a
                # computation: these are the numbers already in hand.
                self._served_grid = {
                    "nx": int(nx), "ny": int(m["ny"]), "nz": int(m["nz"]),
                    "raw": [int(raw[0]), int(raw[1]), int(raw[2])],
                    "stride": [int(v) for v in self._stride],
                    "crop_z_top": int(self._crop_z_top),
                    "moving_window": (list(self._moving_window)
                                      if self._moving_window is not None else None),
                }
        return self._meta_cache[case_idx]

    def served_grid(self) -> dict | None:
        """What this store actually serves, or None before the first case is read."""
        return getattr(self, "_served_grid", None)

    def sample_count(self) -> int:
        return len(self.cases)

    def list_array_names(self) -> list[str]:
        return list(self.field_names)

    def array_shape(self, name: str) -> tuple[int, ...]:
        m = self._meta(0)
        nx = sum(self._moving_window) if self._moving_window is not None else m["nx"]
        return (len(self.cases), m["nt"], nx, m["ny"], m["nz"])

    def has_time_axis(self) -> bool:
        return True

    def time_steps(self, case_idx: int | None = None) -> int:
        return self._meta(0 if case_idx is None else int(case_idx))["nt"]

    def has_uniform_timesteps(self, case_indices: Iterable[int] | None = None) -> bool:
        idx = list(case_indices) if case_indices is not None else range(len(self.cases))
        return len({self.time_steps(case_idx=int(i)) for i in idx}) <= 1

    def spatial_shape(self) -> tuple[int, ...]:
        m = self._meta(0)
        if self._moving_window is not None:
            return (sum(self._moving_window), m["ny"], m["nz"])
        return (m["nx"], m["ny"], m["nz"])

    def scalar_value(self, name: str, case_idx: int, time_idx: int | None = None) -> float:
        del time_idx  # laser scalars are constant over the trajectory here
        m = self._meta(int(case_idx))
        if name == "P":
            return m["power"]
        if name == "V_scan":
            return m["speed"]
        # Case-CONSTANT path channels are synthesised from the case manifest, never
        # stored, so every scalar reader used to fail on them -- which is what left
        # compute_h_norm_denom radius-blind against a radius-carrying numerator. This is
        # the store type the varying-radius datasets actually use.
        from lpno.custom_utils.npz_datasource import (
            CONSTANT_PATH_CHANNELS,
            build_constant_path_channels,
        )

        if name in CONSTANT_PATH_CHANNELS:
            manifest = self._require_manifest(self.cases[int(case_idx)])
            built = build_constant_path_channels([name], manifest, (1,))
            if name in built:
                return float(np.asarray(built[name]).reshape(-1)[0])
        raise KeyError(
            f"[NpzDirStore] scalar_value serves P/V_scan and {sorted(CONSTANT_PATH_CHANNELS)}, "
            f"not '{name}'."
        )

    # ---- heavy per-case payload (LRU) -------------------------------------- #
    def laser_on_frame_stop(self, case_idx: int) -> int:
        """Exclusive index of the last frame on the beam's declared path (see
        `frame_transfer.laser_on_frame_stop`); the case's frame count when the manifest is
        absent or records no moving overrun. Lets the eval's steady transfer window an
        overrun case on the frames its steady target was built from."""
        from .frame_transfer import laser_on_frame_stop

        case = self._case(int(case_idx))
        try:
            manifest = self._require_manifest(self.cases[int(case_idx)])
        except Exception:  # noqa: BLE001 - no manifest means no overrun to know about
            return int(np.asarray(case.t).size)
        return laser_on_frame_stop(manifest, case.t)

    def _case(self, case_idx: int) -> _CaseData:
        """LRU-cached heavy payload, with one retry on a transient archive error.

        The staged runtime cache is a SHARED mutable directory: another job's
        staging, eviction or repair can rewrite a case file under an open
        handle, which surfaces as `BadZipFile: Bad CRC-32` on a file that
        verifies clean seconds later (measured twice on the varspeed cache,
        2026-08-13/14, both times killing a run at step ~8000). One reopen
        absorbs that; a second failure is real corruption and propagates.
        """
        import zipfile as _zipfile

        try:
            return self._case_uncached(case_idx)
        except (_zipfile.BadZipFile, EOFError, OSError) as exc:
            # FileNotFoundError IS an OSError, and a missing case is not
            # transient -- it is eviction or mis-staging, and retrying it hides
            # that behind a doubled load and a "transient" log line (pinned by
            # test_a_genuinely_missing_case_still_raises, which this retry broke
            # the day it landed).
            if isinstance(exc, FileNotFoundError):
                raise
            print(f"[npz-store] transient archive error on case {case_idx} "
                  f"({type(exc).__name__}: {exc}); reopening once", flush=True)
            self._cache.pop(case_idx, None)
            return self._case_uncached(case_idx)

    def _case_uncached(self, case_idx: int) -> _CaseData:
        cached = self._cache.get(case_idx)
        if cached is not None:
            self._cache.move_to_end(case_idx)
            return cached
        path = self._existing_case_path(case_idx)
        sx, sy, sz = self._stride
        # `None` keeps the full extent; an int is the last RAW z cell kept, so the
        # crop is expressed in raw cells and the stride then applies to what is left.
        zk = -self._crop_z_top if self._crop_z_top else None
        # ascontiguousarray, not a bare view: the slice must drop the discarded
        # cells so the LRU holds the small grid, not the full one it came from.
        spatial = (
            (lambda a: a)
            if (sx, sy, sz) == (1, 1, 1) and zk is None
            else (lambda a: np.ascontiguousarray(a[..., ::sx, ::sy, :zk:sz]))
        )
        # Materialised (uncompressed) members can be memory-mapped and sliced per
        # frame instead of loaded whole -- see _zip_member_memmaps. Only when the
        # opener hands back a real path: a pack-index file-like has no stable
        # byte offsets to map.
        _opened_probe = _open_case(path)
        mm = (
            _zip_member_memmaps(path, tuple(self.field_names) + tuple(self._field_path_channels))
            if isinstance(_opened_probe, (str, os.PathLike))
            else {}
        )
        with np.load(_open_case(path)) as data:
            x = np.asarray(data["x"], dtype=np.float32)[::sx]
            y = np.asarray(data["y"], dtype=np.float32)[::sy]
            z = np.asarray(data["z"], dtype=np.float32)[:zk:sz]
            t = np.asarray(data["time_s"], dtype=np.float32)
            power = float(np.asarray(data["power_w"]))
            speed = float(np.asarray(data["scan_speed_m_per_s"]))
            fields: Dict[str, np.ndarray] = {}
            for name in self.field_names:
                if name not in data.files:
                    # A configured field the data does not carry. Warn once and skip
                    # rather than raise: config_flow3D_RIGNO lists X-/Y-/Z-velocity,
                    # which the FLOW-3D exporter drops as redundant with the Velocity
                    # vector, so time_as_batch=true died here with
                    # "KeyError: 'X-velocity is not a file in the archive'" while the
                    # time-averaged path (which never asks the store) was fine.
                    # Downstream is already tolerant: _preprocess_var guards the
                    # optional Pressure/Velocity keys.
                    if name not in _MISSING_FIELDS_WARNED:
                        _MISSING_FIELDS_WARNED.add(name)
                        print(
                            f"[npz-store] configured field {name!r} is not in the "
                            f"data ({os.path.basename(str(path))}); skipping it. "
                            "Trim custom.vtk_fields to silence this.",
                            flush=True,
                        )
                    continue
                if name in mm:
                    arr = mm[name]  # mapped: slicing below touches only needed pages
                    if arr.ndim == 5:
                        arr = np.moveaxis(arr, -1, 1)
                    # keep the VIEW -- ascontiguousarray would materialise the
                    # whole case and defeat the map; downstream take_field()
                    # copies the single frame it serves.
                    fields[name] = (
                        arr if (sx, sy, sz) == (1, 1, 1) and zk is None
                        else arr[..., ::sx, ::sy, :zk:sz]
                    )
                    continue
                arr = np.asarray(data[name], dtype=np.float32)  # (nt, X, Y, Z[, 3])
                if arr.ndim == 5:  # vector field -> (nt, C, X, Y, Z)
                    arr = np.moveaxis(arr, -1, 1)
                fields[name] = spatial(arr)  # spatial axes are the trailing 3 either way
            # Field path channels precomputed into the NPZ (e.g. a co-moving beam_dist
            # from build_steady_split, whose laser-frame footprint can't be rebuilt
            # from the lab-frame manifest). Read them here; the rest come from the
            # manifest below. Per-frame NPZs store none of these, so are unaffected.
            precomp_path: Dict[str, np.ndarray] = {}
            baked_meta = None
            if _BAKED_META_KEY in data.files:
                try:
                    baked_meta = json.loads(str(np.asarray(data[_BAKED_META_KEY]).item()))
                except Exception:  # noqa: BLE001 - unreadable meta must refuse, not guess
                    baked_meta = "unreadable"
            usable = _baked_channels_usable(
                baked_meta,
                power_weighted_dose=self._power_weighted_dose,
                power_weighted_conduction=self._power_weighted_conduction,
                conduction_source_sigma_factor=self._conduction_source_sigma_factor,
            )
            if baked_meta is not None and "x" not in _BAKED_ANNOUNCED:
                # Announce ONCE per process. A refused bake is otherwise silent and
                # looks exactly like the fix not working: the run just falls back to
                # building at the old speed. Say which it is, and why.
                _BAKED_ANNOUNCED.add("x")
                if usable:
                    print(f"[npz-store] using BAKED path channels (mmap) from "
                          f"{os.path.basename(str(path))}", flush=True)
                else:
                    want = (self._power_weighted_dose, self._power_weighted_conduction)
                    got = (baked_meta.get("power_weighted_dose"),
                           baked_meta.get("power_weighted_conduction")) \
                        if isinstance(baked_meta, dict) else baked_meta
                    print(f"[npz-store] REFUSING baked path channels: bake recorded "
                          f"power_weighted (dose, conduction)={got} but this run wants "
                          f"{want}. Falling back to building them -- correct, but slow. "
                          "Re-bake with the run's flags to get the speed back.", flush=True)
            for pch in ([] if not usable else self._field_path_channels):
                if pch in mm:
                    # Mapped: keep the VIEW. These are the biggest arrays in the
                    # store (nt x X x Y x Z), so eager-loading them is what made
                    # the LRU cost ~1 GB per channel per case.
                    a = mm[pch]
                    a = a if (sx, sy, sz) == (1, 1, 1) and zk is None else a[..., ::sx, ::sy, :zk:sz]
                    precomp_path[pch] = a[None, :, None]
                elif pch in data.files:
                    a = spatial(np.asarray(data[pch], dtype=np.float32))  # (nt, X, Y, Z)
                    precomp_path[pch] = a[None, :, None]         # (1, nt, 1, X, Y, Z)
        centres = None
        path_fields: Dict[str, np.ndarray] = dict(precomp_path)
        from_manifest = [p for p in self._field_path_channels if p not in precomp_path]
        if self.path_channels and (self._scalar_path_channels or from_manifest):
            manifest = self._require_manifest(path)
            if self._scalar_path_channels:
                centres = _beam_centres_over_time(manifest, t)  # (nt, 2)
            if from_manifest:
                path_fields.update(_build_path_channels(
                    from_manifest, manifest, x, y, z, t,
                    power_weighted=self._power_weighted_dose,
                    power_weighted_conduction=self._power_weighted_conduction,
                    conduction_sigma_factor=self._conduction_source_sigma_factor,
                ))
        case = _CaseData(x, y, z, t, power, speed, fields, centres, path_fields)
        self._cache[case_idx] = case
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return case

    def _existing_case_path(self, case_idx: int) -> Path:
        """Case NPZ path, repairing a vanished aiida-pino runtime cache once.

        The staged dir is a symlink into `~/.cache/aiida-pino/runtime/<uuid>`,
        which used to be documented as safe to `rm -rf`. Clearing it during a
        run made a DataLoader worker raise FileNotFoundError and killed the job
        -- seven runs so far, one of them ten hours in. One retry turns that
        into a pause; a genuinely absent file still raises rather than looping.
        """
        path = self.cases[case_idx]
        if path.exists():
            return path
        if self._repair_runtime_cache_once() and path.exists():
            print(f"[npz-store] repaired vanished runtime cache for {path.name}", flush=True)
        return path

    def _repair_runtime_cache_once(self) -> bool:
        """Ask aiida-pino to restage the runtime cache. False if it could not run.

        Shared by the two halves of the same failure -- a cleared cache takes
        `<case>.npz` and `<case>.manifest.json` together, so guarding only the NPZ
        meant which error killed the run depended on which file was touched first.
        """
        try:
            from aiida_pino.utils.runtime_repair import repair_dataset_dir
        except Exception:  # noqa: BLE001 - aiida-pino is an optional peer
            return False
        try:
            repair_dataset_dir(self.root)
        except Exception as exc:  # noqa: BLE001 - repair is best effort
            print(f"[npz-store] runtime-cache repair failed ({type(exc).__name__}: {exc})", flush=True)
            return False
        return True

    def _require_manifest(self, path: Path) -> Dict[str, Any]:
        manifest = _load_manifest(path)
        if manifest is None and self._repair_runtime_cache_once():
            # Same one-shot retry as `_existing_case_path`: a cleared runtime cache
            # becomes a pause, a genuinely absent sidecar still raises. Guarded here
            # rather than at the call sites because two of the three index
            # `self.cases` directly and never pass through `_existing_case_path`.
            manifest = _load_manifest(path)
            if manifest is not None:
                print(f"[npz-store] repaired vanished runtime cache for "
                      f"{path.stem}.manifest.json", flush=True)
        if manifest is None:
            raise ValueError(
                f"[NpzDirStore] manifest channels requested but no "
                f"'{path.stem}.manifest.json' beside '{path.name}'."
            )
        return manifest

    # ---- the read the lazy sample builder calls ---------------------------- #
    def read_case(
        self,
        case_idx: int,
        *,
        time_idx: int | None = None,
        spatial_slices: tuple[slice, ...] | None = None,
        keys: Sequence[str] | None = None,
    ) -> Dict[str, np.ndarray]:
        """One case (all frames) or one case/timestep as ``(1, [nt,] C, X, Y, Z)``."""
        case = self._case(int(case_idx))
        nx, ny, nz = case.x.size, case.y.size, case.z.size
        nt = case.t.size
        want = set(keys) if keys is not None else None
        if self._moving_window is not None and time_idx is None:
            raise ValueError(
                "[NpzDirStore] moving_window serves per-frame reads only; a full-case "
                "read has no single beam position to window on (per-frame windows differ).")

        def take_field(arr: np.ndarray) -> np.ndarray:
            # arr: (nt, [C,] X, Y, Z). Return (1, [C,] X, Y, Z) or full (1, nt, C, X, Y, Z).
            if time_idx is None:
                a = arr[np.newaxis]  # (1, nt, [C,] X, Y, Z)
                if a.ndim == 5:      # scalar field -> add channel axis
                    a = a[:, :, np.newaxis]
            else:
                a = arr[time_idx][np.newaxis]  # (1, [C,] X, Y, Z)
                if a.ndim == 4:
                    a = a[:, np.newaxis]
            return np.ascontiguousarray(a, dtype=np.float32)

        # spatial shape of the emitted per-sample tensors
        spatial = (nx, ny, nz)
        if time_idx is None:
            shape = (1, nt, 1, nx, ny, nz)
            coord_shapes = {
                "x": case.x.reshape(1, 1, 1, nx, 1, 1),
                "y": case.y.reshape(1, 1, 1, 1, ny, 1),
                "z": case.z.reshape(1, 1, 1, 1, 1, nz),
                "time_s": case.t.reshape(1, nt, 1, 1, 1, 1),
            }
        else:
            shape = (1, 1, nx, ny, nz)
            coord_shapes = {
                "x": case.x.reshape(1, 1, nx, 1, 1),
                "y": case.y.reshape(1, 1, 1, ny, 1),
                "z": case.z.reshape(1, 1, 1, 1, nz),
                "time_s": np.float32(case.t[time_idx]).reshape(1, 1, 1, 1, 1),
            }

        out: Dict[str, np.ndarray] = {}

        def emit(name: str, value: np.ndarray) -> None:
            if want is None or name in want:
                out[name] = value

        for name in self.field_names:
            emit(name, take_field(case.fields[name]))

        for name, base in coord_shapes.items():
            emit(name, np.ascontiguousarray(np.broadcast_to(base.astype(np.float32), shape)))
        emit("P", np.full(shape, case.power, dtype=np.float32))
        emit("V_scan", np.full(shape, case.speed, dtype=np.float32))

        # beam-centre scalar path channels, built per requested frame(s)
        if case.centres is not None:
            cx = case.centres[:, 0].astype(np.float32)
            cy = case.centres[:, 1].astype(np.float32)
            if time_idx is None:
                bx = cx.reshape(1, nt, 1, 1, 1, 1)
                by = cy.reshape(1, nt, 1, 1, 1, 1)
            else:
                bx = np.float32(cx[time_idx]).reshape(1, 1, 1, 1, 1)
                by = np.float32(cy[time_idx]).reshape(1, 1, 1, 1, 1)
            if "beam_x" in self._scalar_path_channels:
                emit("beam_x", np.ascontiguousarray(np.broadcast_to(bx, shape)))
            if "beam_y" in self._scalar_path_channels:
                emit("beam_y", np.ascontiguousarray(np.broadcast_to(by, shape)))

        # cumulative path fields (beam_dist / path_dose): (1, nt, 1, X, Y, Z) -> slice
        for name, arr in case.path_fields.items():
            emit(name, np.ascontiguousarray(arr if time_idx is None else arr[:, time_idx]))

        if self.material_channels:
            manifest = self._require_manifest(self.cases[int(case_idx)])
            for name, value in _build_material_channels(self.material_channels, manifest, shape).items():
                emit(name, value)

        if self._moving_window is not None:
            # Per-frame laser-tracking window: crop x to [ix0, ix0+w) around the
            # beam's CURRENT cell, clamped inside the domain (constant shape for
            # batching). x becomes beam-relative so every window spans the same
            # coordinate range; beam_x stays absolute (soak boundaries are
            # absolute, and the model still needs to know where the window sits).
            if case.centres is None:
                # `centres` is loaded ONLY when a scalar path channel (beam_x/beam_y) is
                # requested -- see the `self._scalar_path_channels` gate in _case. So the
                # usual cause is the INPUT KEY SET, not the data: the manifest generally does
                # carry the path, it was simply never read. Blaming the manifest sent an
                # investigation to the wrong machine on 2026-08-22 (a six-channel moving-ref
                # arm, [V_scan,P,H_norm,x,y,z], failed identically on two boxes and the staged
                # remote copy was wrongly diagnosed as incomplete).
                raise ValueError(
                    "[NpzDirStore] moving_window needs per-frame beam centres, which are only "
                    "loaded when a scalar path channel is requested. This input set asks for "
                    f"none: scalar_path_channels={sorted(self._scalar_path_channels)!r}. Add "
                    "'beam_x' (and 'beam_y') to FNO_input_keys, or drop the moving window. "
                    "Only if the manifest genuinely lacks a laser path is the data at fault.")
            behind, ahead = self._moving_window
            cx_t = float(case.centres[time_idx, 0])
            sl = (slice(None), slice(None), beam_following_x_slice(
                beam_cx=cx_t,
                x0=float(case.x[0]),
                dx=float(case.x[1] - case.x[0]),
                nx=nx,
                behind=behind,
                ahead=ahead,
            ))
            out = {k: np.ascontiguousarray(v[sl]) for k, v in out.items()}
            if want is None or "x" in want:
                out["x"] = np.ascontiguousarray(
                    (out["x"].astype(np.float64) - cx_t).astype(np.float32))

        if spatial_slices is not None:
            lead = 3 if time_idx is None else 2  # (B, [T,] C) leading axes before spatial
            sl = (slice(None),) * lead + spatial_slices
            out = {k: np.ascontiguousarray(v[sl]) for k, v in out.items()}

        return out


def maybe_build_npz_store(
    input_dir: str | Path, run_cfg: Any, *, force: bool = False
) -> Optional[NpzDirStore]:
    """Return an NpzDirStore for a consolidated-NPZ dir when lazy loading is on.

    `force` skips the `lazy_npz_enabled` opt-in for callers that have no other
    reader available (an aiida-pino npz_file collection), where declining would
    just fail later instead of falling back to anything.
    """
    if not is_consolidated_npz(input_dir):
        return None
    if not (force or lazy_npz_enabled(run_cfg)):
        return None
    if bool(getattr(run_cfg, "moving_ref_frame", False)):
        raise ValueError("[NpzDirStore] lazy NPZ loading does not support moving_ref_frame=true.")
    input_keys = []
    if hasattr(run_cfg, "keys") and getattr(run_cfg.keys, "str_fno_input_keys", None) is not None:
        input_keys = list(run_cfg.keys.str_fno_input_keys)
    # A cached case is ~0.5 GB decompressed (fields + cumulative path channels), and
    # train + val stores each hold one LRU, so keep the cache small: training is
    # shuffled (low hit rate anyway) and validation reads a case's frames in order
    # (size 1-2 already gives full within-case reuse). Default 3; tune with
    # PINO_LAZY_CACHE. cache_size=16 x 2 stores OOM-killed the box under concurrency.
    cache_size = int(os.environ.get("PINO_LAZY_CACHE", "3"))
    # PINO_LAZY_STRIDE="sx,sy,sz" trains on a coarser grid without writing a second
    # copy of the dataset. RIGNO needs it (the full 150x100x40 graph does not fit
    # the 24 GB card); aiida-pino sets it per training profile. Unset = full grid.
    raw_stride = os.environ.get("PINO_LAZY_STRIDE", "").strip()
    if raw_stride:
        parts = [p for p in raw_stride.replace(" ", "").split(",") if p]
        if len(parts) != 3:
            raise ValueError(
                f"[NpzDirStore] PINO_LAZY_STRIDE must be 'sx,sy,sz', got '{raw_stride}'."
            )
        spatial_stride = _validate_stride(parts)
        # Refuse the one composition nothing else states: a variant that is
        # ALREADY coarse plus a read-time stride multiplies to a grid no log,
        # config or manifest names on its own.
        baked = _variant_stride(input_dir)
        if baked != (1, 1, 1):
            effective = tuple(a * b for a, b in zip(baked, spatial_stride))
            raise ValueError(
                f"[NpzDirStore] the dataset staged at '{input_dir}' is already strided "
                f"{baked[0]}x{baked[1]}x{baked[2]} by its derivation profile, and "
                f"PINO_LAZY_STRIDE='{raw_stride}' would stride it again -- effective "
                f"{effective[0]}x{effective[1]}x{effective[2]}. Use ONE mechanism: "
                f"either stage an unstrided variant and keep PINO_LAZY_STRIDE, or "
                f"keep the strided variant and unset PINO_LAZY_STRIDE."
            )
    else:
        spatial_stride = (1, 1, 1)
    # PINO_LAZY_MOVING_REF="behind,ahead" (cells): per-frame laser-tracking x-window,
    # the loader-side twin of PINO_LAZY_STRIDE -- applied at read time so the dataset
    # variant is unchanged. See test_npz_lazy_moving_ref.py.
    raw_window = os.environ.get("PINO_LAZY_MOVING_REF", "").strip()
    moving_window: tuple[int, int] | None = None
    if raw_window:
        parts = [p for p in raw_window.replace(" ", "").split(",") if p]
        try:
            if len(parts) != 2:
                raise ValueError
            moving_window = (int(parts[0]), int(parts[1]))
        except ValueError:
            raise ValueError(
                f"[NpzDirStore] PINO_LAZY_MOVING_REF must be 'behind,ahead' in cells, "
                f"got '{raw_window}'.") from None
    # PINO_LAZY_CROP_Z_TOP="N" (raw cells): gas-side z-crop at read time, the
    # loader-side twin of aiida-pino's `crop_z_top` derivation field. Read-time so
    # the collection stays pack-indexed instead of being materialised uncompressed.
    crop_z_top = _resolve_read_time_crop(
        input_dir, os.environ.get("PINO_LAZY_CROP_Z_TOP", ""))
    store = NpzDirStore(
        input_dir,
        requested_fields=run_cfg.vtk_fields,
        path_channels=[k for k in input_keys if k in PATH_CHANNEL_NAMES],
        material_channels=served_material_channels(run_cfg, input_keys),
        cache_size=cache_size,
        power_weighted_dose=power_weighted_dose_enabled(run_cfg),
        power_weighted_conduction=power_weighted_conduction_enabled(run_cfg),
        conduction_source_sigma_factor=conduction_source_sigma_factor(run_cfg),
        spatial_stride=spatial_stride,
        moving_window=moving_window,
        crop_z_top=crop_z_top,
    )
    _enforce_grid_contract(store, run_cfg)
    return store
