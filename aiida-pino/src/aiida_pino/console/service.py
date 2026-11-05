"""AiiDA console service: a thin, stateless wrapper over the aiida-pino
programmatic surface. Shared by the FastAPI web
console (`web/app`) and the `aiida-pino` CLI.

Browse/monitor are broker-free ORM reads (work whenever a profile is loaded).
Submit/kill need a running `verdi daemon` (RabbitMQ) — the exact precondition
the widget enforced by disabling its submit buttons — so they are gated behind
`daemon_available()` and raise DaemonUnavailableError (503) otherwise.
"""

from __future__ import annotations

import importlib.metadata
import logging
import math
import re
import shlex
import socket
import threading
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from ..utils.architecture import architecture_for_summary
from ..utils.profiles import effective_max_steps
from .config import Settings
from .errors import (
    DaemonUnavailableError,
    FeatureDeprecatedError,
    InvalidRequestError,
    NodeNotFoundError,
    OperationFailedError,
)

# OpenFOAM/LaserbeamFoam features are deliberately disabled (buggy, not fully
# functional); the workchains and adapters stay in the codebase and come back
# once fixed. Remove this guard (and the GUI/README notes) to re-enable.


# "…var_1_time_noref (test split)" — the split is the single most important
# qualifier on a reported metric, and the display name drops it: one model's
# test and val evaluations rendered as two identical rows on the same dataset.
_SPLIT_IN_LABEL = re.compile(r"\((train|val|validation|test)\s+split\)", re.IGNORECASE)


def _split_qualified(display: str | None, label: str | None) -> str | None:
    if not display:
        return display
    match = _SPLIT_IN_LABEL.search(str(label or ""))
    if not match or match.group(1).lower() in str(display).lower():
        return display
    return f"{display} ({match.group(1).lower()} split)"


_DATASET_ROLES = frozenset({"base", "composed", "derived", "subset"})
_DATASET_SOURCES = frozenset({"flow3d", "openfoam"})


# What a caller calls these, as opposed to what AiiDA calls them.
_HUMAN_TYPES = {
    "PinoDatasetCollectionData": "dataset",
    "PinoDatasetData": "dataset",
    "PinoDatasetUnionData": "dataset",
    "PinoSimulationData": "simulation",
    "PinoModelData": "model",
    "PinoModelEvaluationData": "evaluation",
    "ProcessNode": "process",
    "CalcJobNode": "calcjob",
    "WorkChainNode": "workchain",
    "FolderData": "folder",
    "InstalledCode": "code",
}


def _article(word: str) -> str:
    return "an" if word[:1].upper() in "AEIOU" else "a"


def _human_type(expected: type | tuple[type, ...] | None) -> str:
    """`dataset`, not `PinoDatasetCollectionData` -- and `node` when unconstrained."""
    if expected is None:
        return "node"
    types = expected if isinstance(expected, tuple) else (expected,)
    names: list[str] = []
    for kind in types:
        name = _HUMAN_TYPES.get(kind.__name__, kind.__name__)
        if name not in names:
            names.append(name)
    return " or ".join(names) if names else "node"


def _annotate_submitted_process(node, *, description=None, submitted_by=None) -> None:
    """Record WHY a run was launched and WHO launched it, at submit time.

    Both already existed as a second step -- `aiida-pino describe PK "..."` --
    and the developer notes ask for it right after every submission. In practice
    processes got a label and almost never a description: a step after the
    interesting part is done is a step that gets skipped, so these travel with the
    submission instead.

    `submitted_by` is free text and deliberately NOT auto-detected. Several people
    or tools can submit as one OS user (a profile often has a single real AiiDA
    user besides `aiida@localhost`), so any automatic value would be uniformly
    true and answer nobody's question. Who launched it is knowable only to the
    caller.

    A description a caller set deliberately is never overwritten -- the same rule
    the label follows. And nothing here may raise: the run has already launched
    by the time this is called, so a bookkeeping failure must not surface as a
    submission failure.
    """
    try:
        text = str(description or "").strip()
        if text and not str(getattr(node, "description", "") or "").strip():
            node.description = text
        who = str(submitted_by or "").strip()
        if who:
            node.base.extras.set("submitted_by", who)
    except Exception:  # noqa: BLE001 - the run is already submitted
        pass


def _label_submitted_process(node, entry_point: str, inputs: dict[str, Any]) -> None:
    """Give a submitted run a label that says what it is.

    4977 of 5718 processes on this profile have an empty label, so the Runs list
    is a column of identical class names and finding "the varspeed training" means
    traversing links. AiiDA leaves `label` empty by default and nothing here set
    it, so the only findable runs were the ones somebody titled by hand.

    The label is derived from the inputs that identify the run — the profile's
    `run_name`, the case name, the dataset pk — and never overwrites one a
    caller set deliberately.
    """
    try:
        if str(getattr(node, "label", "") or "").strip():
            return
        bits: list[str] = []
        for key in ("training_profile", "evaluation_profile"):
            profile = inputs.get(key)
            name = None
            if profile is not None:
                try:
                    name = str(profile.get_dict().get("run_name") or "").strip()
                except Exception:  # noqa: BLE001
                    name = None
            if name and name != "unnamed":
                bits.append(name)
                break
        run_config = inputs.get("run_config")
        if run_config is not None and not bits:
            try:
                case = str(run_config.get_dict().get("case_name") or "").strip()
                if case:
                    bits.append(case)
            except Exception:  # noqa: BLE001
                pass
        for key in ("train_dataset", "dataset", "target_dataset"):
            dataset = inputs.get(key)
            if dataset is not None:
                bits.append(f"ds{dataset.pk}")
                break
        model = inputs.get("model")
        if model is not None:
            bits.append(f"model{model.pk}")
        short = entry_point.split(".")[-1]
        node.label = " | ".join([short] + bits) if bits else short
    except Exception:  # noqa: BLE001 - a label must never fail a submission
        pass


def _broker_fault(exc: BaseException) -> bool:
    """Is this failure actually the daemon/broker, or something wearing its name?

    `_submit` and `kill_process` relabelled every exception as "the AiiDA
    daemon/broker is not reachable", three lines after `daemon_available()` had
    just confirmed over RPC that it *was* reachable. A saturated connection pool
    read as a dead daemon, so the documented response — restart the container,
    restart the daemon — was aimed at the wrong thing with runs in flight.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        root = type(current).__module__.split(".")[0]
        if root in {"sqlalchemy", "psycopg", "psycopg2"}:
            return False  # a saturated pool is not a dead broker, whatever it is named
        if isinstance(current, (ConnectionError, TimeoutError)):
            return True
        if root in {"kiwipy", "aio_pika", "aiormq", "pamqp", "plumpy"}:
            return True
        current = current.__cause__ or current.__context__
    return False


def _reject_unknown_training_keys(raw: dict[str, Any], base: dict[str, Any]) -> None:
    """Refuse a training_profile key nothing will read, suggesting the near miss.

    The submit path overlays packaged defaults, which is deliberate (a partial
    profile is meant to work) but means a misspelt key passes every check: the
    required-key set is satisfied by the defaults, the merge carries the typo
    through unread, and the run silently uses the default for whatever the user
    meant to set. Four runs died at exit 300 a minute after a "successful"
    submit for the mirror-image of this on the strict path.
    """
    import difflib

    from aiida_pino.utils.profiles import TRAINING_PROFILE_KEYS

    known = set(TRAINING_PROFILE_KEYS) | set(base) | {
        # accepted aliases (merge_training_profile normalises these)
        "VOF_pde_loss", "Diff_loss", "Diff_weight", "use_2D", "split_mode",
    }
    unknown = sorted(key for key in raw if key not in known)
    if not unknown:
        return
    hints = []
    for key in unknown:
        near = difflib.get_close_matches(key, sorted(known), n=1, cutoff=0.7)
        hints.append(f"`{key}`" + (f" (did you mean `{near[0]}`?)" if near else ""))
    raise InvalidRequestError(
        "unknown training_profile key(s): " + ", ".join(hints)
        + ". They would be stored but never read, and the run would use the default instead. "
        "`aiida-pino profile-keys training` lists every accepted key."
    )


def _check_run_config(config: Any, *, where: str = "run_config", runner_root: str = "") -> None:
    """Reject a config the runner will reject, before it costs a queue slot.

    Two layers: the shallow structural rules always run, and — when the runner
    checkout is known — the runner's own `load_and_validate` runs in-process,
    so submit-time acceptance means the same thing the runner means by it.
    """
    from aiida_pino.utils.flow3d_config import validate_run_config, validate_with_runner

    try:
        validate_run_config(config, where=where)
        if runner_root:
            validate_with_runner(config, runner_root, where=where)
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc


# CalcJobs that occupy a GPU. They take the gpu_spool seat lock from inside the
# adapter rather than queueing a task, so they never show up in the spool dirs
# the Monitor tab lists.
GPU_CALCJOB_LABELS = ("PinoTrainCalculation", "PinoEvalCalculation")
# Top-level runs that queue for a resource, so the Monitor can show what is
# waiting behind the file spools that AiiDA submissions never enter.
GPU_RUN_LABELS = frozenset({
    "TrainModelWorkChain", "TrainEvalWorkChain", "EvalOnlyWorkChain",
    "EvaluateModelWorkChain", "KfoldWorkChain", "ImportModelWorkChain",
    "PinoTrainCalculation", "PinoEvalCalculation",
})
FLOW3D_RUN_LABELS = frozenset({
    "Flow3DGenerateWorkChain", "Flow3DBatchWorkChain", "Flow3DRunCalculation",
})


def queued_vram_reservation(node) -> tuple[int | None, bool]:
    """(booked MiB, auto-sized?) for a queued GPU run.

    Reads both profiles: an eval books through `evaluation_profile`, and
    reading only `training_profile` rendered every queued eval as "exclusive
    card" when it had in fact declared a share (seen live 2026-07-28: twelve
    3000 MiB evals shown as exclusive). A training run that declares nothing
    is not exclusive either — `TrainModelWorkChain` sizes it from a measured
    peak (`utils/vram.py`) — so it reports as auto-sized instead.
    """
    declared = None
    for label in ("training_profile", "evaluation_profile"):
        try:
            value = int(getattr(node.inputs, label).get_dict().get("gpu_vram_mb") or 0)
        except Exception:  # noqa: BLE001 - profile absent for this run type
            continue
        if value > 0:
            declared = value if declared is None else max(declared, value)
    if declared is not None:
        return declared, False
    try:
        auto = getattr(node.inputs, "training_profile", None) is not None
    except Exception:  # noqa: BLE001 - no inputs at all
        auto = False
    return None, auto


def classify_run(label: str, progress: dict | None, *, on_gpu: bool, has_seat: bool) -> tuple[str, str]:
    """(category, phase) for a top-level active AiiDA run.

    Running means a live resource: a GPU process on a card, or a running
    Flow3DRunCalculation holding the seat. Otherwise the run is queued —
    submitted, waiting for its resource. A training run that logged `waiting for
    GPU seat` is queued even if it half-started.
    """
    progress = progress or {}
    if label in GPU_RUN_LABELS:
        running = (on_gpu or progress.get("step") is not None) and not progress.get("waiting_for_gpu")
        return "gpu", "running" if running else "queued"
    if label in FLOW3D_RUN_LABELS:
        return "flow3d", "running" if has_seat else "queued"
    return "other", "running"


logger = logging.getLogger(__name__)


KFOLD_DEPRECATION_MESSAGE = (
    "k-fold submission is DEPRECATED from the console. It is not broken and "
    "nothing was deleted: the machinery (KfoldTrainEvalWorkChain, its CalcJob, "
    "parser and adapter) is intact and a deliberate submission through the "
    "workchain still works. It is withdrawn from the console because it is not "
    "tested to the standard of the other submission paths: it had no split "
    "guard until late in development, so it could train k folds on an "
    "evaluation-only dataset silently. See aiida-pino docs/developer_notes.md "
    "for what validating it would take."
)


def is_evaluation_only_split_set(splits) -> bool:
    """True when a label set means "reserved for evaluation, never trained on".

    POSITIVE evidence only: at least one held-out label, and no trainable one.
    The old rule was equality with {"test"}, which let two live shapes through —
    `{eval}` (two validation-campaign datasets keyworded evaluation_only) and any mix
    such as `{test, excluded}`, which is what an ablation evaluation set looks
    like once exclusions are recorded.

    The tempting inversion — "no member resolves to train or val" — is wrong and
    was measured: it reclassifies 987 of 1731 live collections as
    evaluation-only, because `unassigned` is the normal state of a freshly
    finalised batch dataset, and it would refuse training on seven collections
    that already have training links. Absence of labels is not evidence of
    reservation.
    """
    from aiida_pino.utils.annotations import EXCLUDED_SPLIT_VALUES, HELD_OUT_SPLIT_VALUES

    labels = {str(x) for x in (splits or set())}
    if not labels:
        return False
    held_out = labels & set(HELD_OUT_SPLIT_VALUES)
    if not held_out:
        return False
    ignorable = set(HELD_OUT_SPLIT_VALUES) | set(EXCLUDED_SPLIT_VALUES)
    return not (labels - ignorable)


def _compute_case_stats_isolated(uuid: str, *, timeout_s: int = 3600) -> bool:
    """Compute one case's melt-pool stats in a CHILD process. True on success.

    `compute_case_metrics` loads whole field stacks (~810 MB per field for a
    337-frame 600k-cell case). In-process, that memory is never fully returned:
    glibc's dynamic mmap threshold grows after large frees, so the blocks come
    from the arena and stay. Measured on the live console: one case took RSS
    12.03 -> 16.40 GiB, and `malloc_trim` afterwards recovered only 10 of the
    22 GiB another case had added. A long-lived console therefore ratchets --
    it reached 89 GB in a day and drove the box under its MemAvailable floor.
    A child process gives back 100 % of it by exiting, which is the only
    bound that holds. The child writes the extras itself, so nothing large
    crosses back.
    """
    import subprocess
    import sys

    from aiida.manage.configuration import get_profile

    profile = getattr(get_profile(), "name", None)
    code = (
        "import sys\n"
        "from aiida import load_profile\n"
        f"load_profile({profile!r}) if {profile!r} else load_profile()\n"
        "from aiida.orm import load_node\n"
        "from aiida_pino.console import viz\n"
        "viz.get_or_compute_case_stats(load_node(sys.argv[1]), recompute=True)\n"
    )
    try:
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code, str(uuid)],
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("stats compute timed out for sim %s after %ss", uuid, timeout_s)
        return False
    if proc.returncode != 0:
        logger.warning("stats compute failed for sim %s: %s", uuid, (proc.stderr or "")[-400:])
        return False
    return True


def _trim_malloc_arena() -> None:
    """Return glibc's free arena to the OS.

    Python frees the field stacks, but glibc raises its dynamic mmap threshold
    once it has seen large frees: subsequent ~800 MB allocations then come from
    the heap arena, and freeing them does NOT shrink RSS. Without this the
    console grows by GBs per computed case and never gives it back (measured
    12.03 -> 16.40 GiB for a single case; 89 GB over a day, which drove the box
    under its MemAvailable floor).
    """
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001 - non-glibc or restricted; best effort
        pass


def _pino_version() -> str | None:
    for dist in ("lpno", "lpno"):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _code_role(executable: str, label: str) -> str:
    hay = f"{executable} {label}".lower()
    if "pvpython" in hay:
        return "pvpython"
    if executable.rsplit("/", 1)[-1].lower() in {"bash", "sh", "zsh", "dash"}:
        return "shell"
    if "openfoam" in hay or "foam" in hay or "shell" in hay:
        return "openfoam"
    return "python"


def _node_keywords(node) -> list[str]:
    from aiida_pino.utils.annotations import normalize_keywords

    try:
        return normalize_keywords(node.base.extras.get("keywords", []))
    except Exception:  # noqa: BLE001
        return []


# Human-readable dataset names for the known data folders; anything unknown
# falls back to a generic prettification of the folder name. Display-layer
# only — labels/identifiers in the DB are untouched.
_DISPLAY_NAME_OVERRIDES = {
    "material_lhs_300": "Material LHS 300",
    "material_lhs_300_steady_split": "Material LHS 300 — steady",
    "material_lhs_300_steady_val_quick": "Material LHS 300 steady — quick val subset",
    "material_keyhole_300": "Material keyhole 300",
    "ti64_random150": "Ti64 random paths (150)",
    "ti64_random150_ds_split": "Ti64 random paths — downsampled",
    "ti64_random150_ds_valsub": "Ti64 random paths downsampled — quick val subset",
    "original_straight_path_data": "Straight path Ti64",
    "original_straight_path_steady": "Straight path Ti64 — steady",
    "resampled_straight_path_data": "Straight path Ti64 — resampled",
    "single_test": "Straight path Ti64 — 5 µm fine mesh",
    "fine_mesh_vtk": "Straight path Ti64 — 5 µm fine mesh",
    "ti64_random_keyhole150": "Ti64 random keyhole paths (150)",
    "random_paths": "Random paths (fixed P, V)",
}


# Dataset lineage for the known data folders (display layer, like the names
# above): effective role + parent folder for collections that are derived views
# (steady / resampled / downsampled) or quick-val subsets of another
# collection. The stored dataset_role attribute is immutable on stored nodes
# and was imported as "base" everywhere, so the console computes the effective
# role here.
_FOLDER_LINEAGE = {
    "material_lhs_300_steady_split": ("derived", "material_lhs_300"),
    "material_lhs_300_steady_val_quick": ("subset", "material_lhs_300_steady_split"),
    "ti64_random150_ds_split": ("derived", "ti64_random150"),
    "ti64_random150_ds_valsub": ("subset", "ti64_random150_ds_split"),
    "original_straight_path_steady": ("derived", "original_straight_path_data"),
    "resampled_straight_path_data": ("derived", "original_straight_path_data"),
}


def display_name_for_folder(folder: str | None) -> str:
    if not folder:
        return ""
    if folder in _DISPLAY_NAME_OVERRIDES:
        return _DISPLAY_NAME_OVERRIDES[folder]
    pretty = folder.replace("_", " ").strip()
    return pretty[:1].upper() + pretty[1:] if pretty else ""


def _case_name_for_sim(meta: dict[str, Any], node) -> str:
    """Best human name for one simulation: source NPZ stem, else group/identifier."""
    uri = str(meta.get("source_uri") or "")
    if uri.startswith("file://"):
        stem = PurePosixPath(uri[len("file://") :]).stem
        if stem:
            return stem
    group = str(meta.get("source_group") or "").strip("/")
    if group:
        return group.rsplit("/", 1)[-1]
    return str(meta.get("simulation_identifier") or node.uuid)


def _model_architecture(summary: dict[str, Any]) -> str:
    """Coarse architecture family from the config/run names (filterable label).

    Delegates to the shared table in `aiida_pino.utils.architecture`, which is also
    what stamps the architecture keyword onto a model node, so the displayed label and
    the filterable keyword cannot drift. The private list this used to carry had no
    GAOT entry, so every GAOT run displayed as the "FNO / PINO" fall-through.
    """
    return architecture_for_summary(summary)



CURVES_EXTRA = "curves_v1"
CURVES_VERSION = 1


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """{"T": {"rmse": ...}} -> {"T.rmse": ...}; only finite numbers survive."""
    flat: dict[str, float] = {}
    for field, block in (metrics or {}).items():
        if not isinstance(block, dict):
            continue
        for name, value in block.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                flat[f"{field}.{name}"] = float(value)
    return flat


def _member_split_counts(member_uuids: list[str]) -> dict[str, int]:
    return _member_stats(member_uuids)["split_counts"]


def _member_facts(member_uuids: list[str]) -> dict[str, tuple]:
    """Split/timestep facts per member uuid, in one query.

    Listing every dataset means the same simulations get asked about many times
    over; one query for the whole page beats one per row (the per-row version
    took ~5 s for 73 datasets).
    """
    from aiida.orm import QueryBuilder

    from aiida_pino.data import PinoSimulationData

    if not member_uuids:
        return {}
    q = QueryBuilder().append(
        PinoSimulationData,
        filters={"uuid": {"in": list(member_uuids)}},
        project=["uuid", "extras.split", "attributes.num_timesteps", "attributes.effective_end_time_s",
                 "extras.data_evicted"],
    )
    return {str(uuid): (split, n, end, evicted) for uuid, split, n, end, evicted in q.all()}


def _member_stats(member_uuids: list[str], facts: dict[str, tuple] | None = None) -> dict[str, Any]:
    """Split counts plus timestep statistics over a collection's members.

    A single "Steps" number only makes sense when num_timesteps is uniform;
    for variable-length members the (near-)uniform step size Δt is the better
    indicator, so both are reported and the UI picks.
    """
    out: dict[str, Any] = {"split_counts": {}, "steps_min": None, "steps_max": None, "step_size_s": None}
    if not member_uuids:
        return out
    if facts is None:
        facts = _member_facts(member_uuids)
    steps: list[int] = []
    dts: list[float] = []
    for uuid in member_uuids:
        if str(uuid) not in facts:
            continue
        split, n, end = facts[str(uuid)][:3]
        key = str(split or "unassigned")
        out["split_counts"][key] = out["split_counts"].get(key, 0) + 1
        if n:
            steps.append(int(n))
            if end and n > 1:
                dts.append(float(end) / (int(n) - 1))
    if steps:
        out["steps_min"], out["steps_max"] = min(steps), max(steps)
    if dts and max(dts) <= min(dts) * 1.05:  # near-uniform step size
        out["step_size_s"] = sum(dts) / len(dts)
    return out


def silence_rabbitmq_version_noise() -> None:
    """Drop aiida-core's per-connect 'RabbitMQ vX is not supported' warnings.

    The documented mitigation (consumer_timeout raised, see web/README.md) is
    in place; three copies of the warning per submit only bury the
    real output in scripts and logs.
    """
    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "is not supported and will cause unexpected problems" not in record.getMessage()

    logging.getLogger("aiida.broker.rabbitmq").addFilter(_Filter())


def _served_code_version() -> dict[str, Any]:
    """The commit this *running process* imported, and when it started.

    Read once and cached: the answer cannot change without a restart, which is
    the whole point of asking. Best-effort — a deployment from a tarball has no
    git dir and says so rather than guessing.
    """
    global _SERVED_CODE
    if _SERVED_CODE is None:
        import subprocess
        import time

        sha = ""
        try:
            sha = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(Path(__file__).resolve().parents[3]),
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout.strip()
        except Exception:  # noqa: BLE001 — no git, no checkout, no clock to burn
            sha = ""
        _SERVED_CODE = {
            "served_sha": sha or None,
            "started_utc": _utc_now(),
            "started_monotonic": time.time(),
        }
    return dict(_SERVED_CODE)


_SERVED_CODE: dict[str, Any] | None = None


def _corrected_geometry(rows, scale):
    """Apply the recorded anisotropy correction to melt/keyhole geometry.

    `melt_geometry_stats` takes one cell size from the z spacing and uses it on
    all three axes, which is right only on an isotropic grid. A dataset derived
    with `stride=[2, 2, 1]` has dz=10 um but dx=dy=20 um, so every x/y length is
    **2x** too small and every volume **4x** too small. 137 of 141 evaluations
    already record the factors in `extras.geometry_scale`; nothing read them, so
    the console printed the small numbers with a confident "um".

    Depth is a z measurement and is left alone. Corrected values are marked, so
    a reader can tell them from the stored ones rather than wondering why a
    number moved.
    """
    if not isinstance(scale, dict) or scale.get("axes_isotropic", True):
        return rows
    xy = float(scale.get("length_correction_xy") or 1.0)
    vol = float(scale.get("volume_correction") or 1.0)
    if xy == 1.0 and vol == 1.0:
        return rows
    out = []
    for row in rows:
        quantity = str(row.get("quantity") or "")
        factor = vol if quantity == "volume" else (xy if quantity in ("width", "length") else 1.0)
        if factor == 1.0:
            out.append(row)
            continue
        fixed = dict(row)
        for key in ("true", "pred", "abs_err"):
            if isinstance(fixed.get(key), (int, float)):
                fixed[key] = fixed[key] * factor
        fixed["anisotropy_corrected"] = factor
        out.append(fixed)
    return out


def _process_state_filter(state: str | None) -> dict[str, Any]:
    """QueryBuilder filters for the Runs tab's state control.

    `failed` is not a process state in AiiDA: it is "excepted or killed, or
    finished with a non-zero exit status", which is why this has to be a query
    filter rather than a post-fetch test on a page of rows.
    """
    key = (state or "").strip().lower()
    if key in ("", "all"):
        return {}
    if key in ("queued", "running", "active"):
        return {"attributes.process_state": {"in": ["created", "running", "waiting"]}}
    if key == "finished":
        # `exit_status` is absent on workchains that finished cleanly, so "ok"
        # is 0 *or* the key missing. Expressed with has_key rather than a null
        # comparison: `{"==": None}` works on PostgreSQL and raises TypeError on
        # the SQLite backend the tests run against.
        return {
            "attributes.process_state": "finished",
            "or": [
                {"attributes.exit_status": {"==": 0}},
                {"attributes": {"!has_key": "exit_status"}},
            ],
        }
    if key == "failed":
        return {
            "or": [
                {"attributes.process_state": {"in": ["excepted", "killed"]}},
                {
                    "and": [
                        {"attributes.process_state": "finished"},
                        {"attributes": {"has_key": "exit_status"}},
                        {"attributes.exit_status": {"!==": 0}},
                    ]
                },
            ]
        }
    raise InvalidRequestError(
        f"unknown process state filter `{state}`; use all, queued, finished or failed"
    )


# Same precedence as the Runs tab's Type control expects: a label matching an
# earlier row is that category, so `models` beats `conversion` for ImportModel.
_PROCESS_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("generation", "like", ("%Flow3DGenerate%", "%Flow3DBatch%", "%Flow3DRun%", "%LaserbeamFoam%")),
    ("models", "ilike", ("%Train%", "%Eval%", "%Kfold%", "%ImportModel%")),
    ("conversion", "like", ("%Import%", "%Ingest%", "%ToDataset%", "%ToVtk%", "%VtkTo%")),
    ("derivation", "like", ("%Derive%",)),
)


def _process_category_filter(category: str | None) -> dict[str, Any]:
    """QueryBuilder filters for the Runs tab's type control.

    Applied in the query, like `state`: filtering the fetched page of 100 runs
    by type showed "No rows." for FLOW-3D generation whenever the 100 most
    recent processes were all evals.
    """
    key = (category or "").strip().lower()
    if key in ("", "all"):
        return {}
    names = [name for name, _, _ in _PROCESS_CATEGORIES]
    if key != "other" and key not in names:
        raise InvalidRequestError(
            f"unknown process category `{category}`; use all, {', '.join(names)} or other"
        )
    clauses: list[dict[str, Any]] = []
    for name, op, patterns in _PROCESS_CATEGORIES:
        if name == key:
            clauses.append({"or": [{"attributes.process_label": {op: pat}} for pat in patterns]})
            break
        # Earlier categories win, so a later one excludes their patterns.
        clauses.extend({"attributes.process_label": {"!" + op: pat}} for pat in patterns)
    return {"and": clauses}


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _member_digest(members) -> str:
    """Fingerprint of a member list, order included — an assignment made with a
    seed is only reproducible against the list it was made from."""
    import hashlib

    return hashlib.sha256("\n".join(str(m.uuid) for m in members).encode()).hexdigest()


def _compose_dataset_calcfunction(parameters, **inputs):
    """Deprecated alias: the composition calcfunction now lives in
    `utils.compose`, so the console and the dataset-filing paths share one
    implementation (review fix 2)."""
    from aiida_pino.utils.compose import compose_dataset_calcfunction

    return compose_dataset_calcfunction(parameters, **inputs)


def _require_dataset_collection(node, label: str):
    """A `training_source`/`validation_source` is the DATASET, not the training job.

    2026-08-23: passing a training `CalcJobNode` or its `WorkChainNode` failed with a bare
    `TypeError: Cannot convert value of type <CalcJobNode> to AiiDA type` and launched
    nothing. That message comes from AiiDA coercing an input against a port typed
    `orm.Data`, so a PROCESS node dies there before reaching
    `normalize_dataset_collection_node`, which would have said what it wanted. Checking at
    submit time puts the explanation where the caller can act on it.

    A process node is a near-miss rather than a random wrong answer -- the dataset the
    caller means is one link away -- so it is named in the message, turning the recovery
    into one step instead of a hunt through the provenance graph.
    """
    from aiida import orm

    from aiida_pino.data import PinoDatasetCollectionData

    if isinstance(node, PinoDatasetCollectionData):
        return node

    hint = ""
    if isinstance(node, orm.ProcessNode):
        candidates = [
            f"{link.node.pk} ({link.link_label})"
            for link in node.base.links.get_incoming().all()
            if isinstance(link.node, PinoDatasetCollectionData)
        ]
        hint = (
            f" That looks like a training job; the dataset it used is "
            f"{', '.join(candidates)}." if candidates else
            " That looks like a training job, not a dataset; pass the dataset collection "
            "it trained on."
        )
    raise InvalidRequestError(
        f"`{label}` must be a PinoDatasetCollectionData (the dataset the model was "
        f"trained on); pk={getattr(node, 'pk', None)} is a {type(node).__name__}.{hint}"
    )


class _ProjectedAttrs:
    """`.get()`/`.all` over a projected dict, so node-shaped code needs no changes."""

    __slots__ = ("_data",)

    def __init__(self, data):
        self._data = data if isinstance(data, dict) else {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    @property
    def all(self):
        return self._data


class _ProjectedBase:
    __slots__ = ("attributes", "extras")

    def __init__(self, metadata, extras):
        self.attributes = _ProjectedAttrs({"evaluation_metadata": metadata})
        self.extras = _ProjectedAttrs(extras)


class _ProjectedEval:
    """An evaluation built from a PROJECTION, wearing just enough of the node API.

    `_model_eval_rows` reads `pk`, `label`, `ctime`, `base.attributes["evaluation_metadata"]`
    and a handful of extras. Every one of those is projectable, but reading them off a real
    node is a lazy DB fetch each: 120 evaluation nodes cost 3.07 s of a listing.

    Deliberately a stand-in rather than a rewrite of the row builder. The body that turns an
    evaluation into a row is where the correctness risk lives -- Kelvin conversion, geometry
    correction, held-out warnings -- so it is left byte-for-byte alone and fed different
    objects. `tests/test_models_listing_projection.py` compares the result field-for-field
    against the node-built rows on the live profile.
    """

    __slots__ = ("pk", "label", "ctime", "base")

    def __init__(self, pk, label, ctime, metadata, extras):
        self.pk = pk
        self.label = label
        self.ctime = ctime
        self.base = _ProjectedBase(metadata, extras)


def cap_dataset_rows(rows: list, limit: int) -> list:
    """Trim a dataset listing to `limit`, never dropping a CURATED row.

    The cap exists for cost: an uncapped listing of 2529 rows measured 14.3 s. It used to
    protect only `core`, which was too narrow once the store was curated -- 358 of 2529
    rows sit in a filed folder outside the computed `Single-run imports` bucket, so the cap
    could silently drop a curated set and the browser would show a folder missing members
    it demonstrably has. A listing that omits rows without saying so is the same failure
    class as a query whose scope is invisible in its answer.

    Filed means a real folder that is not the computed orphan bucket; the orphan bucket is
    exactly what the cap is for. Measured 2026-09-01: every curated row fits inside the cap
    with ~142 slots spare.

    A module-level function rather than an inline branch so it can be tested directly --
    the test that reached into `list_datasets` to exec its source was unreadable and broke
    on whitespace.
    """
    from aiida_pino.console.folders import ORPHANED

    def curated(row) -> bool:
        folder = str(row.get("folder") or "")
        keywords = row.get("keywords") or []
        return "core" in keywords or (bool(folder) and folder != ORPHANED)

    protected = [r for r in rows if curated(r)]
    remaining = max(0, int(limit) - len(protected))
    head = [r for r in rows if not curated(r)][:remaining]
    out = protected + head
    out.sort(key=lambda r: r.get("ctime") or "", reverse=True)
    return out



def split_resubmit_overrides(overrides: dict | None) -> tuple[dict, dict]:
    """`derivation.<key>` overrides go to the cloned derivation profile; the rest to training.

    The derivation is the field a bad promotion corrupted (2026-09-10), and `resubmit
    --override` could not reach it -- the one command meant for "rerun it correctly" left
    the wrong part untouched (random paths, 2026-09-11).
    """
    training: dict = {}
    derivation: dict = {}
    for key, value in (overrides or {}).items():
        if key.startswith("derivation."):
            derivation[key[len("derivation."):]] = value
        else:
            training[key] = value
    return training, derivation


class AiidaConsoleService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Thread-local, not an instance attribute: every endpoint is a sync
        # `def` on anyio's threadpool sharing one service, so request state
        # kept on `self` is visible to every other request in flight. The
        # facet maps had the same shape and did cross (round-4 m1); this one
        # is only correct by accident, because its entries key on node uuid.
        self._local = threading.local()
        # SHARED across requests on purpose, unlike `_local` above. Collapsing
        # polls is the entire point: two browsers watching one remote job must
        # share the connection attempt, not each open their own. Keys carry the
        # pk, so the worst a race can do is duplicate an attempt — never serve
        # one job's log for another.
        from aiida_pino.console.tail_cache import TailCache

        self._tail_cache = TailCache(ttl_seconds=float(getattr(settings, "tail_cache_seconds", 5.0)))
        # A SEPARATE, longer-lived cache for the monitor's progress tails. The log pane's 5 s
        # is tuned for someone watching a log; the Monitor polls every ~12 s and would then
        # SSH on essentially every poll, for every remote run at once. 30 s costs at most one
        # connection per run per half-minute and still moves a step counter visibly.
        self._progress_tail_cache = TailCache(
            ttl_seconds=float(getattr(settings, "progress_tail_cache_seconds", 30.0))
        )
        silence_rabbitmq_version_noise()

    @classmethod
    def build_default(cls, settings: Settings) -> "AiidaConsoleService":
        from aiida import load_profile

        from aiida_pino.utils.sqlite_tuning import tune_sqlite_connections

        tune_sqlite_connections()  # WAL + busy timeout, before the engine connects
        load_profile(settings.aiida_profile or None)
        return cls(settings)

    # ------------------------------------------------------------------ health
    def health(self) -> dict[str, Any]:
        from aiida.manage import get_manager

        available, reason = self.daemon_available()
        try:
            profile_name = get_manager().get_profile().name
        except Exception:  # pragma: no cover
            profile_name = None
        return {
            "service": "aiida-console",
            "profile": profile_name,
            "daemon": {"available": available, "reason": reason},
            # What this process is actually serving. A long-lived uvicorn keeps
            # the code it imported at startup, so "fixed" and "fixed for the
            # person using it" drifted by three days and four fixes: `limit=0`
            # answered zero rows and `/datasets/1601` had no supersession
            # fields, on a HEAD where both worked. `served_sha != HEAD` is the
            # one check that catches it, and it has to be readable from outside.
            "code": _served_code_version(),
        }

    def daemon_available(self) -> tuple[bool, str]:
        """Best-effort, never raises. True only when a submission would actually
        run (daemon up ⇒ broker up)."""
        try:
            from aiida.engine.daemon.client import get_daemon_client

            client = get_daemon_client()
            if not bool(client.is_daemon_running):
                return (
                    False,
                    "AiiDA daemon is not running; start the RabbitMQ broker and "
                    "`verdi daemon start` to submit workflows",
                )
            # is_daemon_running only checks the PID file; a stale PID or a down
            # broker still reports running. get_daemon_info() does a real RPC to
            # the daemon and raises if it is not truly reachable.
            client.get_daemon_info()
            return True, "daemon running"
        except Exception as exc:  # noqa: BLE001
            return (
                False,
                f"AiiDA daemon not reachable ({type(exc).__name__}); submissions need a "
                "RabbitMQ-backed profile with a running `verdi daemon`",
            )

    # ------------------------------------------------------------- environment
    def env_status(self) -> dict[str, Any]:
        from aiida.manage import get_manager

        available, reason = self.daemon_available()
        try:
            profile_name = get_manager().get_profile().name
        except Exception:  # pragma: no cover
            profile_name = None
        codes = self.list_codes()
        return {
            "lpno_version": _pino_version(),
            "profile": profile_name,
            "daemon": {"available": available, "reason": reason},
            "codes": codes,
            "config_names": self.config_names(),
        }

    def list_codes(self) -> dict[str, Any]:
        from aiida.orm import InstalledCode, QueryBuilder

        rows: list[dict[str, Any]] = []
        for code in QueryBuilder().append(InstalledCode, project=["*"]).all(flat=True):
            try:
                executable = str(code.get_executable())
            except Exception:  # noqa: BLE001
                executable = ""
            label = code.full_label
            rows.append(
                {
                    "pk": code.pk,
                    "label": label,
                    "executable": executable,
                    "role": _code_role(executable, label),
                }
            )
        rows.sort(key=lambda r: r["label"])
        by_role: dict[str, list[str]] = {"python": [], "openfoam": [], "pvpython": []}
        for r in rows:
            by_role.setdefault(r["role"], []).append(r["label"])
        return {"all": rows, "by_role": by_role}

    def config_names(self) -> list[str]:
        from aiida_pino.utils import available_config_names

        try:
            return list(available_config_names())
        except Exception:  # noqa: BLE001
            return []

    def training_defaults(self, config_name: str | None = None) -> dict[str, Any]:
        from aiida_pino.utils import load_training_profile_defaults

        return load_training_profile_defaults(config_name or None)

    # ----------------------------------------------------------- node loading
    @staticmethod
    def _load_node(pk: int, expected: type | tuple[type, ...] | None = None):
        from aiida.orm import load_node

        wanted = _human_type(expected)
        try:
            # A pk OR a uuid (full or a unique prefix). The staging guard has a
            # uuid in hand and prints it as the remedy an operator copies, so
            # rejecting uuids made that remedy die with `invalid int value` --
            # an error whose fix-it line is the part that fails is worse than
            # one with no fix-it line at all.
            identifier = str(pk).strip()
            if identifier.isdigit():
                # An all-digit string is AMBIGUOUS: it is usually a pk, but a
                # uuid PREFIX can be all digits too (caught by a test whose
                # random uuid began `17626629`). Try the pk reading first, then
                # fall back — guessing from the shape alone made the failure
                # intermittent, which is worse than either answer.
                try:
                    node = load_node(int(identifier))
                except Exception:  # noqa: BLE001 - fall through to the uuid reading
                    # `uuid=` explicitly: passing the digits back to load_node()
                    # re-interprets them as a pk and fails the same way, so the
                    # fallback has to NAME the interpretation it wants.
                    node = load_node(uuid=identifier)
            else:
                node = load_node(identifier)
        except Exception as exc:  # noqa: BLE001
            # "NotExistent" is AiiDA's class name, not information: the caller
            # knows the node is missing, they want to know *what* was missing.
            raise NodeNotFoundError(f"{wanted} {pk} not found") from exc
        if expected is not None and not isinstance(node, expected):
            raise NodeNotFoundError(
                f"node {pk} is {_article(type(node).__name__)} {type(node).__name__}, not a {wanted}"
            )
        return node

    # ------------------------------------------------------------ annotations
    def update_annotations(
        self,
        pk: int,
        *,
        title: str | None = None,
        description: str | None = None,
        split: str | None = None,
        split_reason: str | None = None,
        keywords_set: list[str] | None = None,
        keywords_add: list[str] | None = None,
        keywords_remove: list[str] | None = None,
    ) -> dict[str, Any]:
        from aiida_pino.utils.annotations import update_node_annotations

        node = self._load_node(pk)
        try:
            return update_node_annotations(
                node,
                title=title,
                description=description,
                split=split,
                split_reason=split_reason,
                keywords_set=keywords_set,
                keywords_add=keywords_add,
                keywords_remove=keywords_remove,
            )
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc

    def set_folder(self, pks: list[int], path: str | None) -> dict[str, Any]:
        """File datasets under `path` (a slash-separated folder), or unfile them."""
        from . import folders as folders_mod

        moved = [folders_mod.set_folder(self._load_node(int(pk)), path) for pk in pks]
        return {"pks": [int(pk) for pk in pks], "folder": moved[0] if moved else None}

    def list_folders(self, *, include_computed: bool = False) -> dict[str, Any]:
        """Filed folders by default.

        The web client uses this for the Move box's autocomplete, where a
        computed folder is not a legal target, so it must not appear there.
        The CLI asks for them (marked) because its listing is read as an
        inventory, and leaving them out lost 722 of 836 datasets.
        """
        from . import folders as folders_mod

        return {"folders": folders_mod.list_folders(include_computed=include_computed)}

    def rename_folder(self, old: str, new: str) -> dict[str, Any]:
        from . import folders as folders_mod

        try:
            return {"folder": folders_mod.rename_folder(old, new)}
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc

    def remove_folder(self, label: str) -> dict[str, Any]:
        """Delete a folder; its datasets are unfiled, never deleted."""
        from . import folders as folders_mod

        try:
            return {"folder": label, "unfiled": folders_mod.remove_folder(label)}
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc

    # -------------------------------------------------------------- datasets
    def list_datasets(
        self,
        *,
        search: str | None = None,
        role: str | None = None,
        source: str | None = None,
        keywords: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        from aiida_pino.utils import describe_dataset_record
        from aiida_pino.utils.annotations import normalize_keywords
        from aiida_pino.utils.data_nodes import active_dataset_records

        from . import folders as folders_mod

        # Datasets are browsed, not tailed. Build rows for the whole active
        # population, filter, and truncate LAST: filtering after a recency
        # window is what made all four `core` reference sets unreachable from
        # every listing, search and folder on 2026-07-30 (1064 collections, cap
        # 500, the core sets among the oldest).
        # A typo'd filter used to return an empty list under HTTP 200, which
        # reads as "there are none" rather than "that is not a role".
        if role not in (None, "", "all") and role not in _DATASET_ROLES:
            raise InvalidRequestError(
                f"unknown role {role!r}; expected one of: all, {', '.join(sorted(_DATASET_ROLES))}"
            )
        if source not in (None, "", "all") and source not in _DATASET_SOURCES:
            raise InvalidRequestError(
                f"unknown source {source!r}; expected one of: all, {', '.join(sorted(_DATASET_SOURCES))}"
            )
        limit = self.settings.clamp_limit(limit if limit is not None else self.settings.list_max_limit)
        wanted_keywords = normalize_keywords(keywords)
        rows: list[dict[str, Any]] = []
        member_sets: dict[int, set[str]] = {}
        # One projected query for the whole population instead of loading a node
        # per row and reading its metadata, description and extras off it: that
        # was 78-92 s for 734 collections, against 0.1-0.5 s for the same
        # population projected (which `dataset_options` has always done).
        # Filters have to see every candidate before it is known which ones a page holds, so
        # only an UNFILTERED page can be windowed at the fetch. That is the common case --
        # browsing -- and it is the difference between 15.7 s and 0.13 s.
        dataset_filtered = bool(search or wanted_keywords or (role not in (None, "", "all"))
                                or (source not in (None, "", "all")))
        if offset is not None and not dataset_filtered:
            records = active_dataset_records(
                include_superseded=include_superseded,
                limit=(limit or 0) or None,
                offset=max(int(offset), 0),
            )
            windowed = True
        else:
            records = active_dataset_records(include_superseded=include_superseded)
            windowed = False
        every_uuid: set[str] = set()
        for record in records:
            every_uuid.update(record["member_uuids"])
        facts = _member_facts(sorted(every_uuid))
        folder_of = folders_mod.folder_by_pk([record["pk"] for record in records])
        for record in records:
            try:
                meta = record["metadata"]
                extras = record["extras"]
                member_uuids = record["member_uuids"]
                node_pk = record["pk"]
                described = describe_dataset_record(node_pk, meta, record["label"])
                row = {
                    "pk": node_pk,
                    "uuid": record["uuid"],
                    "label": described,
                    "display_name": display_name_for_folder(meta.get("source_folder_name")) or described,
                    "description": record["description"],
                    "keywords": normalize_keywords(extras.get("keywords", []) or []),
                    "dataset_role": meta.get("dataset_role"),
                    # Base-dataset provenance for subset views, so the browser
                    # shows "test split of <base>" instead of leaving the origin
                    # a guess (2026-07-24).
                    "base_collection_pk": meta.get("base_collection_pk"),
                    "base_collection_label": meta.get("base_collection_label"),
                    "subset_split": meta.get("subset_split"),
                    "source_type": meta.get("source_type"),
                    "canonical_layout": meta.get("canonical_layout"),
                    "family_identifier": meta.get("family_identifier"),
                    "source_folder_name": meta.get("source_folder_name"),
                    "num_timesteps": meta.get("num_timesteps"),
                    "simulation_count": meta.get("simulation_count"),
                    "requested_fields": meta.get("requested_fields"),
                    **_member_stats(member_uuids, facts),
                    "evicted_count": sum(
                        1 for u in member_uuids if len(facts.get(str(u), ())) > 3 and facts[str(u)][3]
                    ),
                    "folder": None,   # resolved below: explicit, else inherited, else automatic
                    "folder_explicit": node_pk in folder_of,
                    # Supersession marks 526 of 1,070 collections and appeared in
                    # no payload and nowhere in the UI, so a dataset that had been
                    # replaced simply vanished from every listing with no way to
                    # ask why (review, section 5).
                    "superseded": bool(extras.get("superseded", False)),
                    "replaced_by_pk": extras.get("replaced_by_pk", None),
                    "replaced_by_uuid": extras.get("replaced_by_uuid", None),
                    "_folder_own": folder_of.get(node_pk),
                    "_parent_uuid": meta.get("parent_collection_uuid") or meta.get("base_collection_uuid"),
                    "parent_pk": None,
                    "parent_source": None,
                    "_composed_from": extras.get("composed_from", None) or [],
                    "ctime": record["ctime"].isoformat() if record["ctime"] else None,
                }
            except Exception:  # noqa: BLE001
                continue
            if member_uuids:
                member_sets[row["pk"]] = set(member_uuids)
            rows.append(row)
        # Resolve lineage before filtering, so role filters act on the
        # effective role and children keep their parent link.
        folder_to_pk = {r["source_folder_name"]: r["pk"] for r in rows if r["source_folder_name"]}
        pk_by_uuid = {r["uuid"]: r["pk"] for r in rows}
        for row in rows:
            # Recorded provenance first: a collection that knows its parent is
            # not a guessing problem. The name map and the member-subset
            # heuristic below only fill in for collections predating it.
            parent_pk = pk_by_uuid.get(str(row.get("_parent_uuid") or ""))
            if parent_pk and parent_pk != row["pk"]:
                row["parent_pk"] = parent_pk
                row["parent_source"] = "recorded"
                row["dataset_role"] = row["dataset_role"] or "derived"
                continue
            lineage = _FOLDER_LINEAGE.get(row["source_folder_name"] or "")
            if lineage:
                row["dataset_role"] = lineage[0]
                row["parent_pk"] = folder_to_pk.get(lineage[1])
                row["parent_source"] = "name map"
                continue
            row["dataset_role"] = row["dataset_role"] or "base"
            if row["dataset_role"] != "base":
                continue  # composed/derived keep their stored role
            # Fallback for unlisted quick-eval views: a strict member subset of
            # another collection nests under it. ponytail: O(n^2) set
            # comparison over the listed page — fine for tens of datasets.
            mine = member_sets.get(row["pk"])
            if not mine:
                continue
            best: tuple[int, int] | None = None  # (superset size, pk)
            for other in rows:
                theirs = member_sets.get(other["pk"])
                if other["pk"] == row["pk"] or not theirs:
                    continue
                if mine < theirs and (best is None or len(theirs) < best[0]):
                    best = (len(theirs), other["pk"])
            if best:
                row["parent_pk"] = best[1]
                row["parent_source"] = "inferred"
                row["dataset_role"] = "subset"
        for row in rows:
            # Only recorded lineage relocates a dataset; an inferred parent is a
            # display hint, not provenance.
            row["_inherit_pk"] = row["parent_pk"] if row["parent_source"] in ("recorded", "name map") else None
        folders_mod.resolve_rows(rows)
        for row in rows:
            row.pop("_folder_own", None)
            row.pop("_parent_uuid", None)
            row.pop("_inherit_pk", None)
        uuid_to_pk = {r["uuid"]: r["pk"] for r in rows}
        for row in rows:
            comp = row.pop("_composed_from", []) or []
            row["component_pks"] = [uuid_to_pk.get(str(c.get("uuid"))) or c.get("pk") for c in comp]
        out: list[dict[str, Any]] = []
        for row in rows:
            if role and role != "all" and row["dataset_role"] != role:
                continue
            if source and source != "all" and row["source_type"] != source:
                continue
            # ponytail: Python AND-filter alongside the search/role/source
            # post-filters; move into the QueryBuilder if dataset counts grow.
            if wanted_keywords and not set(wanted_keywords) <= set(row["keywords"]):
                continue
            # Also the name a user reads and the description they wrote: the
            # client used to search `display_name` and `description` on top of
            # this, which subtracted rows the server had matched, and dropping
            # that clause would otherwise have lost the two fields it covered.
            # 734 datasets carry a description; "enthalpy" matched none of them.
            if search and not self._matches(
                search,
                row["label"],
                row["family_identifier"],
                row["source_folder_name"],
                row.get("folder") or "",
                row.get("display_name") or "",
                row.get("description") or "",
            ):
                continue
            out.append(row)
        # Truncation is the last step and is reported, never silent. Protected
        # reference sets are never truncated away: they are the datasets a
        # newcomer is most likely to look for and the least likely to be recent.
        if windowed:
            # `len(out)` is the PAGE when the fetch was windowed, and reporting that as the
            # total would have said "no more datasets" after the first page -- infinite
            # scroll would stop at 60 of 2180. Count the population instead; the count is a
            # single cheap query against the same filter the fetch used.
            from aiida.orm import QueryBuilder

            from aiida_pino.data import PinoDatasetCollectionData as _Collection

            total = (
                QueryBuilder()
                .append(
                    _Collection,
                    filters=None if include_superseded else {"extras.superseded": {"!==": True}},
                )
                .count()
            )
        else:
            total = len(out)
        # Folder aggregates, computed here because only the server has the member
        # uuids. Summing per-dataset counts client-side double-counts every
        # simulation shared by two datasets in the folder: "Core reference"
        # showed 616 or 700 sims (depending on a checkbox) against 550 distinct
        # over 6 datasets, because its members overlap 783 memberships deep.
        folder_totals: dict[str, dict[str, Any]] = {}
        for row in out:
            name = str(row.get("folder") or "")
            if not name:
                continue
            entry = folder_totals.setdefault(name, {"datasets": 0, "_uuids": set()})
            entry["datasets"] += 1
            entry["_uuids"].update(member_sets.get(row["pk"], ()))
        served_folders = {
            name: {"datasets": e["datasets"], "simulations": len(e["_uuids"])}
            for name, e in folder_totals.items()
        }
        # Keyword counts over the *whole* filtered population, for the same
        # reason: built from the loaded page they were wrong by up to 6x
        # (`self-improve` offered as 15 against 65) and two keywords with ten
        # and nine datasets were not offered at all.
        keyword_totals: dict[str, int] = {}
        for row in out:
            for keyword in row.get("keywords") or []:
                keyword_totals[str(keyword)] = keyword_totals.get(str(keyword), 0) + 1
        served_keywords = dict(sorted(keyword_totals.items(), key=lambda kv: (-kv[1], kv[0])))

        if windowed:
            # The fetch already returned exactly this page, so there is nothing left to
            # slice; `total` still counts the whole population, which is what tells the
            # caller more pages exist.
            truncated = total > (max(int(offset), 0) + len(out))
        elif offset is not None:
            # PAGINATED: a plain window, and deliberately WITHOUT the core-protection
            # reshuffle below. That protection exists because truncation HIDES rows -- a
            # `core` reference set dropped off a capped list is invisible. Pagination hides
            # nothing, it defers, so pinning core sets onto page one would instead duplicate
            # them: once at the top and again in their own ctime position further down.
            # Every non-paginated caller (the pickers, which cap at 100) keeps the old
            # behaviour exactly.
            start = max(int(offset), 0)
            out = out[start:start + limit] if limit else out[start:]
            truncated = total > (start + len(out))
        elif limit and total > limit:
            out = cap_dataset_rows(out, limit)
            truncated = True
        else:
            truncated = False
        for row in out:
            row["_total"], row["_truncated"] = total, truncated
            # The facets ride on the payload rather than on the service: parked
            # on the shared singleton they could be read back by a *different*
            # concurrent request.
            row["_folders"], row["_keywords"] = served_folders, served_keywords
        return out

    def dataset_options(self) -> list[dict[str, Any]]:
        """Cheap dataset list for pickers: one projected query, no node loading.

        The browser's `/datasets` builds member statistics and lineage for every
        row (~5 s for 122 datasets); a dropdown needs none of that, and the
        train form was waiting on it six times over.
        """
        from aiida.orm import QueryBuilder

        from aiida_pino.data import PinoDatasetCollectionData

        builder = QueryBuilder()
        builder.append(
            PinoDatasetCollectionData,
            filters={"or": [{"extras": {"!has_key": "superseded"}}, {"extras.superseded": {"==": False}}]},
            project=[
                "id", "label", "ctime", "extras.folder",
                "attributes.simulation_count",
                "attributes.dataset_metadata.source_folder_name",
            ],
        )
        rows = []
        for pk, label, ctime, folder, count, source_folder in builder.all():
            rows.append({
                "pk": int(pk),
                "label": label,
                "display_name": display_name_for_folder(source_folder) or (str(label).split("|")[0].strip() if label else f"Dataset {pk}"),
                "folder": folder,
                "simulation_count": count,
                "ctime": ctime.isoformat() if ctime else None,
            })
        rows.sort(key=lambda r: r["ctime"] or "", reverse=True)
        return rows

    def warm_runtime_cache(self, pk: int, *, runtime_fields=None) -> dict[str, Any]:
        """Build a collection's LOCAL runtime cache now, in this process, off any job's path.

        The submit-time staging note tells a submitter to run `aiida-pino warm <pk>` when
        the resolved collection's cache is not built; until 2026-09-14 no such command
        existed (the audit's one refuted-as-not-mechanical bug). Same export
        `WarmRuntimeCacheCalculation` runs as a job, here in the foreground.
        """
        from aiida_pino.calculations.base import warm_dataset_runtime_cache
        from aiida_pino.data import PinoDatasetCollectionData

        node = self._load_node(pk, PinoDatasetCollectionData)
        path = warm_dataset_runtime_cache(node, runtime_fields=runtime_fields, transport_type="core.local")
        return {"pk": node.pk, "uuid": node.uuid, "cache": str(path)}

    def dataset_detail(self, pk: int) -> dict[str, Any]:
        from aiida.orm import FolderData

        from aiida_pino.data import PinoDatasetCollectionData, PinoDatasetData, PinoDatasetUnionData
        from aiida_pino.utils import (
            dataset_family_nodes,
            describe_dataset_node,
            source_folder_node_for_dataset,
            summarize_dataset_node,
        )

        node = self._load_node(pk, (PinoDatasetCollectionData, PinoDatasetData, PinoDatasetUnionData, FolderData))
        # `FolderData` is in that tuple for legacy datasets stored as plain
        # folders -- but PinoModelData, PinoSimulationData and
        # PinoModelEvaluationData all *subclass* it, so a model pk passed the
        # type check and then died on `member_simulation_uuids` with
        # "Internal error: AttributeError". Exact class, not isinstance.
        if not isinstance(node, (PinoDatasetCollectionData, PinoDatasetData, PinoDatasetUnionData)) and type(node) is not FolderData:
            raise NodeNotFoundError(
                f"node {pk} is {_article(type(node).__name__)} {type(node).__name__}, not a dataset"
            )
        summary = summarize_dataset_node(node)
        summary["describe"] = describe_dataset_node(node)
        summary["display_name"] = (
            display_name_for_folder(summary.get("source_folder_name")) or summary.get("label") or f"Dataset {pk}"
        )
        summary["description"] = node.description
        summary["keywords"] = _node_keywords(node)
        summary["superseded"] = bool(node.base.extras.get("superseded", False))
        summary["replaced_by_pk"] = node.base.extras.get("replaced_by_pk", None)
        summary["replaced_by_uuid"] = node.base.extras.get("replaced_by_uuid", None)
        member_uuids = list(node.base.attributes.get("member_simulation_uuids") or [])
        summary["split_counts"] = _member_split_counts(member_uuids)
        # The list row carries both of these and the pane asked for them by
        # name, but the payload never had them and `kv()` drops an undefined
        # value, so clicking a row lost the two numbers it was showing.
        summary["simulation_count"] = summary.get("num_samples") or len(member_uuids)
        summary["num_timesteps"] = (summary.get("metadata") or {}).get("num_timesteps") or (
            node.get_dataset_metadata().get("num_timesteps") if hasattr(node, "get_dataset_metadata") else None
        )
        from . import folders as folders_mod

        # Newest node per source-folder name, from one narrow query: 0.9 s for
        # 820 collections, against 6.5 s to project their metadata and member
        # lists. It backs both the folder fallback and the lineage chips, which
        # used to scan `recent_dataset_nodes(100)` — a ctime window three days
        # wide, so every older dataset resolved to nothing after a 7-16 s wait.
        folder_index: dict[str, list[int]] = {}

        def by_folder() -> dict[str, list[int]]:
            if not folder_index:
                from aiida.orm import QueryBuilder as _QB

                from aiida_pino.data import PinoDatasetCollectionData as _Coll

                query = _QB().append(
                    _Coll,
                    filters={"extras.superseded": {"!==": True}},
                    project=["id", "attributes.dataset_metadata.source_folder_name"],
                    tag="c",
                )
                query.order_by({"c": {"ctime": "desc"}})
                for other_pk, name in query.all():
                    folder_index.setdefault(str(name or ""), []).append(int(other_pk))
            return folder_index

        def pick(name: str) -> int | None:
            """The node the *list* means by this folder name, not just the newest.

            `original_straight_path_data` names two live collections: 10941, the
            newer, unfiled one, and 1389 in "Core reference", which is the one
            the tree nests under. Taking the newest made the pane's parent chip
            and its inherited folder disagree with the row the user clicked.
            """
            candidates = by_folder().get(name) or []
            filed = folders_mod.folder_by_pk(candidates)
            for candidate in candidates:
                if filed.get(candidate):
                    return candidate
            return candidates[0] if candidates else None

        own_folder = folders_mod.folder_by_pk([node.pk]).get(node.pk)
        summary["folder_explicit"] = own_folder is not None
        summary["folder_inherited_from"] = None
        folder_value = own_folder
        if folder_value is None:
            # A derived dataset lives where its base lives; say so here too, or
            # the pane reads "unfiled" for a row the list shows inside a folder.
            seen: set[str] = set()
            walker = node
            while folder_value is None:
                parent_uuid = walker.base.attributes.all.get("parent_collection_uuid") or walker.base.attributes.all.get(
                    "base_collection_uuid"
                )
                if not parent_uuid or str(parent_uuid) in seen:
                    break
                seen.add(str(parent_uuid))
                try:
                    from aiida.orm import load_node as _load_walk

                    walker = _load_walk(str(parent_uuid))
                except Exception:  # noqa: BLE001
                    break
                folder_value = folders_mod.folder_by_pk([walker.pk]).get(walker.pk)
                if folder_value is not None:
                    summary["folder_inherited_from"] = {"pk": walker.pk, "folder": folder_value}
            # The listing also inherits through the legacy folder-name map, and
            # up the whole chain: a resampled view read "Core reference" in the
            # list and "Folder: —" in the pane because its mapped parent
            # inherits too. A user filling in that blank writes an explicit
            # filing and changes how the dataset inherits, for good — two views
            # disagreeing must not cost that.
            name = str(summary.get("source_folder_name") or "")
            walked: set[str] = {name}
            while folder_value is None:
                entry = _FOLDER_LINEAGE.get(name)
                parent_name = entry[1] if entry else None
                if not parent_name or parent_name in walked:
                    break
                walked.add(parent_name)
                parent_pk = pick(parent_name)
                if parent_pk is None:
                    break
                folder_value = folders_mod.folder_by_pk([parent_pk]).get(parent_pk)
                if folder_value is not None:
                    summary["folder_inherited_from"] = {"pk": parent_pk, "folder": folder_value}
                name = parent_name
        summary["folder"] = folder_value or folders_mod.auto_folder(
            {
                "simulation_count": len(list(node.base.attributes.all.get("member_simulation_uuids") or [])),
                "keywords": summary["keywords"],
            }
        )
        folder = str(summary.get("source_folder_name") or "")
        lineage = _FOLDER_LINEAGE.get(folder)
        summary["dataset_role"] = lineage[0] if lineage else (summary.get("dataset_role") or "base")
        parent_folder = lineage[1] if lineage else None
        # Recorded provenance beats the legacy name map: a collection that knows
        # its parent should say so, and say that it knows.
        recorded_parent = node.base.attributes.all.get("parent_collection_uuid") or node.base.attributes.all.get(
            "base_collection_uuid"
        )
        if recorded_parent:
            try:
                from aiida.orm import load_node as _load_parent

                parent_node = _load_parent(str(recorded_parent))
                if parent_node.pk != node.pk:
                    parent_meta_folder = str(parent_node.get_dataset_metadata().get("source_folder_name") or "")
                    summary["derived_from"] = {
                        "pk": parent_node.pk,
                        "display_name": display_name_for_folder(parent_meta_folder)
                        or parent_node.label
                        or f"Dataset {parent_node.pk}",
                        "source": "recorded",
                    }
                    parent_folder = None
            except Exception:  # noqa: BLE001
                pass
        child_folders = sorted(f for f, (_, p) in _FOLDER_LINEAGE.items() if p == folder)
        summary.setdefault("derived_from", None)
        summary["derived_sets"] = []
        summary["composed_from"] = []
        for src in node.base.extras.get("composed_from", None) or []:
            try:
                from aiida.orm import load_node as _load

                src_node = _load(str(src.get("uuid") or src.get("pk")))
                folder_name = str(src_node.get_dataset_metadata().get("source_folder_name") or "")
                summary["composed_from"].append({
                    "pk": src_node.pk,
                    "display_name": display_name_for_folder(folder_name) or src_node.label or f"Dataset {src_node.pk}",
                })
            except Exception:  # noqa: BLE001
                continue
        if parent_folder or child_folders:
            mapped_parent = pick(parent_folder) if parent_folder else None
            if mapped_parent is not None:
                summary["derived_from"] = {
                    "pk": mapped_parent,
                    "display_name": display_name_for_folder(parent_folder),
                    "source": "name map",
                }
            summary["derived_sets"] = [
                {
                    "pk": pick(f),
                    "display_name": display_name_for_folder(f),
                    "role": _FOLDER_LINEAGE[f][0],
                }
                for f in child_folders
                if pick(f) is not None
            ]
        try:
            family = [
                {"pk": n.pk, "role": (n.get_dataset_metadata().get("dataset_role")), "label": describe_dataset_node(n)}
                for n in dataset_family_nodes(node)
            ]
        except Exception:  # noqa: BLE001
            family = []
        summary["family"] = family
        try:
            folder = source_folder_node_for_dataset(node)
            summary["source_folder_pk"] = folder.pk if folder is not None else None
        except Exception:  # noqa: BLE001
            summary["source_folder_pk"] = None
        return summary

    def dataset_cases(self, pk: int) -> dict[str, Any]:
        """Per-member case rows for one collection: the dataset's case browser."""
        from aiida.orm import QueryBuilder

        from aiida_pino.data import PinoDatasetCollectionData, PinoSimulationData

        node = self._load_node(pk, PinoDatasetCollectionData)
        member_uuids = list(node.base.attributes.get("member_simulation_uuids") or [])
        by_uuid: dict[str, Any] = {}
        if member_uuids:
            q = QueryBuilder().append(PinoSimulationData, filters={"uuid": {"in": member_uuids}}, project=["*"])
            for (sim,) in q.iterall():
                by_uuid[str(sim.uuid)] = sim
        cases: list[dict[str, Any]] = []
        for uuid in member_uuids:  # keep the collection's member order
            sim = by_uuid.get(uuid)
            if sim is None:
                cases.append({"uuid": uuid, "missing": True})
                continue
            meta = sim.get_simulation_metadata()
            shape = meta.get("spatial_shape") or []
            cases.append(
                {
                    "pk": sim.pk,
                    "uuid": uuid,
                    "case_name": _case_name_for_sim(meta, sim),
                    "split": str(sim.base.extras.get("split", "") or "unassigned"),
                    "keywords": _node_keywords(sim),
                    "laser_power": meta.get("laser_power"),
                    "V_scan": meta.get("V_scan"),
                    "beam_radius": meta.get("beam_radius"),
                    "num_timesteps": meta.get("num_timesteps"),
                    "grid": "×".join(str(int(v)) for v in shape) if shape else None,
                    "usable_for_training": sim.base.extras.get("usable_for_training", None),
                    "description": str(sim.base.extras.get("description", "") or ""),
                }
            )
        return {"pk": pk, "count": len(cases), "cases": cases}

    # ---------------------------------------------------------------- models
    def _list_models_via_nodes(self, **kwargs) -> list[dict[str, Any]]:
        """The node-object listing, kept as the REFERENCE the projection is checked against.

        Not used to serve requests -- `list_models` builds rows from a projection instead,
        which is ~6x faster. This stays because the failure mode of the fast path is not a
        crash: a field that silently arrives as None renders as "never evaluated" or a blank
        architecture, which looks like data rather than a bug.
        `tests/test_models_listing_projection.py` compares the two field-for-field against
        the live profile, so the reference has to keep working.
        """
        return self.list_models(_projected_evals=False, **kwargs)

    def list_models(
        self,
        *,
        search: str | None = None,
        evaluated_only: bool = False,
        keywords: list[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
        _projected_evals: bool = True,
    ) -> list[dict[str, Any]]:
        """Model rows for the browser. Evaluations come from ONE projected query.

        `_projected_evals=False` forces the old per-model node path; it exists so
        `_list_models_via_nodes` can serve as the reference the projection is tested
        against, and is not used to answer requests.
        """
        from aiida_pino.utils import recent_model_nodes, summarize_model_node
        from aiida_pino.utils.annotations import normalize_keywords

        from . import folders as folders_mod

        limit = self.settings.clamp_limit(limit)
        wanted_keywords = normalize_keywords(keywords)
        rows: list[dict[str, Any]] = []
        # Filter first, truncate last. `recent_model_nodes(limit)` capped the
        # scan before the predicates ran, so `?limit=20&keyword=rigno` returned
        # 9 of the 50 matching models with nothing saying so.
        filtered = bool(search or evaluated_only or wanted_keywords)
        # Unfiltered, the page is a WINDOW on the fetch, so only its own rows are built.
        # Filtered, every candidate must be examined before it is known which ones the page
        # contains, so the window is applied after the predicates instead.
        offset = max(int(offset or 0), 0)
        if filtered:
            model_nodes = list(recent_model_nodes(None))
            if search:
                # PRE-FILTER on a projection, before the expensive row build. A search
                # returning one model took 17.6 s because every one of the 448 rows was
                # built and only then discarded. The fields the search reads -- label,
                # model_name, training_source_label -- are all projectable, so the
                # population can be narrowed for ~1 s and the loop below runs over the
                # survivors. The predicate is `self._matches`, the SAME one the loop
                # applies, so nothing can pass here and fail there or the reverse.
                try:
                    from aiida.orm import QueryBuilder

                    from aiida_pino.data import PinoModelData as _Model

                    keep: set[int] = set()
                    for pk, label, meta in (
                        QueryBuilder()
                        .append(_Model, project=["id", "label", "attributes.model_metadata"])
                        .all()
                    ):
                        meta = meta if isinstance(meta, dict) else {}
                        name = meta.get("model_name") or meta.get("run_name") or label
                        if self._matches(search, label, name, meta.get("training_source_label")):
                            keep.add(int(pk))
                    model_nodes = [node for node in model_nodes if node.pk in keep]
                except Exception:  # noqa: BLE001 - a failed narrowing must only cost speed
                    pass
        else:
            model_nodes = list(recent_model_nodes((limit or 0) + offset if limit else None))
            if offset:
                model_nodes = model_nodes[offset:]
        with self._memoized():
            folder_of = folders_mod.folder_by_pk([node.pk for node in model_nodes])
            eval_index = self._evaluation_index()
            # One query for every model's evaluations, before the per-model loop turns each
            # into its own filtered query.
            evals_by_uuid = (
                self._evaluations_by_model_uuid(node.uuid for node in model_nodes)
                if _projected_evals
                else {}
            )
            # One query for every training source, before the per-model loop turns each
            # into its own `load_node`.
            self._prefetch_dataset_lookups(
                (node.base.extras.get("training_source_uuid", None)
                 or (node.base.attributes.get("model_metadata") or {}).get("training_source_uuid"))
                for node in model_nodes
            )
            for node in model_nodes:
                try:
                    # No per-model evaluations query and no repository listing: the row
                    # needs a COUNT, and `eval_index` already holds every evaluation.
                    s = summarize_model_node(node, include_evaluations=False, include_files=False)
                    # Once, before the row: the count below and the `evals` field both need
                    # it, and it is the index lookup that replaced the per-model query.
                    eval_rows = self._model_eval_rows(
                        node,
                        eval_index,
                        evals=evals_by_uuid.get(str(node.uuid), []) if _projected_evals else None,
                    )
                    meta = node.base.attributes.get("model_metadata") or {}
                    tp = meta.get("training_profile") or {}
                    row = {
                        "pk": s.get("pk", node.pk),
                        "uuid": node.uuid,
                        "label": s.get("label"),
                        "description": node.description,
                        "keywords": _node_keywords(node),
                        "model_name": s.get("model_name"),
                        "config_name": s.get("config_name"),
                        "architecture": _model_architecture({**s, "run_name": meta.get("run_name")}),
                        # `max_steps` is the ceiling requested; the trainer's
                        # stopping criterion stopped 31 of 158 runs short of it,
                        # worst 4000 of 48000, and the column said 48000.
                        # Read through `effective_max_steps`: a `training.max_steps=`
                        # entry in `hydra_overrides` is emitted after the structured
                        # field and wins, so the field alone is not what ran.
                        "max_steps": effective_max_steps(tp),
                        "steps_trained": node.base.extras.get("steps_trained", None),
                        "evaluation_count": (
                            s.get("evaluation_count")
                            if s.get("evaluation_count") is not None
                            else len(eval_rows)
                        ),
                        "training_source_kind": s.get("training_source_kind"),
                        "training_source_label": s.get("training_source_label"),
                        "training_source_display": self._dataset_display_by_uuid(
                            node.base.extras.get("training_source_uuid", None) or meta.get("training_source_uuid")
                        ) or s.get("training_source_label"),
                        # If that dataset was later merged into another, name the
                        # survivor. The link above still points at the node this
                        # model actually consumed -- that is a fact and stays -- but
                        # two models trained on the same data under two node ids
                        # should not look like they trained on different data.
                        "training_source_canonical": self.canonical_dataset(
                            node.base.extras.get("training_source_uuid", None) or meta.get("training_source_uuid")
                        ),
                        # The resolution the model trained at. An error measured
                        # on a 2x2x1 grid is not comparable with one measured on
                        # the full grid, and 25 of the 35 models with a recorded
                        # stride are NOT full resolution -- including the current
                        # best-T-MAE holder, whose page printed [1,1,1].
                        "training_stride": node.base.extras.get("training_stride", None),
                        "full_resolution": node.base.extras.get("full_resolution", None),
                        # Fetched once and reused below: this call used to run
                        # twice per model, and the first one dropped the eval_index
                        # built for it, re-running model_evaluations_for_model and
                        # the Kelvin lookup for every row (review criticism D).
                        "evals": eval_rows,
                        # Explicit folders only: an auto-folder would be collapsed by
                        # default and would hide starred models inside it.
                        "folder": folder_of.get(node.pk),
                        "auto_keywords": sorted({b for ev in eval_rows for b in ev["badges"]}),
                        "ctime": node.ctime.isoformat() if node.ctime else None,
                    }
                except Exception:  # noqa: BLE001
                    continue
                if evaluated_only and not row["evaluation_count"]:
                    continue
                if wanted_keywords and not set(wanted_keywords) <= set(row["keywords"]):
                    continue
                if search and not self._matches(search, row["label"], row["model_name"], row["training_source_label"]):
                    continue
                rows.append(row)
        total = len(rows)
        if not filtered:
            # Unfiltered, the scan was already capped by `recent_model_nodes(limit)`,
            # so `len(rows)` is the page, not the population: `?limit=100` reported
            # `total: 100, truncated: false` with 158 models stored, and both model
            # pickers silently offered the newest 100 -- including one whose absence
            # the eval form's own placeholder text pointed at.
            from aiida.orm import QueryBuilder

            from aiida_pino.data import PinoModelData

            total = QueryBuilder().append(PinoModelData).count()
        if filtered and offset:
            rows = rows[offset:]
        if limit and len(rows) > limit:
            rows = rows[:limit]
        for row in rows:
            # `_truncated` means "more exist than are being returned", which with a window
            # includes everything BEFORE it as well as after -- otherwise the last page of a
            # paginated listing would claim to be the whole set.
            row["_total"], row["_truncated"] = total, total > (offset + len(rows))
        return rows

    @contextmanager
    def _memoized(self):
        """Memoise node lookups for the life of one request, then forget them.

        The models listing resolved the same handful of dataset uuids 490 times
        — once per model, then again per evaluation row — for 12.6 s of a 50 s
        request, and followed the same supersession chains 174 times on top of
        that. Scoping the memo to one request keeps every number as fresh as it
        was before (a rename or a merge shows on the next reload) and pays for
        each distinct uuid once.
        """
        outer = getattr(self._local, "memo", None)
        self._local.memo = {} if outer is None else outer
        try:
            yield
        finally:
            self._local.memo = outer

    def _memo(self, bucket: str, key: Any, compute):
        store = getattr(self._local, "memo", None)
        if store is None:
            return compute()
        cache = store.setdefault(bucket, {})
        if key not in cache:
            cache[key] = compute()
        return cache[key]

    def _prefetch_dataset_lookups(self, uuids) -> None:
        """Fill the display/canonical memo caches for many datasets in ONE query.

        `_dataset_display_uncached` and `_canonical_dataset_uncached` each call
        `load_node(uuid)`, so a models listing did 273 individual node loads -- ~5 s of a
        10.5 s response, and the console runs a SINGLE uvicorn worker, so that request also
        blocked the 5 s monitor poll behind it. The models tab was still showing "Loading..."
        when the headless render check gave up, which is what made its two interaction
        checks fail.

        Everything both resolvers read is projectable: the label, one attribute and one
        extra. This prefills the SAME memo buckets they consult, so the per-model calls stay
        exactly as they were and simply hit warm cache -- no second implementation of the
        lookup, and no behaviour to keep in sync.

        A dataset that HAS been superseded is deliberately left uncached: the canonical
        lookup follows a chain, and only the common no-replacement case is answered here.
        """
        store = getattr(self._local, "memo", None)
        if store is None:
            return
        wanted = {str(u) for u in uuids if u}
        if not wanted:
            return
        display_cache = store.setdefault("display", {})
        canonical_cache = store.setdefault("canonical", {})
        missing = [u for u in wanted if u not in display_cache or u not in canonical_cache]
        if not missing:
            return
        try:
            from aiida.orm import QueryBuilder

            from aiida_pino.data import PinoDatasetCollectionData

            rows = (
                QueryBuilder()
                .append(
                    PinoDatasetCollectionData,
                    filters={"uuid": {"in": missing}},
                    project=["uuid", "label", "attributes.dataset_metadata", "extras.replaced_by_uuid"],
                )
                .all()
            )
        except Exception:  # noqa: BLE001 - a failed prefetch must only cost speed
            return
        for uuid, label, metadata, replaced_by in rows:
            key = str(uuid)
            if key not in display_cache:
                folder = (metadata or {}).get("source_folder_name") if isinstance(metadata, dict) else None
                display_cache[key] = display_name_for_folder(folder) or label
            if key not in canonical_cache and not replaced_by:
                canonical_cache[key] = None

    def _dataset_display_by_uuid(self, uuid: str | None) -> str | None:
        if not uuid:
            return None
        return self._memo("display", str(uuid), lambda: self._dataset_display_uncached(uuid))

    def _dataset_display_uncached(self, uuid: str | None) -> str | None:
        try:
            from aiida.orm import load_node

            node = load_node(str(uuid))
            meta = node.get_dataset_metadata()
            return display_name_for_folder(meta.get("source_folder_name")) or node.label
        except Exception:  # noqa: BLE001
            return None

    def canonical_dataset(self, uuid: str | None) -> dict[str, Any] | None:
        """Follow `replaced_by_uuid` to the node that now represents this data.

        Merging two duplicate datasets cannot mean re-pointing the models that
        trained on them: an AiiDA input link is immutable, and rewriting it
        would say a model consumed a node it never saw. The link stays truthful
        and the *reading* resolves — so a model trained on a retired twin is
        shown against the surviving dataset, with the node it actually used
        still named.

        Returns None when the dataset was never superseded. Chain-safe and
        cycle-safe: a supersession chain is followed to its end, and a loop
        stops rather than hangs.
        """
        return self._memo("canonical", str(uuid or ""), lambda: self._canonical_dataset_uncached(uuid))

    def _canonical_dataset_uncached(self, uuid: str | None) -> dict[str, Any] | None:
        from aiida.orm import load_node

        seen: set[str] = set()
        current = str(uuid or "")
        hops = 0
        while current and current not in seen:
            seen.add(current)
            try:
                node = load_node(current)
            except Exception:  # noqa: BLE001
                return None
            replacement = node.base.extras.get("replaced_by_uuid", None)
            if not replacement or str(replacement) in seen:
                if hops == 0:
                    return None
                return {
                    "pk": node.pk,
                    "uuid": str(node.uuid),
                    "display": self._dataset_display_by_uuid(str(node.uuid)),
                    "hops": hops,
                }
            current = str(replacement)
            hops += 1
        return None

    # ------------------------------------------------- model metrics / badges
    def _model_hydra_config(self, model) -> dict[str, Any] | None:
        """The model's archived `.hydra/config.yaml`, or None when absent.

        The *fallback* record of `ref_scales.T`, the constant that turns a
        normalised temperature error into Kelvin: evaluations now store the
        factor themselves, and this covers the ones made before they did.
        """
        from . import model_metrics

        return model_metrics.model_hydra_config(model)

    def _kelvin_factor(self, model) -> tuple[float | None, str]:
        from . import model_metrics

        return model_metrics.kelvin_factor(self._model_hydra_config(model))

    def _league_label(self, uuid: str | None, fallback: str | None) -> str:
        return self._memo("league", (str(uuid or ""), str(fallback or "")), lambda: self._league_label_uncached(uuid, fallback))

    def _league_label_uncached(self, uuid: str | None, fallback: str | None) -> str:
        """A dataset name that identifies exactly one dataset.

        Two live datasets can share a display name while holding different data
        (a strided view of the same cases), and a leaderboard grouped on one and
        labelled with the other awards two "best" badges for what reads as one
        dataset.
        """
        base = self._dataset_display_by_uuid(uuid) or fallback or "dataset"
        return f"{base}{self._league_suffix(uuid)}"

    def _league_suffix(self, uuid: str | None) -> str:
        """`' (#24397, stride 2x2x1)'` — what makes a shared display name injective.

        Separate from `_league_label` because the evaluation rows build their
        display name differently (they append the split), and the two must end up
        naming the same league or a badge cannot be matched to the row it came
        from.
        """
        return self._memo("league_suffix", str(uuid or ""), lambda: self._league_suffix_uncached(uuid))

    def _league_suffix_uncached(self, uuid: str | None) -> str:
        from aiida.orm import load_node

        try:
            node = load_node(str(uuid))
        except Exception:  # noqa: BLE001
            return ""
        bits = [f"#{node.pk}"]
        stride = ((node.get_dataset_metadata() or {}).get("derivation_profile") or {}).get("stride")
        if stride and list(stride) != [1, 1, 1]:
            bits.append(f"stride {'x'.join(str(s) for s in stride)}")
        return f" ({', '.join(bits)})"

    #: Seconds an evaluation index may be reused across requests. Short on purpose --
    #: see `_evaluation_index`.
    EVAL_INDEX_TTL_S = 15.0

    def _evaluation_index(self) -> dict[int, dict[str, Any]]:
        """Badges and ranks for every evaluation, rebuilt at most every few seconds.

        Nothing is PERSISTED: a model that gets beaten loses its badge, and no machine
        string can ever land in the user's keywords. That was the point of recomputing it
        per request, and it still holds -- a 15 s window cannot make a badge outlive the
        eval that beat it by more than a page refresh.

        Why it needs a window at all: measured at **2.95 s for 564 evaluations**, and it ran
        on EVERY models request. The console is one uvicorn worker, so those seconds also
        sat in front of the 5 s monitor poll queued behind them -- the models tab was still
        rendering "Loading..." when the headless render check timed out, which is what made
        its two interaction checks fail.
        """
        import time as _time

        cached = getattr(self, "_eval_index_cache", None)
        if cached is not None:
            built_at, value = cached
            if _time.monotonic() - built_at < self.EVAL_INDEX_TTL_S:
                return value
        value = self._evaluation_index_uncached()
        self._eval_index_cache = (_time.monotonic(), value)
        return value

    def _evaluation_index_uncached(self) -> dict[int, dict[str, Any]]:
        """Build the index. See `_evaluation_index` for why this is not called directly."""
        from aiida.orm import QueryBuilder

        from aiida_pino.data import PinoModelData, PinoModelEvaluationData

        from . import model_metrics

        factor_by_uuid: dict[str, float | None] = {}
        canonical_by_uuid: dict[str, str] = {}
        for uuid, in QueryBuilder().append(PinoModelData, project=["uuid"]).all():
            factor_by_uuid[str(uuid)] = None
        rows: list[dict[str, Any]] = []
        builder = QueryBuilder().append(
            PinoModelEvaluationData,
            project=[
                "id",
                "extras.model_uuid",
                "attributes.evaluation_metadata.dataset_source_uuid",
                "attributes.evaluation_metadata.dataset_source_label",
                "attributes.evaluation_metadata.metrics.T.abs_err_mean",
                "attributes.evaluation_metadata.metrics.T.rmse",
                "attributes.evaluation_metadata.metric_units",
                "extras.metric_units",
                "extras.held_out",
                "extras.train_overlap_n",
            ],
        )
        for pk, model_uuid, ds_uuid, ds_label, mae, rmse, units, units_extra, held_out, overlap in builder.all():
            # The evaluation's own unit record wins; the model's archived config
            # is the fallback for evaluations made before units were stored.
            factor, _ = model_metrics.recorded_kelvin_factor(
                {"metric_units": units}, {"metric_units": units_extra}
            )
            if factor is None:
                factor = self._factor_for_model_uuid(model_uuid, factor_by_uuid)
            # League on the *canonical* dataset. Grouping on the raw uuid gave a
            # model evaluated on a superseded twin its own league of one and an
            # automatic first place: three nodes hold the identical 53-case
            # random-path test set, and a superseded duplicate badged
            # 13.436 K "best" while the real best on that same set is 13.138 K.
            canonical = canonical_by_uuid.get(str(ds_uuid or ""))
            if canonical is None:
                resolved = self.canonical_dataset(ds_uuid)
                canonical = (resolved or {}).get("uuid") or ds_uuid
                canonical_by_uuid[str(ds_uuid or "")] = canonical
            rows.append({
                "eval_pk": int(pk),
                "model_uuid": model_uuid,
                "dataset_key": canonical or ds_label,
                # The display name is not injective, and two datasets sharing it
                # are not necessarily the same data: 12968 and 24397 hold the
                # identical 53 cases but 24397 is `stride [2, 2, 1]`, i.e. half
                # resolution. Badging both "best T MAE (Variable length random
                # paths test split)" compares a downsampled score against a
                # full-resolution one under one name, so the label carries the
                # pk and the stride whenever the derivation is not the default.
                "dataset_display": self._league_label(canonical or ds_uuid, ds_label),
                # A score on the model's own training data must never rank or
                # badge: strip the Kelvin numbers from the ranking row but keep
                # the row itself, so the render still shows the warning.
                "t_mae_k": (mae * factor) if (factor and isinstance(mae, (int, float)) and held_out is not False) else None,
                "t_rmse_k": (rmse * factor) if (factor and isinstance(rmse, (int, float)) and held_out is not False) else None,
                "held_out": held_out,
                "train_overlap_n": overlap,
            })
        return model_metrics.rank_evaluations(rows)

    def _factor_for_model_uuid(self, model_uuid, cache: dict[str, float | None]) -> float | None:
        if not model_uuid:
            return None
        key = str(model_uuid)
        if key in cache and cache[key] is not None:
            return cache[key]
        try:
            from aiida.orm import load_node

            factor, _ = self._kelvin_factor(load_node(key))
        except Exception:  # noqa: BLE001
            factor = None
        cache[key] = factor
        return factor

    def _evaluations_by_model_uuid(self, uuids) -> dict[str, list]:
        """Every model's evaluations in ONE query, keyed by model uuid.

        `model_evaluations_for_model` filters on `extras.model_uuid` per model, so a
        100-model listing ran 100 queries for 564 evaluation records -- measured 2.68 s of a
        10.9 s response. The filter takes an `in` just as happily.

        Returns the same node objects the per-model helper returns, in the same ctime-desc
        order, so `_model_eval_rows` cannot tell the difference. A model with no evaluations
        maps to an empty list rather than being absent, so callers need no special case.
        """
        wanted = [str(u) for u in uuids if u]
        if not wanted:
            return {}
        grouped: dict[str, list] = {u: [] for u in wanted}
        try:
            from aiida.orm import QueryBuilder

            from aiida_pino.data import PinoModelEvaluationData

            builder = QueryBuilder().append(
                PinoModelEvaluationData,
                filters={"extras.model_uuid": {"in": wanted}},
                project=[
                    "id",
                    "label",
                    "ctime",
                    "attributes.evaluation_metadata",
                    "extras",
                    "extras.model_uuid",
                ],
            )
            builder.order_by({PinoModelEvaluationData: {"ctime": "desc"}})
            rows = builder.all()
        except Exception:  # noqa: BLE001 - a failed prefetch must only cost speed
            return {}
        for pk, label, ctime, metadata, extras, model_uuid in rows:
            grouped.setdefault(str(model_uuid), []).append(
                _ProjectedEval(pk, label, ctime, metadata, extras)
            )
        return grouped

    def _model_eval_rows(self, model, index: dict[int, dict] | None = None, evals=None) -> list[dict[str, Any]]:
        from aiida_pino.utils import model_evaluations_for_model

        rows = []
        try:
            evals = model_evaluations_for_model(model) if evals is None else evals
        except Exception:  # noqa: BLE001
            return rows
        from . import model_metrics

        # Read only when an evaluation has no unit record of its own. Reading
        # it up front cost 9.2 s of a 50 s listing for a fallback almost every
        # row now declines.
        model_factor: float | None = None
        factor_read = False
        index = index if index is not None else {}
        for ev in evals:
            em = ev.base.attributes.get("evaluation_metadata") or {}
            # Prefer the unit the evaluation recorded for itself.
            factor, _source = model_metrics.recorded_kelvin_factor(em, ev.base.extras.all)
            if factor is None:
                if not factor_read:
                    model_factor, _ = self._kelvin_factor(model)
                    factor_read = True
                factor = model_factor
            flat = _flatten_metrics(em.get("metrics") or {})
            ranked = index.get(int(ev.pk)) or {}
            ds_uuid = em.get("dataset_source_uuid")
            rows.append(
                {
                    "pk": ev.pk,
                    "name": ev.label or em.get("evaluation_name"),
                    "dataset_label": em.get("dataset_source_label"),
                    # The badge path already qualified its label; this row, the
                    # line a reader actually looks at, did not. 14 evaluations —
                    # 9% of the record — print "Variable length random paths test
                    # split" while sitting on three different nodes, one of them
                    # `stride [2, 2, 1]`. Downsampling smooths the field, so the
                    # half-resolution scores are systematically lower and a reader
                    # ranking by the printed Kelvin gets the wrong order. Same
                    # suffix as the badge, so a badge can be matched to its row.
                    "dataset_display": self._qualified_eval_dataset(ev, em, ds_uuid),
                    "dataset_pk": self._pk_for_uuid(ds_uuid),
                    "ctime": ev.ctime.isoformat() if ev.ctime else None,
                    "metrics": flat,
                    "kelvin": model_metrics.to_kelvin(flat, factor),
                    "iou_pct": model_metrics.iou_error_pct(flat),
                    "geometry": _corrected_geometry(
                        model_metrics.geometry_rows(flat),
                        ev.base.extras.get("geometry_scale", None),
                    ),
                    "geometry_scale": ev.base.extras.get("geometry_scale", None),
                    "badges": ranked.get("badges") or [],
                    "ranks": ranked.get("ranks") or {},
                    # Present on every row so the front end never has to guess:
                    # None = unknown provenance, True = held out, False = the
                    # model was trained on some of the scored cases.
                    "held_out": ev.base.extras.get("held_out", None),
                    "train_overlap_n": ev.base.extras.get("train_overlap_n", None),
                    "warning": ranked.get("warning"),
                }
            )
        return rows

    def _qualified_eval_dataset(self, ev, em: dict, ds_uuid: str | None) -> str | None:
        """The name on an evaluation row, naming exactly one dataset."""
        base = _split_qualified(
            self._dataset_display_by_uuid(ds_uuid) or em.get("dataset_source_label"),
            # The split is named in the *dataset* label the evaluation recorded,
            # not in the evaluation's own name.
            f"{em.get('dataset_source_label') or ''} {ev.label or em.get('evaluation_name') or ''}",
        )
        return f"{base}{self._league_suffix(ds_uuid)}" if base else base

    def _pk_for_uuid(self, uuid: str | None) -> int | None:
        if not uuid:
            return None
        return self._memo("pk", str(uuid), lambda: self._pk_for_uuid_uncached(uuid))

    def _pk_for_uuid_uncached(self, uuid: str | None) -> int | None:
        try:
            from aiida.orm import load_node

            return int(load_node(str(uuid)).pk)
        except Exception:  # noqa: BLE001
            return None

    def _model_curve_availability(self, model) -> dict[str, Any]:
        """Which curves can be drawn for this model, and why not when they cannot.

        The model node keeps only checkpoints and the Hydra config; the tfevents
        file and the job stdout live on the training-artifacts folder and the
        calcjob behind it, so a model imported without them simply has no
        curves — the pane says so rather than drawing an empty chart.
        """
        out: dict[str, Any] = {"validation": False, "loss": False, "reason": None}
        folder = self._training_artifacts_folder(model)
        if folder is None:
            out["reason"] = "no training artifacts linked to this model"
            return out
        try:
            names = folder.base.repository.list_object_names()
        except Exception:  # noqa: BLE001
            out["reason"] = "training artifacts unreadable"
            return out
        out["validation"] = any(name.startswith("events.out.tfevents") for name in names)
        out["loss"] = self._training_stdout(model) is not None
        if not out["validation"] and not out["loss"]:
            out["reason"] = "no tfevents file and no job output recorded"
        return out

    def _training_artifacts_folder(self, model):
        try:
            return model.creator.inputs.training_artifacts
        except Exception:  # noqa: BLE001
            return None

    def _training_stdout(self, model) -> str | None:
        """The training job stdout, which carries the dense `[step: N] loss:` series."""
        folder = self._training_artifacts_folder(model)
        if folder is None:
            return None
        try:
            retrieved = folder.creator.outputs.retrieved
            for name in retrieved.base.repository.list_object_names():
                if name.endswith(".out"):
                    return retrieved.base.repository.get_object_content(name)
        except Exception:  # noqa: BLE001
            return None
        return None

    def model_curves(self, model) -> dict[str, Any]:
        """Parsed training curves for a model, cached on the node.

        Parsing a tfevents file costs a couple of seconds, so the result is
        cached in an extra keyed by the artifacts node — a re-run that produces
        new artifacts invalidates it, nothing else has to remember.
        """
        import tempfile
        from pathlib import Path

        from . import viz

        folder = self._training_artifacts_folder(model)
        if folder is None:
            return {"tags": {}, "loss_stdout": []}
        cached = model.base.extras.get(CURVES_EXTRA, None)
        if isinstance(cached, dict) and cached.get("artifacts_uuid") == str(folder.uuid) \
                and cached.get("version") == CURVES_VERSION:
            return cached
        tags: dict[str, list] = {}
        try:
            names = [n for n in folder.base.repository.list_object_names() if n.startswith("events.out.tfevents")]
        except Exception:  # noqa: BLE001
            names = []
        if names:
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / names[0]
                target.write_bytes(folder.base.repository.get_object_content(names[0], mode="rb"))
                try:
                    tags = viz.parse_tfevents(tmp)
                except ImportError:
                    tags = {}     # tensorboard missing: the loss panel still works
                except Exception:  # noqa: BLE001 - a corrupt event file is not fatal
                    tags = {}
        stdout = self._training_stdout(model)
        curves = {
            "version": CURVES_VERSION,
            "artifacts_uuid": str(folder.uuid),
            "tags": tags,
            "loss_stdout": viz.parse_stdout_loss(stdout) if stdout else [],
        }
        try:
            model.base.extras.set(CURVES_EXTRA, curves)
        except Exception:  # noqa: BLE001 - a read-only node is still renderable
            pass
        return curves

    def model_curves_plot(self, pk: int, *, kind: str = "validation") -> bytes:
        from aiida_pino.data import PinoModelData

        from . import viz

        if kind not in ("validation", "loss"):
            # It used to fall through to the validation plot, byte-identical, so
            # a typo returned a real-looking chart of something else. The
            # neighbouring preview endpoint 422s on an unknown field.
            raise InvalidRequestError(f"kind must be 'validation' or 'loss' (got {kind!r})")
        node = self._load_node(pk, PinoModelData)
        curves = self.model_curves(node)
        if not curves.get("tags") and not curves.get("loss_stdout"):
            raise InvalidRequestError(f"no training curves recorded for model {pk}")
        factor, _ = self._kelvin_factor(node)
        return viz.render_model_curves_png(curves, kind=kind, t_factor=factor)

    def evaluation_preview_arrays(self, pk: int):
        """The stored ground-truth/prediction preview for one evaluation."""
        import io as _io

        import numpy as np

        node = self._load_node(pk)
        folder = None
        try:
            folder = node.creator.caller.outputs.eval_artifacts
        except Exception:  # noqa: BLE001
            folder = None
        if folder is None:
            raise InvalidRequestError(f"evaluation {pk} stored no preview arrays")
        names = [n for n in folder.base.repository.list_object_names() if n.endswith(".npz")]
        if not names:
            raise InvalidRequestError(f"evaluation {pk} stored no preview arrays")
        raw = folder.base.repository.get_object_content(names[0], mode="rb")
        return dict(np.load(_io.BytesIO(raw)))

    def evaluation_preview(self, pk: int) -> dict[str, Any]:
        arrays = self.evaluation_preview_arrays(pk)
        fields = sorted({k[5:] for k in arrays if k.startswith("true_") and f"pred_{k[5:]}" in arrays})
        shape = list(arrays[f"true_{fields[0]}"].shape[-3:]) if fields else []
        # `"<uuid>" and 0` evaluates to 0, so this was `None` for every
        # evaluation ever previewed. The helper below resolves it properly.
        model_pk = self._pk_for_uuid(self._load_node(pk).base.extras.get("model_uuid", None))
        return {
            "pk": pk,
            "fields": fields,
            "shape": shape,
            "nz": shape[-1] if shape else 0,
            "z_default": (shape[-1] // 2) if shape else 0,
            "samples": 1,
            "model_pk": model_pk,
        }

    def evaluation_preview_plot(self, pk: int, *, field: str = "T", z: int = -1) -> bytes:
        from . import viz

        arrays = self.evaluation_preview_arrays(pk)
        if f"true_{field}" not in arrays:
            raise InvalidRequestError(f"evaluation {pk} has no stored {field!r} field")
        factor = None
        try:
            model_uuid = self._load_node(pk).base.extras.get("model_uuid", None)
            if model_uuid:
                from aiida.orm import load_node

                factor, _ = self._kelvin_factor(load_node(str(model_uuid)))
        except Exception:  # noqa: BLE001
            factor = None
        return viz.render_eval_preview_png(arrays, field=field, z=z, t_factor=factor)

    def compare_models(self, pk_a: int, pk_b: int) -> dict[str, Any]:
        """The ablation referee: every difference between two models, classified.

        Confounds this database has already paid for, checked automatically:
        seed (~1700 K swing across replicates), steps actually trained (31 of
        158 models under their declared budget), resolution (18 models at 2x2x1
        under a full-resolution record), training-data overlap, and evals scored
        on training data. See utils/model_compare.py.
        """
        from aiida_pino.data import PinoModelData
        from aiida_pino.utils.model_compare import compare_models

        node_a = self._load_node(pk_a, PinoModelData)
        node_b = self._load_node(pk_b, PinoModelData)
        return compare_models(node_a, node_b)

    def models_trained_on(self, pk: int) -> dict[str, Any]:
        """Everything that trained on a dataset -- models, runs, and evals.

        "Has anything trained on this dataset" was a hand-written join
        (2026-08-10): the actual train input is usually a SPLIT or
        layout variant reached only through the `base_collection_uuid` /
        `parent_collection_uuid` attributes -- splits mint their own
        family_identifier and there is no graph edge from the parent, so
        provenance traversal from the base finds nothing. This closes over both
        pointer attributes in both directions, then reports:

        - models whose training_source_uuid / train_dataset_uuid is in the
          closure (the cheap-extras answer),
        - training PROCESSES whose train input is in the closure -- catching
          runs still in flight and runs whose model never materialized, which
          extras can never see,
        - evaluations scored on the closure, listed separately so an eval named
          like a model (e4_*) cannot be mistaken for a trained control.
        """
        from aiida import orm
        from aiida.orm import QueryBuilder

        from aiida_pino.data import PinoDatasetCollectionData, PinoModelData

        node = self._load_node(pk, PinoDatasetCollectionData)
        # Parent-pointer table for the whole population: one projected query,
        # closure in python (1,700 rows; the loop is trivial next to a per-node
        # QueryBuilder round trip).
        rows = QueryBuilder().append(
            PinoDatasetCollectionData,
            project=["uuid", "attributes.base_collection_uuid",
                     "attributes.parent_collection_uuid", "id", "label"],
        ).all()
        parents: dict[str, set[str]] = {}
        info: dict[str, tuple[int, str]] = {}
        for uuid, base, parent, node_pk, label in rows:
            info[uuid] = (node_pk, label)
            parents[uuid] = {p for p in (base, parent) if p}
        closure = {node.uuid}
        # up: the base chain of the asked-for node
        frontier = set(parents.get(node.uuid) or [])
        while frontier - closure:
            closure |= frontier
            frontier = {g for u in frontier for g in (parents.get(u) or set())}
        # down: everything that points into the closure, to a fixpoint
        changed = True
        while changed:
            changed = False
            for uuid, ups in parents.items():
                if uuid not in closure and ups & closure:
                    closure.add(uuid)
                    changed = True

        members = sorted(
            ({"pk": info[u][0], "uuid": u, "label": info[u][1]} for u in closure if u in info),
            key=lambda r: r["pk"],
        )
        model_rows = [
            {"pk": mpk, "model_name": name, "trained_on_uuid": src or tds}
            for mpk, name, src, tds in QueryBuilder().append(
                PinoModelData,
                project=["id", "extras.model_name", "extras.training_source_uuid",
                         "extras.train_dataset_uuid"],
            ).all()
            if src in closure or tds in closure
        ]
        procs = QueryBuilder().append(
            orm.Data, filters={"uuid": {"in": sorted(closure)}}, tag="d",
        ).append(
            orm.ProcessNode, with_incoming="d",
            edge_filters={"label": {"like": "train%"}},
            project=["id", "attributes.process_label", "attributes.process_state",
                     "attributes.exit_status"],
        ).all()
        # A TrainEval chain hands the same dataset to its nested TrainModel and
        # calcjob, so the raw edge query returns each run up to three times.
        # Keep only processes not called by another process in the set.
        proc_pks = [ppk for ppk, *_ in procs]
        called: set[int] = set()
        if proc_pks:
            for child, caller in QueryBuilder().append(
                orm.ProcessNode, filters={"id": {"in": proc_pks}}, tag="c", project=["id"],
            ).append(
                orm.ProcessNode, with_outgoing="c", project=["id"],
            ).all():
                if caller in proc_pks:
                    called.add(child)
        training_rows = [
            {"pk": ppk, "process": plabel, "state": state, "exit_status": exit_status}
            for ppk, plabel, state, exit_status in procs
            if ppk not in called
        ]
        eval_rows = [
            # bool(), not the raw projection: the sqlite backend hands JSON
            # booleans back as 0/1, and the consumers compare strictly.
            {"pk": epk, "label": label, "held_out": bool(held) if held is not None else None}
            for epk, label, held in QueryBuilder().append(
                orm.Data,
                filters={"node_type": {"like": "%pino_model_evaluation%"},
                         "extras.dataset_source_uuid": {"in": sorted(closure)}},
                project=["id", "label", "extras.held_out"],
            ).all()
        ]
        return {
            "dataset": {"pk": node.pk, "uuid": node.uuid, "label": node.label},
            "closure": members,
            "models": sorted(model_rows, key=lambda r: r["pk"]),
            "trainings": sorted(training_rows, key=lambda r: r["pk"]),
            "evaluations_scored_on_it": sorted(eval_rows, key=lambda r: r["pk"]),
            "note": "trainings includes in-flight runs and runs whose model never "
                    "materialized; evaluations are NOT models, however model-like "
                    "their labels look.",
        }

    def model_detail(self, pk: int) -> dict[str, Any]:
        from aiida_pino.data import PinoModelData
        from aiida_pino.utils import summarize_model_node

        node = self._load_node(pk, PinoModelData)
        with self._memoized():
            summary = summarize_model_node(node)
            meta = node.base.attributes.get("model_metadata") or {}
            summary["architecture"] = _model_architecture({**summary, "run_name": meta.get("run_name")})
            summary["description"] = node.description
            summary["keywords"] = _node_keywords(node)
            summary["superseded"] = bool(node.base.extras.get("superseded", False))
            summary["replaced_by_pk"] = node.base.extras.get("replaced_by_pk", None)
            summary["replaced_by_uuid"] = node.base.extras.get("replaced_by_uuid", None)
            from . import folders as folders_mod

            summary["folder"] = folders_mod.folder_by_pk([node.pk]).get(node.pk)
            summary["training_source_display"] = (
                self._dataset_display_by_uuid(
                    node.base.extras.get("training_source_uuid", None) or meta.get("training_source_uuid")
                )
                or summary.get("training_source_label")
            )
            # The listing computes this on every row and rendered it nowhere; the
            # pane did not carry it at all. A model trained on a dataset that was
            # later merged should name the survivor as well as the node it read.
            summary["training_stride"] = node.base.extras.get("training_stride", None)
            summary["full_resolution"] = node.base.extras.get("full_resolution", None)
            summary["training_stride_source"] = node.base.extras.get("training_stride_source", None)
            summary["training_source_canonical"] = self.canonical_dataset(
                node.base.extras.get("training_source_uuid", None) or meta.get("training_source_uuid")
            )
            factor, provenance = self._kelvin_factor(node)
            summary["t_ref_k"] = factor
            summary["t_ref_source"] = provenance
            summary["curves"] = self._model_curve_availability(node)
            index = self._evaluation_index()
            # `evaluations` (raw, carrying /tmp scratch paths) and `eval_rows`
            # (processed, a strict superset) were both shipped, roughly doubling
            # an already large payload. The raw list stays in
            # `summarize_model_node` for the widget; the console sends one.
            summary.pop("evaluations", None)
            summary["eval_rows"] = self._model_eval_rows(node, index)
            summary["auto_keywords"] = [b for row in summary["eval_rows"] for b in row["badges"]]
            # First line of the CLI dict render, before any metric: a
            # training-fit number must not survive a glance as a held-out one.
            tainted = [r for r in summary["eval_rows"] if r.get("held_out") is False]
            if tainted:
                summary = {
                    "WARNING": "; ".join(
                        f"eval {r['pk']} scored {r['train_overlap_n']} training cases"
                        if isinstance(r.get("train_overlap_n"), int)
                        else f"eval {r['pk']} scored training data"
                        for r in tainted
                    ) + " -- training-fit, not held-out",
                    **summary,
                }
            return summary

    # ------------------------------------------------------------ kfold runs
    def list_kfold_runs(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        from aiida_pino.utils import describe_kfold_run_node, recent_kfold_run_nodes

        limit = self.settings.clamp_limit(limit)
        rows: list[dict[str, Any]] = []
        for node in recent_kfold_run_nodes(limit):
            try:
                rows.append(
                    {
                        "pk": node.pk,
                        "uuid": node.uuid,
                        "label": describe_kfold_run_node(node),
                        "ctime": node.ctime.isoformat() if node.ctime else None,
                    }
                )
            except Exception:  # noqa: BLE001
                continue
        return rows

    def kfold_detail(self, pk: int) -> dict[str, Any]:
        from aiida_pino.data import PinoKfoldRunData
        from aiida_pino.utils import summarize_kfold_run_node

        return summarize_kfold_run_node(self._load_node(pk, PinoKfoldRunData))

    # ----------------------------------------------------------- simulations
    def list_simulations(
        self,
        *,
        source: str | None = None,
        folder: str | None = None,
        fields: Iterable[str] | None = None,
        split: str | None = None,
        keywords: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        from aiida_pino.utils import query_simulation_nodes
        from aiida_pino.utils.annotations import normalize_keywords

        limit = self.settings.clamp_limit(limit)
        rows: list[dict[str, Any]] = []
        nodes = query_simulation_nodes(
            source_type=(source or None) if source not in (None, "all") else None,
            source_folder_contains=folder or None,
            required_fields=list(fields) if fields else None,
            split=split or None,
            keywords=normalize_keywords(keywords) or None,
            limit=limit,
        )
        for node in nodes:
            try:
                meta = node.get_simulation_metadata()
                rows.append(
                    {
                        "pk": node.pk,
                        "uuid": node.uuid,
                        "label": meta.get("simulation_identifier") or node.uuid,
                        "case_name": _case_name_for_sim(meta, node),
                        "source_type": meta.get("source_type"),
                        "source_folder_name": meta.get("source_folder_name"),
                        "source_display": display_name_for_folder(meta.get("source_folder_name")),
                        "V_scan": meta.get("V_scan"),
                        "laser_power": meta.get("laser_power"),
                        "beam_radius": meta.get("beam_radius"),
                        "num_timesteps": meta.get("num_timesteps"),
                        "requested_fields": meta.get("requested_fields"),
                        "usable_for_training": node.base.extras.get("usable_for_training", None),
                        "batch_keyword": node.base.extras.get("batch_keyword", None),
                        "split": str(node.base.extras.get("split", "") or "unassigned"),
                        "keywords": _node_keywords(node),
                    }
                )
            except Exception:  # noqa: BLE001
                continue
        return rows

    # ------------------------------------------------- per-case visualization
    def simulation_frames(self, pk: int) -> dict[str, Any]:
        from aiida_pino.data import PinoSimulationData

        from . import viz

        sim = self._load_node(pk, PinoSimulationData)
        info = viz.frame_info(sim)
        aliases = sim.base.attributes.get("array_aliases") or {}
        info["fields"] = [k for k in aliases if k not in ("x", "y", "z", "P", "V_scan", "time_s")]
        info["pk"] = pk
        return info

    def simulation_plot(
        self, pk: int, *, field: str, view: str = "long", frame: int = -1, contour: bool = True
    ) -> bytes:
        from aiida_pino.data import PinoSimulationData

        from . import viz

        sim = self._load_node(pk, PinoSimulationData)
        try:
            return viz.render_case_frame_png(sim, field=field, view=view, frame=frame, contour=contour)
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc

    # ---------------------------------------------------------- meltpool stats
    def simulation_stats(self, pk: int, *, recompute: bool = False, compute: bool = False) -> dict[str, Any]:
        """Cached melt-pool stats. Computing is opt-in, because it is slow.

        `compute_case_metrics` reads every frame of the case: 197 s for a
        331-frame case under load, 67 s for a 119-frame one. The case browser
        fires this the moment a row is clicked, so an idempotent-looking GET
        blocked for minutes, and a second tab or a prefetch paid it again. The
        cheap answer is `computed: false` and the caller asks for the work.
        """
        from aiida_pino.data import PinoSimulationData

        from . import viz

        sim = self._load_node(pk, PinoSimulationData)
        if not (compute or recompute):
            cached = sim.base.extras.get(viz.STATS_EXTRA_KEY, None)
            if not (isinstance(cached, dict) and cached.get("version") == viz.STATS_VERSION):
                return {"pk": pk, "stats": None, "computed": False}
            return {"pk": pk, "stats": cached, "computed": True}
        try:
            stats = viz.get_or_compute_case_stats(sim, recompute=recompute)
        except KeyError as exc:
            # Collapsed/steady sims store a reduced field set; melt-pool metrics
            # need the raw fields. Having no stats is a normal state for such a
            # simulation, not a failed request: say so with a 200.
            return {"pk": pk, "stats": None, "computed": True, "unavailable": f"missing field {exc.args[0]}"}
        return {"pk": pk, "stats": stats, "computed": True}

    def _dataset_stats_rows(self, pk: int) -> list[dict[str, Any]]:
        from . import viz

        rows = []
        for case in self.dataset_cases(pk)["cases"]:
            if case.get("missing"):
                continue
            stats = None
            unavailable = None
            try:
                from aiida.orm import load_node

                from aiida_pino.utils.stats_autocompute import UNAVAILABLE_KEY

                extras = load_node(case["uuid"]).base.extras
                cached = extras.get(viz.STATS_EXTRA_KEY, None)
                if isinstance(cached, dict) and cached.get("version") == viz.STATS_VERSION:
                    stats = cached
                else:
                    # The background worker tombstones a case whose fields cannot
                    # yield melt metrics at all (reduced-field datasets carry no
                    # `Melt Region`). Without honouring it here the console offers
                    # those cases as "missing" forever and each click re-runs a
                    # multi-minute compute that is guaranteed to raise KeyError.
                    dead = extras.get(UNAVAILABLE_KEY, None)
                    if isinstance(dead, dict) and dead.get("version") == viz.STATS_VERSION:
                        unavailable = dead
            except Exception:  # noqa: BLE001
                pass
            rows.append({**case, "stats": stats, "unavailable": unavailable})
        return rows

    def dataset_stats(self, pk: int) -> dict[str, Any]:
        """Cached-only aggregate; computing happens via dataset_stats_compute."""
        rows = self._dataset_stats_rows(pk)
        done = [r for r in rows if r["stats"]]
        summary: dict[str, Any] = {
            "total": len(rows),
            "computed": len(done),
            "unavailable": sum(1 for r in rows if r.get("unavailable")),
        }
        if done:
            import statistics

            regimes: dict[str, int] = {}
            for r in done:
                reg = r["stats"].get("regime")
                if reg:
                    regimes[reg] = regimes.get(reg, 0) + 1
            summary["regime_counts"] = regimes
            if regimes:
                summary["keyhole_fraction"] = round(regimes.get("keyhole", 0) / max(1, sum(regimes.values())), 3)
            # (stats key, also report a max) -- melt/cavity width and length are
            # the full ellipse axes added with stats v5; cavity_rmin_um stays for
            # continuity with summaries recorded before it.
            for key, with_max in (
                ("melt_vol_um3", True), ("melt_depth_um", True), ("depth_um", True),
                ("melt_width_um", True), ("melt_length_um", True),
                ("cavity_width_um", False), ("cavity_length_um", False),
                ("cavity_rmin_um", False), ("tmax_k", True),
            ):
                vals = [r["stats"][key] for r in done if r["stats"].get(key) is not None]
                if not vals:
                    continue
                summary[f"{key}_mean"] = statistics.fmean(vals)
                if with_max:
                    summary[f"{key}_max"] = max(vals)
        return {"pk": pk, "summary": summary, "cases": rows}

    def dataset_stats_compute(self, pk: int, *, limit: int = 25) -> dict[str, Any]:
        """Compute up to ``limit`` missing per-case stats (batched so the HTTP
        request stays bounded; the client loops until remaining == 0)."""
        from . import viz

        rows = self._dataset_stats_rows(pk)
        missing = [r for r in rows if not r["stats"] and not r.get("unavailable")]
        computed_now = 0
        failed: list[str] = []
        for r in missing[: max(1, int(limit))]:
            try:
                # Child process, not this one: see `_compute_case_stats_isolated`.
                if _compute_case_stats_isolated(str(r["uuid"])):
                    computed_now += 1
                else:
                    failed.append(f"{r.get('case_name') or r.get('pk')}: compute failed")
            except Exception as exc:  # noqa: BLE001 - keep batch going, report at the end
                logger.warning("stats compute failed for sim %s: %r", r.get("pk"), exc)
                failed.append(f"{r.get('case_name') or r.get('pk')}: {type(exc).__name__}")
        # A batch visits each case once, so nothing here benefits from the
        # viewer's field-stack cache -- drop it and hand the arena back, or the
        # web process keeps every stack it touched (see `_trim_malloc_arena`).
        viz.clear_caches()
        _trim_malloc_arena()
        # `failed` cases keep no stats, so they stay in `missing` on the next
        # call: count them as consumed or the client's loop never terminates.
        return {
            "pk": pk,
            "computed_now": computed_now,
            "failed": failed,
            "remaining": max(0, len(missing) - computed_now - len(failed)),
            "total": len(rows),
        }

    def dataset_stats_plot(self, pk: int, *, kind: str = "regime") -> bytes:
        from . import viz

        rows = self._dataset_stats_rows(pk)
        try:
            return viz.render_dataset_stats_png(rows, kind=kind)
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc

    # ------------------------------------------------- object-store reclaim
    def reclaim_space(self, *, dry_run: bool = True) -> dict[str, Any]:
        """Reclaim disk from evicted/unreferenced objects, repacking only the
        packs that hold them.

        Dry run by default: reports how many objects are unreferenced, how much
        that frees, and which packs would be repacked. The real run deletes the
        unreferenced objects (the safe `get_unreferenced_keyset` set, same as
        `verdi storage maintain`) and repacks just those packs under the
        exclusive profile lock — so the daemon and web console must be stopped
        first, exactly as `--full` requires.
        """
        from aiida.common.exceptions import LockedProfileError
        from aiida.manage import get_manager
        from aiida.manage.profile_access import ProfileAccessManager

        from . import maintenance

        backend = get_manager().get_profile_storage()
        repository = backend.get_repository()
        container = repository._container
        unref = list(backend.get_unreferenced_keyset())
        plan = maintenance.plan_reclaim(container, unref)
        plan["dry_run"] = bool(dry_run)
        if dry_run or not unref:
            return plan
        try:
            with ProfileAccessManager(backend.profile).lock():
                repository.delete_objects(unref)
                for pack_id in plan["packs"]:
                    container.repack_pack(str(pack_id))
                container._vacuum()
        except LockedProfileError as exc:
            raise InvalidRequestError(
                f"cannot reclaim while the profile is in use ({exc}). Stop the AiiDA daemon and the "
                "web console first, then retry — repacking needs exclusive access, like `verdi storage "
                "maintain --full`."
            ) from exc
        plan["reclaimed_bytes"] = plan["reclaimable_bytes"]
        plan["packs_repacked"] = len(plan["packs"])
        return plan

    # ------------------------------------------------------- data eviction
    def _core_member_uuids(self) -> set[str]:
        """Member uuids of every `core`-tagged collection, to protect their data."""
        from aiida.orm import QueryBuilder

        from aiida_pino.data import PinoDatasetCollectionData

        builder = QueryBuilder().append(
            PinoDatasetCollectionData,
            filters={"extras.keywords": {"contains": ["core"]}},
            project=["attributes.member_simulation_uuids"],
        )
        out: set[str] = set()
        for (uuids,) in builder.all():
            out.update(str(u) for u in (uuids or []))
        return out

    def restore_deposit_payloads(
        self,
        deposit: str,
        *,
        collections: list[str] | None = None,
        for_nodes: list[str] | None = None,
        all_evicted: bool = False,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Put the arrays of a published deposit back into its imported simulation nodes.

        The inverse of eviction for a profile made by `verdi archive import` of the deposit:
        each member of the named collections (or every evicted simulation with
        `all_evicted`) is read from its payload tar, checked against the deposit's sha256 and
        written back into the node itself, keeping its UUID. `for_nodes` names what is to be
        re-run instead of its collections: an evaluation restores the dataset it scored, a
        training workflow (or the model it produced) its training and validation datasets.
        Defaults to a dry run.
        """
        import sys

        from aiida import orm
        from aiida.common.links import LinkType

        from aiida_pino.data import PinoDatasetCollectionData, PinoSimulationData
        from aiida_pino.data.pino_model import PinoModelData
        from aiida_pino.utils.deposit_restore import DepositPayloads, restore_from_deposit

        def _load(ident):
            return orm.load_node(int(ident) if str(ident).isdigit() else str(ident))

        def _consumed(ident) -> list[str]:
            node = _load(ident)
            metadata = node.base.attributes.all.get("evaluation_metadata") or {}
            if metadata.get("dataset_source_uuid"):
                return [str(metadata["dataset_source_uuid"])]
            if isinstance(node, PinoModelData):
                creators = [link.node for link in node.base.links.get_incoming(link_type=LinkType.CREATE).all()]
                node = creators[0] if creators else node
                while getattr(node, "caller", None) is not None:
                    node = node.caller
            inputs = {link.link_label: link.node for link in
                      node.base.links.get_incoming(link_type=(LinkType.INPUT_WORK, LinkType.INPUT_CALC)).all()}
            found = [str(inputs[name].uuid) for name in ("train_dataset", "val_dataset", "dataset") if name in inputs]
            if not found:
                raise InvalidRequestError(
                    f"{ident} is a {type(node).__name__} with no dataset input; name its collections with --collection")
            return found

        wanted = list(collections or [])
        for ident in for_nodes or []:
            wanted += _consumed(ident)
        if not wanted and not all_evicted:
            raise InvalidRequestError("name at least one --collection or --for, or pass --all")
        payloads = DepositPayloads(deposit)
        sims: dict[str, Any] = {}
        for ident in wanted:
            node = _load(ident)
            if not isinstance(node, PinoDatasetCollectionData):
                raise InvalidRequestError(f"{ident} is a {type(node).__name__}, not a dataset collection")
            for sim in node.iter_member_simulations():
                sims[str(sim.uuid)] = sim
        if all_evicted:
            query = orm.QueryBuilder().append(
                PinoSimulationData, filters={"extras": {"has_key": PinoSimulationData.DATA_EVICTED_EXTRA}}
            )
            for (sim,) in query.iterall():
                sims[str(sim.uuid)] = sim
        return restore_from_deposit(
            payloads, sims.values(), dry_run=dry_run,
            progress=lambda line: print(line, file=sys.stderr, flush=True),
        )

    def evict_dataset_data(self, pk: int, *, dry_run: bool = True, reason: str | None = None) -> dict[str, Any]:
        """Delete a dataset's array data, keeping all metadata and provenance.

        Eviction is per simulation and the data is shared: a simulation belongs
        to one base dataset but can be unioned into composed/derived sets, so
        evicting here empties it everywhere it is referenced — the metadata and
        provenance survive in every one of them. `core` datasets are refused,
        and so is any dataset whose members are part of a core set, so reference
        data cannot be evicted through a non-core view. Defaults to a dry run.
        """
        from aiida_pino.data import PinoDatasetCollectionData

        node = self._load_node(pk, PinoDatasetCollectionData)
        if "core" in (node.base.extras.get("keywords", None) or []):
            raise InvalidRequestError(
                f"refusing to evict data of core dataset {pk}: it is a protected reference set. "
                "Remove the 'core' keyword first if this is deliberate."
            )
        members = self._member_sims(node)
        protected = self._core_member_uuids()
        core_members = [m.pk for m in members if str(m.uuid) in protected]
        if core_members:
            shown = ", ".join(str(p) for p in core_members[:5])
            more = f" (+{len(core_members) - 5} more)" if len(core_members) > 5 else ""
            raise InvalidRequestError(
                f"refusing to evict dataset {pk}: {len(core_members)} of its simulations "
                f"(e.g. {shown}{more}) are also members of a core dataset. Evicting here would "
                "delete protected reference data."
            )
        evictable = [m for m in members if not m.is_data_evicted()]
        if dry_run:
            reclaimable = sum(m.data_size_bytes() for m in evictable)
            return {
                "pk": pk, "dry_run": True, "members": len(members),
                "would_evict": len(evictable), "already_evicted": len(members) - len(evictable),
                "reclaimable_bytes": reclaimable,
            }
        freed = 0
        for m in evictable:
            freed += m.evict_data(reason=reason)
        return {
            "pk": pk, "dry_run": False, "members": len(members),
            "evicted": len(evictable), "bytes_freed": freed,
            # The bytes are unreferenced now; `aiida-pino reclaim` returns the disk space.
            "reclaim_hint": "run `aiida-pino reclaim` to reclaim the freed disk space",
        }

    def staged_datasets(self, *, computer: str, verify: bool = False) -> dict[str, Any]:
        """What runtime caches are staged on a computer."""
        from aiida import orm

        from aiida_pino.utils.stage_remote import staged_caches_on

        try:
            target = orm.load_computer(label=computer)
        except Exception as exc:  # noqa: BLE001
            raise InvalidRequestError(f"No computer labelled `{computer}`: {exc}") from exc
        rows = staged_caches_on(target, verify=verify)
        return {"computer": computer, "count": len([r for r in rows if r.get("cache_name")]), "staged": rows}

    def staged_check(
        self,
        dataset: str,
        *,
        computer: str,
        derivation: dict | None = None,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Is the collection a SUBMISSION would consume staged on `computer`?

        `staged_datasets` lists what is present and cannot answer this: a remote job does not
        consume the dataset that was passed, it consumes whatever the derivation profile
        resolves to, keyed by `cache_discriminator_for` + `runtime_cache_name`. Confirming a
        base uuid in the listing is a WEAKER claim, and it read as confirmation twice before
        a submission failed on it (2026-09-06).

        Three answers, and the third is the one that previously had none:

        - `staged`: the derivation resolves to an existing collection and its cache is there.
        - `not_staged`: it resolves, but the cache is absent -- `stage_remote` names the pk.
        - `would_be_created`: the derivation resolves to NO existing collection, so submission
          will create it and it cannot be staged in advance at all. Answering "not staged"
          there would send someone to stage a node that does not exist.
        """
        from aiida import orm

        from aiida_pino.calculations.base import cache_discriminator_for
        from aiida_pino.utils.dataset_derivation import (
            find_reusable_derived_dataset,
            validate_derivation_profile,
        )
        from aiida_pino.utils.runtime_cache import runtime_cache_name
        from aiida_pino.utils.data_nodes import find_remote_runtime_cache

        try:
            target = orm.load_computer(label=computer)
        except Exception as exc:  # noqa: BLE001
            raise InvalidRequestError(f"No computer labelled `{computer}`: {exc}") from exc
        try:
            base = orm.load_node(int(dataset)) if str(dataset).isdigit() else orm.load_node(uuid=dataset)
        except Exception as exc:  # noqa: BLE001
            raise InvalidRequestError(f"No dataset `{dataset}`: {exc}") from exc

        node = base
        resolved_note = "the dataset itself (no derivation given)"
        if derivation:
            profile = validate_derivation_profile(dict(derivation))
            found = find_reusable_derived_dataset(base, profile)
            if found is None:
                return {
                    "verdict": "would_be_created",
                    "computer": computer,
                    "dataset_pk": base.pk,
                    "detail": (
                        "This derivation resolves to no existing collection, so the submission "
                        "will CREATE it and it cannot be staged in advance. Submit; the eval "
                        "will stop with exit 307 naming the new pk, then "
                        "`aiida-pino stage-remote <that pk> --computer "
                        f"{computer}` and resubmit."
                    ),
                }
            node = found
            resolved_note = f"derived collection pk {found.pk} (resolved from the profile)"

        transport_type = getattr(target, "transport_type", "core.local")
        discriminator = cache_discriminator_for(
            node, runtime_fields=list(fields) if fields else None, transport_type=transport_type
        )
        cache_name = runtime_cache_name(node.uuid, discriminator)
        staged = find_remote_runtime_cache(target, cache_name)
        hint = " ".join(fields) if fields else ""
        return {
            "verdict": "staged" if staged is not None else "not_staged",
            "computer": computer,
            "dataset_pk": base.pk,
            "consumed_pk": node.pk,
            "consumed": resolved_note,
            "cache_name": cache_name,
            "remote_path": staged.get_remote_path() if staged is not None else None,
            "stage_remote": None if staged is not None else (
                f"aiida-pino stage-remote {node.pk} --computer {computer}"
                + (f" --fields {hint}" if hint else "")
            ),
        }

    def unstage_remote_dataset(
        self,
        dataset: str,
        *,
        computer: str,
        derivation: dict | None = None,
        fields: list[str] | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Delete a staged runtime cache from a computer and retract its record.

        Nothing did this before. `retract_remote_runtime_cache` only marks the RECORD
        superseded -- deliberately, so provenance keeps the fact that a copy once existed --
        but it never removes the bytes, so staged caches accumulated on the remote with no
        way to reclaim them short of a manual `rm` over ssh. That matters more now that a
        full-resolution uncompressed stage is ~69 GB for a 96-member set.

        Dry-run by default, like `evict` and `compress`: this deletes data on another machine,
        and the copy here is the only other one.
        """
        from aiida import orm

        from aiida_pino.calculations.base import cache_discriminator_for
        from aiida_pino.utils.data_nodes import find_remote_runtime_cache
        from aiida_pino.utils.dataset_derivation import (
            find_reusable_derived_dataset,
            validate_derivation_profile,
        )
        from aiida_pino.utils.runtime_cache import runtime_cache_name
        from aiida_pino.utils.stage_remote import (
            remote_cache_root,
            retract_remote_runtime_cache,
        )

        try:
            target = orm.load_computer(label=computer)
        except Exception as exc:  # noqa: BLE001
            raise InvalidRequestError(f"No computer labelled `{computer}`: {exc}") from exc
        try:
            node = orm.load_node(int(dataset)) if str(dataset).isdigit() else orm.load_node(uuid=dataset)
        except Exception as exc:  # noqa: BLE001
            raise InvalidRequestError(f"No dataset `{dataset}`: {exc}") from exc
        if derivation:
            found = find_reusable_derived_dataset(node, validate_derivation_profile(dict(derivation)))
            if found is None:
                raise InvalidRequestError(
                    "That derivation resolves to no existing collection, so nothing is staged "
                    "for it. Check with `aiida-pino staged --check` first."
                )
            node = found
        transport_type = getattr(target, "transport_type", "core.local")
        discriminator = cache_discriminator_for(
            node, runtime_fields=list(fields) if fields else None, transport_type=transport_type
        )
        cache_name = runtime_cache_name(node.uuid, discriminator)
        record = find_remote_runtime_cache(target, cache_name)
        result = {
            "computer": computer,
            "dataset_pk": node.pk,
            "cache_name": cache_name,
            "recorded": record is not None,
            "dry_run": dry_run,
        }
        with target.get_transport() as transport:
            root = remote_cache_root(transport, None)
            remote_dir = f"{root}/{cache_name}"
            result["remote_path"] = remote_dir
            result["present_on_remote"] = bool(transport.path_exists(remote_dir))
            if dry_run:
                result["note"] = "dry run: nothing deleted, nothing retracted"
                return result
            if result["present_on_remote"]:
                transport.rmtree(remote_dir)
                result["deleted"] = True
        # Retract only AFTER the bytes are gone, so a record never says "absent" while the
        # directory is still occupying the disk.
        result["retracted"] = bool(retract_remote_runtime_cache(target, cache_name))
        return result

    def compare_evaluations(self, pks: list[int]) -> dict[str, Any]:
        """Whether these evaluations measure the same thing, axis by axis.

        Reports differences rather than blocking: the point is that a
        difference is VISIBLE to someone who did not know to look. Max
        statistics are called out separately, because peak metrics grow as
        sqrt(2 ln N) and a frame-count difference manufactures a gap that per-
        cell means do not show.
        """
        from aiida_pino.utils.comparability import compare_evaluations as _compare

        if len(pks) < 1:
            raise InvalidRequestError("compare needs at least one evaluation pk")
        try:
            return _compare(pks)
        except Exception as exc:  # noqa: BLE001 - surfaces as a request error
            raise InvalidRequestError(f"could not compare {pks}: {exc}") from exc

    def stage_remote_dataset(
        self,
        pk: int,
        *,
        computer: str,
        fields: list[str] | None = None,
        remote_root: str | None = None,
        dry_run: bool = False,
        for_training: bool = False,
    ) -> dict[str, Any]:
        """Copy a dataset's runtime cache onto a compute box and verify it there.

        With `for_training`, `pk` is a past TRAINING process and every collection
        its CalcJobs consumed is staged — which is what you want, because a job
        reads a DERIVED collection the workchain built at submit time, not the
        base collection you passed it.

        A remote job cannot build its own cache: the payloads live in this
        machine's object store. Staging is therefore a separate, deliberate step
        with a one-time cost per (dataset, field set, machine) — tens of minutes
        for ~51 GB on 1 GbE — after which every submission to that box reuses it.

        The copy is named by the same function `stage_dataset_input` looks it up
        with, and it is verified on the remote against an exact set of relative
        paths and sizes before a `RemoteData` records it. That check is not
        belt-and-braces: `puttree` reports nothing, creates missing parents
        silently, and nests a duplicate tree on a re-put, so a count- or
        size-based check passes on the most likely corruption.
        """
        from aiida import orm

        from aiida_pino.data import PinoDatasetCollectionData
        from aiida_pino.utils.stage_remote import stage_collection_to_computer

        if for_training:
            from aiida_pino.utils.stage_remote import collections_consumed_by

            consumed = collections_consumed_by(pk)
            if not consumed:
                raise InvalidRequestError(
                    f"process {pk} has no dataset collections on its CalcJobs to stage. "
                    "Pass a finished or running training pk, or a collection pk without "
                    "--for-training."
                )
            staged = []
            for role, node in consumed:
                result = self.stage_remote_dataset(
                    node.pk, computer=computer, fields=fields, remote_root=remote_root, dry_run=dry_run
                )
                result["role"] = role
                staged.append(result)
            return {"for_training": pk, "staged": staged, "count": len(staged)}

        collection = self._load_node(pk, PinoDatasetCollectionData)
        try:
            target = orm.load_computer(label=computer)
        except Exception as exc:  # noqa: BLE001 - surfaces as a request error
            raise InvalidRequestError(f"No computer labelled `{computer}`: {exc}") from exc
        try:
            return stage_collection_to_computer(
                collection,
                target,
                runtime_fields=fields or None,
                remote_root=remote_root,
                dry_run=dry_run,
            )
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc

    def compress_dataset_data(self, pk: int, *, dry_run: bool = True) -> dict[str, Any]:
        """Convert a dataset's legacy arrays-mode members to compressed npz_file.

        Lossless: every member's reads are verified byte-identical before the
        swap, so `core` datasets are allowed. Members already in npz_file mode
        (or evicted) are skipped. The old uncompressed objects become
        unreferenced; disk returns at the next `aiida-pino reclaim`. Defaults
        to a dry run reporting what would be converted.
        """
        from aiida_pino.data import PinoDatasetCollectionData, PinoSimulationData

        node = self._load_node(pk, PinoDatasetCollectionData)
        members = self._member_sims(node)
        convertible = [
            m
            for m in members
            if m.get_storage_mode() == PinoSimulationData.STORAGE_MODE_ARRAYS and not m.is_data_evicted()
        ]
        if dry_run:
            return {
                "pk": pk, "dry_run": True, "members": len(members),
                "would_convert": len(convertible),
                "bytes_before": sum(m.data_size_bytes() for m in convertible),
            }
        converted = 0
        skipped: list[dict[str, Any]] = []
        bytes_before = bytes_after = 0
        for m in convertible:
            try:
                report = m.compress_to_npz_file()
            except ValueError as exc:
                # Refused (not an exact canonicalization image): keep going,
                # report it — a partial conversion is still a win.
                skipped.append({"pk": m.pk, "error": str(exc)})
                continue
            converted += 1
            bytes_before += report["bytes_before"]
            bytes_after += report["bytes_after"]
        return {
            "pk": pk, "dry_run": False, "members": len(members),
            "converted": converted, "skipped": skipped,
            "bytes_before": bytes_before, "bytes_after": bytes_after,
            "reclaim_hint": "run `aiida-pino reclaim` to free the old uncompressed objects",
        }

    # ------------------------------------------------------------- processes
    def list_processes(
        self,
        *,
        limit: int | None = None,
        label_filter: str | None = None,
        only_pino: bool = True,
        include_smokes: bool = False,
        state: str | None = None,
        category: str | None = None,
        recent: bool = False,
    ) -> list[dict[str, Any]]:
        """Recent processes, newest first.

        `state` is applied **in the query**, not to the fetched page. It used to
        be a front-end filter over whatever 100 rows had been fetched, so asking
        for failures on a database with 395 killed processes returned two rows
        and implied nothing had ever been killed (review criticism K).
        """
        from aiida.orm import ProcessNode, QueryBuilder

        from aiida_pino.utils.vram import SMOKE_KEYWORD

        limit = self.settings.clamp_limit(limit)
        # Post-query filters (pino-only, smoke hiding) drop rows, so over-fetch
        # to still fill a page.
        over_fetch = only_pino or not include_smokes
        parts = [f for f in (_process_state_filter(state), _process_category_filter(category)) if f]
        filters = {"and": parts} if len(parts) > 1 else (parts[0] if parts else {})
        q = (
            QueryBuilder()
            .append(ProcessNode, project=["*"], filters=filters or None)
            # `recent` orders by mtime: "what changed state most recently" is
            # the triage question, and `verdi process list -O` cannot ask it --
            # upstream accepts only `id` and `ctime`, so a 20-day-old workchain
            # that just moved sorts below a minute-old one that has not.
            .order_by({ProcessNode: {("mtime" if recent else "ctime"): "desc"}})
            # `limit` is None when no cap was asked for, which QueryBuilder
            # already reads as unlimited; only multiply when there is a number.
            .limit(max(limit * 3, limit) if (over_fetch and limit) else limit)
        )
        rows: list[dict[str, Any]] = []
        for node in q.all(flat=True):
            if only_pino and not self._is_pino_process(node):
                continue
            if not include_smokes and SMOKE_KEYWORD in _node_keywords(node):
                continue
            label = node.process_label or ""
            if label_filter and label_filter.lower() not in label.lower() and label_filter.lower() not in (
                node.process_type or ""
            ).lower():
                continue
            rows.append(
                {
                    "pk": node.pk,
                    "uuid": node.uuid,
                    "label": label,
                    "title": node.label or None,
                    "description": node.description or None,
                    "state": getattr(node.process_state, "value", None),
                    "exit_status": node.exit_status,
                    "process_type": node.process_type,
                    "ctime": node.ctime.isoformat() if node.ctime else None,
                    "finished_ok": bool(node.is_finished_ok),
                    "terminated": bool(node.is_terminated),
                    # Pausing a workchain correctly leaves its in-flight CalcJob
                    # running, so the row said "Paused" while a solve was still
                    # holding the FLOW-3D seat. Name the child that is still up.
                    "paused_with_unfinished_child": self._running_child_pk(node),
                    "planned_runs": node.base.extras.get("planned_runs", None),
                    "completed_runs": node.base.extras.get("completed_runs", None),
                }
            )
            if limit and len(rows) >= limit:
                break
        return rows

    @staticmethod
    def _running_child_pk(node) -> int | None:
        """The descendant CalcJob still running under a paused process.

        Pausing a workchain does not stop a CalcJob it already submitted -- that
        is correct, killing a solve mid-flight would waste it -- but the row
        then read "Paused" while the seat was still held. Returns None unless
        the process is actually paused, so it costs nothing on a normal row.
        """
        if not getattr(node, "paused", False):
            return None
        try:
            descendants = node.called_descendants
        except Exception:  # noqa: BLE001
            return None
        for child in descendants:
            if child.process_label in ("Flow3DRunCalculation", "PinoTrainCalculation", "PinoEvalCalculation"):
                if not child.is_terminated:
                    return int(child.pk)
        return None

    def _wrong_type_hint(self, pk: int, attempted: str) -> str:
        """Name the command that *does* work for this node type.

        Only commands that exist: there is no `simulation` or `evaluation` subcommand,
        which this hint once suggested.
        """
        from aiida.orm import load_node

        try:
            node = load_node(pk)
        except Exception:  # noqa: BLE001
            return f"node {pk} not found"
        kind = type(node).__name__
        if kind == "PinoModelEvaluationData":
            better = f"`aiida-pino compare {pk}` (what it scored, on which data and frames)"
            model_uuid = node.base.extras.get("model_uuid", None)
            try:
                model_pk = load_node(model_uuid).pk if model_uuid else None
            except Exception:  # noqa: BLE001 - the hint must not fail on a dangling link
                model_pk = None
            if model_pk is not None:
                better += f", or `aiida-pino model {model_pk}` (its model, with its metrics)"
        else:
            better = {
                "PinoDatasetCollectionData": f"`aiida-pino dataset {pk}`",
                "PinoSimulationData": (
                    f"`aiida-pino case-stats {pk}` (melt-pool statistics) or "
                    f"`aiida-pino plot {pk}` (a field slice); `aiida-pino sims` lists them"
                ),
                "PinoModelData": f"`aiida-pino model {pk}`",
            }.get(kind)
        if better:
            return f"node {pk} is a {kind}; `aiida-pino {attempted}` only takes processes — use {better}"
        return f"node {pk} is a {kind}, not a process"

    def process_detail(self, pk: int) -> dict[str, Any]:
        from aiida.orm import ProcessNode

        try:
            node = self._load_node(pk, ProcessNode)
        except NodeNotFoundError:
            # `show` is the most natural first command for someone handed a pk.
            # Saying only "not the expected type" leaves them guessing which of
            # six commands is the right one.
            raise NodeNotFoundError(self._wrong_type_hint(pk, "show")) from None
        outputs, metrics = self._process_outputs(node)
        return {
            "pk": node.pk,
            "uuid": node.uuid,
            "label": node.process_label,
            "title": node.label or None,
            "description": node.description or None,
            "process_type": node.process_type,
            "state": getattr(node.process_state, "value", None),
            "exit_status": node.exit_status,
            "process_status": node.process_status,
            "exception": node.exception,
            "finished_ok": bool(node.is_finished_ok),
            "terminated": bool(node.is_terminated),
            "ctime": node.ctime.isoformat() if node.ctime else None,
            "mtime": node.mtime.isoformat() if node.mtime else None,
            "outputs": outputs,
            "metrics": metrics,
        }

    @staticmethod
    def _process_outputs(node) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from aiida.common.links import LinkType

        outputs: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {}
        try:
            links = node.base.links.get_outgoing(link_type=(LinkType.RETURN, LinkType.CREATE)).all()
        except Exception:  # noqa: BLE001
            links = []
        seen = set()
        for link in links:
            label = link.link_label
            if label in seen:
                continue
            seen.add(label)
            out_node = link.node
            outputs.append({"label": label, "pk": out_node.pk, "type": type(out_node).__name__})
            if label in ("metrics", "train_summary", "eval_summary", "run_summary", "generation_summary", "batch_summary"):
                try:
                    metrics[label] = out_node.get_dict()
                except Exception:  # noqa: BLE001
                    pass
        return outputs, metrics

    def process_logs(self, pk: int) -> dict[str, Any]:
        from aiida.orm import Log, ProcessNode

        node = self._load_node(pk, ProcessNode)
        family = [node] + list(getattr(node, "called_descendants", []) or [])
        report: list[dict[str, Any]] = []
        for proc in family:
            try:
                for entry in Log.collection.get_logs_for(proc):
                    report.append(
                        {
                            "time": entry.time.isoformat() if entry.time else None,
                            "level": entry.levelname,
                            "message": entry.message,
                            "node_pk": proc.pk,
                        }
                    )
            except Exception:  # noqa: BLE001
                continue
        report.sort(key=lambda r: r["time"] or "")
        console = self._collect_console(family)
        cap = self.settings.logs_max_chars
        if len(console) > cap:
            console = console[-cap:]
        return {"pk": node.pk, "report": report, "console": console}

    @staticmethod
    def _collect_console(family) -> str:
        from aiida.orm import CalcJobNode

        chunks: list[str] = []
        for proc in family:
            if not isinstance(proc, CalcJobNode):
                continue
            header = f"===== CalcJob {proc.pk} ({proc.process_label}) ====="
            body = []
            for name, getter in (("scheduler stdout", "get_scheduler_stdout"), ("scheduler stderr", "get_scheduler_stderr")):
                try:
                    text = getattr(proc, getter)()
                    if text:
                        body.append(f"--- {name} ---\n{text}")
                except Exception:  # noqa: BLE001
                    pass
            try:
                retrieved = proc.outputs.retrieved
                out_name = proc.get_option("output_filename")
                if out_name and out_name in retrieved.base.repository.list_object_names():
                    with retrieved.base.repository.open(out_name) as fh:
                        body.append(f"--- {out_name} ---\n{fh.read()}")
            except Exception:  # noqa: BLE001
                pass
            if body:
                chunks.append(header + "\n" + "\n\n".join(body))
        return "\n\n".join(chunks)

    def process_tail(self, pk: int, *, lines: int = 200, path: str | None = None) -> dict[str, Any]:
        """One-shot remote tail of the first CalcJob's live log. Best-effort:
        returns a note instead of raising when no remote folder is reachable."""
        from aiida.orm import CalcJobNode, ProcessNode

        lines = max(1, min(self.settings.tail_max_lines, int(lines)))
        # Validate the caller-supplied path before doing any work with it. This
        # service listens on 0.0.0.0 with no auth, and
        # `PurePosixPath(base) / "../../../etc/hostname"` escapes the job
        # directory happily -- shlex.quote stops command injection, not
        # traversal. Reproduced: GET /processes/{pk}/tail?path=../../../../../
        # etc/hostname returned the file's contents.
        if path is not None:
            candidate = PurePosixPath(str(path))
            if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
                raise InvalidRequestError(
                    "tail path must be relative to the job's working directory and may "
                    f"not contain '..' (got {path!r})"
                )
        node = self._load_node(pk, ProcessNode)
        family = [node] + list(getattr(node, "called_descendants", []) or [])
        calcjob = next((p for p in family if isinstance(p, CalcJobNode)), None)
        if calcjob is None:
            return {"pk": node.pk, "tail": None, "note": "no CalcJob in this process family yet"}
        if calcjob.is_terminated:
            # A live tail of a finished job is a category error: the scratch
            # directory is usually gone, and `tail`'s "No such file" came back
            # as the tail *body*, styled as output, with no next step. The
            # retrieved log is what the user wants and /processes/{pk}/logs
            # already serves it.
            return {
                "pk": node.pk,
                "tail": None,
                "note": f"run {calcjob.pk} has finished ({calcjob.process_state.value if calcjob.process_state else 'terminated'})"
                        " — there is no live tail; its retrieved output is under Logs",
            }
        try:
            remote = calcjob.outputs.remote_folder
        except Exception:  # noqa: BLE001
            return {"pk": node.pk, "tail": "", "note": "no remote_folder output (job not yet running)"}
        rel_path = PurePosixPath(str(path or self._default_tail_rel(calcjob)))
        remote_path = str(PurePosixPath(remote.get_remote_path()) / rel_path)

        # A tail of a REMOTE job is an SSH connect inside the web process, and an
        # open Monitor tab polls. Collapse repeated polls onto one attempt so an
        # unreachable box costs one slow request per TTL instead of one per
        # poll. Local tails are not cached: they are cheap, and freshness is the
        # whole point there.
        computer = getattr(calcjob, "computer", None)
        transport_type = getattr(computer, "transport_type", "core.local")
        hostname = getattr(computer, "hostname", "") or ""
        if transport_type != "core.local":
            return self._tail_cache.get_or_call(
                (int(node.pk), str(rel_path), int(lines)),
                lambda: self._tail_over_transport(
                    node, calcjob, remote, rel_path, remote_path, lines, hostname
                ),
            )
        return self._tail_over_transport(node, calcjob, remote, rel_path, remote_path, lines, hostname)

    def _live_tail_text(self, calcjob, workdir: str) -> str:
        """The live log tail of a running training, wherever it is running.

        This read used to be `tail_file(Path(workdir) / LIVE_TAIL_FILENAME)` with the comment
        that the computer is the local machine. `get_remote_workdir()` returns a path on the
        EXECUTING box, so for a job on a compute box that path does not exist here: the read
        returned empty, `last_step` returned None, and the row showed "starting..." with no
        step and no beam-box MAE for the entire run. Local runs were unaffected, which is why
        it looked like a problem with the remote box and not with the monitor.

        Local stays a direct read -- it is cheap and freshness is the point. Remote goes over
        the same transport the Logs pane uses, through a 30 s cache, so an unreachable or slow
        box costs one attempt per TTL instead of one per poll.

        Returns "" on any failure: an unreadable tail means "no progress known", which renders
        as "starting..." exactly as before. It must never raise into the monitor payload.
        """
        # Imported here, as everywhere else in this module: `monitor` and the live-tail
        # filename are not module-level names.
        from aiida_pino.console import monitor
        from aiida_pino.scripts.live_tail import LIVE_TAIL_FILENAME

        computer = getattr(calcjob, "computer", None)
        transport_type = getattr(computer, "transport_type", "core.local")
        if transport_type == "core.local":
            return monitor.tail_file(Path(workdir) / LIVE_TAIL_FILENAME)
        try:
            remote = calcjob.outputs.remote_folder
        except Exception:  # noqa: BLE001 - not yet running
            return ""
        rel_path = PurePosixPath(LIVE_TAIL_FILENAME)
        remote_path = str(PurePosixPath(remote.get_remote_path()) / rel_path)
        hostname = getattr(computer, "hostname", "") or ""

        def _fetch():
            try:
                return self._tail_over_transport(
                    calcjob, calcjob, remote, rel_path, remote_path, 400, hostname
                )
            except Exception:  # noqa: BLE001 - an unreachable box is not an error here
                return {"tail": ""}

        result = self._progress_tail_cache.get_or_call(
            ("progress", int(calcjob.pk)), _fetch
        )
        return str((result or {}).get("tail") or "")

    def _live_tail_age_s(self, calcjob, workdir: str) -> float | None:
        """Seconds since the live log was last written, or None if unknown.

        A tail says what a run REACHED, never that it is still running: a dead or finished
        job's log keeps its final step forever. Reading one without this made the monitor
        show a stopped run as though it were training -- a compute box sat at 0% util with no
        compute processes while two of its runs displayed a moving step count and a beam-box
        MAE. Before the remote tail existed those rows read "starting...", which was useless
        but not misleading; this is worse, so the age travels with the number.
        """
        import time as _time

        computer = getattr(calcjob, "computer", None)
        transport_type = getattr(computer, "transport_type", "core.local")
        if transport_type == "core.local":
            from aiida_pino.scripts.live_tail import LIVE_TAIL_FILENAME

            for name in (LIVE_TAIL_FILENAME, "aiida.out"):
                try:
                    return max(0.0, _time.time() - (Path(workdir) / name).stat().st_mtime)
                except OSError:
                    continue
            return None
        try:
            remote = calcjob.outputs.remote_folder
        except Exception:  # noqa: BLE001
            return None

        def _fetch_age():
            import shlex

            from aiida_pino.scripts.live_tail import LIVE_TAIL_FILENAME

            base = PurePosixPath(remote.get_remote_path())
            # The SAME fallback the text path takes. `_tail_over_transport` falls back to
            # `aiida.out` when the live-tail file is absent -- and it IS absent on some runs
            # (71840 had only `aiida.out`) -- so statting the live-tail name alone returned
            # "age unknown" for exactly the rows whose text had come from the fallback. An
            # age that measures a different file than the number beside it is worse than no
            # age: it silently exempts a stale row from the staleness check.
            try:
                fallback = calcjob.get_option("output_filename") or "aiida.out"
            except Exception:  # noqa: BLE001
                fallback = "aiida.out"
            candidates = [str(base / LIVE_TAIL_FILENAME), str(base / fallback)]
            # `2>/dev/null` per candidate, first hit wins, so one round trip covers both.
            script = " || ".join(
                f"stat -c %Y {shlex.quote(c)} 2>/dev/null" for c in candidates
            )
            try:
                transport = remote.get_authinfo().get_transport()
                with transport:
                    # The remote clock on both sides of the subtraction, so the two boxes
                    # do not have to agree on the time.
                    retval, stdout, _ = transport.exec_command_wait(
                        f"m=$({script}); [ -n \"$m\" ] && expr $(date +%s) - $m"
                    )
                if retval:
                    return None
                return float(str(stdout).strip())
            except (OSError, ValueError):
                return None
            except Exception:  # noqa: BLE001
                return None

        return self._progress_tail_cache.get_or_call(("age", int(calcjob.pk)), _fetch_age)



    def _remote_gpu_snapshots(self):
        """GPU rows for every configured non-local computer, cached.

        Shares the tail cache's TTL for the same reason it exists: an open
        Monitor tab polls, and an unreachable box must cost one slow request per
        TTL rather than one per poll. Each computer is cached separately so one
        unreachable box does not hide a reachable one.
        """
        from aiida import orm

        from aiida_pino.console import monitor

        rows, notes = [], []
        try:
            computers = list(orm.Computer.collection.all())
        except Exception:  # noqa: BLE001 - a monitor panel must not fail the page
            return rows, notes
        for computer in computers:
            if getattr(computer, "transport_type", "core.local") == "core.local":
                continue
            label = computer.label

            # LABEL only into the closure -- never the ORM object. Capturing a
            # live `orm.Computer` made a healthy box report
            # "unavailable: DetachedInstanceError" on 2026-08-21: in a
            # long-lived console the session behind it expires between this
            # listing and `get_transport()`, which queries authinfo.
            def _fetch(label=label):
                return monitor.gpu_rows_for_computer(
                    label, load_computer=orm.load_computer, log=logger.warning
                )

            found, note = self._tail_cache.get_or_call(("gpus", label), _fetch)
            rows.extend(found)
            if note:
                notes.append(note)
        return rows, notes

    def _tail_over_transport(self, node, calcjob, remote, rel_path, remote_path, lines, hostname):
        """The transport half of `process_tail`, split so it can be cached.

        Every return carries `hostname`: a log pane that does not say WHICH
        machine it is showing is the same ambiguity as a metric that does not
        say which regime produced it, and it matters more now that a job may be
        on a box the reader has never logged into.
        """
        try:
            transport = remote.get_authinfo().get_transport()
            with transport:
                retval, stdout, stderr = transport.exec_command_wait(
                    f"tail -n {lines} {shlex.quote(remote_path)}"
                )
                if retval:
                    # `tail`'s own error is not the job's output. Returning it as
                    # `tail` put "tail: cannot open ...: No such file or
                    # directory" inside the log pane in log styling, under a 200,
                    # for every poll of every Flow3DRunCalculation — which never
                    # writes the live-tail file at all. Fall back to the job's
                    # stdout, which is the file the user wanted anyway.
                    fallback_rel = None
                    try:
                        fallback_rel = calcjob.get_option("output_filename") or "aiida.out"
                    except Exception:  # noqa: BLE001
                        fallback_rel = "aiida.out"
                    if str(rel_path) != str(fallback_rel):
                        fallback_path = str(PurePosixPath(remote.get_remote_path()) / fallback_rel)
                        retval2, stdout2, _ = transport.exec_command_wait(
                            f"tail -n {lines} {shlex.quote(fallback_path)}"
                        )
                        if not retval2:
                            return {
                                "pk": node.pk,
                                "tail": stdout2 or "",
                                "path": fallback_path,
                                "note": f"{rel_path} is not written by this job; showing {fallback_rel}",
                                "hostname": hostname,
                            }
                    return {
                        "pk": node.pk,
                        "tail": None,
                        "path": remote_path,
                        "note": f"no live tail: {(stderr or '').strip() or f'tail exited {retval}'}",
                        "hostname": hostname,
                    }
            return {"pk": node.pk, "tail": stdout, "path": remote_path, "hostname": hostname}
        except Exception as exc:  # noqa: BLE001
            return {"pk": node.pk, "tail": "", "note": f"tail unavailable: {type(exc).__name__}: {exc}", "path": remote_path, "hostname": hostname}

    @staticmethod
    def _default_tail_rel(calcjob) -> str:
        try:
            from aiida_pino.scripts.live_tail import LIVE_TAIL_FILENAME

            return LIVE_TAIL_FILENAME
        except Exception:  # noqa: BLE001
            pass
        try:
            return calcjob.get_option("output_filename") or "aiida.out"
        except Exception:  # noqa: BLE001
            return "aiida.out"

    # ----------------------------------------------------------- previews

    def preview_collection(self, simulation_pks: list[int]) -> dict[str, Any]:
        from aiida_pino.data import validate_collection_members

        sims = [self._load_node(pk) for pk in simulation_pks]
        try:
            members, signature = validate_collection_members(sims)
        except Exception as exc:  # noqa: BLE001
            raise InvalidRequestError(f"incompatible members: {exc}") from exc
        # Structured pks, not a str() of the (members, signature) tuple: the console
        # used to print the Python repr of the ORM objects inside its JSON block.
        return {"member_pks": [m.pk for m in members], "signature": signature, "member_count": len(members)}

    def _member_sims(self, node) -> list[Any]:
        """A collection's member simulations, in member order."""
        from aiida.orm import QueryBuilder

        from aiida_pino.data import PinoSimulationData

        member_uuids = list(node.base.attributes.get("member_simulation_uuids") or [])
        by_uuid: dict[str, Any] = {}
        if member_uuids:
            q = QueryBuilder().append(PinoSimulationData, filters={"uuid": {"in": member_uuids}}, project=["*"])
            for (sim,) in q.iterall():
                by_uuid[str(sim.uuid)] = sim
        return [by_uuid[u] for u in member_uuids if u in by_uuid]

    def assign_splits(
        self,
        pk: int,
        *,
        train: float,
        val: float,
        test: float,
        only_unassigned: bool = True,
        seed: int = 0,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Randomly assign train/val/test splits to a collection's members.

        Fractions must be non-negative and sum to 1 (±1e-6). With
        only_unassigned (the default) already-assigned members keep their
        split; pass only_unassigned=False to reshuffle everything. The
        assignment is deterministic for a given seed and member set.
        """
        import random

        from aiida_pino.data import PinoDatasetCollectionData
        from aiida_pino.utils.annotations import update_node_annotations

        fractions = {"train": float(train), "val": float(val), "test": float(test)}
        if any(f < 0 for f in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1e-6:
            raise InvalidRequestError(
                f"train/val/test fractions must be >= 0 and sum to 1, got {fractions}"
            )
        node = self._load_node(pk, PinoDatasetCollectionData)
        members = self._member_sims(node)
        before_labels = {
            str(sim.uuid): str(sim.base.extras.get("split", "") or "unassigned") for sim in members
        }
        if only_unassigned:
            candidates = [s for s in members if (s.base.extras.get("split", None) or "unassigned") == "unassigned"]
        else:
            candidates = list(members)
        skipped = len(members) - len(candidates)
        random.Random(int(seed)).shuffle(candidates)
        n = len(candidates)
        # Largest-remainder rounding so counts always sum to n.
        raw = {k: f * n for k, f in fractions.items()}
        counts = {k: int(v) for k, v in raw.items()}
        for k, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[: n - sum(counts.values())]:
            counts[k] += 1
        i = 0
        for split_name in ("train", "val", "test"):
            for sim in candidates[i : i + counts[split_name]]:
                update_node_annotations(sim, split=split_name)
            i += counts[split_name]
        # The labels live in mutable extras, so the *recipe* that produced them
        # is recorded on the base dataset and frozen into the Dict input of the
        # split calcfunction when a subset is first materialised. Seed plus the
        # member list reproduces the assignment exactly; the digest says whether
        # that member list is still the one that was split.
        recipe = {
            "method": "random",
            "stratify_by": None,  # this assigner does not stratify
            "seed": int(seed),
            "fractions": fractions,
            "only_unassigned": bool(only_unassigned),
            "assigned": counts,
            "skipped": skipped,
            "member_count": len(members),
            "member_uuids_sha256": _member_digest(members),
            "assigned_utc": _utc_now(),
        }
        node.base.extras.set("split_recipe", recipe)
        # History, which is what was actually asked for: 38427 carries a recipe
        # claiming random 0.8/0.05/0.15 while its labels are family-inverted, and
        # nothing anywhere says who changed them or when. Append-only.
        from aiida_pino.utils.annotations import append_split_audit, audit_actor

        audit = list(node.base.extras.get("split_audit", None) or [])
        append_split_audit(
            audit,
            before=before_labels,
            after={
                str(sim.uuid): str(sim.base.extras.get("split", "") or "unassigned")
                for sim in members
            },
            reason=str(reason or "assign_splits"),
            actor=audit_actor(),
            method="random",
        )
        node.base.extras.set("split_audit", audit)
        return {
            "pk": node.pk,
            "total": len(members),
            "assigned": counts,
            "skipped": skipped,
            "recipe": recipe,
        }

    def _compose_collection(self, sims, metadata, *, extra_parameters=None, **inputs):
        """Reuse an identical collection, else create one through the compose
        calcfunction. Thin wrapper over `utils.compose.compose_collection`, which
        the dataset-filing paths use too."""
        from aiida_pino.utils.compose import compose_collection

        return compose_collection(sims, metadata, extra_parameters=extra_parameters, **inputs)

    def create_collection(
        self,
        simulation_pks: list[int],
        collection_name: str,
        *,
        description: str | None = None,
        keywords: list[str] | None = None,
        dataset_pks: list[int] | None = None,
        skip_description: bool = False,
        field_policy: str | None = None,
    ) -> dict[str, Any]:
        """Compose a new collection from simulations and/or whole datasets.

        Members of `dataset_pks` collections are unioned (order kept, deduped
        by uuid) with any explicitly picked simulations. Console-composed
        collections carry dataset_role "composed" (not base/derived) and are
        created through a calcfunction, so the source datasets and picked
        simulations are real provenance inputs in the AiiDA graph.
        """
        from aiida_pino.data import PinoDatasetCollectionData
        from aiida_pino.utils import build_collection_metadata_from_simulations
        from aiida_pino.utils.annotations import update_node_annotations

        if not collection_name.strip():
            raise InvalidRequestError("collection_name is required")
        # 713 of 802 collections have no description, so a dataset's purpose
        # lives in whoever made it. A name is not a description: "bundle_gate"
        # does not say what was gated or why. Required unless the caller says
        # explicitly that it should be skipped.
        if not str(description or "").strip() and not skip_description:
            raise InvalidRequestError(
                "description is required: say what this dataset is for and how it was "
                "selected, so the next reader does not have to reconstruct it. Pass "
                "skip_description=true (CLI: --no-description) if it genuinely needs none."
            )
        sims, seen, source_nodes, picked = [], set(), [], []
        for ds_pk in dataset_pks or []:
            ds = self._load_node(ds_pk, PinoDatasetCollectionData)
            source_nodes.append(ds)
            for sim in self._member_sims(ds):
                if sim.uuid not in seen:
                    seen.add(sim.uuid)
                    sims.append(sim)
        for pk in simulation_pks:
            sim = self._load_node(pk)
            if sim.uuid not in seen:
                seen.add(sim.uuid)
                sims.append(sim)
                picked.append(sim)
        if not sims:
            raise InvalidRequestError("no simulations: pass simulation_pks and/or dataset_pks")
        metadata = build_collection_metadata_from_simulations(
            sims,
            collection_name=collection_name.strip(),
            dataset_role="composed",
            composition_kind="composed",
            # "intersect" admits members whose field sets differ and trains on
            # the shared fields — what the datasource does with such a folder
            # anyway; the strict default refuses, naming the odd fields out.
            extra_metadata={"field_policy": field_policy} if field_policy else None,
        )
        from aiida_pino.utils.compose import member_inputs

        inputs = {f"dataset_{i}": ds for i, ds in enumerate(source_nodes)}
        # Every member is an input, not just the ones picked by hand: a member
        # that arrived via a source dataset is as much a part of this collection
        # as one named explicitly, and only inputs put it in the graph.
        inputs.update(member_inputs(sims))
        node, created = self._compose_collection(sims, metadata, **inputs)
        sources = [{"pk": ds.pk, "uuid": ds.uuid} for ds in source_nodes]
        if sources and not node.base.extras.get("composed_from", None):
            # UI-layer mirror of the provenance links. The member data itself
            # is shared by reference, never duplicated.
            node.base.extras.set("composed_from", sources)
        if description is not None or keywords is not None:
            update_node_annotations(
                node,
                description=description,
                keywords_set=list(keywords) if keywords is not None else None,
            )
        return {"pk": node.pk, "uuid": node.uuid, "created": bool(created)}

    # -------------------------------------------------------- deletions/kill
    def delete_nodes(self, pks: list[int], *, dry_run: bool = True) -> dict[str, Any]:
        from aiida.tools import delete_nodes

        pk_ints = [int(p) for p in pks]
        # Confirm each exists first (clear 404 instead of a silent no-op), and
        # refuse to touch nodes the user marked `core` — those are load-bearing
        # reference datasets/models; untag deliberately before deleting.
        protected = []
        for pk in pk_ints:
            node = self._load_node(pk)
            if "core" in (node.base.extras.get("keywords", None) or []):  # get() raises without a default
                protected.append(pk)
        if protected:
            raise InvalidRequestError(
                f"refusing to delete node(s) {protected}: tagged 'core' (protected). "
                "Remove the 'core' keyword first if this is deliberate."
            )
        deleted, was_deleted = delete_nodes(pk_ints, dry_run=dry_run)
        return {"requested": pk_ints, "affected": sorted(int(p) for p in deleted), "dry_run": bool(dry_run), "deleted": bool(was_deleted)}

    def kill_process(self, pk: int) -> dict[str, Any]:
        available, reason = self.daemon_available()
        if not available:
            raise DaemonUnavailableError(reason)
        from aiida.engine.processes.control import kill_processes

        node = self._load_node(pk)
        try:
            kill_processes([node], timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("kill failed for pk %s: %r", node.pk, exc)
            if _broker_fault(exc):
                raise DaemonUnavailableError("kill failed: the AiiDA daemon is not reachable") from exc
            raise OperationFailedError(f"kill failed ({type(exc).__name__}); the process was not killed") from exc
        return {"pk": node.pk, "requested_kill": True}

    # -------------------------------------------------------------- submits
    def _submit(self, entry_point: str, *, description=None, submitted_by=None, **inputs) -> dict[str, Any]:
        # `AIIDA_PINO_FOREGROUND=1` (CLI: `aiida-pino --foreground ...`) runs the workflow in
        # this process until it finishes, which needs neither a daemon nor a message broker:
        # the route for a profile made by a plain `verdi presto`, which has no broker
        # (2026-10-07, reproduction test from the deposit).
        import os

        foreground = os.environ.get("AIIDA_PINO_FOREGROUND", "").strip().lower() in ("1", "true", "yes")
        if not foreground:
            available, reason = self.daemon_available()
            if not available:
                raise DaemonUnavailableError(
                    f"{reason}; or run it in this terminal without a daemon: `aiida-pino --foreground ...`"
                )
        from aiida.engine import run_get_node, submit
        from aiida.plugins import WorkflowFactory

        workflow = WorkflowFactory(entry_point)
        try:
            if foreground:
                _, node = run_get_node(workflow, **inputs)
            else:
                node = submit(workflow, **inputs)
            _label_submitted_process(node, entry_point, inputs)
            _annotate_submitted_process(
                node, description=description, submitted_by=submitted_by
            )
        except Exception as exc:  # noqa: BLE001
            # Do not leak the raw AiiDA/broker exception (it can carry DB/RabbitMQ
            # connection details) to the client; log it server-side instead.
            logger.warning("submit failed for %s: %r", entry_point, exc)
            if foreground:
                raise OperationFailedError(
                    f"the run in this terminal raised {type(exc).__name__}; `verdi process list -a` "
                    "shows what it left behind"
                ) from exc
            if _broker_fault(exc):
                raise DaemonUnavailableError("submission failed: the AiiDA daemon/broker is not reachable") from exc
            # Name the class, not the message: the message can carry connection
            # strings, the class name tells the user what to do about it. And
            # say plainly that nothing launched — there is no idempotency key
            # on this path, so a blind resubmit is how you get two runs.
            raise OperationFailedError(
                f"submission failed ({type(exc).__name__}); nothing was launched — "
                "check the Runs tab before resubmitting"
            ) from exc
        if foreground and not node.is_finished_ok:
            # A foreground run has finished by now; reporting it like a submission (exit 0, a pk)
            # read as success when the workflow had failed (2026-10-08, reproduction test).
            message = f": {node.exit_message}" if node.exit_message else ""
            raise OperationFailedError(
                f"{node.process_label} {node.pk} ({node.uuid}) ran in this terminal and failed, "
                f"exit status {node.exit_status}{message}; `verdi process report {node.pk}` says why"
            )
        return {"pk": node.pk, "uuid": node.uuid, "label": node.process_label, "process_type": node.process_type}

    def _code(self, label: str):
        from aiida.orm import load_code

        try:
            return load_code(label)
        except Exception as exc:  # noqa: BLE001
            raise InvalidRequestError(f"code '{label}' not found: {type(exc).__name__}") from exc

    def _training_target(self, requested_code: str, supplied_profile: dict, training: dict):
        """Resolve WHERE a training runs and WHICH card it asks for, together.

        These two cannot be decided apart. `cuda_device` ships as `"0"`, which
        on a two-card box pins every run to card 0 and serialises them silently,
        so a remote submission must promote it to `"auto"` — and whether a
        submission is remote is only knowable once the code is resolved.

        `code_label` in the training profile wins over the request's code, so a
        saved profile can name its machine. Returns `(code, training)` with
        `training` already carrying the resolved `cuda_device`.
        """
        from aiida_pino.utils.profiles import resolve_cuda_device_for_target

        label = (training or {}).get("code_label") or requested_code
        code = self._code(label)
        transport = getattr(getattr(code, "computer", None), "transport_type", "core.local")
        resolved = dict(training or {})
        resolved["cuda_device"] = resolve_cuda_device_for_target(
            supplied_profile or {},
            resolved.get("cuda_device"),
            remote=transport != "core.local",
        )
        return code, resolved

    def _validated_training_profile(
        self, raw: dict[str, Any], *, split_mode: str, for_import: bool = False
    ) -> dict[str, Any]:
        from aiida_pino.utils import merge_training_profile, validate_training_profile
        from aiida_pino.utils.profiles import DEFAULT_TRAINING_PROFILE

        from aiida_pino.utils.profiles import load_training_profile_defaults

        try:
            # Partial profiles overlay the packaged defaults, so a CLI call
            # with just `run_name=x max_steps=500` works instead of demanding
            # the full required-key set every time. The pre-fill MUST come from
            # the profile's own config_name: overlaying DEFAULT_TRAINING_PROFILE
            # stamps the default config's values (time_as_batch=False, lr=6e-5,
            # batch=20 ...) as explicitly-provided keys, silently overriding
            # everything the selected config sets differently.
            config_name = str((raw or {}).get("config_name") or DEFAULT_TRAINING_PROFILE["config_name"])
            base = load_training_profile_defaults(config_name)
            # A key nobody recognises is a typo, and defaults-overlay hides it:
            # `wandb_offlien=true` or `max_step=500` passed submit, rode into the
            # stored profile as an unknown key, and the run trained with the
            # DEFAULT value of the key the user meant. Catch it while the user is
            # still looking at the command they typed.
            _reject_unknown_training_keys(raw or {}, base)
            merged = merge_training_profile(
                {**base, **(raw or {}), "split_mode": split_mode}, for_import=for_import
            )
            return validate_training_profile(
                merged, expected_split_mode=split_mode, for_import=for_import
            )
        except InvalidRequestError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise InvalidRequestError(f"invalid training_profile: {exc}") from exc

    def _validated_derivation_profile(self, raw: dict[str, Any] | None) -> dict[str, Any] | None:
        if raw is None:
            return None
        from aiida_pino.utils import validate_derivation_profile

        try:
            return validate_derivation_profile(raw)
        except Exception as exc:  # noqa: BLE001
            raise InvalidRequestError(f"invalid derivation_profile: {exc}") from exc

    def _validated_evaluation_profile(self, raw: dict[str, Any] | None) -> dict[str, Any] | None:
        from aiida_pino.utils import validate_evaluation_profile

        try:
            return validate_evaluation_profile(raw)
        except Exception as exc:  # noqa: BLE001
            raise InvalidRequestError(f"invalid evaluation_profile: {exc}") from exc

    # ---------------------------------------------------- training presets
    def list_training_presets(self) -> list[dict[str, Any]]:
        from aiida.orm import Dict, QueryBuilder

        q = (
            QueryBuilder()
            .append(Dict, filters={"extras.aiida_pino_kind": "training_preset"}, project=["*"])
            .order_by({Dict: {"ctime": "desc"}})
            .limit(100)
        )
        return [
            {
                "pk": node.pk,
                "name": node.base.extras.get("preset_name", None) or node.label,
                "description": node.description,
                "training_profile": node.get_dict(),
                "ctime": node.ctime.isoformat() if node.ctime else None,
            }
            for node in q.all(flat=True)
        ]

    def save_training_preset(
        self, *, name: str, description: str | None = None, training_profile: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        from aiida.orm import Dict

        if not (name or "").strip():
            raise InvalidRequestError("preset name is required")
        profile = dict(training_profile or {})
        # Partial presets are fine (they overlay the defaults at submit time),
        # but invalid values should fail here, not at 3am in a workchain.
        self._validated_training_profile(profile, split_mode=str(profile.get("split_mode") or "standard"))
        node = Dict(dict=profile).store()
        node.label = f"training preset: {name.strip()}"
        node.base.extras.set("aiida_pino_kind", "training_preset")
        node.base.extras.set("preset_name", name.strip())
        if description:
            node.description = str(description)
        return {"pk": node.pk, "name": name.strip()}

    def training_run_inputs(self, pk: int) -> dict[str, Any]:
        """The reusable inputs of a past training run, for form prefill."""
        node = self._load_node(pk)
        incoming = {
            link.link_label: link.node
            for link in node.base.links.get_incoming().all()
        }
        out: dict[str, Any] = {"pk": node.pk, "label": node.process_label or ""}
        profile = incoming.get("training_profile")
        if profile is None:
            raise InvalidRequestError(f"process {pk} has no training_profile input")
        out["training_profile"] = profile.get_dict()
        for label in ("train_dataset", "val_dataset", "init_model"):
            if label in incoming:
                out[f"{label}_pk"] = incoming[label].pk
        return out

    def resolved_layout(self, pk: int) -> dict[str, Any] | None:
        """The dataset layout a past training ACTUALLY used.

        Needed because the F1 guard refuses a submission whose config-derived
        layout disagrees with the selected dataset, and 271 historical
        `TrainModelWorkChain`s are in exactly that shape — so cloning their
        inputs verbatim now fails at exit 305. The unambiguous answer is not the
        config: for **77 of the 257** recoverable ones the layout that really ran
        disagrees with what the config implies, because the workchain had
        substituted a derived dataset. The truth is the dataset the calcjob was
        handed.

        Falls back to the config-derived layout only for runs that never reached
        a calcjob (14 of 271), where nothing better exists.
        """
        from aiida_pino.utils import build_derivation_profile_from_training_profile
        from aiida_pino.utils.dataset_derivation import dataset_layout_profile_from_metadata

        node = self._load_node(pk)
        for child in getattr(node, "called_descendants", []) or []:
            if child.process_label != "PinoTrainCalculation":
                continue
            try:
                return dict(dataset_layout_profile_from_metadata(
                    child.inputs.train_dataset.get_dataset_metadata()
                ))
            except Exception:  # noqa: BLE001 - fall through to the config
                break
        try:
            profile = self._validated_training_profile(
                dict(node.inputs.training_profile.get_dict()), split_mode="standard"
            )
            return dict(build_derivation_profile_from_training_profile(profile))
        except Exception:  # noqa: BLE001
            return None

    def resubmit_training(
        self,
        pk: int,
        *,
        overrides: dict[str, Any] | None = None,
        run_name: str | None = None,
        mode: str | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Re-run a past training with its own inputs, optionally tweaked.

        Cloning a run's inputs is the only reliable way to submit here — a
        hand-written profile has to satisfy a required-key set nothing prints
        and a config's own defaults nobody memorises, and two attempts in one
        day died a minute after a "successful" submit. This makes the reliable
        path a command instead of a bespoke script per use.

        Overrides are applied to the *stored* profile, then the whole thing goes
        through the same validation as any other submit, so a typo'd key is
        refused here rather than at exit 300.
        """
        from aiida_pino.console.schemas import TrainEvalRequest

        source = self.training_run_inputs(pk)
        profile = dict(source["training_profile"])
        applied, derivation_overrides = split_resubmit_overrides(overrides)
        profile.update(applied)
        if run_name:
            profile["run_name"] = run_name
        elif not applied and not run_name:
            # A rerun that reuses the name is indistinguishable from the
            # original in wandb and in the Runs list.
            profile["run_name"] = f"{profile.get('run_name', 'run')}_rerun"

        train_pk = source.get("train_dataset_pk")
        if train_pk is None:
            raise InvalidRequestError(f"process {pk} has no train_dataset input to reuse")

        # The original run's code, which is the right DEFAULT for a rerun. It is
        # not the last word: `submit_train_eval` calls `_training_target`, which
        # prefers a `code_label` in the training profile, so
        # `--override code_label=<code>` genuinely retargets a resubmit to
        # another machine. Said here because reading this line alone suggests
        # otherwise, and one reviewer reasonably concluded the override was
        # silently ignored (2026-08-19); it is not, but nothing here showed that.
        code = self._resubmit_code_label(pk)
        # Pass the layout the original run actually used. Without it a clone of
        # any pre-guard training is refused at exit 305, because those runs are
        # precisely the ones whose config disagreed with their dataset -- 271 of
        # them. Stating it explicitly is what the guard asks for, and it also
        # makes the rerun reproduce the original instead of whatever the
        # packaged config means today.
        derivation = self.resolved_layout(pk)
        if derivation_overrides:
            derivation = {**dict(derivation or {}), **derivation_overrides}
            applied = {**applied, **{f"derivation.{k}": v for k, v in derivation_overrides.items()}}
        request = TrainEvalRequest(
            mode=mode or "train_eval",
            train_dataset_pk=int(train_pk),
            val_dataset_pk=source.get("val_dataset_pk"),
            python_code=code,
            training_profile=profile,
            init_model_pk=source.get("init_model_pk"),
            derivation_profile=derivation,
        )
        result = self.submit_train_eval(request)
        result["cloned_from"] = pk
        result["overrides_applied"] = applied
        if title or description:
            self.update_annotations(result["pk"], title=title, description=description)
        return result

    def _resubmit_code_label(self, pk: int) -> str:
        """The code the original run used: the DEFAULT for a rerun, not a lock.

        Matching the original environment is what a rerun usually wants, so this
        is what a resubmit uses when nothing says otherwise. A `code_label` in
        the training profile overrides it downstream in `_training_target`,
        which is how a training gets moved to another computer.
        """
        node = self._load_node(pk)
        for member in [node] + list(getattr(node, "called_descendants", []) or []):
            for link in member.base.links.get_incoming().all():
                if link.link_label in ("python_code", "code"):
                    return str(link.node.label or link.node.full_label)
        raise InvalidRequestError(
            f"process {pk} has no python_code input to reuse; pass one explicitly"
        )

    def flow3d_config_schema(self) -> dict[str, Any]:
        """What each FLOW-3D config field means, in what unit, within what range.

        The runner ships `python/config_schema.py` — 50 rows of
        (field, meaning, constraint) that generate its README table — and the
        console rendered none of it: the physics blocks were single-line JSON
        text inputs (the `laser` block is 191 characters) with no units, no
        ranges, no defaults and no link to the docs. The knowledge existed; it
        just never reached the form.
        """
        import importlib.util
        import sys

        root = str(getattr(self.settings, "flow3d_scripts_root", "") or "").strip()
        module_path = Path(root) / "python" / "config_schema.py" if root else None
        if module_path is None or not module_path.is_file():
            return {"fields": {}, "note": "flow3d_scripts_root is not configured; no schema available"}
        try:
            spec = importlib.util.spec_from_file_location("_flow3d_config_schema", module_path)
            module = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(module_path.parent))
            try:
                spec.loader.exec_module(module)
            finally:
                sys.path.remove(str(module_path.parent))
            rows = list(getattr(module, "CONFIG_SCHEMA", []))
        except Exception as exc:  # noqa: BLE001 - documentation is never fatal
            return {"fields": {}, "note": f"schema unavailable: {type(exc).__name__}"}

        fields: dict[str, dict[str, str]] = {}
        for row in rows:
            if len(row) < 3:
                continue
            name = str(row[0]).strip().strip("`")
            fields[name] = {"meaning": str(row[1]).strip(), "constraint": str(row[2]).strip()}
        return {"fields": fields, "count": len(fields), "source": str(module_path)}

    def profile_keys(self, which: str = "training", config_name: str | None = None) -> dict[str, Any]:
        """Every accepted key of a profile, with its default. Answers "which keys can be set?".

        The required-key list used to be learnable only by failing:
        `training_profile is missing required keys: cuda_device, hydra_overrides,
        wandb_offline` is not documented anywhere a user reads first.

        `config_name` matters more than it looks. Without it this printed the
        defaults of `config_NS_energy_PINO_flow3D` whatever config you were
        actually going to submit — including `moving_ref_frame: true` and
        `time_avg_window: 30`, the exact keys that silently decided which dataset
        a run trained on (F1). Asking what a config would do got another
        config's answer.
        """
        from aiida_pino.utils.dataset_derivation import DEFAULT_DERIVATION_PROFILE
        from aiida_pino.utils.model_registry import EVALUATION_OVERRIDE_KEYS
        from aiida_pino.utils.profiles import (
            DEFAULT_TRAINING_PROFILE,
            TRAINING_PROFILE_REQUIRED_KEYS,
        )

        which = (which or "training").strip().lower()
        if which == "training":
            selected = str(config_name or "").strip()
            if selected:
                from aiida_pino.utils.profiles import load_training_profile_defaults

                try:
                    defaults = load_training_profile_defaults(selected)
                except Exception as exc:  # noqa: BLE001 - a bad name is a user error
                    raise InvalidRequestError(f"unknown training config {selected!r}: {exc}") from exc
            else:
                defaults = DEFAULT_TRAINING_PROFILE
            required = sorted(TRAINING_PROFILE_REQUIRED_KEYS)
        elif which == "evaluation":
            # An eval profile is a set of allowed overrides, not a defaults dict.
            #
            # `hydra_overrides` needs its OWN wording: it EXTENDS the model's list rather
            # than replacing it, and the blanket phrasing reads as a whole-list swap for a
            # list-valued key. That misreading cost three profiles
            # of defensive copying (2026-08-25) and nearly a wrong number, because the
            # feared behaviour -- supplying one key silently dropping the donor's
            # `FNO_input_keys` -- would return a plausible score for a different input set
            # rather than failing. The append is real and tested; only the description
            # said otherwise.
            _EXTENDS = (
                "(override; EXTENDS the model's own list -- your entries are appended, so "
                "hydra's last-wins rule lets you restate a key to change it, and anything "
                "you omit is kept)"
            )
            defaults = {
                key: _EXTENDS if key == "hydra_overrides" else "(override; unset means the model's own value)"
                for key in sorted(EVALUATION_OVERRIDE_KEYS)
            }
            required = []
        elif which == "derivation":
            defaults, required = DEFAULT_DERIVATION_PROFILE, []
        elif which in ("flow3d_run", "execution"):
            # 13 of these 16 keys had no control anywhere in the console and no
            # way to discover them — including `timeout_s` and the watchdog.
            from aiida_pino.utils.profiles import FLOW3D_RUN_EXECUTION_PROFILE_KEYS

            defaults = {key: "(unset; see the FLOW-3D README)" for key in sorted(FLOW3D_RUN_EXECUTION_PROFILE_KEYS)}
            required = []
        else:
            raise InvalidRequestError(
                f"unknown profile {which!r}; expected one of: training, evaluation, derivation, flow3d_run"
            )
        return {
            "profile": which,
            "required": required,
            "keys": {key: defaults[key] for key in sorted(defaults)},
            "note": (
                "Partial profiles are allowed: anything omitted comes from the selected "
                "config's packaged defaults. Unknown keys are refused at submit time."
            ),
        }

    def _split_recipe(self, base_node, split_name, member_sims) -> dict[str, Any]:
        """How this split's labels were decided, as far as it is knowable.

        ``assign_splits`` leaves a ``split_recipe`` extra on the base dataset;
        anything older (labels written by an import script or by hand) has no
        recorded method, and says so rather than implying one.
        """
        recipe = dict(base_node.base.extras.get("split_recipe", None) or {})
        if not recipe:
            recipe = {
                "method": "unrecorded",
                "note": "split labels predate recipe capture; assigner, seed and ratios unknown",
            }
        recipe.update(
            {
                "split": split_name,
                "labels_from": "member `split` extras",
                "split_member_count": len(member_sims),
                "base_collection_uuid": str(base_node.uuid),
                "base_member_uuids_sha256": _member_digest(self._member_sims(base_node)),
                "composed_utc": _utc_now(),
            }
        )
        return recipe

    def _split_subset_collection(self, base_node, member_sims, split_name):
        """Stable `subset`-role collection for one split of ``base_node``.

        These are machine-made views of one base dataset, not unions of several,
        so they are classified **subset** (not composed) and record their base
        explicitly (``base_collection_pk``/``uuid`` plus a human
        ``base_collection_label``), so the browser can say "test split of <base>"
        rather than leaving the origin a guess. Deduped — the same split of the
        same base is composed once and reused, never re-split on every submit.
        Filed under a per-base subfolder of "Training splits (auto)" so splits
        group by their source dataset.

        Created through the compose calcfunction with the base dataset as an
        input and the split recipe as a stored ``Dict``, so "which members, from
        which dataset, chosen how" is answerable from the graph alone instead of
        from a sentence in a paper.
        """
        from aiida_pino.utils import build_collection_metadata_from_simulations

        base_label = str(base_node.label or base_node.get_dataset_metadata().get("source_folder_name") or f"dataset {base_node.pk}")
        folder = str(base_node.get_dataset_metadata().get("source_folder_name") or f"dataset_{base_node.pk}")
        metadata = build_collection_metadata_from_simulations(
            member_sims,
            collection_name=f"{folder}_{split_name}_split",
            dataset_role="subset",
            base_collection=base_node,
            parent_collection=base_node,
            extra_metadata={
                "base_collection_label": base_label,
                "subset_split": split_name,
                "description": f"{split_name} split of {base_label} (#{base_node.pk})",
            },
        )
        from aiida_pino.utils.compose import member_inputs

        node, _created = self._compose_collection(
            member_sims,
            metadata,
            extra_parameters={"split_recipe": self._split_recipe(base_node, split_name, member_sims)},
            base_dataset=base_node,
            # The members are inputs too, not just the base. Passing only the
            # base looked sufficient — the calcfunction can read members out of
            # it — but it puts no simulation in the graph, so `verdi archive
            # create -N <split>` exported 4 nodes and none of the data for the
            # six splits made after the membership fix landed. These splits are
            # exactly what the reported metrics are computed on.
            **member_inputs(member_sims),
        )
        # ONE flat folder, not one subfolder per base dataset. This is the only automatic
        # writer of a folder extra in the codebase (the others are user-driven, via the CLI
        # and the browser), and it fires on every train submit without an explicit val set
        # and every eval on a split-bearing dataset -- so a per-dataset path mints a new
        # subfolder per dataset forever. It had produced 18 of them holding 108 sets, 27
        # under `ti64_preheat60` alone, which is what the 2026-09-01 curation flattened.
        # A per-dataset grouping belongs in the browser's
        # rendering, where it costs nothing and can be undone, not in a stored extra.
        self.set_folder([node.pk], "Training splits")
        return node

    def _split_subsets(self, node) -> tuple[Any, Any, str] | None:
        """Train/val subset collections from the members' split extras, or None.

        Honours the annotated split when the user picks one dataset: composes
        (with provenance) a train subset and a val subset instead of silently
        validating on the training data. Reuses existing identical subsets.
        """
        members = self._member_sims(node)
        by_split: dict[str, list[Any]] = {}
        for sim in members:
            by_split.setdefault(str(sim.base.extras.get("split", None) or "unassigned"), []).append(sim)
        if not by_split.get("train") or not by_split.get("val"):
            return None
        out = [self._split_subset_collection(node, by_split[s], s) for s in ("train", "val")]
        held_out = len(by_split.get("test") or [])
        note = (
            f"honoured split extras of dataset {node.pk}: {len(by_split['train'])} train / "
            f"{len(by_split['val'])} val" + (f" ({held_out} test members held out)" if held_out else "")
        )
        return out[0], out[1], note

    #: Marker text `_test_subset` uses when it falls back off the test split.
    NOT_HELD_OUT_MARKER = "NOT a held-out test metric"

    @staticmethod
    def _test_split_caveat(pk: int, *, sibling: dict | None) -> str:
        """Say why a metric is not held out, in terms of the thing to change.

        This used to read "dataset N has split extras but an EMPTY test split",
        which is literally true and points at the data. The data is usually
        fine: the campaign that hit this had 96/96 train, 14/14 val and 13/13
        test, cleanly disjoint across sibling subsets of one parent. The cause
        is that `TrainEvalRequest` has no eval-dataset field, so the auto-eval
        scores the val split — plumbing, not the dataset. Naming the sibling
        that DOES hold the test split turns a dead end into the next command.
        """
        where = ""
        if sibling:
            where = (
                f" Collection {sibling['pk']} ({sibling.get('label') or 'unnamed'}) in the same "
                f"family holds {sibling['count']} test members — score against it with "
                f"`aiida-pino submit eval --dataset {sibling['pk']} --model <MODEL_PK> --python-code <CODE>`."
            )
        return (
            f"WARNING: no test dataset was supplied, so this scored the VAL split of dataset "
            f"{pk} — the same data the run's own checkpoints were selected against. "
            f"This is {AiidaConsoleService.NOT_HELD_OUT_MARKER}.{where}"
        )

    @staticmethod
    def _test_split_sibling(node) -> dict | None:
        """A sibling subset of the same parent that carries test members."""
        from aiida import orm

        from aiida_pino.data import PinoDatasetCollectionData

        try:
            parent = (node.get_dataset_metadata() or {}).get("parent_collection_pk")
            if not parent:
                return None
            builder = orm.QueryBuilder().append(
                PinoDatasetCollectionData,
                filters={"attributes.parent_collection_pk": {"==": int(parent)}},
                project=["*"],
            )
            for candidate in builder.all(flat=True):
                if candidate.pk == node.pk:
                    continue
                members = list(candidate.iter_member_simulations())
                tests = [m for m in members if str(m.base.extras.get("split", "")) == "test"]
                if tests and len(tests) == len(members):
                    return {"pk": candidate.pk, "label": candidate.label, "count": len(tests)}
        except Exception:  # noqa: BLE001 - a better message must not fail a submit
            return None
        return None

    def _stamp_metric_caveat(self, pk: int, notes: list[str]) -> None:
        """Record a not-held-out evaluation on the NODE, not just in the reply.

        The warning was returned in `result["notes"]`, which reaches whoever ran
        the command and nothing else. The NUMBER outlives that: it sits on the
        eval node, gets read by model comparisons, tables and anyone browsing
        months later, and looks exactly like a test metric. A caveat that lives
        only in a submit-time reply is a caveat nobody downstream can see.

        Cheap and additive: an extra, so nothing that already reads these nodes
        changes, and anything that cares can ask.
        """
        caveats = [note for note in notes or [] if self.NOT_HELD_OUT_MARKER in note]
        if not caveats:
            return
        try:
            from aiida import orm

            node = orm.load_node(int(pk))
            node.base.extras.set("eval_held_out_test", False)
            node.base.extras.set("eval_metric_caveat", caveats[0])
        except Exception:  # noqa: BLE001 - a caveat must never fail a submission
            pass

    def _test_subset(self, node) -> tuple[Any | None, str | None]:
        """Test-split subset collection from the members' split extras.

        Eval scores the held-out `test` split of the dataset it is given, so a
        caller passes the full dataset (train/val/test) rather than a hand-built
        test collection. Returns ``(subset, note)``:

        - a proper `test` subset -> ``(subset, note)``; caller evaluates it;
        - the whole dataset is already test -> ``(None, note)`` used as-is, no
          pointless re-compose;
        - split extras present but the test split is EMPTY -> ``(None, WARNING)``
          — a loud note, never a silent eval on train/val data;
        - no split extras at all -> ``(None, None)``, backward-compatible: the
          passed collection is simply the eval set.
        """
        members = self._member_sims(node)
        by_split: dict[str, list[Any]] = {}
        for sim in members:
            by_split.setdefault(str(sim.base.extras.get("split", None) or "unassigned"), []).append(sim)
        test = by_split.get("test") or []
        has_splits = bool(by_split.get("train") or by_split.get("val") or test)
        if test and len(test) == len(members):
            return None, f"dataset {node.pk} is entirely test split ({len(test)} members); evaluating as-is"
        if test:
            subset = self._split_subset_collection(node, test, "test")
            return subset, f"evaluating on the test split of dataset {node.pk}: {len(test)} test members"
        if has_splits:
            # Deliberately does not name the fallback target: submit_eval_only
            # falls back to the passed dataset, train_eval falls back to
            # val_dataset. Naming one of them would be wrong in the other path.
            return None, self._test_split_caveat(node.pk, sibling=self._test_split_sibling(node))
        return None, None

    def _refuse_if_evaluation_only(self, node, what: str) -> None:
        """Refuse to train on a dataset whose every member is held-out test data.

        Marking a dataset evaluation-only means exactly this: every member
        carries `split="test"`. Without a guard the training path treats "no
        train members" as "no split information at all" and falls back to using
        the whole dataset — so a set reserved for evaluation would be trained on
        and its held-out metric would be a training metric.
        """
        splits = {str(s.base.extras.get("split", "") or "unassigned") for s in self._member_sims(node)}
        if is_evaluation_only_split_set(splits):
            raise InvalidRequestError(
                f"dataset {node.pk} is evaluation-only: all {node.get_simulation_count()} members are "
                f'split="test", so it cannot be used as {what}. Evaluate against it with '
                "`submit eval`, or assign train/val splits first if it really is training data."
            )

    def mark_evaluation_only(self, pk: int) -> dict[str, Any]:
        """Reserve a dataset for evaluation: every member becomes `test`.

        The counterpart to the guard above. Eval-only data is the thing that
        must never leak into training, and "all members are test" is the state
        the rest of the pipeline already understands — `_test_subset` uses the
        whole dataset as-is, and training now refuses it outright.
        """
        from aiida_pino.data import PinoDatasetCollectionData
        from aiida_pino.utils.annotations import update_node_annotations

        node = self._load_node(pk, PinoDatasetCollectionData)
        members = self._member_sims(node)
        changed = 0
        for sim in members:
            if str(sim.base.extras.get("split", "") or "") != "test":
                update_node_annotations(sim, split="test")
                changed += 1
        node.base.extras.set("evaluation_only", True)
        return {"pk": node.pk, "members": len(members), "changed": changed, "split": "test"}

    def presubmit_train(self, req) -> dict[str, Any]:
        """The submission sequence WITHOUT the submission: what would this training do?

        Same code path as `TrainModelWorkChain.validate_inputs` -- literally the same
        function, `resolve_training_submission`, with `materialize=False` -- so the answer
        cannot drift from what a real submission would decide. Reports the effective
        profiles after demotion, the per-source resolution plan (as-is / base /
        reuse-derived / would-materialize with an estimated size), every note the workchain
        would have reported, the first problem it would have refused on, and whether the
        resolved collection is staged on the target computer.
        """
        from aiida_pino.data import PinoDatasetCollectionData
        from aiida_pino.utils.presubmit import resolve_training_submission, staging_report

        train = self._load_node(req.train_dataset_pk, PinoDatasetCollectionData)
        val = self._load_node(req.val_dataset_pk, PinoDatasetCollectionData) if req.val_dataset_pk else train
        # The SAME code resolution as a real submission (`code_label` in the profile wins
        # over the request), or the answer would describe a submission to a different box.
        code, _training = self._training_target(
            req.python_code, req.training_profile or {}, dict(req.training_profile or {})
        )
        outcome = resolve_training_submission(
            train, val,
            dict(req.training_profile or {}),
            code,
            dict(req.derivation_profile) if req.derivation_profile else None,
            allow_split_violation=bool(getattr(req, "allow_split_violation", False)),
            materialize=False,
        )
        plans = {}
        for name, plan in (outcome.get("plans") or {}).items():
            resolved = plan.get("resolved")
            entry = {"action": plan.get("action"),
                     "resolved_pk": resolved.pk if resolved is not None else None,
                     "resolved_label": resolved.label if resolved is not None else None}
            if plan.get("estimated_bytes") is not None:
                entry["estimated_gb"] = round(plan["estimated_bytes"] / 1e9, 1)
            entry["staging"] = staging_report(plan, code)
            plans[name] = entry
        return {
            "ok": not outcome["problems"],
            "problems": outcome["problems"],
            "notes": outcome["notes"],
            "plans": plans,
            "effective_env": (outcome.get("training_profile") or {}).get("env"),
        }

    def submit_train_eval(self, req) -> dict[str, Any]:
        from aiida.orm import Bool, Dict

        from aiida_pino.data import PinoDatasetCollectionData

        # Typed, like `submit_derive`: an untyped load of, say, a code pk reached
        # `attributes.get("member_simulation_uuids")`, which AiiDA raises on
        # rather than returning None, giving "Internal error: AttributeError"
        # over HTTP and a raw traceback on the CLI. Defaulting the attribute
        # instead would be worse -- the port is `orm.Data`, so a model or a
        # simulation would sail through the guards and actually launch a
        # training run on a non-dataset.
        train = self._load_node(req.train_dataset_pk, PinoDatasetCollectionData)
        self._refuse_if_evaluation_only(train, "training data")
        notes: list[str] = [n for n in (self._superseded_note(train, "training"),) if n]
        if req.val_dataset_pk:
            val = self._load_node(req.val_dataset_pk, PinoDatasetCollectionData)
        else:
            split = self._split_subsets(train)
            if split is not None:
                train, val, note = split
                notes.append(note)
            else:
                val = train
                notes.append(
                    "no validation dataset and no split extras: validation metrics will be "
                    "computed on the training data"
                )
        training = self._validated_training_profile(req.training_profile, split_mode="standard")
        derivation = self._validated_derivation_profile(req.derivation_profile)
        _training_code, training = self._training_target(req.python_code, req.training_profile, training)
        # Say at SUBMIT time what the upload step will cost. A local job whose resolved
        # collection has no runtime cache exports it inside `do_upload` first -- minutes a
        # member, GPUs idle -- and until this line nothing told the submitter (2026-09-11).
        try:
            from aiida_pino.utils.presubmit import resolve_training_submission, staging_report

            dry = resolve_training_submission(train, val, training, _training_code, derivation, materialize=False)
            # Always one line per source, so "zero-copy" is a statement the submitter reads,
            # not an absence they have to interpret (random paths, 2026-09-11: a silent
            # submit was indistinguishable from one that had not checked).
            for name, plan in (dry.get("plans") or {}).items():
                report = staging_report(plan, _training_code)
                notes.append(f"{name} staging: {report['detail']}")
        except Exception as exc:  # noqa: BLE001 - an advisory must never block a submit
            notes.append(f"could not estimate staging cost: {type(exc).__name__}: {exc}")
        inputs: dict[str, Any] = {
            "train_dataset": train,
            "val_dataset": val,
            "training_profile": Dict(dict=training),
            "python_code": _training_code,
        }
        if derivation is not None:
            inputs["derivation_profile"] = Dict(dict=derivation)
        if getattr(req, "init_model_pk", None):
            from aiida import orm as _orm

            from aiida_pino.data import PinoModelData

            # A model node, or checkpoint artifacts directly. The calcjob's
            # `init_artifacts` port always took all three; only the layers above
            # narrowed it, forcing an import-model round trip to resume a killed
            # run from its own scratch training_output/.
            node = self._load_node(req.init_model_pk)
            if not isinstance(node, (PinoModelData, _orm.FolderData, _orm.RemoteData)):
                raise InvalidRequestError(
                    f"init_model {req.init_model_pk} is a {type(node).__name__}; expected a "
                    "model, FolderData or RemoteData holding checkpoint files"
                )
            inputs["init_model"] = node
        entry = "pino.train_model" if req.mode == "train" else "pino.train_eval"
        if req.mode == "train_eval":
            evaluation = self._validated_evaluation_profile(req.evaluation_profile)
            if evaluation:
                inputs["evaluation_profile"] = Dict(dict=evaluation)
            # Evaluate on the held-out test split of the original dataset, not on
            # val. Reuses the test subset (create_collection dedupes); a loud
            # note if the dataset has splits but no test members.
            source = self._load_node(req.train_dataset_pk)
            test_subset, test_note = self._test_subset(source)
            if test_subset is None:
                # `_test_subset` looks for test members INSIDE the dataset it is
                # given, which assumes the caller passes the whole train/val/test
                # collection. A split-based submission passes the TRAIN subset, so
                # the lookup can never succeed and eval quietly fell back to val.
                # The family's test sibling is the right target, and the lookup for
                # it already existed -- it was only used to build the warning.
                # Deliberately NOT done inside `_test_subset`: submit_eval_only
                # shares it, and there the dataset is an explicit choice that must
                # never be silently redirected.
                sibling = self._test_split_sibling(source)
                if sibling:
                    test_subset = self._load_node(sibling["pk"], PinoDatasetCollectionData)
                    test_note = (
                        f"dataset {source.pk} holds no test members; evaluating on its "
                        f"family's test split {sibling['pk']} ({sibling['count']} members)"
                    )
            if test_subset is not None:
                inputs["eval_dataset"] = test_subset
                # The pair travels together or not at all. TrainEvalWorkChain reads
                # `evaluate_on_test` and silently falls back to val_dataset without
                # it, so passing the subset alone composed a test collection and
                # then threw it away -- every console train_eval since 2026-08-06.
                if getattr(req, "evaluate_on_test", True):
                    inputs["evaluate_on_test"] = Bool(True)
                else:
                    notes.append(
                        f"evaluate_on_test=False: scoring the VAL split, not the test "
                        f"subset {test_subset.pk}. This is a health readout, not a "
                        f"held-out metric."
                    )
            if test_note:
                notes.append(test_note)
        result = self._submit(
            entry,
            description=getattr(req, "description", None),
            submitted_by=getattr(req, "submitted_by", None),
            **inputs,
        )
        # A run nobody can attribute or explain is the one that gets rediscovered
        # months later as "what was this?". Nudge rather than refuse: a legitimate
        # quick experiment should not be blocked, but it should not be silent.
        missing = [
            name
            for name, value in (
                ("description", getattr(req, "description", None)),
                ("submitted_by", getattr(req, "submitted_by", None)),
            )
            if not str(value or "").strip()
        ]
        if missing:
            notes.append(
                f"no {' or '.join(missing)} recorded for this run; set it with "
                f"`aiida-pino describe {result.get('pk', '<pk>')} \"what and why\" "
                "--title \"short name\"`"
            )
        if notes:
            result["notes"] = notes
            self._stamp_metric_caveat(result.get("pk"), notes)
        return result

    def submit_eval_only(self, req) -> dict[str, Any]:
        from aiida.orm import Dict

        from aiida_pino.data import PinoModelData

        from aiida_pino.data import PinoDatasetCollectionData

        dataset = self._load_node(req.dataset_pk, PinoDatasetCollectionData)
        model = self._load_node(req.model_pk, PinoModelData)
        superseded_note = self._superseded_note(dataset, "evaluation")
        # Score the held-out test split of the dataset given, composing it once
        # and reusing it thereafter (create_collection dedupes). A caller passes
        # the full dataset; a pre-built collection with no split extras is used
        # as-is.
        notes: list[str] = [n for n in (superseded_note,) if n]
        test_subset, note = self._test_subset(dataset)
        if test_subset is not None:
            dataset = test_subset
        if note:
            notes.append(note)
        derivation = self._validated_derivation_profile(req.derivation_profile)
        evaluation = self._validated_evaluation_profile(req.evaluation_profile)
        inputs: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "python_code": self._code(req.python_code),
        }
        if evaluation:
            inputs["evaluation_profile"] = Dict(dict=evaluation)
        if derivation is not None:
            inputs["derivation_profile"] = Dict(dict=derivation)
        from aiida.orm import Bool

        if getattr(req, "allow_domain_change", False):
            inputs["allow_domain_change"] = Bool(True)
            notes.append(
                "allow_domain_change: scoring on a different domain extent than the model "
                "trained on, by explicit request"
            )
        if getattr(req, "allow_training_data", False):
            inputs["allow_training_data"] = Bool(True)
            notes.append(
                "allow_training_data: scoring cases the model trained on, by explicit "
                "request; the record is stamped held_out=False"
            )
        result = self._submit(
            "pino.eval_only",
            description=getattr(req, "description", None),
            submitted_by=getattr(req, "submitted_by", None),
            **inputs,
        )
        if notes:
            result["notes"] = notes
            self._stamp_metric_caveat(result.get("pk"), notes)
        return result

    def submit_kfold(self, req) -> dict[str, Any]:
        # Deprecated from the console; the workchain itself is untouched. Kept
        # as a refusing stub rather than deleted so the route, the CLI and the
        # UI all fail with one explanation instead of a 404 or an AttributeError.
        raise InvalidRequestError(KFOLD_DEPRECATION_MESSAGE)

    def _submit_kfold_disabled(self, req) -> dict[str, Any]:
        from aiida.orm import Dict

        from aiida_pino.data import PinoDatasetCollectionData

        dataset = self._load_node(req.dataset_pk, PinoDatasetCollectionData)
        # k-fold trains on every fold, so an evaluation-only set is exactly as
        # unusable here as it is for a straight training run.
        self._refuse_if_evaluation_only(dataset, "k-fold training data")
        training = self._validated_training_profile(req.training_profile, split_mode="kfold")
        derivation = self._validated_derivation_profile(req.derivation_profile)
        _training_code, training = self._training_target(req.python_code, req.training_profile, training)
        inputs: dict[str, Any] = {
            "dataset": dataset,
            "training_profile": Dict(dict=training),
            "python_code": _training_code,
        }
        if derivation is not None:
            inputs["derivation_profile"] = Dict(dict=derivation)
        result = self._submit("pino.kfold_train_eval", **inputs)
        note = self._superseded_note(dataset, "k-fold")
        if note:
            result["notes"] = [note]
        return result

    def submit_raw_flow3d(self, req) -> dict[str, Any]:
        from aiida.orm import Dict, Str

        inputs: dict[str, Any] = {
            "source_path": Str(str(Path(req.source_path).expanduser())),
            "dataset_profile": Dict(dict=req.dataset_profile or {}),
            "python_code": self._code(req.python_code),
        }
        if req.materials_root:
            inputs["materials_root"] = Str(str(Path(req.materials_root).expanduser()))
        return self._submit("pino.raw_flow3d_import", **inputs)

    def _flow3d_runner_root(self, req) -> str:
        """The checkout a submit will actually use, for submit-time validation.

        Falls back to the sibling checkout so the deep, runner-owned validation
        below always runs. Without that fallback an unset root skipped the
        validation *and* let the adapter land on the vendored submodule — the
        two halves of the same failure.
        """
        from aiida_pino.utils.flow3d_config import default_runner_root

        explicit = str((getattr(req, "execution_profile", None) or {}).get("flow3d_scripts_root") or "").strip()
        configured = str(getattr(self.settings, "flow3d_scripts_root", "") or "").strip()
        return explicit or configured or default_runner_root()

    def _flow3d_inputs(self, req) -> dict[str, Any]:
        """Shared validation + input building for FLOW-3D generate/batch."""
        from aiida.orm import Dict, Str

        from aiida_pino.utils import validate_flow3d_npz_profile, validate_flow3d_run_execution_profile

        try:
            validate_flow3d_run_execution_profile(req.execution_profile or {})
            validate_flow3d_npz_profile(req.dataset_profile or {})
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
        execution = dict(req.execution_profile or {})
        # Pin the runner checkout. Left unset, the adapter falls through to the
        # bundled `external/flow3D_scripts` submodule, which predates the
        # calibration campaign and IGNORES `recoil_model` /
        # `void_vapor_pressure_pa` rather than rejecting them — so a console run
        # asking for calibrated physics quietly produced old-physics data. The
        # console was the only launch path that left this unset: of 1178
        # Flow3DRunCalculations, the 3 without a root are the 3 console submits.
        configured_root = self._flow3d_runner_root(req)
        if configured_root and not execution.get("flow3d_scripts_root"):
            execution["flow3d_scripts_root"] = configured_root
        inputs: dict[str, Any] = {
            "execution_profile": Dict(dict=execution),
            "dataset_profile": Dict(dict=req.dataset_profile or {}),
            "python_code": self._code(req.python_code),
        }
        if req.materials_root:
            inputs["materials_root"] = Str(str(Path(req.materials_root).expanduser()))
        return inputs

    def _superseded_note(self, node, role: str) -> str | None:
        """A note, not a refusal, when work is launched on a retired dataset.

        `_add_target_dataset` refuses one as a *filing target* -- writing new
        cases into a retired set is a mistake with no upside. Reading one is
        different: re-running an old training deliberately, to reproduce a
        published number, is legitimate. But the pk can equally come from a
        stale note or a bookmark, and the picker hides superseded rows, so
        nothing on the console path would have said a word.
        """
        if not bool(node.base.extras.get("superseded", False)):
            return None
        replacement = node.base.extras.get("replaced_by_pk", None)
        return (
            f"{role} dataset {node.pk} is superseded"
            + (f" (replaced by #{replacement})" if replacement else "")
            + " — the run was submitted against it as asked; use the replacement if that was not deliberate"
        )

    def _add_target_dataset(self, inputs: dict[str, Any], req) -> None:
        """Attach an explicit filing target, if the caller named one.

        Resolved here rather than in the workchain so a bad pk is a 404 at
        submit time instead of a report line an hour into a solve.
        """
        from aiida_pino.data import PinoDatasetCollectionData

        pk = getattr(req, "target_dataset_pk", None)
        if pk is None:
            return
        node = self._load_node(int(pk), PinoDatasetCollectionData)
        if bool(node.base.extras.get("superseded", False)):
            raise InvalidRequestError(
                f"dataset {node.pk} is superseded (replaced by "
                f"{node.base.extras.get('replaced_by_pk', '?')}); file into the replacement"
            )
        inputs["target_dataset"] = node

    def submit_flow3d_generate(self, req) -> dict[str, Any]:
        """One FLOW-3D case through flow3D_scripts (+ NPZ ingest unless prepare-only)."""
        from aiida.orm import Dict

        if not (req.run_config or {}):
            raise InvalidRequestError("run_config is required (a full flow3D_scripts JSON config)")
        _check_run_config(req.run_config, runner_root=self._flow3d_runner_root(req))
        inputs = self._flow3d_inputs(req)
        inputs["run_config"] = Dict(dict=req.run_config)
        self._add_target_dataset(inputs, req)
        return self._submit("pino.flow3d_generate", **inputs)

    def submit_flow3d_batch(self, req) -> dict[str, Any]:
        from aiida.orm import Int, List

        if not req.run_configs:
            raise InvalidRequestError("run_configs must be a non-empty list of flow3D_scripts configs")
        runner_root = self._flow3d_runner_root(req)
        for position, config in enumerate(req.run_configs):
            _check_run_config(config, where=f"run_configs[{position}]", runner_root=runner_root)
        inputs = self._flow3d_inputs(req)
        inputs["run_configs"] = List(list=list(req.run_configs))
        inputs["max_concurrent_runs"] = Int(int(req.max_concurrent_runs or 1))
        self._add_target_dataset(inputs, req)
        return self._submit("pino.flow3d_batch", **inputs)

    def _aiida_flow3d_queue(self) -> dict[str, Any]:
        """FLOW-3D work queued *through AiiDA*, which the spool never sees.

        AiiDA is the default entry point, and an AiiDA-launched solve takes the
        same seat lock but writes no `.task` file — so a spool-only view reports
        "0 queued" while batches are mid-flight. Observed 2026-08-02: the console
        showed nothing queued while four `Flow3DBatchWorkChain`s had been waiting
        since 07-31, three run calcjobs were blocked on the seat, and a fourth
        (`ti64_lhs_009`) was solving.

        A calcjob that is `waiting` is either running the solver or blocked on
        the seat; the seat holder is the one whose scheduler job is live, so
        both are reported and the caller does not have to guess.

        A batch also reports its PENDING cases. With `max_concurrent_runs=1`
        only one child CalcJob exists at a time; the other N-1 cases live only
        in the workchain's `run_configs` input, so no process count sees them —
        one batch read as "nothing queued" while 299 cases (~3 days of seat
        time) were committed. `planned_runs`/`completed_runs` extras carry the
        truth; absent extras stay None rather than reading as an empty queue.
        """
        from aiida.orm import ProcessNode, QueryBuilder

        active = {"in": ["created", "running", "waiting"]}
        out: dict[str, Any] = {"runs": [], "batches": [], "run_count": 0, "batch_count": 0}
        builder = QueryBuilder().append(
            ProcessNode,
            filters={
                "attributes.process_state": active,
                "attributes.process_label": {"in": ["Flow3DRunCalculation", "Flow3DBatchWorkChain", "Flow3DGenerateWorkChain"]},
            },
            project=[
                "id", "attributes.process_label", "attributes.process_state", "ctime", "label",
                "extras.planned_runs", "extras.completed_runs",
            ],
        )
        builder.order_by({ProcessNode: {"ctime": "asc"}})
        pending_cases = 0
        pending_known = False
        for pk, label, state, ctime, title, planned, completed in builder.all():
            row = {
                "pk": int(pk),
                "label": str(label),
                "state": str(state),
                "title": title or None,
                "since": ctime.isoformat() if ctime else None,
            }
            if label == "Flow3DBatchWorkChain":
                if planned is not None:
                    row["planned_runs"] = int(planned)
                    row["completed_runs"] = int(completed or 0)
                    row["pending_runs"] = max(0, int(planned) - int(completed or 0))
                    pending_cases += row["pending_runs"]
                    pending_known = True
                else:
                    # Pre-extras batch: unknown is not zero.
                    row["planned_runs"] = row["completed_runs"] = row["pending_runs"] = None
                out["batches"].append(row)
            elif label == "Flow3DRunCalculation":
                out["runs"].append(row)
        out["run_count"] = len(out["runs"])
        out["batch_count"] = len(out["batches"])
        # The headline number the queue view was missing: cases committed to the
        # seat that no process tally can see. None when no live batch records it.
        out["pending_batch_cases"] = pending_cases if pending_known else None
        out["runs"] = out["runs"][:10]
        out["batches"] = out["batches"][:10]
        return out

    def paused_work(self) -> dict[str, Any]:
        """Everything deliberately parked, in both places it can be parked.

        Pausing is the only way to hold work back — `seat_priority` cannot do it.
        Deference is evaluated per seat acquisition against whatever is queued at
        that moment, so it can order a queue but cannot express "stay out of the
        way until released", which is what parking a sweep means.

        Which leaves pausing, whose real hazard is being forgotten: a paused
        workchain sits indefinitely with nothing scheduled to resume it, and
        neither AiiDA's process list nor the spool's held queues put it next to
        the other. This is the one view that does, with an age so "parked
        yesterday" is visibly different from "parked in March".
        """
        import datetime as _dt

        from aiida import orm

        now = _dt.datetime.now(_dt.timezone.utc)
        rows: list[dict[str, Any]] = []
        for node in orm.QueryBuilder().append(orm.ProcessNode, project=["*"]).all(flat=True):
            # `paused` is not cleared when a process terminates, so a KILLED
            # workchain keeps reporting True forever: one batch read as "paused
            # for 38.4 hours" while its status said `Killed through verdi
            # process kill`. A finished process is not parked work, and listing
            # it as such is how the real ones get lost in the noise.
            if not getattr(node, "paused", False) or node.is_terminated:
                continue
            mtime = getattr(node, "mtime", None)
            rows.append({
                "pk": int(node.pk),
                "process_label": node.process_label or "",
                "title": node.label or None,
                "description": node.description or None,
                "paused_for_hours": round((now - mtime).total_seconds() / 3600.0, 1) if mtime else None,
                "status": node.process_status or None,
                "unfinished_child": self._running_child_pk(node),
                "planned_runs": node.base.extras.get("planned_runs", None),
                "completed_runs": node.base.extras.get("completed_runs", None),
            })
        rows.sort(key=lambda row: -(row["paused_for_hours"] or 0))

        held: list[dict[str, Any]] = []
        try:
            queue = self.flow3d_queue_status()
            held = list(queue.get("held") or [])
        except Exception:  # noqa: BLE001 - the spool is a fallback path, not a dependency
            held = []
        return {
            "paused_processes": rows,
            "paused_count": len(rows),
            "held_spool_queues": held,
            "held_task_count": sum(int(entry.get("count") or 0) for entry in held),
            "undescribed": [row["pk"] for row in rows if not row["description"]],
        }

    def flow3d_queue_status(self) -> dict[str, Any]:
        """Broker-free view of FLOW-3D work: the task spooler *and* AiiDA.

        The spool (`~/.flow3d/spool`) is the manual fallback path; AiiDA is the
        default one. Both take the same seat lock, so a view of only one of them
        can report an idle machine while the other is saturated.

        The seat probe briefly acquires and releases the flock when it is free;
        that cannot disturb a real holder.
        """
        import fcntl
        import json as _json
        import os

        from aiida_pino.console import monitor

        spool = Path(os.environ.get("F3D_SPOOL_DIR") or (Path.home() / ".flow3d" / "spool"))
        out: dict[str, Any] = {
            "spool_dir": str(spool),
            "exists": spool.is_dir(),
            "seat_held": None,
            "queued": [],
            "queued_count": 0,
            "running": [],
        }
        # A missing spool dir used to return here, skipping the AiiDA half —
        # the exact blind-spot class this view exists to close: "no spool"
        # read as "nothing anywhere" while AiiDA batches held days of work.
        if out["exists"]:
            for sub, key in (("queue", "queued"), ("running", "running")):
                rows = []
                for t in sorted((spool / sub).glob("*.task")):
                    try:
                        body = _json.loads(t.read_text())
                    except Exception:  # noqa: BLE001
                        body = {}
                    rows.append({"name": body.get("name") or t.stem, "added": body.get("added")})
                if key == "queued":
                    out["queued_count"] = len(rows)
                    out["batches"] = monitor.group_batches([r["name"] for r in rows])
                    rows = rows[:10]
                out[key] = rows
            # queue_held_* dirs are manually parked work; count them so the Monitor
            # tab shows the whole backlog, not just what will drain next.
            out["held"] = [
                {"name": d.name, "count": len(list(d.glob("*.task")))} for d in sorted(spool.glob("queue_*")) if d.is_dir()
            ]
            # Who is actually on the seat, from the kernel's lock table: the spool's
            # own `running/` entry names the task its worker started, which can be
            # (and on 2026-08-04 was) blocked behind an AiiDA run for two days.
            out["seat_holder"] = monitor.seat_holder(spool)
            lock = spool / ".seat.lock"
            if lock.exists():
                try:
                    with open(lock) as fh:
                        try:
                            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            fcntl.flock(fh, fcntl.LOCK_UN)
                            out["seat_held"] = False
                        except OSError:
                            out["seat_held"] = True
                except OSError:
                    pass
        # The AiiDA half. Reported separately rather than merged into `queued`,
        # because the two paths queue in different places and conflating them
        # would hide which one is holding the machine.
        try:
            out["aiida"] = self._aiida_flow3d_queue()
        except Exception:  # noqa: BLE001 — a broker-free view must survive a
            # profile that cannot be queried; the spool half is still useful.
            out["aiida"] = {
                "runs": [], "batches": [], "run_count": 0, "batch_count": 0,
                "pending_batch_cases": None,
            }
        # Unmaterialised batch cases count as waiting work: `completed_runs` is
        # the submit cursor, so planned - completed is exactly the cases that
        # exist only in `run_configs` — no overlap with `run_count`, which sees
        # the already-submitted children.
        out["total_waiting"] = (
            out["queued_count"]
            + out["aiida"]["run_count"]
            + int(out["aiida"].get("pending_batch_cases") or 0)
        )
        return out

    def monitor_status(self) -> dict[str, Any]:
        """Live snapshot for the Monitor tab: GPUs, CPU, active AiiDA runs with
        training progress, the gpu_spool training queue, and the FLOW-3D spool.
        Broker-free; every section degrades to a note instead of failing."""
        from aiida.orm import CalcJobNode, ProcessNode, QueryBuilder

        from aiida_pino.console import monitor
        from aiida_pino.scripts.live_tail import LIVE_TAIL_FILENAME

        gpus, gpu_note = monitor.gpu_snapshot()
        local_hostname = socket.gethostname()
        for gpu in gpus:
            gpu.setdefault("hostname", local_hostname)
        # Cards on the compute boxes. Without this the panel shows only THIS
        # machine, so a training running on another one appears to be using no
        # GPU at all -- the reading is not wrong, it is about a different
        # computer, which is worse.
        remote_gpus, remote_notes = self._remote_gpu_snapshots()
        gpus.extend(remote_gpus)
        if remote_notes:
            gpu_note = "; ".join([n for n in ([gpu_note] if gpu_note else []) + remote_notes])

        q = (
            QueryBuilder()
            .append(
                ProcessNode,
                filters={"attributes.process_state": {"in": ["created", "waiting", "running"]}},
                project=["*"],
            )
            .order_by({ProcessNode: {"ctime": "asc"}})
        )
        # A running calcjob is itself active, so the one query above already has
        # the whole family: walking `caller` up beats asking each top-level node
        # for `called_descendants`, which is a recursive graph query (7.6 s for
        # ten runs, against ~0 for the walk).
        active = [node for node in q.all(flat=True) if self._is_pino_process(node)]
        caller_pk = {node.pk: (node.caller.pk if node.caller is not None else None) for node in active}
        by_pk = {node.pk: node for node in active}

        def top_of(node):
            seen = {node.pk}
            while caller_pk.get(node.pk) is not None and caller_pk[node.pk] in by_pk:
                node = by_pk[caller_pk[node.pk]]
                if node.pk in seen:  # cycles cannot happen in a provenance graph, but do not hang on one
                    break
                seen.add(node.pk)
            return node

        calcjobs_of: dict[int, list] = {}
        # Both queue panels below read spool directories, which AiiDA-driven work
        # never enters -- a sweep submitted through Flow3DBatchWorkChain takes the
        # seat without queueing a task, and an AiiDA training holds a GPU without
        # a gpu_spool entry. Counting the active calcjobs here (no extra query --
        # `active` is already loaded) is what stops an empty spool from reading as
        # an idle machine.
        aiida_calcjobs: dict[str, int] = {}
        for node in active:
            if isinstance(node, CalcJobNode):
                calcjobs_of.setdefault(top_of(node).pk, []).append(node)
                label = node.process_label or "unknown"
                aiida_calcjobs[label] = aiida_calcjobs.get(label, 0) + 1

        workdir_runs: dict[str, dict[str, Any]] = {}
        rows: list[dict[str, Any]] = []
        for node in active:
            if caller_pk.get(node.pk) is not None and caller_pk[node.pk] in by_pk:
                continue  # children fold into their top-level parent row
            row: dict[str, Any] = {
                "pk": node.pk,
                "label": node.process_label,
                "title": node.label or None,
                "state": getattr(node.process_state, "value", None),
                "ctime": node.ctime.isoformat() if node.ctime else None,
                "progress": None,
            }
            for member in calcjobs_of.get(node.pk, []):
                workdir = member.get_remote_workdir()
                if not workdir:
                    continue
                workdir_runs[workdir] = {"run_pk": node.pk, "run_label": node.process_label, "run_title": node.label or None}
                if member.process_label == "PinoTrainCalculation":
                    # Local reads straight from the workdir; remote goes over the same
                    # transport the Logs pane uses, cached. See `_live_tail_text`.
                    tail = self._live_tail_text(member, workdir)
                    try:
                        profile = member.inputs.training_profile.get_dict()
                    except Exception:  # noqa: BLE001
                        profile = {}
                    # The beam-box MAE the trainer prints beside its step counter. A step
                    # count says a run is ADVANCING; this says whether it is advancing well,
                    # and it is already in the tail being read, so it costs nothing.
                    #
                    # The logged value is NORMALISED. A live run has no archived model to
                    # take a scale from, so the factor comes from the SHIPPED config of the
                    # name the run used -- `named_config_scale`, whose own docstring calls
                    # it the weaker source because that file can change. Its provenance
                    # string travels with the number so the UI can say where the Kelvin came
                    # from, and when no factor resolves the raw value is sent and labelled
                    # normalised rather than converted by a guess.
                    beambox = monitor.last_beambox_mae(tail)
                    bb_factor, bb_source = (None, "")
                    if beambox:
                        from . import model_metrics as _mm

                        bb_factor, bb_source = _mm.named_config_scale(
                            str(profile.get("config_name") or ""),
                            because="live training progress",
                        )
                    # How long since anything was written. A step with no recent write is a
                    # run that STOPPED at that step, not one passing through it.
                    tail_age = self._live_tail_age_s(member, workdir)
                    row["progress"] = {
                        "tail_age_s": tail_age,
                        "step": monitor.last_step(tail),
                        # The bar denominator must match what hydra resolved, or a
                        # run with an override reads 100% early (one run read 19900/20000
                        # on a 35000-step run).
                        "max_steps": effective_max_steps(profile),
                        "waiting_for_gpu": monitor.waiting_for_gpu(tail),
                        "calcjob_pk": member.pk,
                        "beambox_mae": beambox,
                        "beambox_kelvin": (
                            round(beambox["value"] * bb_factor, 1)
                            if beambox and bb_factor else None
                        ),
                        "beambox_kelvin_source": bb_source,
                    }
            rows.append(row)

        for gpu in gpus:
            for proc in gpu["procs"]:
                cwd = proc.get("cwd") or ""
                cmd = proc.get("cmd") or ""
                hit = next((v for wd, v in workdir_runs.items() if (cwd and cwd.startswith(wd)) or wd in cmd), None)
                if hit:
                    proc.update(hit)

        # Classify each active AiiDA run as running or queued so the daemon's
        # queue is visible, not just the file spools. Ground truth for "running"
        # is a live resource: a GPU process actually on a card (from nvidia-smi),
        # or a running Flow3DRunCalculation holding the seat. Everything else
        # active is queued — submitted, waiting for its resource. A training run
        # that logged `waiting for GPU seat` is queued even if half-started.
        running_gpu_pks = {p["run_pk"] for gpu in gpus for p in gpu["procs"] if p.get("run_pk")}
        # A running GPU calcjob is a "running" signal too: nvidia-smi proc
        # matching is best-effort, so its absence must not demote a real run to
        # queued. `waiting_for_gpu` (below) is what actually marks a started-but-
        # waiting job as queued.
        gpu_calcjob_running_pks = {
            top_of(node).pk for node in active
            if isinstance(node, CalcJobNode) and node.process_label in GPU_CALCJOB_LABELS
            and getattr(node.process_state, "value", None) == "running"
        }
        # The seat lock serializes solves, so at most one FLOW-3D case can be solving.
        # Counting every active Flow3DRunCalculation as "running" reported three
        # runs under one lock and an empty queue; the badge then read
        # "1 queued" over a 23-deep backlog. The kernel knows who holds the
        # flock, and the holder's working directory names the calcjob — the
        # spool's own `running/` entry does not, it named a task that had been
        # blocked behind an AiiDA run since two days earlier.
        spool_status = self.flow3d_queue_status()
        seat = spool_status.get("seat_holder")
        holder_dir = (seat or {}).get("workdir") or ""
        flow3d_running_pks = set()
        for node in active:
            if not (isinstance(node, CalcJobNode) and node.process_label == "Flow3DRunCalculation"):
                continue
            if not holder_dir:
                # No lock table to read (no spool directory, another platform):
                # fall back to the process state, which is what this used before.
                if getattr(node.process_state, "value", None) == "running":
                    flow3d_running_pks.add(top_of(node).pk)
                continue
            try:
                workdir = str(node.outputs.remote_folder.get_remote_path())
            except Exception:  # noqa: BLE001 - no remote folder yet: not on the seat
                continue
            if holder_dir == workdir or holder_dir.startswith(workdir.rstrip("/") + "/"):
                flow3d_running_pks.add(top_of(node).pk)
                seat["pk"] = top_of(node).pk
        queue = {"gpu": {"running": 0, "queued": 0}, "flow3d": {"running": 0, "queued": 0}}
        # Which boxes actually answered, so "no process for this run" can be told apart from
        # "no snapshot of that box". Only the first is evidence that a run has stopped.
        gpu_hosts_seen = {str(g.get("hostname") or "") for g in gpus if g is not None}
        for row in rows:
            on_gpu = row["pk"] in running_gpu_pks or row["pk"] in gpu_calcjob_running_pks
            progress = row.get("progress")
            if isinstance(progress, dict):
                # The AUTHORITATIVE liveness signal, and the reason the age heuristic alone
                # was not enough: a training that logs every ~66 s (an AR run at step
                # 800/32000 after 14.7 h) looks stale on any threshold short enough to catch
                # a run that finished ten minutes ago. Holding a GPU is not a heuristic.
                progress["on_gpu"] = bool(on_gpu)
                progress["gpu_snapshot_seen"] = bool(gpu_hosts_seen)
            category, phase = classify_run(
                row["label"] or "", progress,
                on_gpu=on_gpu,
                has_seat=row["pk"] in flow3d_running_pks,
            )
            row["category"], row["phase"] = category, phase
            if category in queue:
                queue[category][phase] += 1

        # One unified queue per resource, so AiiDA-submitted and spool-submitted
        # work share a single ordered list instead of the daemon queue hiding
        # behind an empty spool. AiiDA runs first, in submission (ctime) order --
        # `rows` is already ctime-ascending; then any spool-queued tasks.
        def _run_vram_mb(pk):
            return queued_vram_reservation(by_pk.get(pk))

        gpu_spool_rows = monitor.attach_seats_to_spool(
            monitor.attach_aiida_to_spool(monitor.gpu_spool_snapshot(), gpus)
        )
        gpu_queue: list[dict[str, Any]] = []
        flow3d_aiida_queue: list[dict[str, Any]] = []
        for row in rows:
            if row.get("phase") != "queued":
                continue
            entry = {"pk": row["pk"], "name": row["label"], "title": row.get("title"), "source": "aiida"}
            if row["category"] == "gpu":
                entry["vram_mb"], entry["vram_auto"] = _run_vram_mb(row["pk"])
                gpu_queue.append(entry)
            elif row["category"] == "flow3d":
                flow3d_aiida_queue.append(entry)
        for srow in gpu_spool_rows:
            for name in srow.get("queued", []):
                gpu_queue.append({"pk": None, "name": name, "source": "spool", "gpu": srow.get("gpu"), "vram_mb": None})
        for i, entry in enumerate(gpu_queue, 1):
            entry["pos"] = i

        flow3d = spool_status
        flow3d["aiida_running"] = queue["flow3d"]["running"]
        flow3d["aiida_queued"] = queue["flow3d"]["queued"]
        # Cases a live batch has not materialised yet (max_concurrent_runs=1
        # keeps N-1 of them out of every process count): without this the
        # Monitor read "nothing queued" while one batch had 299 cases —
        # ~3 days of seat time — committed. Already computed inside
        # flow3d_queue_status; lifted to the top level for the panel.
        flow3d["pending_batch_cases"] = (flow3d.get("aiida") or {}).get("pending_batch_cases")
        # Merge the AiiDA-queued FLOW-3D runs with the spool batch queue, in order.
        flow3d_queue = flow3d_aiida_queue + [
            {"pk": None, "name": b.get("batch"), "count": b.get("count"), "source": "spool"}
            for b in (flow3d.get("batches") or [])
        ]
        for i, entry in enumerate(flow3d_queue, 1):
            entry["pos"] = i
        flow3d["queue"] = flow3d_queue
        return {
            "gpus": gpus,
            "gpu_note": gpu_note,
            "cpu": monitor.cpu_snapshot(),
            "active_processes": rows,
            # gpu_spool reads the lease ledger (who booked a card, including
            # jobs still waiting for a seat) merged with the proc/run correlation
            # (which running procs belong to a tracked run). gpu_queue is the one
            # ordered waiting list the panel renders.
            "gpu_spool": gpu_spool_rows,
            "gpu_queue": gpu_queue,
            # Active calcjobs by class. Rendered as the "what is on the machine"
            # line: it shipped with nothing reading it, next to two other
            # "running" counts, and a reader could not tell which one the UI
            # believed.
            "aiida_calcjobs": aiida_calcjobs,
            # Genuinely running vs waiting, so the badge no longer counts a
            # queued job as running (the old sum over active calcjobs did).
            "aiida_gpu_running": queue["gpu"]["running"],
            "aiida_gpu_queued": queue["gpu"]["queued"],
            "flow3d": flow3d,
        }

    def _recorded_training_profile(self, artifacts) -> tuple[dict, str]:
        """The EFFECTIVE profile a finished run recorded for itself, or `({}, "")`.

        `train_summary.json` in the run directory holds the profile the training actually
        ran with -- `env`, bookings, config name, resolved hydra overrides -- which is the
        one thing a hand-passed `--training-profile` reliably gets wrong. Two defects
        found on 2026-08-27 are both this:

        - the import recorded the RAW profile while the native path records the effective
          one, so `compare-models` reported 41 false confounds between an imported model and
          a native one whose overrides were byte-identical;
        - a partial profile was accepted silently and `env` defaulted to `{}`.

        The second is the dangerous one now. Since the gas crop became the default, `env`
        carries `PINO_LAZY_CROP_Z_TOP`, which DEFINES the input geometry, and a moving-frame
        run also carries `PINO_LAZY_MOVING_REF`. A model imported without them reads as real
        and misstates what it was trained on -- exactly the malformed node the reporter
        refused to hand-construct.

        Best-effort by design: an unreadable summary returns empty and the caller's profile
        stands, because refusing an import over a missing convenience would be worse than
        the partial record it replaces.
        """
        import json as _json
        import shlex

        name = "train_summary.json"
        try:
            path = artifacts.get_remote_path()
        except Exception:  # noqa: BLE001
            return {}, ""
        # Local artifacts read straight off disk; a remote box goes over the same transport
        # the Logs pane uses.
        computer = getattr(artifacts, "computer", None)
        transport_type = getattr(computer, "transport_type", "core.local")
        raw = ""
        if transport_type == "core.local":
            try:
                raw = (Path(path) / name).read_text(encoding="utf-8")
            except OSError:
                return self._profile_from_hydra_record(Path(path))
        else:
            try:
                transport = artifacts.get_authinfo().get_transport()
                with transport:
                    retval, stdout, _ = transport.exec_command_wait(
                        f"cat {shlex.quote(str(PurePosixPath(path) / name))}"
                    )
                if retval:
                    return {}, ""
                raw = str(stdout)
            except Exception:  # noqa: BLE001
                return {}, ""
        try:
            recorded = (_json.loads(raw) or {}).get("profile") or {}
        except ValueError:
            return {}, ""
        if not isinstance(recorded, dict) or not recorded:
            return {}, ""
        return recorded, str(PurePosixPath(path) / name)

    @staticmethod
    def _profile_from_hydra_record(run_dir: Path) -> tuple[dict, str]:
        """A profile for a run trained outside aiida-pino, from the `.hydra` record it carries.

        Such a run has no `train_summary.json`, and the import used to fall back to defaults:
        another config name, no overrides, 6000 steps, wandb online. The evaluation builds on
        that profile, so a model imported from a plain run scored a different beam box and
        logged to wandb (2026-10-08, registering cluster retrains). The record gives the
        config name, the overrides and the step count. Overrides whose value is an absolute
        path (data, network and run directories) say where the run happened, not how it was
        trained, and are left out. wandb stays offline.
        """
        import yaml as _yaml

        hydra_dir = run_dir / ".hydra"
        try:
            config = _yaml.safe_load((hydra_dir / "config.yaml").read_text(encoding="utf-8")) or {}
            hydra = _yaml.safe_load((hydra_dir / "hydra.yaml").read_text(encoding="utf-8")) or {}
            overrides = _yaml.safe_load((hydra_dir / "overrides.yaml").read_text(encoding="utf-8")) or []
        except (OSError, ValueError):
            return {}, ""
        kept = [str(o) for o in overrides
                if not str(o).split("=", 1)[-1].strip().strip("'\"").startswith("/")]
        profile: dict[str, Any] = {"hydra_overrides": kept, "wandb_offline": True}
        config_name = ((hydra.get("hydra") or {}).get("job") or {}).get("config_name")
        if config_name:
            profile["config_name"] = str(config_name)
        max_steps = (config.get("training") or {}).get("max_steps")
        if max_steps:
            profile["max_steps"] = int(max_steps)
        return profile, str(hydra_dir)

    def submit_import_model(self, req) -> dict[str, Any]:
        from aiida import orm
        from aiida.orm import Dict, Str

        from aiida_pino.data import PinoModelData
        from aiida_pino.utils import get_or_create_remote_data, resolve_artifacts_dir

        code = self._code(req.python_code)
        source = _require_dataset_collection(self._load_node(req.training_source_pk), "training_source")
        if req.artifacts_pk:
            artifacts = self._load_node(req.artifacts_pk)
        elif req.artifacts_path:
            artifacts, _ = get_or_create_remote_data(
                resolve_artifacts_dir(req.artifacts_path),
                code.computer,
                kind="training_artifacts",
                description="imported via console",
            )
        else:
            raise InvalidRequestError("one of artifacts_pk or artifacts_path is required")
        # The run's OWN record underneath, the caller's profile on top. A caller who passes
        # nothing now gets the effective profile the training ran with -- crucially its
        # `env` -- instead of a defaulted one; a caller who passes a key still wins.
        recorded, recorded_from = self._recorded_training_profile(artifacts)
        merged = {**recorded, **(req.training_profile or {})}
        training = self._validated_training_profile(
            merged, split_mode="standard", for_import=True
        )
        if recorded_from and not (req.training_profile or {}).get("env") and recorded.get("env"):
            print(
                f"[import-model] adopted the profile recorded in {recorded_from}, including "
                f"env={sorted(recorded['env'])}; pass --training-profile to override.",
                flush=True,
            )
        derivation = self._validated_derivation_profile(req.derivation_profile)
        inputs: dict[str, Any] = {
            "training_artifacts": artifacts,
            "training_source": source,
            "training_profile": Dict(dict=training),
        }
        if getattr(req, "validation_source_pk", None):
            inputs["validation_source"] = _require_dataset_collection(
                self._load_node(req.validation_source_pk), "validation_source"
            )
        if req.model_name:
            inputs["model_name"] = Str(req.model_name)
        if derivation is not None:
            inputs["derivation_profile"] = Dict(dict=derivation)

        # Nothing stopped importing the same run twice into two model nodes
        # (2026-08-10 friction, item 8). Look for a model already built from
        # these artifacts and report it; --dry-run stops here either way.
        existing = [
            {"pk": pk, "label": label}
            for pk, label in orm.QueryBuilder().append(
                orm.Node, filters={"id": artifacts.pk}, tag="a",
            ).append(
                orm.ProcessNode, with_incoming="a", tag="proc",
            ).append(
                PinoModelData, with_incoming="proc", project=["id", "label"],
            ).all()
        ]
        if getattr(req, "dry_run", False):
            return {
                "dry_run": True,
                "would_import": {
                    "artifacts_pk": artifacts.pk,
                    "artifacts_path": getattr(artifacts, "get_remote_path", lambda: None)()
                    if hasattr(artifacts, "get_remote_path") else None,
                    "training_source_pk": source.pk,
                    "validation_source_pk": getattr(req, "validation_source_pk", None),
                    "model_name": req.model_name,
                    "derivation_profile_given": derivation is not None,
                },
                "already_imported": existing,
                "note": "nothing was submitted"
                + ("; these artifacts already have a model node" if existing else ""),
            }
        result = self._submit("pino.import_model", **inputs)
        if existing:
            result.setdefault("notes", []).append(
                f"these artifacts already produced model(s) {[m['pk'] for m in existing]}; "
                "this import creates another node"
            )
        return result

    def submit_derive(self, req) -> dict[str, Any]:
        from aiida.orm import Dict

        from aiida_pino.data import PinoDatasetCollectionData

        dataset = self._load_node(req.dataset_pk, PinoDatasetCollectionData)
        derivation = self._validated_derivation_profile(req.derivation_profile) or {}
        result = self._submit("pino.derive_dataset", dataset=dataset, derivation_profile=Dict(dict=derivation))
        note = self._superseded_note(dataset, "source")
        if note:
            result["notes"] = [note]
        return result

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _is_pino_process(node) -> bool:
        ptype = (node.process_type or "").lower()
        label = (node.process_label or "").lower()
        if "pino" in ptype or "aiida_pino" in ptype or "pino" in label:
            return True
        # A calcfunction run from a script is `__main__.<fn>`, with "pino"
        # nowhere in it: 47 provenance nodes were hidden by the default filter,
        # including the 19 `select_acquisitions` records of which cases each
        # self-improvement round chose — exactly the provenance a reviewer asks
        # for. If it produced or consumed a pino node, it is ours.
        try:
            for entry in list(node.base.links.get_incoming().all()) + list(node.base.links.get_outgoing().all()):
                if "pino" in type(entry.node).__name__.lower():
                    return True
        except Exception:  # noqa: BLE001 - link traversal is best effort
            pass
        return False

    @staticmethod
    def _matches(search: str, *fields: Any) -> bool:
        s = search.strip().lower()
        if not s:
            return True
        return any(s in str(f).lower() for f in fields if f is not None)
