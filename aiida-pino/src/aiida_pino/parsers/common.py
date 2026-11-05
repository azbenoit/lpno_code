"""Common parser helpers."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

from aiida import orm
from aiida.common import exceptions
from aiida.parsers import Parser

from aiida_pino.utils import parse_stdout_metric_dicts


def stamp_performance_extras(node, stdout: str | None) -> None:
    """Graph size and throughput, from the stdout AiiDA just retrieved.

    The GAOT graph line — nodes, encoder/decoder edges, the thing that decides
    memory — was believed to exist only in the calcjob's scratch live-tail log,
    which is deleted with the workdir. It is on stdout, which is retrieved and
    kept: present in 23 of 151 retained training stdouts. Recording it here
    means "how big was the graph, how fast did it run" survives the workdir.
    """
    from aiida_pino.utils.run_metrics import parse_graph_size, parse_throughput

    if not stdout:
        return
    for key, value in {**parse_graph_size(stdout), **parse_throughput(stdout)}.items():
        if node.base.extras.get(key, None) is None:
            node.base.extras.set(key, value)


def stamp_progress_extras(node, stdout: str | None) -> None:
    """How far the run actually got, from its own stdout.

    `exit_status 300` says "the step failed"; it does not distinguish a job that
    died on step 1 (out of memory, bad config -- nothing was learned) from one
    that trained for hours and fell over near the end. Measured 2026-08-10: 179
    processes carry exit 300 and NONE carried any step count, so telling those
    two apart meant reading `[step: N]` lines out of live-tail logs by hand --
    logs that live in the scratch workdir and are deleted with it.

    Stamped from the retrieved stdout, which survives, and stamped on the
    FAILURE path too: a run with no train_summary.json is exactly the case where
    this is the only evidence left.
    """
    import re

    if not stdout:
        return
    steps = re.findall(r"\[step: *(\d+)", stdout)
    if not steps:
        # Distinguish "produced no step line" from "never asked": a job that
        # died before its first step is the headline case, not a blank.
        if node.base.extras.get("max_step_reached", None) is None:
            node.base.extras.set("max_step_reached", 0)
        return
    node.base.extras.set("max_step_reached", int(steps[-1]))
    node.base.extras.set("step_lines_logged", len(steps))


#: Ordered because the first match wins and the specific must beat the generic:
#: an OOM traceback also contains a "Traceback", and a disk-full error surfaces
#: as an OSError that a generic exception rule would swallow. Every pattern here
#: was taken from a real failed run's stdout on this profile (2026-08-10), not
#: from what a failure might plausibly look like.
FAILURE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("oom_gpu", r"CUDA out of memory|torch\.cuda\.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED"),
    ("oom_host", r"MemoryError|Out of memory: Killed|Killed process \d+"),
    ("disk_full", r"No space left on device|Errno 28"),
    ("waiting_for_resources", r"\[gpu-spool\] waiting for .*MiB"),
    ("config_error", r"hydra\.errors|ConfigAttributeError|Error executing job with overrides"),
    ("import_error", r"ModuleNotFoundError|ImportError"),
    ("nan_divergence", r"loss is nan|loss is inf|NaN detected"),
    ("python_exception", r"^Traceback \(most recent call last\)"),
)


def classify_failure(stdout: str | None) -> str:
    """Which KIND of failure this was, from the run's own output.

    `exit_status 300` is uniform across a config crash, an OOM and a stall-kill,
    so diagnosis always meant opening the workdir's live-tail file -- a file
    that is deleted with the workdir, and reached only through
    node -> calcjob -> remote workdir -> log (2026-08-10 friction). The class is
    decided once, at parse time, from stdout that AiiDA retains.

    Returns "no_output" when nothing was captured and "unclassified" when output
    exists but matches no known pattern -- deliberately distinct, because the
    first is a retrieval problem and the second is a gap in this table.
    """
    import re

    if not stdout or not stdout.strip():
        return "no_output"
    clean = re.sub(r"\x1b\[[0-9;]*m", "", stdout)
    for name, pattern in FAILURE_PATTERNS:
        if re.search(pattern, clean, re.IGNORECASE | re.MULTILINE):
            return name
    return "unclassified"


def stamp_failure_class(node, stdout: str | None) -> None:
    """Record the failure kind on the node, so triage is a query."""
    node.base.extras.set("failure_class", classify_failure(stdout))


def stamp_runtime_extras(node, summary: dict | None) -> None:
    """Write `runtime_extras_from_summary` onto the node, never overwriting.

    Split from the mapping itself so the mapping can be tested without a stored
    node: what belongs in an extra is a decision worth pinning, and it was
    previously only reachable through a parser run.
    """
    for key, value in runtime_extras_from_summary(summary).items():
        if node.base.extras.get(key, None) is None:
            node.base.extras.set(key, value)


def runtime_extras_from_summary(summary: dict | None) -> dict:
    """Promote the run summary's compute-cost aggregates to queryable extras.

    The summaries carry full detail (adapter `runtime` block with the software/
    hardware identity, and for FLOW-3D the solver's own `manifest.run` block),
    but a Dict output cannot answer "total solver CPU-hours this month" without
    loading every node. The few aggregate numbers land as calcjob extras — the
    same pattern that makes `gpu_vram_peak_mb` drive automatic reservations.
    Existing extras are never overwritten, so a re-parse or backfill is safe.
    """
    summary = summary or {}
    extras: dict[str, object] = {}
    runtime = summary.get("runtime") or {}
    if runtime.get("wall_seconds") is not None:
        extras["runtime_wall_seconds"] = float(runtime["wall_seconds"])
    cpu = float(runtime.get("cpu_user_seconds") or 0) + float(runtime.get("cpu_system_seconds") or 0)
    if cpu:
        extras["runtime_cpu_seconds"] = round(cpu, 2)
    if runtime.get("max_rss_mb"):
        extras["runtime_max_rss_mb"] = int(runtime["max_rss_mb"])
    # The contention witness, flattened. A wall clock is only comparable with
    # another wall clock measured under similar load, and `solo_run` is the
    # filter that makes that possible: no other GPU busy and load below one per
    # core at BOTH ends of the run. Without it every timing average silently
    # mixes an idle box with one running two trainings and a solve.
    start = runtime.get("contention_start") or {}
    end = runtime.get("contention_end") or {}
    load_start = (start.get("loadavg") or [None])[0]
    load_end = (end.get("loadavg") or [None])[0]
    if load_start is not None:
        extras["loadavg_start"] = float(load_start)
    if load_end is not None:
        extras["loadavg_end"] = float(load_end)
    for key, snapshot in (("gpu_busy_start", start), ("gpu_busy_end", end)):
        if snapshot.get("gpu_busy") is not None:
            extras[key] = int(snapshot["gpu_busy"])
    cores = start.get("cpu_count") or end.get("cpu_count")
    if load_start is not None and load_end is not None and cores:
        quiet = max(load_start, load_end) <= cores
        others = max(int(start.get("gpu_busy") or 0), int(end.get("gpu_busy") or 0))
        extras["solo_run"] = bool(quiet and others <= 1)
    environment = runtime.get("environment") or {}
    for key in ("hostname", "gpu_name"):
        if environment.get(key):
            extras[f"env_{key}"] = environment[key]
    # Which numerical regime produced these numbers. An extra rather than a
    # value inside the summary Dict: a Dict is reachable by someone who already
    # suspects there is a question, an extra is reachable by someone who does
    # not. `source` distinguishes this — observed on the box at capture time —
    # from a value back-stamped later by inference.
    numerics = environment.get("numerics") or {}
    if numerics.get("numerical_regime"):
        extras["numerical_regime"] = str(numerics["numerical_regime"])
        extras["numerical_regime_source"] = "captured"
    peak = summary.get("gpu_vram_peak_mb")
    if isinstance(peak, int) and peak > 0:
        extras["gpu_vram_peak_mb"] = peak
    # FLOW-3D: the solver reports its own clock (excludes seat wait and export).
    run_block = (summary.get("manifest") or {}).get("run") or {}
    if run_block.get("elapsed_seconds"):
        extras["solver_elapsed_seconds"] = float(run_block["elapsed_seconds"])
    if run_block.get("cpu_seconds"):
        extras["solver_cpu_seconds"] = float(run_block["cpu_seconds"])
    cores = ((summary.get("manifest") or {}).get("parallel") or {}).get("cores")
    if cores:
        extras["solver_cores"] = int(cores)
    # How much work the run actually did, on the calcjob itself. This is the
    # contention-INDEPENDENT half of timing: seconds-per-cycle absorbs a solve
    # that simply had more to do, where raw wall clock cannot. The runner writes
    # `cycles` as a string, so coerce.
    try:
        cycles = int(str(run_block.get("cycles")).strip())
    except (TypeError, ValueError):
        cycles = 0
    if cycles > 0:
        extras["cycles"] = cycles
    steps = summary.get("steps_trained")
    if isinstance(steps, int) and steps > 0:
        extras["steps_trained"] = steps
    # Which commercial solver build produced this. Parsed from the run's own
    # runhyd.log banner; absent for prepare-only runs and for anything that died
    # before hydr3d started.
    solver = summary.get("solver") or {}
    if solver.get("version"):
        extras["solver_version"] = str(solver["version"])
    if solver.get("build"):
        extras["solver_build"] = str(solver["build"])
    return extras


class FolderResultParser(Parser):
    """Base parser that extracts JSON summaries and subdirectories into outputs."""

    folder_label: str = ""
    folder_subpath: str = ""
    summary_label: str = ""
    summary_filename: str = ""

    def read_json(self, filename: str, *, required: bool = True) -> dict | None:
        """Read a JSON file from the retrieved folder.

        With ``required=False`` a missing or corrupt file returns ``None`` instead of raising.
        """
        try:
            with self.retrieved.base.repository.open(filename, "r") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            if required:
                raise exceptions.ParsingError(f"Failed to read parser JSON file `{filename}`.") from exc
            return None

    def read_metrics_with_stdout_fallback(self) -> dict:
        """Read `metrics.json`, falling back to metric dicts printed on stdout, else `{}`."""
        metrics_payload = self.read_json("metrics.json", required=False)
        if not metrics_payload:
            try:
                with self.retrieved.base.repository.open(self.node.get_option("output_filename"), "r") as handle:
                    metrics_payload = parse_stdout_metric_dicts(handle.read())
            except OSError:
                metrics_payload = {}
        return metrics_payload

    def _root_exclude_names(self) -> set[str]:
        """Return root-level retrieved files that should not be folded into artifact folders."""
        excluded = {self.summary_filename}
        output_filename = self.node.get_option("output_filename")
        if output_filename:
            excluded.add(output_filename)
        scheduler_stdout = self.node.get_option("scheduler_stdout")
        if scheduler_stdout:
            excluded.add(scheduler_stdout)
        scheduler_stderr = self.node.get_option("scheduler_stderr")
        if scheduler_stderr:
            excluded.add(scheduler_stderr)
        return excluded

    def _build_flattened_folder_output(self) -> orm.FolderData:
        """Build a `FolderData` node from retrieved root contents when the folder was flattened on retrieval."""
        root_entries = self.retrieved.base.repository.list_object_names()
        excluded = self._root_exclude_names()
        payload_entries = [name for name in root_entries if name not in excluded]
        if not payload_entries:
            raise exceptions.ParsingError(
                f"Retrieved output folder `{self.folder_subpath}` is missing and no fallback payload exists in root."
            )

        with tempfile.TemporaryDirectory() as tmp_path:
            tmpdir = Path(tmp_path)
            self.retrieved.base.repository.copy_tree(tmpdir)
            for name in excluded:
                path = tmpdir / name
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            return orm.FolderData(tree=str(tmpdir))

    def build_folder_output(self, subpath: str) -> orm.FolderData:
        """Build a `FolderData` node from a subdirectory in `retrieved`."""
        try:
            with self.retrieved.base.repository.as_path(subpath) as path:
                if not Path(path).exists():
                    raise exceptions.ParsingError(f"Retrieved output folder `{subpath}` is missing.")
                return orm.FolderData(tree=str(path))
        except FileNotFoundError:
            return self._build_flattened_folder_output()

    def parse(self, **kwargs):
        # Read stdout FIRST, and stamp what it proves, before any early return.
        # A job that dies without a summary is precisely the one whose progress
        # nothing else records (2026-08-10 friction: 179 exit-300 processes, 0
        # step counts).
        stdout_text: str | None = None
        try:
            retrieved = self.retrieved
            for name in retrieved.base.repository.list_object_names():
                if name.endswith(".out"):
                    stdout_text = retrieved.base.repository.get_object_content(name)
                    break
        except Exception:  # noqa: BLE001 - no retrieved folder: nothing to read
            stdout_text = None
        if stdout_text is not None:
            # Step progress only where steps exist. An eval job logs none, so
            # stamping it 0 would read as "never got going" for a job that
            # completed normally.
            if getattr(self, "tracks_steps", False):
                stamp_progress_extras(self.node, stdout_text)
            stamp_performance_extras(self.node, stdout_text)
            stamp_failure_class(self.node, stdout_text)

        summary_payload: dict | None = None
        if self.summary_label:
            summary_payload = self.read_json(self.summary_filename, required=False)
            if summary_payload is None:
                return self.exit_codes.ERROR_MISSING_SUMMARY
            stamp_runtime_extras(self.node, summary_payload)
        if self.folder_label:
            try:
                folder = self.build_folder_output(self.folder_subpath)
            except exceptions.ParsingError:
                return self.exit_codes.ERROR_MISSING_ARTIFACTS
            folder.label = self.folder_label
            folder.base.extras.set("aiida_pino_kind", self.folder_label)
            self.out(self.folder_label, folder)
        if self.summary_label:
            summary = orm.Dict(dict=summary_payload)
            summary.label = self.summary_label
            self.out(self.summary_label, summary)
