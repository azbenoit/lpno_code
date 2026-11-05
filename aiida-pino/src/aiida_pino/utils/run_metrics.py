"""Performance metadata as a first-class record, not a log file that gets deleted.

AiiDA holds provenance well and performance not at all. Every question of the
form *how big / how fast / how much memory* needed a log in the calcjob's
scratch workdir, and for finished runs those are gone. Reported 2026-08-06:

* the GAOT graph size — nodes, encoder/decoder edges, the thing that decides
  memory — was believed to live only in the live-tail log. It does not: it is
  printed to stdout, which AiiDA **retrieves and keeps**, and it is present in
  23 of 151 retained training stdouts. Recoverable for every GAOT run ever done.
* throughput (ms/iter) was recorded nowhere at all, so "how long does a
  full-resolution GAOT run take" could not be answered from the database.
* getting from a model to its calcjob took four hops: model -> creator ->
  `.caller` -> `called_descendants`.

This module turns all three into flat extras on the model node. The parsing is
deliberately tolerant: a line that does not match yields nothing rather than a
wrong number, because a wrong performance figure is worse than a missing one.
"""

from __future__ import annotations

import re
from typing import Any

#: `GAOT graph: grid (150, 100, 40) (600000 nodes) <-> latent (50, 34, 40)
#: (68000 tokens); encoder edges 2400000, decoder edges 4800000`
_GRAPH = re.compile(
    r"GAOT graph:\s*grid\s*\(([^)]*)\)\s*\((\d+)\s*nodes\)\s*<->\s*latent\s*\(([^)]*)\)\s*"
    r"\((\d+)\s*tokens\);\s*encoder edges\s*(\d+),\s*decoder edges\s*(\d+)"
)

#: `[14:33:42] - [step:      11000] loss: ...`
_STEP = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\][^\[]*\[step:\s*(\d+)\]")


def _shape(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out


def parse_graph_size(stdout: str | None) -> dict[str, Any]:
    """Graph geometry from the line the architecture prints when it builds one."""
    match = _GRAPH.search(stdout or "")
    if match is None:
        return {}
    grid, nodes, latent, tokens, enc, dec = match.groups()
    return {
        "graph_grid": _shape(grid),
        "graph_nodes": int(nodes),
        "graph_latent": _shape(latent),
        "graph_latent_tokens": int(tokens),
        "graph_encoder_edges": int(enc),
        "graph_decoder_edges": int(dec),
    }


def parse_throughput(stdout: str | None) -> dict[str, Any]:
    """Milliseconds per training step, from the timestamps already in stdout.

    Measured across the widest pair of step lines, so a slow first step (graph
    build, dataset staging, CUDA warm-up) is diluted rather than counted as the
    steady rate. Times are wall-clock HH:MM:SS, so a run crossing midnight is
    corrected by assuming no single gap exceeds 24 h.
    """
    marks = [
        (int(h) * 3600 + int(m) * 60 + int(s), int(step))
        for h, m, s, step in _STEP.findall(stdout or "")
    ]
    if len(marks) < 2:
        return {}
    first, last = marks[0], marks[-1]
    seconds = last[0] - first[0]
    if seconds < 0:
        seconds += 24 * 3600
    steps = last[1] - first[1]
    if steps <= 0 or seconds <= 0:
        return {}
    return {
        "ms_per_iter": round(1000.0 * seconds / steps, 2),
        "throughput_sampled_steps": steps,
    }


def calcjob_pointer(model) -> dict[str, Any]:
    """The calcjob that produced this model, without the four-hop walk."""
    # The model is CREATED by the `materialize_model_node` calcfunction; the
    # artifacts are an INPUT to that calcfunction, not to the model. Walking the
    # model's own incoming links finds the calcfunction and stops.
    maker = getattr(model, "creator", None)
    artifacts = None
    if maker is not None:
        try:
            artifacts = maker.inputs.training_artifacts
        except Exception:  # noqa: BLE001 - fall back to any FolderData input
            for node in maker.base.links.get_incoming().all_nodes():
                if node.__class__.__name__ == "FolderData":
                    artifacts = node
                    break
    creator = getattr(artifacts, "creator", None) if artifacts is not None else None
    if creator is None:
        return {}
    out = {"calcjob_pk": int(creator.pk), "calcjob_uuid": str(creator.uuid)}
    workdir = creator.base.attributes.get("remote_workdir", None)
    if workdir:
        out["calcjob_workdir"] = str(workdir)
    return out


def performance_extras(train_summary: dict | None, stdout: str | None = None) -> dict[str, Any]:
    """Everything about cost that the run already measured, flattened.

    `train_summary` is the durable half — the adapter writes it into the node —
    and `stdout` is what AiiDA retrieved. Neither is required.
    """
    summary = train_summary or {}
    runtime = summary.get("runtime") or {}
    extras: dict[str, Any] = {}

    for key, source in (
        ("wall_seconds", runtime.get("wall_seconds")),
        ("cpu_user_seconds", runtime.get("cpu_user_seconds")),
        ("gpu_vram_peak_mb", summary.get("gpu_vram_peak_mb")),
        ("host_ram_peak_mb", summary.get("host_ram_peak_mb") or runtime.get("max_rss_mb")),
    ):
        if isinstance(source, (int, float)) and not isinstance(source, bool) and source > 0:
            extras[key] = float(source) if key.endswith("seconds") else int(source)

    steps = summary.get("steps_trained")
    wall = extras.get("wall_seconds")
    if isinstance(steps, int) and steps > 0 and wall:
        # From the summary, which is durable; parse_throughput refines it from
        # stdout when that is still around.
        extras.setdefault("ms_per_iter", round(1000.0 * wall / steps, 2))

    extras.update(parse_graph_size(stdout))
    extras.update(parse_throughput(stdout))
    return extras
