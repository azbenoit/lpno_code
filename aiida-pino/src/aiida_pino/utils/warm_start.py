"""Inherit a warm-start donor's architecture, so a moved default cannot silently reset it.

`physicsnemo/sym/trainer.py` loads a warm-start checkpoint under a BARE `except`: a
state-dict mismatch logs one red line and the run then **trains from random init to
completion**, producing a perfectly normal-looking curve. Arm 65831 (6000 steps) was lost
that way on 2026-08-23 and was only caught because someone asked whether it was training
from scratch.

Triggering it needs no mistake by anyone. That arm was submitted 15 minutes BEFORE the
commits that moved `num_graph_variants` 1 -> 8, then queued overnight and resolved its
config at RUN time -- so it built an 8-variant model from a 1-variant donor. Neither run
was wrong; the default moved between them.

The donor carries the answer: `.hydra/config.yaml` in its repository is the fully composed
config its own run used. Inheriting `arch.*` from it makes the mismatch class impossible
rather than merely detectable, which is the difference between "the submitter must know to
pin six keys" and "it works".

**Known cost: this changes the VRAM footprint key.** `vram_footprint_key` hashes
`hydra_overrides`, so appending inherited `arch.*` pins produces a new key even when the
donor's architecture is IDENTICAL to today's default and nothing physical has changed. The
run therefore misses any previously measured peak for that footprint and falls back to a
whole-card smoke (for a training) or the p90 (for an eval). Measured from the other side:
pinning `arch.rigno.processor_steps` on an eval lost a model's
measured 3800 MiB peak and fell back to a p90 of 9248.

That is a real cost and it is accepted deliberately, because the alternative is worse: the
key must change when the architecture genuinely differs, or a depth-18 run gets sized from
a depth-1 peak and OOMs. Excluding these overrides from the key would reintroduce exactly
that. The clean fix is for the key to hash the RESOLVED architecture rather than the
override text, which needs hydra composition (see the roadmap in `docs/developer_notes.md`). Until then,
hand-setting `gpu_vram_mb` on a warm start skips the smoke entirely.

PLAIN overrides, never `+`/`++`: pino's `remap_plus_args_to_custom` rewrites ANY arg
starting with `+` into the `custom.` namespace, so `++arch.rigno.processor_steps=8` sets
`custom.arch.rigno.processor_steps` -- a key nothing reads -- while the real architecture
keeps whatever today's config says. That is the same trap that made the VRAM smoke's
cadence clamp inert (2026-08-22).
"""

from __future__ import annotations

from typing import Any

#: Inherited from the donor. `arch` is the architecture proper; `custom.FNO_input_keys`
#: decides the input channel count, which changes the first layer's shape and so belongs
#: to the same must-match set.
INHERITED_ROOTS = ("arch",)
INHERITED_LEAVES = ("custom.FNO_input_keys",)

#: Never inherited: filesystem paths. `arch.rigno.cache_dir` sits inside the `arch` block
#: but cannot change a state dict, and carrying a donor's path into a new run is a hazard
#: rather than a fidelity gain -- a donor that recorded an absolute path from another
#: machine would point this run at a directory that does not exist here. Matched by suffix
#: so a future `*_dir` under `arch` is excluded by default, which is the safe direction:
#: a path wrongly inherited breaks a run, a path wrongly omitted changes nothing.
EXCLUDED_SUFFIXES = ("_dir",)


def _flatten(node: Any, prefix: str) -> dict[str, Any]:
    """`{'arch.rigno.processor_steps': 8, ...}` from a nested mapping."""
    if not isinstance(node, dict):
        return {prefix: node}
    out: dict[str, Any] = {}
    for key, value in node.items():
        out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return out


def _select(mapping: Any, dotted: str) -> Any:
    current = mapping
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def donor_archived_config(node) -> dict[str, Any] | None:
    """The donor's composed `.hydra/config.yaml`, or None if it archived none.

    Best-effort by construction: a donor without one simply cannot be inherited from, and
    that must not block the submission -- it is the situation every warm start was in
    before this existed.
    """
    try:
        import yaml

        text = node.base.repository.get_object_content(".hydra/config.yaml")
    except Exception:  # noqa: BLE001 - FolderData/RemoteData donors, older models, all fine
        return None
    try:
        loaded = yaml.safe_load(text)
    except Exception:  # noqa: BLE001
        return None
    return loaded if isinstance(loaded, dict) else None


#: Characters that stop a bare Hydra override from meaning the string itself. Measured
#: against Hydra's own parser, not guessed: a comma makes `k=x, y, z` a five-value
#: ChoiceSweep (a MULTIRUN sweep, which is why the failure reads "ambiguous value" rather
#: than a syntax error), brackets and braces make it a list or dict, and the rest are parse
#: errors. Everything else -- `a:b`, `a/b`, `a-b`, `a.b`, `a_b`, a backslash -- round-trips
#: bare and is left exactly as it was.
_HYDRA_BREAKS_BARE = frozenset(",[]{}='\"()")


def hydra_value(value: Any) -> str:
    """Hydra-literal form. Lists must not carry spaces or the shell splits the override.

    A string that would not survive the round trip is QUOTED rather than split. physicsnemo
    archives `arch.fno.input_keys` as the string `'x, y, z, V_scan, P'`, and
    emitting it bare made every FNO warm start die at step 0 -- after taking a GPU seat and
    loading the donor weights, so the cost was a seat and a queue wait (2026-09-03). Splitting on commas would have fixed that one value
    by RETYPING it, which silently corrupts any comma-bearing string that is not a list.

    Scalar-literal strings (`"true"`, `"3"`, `"null"`) also come back as bool/int/None, but
    they yield a scalar the target coerces and re-typing them would change overrides that
    work today, so they are deliberately left bare.

    The command reaches `subprocess.Popen` as an argv list with no shell, so these quotes
    are read by Hydra rather than eaten by a shell.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(hydra_value(item) for item in value) + "]"
    if value is None:
        return "null"
    if isinstance(value, str) and any(char in _HYDRA_BREAKS_BARE for char in value):
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    return str(value)


def architecture_overrides_from_donor(donor, existing_overrides) -> tuple[list[str], list[str]]:
    """(overrides to append, human-readable conflicts the caller already set differently).

    An explicit caller override WINS and is never rewritten -- deliberately deviating from
    the donor is legitimate (widening a channel set, say) -- but it is returned as a
    conflict so the caller can say so out loud. Silent inheritance is its own trap.
    """
    config = donor_archived_config(donor)
    if not config:
        return [], []

    wanted: dict[str, Any] = {}
    for root in INHERITED_ROOTS:
        subtree = config.get(root)
        if isinstance(subtree, dict):
            wanted.update(_flatten(subtree, root))
    for leaf in INHERITED_LEAVES:
        value = _select(config, leaf)
        if value is not None:
            wanted[leaf] = value

    already: dict[str, str] = {}
    for item in existing_overrides or []:
        text = str(item)
        key, sep, value = text.partition("=")
        if sep:
            already[key.strip().lstrip("+~")] = value.strip()

    additions: list[str] = []
    conflicts: list[str] = []
    for key in sorted(wanted):
        if key.endswith(EXCLUDED_SUFFIXES):
            continue
        rendered = hydra_value(wanted[key])
        if key in already:
            if already[key] != rendered:
                conflicts.append(f"{key}={already[key]} (donor has {rendered})")
            continue
        additions.append(f"{key}={rendered}")
    return additions, conflicts
