"""CLI remapping helpers for routing ``+...`` overrides into ``custom.*``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import yaml


HYDRA_CONFIG_DIR = Path(__file__).resolve().parents[1] / "conf" / "hydra"

#: Top-level config sections. A `+`/`++` override naming one of these is almost
#: certainly meant for the real key, not for a `custom.` twin that nothing reads:
#: `++arch.rigno.processor_steps=8` silently became
#: `++custom.arch.rigno.processor_steps=8` and left the depth at its default,
#: which cost four RIGNO attempts (66684/66690 and two before) -- they trained at
#: depth 18 and died with CUDA OOM against a booking measured at depth 8, so the
#: failure read as a resource problem. Warn rather than refuse: the remap is
#: long-standing behaviour and someone may rely on it.
_CONFIG_TOP_LEVEL_SECTIONS = (
    "arch", "training", "optimizer", "scheduler", "loss", "stop_criterion",
)


def _extract_config_name(args: Sequence[str], default: str) -> str:
    """Extract ``--config-name`` from CLI args, falling back to ``default``."""

    for idx, arg in enumerate(args):
        if arg == "--config-name" and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith("--config-name="):
            return arg.split("=", 1)[1]
    return default


def _collect_declared_paths(node: Any, prefix: str, out: set[str]) -> None:
    """Collect dot-paths for declared custom keys (including nested dict paths)."""

    if prefix:
        out.add(prefix)
    if isinstance(node, dict):
        for key, value in node.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _collect_declared_paths(value, child_prefix, out)


def _load_declared_custom_paths(config_name: str) -> set[str]:
    """Load declared ``custom`` key paths from the active Hydra YAML file."""

    normalized_name = config_name[:-5] if config_name.endswith(".yaml") else config_name
    config_path = HYDRA_CONFIG_DIR / f"{normalized_name}.yaml"
    if not config_path.is_file():
        return set()

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    custom_cfg = cfg.get("custom", {})
    paths: set[str] = set()
    _collect_declared_paths(custom_cfg, "custom", paths)
    return paths


def remap_plus_args_to_custom(
    argv: Sequence[str] | None = None,
    default_config_name: str = "config_NS_energy_PINO",
) -> list[str]:
    """Remap ``+foo=bar`` to ``custom.foo=bar`` or ``+custom.foo=bar``.

    Rule:
    - if ``custom.foo`` is declared in active Hydra YAML -> ``custom.foo=bar`` (no leading ``+``)
    - otherwise -> ``+custom.foo=bar``

    ``++foo=bar`` (Hydra force-add: set it whatever the config says) keeps its
    ``++`` and skips the declared-path lookup, which that form does not need.
    Stripping only ONE ``+`` left the key as ``+custom.seed``, which fails the
    ``custom.`` test and was re-qualified into ``+custom.+custom.seed=42`` --
    a lexer error that killed every aiida-pino training on 2026-08-10, once
    that repo started making the resolved seed explicit as ``++custom.seed=N``.
    """

    args = list(sys.argv[1:] if argv is None else argv)
    config_name = _extract_config_name(args, default=default_config_name)
    declared_custom_paths = _load_declared_custom_paths(config_name)

    remapped: list[str] = []
    for arg in args:
        if not arg.startswith("+") or "=" not in arg:
            remapped.append(arg)
            continue

        prefix = "++" if arg.startswith("++") else "+"
        key, value = arg[len(prefix):].split("=", 1)
        if not key:
            remapped.append(arg)
            continue

        target_key = key if key.startswith("custom.") else f"custom.{key}"
        if target_key != key and key.split(".", 1)[0] in _CONFIG_TOP_LEVEL_SECTIONS:
            print(
                f"WARNING: '{prefix}{key}={value}' was rewritten to "
                f"'{prefix}{target_key}={value}'. '{key.split('.', 1)[0]}' is a "
                f"top-level config section, so '{target_key}' is a key nothing "
                f"reads and '{key}' keeps its config default. Pass it without the "
                f"'{prefix}' (as '{key}={value}') to set the real key.",
                flush=True,
            )
        if prefix == "++":
            # Force-add is valid whether or not the config declares the key, so
            # it needs qualifying but never the declared-path lookup.
            remapped.append(f"++{target_key}={value}")
        elif target_key in declared_custom_paths:
            remapped.append(f"{target_key}={value}")
        else:
            remapped.append(f"+{target_key}={value}")

    if argv is None:
        sys.argv = [sys.argv[0], *remapped]
    return remapped


__all__ = ["remap_plus_args_to_custom"]
