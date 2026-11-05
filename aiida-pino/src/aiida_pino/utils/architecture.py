"""Canonical mapping from a run/config name to its model architecture.

One table, used by both the console's filterable label and the keyword stamped on a
model node at creation, so the two cannot disagree. Previously the console carried its
own private list; it had no GAOT entry, so every GAOT run was displayed as the
fall-through default "FNO / PINO".

Matching is by NAME rather than by resolved config, deliberately: a model node records
its `run_name` and `config_name` but not its `custom.model_family`, and the GAOT
configs inherit from the RIGNO ones (`config_flow3D_GAOT_randompath` has
`defaults: [config_flow3D_RIGNO_randompath_onecycle]`), so reading the resolved config
would report the parent's family. Order matters for the same reason: the most specific
token is checked first, so a name mentioning both wins for the actual architecture.
"""
from __future__ import annotations

from typing import Any, Iterable

# (token searched for in the lowercased name, display label, keyword)
# GAOT first: its configs inherit from RIGNO ones, so a name or a config chain that
# mentions both must resolve to GAOT.
ARCHITECTURE_TOKENS: tuple[tuple[str, str, str], ...] = (
    ("gaot", "GAOT", "gaot"),
    ("rigno", "RIGNO", "rigno"),
    ("unet", "U-Net", "unet"),
    ("u_net", "U-Net", "unet"),
    ("fcn", "Pointwise FC", "fcn"),
    ("deeponet", "DeepONet", "deeponet"),
)

DEFAULT_ARCHITECTURE = "FNO / PINO"
DEFAULT_ARCHITECTURE_KEYWORD = "fno"

_NAME_FIELDS = ("config_name", "model_name", "label", "run_name")


def _haystack(*values: Any) -> str:
    return " ".join(str(v or "") for v in values).lower()


def architecture_from_names(*values: Any) -> str:
    """Display label, e.g. "GAOT". Falls back to "FNO / PINO"."""
    hay = _haystack(*values)
    for token, label, _ in ARCHITECTURE_TOKENS:
        if token in hay:
            return label
    return DEFAULT_ARCHITECTURE


def architecture_keyword_from_names(*values: Any) -> str:
    """Lowercase keyword, e.g. "gaot". Matches the convention already used by the
    self-improve runs, whose keyword lists carry "fno" alongside the run tags."""
    hay = _haystack(*values)
    for token, _, keyword in ARCHITECTURE_TOKENS:
        if token in hay:
            return keyword
    return DEFAULT_ARCHITECTURE_KEYWORD


def architecture_for_summary(summary: dict) -> str:
    """Display label from a summary/metadata dict, reading the usual name fields."""
    return architecture_from_names(*((summary or {}).get(k) for k in _NAME_FIELDS))


def architecture_keyword_for_metadata(metadata: dict) -> str:
    """Keyword from a model-metadata dict, reading the usual name fields."""
    return architecture_keyword_from_names(*((metadata or {}).get(k) for k in _NAME_FIELDS))


def merge_keywords(existing: Iterable[str] | None, *add: str) -> list[str]:
    """Union of keywords, order-stable and de-duplicated CASE-INSENSITIVELY.

    Never drops what a caller already set -- the self-improve driver stamps its own run
    tags and those must survive the architecture keyword being added -- and never
    changes the case of a tag that is already there.

    Case-insensitive matters: several older demo models are tagged `["FNO", "demo",
    "straight path"]`, and a case-sensitive union would append a second, lowercase
    "fno" to every one of them.
    """
    out: list[str] = []
    seen: set[str] = set()
    for value in list(existing or []) + list(add):
        text = str(value).strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        out.append(text)
    return out
