"""The base RIGNO configs must FiLM the broadcast-constant scalars P / V_scan / H_norm.

WHY. FiLM routes broadcast-constant scalar inputs to conditioning instead of letting
them enter the pnode features as spatially-uniform channels. The parity ablation
measured FiLM at roughly -28% T MAE on straight-path (2026-08-08), so leaving
``film_keys: []`` in the base configs silently costs that on every arm that does not
inherit ``config_flow3D_RIGNO_pvlhs``. That config's own comment records the same
defect: "FiLM was claimed for these channels and was NOT enabled: film_keys defaults
to [] in the parent and nothing populates it, so broadcast scalars were entering the
pnode features directly."

WHY ONLY THESE THREE. ``rigno_arch.py`` hard-asserts
``film_keys not in input keys``, so a key listed here that some inheriting config does
not declare as an input breaks that arm at construction. P, V_scan and H_norm are the
first three ``FNO_input_keys`` of every RIGNO config, so they are the safe set. Richer
lists (preheat_T0, focus_factor, beam_shape_uniform) stay in the sweep-specific
configs that actually declare those channels.

Config-only: no GPU, no license, no dataset.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONF = Path(__file__).resolve().parents[1] / "src" / "lpno" / "conf" / "hydra"

BASE_CONFIGS = [
    "config_flow3D_RIGNO.yaml",
    "config_flow3D_RIGNO_randompath_onecycle.yaml",
    "config_flow3D_RIGNO_onecycle.yaml",
    "config_flow3D_RIGNO_randompath.yaml",
]

REQUIRED = {"P", "V_scan", "H_norm"}


def _load(name):
    with (CONF / name).open() as fh:
        return yaml.safe_load(fh)


@pytest.mark.parametrize("name", BASE_CONFIGS)
def test_base_config_films_the_broadcast_scalars(name):
    cfg = _load(name)
    film = cfg["arch"]["rigno"].get("film_keys") or []
    missing = REQUIRED - set(film)
    assert not missing, (
        f"{name} does not FiLM {sorted(missing)}. Broadcast-constant scalars entering "
        "the pnode features directly cost ~28% T MAE in the parity ablation."
    )


@pytest.mark.parametrize("name", BASE_CONFIGS)
def test_every_film_key_is_an_input_key(name):
    """The invariant that keeps rigno_arch's assert from firing on an inheriting arm."""
    cfg = _load(name)
    film = cfg["arch"]["rigno"].get("film_keys") or []
    inputs = set(cfg["custom"]["FNO_input_keys"])
    unknown = [k for k in film if k not in inputs]
    assert not unknown, (
        f"{name} lists film_keys {unknown} that are not in FNO_input_keys. "
        "rigno_arch.py asserts `film_keys not in input keys` at construction."
    )


def test_pvlhs_still_films_its_own_richer_set():
    """The sweep config keeps its extra channels; the base change must not shrink it."""
    cfg = _load("config_flow3D_RIGNO_pvlhs.yaml")
    film = set(cfg["arch"]["rigno"]["film_keys"])
    assert REQUIRED <= film
    assert {"preheat_T0", "focus_factor", "beam_shape_uniform"} <= film


def test_no_rigno_config_declares_film_keys_empty_while_able_to_fill_them():
    """Completeness guard: catch a config left at [] that COULD carry the three keys.

    The first version of this change updated only the two configs I happened to be
    looking at, leaving `config_flow3D_RIGNO_onecycle` and `_randompath` at []. A
    per-file parametrisation cannot catch that -- it only checks the files it lists.
    This walks every RIGNO config instead, so a new one added at [] fails here.

    Configs with no `film_keys` of their own are skipped: they inherit from a parent
    via `defaults:`, which is the intended propagation path.
    """
    offenders = []
    for path in sorted(CONF.glob("*RIGNO*.yaml")):
        cfg = _load(path.name)
        rigno = (cfg.get("arch") or {}).get("rigno") or {}
        if "film_keys" not in rigno:
            continue                      # inherits from its parent
        film = rigno.get("film_keys") or []
        inputs = set((cfg.get("custom") or {}).get("FNO_input_keys") or [])
        if not inputs:
            continue                      # inherits its input keys too
        if REQUIRED <= inputs and not REQUIRED <= set(film):
            offenders.append(path.name)
    assert not offenders, (
        f"these RIGNO configs declare P/V_scan/H_norm as inputs but do not FiLM them: "
        f"{offenders}"
    )
