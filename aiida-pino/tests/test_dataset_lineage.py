"""Where a derived dataset came from, and which folder it lives in.

The console used to infer lineage from member-set inclusion and a hardcoded map
of folder names, so a derived dataset showed up wherever the smallest superset
happened to be — or nowhere. Collections record `parent_collection_uuid`; that
is the answer when it exists, and the fallbacks only fill the gaps.
"""

from __future__ import annotations

import pytest

from aiida_pino.console import folders
from aiida_pino.data.dataset_hierarchy import create_base_dataset
from aiida_pino.utils.dataset_derivation import build_derived_collection_metadata
from aiida_pino.utils.model_registry import get_or_create_dataset_collection
from web.app.aiida_service import AiidaConsoleService
from web.app.config import Settings


@pytest.fixture
def svc():
    return AiidaConsoleService(Settings())


def _base(factory, name, n=3):
    node, _ = create_base_dataset([factory(f"{name}_{i}") for i in range(n)], name=name)
    return node


def _derived(parent, name, super_res=2):
    """A derived collection exactly as materialize_derived_dataset makes one.

    `super_res_factor` varies per level so each derivation is genuinely distinct
    — with an identical profile the reuse lookup rightly returns the same node.
    The default is 2, not 1: since collections became content-addressed
    (`content_signature` = members + derivation profile), factor 1 *is* the
    canonical base profile, so it no longer mints a child at all. See
    `test_a_noop_derivation_returns_the_base_itself` below, which pins that.
    """
    metadata = build_derived_collection_metadata(parent, {"super_res_factor": super_res}, parent_collection=parent)
    metadata["source_folder_name"] = name
    node, _ = get_or_create_dataset_collection(parent.iter_member_simulations(), metadata=metadata)
    return node


def _rows(svc):
    return {row["pk"]: row for row in svc.list_datasets(role="all")}


def test_recorded_parent_wins_over_the_subset_guess(aiida_profile_clean, pino_simulation_factory, svc) -> None:
    base = _base(pino_simulation_factory, "base", n=3)
    other = _base(pino_simulation_factory, "unrelated", n=4)
    derived = _derived(base, "base_derived")

    rows = _rows(svc)
    assert rows[derived.pk]["parent_pk"] == base.pk
    assert rows[derived.pk]["parent_source"] == "recorded"
    assert rows[base.pk]["parent_pk"] is None
    assert rows[other.pk]["parent_pk"] is None


def test_a_derived_dataset_follows_its_base_into_a_folder(aiida_profile_clean, pino_simulation_factory, svc) -> None:
    base = _base(pino_simulation_factory, "base")
    derived = _derived(base, "base_derived")

    svc.set_folder([base.pk], "reference/keyhole")

    rows = _rows(svc)
    assert rows[base.pk]["folder"] == "reference/keyhole"
    assert rows[derived.pk]["folder"] == "reference/keyhole", "the derived set stayed behind"
    # Nothing was written to the child: it inherits, so moving the base again
    # moves it too and unfiling the base takes it back out.
    assert folders.folder_by_pk([derived.pk]) == {}
    assert rows[derived.pk]["folder_explicit"] is False


def test_filing_a_derived_dataset_explicitly_beats_inheritance(aiida_profile_clean, pino_simulation_factory, svc) -> None:
    base = _base(pino_simulation_factory, "base")
    derived = _derived(base, "base_derived")
    svc.set_folder([base.pk], "reference")
    svc.set_folder([derived.pk], "scratch")

    rows = _rows(svc)
    assert rows[derived.pk]["folder"] == "scratch"
    assert rows[derived.pk]["folder_explicit"] is True
    assert rows[derived.pk]["parent_pk"] == base.pk, "filing it elsewhere must not cut the lineage"


def test_a_noop_derivation_returns_the_base_itself(aiida_profile_clean, pino_simulation_factory, svc) -> None:
    """A derivation that changes nothing is not a new dataset.

    `super_res_factor: 1` over the same members produces the canonical base
    layout — identical content, so content-addressed reuse hands back the base
    node instead of minting a twin under a second name. That is the whole point
    of `content_signature`; it is recorded here because it is surprising: the
    console's derive request defaults to an empty profile, which normalises to
    exactly this no-op.
    """
    base = _base(pino_simulation_factory, "base")
    noop = _derived(base, "base_noop", super_res=1)
    assert noop.pk == base.pk
    assert noop.base.attributes.all["content_signature"] == base.base.attributes.all["content_signature"]
    # ...while a real derivation still makes a child with a recorded parent
    real = _derived(base, "base_real", super_res=2)
    assert real.pk != base.pk
    assert _rows(svc)[real.pk]["parent_pk"] == base.pk


def test_a_composed_dataset_can_itself_have_a_derived_child(aiida_profile_clean, pino_simulation_factory, svc) -> None:
    """The chain base -> composed -> derived must survive intact: the middle
    node keeps both its parent link and its own child."""
    base = _base(pino_simulation_factory, "base")
    composed = _derived(base, "composed", super_res=2)
    grandchild = _derived(composed, "composed_derived", super_res=3)

    svc.set_folder([base.pk], "reference")
    rows = _rows(svc)

    assert rows[composed.pk]["parent_pk"] == base.pk
    assert rows[grandchild.pk]["parent_pk"] == composed.pk
    assert rows[grandchild.pk]["folder"] == "reference", "inheritance stopped at the first hop"


def test_an_inferred_parent_does_not_relocate_a_dataset(aiida_profile_clean, pino_simulation_factory, svc) -> None:
    """A member-subset guess is a display hint. Letting it inherit folders put
    every single-case import into whichever curated folder happened to contain
    it (56 datasets landed in "Materials" on the live profile)."""
    big = _base(pino_simulation_factory, "big", n=4)
    svc.set_folder([big.pk], "curated")

    rows = _rows(svc)
    guessed = [r for r in rows.values() if r["parent_source"] == "inferred"]
    for row in guessed:
        assert row["folder"] != "curated" or row["folder_explicit"], "a guess moved a dataset into a folder"


def test_folder_inheritance_survives_a_lineage_cycle() -> None:
    """Nothing should create a parent cycle, but a hand-edited attribute could,
    and a listing that never returns is worse than a wrong folder."""
    rows = [
        {"pk": 1, "_inherit_pk": 2, "_folder_own": None, "folder": None, "simulation_count": 5},
        {"pk": 2, "_inherit_pk": 1, "_folder_own": None, "folder": None, "simulation_count": 5},
        {"pk": 3, "_inherit_pk": 1, "_folder_own": "kept", "folder": None, "simulation_count": 5},
    ]
    folders.resolve_rows(rows)   # must terminate
    assert rows[2]["folder"] == "kept"
    assert rows[0]["folder"] is None and rows[1]["folder"] is None
