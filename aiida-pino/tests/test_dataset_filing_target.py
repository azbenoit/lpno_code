"""Where a finished FLOW-3D case gets filed, and who decides.

Two behaviours:

- **every** run is ingested and filed, single or batch child — a one-off case
  used to be ingested and then left unattached unless it happened to go through
  the batch path;
- a caller can **name the dataset**, instead of relying on a display-name lookup
  against `dataset_profile.batch_keyword`.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from aiida_pino.workflows.dataset_filing import file_simulations, resolve_filing_target


def _base(factory, name, n=2):
    from aiida_pino.data.dataset_hierarchy import create_base_dataset

    node, _ = create_base_dataset([factory(f"{name}_{i}") for i in range(n)], name=name)
    return node


def test_an_explicit_target_wins_over_the_name_lookup(aiida_profile_clean, pino_simulation_factory):
    """Naming a node is more specific than naming a string, and display names
    are not unique — two sweeps under one batch keyword have collided before."""
    wanted = _base(pino_simulation_factory, "wanted")
    _decoy = _base(pino_simulation_factory, "by_name")

    fresh = [pino_simulation_factory("newcase", group_name="nc-1")]
    dataset, action = file_simulations(fresh, name="by_name", target_dataset=wanted)

    assert action == "extended"
    assert fresh[0].uuid in set(dataset.get_member_simulation_uuids())
    # ...and it went to the named node's lineage, not the same-named decoy
    assert wanted.uuid in {wanted.uuid, dataset.uuid} and dataset.uuid != _decoy.uuid


def test_filing_by_name_still_extends_an_existing_dataset(aiida_profile_clean, pino_simulation_factory):
    existing = _base(pino_simulation_factory, "sweep_a")
    fresh = [pino_simulation_factory("newcase", group_name="nc-2")]

    dataset, action = file_simulations(fresh, name="sweep_a")
    assert action == "extended"
    assert dataset.get_simulation_count() == existing.get_simulation_count() + 1


def test_filing_creates_the_dataset_when_the_name_is_new(aiida_profile_clean, pino_simulation_factory):
    fresh = [pino_simulation_factory("solo", group_name="solo-1")]
    dataset, action = file_simulations(fresh, name="brand_new")
    assert action == "created"
    assert dataset.get_simulation_count() == 1


def test_an_already_owned_simulation_is_never_stolen(aiida_profile_clean, pino_simulation_factory):
    """A case belongs to one base dataset. A requeue must not move it out of the
    sweep it already belongs to."""
    owned = _base(pino_simulation_factory, "owner", n=1)
    member = owned.iter_member_simulations()[0]

    dataset, action = file_simulations([member], name="somewhere_else")
    assert (dataset, action) == (None, "nothing")


def test_a_superseded_target_is_refused(aiida_profile_clean, pino_simulation_factory):
    """Filing into a retired node would put the cases where no listing shows
    them: its replacement is what everything else reads."""
    retired = _base(pino_simulation_factory, "retired")
    retired.base.extras.set("superseded", True)
    retired.base.extras.set("replaced_by_pk", 4242)

    with pytest.raises(ValueError, match="superseded"):
        resolve_filing_target(target_dataset=retired, name="retired")


def test_a_single_run_files_itself_and_a_batch_child_defers(aiida_profile_clean):
    """The rule that keeps a sweep from becoming N one-member datasets — unless
    the caller named a target, which is a deliberate instruction either way."""
    from aiida_pino.workflows.flow3d_generate import should_file_own_dataset

    assert should_file_own_dataset(caller=None) is True
    assert should_file_own_dataset(caller=object()) is False


def test_both_workchains_accept_a_target_dataset():
    from aiida_pino.data import PinoDatasetCollectionData
    from aiida_pino.workflows.flow3d_batch import Flow3DBatchWorkChain
    from aiida_pino.workflows.flow3d_generate import Flow3DGenerateWorkChain

    for cls in (Flow3DGenerateWorkChain, Flow3DBatchWorkChain):
        port = cls.spec().inputs["target_dataset"]
        assert port.required is False
        assert PinoDatasetCollectionData in tuple(port.valid_type)


def test_a_merged_dataset_resolves_to_its_survivor(aiida_profile_clean, pino_simulation_factory):
    """Collapsing duplicates must not rewrite what a model consumed.

    An AiiDA input link is immutable, and re-pointing it would claim a model
    saw a node it never did. So the merge happens at read time: the retired
    node keeps every link, and anything displaying it names the survivor.
    """
    from aiida_pino.console.config import Settings
    from aiida_pino.console.service import AiidaConsoleService
    from aiida_pino.utils import build_collection_metadata_from_simulations, get_or_create_dataset_collection

    sims = [pino_simulation_factory("merge", group_name=f"mg-{i}") for i in range(2)]
    keeper, _ = get_or_create_dataset_collection(
        sims, metadata=build_collection_metadata_from_simulations(sims, collection_name="keeper")
    )
    # Different members: identical ones would collapse into `keeper` by content identity.
    others = [pino_simulation_factory("merge", group_name=f"mg-other-{i}") for i in range(2)]
    retired, _ = get_or_create_dataset_collection(
        others, metadata=build_collection_metadata_from_simulations(others, collection_name="retired_twin")
    )
    assert retired.uuid != keeper.uuid
    retired.base.extras.set("superseded", True)
    retired.base.extras.set("replaced_by_uuid", str(keeper.uuid))

    svc = AiidaConsoleService(Settings())
    resolved = svc.canonical_dataset(str(retired.uuid))
    assert resolved is not None and resolved["pk"] == keeper.pk
    # a dataset that was never superseded resolves to nothing, not to itself
    assert svc.canonical_dataset(str(keeper.uuid)) is None


def test_automated_filing_never_leaves_a_blank_description(aiida_profile_clean, pino_simulation_factory):
    """713 of 802 collections had no description, so a dataset's purpose lived
    only in whoever made it. Automated filing must not block a finished solve on
    prose — but it must not leave an empty string either."""
    from aiida_pino.workflows.dataset_filing import file_simulations

    sims = [pino_simulation_factory("autodesc", group_name=f"ad-{i}") for i in range(2)]
    dataset, action = file_simulations(sims, name="autodesc_batch",
                                       extra_metadata={"batch_keyword": "autodesc_batch"})
    assert action == "created"
    described = dataset.get_dataset_metadata().get("description") or dataset.description or ""
    assert "autodesc_batch" in described
    assert "auto-generated" in described


def test_a_supplied_description_is_kept_verbatim(aiida_profile_clean, pino_simulation_factory):
    from aiida_pino.workflows.dataset_filing import file_simulations

    sims = [pino_simulation_factory("mydesc", group_name="md-1")]
    dataset, _ = file_simulations(sims, name="mydesc_batch",
                                  extra_metadata={"description": "Gate cases for the SS304L arm."})
    described = dataset.get_dataset_metadata().get("description") or dataset.description or ""
    assert described == "Gate cases for the SS304L arm."


# --- keywords reached the batch path and not the generate path -------------
# 2026-08-21 duplication audit: `dataset_profile.keywords` is what the
# `old_physics` / `experimentally_anchored` era selection reads, and every
# console keyword query. `flow3d_batch.py` forwarded it (with a comment citing
# the 2026-08-10 friction); `flow3d_generate.py` hand-built its own
# `extra_metadata` dict with `description` and `batch_keyword` only, so a
# one-off or a requeue silently produced a dataset with zero keywords and no
# warning -- `dataset_profile_metadata_gaps` sees both its fields present.
# The two call sites now take the mapping from one place.

def test_profile_metadata_carries_keywords_not_just_the_batch_keyword():
    from aiida_pino.workflows.dataset_filing import dataset_metadata_from_profile

    profile = {
        "description": "evaluation only -- do not train on this",
        "keywords": ["experimentally_anchored", "ti64"],
        "batch_keyword": "ti64_v5_measured",
    }
    md = dataset_metadata_from_profile(profile)

    # `batch_keyword` NAMES the dataset; `keywords` are its tags. Carrying one
    # is not carrying the other -- that confusion is the whole bug.
    assert md["keywords"] == ["experimentally_anchored", "ti64"]
    assert md["batch_keyword"] == "ti64_v5_measured"
    assert md["description"] == "evaluation only -- do not train on this"


def test_profile_metadata_accepts_call_site_extras():
    from aiida_pino.workflows.dataset_filing import dataset_metadata_from_profile

    md = dataset_metadata_from_profile({"keywords": ["a"]}, batch_workchain_pk=17)
    assert md["batch_workchain_pk"] == 17
    assert md["keywords"] == ["a"]


def test_neither_workflow_hand_builds_the_profile_metadata_dict():
    """The regression: a second hand-written mapping is how they diverged."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "aiida_pino" / "workflows"
    for name in ("flow3d_generate.py", "flow3d_batch.py"):
        text = (root / name).read_text()
        assert "dataset_metadata_from_profile" in text, f"{name} must use the shared mapping"
        # a literal `"batch_keyword": profile.get(...)` is the hand-built form
        assert '"batch_keyword": profile.get' not in text, (
            f"{name} hand-builds the profile metadata dict again; use "
            "dataset_metadata_from_profile so `keywords` cannot be dropped"
        )
