"""A sweep can set its dataset's keywords, and its description is not discarded.

Both measured on the live DB before the fix (2026-08-10):

* `flow3d_batch.py` had zero `keywords` handling -- only `batch_keyword`, which
  NAMES the dataset and tags its simulations. So a sweep structurally could not
  set dataset keywords, however carefully written, and the field that looks like
  it does is the trap.
* 25 of the 56 collections carrying a `dataset_metadata.description` had an
  EMPTY `node.description` -- the field the console displays. The description
  was accepted and then invisible.
"""

from __future__ import annotations

from aiida import orm

from aiida_pino.data import PinoDatasetCollectionData
from aiida_pino.utils.profiles import FLOW3D_NPZ_PROFILE_KEYS


def _sims(tmp_path, n=2):
    from test_pack_index_export import _npz_file_sim

    return [_npz_file_sim(tmp_path / f"src_{i}", f"case_{i}", seed=i) for i in range(n)]


def _collection(sims, name, **extra):
    from aiida_pino.utils import build_collection_metadata_from_simulations
    from aiida_pino.utils.compose import compose_collection, member_inputs

    metadata = build_collection_metadata_from_simulations(
        sims, collection_name=name, dataset_role="base", composition_kind="import",
        extra_metadata=extra or None,
    )
    return compose_collection(sims, metadata, **member_inputs(sims))[0]


def test_dataset_profile_accepts_keywords():
    assert "keywords" in FLOW3D_NPZ_PROFILE_KEYS
    assert "batch_keyword" in FLOW3D_NPZ_PROFILE_KEYS, "the naming field still exists"


def test_a_sweep_description_reaches_the_field_the_console_shows(aiida_profile_clean, tmp_path):
    text = "Ti64 multi-beam random paths, TOTAL power capped at 195 W."
    node = _collection(_sims(tmp_path), "described_batch", description=text)
    assert node.description == text, "metadata-only description is invisible in the console"
    assert node.get_dataset_metadata()["description"] == text


def test_keywords_from_the_profile_land_as_queryable_extras(aiida_profile_clean, tmp_path):
    node = _collection(
        _sims(tmp_path), "kw_batch", keywords=["multi-beam", "ti64", "power capped"],
    )
    assert set(node.base.extras.get("keywords")) == {"multi-beam", "ti64", "power capped"}
    found = orm.QueryBuilder().append(
        PinoDatasetCollectionData,
        filters={"extras.keywords": {"contains": ["multi-beam"]}},
        project=["id"],
    ).all(flat=True)
    assert node.pk in found


def test_an_existing_description_is_not_overwritten(aiida_profile_clean, tmp_path):
    """Mirroring fills a blank; it must never clobber what a human wrote."""
    node = _collection(_sims(tmp_path), "kept", description="from the sweep")
    node.description = "a human corrected this"
    assert node.description == "a human corrected this"


def test_no_keywords_means_no_extra(aiida_profile_clean, tmp_path):
    node = _collection(_sims(tmp_path), "plain_batch")
    assert node.base.extras.get("keywords", None) in (None, [])


def test_missing_findability_fields_are_reported_not_refused():
    """A submission omitting keywords/description must WARN, never fail.

    Inputs freeze at submit, so an omission can only be repaired by a manual
    post-hoc write — six batches
    needed one because nothing said anything at submit time. But refusing is
    the wrong cure: a gate that rejects a sweep costing days of seat time turns
    a missing tag into a dead batch (the 2026-07-21 disk-floor lesson, where a
    refusing gate killed 300 queued cases in five minutes). So: report.
    """
    from aiida_pino.utils.profiles import dataset_profile_metadata_gaps

    assert dataset_profile_metadata_gaps({"keywords": ["a"], "description": "why"}) is None
    # Empty values count as missing, not present.
    assert dataset_profile_metadata_gaps({"keywords": [], "description": "  "}) is not None

    both = dataset_profile_metadata_gaps({})
    assert both is not None
    assert "keywords" in both and "description" in both
    assert "unfindable" in both.lower()

    # Assert on the "omits ..." clause, not the whole message: the remediation
    # tail legitimately mentions a description write either way.
    only_kw = dataset_profile_metadata_gaps({"description": "why this sweep exists"})
    assert only_kw is not None
    named = only_kw.split(":", 1)[0]
    assert "keywords" in named and "description" not in named

    # `batch_keyword` NAMES the dataset and is not a substitute for keywords.
    assert dataset_profile_metadata_gaps({"batch_keyword": "mysweep"}) is not None
