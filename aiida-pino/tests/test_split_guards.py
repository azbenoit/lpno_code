"""Split-safety guards: the cheap fixes that replace the shelved table redesign.

Five adversarial review passes concluded that the contamination the table
redesign targeted is already blocked by `evaluate_model`'s member-uuid overlap
guard, which never consults a split label. What remained genuinely open was
narrow, and is what these tests pin:

* the eval-only predicate misses `eval` and any mixed set, and `submit_kfold`
  has no split guard at all -- so kfold trains on the NIST/Galiegue validation
  sets today;
* the overlap guard sees only a model's LAST training stage, so a warm-started
  model is not protected by its ancestor's training set;
* nothing records who re-split a dataset, when, or why -- the failure that let
  38427's recipe drift from its labels with no trail.
"""

from __future__ import annotations

import pytest


# --- vocabulary -----------------------------------------------------------

def test_excluded_and_eval_are_sanctioned_split_values():
    """`eval` and an exclusion value exist in production and must be writable.

    Measured on the authors' database: 4 simulations carry `eval`, 5 carry
    `duplicate_excluded` (38427's centre-point replicas). Both were written by
    ad-hoc scripts because `SPLIT_VALUES` rejects them, which is how a
    deliberate, correct decision ended up undocumented.
    """
    from aiida_pino.utils.annotations import SPLIT_VALUES

    assert "eval" in SPLIT_VALUES
    assert "excluded" in SPLIT_VALUES
    # The legacy value stays writable so 38427's five replicas can be restated
    # without inventing a migration.
    assert "duplicate_excluded" in SPLIT_VALUES


# --- eval-only predicate --------------------------------------------------

def test_evaluation_only_detects_eval_and_mixed_sets_but_not_unlabelled():
    """Positive evidence only: labels present, some held-out, none trainable.

    The old rule was `splits == {"test"}`, which misses `{eval}` (two datasets,
    both keyworded evaluation_only) and misses the mixed
    `{test, excluded}` shape ablation30 will have. The tempting inversion --
    "no member resolves to train or val" -- is worse: measured over the authors' database it
    reclassifies 987 of 1731 live collections as evaluation-only, because
    `unassigned` is the normal state of a freshly finalised batch.
    """
    from aiida_pino.console.service import is_evaluation_only_split_set as f

    assert f({"test"}) is True
    assert f({"eval"}) is True
    assert f({"test", "excluded"}) is True
    assert f({"test", "duplicate_excluded"}) is True

    assert f(set()) is False
    assert f({"unassigned"}) is False, "an unlabelled dataset is not evaluation-only"
    assert f({"train", "val", "test"}) is False
    assert f({"train"}) is False
    assert f({"unassigned", "test"}) is False, "partially labelled is not reserved"


# --- warm-start lineage ---------------------------------------------------

def test_training_lineage_is_transitive_over_warm_starts():
    """A warm-started model must carry its ancestor's training members.

    `training_source_summary` records only the last stage, so a model
    fine-tuned from an ancestor is not protected against being scored on what
    the ANCESTOR trained on -- and 8 of 13 warm-start chains are this
    campaign's established pattern. The eval guard is label-independent and
    correct; it was just being handed an incomplete set.
    """
    from aiida_pino.utils.model_registry import lineage_member_uuids

    ancestor = {"members": [{"uuid": "a1"}, {"uuid": "a2"}]}
    child = {"members": [{"uuid": "c1"}], "init_lineage_uuids": ["a1", "a2"]}

    assert lineage_member_uuids(ancestor) == {"a1", "a2"}
    assert lineage_member_uuids(child) == {"c1", "a1", "a2"}, "must union the ancestry"
    assert lineage_member_uuids(None) == set()
    assert lineage_member_uuids({}) == set()


# --- split-change audit log ----------------------------------------------

def test_split_change_is_recorded_with_actor_time_and_reason():
    """Every split write appends an immutable-by-convention audit entry.

    The requirement: resplitting must remain possible, with a clear
    history of how and when it was done. One dataset has a
    stored recipe claiming random 0.8/0.05/0.15 while its labels are
    family-inverted, and nothing anywhere says who changed them or why.
    """
    from aiida_pino.utils.annotations import append_split_audit

    log = []
    entry = append_split_audit(
        log, before={"a": "train"}, after={"a": "test"}, reason="fix family inversion",
        actor="reviewer", method="stratified",
    )
    assert log == [entry]
    for key in ("utc", "actor", "reason", "method", "changed", "before_digest", "after_digest"):
        assert key in entry, key
    assert entry["changed"] == 1
    assert entry["reason"] == "fix family inversion"

    # A second change appends; it never rewrites history.
    append_split_audit(log, before={"a": "test"}, after={"a": "val"}, reason="second thoughts",
                       actor="reviewer", method="manual")
    assert len(log) == 2
    assert log[0]["reason"] == "fix family inversion"

    # A no-op write is still recorded, with changed=0 -- "nothing happened" is
    # itself the answer to "who touched this and when".
    append_split_audit(log, before={"a": "val"}, after={"a": "val"}, reason="rerun",
                       actor="reviewer", method="random")
    assert log[2]["changed"] == 0
