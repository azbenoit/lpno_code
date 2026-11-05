"""Evaluation defaults to the test split of the dataset it is given.

You pass the *full* dataset — the one holding train/val/test — and eval scores
the held-out `test` split, composing that subset with provenance the same way
`submit_train_eval` composes train/val. This removes the hand-built test
collections that eval used to require, and makes `train_eval` a true one-shot
path: one dataset in, train on train, validate on val, evaluate on test.

Rules:
- a `test` split that is a proper subset is composed and used;
- a dataset whose members are *all* test is used as-is (no pointless re-compose);
- split extras present but an EMPTY test split is a **loud warning**, not a
  silent eval on the wrong data;
- no split extras at all is backward-compatible: the passed collection is the
  eval set (a pre-built test collection still works).
"""
from __future__ import annotations

import pytest

from aiida_pino.console.schemas import EvalOnlyRequest, TrainEvalRequest
from aiida_pino.console.service import AiidaConsoleService
from aiida_pino.console.errors import DaemonUnavailableError
from web.app.config import Settings


@pytest.fixture
def svc():
    return AiidaConsoleService(Settings())


def _collection_of(sims, name):
    from aiida_pino.utils import build_collection_metadata_from_simulations, get_or_create_dataset_collection

    meta = build_collection_metadata_from_simulations(sims, collection_name=name)
    node, _ = get_or_create_dataset_collection(sims, metadata=meta)
    return node


def _split_ds(svc, factory, folder, splits):
    sims = [factory(folder, group_name=f"{folder}-{i}") for i in range(len(splits))]
    for sim, split in zip(sims, splits):
        if split:
            sim.base.extras.set("split", split)
    return _collection_of(sims, folder), sims


# -------------------------------------------------------- _test_subset unit
def test_proper_test_split_is_composed(svc, pino_simulation_factory):
    ds, _ = _split_ds(svc, pino_simulation_factory, "tsplit",
                      ["train", "train", "val", "test", "test"])
    subset, note = svc._test_subset(ds)
    assert subset is not None
    assert subset.get_simulation_count() == 2
    md = subset.get_dataset_metadata()
    # subset role (not composed) with the base recorded for clear provenance
    assert md["dataset_role"] == "subset"
    assert md["base_collection_pk"] == ds.pk
    assert md["base_collection_label"]
    assert md["subset_split"] == "test"
    assert "test split" in note.lower()


def test_test_subset_is_reused_not_resplit(svc, pino_simulation_factory):
    """The same test split of the same base must dedupe, not clone."""
    ds, _ = _split_ds(svc, pino_simulation_factory, "reuse",
                      ["train", "val", "test", "test"])
    first, _ = svc._test_subset(ds)
    second, _ = svc._test_subset(ds)
    assert first.pk == second.pk


def test_split_subsets_grouped_by_base_folder(svc, pino_simulation_factory):
    """Auto splits nest under a per-base subfolder of Training splits (auto)."""
    from aiida_pino.console import folders as folders_mod

    ds, _ = _split_ds(svc, pino_simulation_factory, "grp",
                      ["train", "train", "val", "test"])
    subset, _ = svc._test_subset(ds)
    path = folders_mod.folder_by_pk([subset.pk]).get(subset.pk)
    # ONE flat folder. This asserted a per-BASE-DATASET subfolder, which is what made the
    # only automatic folder writer mint a new subfolder per dataset forever -- 18 of them
    # holding 108 sets before the 2026-09-01 curation flattened them. The property that
    # matters, and is still pinned, is that an auto-split IS filed somewhere predictable;
    # grouping by base dataset belongs in the browser, not in a stored extra.
    assert path == "Training splits", path


def _creator_inputs(node):
    assert node.creator is not None, "subset has no creator process"
    return {link.link_label: link.node for link in node.creator.base.links.get_incoming().all()}


def test_split_subset_records_its_recipe_in_provenance(svc, pino_simulation_factory):
    """A subset must answer "which members, from which dataset, chosen how"
    from the graph alone: creator process, base dataset as an input, and the
    assigner's seed and ratios in a stored Dict."""
    sims = [pino_simulation_factory("recipe", group_name=f"recipe-{i}") for i in range(10)]
    ds = _collection_of(sims, "recipe")
    assert ds.creator is None  # plain get_or_create: stored, no process — the old subset path
    svc.assign_splits(ds.pk, train=0.6, val=0.2, test=0.2, seed=7)

    subset, _ = svc._test_subset(ds)
    inputs = _creator_inputs(subset)
    assert inputs["base_dataset"].pk == ds.pk
    recipe = inputs["parameters"]["split_recipe"]
    assert recipe["method"] == "random"
    assert recipe["seed"] == 7
    assert recipe["fractions"] == {"train": 0.6, "val": 0.2, "test": 0.2}
    assert recipe["only_unassigned"] is True
    assert recipe["stratify_by"] is None
    assert recipe["split"] == "test"
    assert recipe["split_member_count"] == subset.get_simulation_count()
    # the member list the seed was applied to, so a later reshuffle is detectable
    assert recipe["base_member_uuids_sha256"] == ds.base.extras.get("split_recipe")["member_uuids_sha256"]


def test_split_subset_recipe_is_honest_when_labels_predate_capture(svc, pino_simulation_factory):
    """Hand-written or imported split extras have no known seed; the recipe
    says so instead of implying an assigner that never ran."""
    ds, _ = _split_ds(svc, pino_simulation_factory, "norecipe", ["train", "val", "test", "test"])
    subset, _ = svc._test_subset(ds)
    recipe = _creator_inputs(subset)["parameters"]["split_recipe"]
    assert recipe["method"] == "unrecorded"
    assert "seed" not in recipe
    assert recipe["split"] == "test"


def test_all_test_dataset_is_used_as_is(svc, pino_simulation_factory):
    """A pre-built test collection must not be needlessly re-composed."""
    ds, _ = _split_ds(svc, pino_simulation_factory, "alltest", ["test", "test", "test"])
    subset, note = svc._test_subset(ds)
    assert subset is None  # None => caller keeps the passed node
    assert "as-is" in note.lower() or "entirely test" in note.lower()


def test_empty_test_split_is_a_loud_warning(svc, pino_simulation_factory):
    ds, _ = _split_ds(svc, pino_simulation_factory, "notest", ["train", "train", "val"])
    subset, note = svc._test_subset(ds)
    assert subset is None
    assert note is not None and "WARNING" in note
    assert "empty test" in note.lower() or "no test" in note.lower()
    # must not name a specific fallback target: eval-only falls back to the
    # passed dataset, train_eval falls back to val_dataset.
    assert "whole dataset" not in note.lower()


def test_no_split_extras_is_silent_backward_compat(svc, pino_simulation_factory):
    ds, _ = _split_ds(svc, pino_simulation_factory, "nosplit", [None, None])
    subset, note = svc._test_subset(ds)
    assert subset is None
    assert note is None  # a bare collection is just the eval set, no fuss


# -------------------------------------------------------- submit_eval_only wiring
def test_eval_submit_resolves_the_test_split(svc, pino_simulation_factory, local_code_factory, pino_model_factory):
    local_code_factory("/bin/bash", label="pino-python-evalsplit")
    model = pino_model_factory()
    ds, _ = _split_ds(svc, pino_simulation_factory, "evalsplit",
                      ["train", "train", "val", "test", "test"])
    req = EvalOnlyRequest(dataset_pk=ds.pk, model_pk=model.pk, python_code="pino-python-evalsplit")
    # the split subset is composed before the daemon gate fires
    with pytest.raises(DaemonUnavailableError):
        svc.submit_eval_only(req)
    rows = {r["source_folder_name"]: r for r in svc.list_datasets(role="all", limit=50)}
    assert rows["evalsplit_test_split"]["simulation_count"] == 2
    assert rows["evalsplit_test_split"]["dataset_role"] == "subset"


def test_eval_submit_warns_on_empty_test(svc, pino_simulation_factory, local_code_factory, pino_model_factory):
    """An empty test split must still surface the warning even though the daemon
    gate raises first: the warning is attached before submission."""
    local_code_factory("/bin/bash", label="pino-python-evalwarn")
    model = pino_model_factory()
    ds, _ = _split_ds(svc, pino_simulation_factory, "evalwarn", ["train", "val"])
    req = EvalOnlyRequest(dataset_pk=ds.pk, model_pk=model.pk, python_code="pino-python-evalwarn")
    with pytest.raises(DaemonUnavailableError):
        svc.submit_eval_only(req)  # must not crash resolving the split


# -------------------------------------------------------- train_eval evaluates on test
def test_train_eval_targets_the_test_split(svc, pino_simulation_factory, local_code_factory):
    local_code_factory("/bin/bash", label="pino-python-te-test")
    ds, _ = _split_ds(svc, pino_simulation_factory, "tetest",
                      ["train", "train", "train", "val", "test", "test"])
    req = TrainEvalRequest(mode="train_eval", train_dataset_pk=ds.pk, python_code="pino-python-te-test")
    with pytest.raises(DaemonUnavailableError):
        svc.submit_train_eval(req)
    rows = {r["source_folder_name"]: r for r in svc.list_datasets(role="all", limit=50)}
    # train/val AND a test subset are all composed from the one dataset
    assert rows["tetest_train_split"]["simulation_count"] == 3
    assert rows["tetest_val_split"]["simulation_count"] == 1
    assert rows["tetest_test_split"]["simulation_count"] == 2


def test_a_not_held_out_evaluation_is_recorded_on_the_NODE(svc, aiida_profile_clean):
    """The warning is transient; the number is not.

    `_test_subset` returns a loud note when a dataset has split extras but an
    empty test split, and that note went into the submit-time reply and nowhere
    else. The metric it qualifies lives on the eval node, gets read by model
    comparisons and by anyone browsing months later, and looks exactly like a
    test metric. Observed on a real remote run, 2026-08-19.
    """
    from aiida import orm

    node = orm.Dict(dict={"stands in for": "an eval process"}).store()
    svc._stamp_metric_caveat(node.pk, [f"WARNING: ... this is {svc.NOT_HELD_OUT_MARKER}"])

    assert node.base.extras.get("eval_held_out_test") is False
    assert svc.NOT_HELD_OUT_MARKER in node.base.extras.get("eval_metric_caveat")


def test_an_ordinary_evaluation_gets_no_caveat(svc, aiida_profile_clean):
    """Only the fallback is flagged; a real test-split eval stays unmarked."""
    from aiida import orm

    node = orm.Dict(dict={"stands in for": "an eval process"}).store()
    svc._stamp_metric_caveat(node.pk, ["evaluating on the test split of dataset 7: 12 test members"])

    assert "eval_held_out_test" not in node.base.extras.all
    assert "eval_metric_caveat" not in node.base.extras.all


def test_stamping_never_fails_a_submission(svc, aiida_profile_clean):
    """A caveat is bookkeeping; losing it must not lose the run."""
    svc._stamp_metric_caveat(None, [f"x {svc.NOT_HELD_OUT_MARKER}"])
    svc._stamp_metric_caveat(999999999, [f"x {svc.NOT_HELD_OUT_MARKER}"])


# ------------------------------------------------- the 2026-08-06 flag skew
# `evaluate_on_test` was added to TrainEvalWorkChain on 2026-08-06 (f1739fc) with
# default False, and the console was never updated to set it. The console still
# computes a test subset and passes it as `eval_dataset`; the workchain then
# discards it and scores val, reporting one line nobody reads. Every console
# train_eval since that date silently produced a val metric wearing the shape of a
# test metric. `git log -S evaluate_on_test` on console/service.py returns nothing.


def test_console_sets_evaluate_on_test_when_it_supplies_eval_dataset(
    svc, pino_simulation_factory, local_code_factory, monkeypatch
):
    """Passing eval_dataset without the flag is a no-op -- the pair must travel together."""
    local_code_factory("/bin/bash", label="pino-python-flagskew")
    ds, _ = _split_ds(svc, pino_simulation_factory, "flagskew",
                      ["train", "train", "val", "test"])
    captured = {}

    def fake_submit(entry, **inputs):
        captured.update(entry=entry, inputs=inputs)
        return {"pk": 1}

    monkeypatch.setattr(svc, "_submit", fake_submit)
    svc.submit_train_eval(TrainEvalRequest(
        mode="train_eval", train_dataset_pk=ds.pk, val_dataset_pk=ds.pk,
        python_code="pino-python-flagskew", training_profile={"run_name": "x"}))

    assert "eval_dataset" in captured["inputs"], "console stopped resolving the test split"
    flag = captured["inputs"].get("evaluate_on_test")
    assert flag is not None, "eval_dataset passed without evaluate_on_test: the workchain ignores it"
    assert bool(getattr(flag, "value", flag)) is True


def test_evaluate_on_test_can_be_turned_off_for_expensive_sets(
    svc, pino_simulation_factory, local_code_factory, monkeypatch
):
    """The 2026-08-06 cost argument stays available: a full-res test split took 2.9 h."""
    local_code_factory("/bin/bash", label="pino-python-flagoff")
    ds, _ = _split_ds(svc, pino_simulation_factory, "flagoff",
                      ["train", "train", "val", "test"])
    captured = {}
    monkeypatch.setattr(svc, "_submit", lambda entry, **kw: captured.update(inputs=kw) or {"pk": 1})
    svc.submit_train_eval(TrainEvalRequest(
        mode="train_eval", train_dataset_pk=ds.pk, val_dataset_pk=ds.pk,
        python_code="pino-python-flagoff", training_profile={"run_name": "x"},
        evaluate_on_test=False))
    flag = captured["inputs"].get("evaluate_on_test")
    assert flag is None or bool(getattr(flag, "value", flag)) is False


def test_train_eval_resolves_the_test_SIBLING_of_a_pre_split_train_set(
    svc, pino_simulation_factory, local_code_factory, monkeypatch
):
    """A split-based submission passes a TRAIN subset, which holds no test members.

    `_test_subset` looks for test members INSIDE the dataset it is given, and its
    docstring assumes the caller passes the full train/val/test collection. A batch
    that has already split its data passes the train subset instead, so the lookup
    can never succeed and the eval silently falls back to val. The family's test
    sibling is the right answer and the lookup for it already existed -- it was only
    ever used to build the warning message.
    """
    from aiida_pino.data import PinoDatasetCollectionData

    parent, sims = _split_ds(svc, pino_simulation_factory, "family",
                             ["train", "train", "val", "test"])
    train_only = svc._split_subset_collection(
        parent, [s for s in sims if str(s.base.extras.get("split", "")) == "train"], "train")
    test_sib = svc._split_subset_collection(
        parent, [s for s in sims if str(s.base.extras.get("split", "")) == "test"], "test")

    # precondition: the train subset genuinely has no test members of its own
    subset, _ = svc._test_subset(train_only)
    assert subset is None

    local_code_factory("/bin/bash", label="pino-python-sibling")
    captured = {}
    monkeypatch.setattr(svc, "_submit", lambda entry, **kw: captured.update(inputs=kw) or {"pk": 1})
    svc.submit_train_eval(TrainEvalRequest(
        mode="train_eval", train_dataset_pk=train_only.pk, val_dataset_pk=parent.pk,
        python_code="pino-python-sibling", training_profile={"run_name": "x"}))

    got = captured["inputs"].get("eval_dataset")
    assert got is not None, "train_eval did not fall back to the family's test sibling"
    assert got.pk == test_sib.pk
    assert bool(captured["inputs"]["evaluate_on_test"].value) is True


def test_eval_only_is_NOT_redirected_to_a_sibling(svc, pino_simulation_factory):
    """An explicitly-passed eval dataset is a choice, not a guess -- never override it.

    submit_eval_only takes the dataset the caller named. Redirecting it to a test
    sibling would silently score something the caller did not ask for, which is the
    same class of defect as scoring val while claiming test.
    """
    parent, sims = _split_ds(svc, pino_simulation_factory, "noredirect",
                             ["train", "train", "val", "test"])
    val_only = svc._split_subset_collection(
        parent, [s for s in sims if str(s.base.extras.get("split", "")) == "val"], "val")
    subset, note = svc._test_subset(val_only)
    assert subset is None, "_test_subset must not redirect an explicitly chosen dataset"
    assert note is not None and svc.NOT_HELD_OUT_MARKER in note
