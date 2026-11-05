"""A caveat must point at the cause, not at the nearest innocent thing.

`TrainEvalWorkChain` has no eval-dataset input, so its auto-eval scores the VAL
split — the same data the run's checkpoints were selected against. The warning
said "dataset N has split extras but an EMPTY test split; falling back to the
non-test data", which reads as "your dataset is malformed".

The dataset was fine. For the campaign that hit this, the splits are clean and
disjoint across sibling subsets of one parent: 55011 is 96/96 train, 55014 is
14/14 val, 55017 is 13/13 test. The test split existed the whole time; nothing
ever passed it. Blaming the data sent the reader to the wrong place.
"""

from __future__ import annotations

from aiida_pino.console.service import AiidaConsoleService


def test_the_caveat_names_the_plumbing_not_the_data():
    note = AiidaConsoleService._test_split_caveat(55011, sibling=None)
    assert "EMPTY test split" not in note
    assert "no test dataset" in note.lower()
    assert "val" in note.lower(), "say WHAT it scored instead"


def test_the_caveat_names_a_sibling_that_holds_the_test_split():
    """Turning a dead end into the command that fixes it."""
    note = AiidaConsoleService._test_split_caveat(
        55011, sibling={"pk": 55017, "label": "…_test", "count": 13}
    )
    assert "55017" in note
    assert "13" in note


def test_the_caveat_still_says_the_number_is_not_held_out():
    """The marker other code keys on must survive the rewording."""
    note = AiidaConsoleService._test_split_caveat(55011, sibling=None)
    assert AiidaConsoleService.NOT_HELD_OUT_MARKER in note


def test_the_stamped_extra_still_fires_on_the_new_wording(aiida_profile_clean):
    """`_stamp_metric_caveat` matches on the marker, so rewording must not break it."""
    from aiida import orm

    from aiida_pino.console.config import Settings

    service = AiidaConsoleService(Settings())
    node = orm.Dict(dict={"stands in for": "an eval"}).store()
    service._stamp_metric_caveat(node.pk, [AiidaConsoleService._test_split_caveat(55011, sibling=None)])
    assert node.base.extras.get("eval_held_out_test") is False
