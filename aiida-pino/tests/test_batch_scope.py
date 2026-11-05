"""Batch scope and the calcjob pointer: executed, but never asserted.

Round 7, major 4: `flow3d_batch.py` and `train_model.build_model` were covered
in the sense that a crash would have been caught, but nothing checked the
values. "A crash would be caught; a wrong value would not."
"""

from __future__ import annotations

from aiida import orm



def test_a_batch_records_what_it_planned(aiida_profile_clean, monkeypatch):
    """With max_concurrent_runs=1, `called_descendants` shows one child.

    So "4 arms, 1 done" was not a queryable property -- the planned work lived
    only inside the `run_configs` input list and had to be counted by hand.
    """
    from aiida_pino.workflows.flow3d_batch import Flow3DBatchWorkChain

    # A real stored node, so `base.extras` is the genuine API rather than a
    # hand-rolled stand-in that can drift from it.
    node = orm.Dict(dict={}).store()
    statuses: list[str] = []
    node.set_process_status = statuses.append  # type: ignore[attr-defined]

    chain = object.__new__(Flow3DBatchWorkChain)
    monkeypatch.setattr(type(chain), "node", property(lambda self: node), raising=False)
    monkeypatch.setattr(
        type(chain), "ctx", property(lambda self: type("C", (), {"cursor": 2})()), raising=False
    )

    node.base.extras.set("planned_runs", 4)
    chain._record_progress()

    assert node.base.extras.get("completed_runs") == 2
    assert statuses == ["run 3 of 4"], statuses
