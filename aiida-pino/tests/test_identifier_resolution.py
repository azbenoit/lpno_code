"""An error's remedy must be a command that runs.

`remote_dataset_symlink_entry` refuses a job whose dataset is not staged and
prints the command that fixes it — with the collection's UUID, because that is
what the guard has in hand. `stage-remote` declared its argument `type=int`, so
the prescribed command died with `argparse: invalid int value`.

That is worse than an error with no remedy: the remedy is the part an operator
copies without reading, and it was the part that failed. Nothing was staged, so
the next thing seen was the same guard firing again — which reads as "staging
did not work" rather than "staging never ran".

Fixed from both sides: the loader accepts a pk OR a uuid, and the guard prints
a pk.
"""

from __future__ import annotations

import pytest

from aiida_pino.console.config import Settings
from aiida_pino.console.service import AiidaConsoleService, NodeNotFoundError


def test_a_uuid_loads_as_well_as_a_pk(aiida_profile_clean):
    from aiida import orm

    node = orm.Dict(dict={"x": 1}).store()
    service = AiidaConsoleService(Settings())
    assert service._load_node(node.pk).pk == node.pk
    assert service._load_node(str(node.uuid)).pk == node.pk, "a uuid is a legitimate identifier"


def test_a_partial_uuid_prefix_also_loads(aiida_profile_clean):
    """AiiDA quotes short uuids everywhere; an operator will paste one."""
    from aiida import orm

    node = orm.Dict(dict={"x": 2}).store()
    service = AiidaConsoleService(Settings())
    assert service._load_node(str(node.uuid)[:8]).pk == node.pk


def test_an_ALL_DIGIT_uuid_prefix_is_not_mistaken_for_a_pk(aiida_profile_clean):
    """The ambiguous case, made deterministic instead of left to chance.

    An all-digit string is usually a pk, but a uuid prefix can be all digits
    too. Deciding from the shape alone routed such a prefix to a pk lookup and
    failed — INTERMITTENTLY, only when a random uuid happened to start with
    eight digits, which is the worst way for a bug to present.
    """
    from aiida import orm

    digit_prefixed = None
    for value in range(400):
        node = orm.Dict(dict={"probe": value}).store()
        if str(node.uuid)[:8].isdigit():
            digit_prefixed = node
            break
    if digit_prefixed is None:
        import pytest

        pytest.skip("no all-digit uuid prefix drawn in 400 nodes")

    service = AiidaConsoleService(Settings())
    prefix = str(digit_prefixed.uuid)[:8]
    assert prefix.isdigit()
    assert service._load_node(prefix).pk == digit_prefixed.pk, (
        f"{prefix!r} is a uuid prefix, not pk {int(prefix)}"
    )


def test_an_unknown_identifier_still_says_what_was_missing(aiida_profile_clean):
    service = AiidaConsoleService(Settings())
    with pytest.raises(NodeNotFoundError) as excinfo:
        service._load_node("not-a-real-uuid-at-all")
    assert "not-a-real-uuid-at-all" in str(excinfo.value)


def test_the_staging_guard_prints_an_identifier_the_cli_accepts(aiida_profile_clean):
    """The remedy must be copy-pasteable, which is the whole point of printing it."""
    from aiida import orm

    from aiida_pino.calculations.base import remote_dataset_symlink_entry

    computer = orm.Computer(
        label="remedy-box",
        hostname="localhost",
        transport_type="core.ssh_async",
        scheduler_type="core.direct",
        workdir="/tmp/aiida_run",
    )
    computer.store()

    class _Collection:
        uuid = "11111111-2222-3333-4444-555555555555"
        pk = 4242
        label = "ds"

        def export_runtime_arraydir(self, path, runtime_fields=None, allow_pack_index=True):
            raise NotImplementedError

    with pytest.raises(ValueError) as excinfo:
        remote_dataset_symlink_entry(_Collection(), computer, runtime_fields=None, target="train_dataset")
    message = str(excinfo.value)
    assert "stage-remote 4242" in message, "print the pk, which the CLI has always accepted"
