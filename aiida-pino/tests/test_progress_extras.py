"""How far a run got, from the node instead of from a deleted scratch log.

`exit_status 300`/`500` says "the step failed"; it does not separate a job that
died before step 1 (OOM, bad config -- nothing learned) from one that trained
for hours and fell over late. Measured 2026-08-10: 179 processes carried exit
300 and NOT ONE carried a step count, so the distinction had to be read out of
`[step: N]` lines in live-tail logs -- which live in the scratch workdir and are
deleted with it. After the backfill the question is one query: of 189 failed
training calcjobs, 116 never reached step 1 and 31 got past step 1000.
"""

from __future__ import annotations

from aiida_pino.parsers.common import stamp_progress_extras


class _Node:
    def __init__(self):
        self._extras: dict = {}

        class _Base:
            extras = self

        self.base = _Base()

    # minimal extras API
    def get(self, key, default=None):
        return self._extras.get(key, default)

    def set(self, key, value):
        self._extras[key] = value


def test_the_last_step_is_recorded():
    node = _Node()
    stamp_progress_extras(node, "[step: 10] loss 1.0\n[step: 20] loss 0.5\n[step: 30] loss 0.4\n")
    assert node._extras["max_step_reached"] == 30
    assert node._extras["step_lines_logged"] == 3


def test_a_job_that_never_stepped_records_zero_not_nothing():
    """Zero IS the finding -- it is the OOM-on-first-step signature.

    Leaving the key absent would make "died instantly" indistinguishable from
    "never asked", which is the ambiguity this exists to remove.
    """
    node = _Node()
    stamp_progress_extras(node, "CUDA out of memory\nTraceback...\n")
    assert node._extras["max_step_reached"] == 0


def test_an_existing_count_is_not_overwritten_by_a_stdout_without_steps():
    node = _Node()
    node._extras["max_step_reached"] = 4000
    stamp_progress_extras(node, "no step lines here")
    assert node._extras["max_step_reached"] == 4000, "a later empty log must not erase progress"


def test_no_stdout_stamps_nothing():
    node = _Node()
    stamp_progress_extras(node, None)
    stamp_progress_extras(node, "")
    assert node._extras == {}


def test_only_the_training_parser_tracks_steps():
    """An eval job logs no steps; stamping it 0 would read as "never got going"
    for a job that finished normally. 254 eval calcjobs were cleared of exactly
    that misleading zero."""
    from aiida_pino.parsers.eval import PinoEvalParser
    from aiida_pino.parsers.train import PinoTrainParser

    assert getattr(PinoTrainParser, "tracks_steps", False) is True
    assert getattr(PinoEvalParser, "tracks_steps", False) is False


def test_step_format_tolerates_spacing():
    node = _Node()
    stamp_progress_extras(node, "[step:5]\n[step:  1200]\n")
    assert node._extras["max_step_reached"] == 1200
