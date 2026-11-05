"""A warm start must inherit its donor's architecture, not today's defaults.

2026-08-24. `physicsnemo/sym/trainer.py` loads a
warm-start checkpoint under a BARE `except`: a state-dict mismatch logs one line and the
run trains from random init to completion, producing a normal-looking curve. One arm
(6000 steps) was lost that way and only caught because someone asked whether it was
training from scratch.

No mistake is required to trigger it. That arm was submitted 15 minutes before the commits
that moved `num_graph_variants` 1 -> 8, then queued overnight and resolved its config at
RUN time -- an 8-variant model built from a 1-variant donor. Neither run was wrong; the
default moved between them.
"""

from __future__ import annotations

from aiida_pino.utils.warm_start import (
    architecture_overrides_from_donor,
    hydra_value,
)


class _Repo:
    def __init__(self, text):
        self._text = text

    def get_object_content(self, name):
        if name != ".hydra/config.yaml" or self._text is None:
            raise FileNotFoundError(name)
        return self._text


class _Donor:
    def __init__(self, text):
        self.base = type("B", (), {"repository": _Repo(text)})()


DONOR_YAML = """
arch:
  rigno:
    processor_steps: 18
    num_graph_variants: 1
    grad_checkpointing: true
custom:
  FNO_input_keys: [T, P, V_scan]
  unrelated: 7
"""


def test_it_inherits_the_donors_architecture():
    additions, conflicts = architecture_overrides_from_donor(_Donor(DONOR_YAML), [])
    assert "arch.rigno.processor_steps=18" in additions
    assert "arch.rigno.num_graph_variants=1" in additions
    assert "arch.rigno.grad_checkpointing=true" in additions
    assert "custom.FNO_input_keys=[T,P,V_scan]" in additions
    assert not conflicts


def test_the_overrides_are_PLAIN_never_plus_prefixed():
    """`remap_plus_args_to_custom` rewrites any `+`-prefixed arg into `custom.`.

    `++arch.rigno.processor_steps=18` would set `custom.arch.rigno.processor_steps` -- read
    by nothing -- while the real architecture keeps today's default. That is exactly how
    the VRAM smoke's cadence clamp was inert for a day (2026-08-22).
    """
    additions, _ = architecture_overrides_from_donor(_Donor(DONOR_YAML), [])
    assert additions and not any(item.startswith("+") for item in additions)


def test_unrelated_custom_keys_are_not_inherited():
    """Only the must-match set. Inheriting the donor's whole `custom` block would drag in
    its dataset, its run name and its channels."""
    additions, _ = architecture_overrides_from_donor(_Donor(DONOR_YAML), [])
    assert not any("unrelated" in item for item in additions)


def test_an_explicit_caller_override_wins_and_is_reported_as_a_conflict():
    """Deviating from the donor is legitimate; doing it silently is not."""
    additions, conflicts = architecture_overrides_from_donor(
        _Donor(DONOR_YAML), ["arch.rigno.processor_steps=8"]
    )
    assert not any(item.startswith("arch.rigno.processor_steps=") for item in additions)
    assert any("processor_steps" in c and "donor has 18" in c for c in conflicts)


def test_a_caller_override_that_MATCHES_the_donor_is_not_a_conflict():
    _, conflicts = architecture_overrides_from_donor(
        _Donor(DONOR_YAML), ["arch.rigno.processor_steps=18"]
    )
    assert not conflicts


def test_a_donor_with_no_archived_config_is_not_an_error():
    """Older donors archived none; that must not block a submission that worked before."""
    assert architecture_overrides_from_donor(_Donor(None), []) == ([], [])


def test_list_values_carry_no_spaces():
    """A space in an override splits the argument and hydra sees a truncated list."""
    assert " " not in hydra_value([1, 2, 3])
    assert hydra_value(True) == "true" and hydra_value(False) == "false"


def test_filesystem_paths_are_never_inherited():
    """`arch.rigno.cache_dir` lives in the arch block but is a PATH, not architecture.

    It cannot change a state dict, and a donor that recorded an absolute path from another
    machine would point this run at a directory that does not exist here. A real donor
    carries `arch.rigno.cache_dir=outputs/rigno_graphs`.
    """
    donor = _Donor(
        "arch:\n  rigno:\n    cache_dir: outputs/rigno_graphs\n    processor_steps: 8\n"
    )
    additions, _ = architecture_overrides_from_donor(donor, [])
    assert "arch.rigno.processor_steps=8" in additions
    assert not any("cache_dir" in item for item in additions)
