"""Three high-severity defects found in review, each with its evidence.

Every one fails silently in the damaging direction, which is why each gets a
test rather than a comment.
"""

from __future__ import annotations

import pytest
from aiida_pino.scripts.train_adapter import _config_amp_enabled, amp_enabled_in_profile


@pytest.mark.requires_lpno
def test_amp_comes_from_the_config_not_an_assumed_default():
    """`config_flow3D_GAOT_randompath.yaml` sets `amp.enabled: false` on purpose.

    GAOT's segment softmax over ~1.2M edges is an fp16 overflow site. Assuming
    True made plan_warm_start reject a restorable step-21000 checkpoint with
    "amp.enabled is true but scaler_state_dict is empty" (2026-08-09), which
    downgrades the resume to weights-only: the run trains on, and only the step
    counter and the one-cycle anneal position are wrong.
    """
    assert _config_amp_enabled("config_flow3D_GAOT_randompath") is False
    assert amp_enabled_in_profile({"config_name": "config_flow3D_GAOT_randompath"}) is False
    # A config that leaves amp on still reads on.
    assert amp_enabled_in_profile({"config_name": "config_flow3D_RIGNO_randompath"}) is True


def test_an_explicit_override_still_wins_over_the_config():
    profile = {"config_name": "config_flow3D_GAOT_randompath",
               "hydra_overrides": ["amp.enabled=true"]}
    assert amp_enabled_in_profile(profile) is True
    # Hydra's +/~ prefixes are override syntax, not part of the key.
    assert amp_enabled_in_profile(
        {"config_name": "config_flow3D_RIGNO_randompath",
         "hydra_overrides": ["+amp.enabled=false"]}) is False
    # Last override wins, as hydra does.
    assert amp_enabled_in_profile(
        {"config_name": "config_flow3D_RIGNO_randompath",
         "hydra_overrides": ["amp.enabled=false", "amp.enabled=true"]}) is True


def test_an_unreadable_config_keeps_the_old_answer():
    """Preserve historical behaviour rather than flipping every run we cannot see."""
    assert amp_enabled_in_profile({"config_name": "no_such_config_anywhere"}) is True
    assert amp_enabled_in_profile({}) is True


def test_the_canonical_layout_and_the_time_axis_stamp_agree():
    """The 2026-07-22 flip left three literal `time_as_dim: True` stamps behind.

    Each sat beside a `canonical_layout` saying the opposite, so every dataset
    imported after the flip carried self-contradictory metadata (one example:
    `mck_294`, base role, created five days later). Deriving both from
    DEFAULT_DERIVATION_PROFILE removes the second source of truth.
    """
    from aiida_pino.utils import canonicalize_import_metadata
    from aiida_pino.utils.dataset_derivation import DEFAULT_DERIVATION_PROFILE

    metadata = canonicalize_import_metadata({"source_folder_name": "x"})
    mode = metadata["time_axis_mode"]
    assert mode["time_as_batch"] is bool(DEFAULT_DERIVATION_PROFILE["time_as_batch"])
    assert mode["time_as_dim"] is bool(DEFAULT_DERIVATION_PROFILE["time_as_dim"])
    assert mode["time_as_batch"] != mode["time_as_dim"], "a layout cannot be both"
    assert metadata["canonical_layout"].startswith("time_batch"), metadata["canonical_layout"]


def test_no_literal_time_as_dim_stamp_survives():
    """A grep-level guard: the flip failed because copies were missed, not
    because the intent was unclear."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "aiida_pino"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if '"time_as_dim": True' in line or "'time_as_dim': True" in line:
                offenders.append(f"{path.relative_to(root)}:{i}")
    assert not offenders, f"literal canonical time-axis stamps: {offenders}"


def test_only_terminated_processes_are_sealed(aiida_profile_clean, monkeypatch):
    """ONE unsealed node aborts every full-profile export (2026-07-21).

    Sealing a LIVE process would freeze a node still writing to itself, which is
    a worse failure than a blocked export -- so live ones are counted and left.
    """
    from aiida_pino.utils import integrity

    class _Node:
        def __init__(self, pk, terminated):
            self.pk = pk
            self.is_terminated = terminated
            self.sealed = False

        def seal(self):
            self.sealed = True

    done, running = _Node(1, True), _Node(2, False)

    class _QB:
        def append(self, *a, **k):
            return self

        def all(self, flat=False):
            return [done, running]

    monkeypatch.setattr("aiida.orm.QueryBuilder", lambda *a, **k: _QB())

    dry = integrity.seal_orphaned_processes()
    assert dry["would_seal"] == [1] and dry["sealed"] == []
    assert not done.sealed, "a dry run must seal nothing"

    applied = integrity.seal_orphaned_processes(apply=True)
    assert applied["sealed"] == [1]
    assert applied["still_running"] == 1
    assert done.sealed and not running.sealed


def test_unconfigured_computer_is_flagged_by_the_doctor(aiida_profile_clean):
    """Authinfo is per-user and does not live in the node tables, so a node-level
    migration (SQLite -> PostgreSQL, 2026-07-21) left every computer unconfigured
    while the codes were intact -- and the first symptom was `verdi code list`
    printing "No codes found", the one thing that was fine. The doctor now says
    which computer to configure instead of leaving a 20-minute hunt.
    """
    from aiida import orm

    from aiida_pino.utils import integrity

    orm.Computer(
        label="ghost",
        hostname="localhost",
        transport_type="core.local",
        scheduler_type="core.direct",
    ).store()
    report = integrity.check_database(check_sources=False)
    configured = {c["label"]: c["configured"] for c in report["computers"]}
    assert configured["ghost"] is False
