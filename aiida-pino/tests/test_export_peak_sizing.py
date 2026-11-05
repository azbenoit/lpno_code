"""Export workers are sized from what a MEMBER's derivation will hold in RAM, not a flat constant.

2026-09-11: ONE export worker reached 74 GB RSS -- 56 % of the machine's RAM -- on a
THREE-member collection (`derived_fields: ['conduction_field']`,
`moving_ref_frame: true`, `time_avg_window: 30`). MemAvailable fell to 19 GB, swap filled
7/7, three monitoring processes were OOM-killed and two live trainings came close.
`EXPORT_PER_MEMBER_GB = 6.0` was calibrated on ~3.3 GB-uncompressed members with no
averaging; this member is 11.8 GB uncompressed and its derivation multiplies that by 6.3.
The worker cap (`AIIDA_PINO_EXPORT_WORKERS=2`) was in force and irrelevant: the blow-up is
per member, and no cap bounds a single member.

The rule now sizes each member from its own `array_shapes` and its derivation, serialises to
one worker when a member alone exceeds the budget (a warning, never a refusal -- user, this
week: refusals must stay bypassable), and prints the expected peak beside the member count
where a submitter reads it.
"""

from __future__ import annotations

from types import SimpleNamespace

from aiida_pino.utils.export_pool import (
    EXPORT_RESERVED_GB,
    export_worker_count,
    per_member_export_peak_bytes,
)

GB = 1e9


def _member(shape=(689, 1, 75, 50, 60), arrays=19):  # 19 x 620 MB = the real 11.8 GB
    return SimpleNamespace(get_array_shapes=lambda: {f"f{i}": list(shape) for i in range(arrays)})


MIXED = {"derived_fields": ["conduction_field"], "moving_ref_frame": True, "time_avg_window": 30}


def test_the_measured_member_is_sized_at_its_measured_peak():
    """The measured member: 11.8 GB uncompressed. It peaked at 74 GB on 2026-09-11 and
    at 12.56 GB on 2026-09-14 with the current `derived_member_worker` (fresh process, empty
    conduction cache) -- and another member, 18.01 GB, at 19.22 GB the same day: a factor of 1.07
    both times. The 74 GB belonged to that day's export path; sizing a steady member at
    6.5x clamped a 61-member warm to one worker (11 h instead of ~2 h, self-improve). The
    estimate must not be below what was measured, and must not be absurd either."""
    peak = per_member_export_peak_bytes([_member()], MIXED)
    assert 12.56 * GB <= peak <= 20 * GB, f"{peak / GB:.1f} GB against 12.56 GB measured"


def test_a_derivation_that_keeps_the_frame_axis_is_still_sized_as_heavy():
    """Unmeasured with today's worker, so it keeps the conservative factor."""
    full_time = {"derived_fields": ["conduction_field"]}
    assert per_member_export_peak_bytes([_member()], full_time) >= 6 * 11.8 * GB
    assert per_member_export_peak_bytes([_member()], {"moving_ref_frame": True}) >= 6 * 11.8 * GB


def test_a_plain_copy_is_sized_far_below_the_full_time_heavy_case():
    plain = per_member_export_peak_bytes([_member()], {"crop_z_top": 10, "stride": [2, 2, 1]})
    heavy = per_member_export_peak_bytes([_member()], {"derived_fields": ["conduction_field"]})
    assert plain < heavy / 2
    assert plain >= 11.8 * GB, "at least the raw arrays are held once"


def test_the_largest_member_sets_the_figure_not_the_first():
    small, big = _member(shape=(10, 1, 4, 4, 4)), _member()
    assert per_member_export_peak_bytes([small, big], MIXED) == per_member_export_peak_bytes([big], MIXED)


def test_a_member_that_alone_exceeds_the_budget_serialises_to_one_worker(monkeypatch):
    monkeypatch.delenv("AIIDA_PINO_EXPORT_WORKERS", raising=False)
    avail = 60 * GB
    n = export_worker_count(cpu_count=20, n_members=3, per_member_bytes=74 * GB, available_bytes=avail)
    assert n == 1


def test_workers_are_the_budget_divided_by_the_member_peak(monkeypatch):
    monkeypatch.delenv("AIIDA_PINO_EXPORT_WORKERS", raising=False)
    avail = (EXPORT_RESERVED_GB + 30) * GB
    assert export_worker_count(cpu_count=20, n_members=10, per_member_bytes=7 * GB, available_bytes=avail) == 4


def test_the_env_override_is_still_clamped_by_the_member_peak(monkeypatch):
    """`AIIDA_PINO_EXPORT_WORKERS=2` was in force on 2026-09-11 and could not help; an
    override must never admit more workers than members that fit."""
    monkeypatch.setenv("AIIDA_PINO_EXPORT_WORKERS", "2")
    assert export_worker_count(cpu_count=20, n_members=3, per_member_bytes=74 * GB, available_bytes=60 * GB) == 1


def test_the_report_names_the_expected_peak(monkeypatch, tmp_path):
    from aiida_pino.utils.presubmit import local_build_report

    monkeypatch.setenv("AIIDA_PINO_RUNTIME_CACHE", str(tmp_path))

    class _Collection:
        pk, uuid = 7, "cafe0000-0000-0000-0000-000000000000"

        def get_dataset_metadata(self):
            return {"derivation_profile": dict(MIXED), "dataset_role": "derived"}

        def iter_member_simulations(self):
            return [SimpleNamespace(get_simulation_metadata=lambda: {"array_shapes": {"T": [689, 1, 75, 50, 60]}},
                                    get_array_shapes=lambda: {f"f{i}": [689, 1, 75, 50, 60] for i in range(19)})]

        def export_runtime_arraydir(self, *a, **k):
            raise AssertionError("a report must never export")

    report = local_build_report(_Collection())
    assert report["peak_ram_bytes"] == per_member_export_peak_bytes(_Collection().iter_member_simulations(), MIXED)
    assert report["peak_ram_bytes"] >= 12.56 * GB  # the 2026-09-14 measurement of this member
    assert "peak" in report["detail"].lower() and "GB" in report["detail"]
