"""A spot weld scans at 0 m/s, and that zero is the measurement.

The conditioning guard exists to stop a *dropped* scan speed becoming a silent
zero channel. Applied without exception it also made stationary (spot/dwell)
welds impossible to store: the adapter learned to allow them, but
`PinoSimulationData.from_npz_file` refused independently, so the case died at
the node constructor with "refusing to store silent-zero/NaN P/V_scan".
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from aiida_pino.data.pino_simulation import check_laser_scalars, is_stationary_case


def test_a_spot_weld_is_recognised_from_either_source():
    assert is_stationary_case({"laser_stationary": True}) is True
    assert is_stationary_case({"flow3d_case_manifest": {"derived": {"laser_stationary": True}}}) is True
    # the explicit field wins, so a re-import can correct an old manifest
    assert is_stationary_case(
        {"laser_stationary": False, "flow3d_case_manifest": {"derived": {"laser_stationary": True}}}
    ) is False
    assert is_stationary_case({}) is False
    assert is_stationary_case(None) is False


def test_zero_scan_speed_is_accepted_for_a_spot_weld():
    meta = {"laser_power": 120.0, "V_scan": 0.0, "laser_stationary": True}
    check_laser_scalars(meta, where="spot")  # must not raise


def test_a_dwell_case_with_no_scan_speed_at_all_is_normalised_to_zero():
    """A dwell config omits `laser.scan_speed_m_per_s` entirely, so the manifest
    carries no value — for a declared-stationary run the physical truth is 0."""
    meta = {"laser_power": 120.0, "laser_stationary": True}
    check_laser_scalars(meta, where="dwell")
    assert meta["V_scan"] == 0.0


def test_zero_scan_speed_is_still_refused_for_a_moving_case():
    """The original bug this guard was written for: a dropped speed silently
    becoming a zero conditioning channel."""
    with pytest.raises(ValueError, match="not marked stationary"):
        check_laser_scalars({"laser_power": 120.0, "V_scan": 0.0}, where="moving")


def test_zero_power_is_refused_even_for_a_spot_weld():
    """Stationary excuses a zero *speed*, never a zero power — a spot weld with
    no laser is not an experiment."""
    with pytest.raises(ValueError, match="laser_power"):
        check_laser_scalars({"laser_power": 0.0, "V_scan": 0.0, "laser_stationary": True}, where="spot")


def test_nan_is_refused_in_every_case():
    for meta in (
        {"laser_power": 120.0, "V_scan": float("nan"), "laser_stationary": True},
        {"laser_power": float("nan"), "V_scan": 0.0, "laser_stationary": True},
    ):
        with pytest.raises(ValueError):
            check_laser_scalars(meta, where="nan case")


def test_the_stored_flag_resolves_from_the_manifest_too():
    """The check accepted V=0 from the manifest; the stored flag must agree.

    `is_stationary_case` falls back to `flow3d_case_manifest.derived`, so ingest
    accepted the 11 SS304L spot welds. The storage path read
    `normalized.get("laser_stationary", False)` with no such fallback, so every
    one of them landed in AiiDA as `laser_stationary=False` beside `V_scan=0.0` --
    a combination the guard is written to reject, describing a moving laser at
    zero speed. Normalizing the flag once keeps the attribute, the extra and
    `simulation_metadata` consistent with the value the guard actually used.
    """
    from aiida_pino.data.pino_simulation import PinoSimulationData

    normalized = {
        "laser_power": 355.0,
        "V_scan": 0.0,
        "flow3d_case_manifest": {"derived": {"laser_stationary": True}},
    }
    PinoSimulationData._normalize_metadata(
        normalized, aliases={}, array_shapes={}, requested_fields_default=[],
        time_values_array=None,
    )
    assert normalized["laser_stationary"] is True


def test_a_moving_case_stays_false():
    from aiida_pino.data.pino_simulation import PinoSimulationData

    normalized = {"laser_power": 200.0, "V_scan": 0.6}
    PinoSimulationData._normalize_metadata(
        normalized, aliases={}, array_shapes={}, requested_fields_default=[],
        time_values_array=None,
    )
    assert normalized["laser_stationary"] is False
