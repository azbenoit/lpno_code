"""Multi-timescale thermal-memory channels.

`path_dose` (tau -> inf) and `path_dose_recent` (tau = 1 ms) are a two-number summary
of the exposure history, which aliases a variable-speed path: measured on
ti64_varspeed150, every speed segment (median dwell 190.8 us) is shorter than the one
tau, so ~5.2 segments fall inside a single window and "fast then slow" is
indistinguishable from "slow then fast". The bank supplies the missing basis
functions.
"""
import numpy as np
import pytest

from lpno.custom_utils.npz_datasource import (
    _DOSE_RECENT_TAU_S,
    _DOSE_TAU_BANK_S,
    PATH_CHANNEL_NAMES,
    _leaky_dose,
    _tau_bank_for_prefix,
)


def test_bank_channels_are_registered_for_every_prefix():
    for prefix in ('', 'sigma', 'sigma_exp'):
        for name in _tau_bank_for_prefix(prefix):
            assert name in PATH_CHANNEL_NAMES, name


def test_bank_does_not_duplicate_the_existing_1ms_channel():
    """`path_dose_recent` already is tau = 1 ms; a duplicate would waste a channel."""
    assert _DOSE_RECENT_TAU_S == pytest.approx(1e-3)
    assert all(tau != pytest.approx(_DOSE_RECENT_TAU_S) for tau in _DOSE_TAU_BANK_S.values())


def test_leaky_dose_decays_at_the_configured_rate():
    """After a single unit deposit, the accumulator must fall as exp(-t/tau)."""
    nt, dt_s, tau = 60, 1e-5, 1e-4
    footprint = np.zeros((nt, 2, 2), dtype=np.float32)
    footprint[0] = 1.0
    dt = np.full(nt, dt_s)
    on = np.ones(nt)
    out = _leaky_dose(footprint, dt, on, tau)
    peak = out[0, 0, 0]
    # 10 frames later is exactly one tau
    assert out[10, 0, 0] / peak == pytest.approx(np.exp(-1.0), rel=0.05)
    assert out[20, 0, 0] / peak == pytest.approx(np.exp(-2.0), rel=0.05)


def test_short_tau_forgets_faster_than_long_tau():
    """The point of the bank: different tau must give DIFFERENT histories, or the
    extra channels carry no information."""
    nt, dt_s = 80, 1e-5
    footprint = np.zeros((nt, 1, 1), dtype=np.float32)
    footprint[0] = 1.0
    dt, on = np.full(nt, dt_s), np.ones(nt)
    short = _leaky_dose(footprint, dt, on, 3e-5)
    long = _leaky_dose(footprint, dt, on, 3e-3)
    # normalised by their own peaks, the short-tau channel has decayed much further
    assert short[40, 0, 0] / short[0, 0, 0] < long[40, 0, 0] / long[0, 0, 0]


def test_bank_distinguishes_fast_then_slow_from_slow_then_fast():
    """The aliasing this exists to remove.

    Two exposure histories with the SAME total deposit, ordered oppositely. A single
    tau can coincide for both; the bank must separate them.
    """
    nt, dt_s = 60, 1e-5
    dt, on = np.full(nt, dt_s), np.ones(nt)

    fast_then_slow = np.zeros((nt, 1, 1), dtype=np.float32)
    fast_then_slow[5:15] = 1.0          # dense early
    fast_then_slow[40:50] = 0.25        # sparse late

    slow_then_fast = np.zeros((nt, 1, 1), dtype=np.float32)
    slow_then_fast[5:15] = 0.25
    slow_then_fast[40:50] = 1.0

    assert fast_then_slow.sum() == pytest.approx(slow_then_fast.sum())

    # final-frame signature across the bank
    sig_a = [float(_leaky_dose(fast_then_slow, dt, on, tau)[-1, 0, 0])
             for tau in sorted(_DOSE_TAU_BANK_S.values())]
    sig_b = [float(_leaky_dose(slow_then_fast, dt, on, tau)[-1, 0, 0])
             for tau in sorted(_DOSE_TAU_BANK_S.values())]
    # they must differ somewhere in the bank
    assert any(abs(a - b) > 1e-4 * max(abs(a), abs(b), 1e-9) for a, b in zip(sig_a, sig_b)), \
        (sig_a, sig_b)


def test_normalisation_keeps_channels_comparable_across_tau():
    """Dividing by tau keeps a sustained exposure at O(1) for every tau, so the bank
    does not hand the network four channels of wildly different scale."""
    nt, dt_s = 400, 1e-5
    footprint = np.ones((nt, 1, 1), dtype=np.float32)  # continuous exposure
    dt, on = np.full(nt, dt_s), np.ones(nt)
    finals = [float(_leaky_dose(footprint, dt, on, tau)[-1, 0, 0])
              for tau in sorted(_DOSE_TAU_BANK_S.values())]
    assert max(finals) / min(finals) < 5.0, finals
