"""The monitor shows the live beam-box MAE beside the step counter.

A step counter alone says a run is advancing, not whether it is advancing WELL. The
trainer already prints its best beam-box MAE to the same log the monitor tails for
`[step: N]`, so the number costs nothing extra to surface.

Kelvin discipline applies: `beambox_mae_T` is NORMALISED in the log, and this repo's
standing rule is to report Kelvin and never DERIVE the factor. So the parser returns the
raw value and the step it belongs to, and conversion is the caller's job using the factor
recorded in the model's own config. A normalised number is never labelled as Kelvin.
"""

from __future__ import annotations

from aiida_pino.console.monitor import last_beambox_mae, last_step

LOG = """
[12:05:42] - [step:      16000] record validators time:  8.711e+00s
[12:05:42] - best checkpoint: beambox_mae_T=0.153105 at step 16000
[12:16:19] - best checkpoint: beambox_mae_T=0.135708 at step 19500
[12:22:24] - best checkpoint: beambox_mae_T=0.134686 at step 21500
[12:54:16] - [step:      32000] reached maximum training steps, finished training!
"""


def test_it_reads_the_most_recent_beambox_mae():
    got = last_beambox_mae(LOG)
    assert got is not None
    assert got["value"] == 0.134686
    assert got["step"] == 21500


def test_it_is_the_LAST_not_the_best():
    """The log line says "best checkpoint", but the monitor must show the latest reading.

    Those coincide while the model improves and diverge the moment it stops. Reporting a
    stale best as the current number is how a plateau looks like progress.
    """
    log = LOG + "[13:00:00] - best checkpoint: beambox_mae_T=0.900000 at step 24000\n"
    assert last_beambox_mae(log)["value"] == 0.9


def test_no_validation_yet_is_none_not_zero():
    assert last_beambox_mae("[step:  10] training\n") is None
    assert last_beambox_mae("") is None


def test_it_tolerates_scientific_notation():
    assert last_beambox_mae("best checkpoint: beambox_mae_T=1.23e-02 at step 500")["value"] == 0.0123


def test_it_does_not_disturb_the_step_parser():
    """Both read the same tail; the step counter must be unchanged by this addition."""
    assert last_step(LOG) == 32000
