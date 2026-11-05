"""A moving window without a beam channel must blame the KEY SET, not the data.

Origin (2026-08-22): a six-channel moving-ref arm `[V_scan,P,H_norm,x,y,z]` failed with
"the case manifest has no laser path". The manifest was fine. `centres` is populated only when
`_scalar_path_channels` is non-empty (the `beam_x`/`beam_y` gate in `_case`), so a key set
without them leaves it None and the moving window raises. The message blamed the data, which sent
the investigation to a remote box and produced a wrong conclusion about its staged copy before the
same failure reproduced locally.
"""
import inspect

from lpno.models.pipeline import npz_lazy_store


def test_the_message_names_the_key_set_not_the_manifest():
    src = inspect.getsource(npz_lazy_store.NpzDirStore.read_case)
    i = src.index("moving_window needs per-frame beam centres")
    msg = src[i:i + 700]
    assert "scalar path channel" in msg, "must say what actually gates centres"
    assert "beam_x" in msg, "must name the channel that fixes it"
    assert "scalar_path_channels=" in msg, "must report the observed key set"


def test_centres_are_gated_on_the_scalar_path_channels():
    """Pins the gate itself, so the message cannot drift from the mechanism."""
    src = inspect.getsource(npz_lazy_store)
    i = src.index("centres = _beam_centres_over_time")
    preceding = src[:i]
    assert "if self._scalar_path_channels:" in preceding[-400:], (
        "centres must stay gated on the scalar path channels; if this gate moves, the error "
        "message above no longer describes the mechanism"
    )
