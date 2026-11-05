"""A beam-following window pinned at the domain edge must say so.

`beam_following_x_slice` keeps a constant width by clamping the window inside the
domain, which is right for batching and silent about its consequence: the beam is then
NOT at index `behind`. On the anchorbox anchor replicates (collection 83525, 2026-09-01)
the laser overran to ix 86-90 of 90, the 41+20 window pinned at start 29 for every tail
frame, and the beam sat at window index 57-61. The derived steady target read as "cavity
5-12 cells AHEAD of the beam"; that was diagnosed as a frozen beam track, the track was
fixed (0b04707), the model's inputs moved -- and the target came back byte-identical,
because the clamp decides the window once the beam is within `ahead` cells of the edge.
"""
import numpy as np

from lpno.models.pipeline import frame_transfer as ft

X0, DX, NX = -4.45e-4, 1e-5, 90
BEHIND, AHEAD = ft.legacy_window_split(61)   # (41, 20)


def _slice_for(ix):
    return ft.beam_following_x_slice(beam_cx=X0 + ix * DX, x0=X0, dx=DX, nx=NX,
                                     behind=BEHIND, ahead=AHEAD)


def test_edge_clamp_is_announced_once_and_slice_unchanged(capsys):
    ft._CLAMP_ANNOUNCED.clear()
    moving = _slice_for(86)      # overrun:  unclamped start would be 45 -> announces
    frozen = _slice_for(72)      # path end: unclamped start would be 31 -> same geometry, silent
    assert frozen.start == moving.start == NX - 61 == 29, "both pinned at the same start"
    out = capsys.readouterr().out
    assert out.count("[moving-frame] beam-following window clamped") == 1, "once per geometry"
    assert "window index 57, not 41" in out


def test_centred_window_is_silent(capsys):
    ft._CLAMP_ANNOUNCED.clear()
    sl = _slice_for(50)
    assert sl.start == 50 - BEHIND
    assert "[moving-frame]" not in capsys.readouterr().out
