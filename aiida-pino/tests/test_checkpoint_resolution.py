"""Which checkpoint a model resolves to, and whether anyone can see the choice.

`best_checkpoint.py` writes `u_network.best.pth` to protect against late
divergence. Nothing reads it: both candidates tie on every ranked field, so the
sort falls through to the raw path string, where `'0'` (0x30) sorts under `'b'`
(0x62) — the FINAL checkpoint wins by an accident of ASCII, not by intent.

Measured, val mae_T x3000: 58674 best 76.2 K @22000 vs final 143.6 K (1.88x);
58693 76.2 vs 141.7 (1.87x); 58702 29.4 vs 36.8 (1.25x). Both RIGNO arms
finished exit 0, so nothing flagged them.

These tests pin the CURRENT behaviour and make the choice visible. Preferring
best over final would silently change what every existing model node resolves
to, including ones whose published numbers were measured on the final
checkpoint — a behaviour change with reach, and not one to make as a side
effect of noticing it.
"""

from __future__ import annotations

from aiida_pino.data.pino_model import _checkpoint_sort_key, checkpoint_choice_note


def test_the_final_checkpoint_wins_on_a_string_tiebreak():
    """Documents the accident rather than endorsing it."""
    ranked = sorted(["u_network.best.pth", "u_network.0.pth"], key=_checkpoint_sort_key)
    assert ranked[0] == "u_network.0.pth"
    a, b = _checkpoint_sort_key("u_network.0.pth"), _checkpoint_sort_key("u_network.best.pth")
    assert a[:3] == b[:3], "they tie on every ranked field; only the raw path separates them"


def test_one_sort_key_serves_both_call_sites():
    """The key was duplicated in model_registry; fixing one would not be a fix."""
    import inspect

    from aiida_pino.utils import model_registry

    source = inspect.getsource(model_registry)
    assert '{".pth": 0, ".pt": 1, ".ckpt": 2}' not in source, "the second copy must be gone"
    assert "_checkpoint_sort_key" in source or "checkpoint_sort_key" in source


def test_a_shadowed_best_checkpoint_is_reported():
    """The choice must be visible without anyone knowing to look for it."""
    note = checkpoint_choice_note("u_network.0.pth", ["u_network.0.pth", "u_network.best.pth"])
    assert note is not None
    assert "u_network.best.pth" in note
    assert "best" in note.lower()


def test_no_note_when_there_is_no_alternative():
    assert checkpoint_choice_note("u_network.0.pth", ["u_network.0.pth"]) is None


def test_no_note_when_the_best_checkpoint_is_the_one_chosen():
    assert checkpoint_choice_note("u_network.best.pth", ["u_network.0.pth", "u_network.best.pth"]) is None
