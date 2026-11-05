"""`++key=value` (Hydra force-add) must survive the custom-namespace remapper.

The remapper exists to turn a bare `+foo=bar` into the `custom.` namespace. It
stripped exactly ONE leading `+`, so a force-add override arrived as
`key = "+custom.seed"`, failed the `startswith("custom.")` test, and was
re-qualified into `custom.+custom.seed` — emitted as
`+custom.+custom.seed=42`, which dies in Hydra's lexer:

    LexerNoViableAltException: +custom.+custom.seed=42

Measured 2026-08-10: aiida-pino began injecting the resolved seed as
`++custom.seed=<n>` so the archived config stops being silent about it, and
every training launched through that path died at startup on a config that does
not declare `custom.seed`. Both halves are individually reasonable; the
interaction is the bug.

`++` means "add or override, whatever the config says", so a force-add is
always valid and never needs the declared-path lookup — it only needs its key
qualified into the custom namespace when it is not already.
"""

from __future__ import annotations

import pytest

from lpno.models.argv_custom_remap import remap_plus_args_to_custom


def _remap(*args):
    return remap_plus_args_to_custom(argv=list(args), default_config_name="config_flow3D_RIGNO")


def test_force_add_on_an_already_qualified_key_is_untouched():
    """The exact string aiida-pino emits."""
    assert _remap("++custom.seed=42") == ["++custom.seed=42"]


def test_force_add_on_a_bare_key_is_qualified_once():
    assert _remap("++seed=42") == ["++custom.seed=42"]


def test_force_add_never_doubles_the_namespace():
    for arg in ("++custom.seed=42", "++seed=42", "++custom.nested.knob=1"):
        out = _remap(arg)[0]
        assert "custom.+" not in out, f"double-qualified: {out}"
        assert out.count("custom.") == 1, f"namespace doubled: {out}"


def test_single_plus_behaviour_is_unchanged():
    """The original contract: bare +foo=bar joins the custom namespace."""
    assert _remap("+seed=42") == ["+custom.seed=42"]
    assert _remap("+custom.seed=42") == ["+custom.seed=42"]


def test_declared_paths_still_lose_their_plus():
    """A key the active config declares is a plain override, not an add."""
    # config_flow3D_RIGNO declares custom.alpha_masking
    assert _remap("+alpha_masking=true") == ["custom.alpha_masking=true"]
    assert _remap("+custom.alpha_masking=true") == ["custom.alpha_masking=true"]


def test_non_plus_args_pass_through():
    assert _remap("custom.seed=42", "training.max_steps=10") == [
        "custom.seed=42",
        "training.max_steps=10",
    ]


def test_the_produced_overrides_actually_parse():
    """End of the chain: whatever we emit must survive Hydra's lexer."""
    from hydra.core.override_parser.overrides_parser import OverridesParser

    parser = OverridesParser.create()
    for arg in ("++custom.seed=42", "++seed=42", "+seed=42", "+custom.seed=42"):
        emitted = _remap(arg)
        parser.parse_overrides(emitted)  # raises if malformed
