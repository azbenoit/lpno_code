"""Choosing the machine (`code_label`) and the card (`cuda_device`) — item E.

`cuda_device` ships as `"0"` in DEFAULT_TRAINING_PROFILE and in every packaged
hydra config. On a two-card box that pins both trainings to card 0 and
serialises them: the box looks busy, both runs are slow, and nothing errors.
For a repo whose point is adding GPU capacity that default is exactly backwards.

So a remote submission promotes it to `"auto"` — `gpu_spool --gpu auto` already
ranks by least-committed card. "Promotes the default" is decided by whether the
caller actually SUPPLIED the key, not by comparing against `"0"`: an explicit
`"0"` and the default `"0"` are the same string, and only the caller's own dict
knows which happened.
"""

from __future__ import annotations

import pytest

from aiida_pino.utils.profiles import (
    DEFAULT_TRAINING_PROFILE,
    TRAINING_PROFILE_KEYS,
    resolve_cuda_device_for_target,
)


def test_code_label_is_an_accepted_training_profile_key():
    """Unknown keys pass validation silently, so the key set is the only gate."""
    assert "code_label" in TRAINING_PROFILE_KEYS
    assert DEFAULT_TRAINING_PROFILE["code_label"] is None


def test_a_local_submission_that_said_nothing_also_gets_auto():
    """Changed 2026-08-23; was `assert ... == "0"` under "nothing about local changes".

    That was a scoping decision when remote support landed, not a finding that local
    pinning is correct -- and it was costing throughput. Of the last 120
    PinoTrainCalculations, all 8 remote carried `auto` while 25 LOCAL ones were pinned to
    card 0, with six arms observed queued on card 0 while card 1 sat at 0/22528 MiB.
    `gpu_spool` resolves `auto` locally too (87 local runs in that window did).
    """
    assert resolve_cuda_device_for_target({}, "0", remote=False) == "auto"
    assert resolve_cuda_device_for_target({"cuda_device": "1"}, "1", remote=False) == "1"


def test_an_explicit_card_zero_is_never_promoted():
    """The whole reason the check reads the SUPPLIED dict rather than the merged value.

    An explicit "0" and a defaulted "0" are the same string once merged, so promoting on
    the merged value would silently overrule someone who asked for card 0 on purpose.
    """
    for remote in (True, False):
        assert resolve_cuda_device_for_target({"cuda_device": "0"}, "0", remote=remote) == "0"


def test_a_present_but_blank_choice_is_not_a_choice():
    """The console assigns cuda_device on every submission, so blank means unchosen."""
    for remote in (True, False):
        assert resolve_cuda_device_for_target({"cuda_device": ""}, "", remote=remote) == "auto"
        assert resolve_cuda_device_for_target({"cuda_device": "  "}, "  ", remote=remote) == "auto"


def test_a_remote_submission_that_said_nothing_gets_auto():
    assert resolve_cuda_device_for_target({}, "0", remote=True) == "auto"


def test_a_remote_submission_that_named_a_card_keeps_it():
    """An explicit choice is a choice, including an explicit card 0."""
    assert resolve_cuda_device_for_target({"cuda_device": "1"}, "1", remote=True) == "1"
    assert resolve_cuda_device_for_target({"cuda_device": "0"}, "0", remote=True) == "0"


def test_an_explicit_auto_survives_everywhere():
    assert resolve_cuda_device_for_target({"cuda_device": "auto"}, "auto", remote=True) == "auto"
    assert resolve_cuda_device_for_target({"cuda_device": "auto"}, "auto", remote=False) == "auto"


def test_the_promotion_reads_the_callers_dict_not_the_merged_value():
    """The merged value is `"0"` in both cases; only the caller's dict differs."""
    supplied = resolve_cuda_device_for_target({"cuda_device": "0"}, "0", remote=True)
    defaulted = resolve_cuda_device_for_target({"run_name": "x"}, "0", remote=True)
    assert supplied == "0" and defaulted == "auto"


def test_the_doctor_names_the_right_configure_command_per_transport(aiida_profile_clean):
    """Printing the local command for an SSH computer sends you to a dead end."""
    from aiida import orm

    from aiida_pino.utils.integrity import check_database, unconfigured_computer_hint

    computer = orm.Computer(
        label="doctor-box",
        hostname="localhost",
        transport_type="core.ssh_async",
        scheduler_type="core.direct",
        workdir="/tmp/aiida_run",
    )
    computer.store()

    report = check_database(check_sources=False)
    row = next(r for r in report["computers"] if r["label"] == "doctor-box")
    assert row["transport_type"] == "core.ssh_async"
    assert row["configured"] is False, "a bare SSH computer has no authinfo yet"

    hint = unconfigured_computer_hint(row)
    assert "verdi computer configure core.ssh_async doctor-box" in hint
    assert "core.local" not in hint


def test_the_hint_falls_back_to_local_for_a_row_without_a_transport():
    """Older report rows carry no transport_type; the hint must still be valid."""
    from aiida_pino.utils.integrity import unconfigured_computer_hint

    assert "configure core.local old-box" in unconfigured_computer_hint({"label": "old-box"})


def test_a_resubmit_override_retargets_the_training(aiida_profile_clean):
    """`--override code_label=<code>` must decide where a rerun executes.

    `_resubmit_code_label` returns the ORIGINAL run's code, and reading that
    line alone suggests an override cannot take effect — a reviewer concluded
    exactly that. It can: `_training_target` prefers the profile's `code_label`,
    and this pins that so the next refactor cannot quietly make the inference
    true.
    """
    from aiida_pino.console.config import Settings
    from aiida_pino.console.service import AiidaConsoleService, InvalidRequestError

    service = AiidaConsoleService(Settings())
    with pytest.raises(InvalidRequestError) as excinfo:
        service._training_target(
            "original-code@local",
            {"run_name": "x"},
            {"code_label": "override-code@box", "cuda_device": "0"},
        )
    message = str(excinfo.value)
    assert "override-code@box" in message, "the override must be the code that is resolved"
    assert "original-code@local" not in message


def test_without_an_override_the_original_code_is_used(aiida_profile_clean):
    """The default has to keep working: a rerun reproduces its environment."""
    from aiida_pino.console.config import Settings
    from aiida_pino.console.service import AiidaConsoleService, InvalidRequestError

    service = AiidaConsoleService(Settings())
    with pytest.raises(InvalidRequestError) as excinfo:
        service._training_target("original-code@local", {"run_name": "x"}, {"cuda_device": "0"})
    assert "original-code@local" in str(excinfo.value)


# --- the promotion could never fire from the console -----------------------
# 2026-08-21, reported from a two-card compute box: two arms both landed on GPU 0 while GPU 1
# sat idle ~45 min; the starved arm reached step 200 with a 930-second-old log
# while its neighbour was at 1400 and healthy. Nothing errored.
#
# `resolve_cuda_device_for_target` decides "was this defaulted?" by asking
# whether the caller's dict CONTAINS the key. That is the right question, but
# the console form answers it wrong: `app.js` defaults the input to "0" and
# then assigns `tp.cuda_device = cuda.value` unconditionally, so the key is
# ALWAYS present and the value is always at least "0". The guard therefore
# returned "0" for every console submission and the "auto" promotion was dead
# code on the path that matters most.

def test_a_blank_cuda_device_counts_as_not_supplied():
    """Present-but-empty is how a form says "the user did not choose"."""
    from aiida_pino.utils.profiles import resolve_cuda_device_for_target

    for blank in ("", "   ", None):
        assert resolve_cuda_device_for_target(
            {"cuda_device": blank}, "0", remote=True
        ) == "auto", f"blank {blank!r} must not read as an explicit choice"


def test_an_explicit_card_is_still_honoured():
    """The other half: someone who asks for card 0 on purpose keeps card 0."""
    from aiida_pino.utils.profiles import resolve_cuda_device_for_target

    assert resolve_cuda_device_for_target({"cuda_device": "0"}, "0", remote=True) == "0"
    assert resolve_cuda_device_for_target({"cuda_device": "1"}, "1", remote=True) == "1"


def test_a_blank_local_choice_never_becomes_an_empty_device():
    """Blank must not become an EMPTY device on a local box.

    The merged value really is "" there -- a blank supplied value overrides the default
    during the merge -- so returning `merged_value` unchanged would send an empty card
    downstream. That concern is unchanged; only the fallback moved. It was the shipped
    default "0" while promotion was remote-only, and is now "auto" for local too
    (2026-08-23), which satisfies the same requirement without pinning the card.
    """
    from aiida_pino.utils.profiles import resolve_cuda_device_for_target

    resolved = resolve_cuda_device_for_target({"cuda_device": ""}, "", remote=False)
    assert resolved and resolved.strip(), "an empty device must never reach the adapter"
    assert resolved == "auto"
    assert resolve_cuda_device_for_target({"cuda_device": "1"}, "1", remote=False) == "1"


def test_the_console_form_does_not_send_a_card_nobody_chose():
    """Server-side leniency is useless if the client keeps inventing a value."""
    import pathlib

    app_js = (pathlib.Path(__file__).resolve().parents[1] / "web" / "public" / "app.js").read_text()
    assert "tp.cuda_device = cuda.value;" not in app_js, (
        "app.js assigns cuda_device unconditionally, so the training profile always "
        "carries a card the user may never have chosen and the auto-promotion cannot fire"
    )


def test_the_shipped_default_card_is_auto_not_a_pinned_zero():
    """The console form pre-fills this field and submits it, so the default IS the choice.

    A shipped "0" arrives at `resolve_cuda_device_for_target` as an explicit choice, which
    it correctly honours -- so promotion alone cannot fix the console path, and only the
    default can. Measured 2026-08-23: 25 of the last 120 local PinoTrainCalculations were
    pinned to card 0, six arms queued there while card 1 sat at 0/22528 MiB reserved.
    """
    from aiida_pino.utils.profiles import DEFAULT_TRAINING_PROFILE

    assert DEFAULT_TRAINING_PROFILE["cuda_device"] == "auto"
