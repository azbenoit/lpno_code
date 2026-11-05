"""Unpinned runs choose their GPU when the gpu_spool seat admits them.

Older `gpu_spool seat` needed an explicit card index, so a submitter that pins
`cuda_device` hours ahead is guessing which card will be free when the job
finally starts. On 2026-07-28 twelve evals pinned to GPU 1 sat queued behind a
21 GiB training while GPU 0 was completely idle. An empty (or "auto") value now
reaches `gpu_spool seat --gpu auto`; the seat selects a card at admission and
injects `custom.cuda_device=N` into the payload then.
"""

from __future__ import annotations

import pytest

from aiida_pino.scripts import eval_adapter, train_adapter

ADAPTERS = pytest.mark.parametrize("adapter", [train_adapter, eval_adapter], ids=["train", "eval"])

SMI_TWO_CARDS = "0, 512\n1, 20480\n"


def _fake_smi(adapter, monkeypatch, stdout, returncode=0):
    class _Result:
        def __init__(self):
            self.returncode = returncode
            self.stdout = stdout

    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **k: _Result())


@ADAPTERS
def test_picks_the_emptiest_card(adapter, monkeypatch) -> None:
    _fake_smi(adapter, monkeypatch, SMI_TWO_CARDS)
    assert adapter.pick_free_gpu() == 1


@ADAPTERS
def test_falls_back_to_default_without_nvidia_smi(adapter, monkeypatch) -> None:
    def _boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(adapter.subprocess, "run", _boom)
    assert adapter.pick_free_gpu() == 0
    assert adapter.pick_free_gpu(default=1) == 1


@ADAPTERS
@pytest.mark.parametrize("unpinned", ["", "auto", "AUTO", None])
def test_unpinned_defers_to_gpu_spool_auto(adapter, monkeypatch, unpinned) -> None:
    _fake_smi(adapter, monkeypatch, SMI_TWO_CARDS)
    profile = {"cuda_device": unpinned}
    adapter.resolve_cuda_device(profile)
    assert profile["cuda_device"] == "auto", "unpinned run must defer selection to the seat"


@ADAPTERS
def test_explicit_index_is_always_honoured(adapter, monkeypatch) -> None:
    """An explicit pin is a deliberate choice (co-locating with a sibling run,
    reproducing a placement) and must never be second-guessed."""
    _fake_smi(adapter, monkeypatch, SMI_TWO_CARDS)
    profile = {"cuda_device": "0"}
    adapter.resolve_cuda_device(profile)
    assert profile["cuda_device"] == "0"


@ADAPTERS
def test_auto_reaches_the_seat_without_payload_override(adapter, monkeypatch) -> None:
    _fake_smi(adapter, monkeypatch, SMI_TWO_CARDS)
    monkeypatch.setattr(adapter.importlib.util, "find_spec", lambda name: object())
    profile = {"cuda_device": ""}
    adapter.resolve_cuda_device(profile)
    wrapped = adapter.wrap_with_gpu_seat(["python", "-m", "payload"], profile)
    assert wrapped[wrapped.index("--gpu") + 1] == "auto"
    assert "custom.cuda_device=auto" not in wrapped
    assert profile["cuda_device"] == "auto"
