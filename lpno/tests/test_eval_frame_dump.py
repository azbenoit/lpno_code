"""The eval's full-series frame dump (PINO_DUMP_FRAMES).

The preview keeps exactly ONE batch, which is enough to eyeball a model but not
to render a case as a video. These pin the env gate and the accumulation shape,
because a dump that silently keeps one batch looks identical to a working one
until you try to make 387 frames out of it.
"""
from __future__ import annotations

import numpy as np
import pytest

from lpno.models.pipeline.postprocessing import (
    frame_dump_stride,
    frame_dump_enabled,
    accumulate_frame_dump,
    write_frame_dump,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("False", False),
        ("1", True),
        ("true", True),
        ("yes", True),
    ],
)
def test_gate_defaults_off(monkeypatch, value, expected):
    """Off unless explicitly asked for: every eval writes this otherwise."""
    if value is None:
        monkeypatch.delenv("PINO_DUMP_FRAMES", raising=False)
    else:
        monkeypatch.setenv("PINO_DUMP_FRAMES", value)
    assert frame_dump_enabled() is expected


def test_accumulate_keeps_every_batch():
    """The defect this guards: keeping only the last (or first) batch."""
    acc: dict[str, list] = {}
    for i in range(5):
        preds = {"T": np.full((1, 1, 2, 2, 2), float(i)), "alpha": np.zeros((1, 1, 2, 2, 2))}
        truth = {"T": np.full((1, 1, 2, 2, 2), float(i) + 100.0), "alpha": np.ones((1, 1, 2, 2, 2))}
        accumulate_frame_dump(acc, preds, truth)

    assert set(acc) == {"pred_T", "pred_alpha", "true_T", "true_alpha"}
    assert len(acc["pred_T"]) == 5
    # order preserved, so frame i is really frame i
    assert [float(a.flat[0]) for a in acc["pred_T"]] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_accumulate_ignores_absent_keys():
    """A model without alpha must not raise, and must not invent the key."""
    acc: dict[str, list] = {}
    accumulate_frame_dump(acc, {"T": np.zeros((1, 1, 2, 2, 2))}, {"T": np.zeros((1, 1, 2, 2, 2))})
    assert set(acc) == {"pred_T", "true_T"}


def test_write_concatenates_along_frames(tmp_path):
    acc: dict[str, list] = {}
    for i in range(3):
        accumulate_frame_dump(
            acc,
            {"T": np.full((2, 1, 2, 2, 2), float(i))},
            {"T": np.full((2, 1, 2, 2, 2), float(i))},
        )
    out = write_frame_dump(acc, tmp_path)
    assert out is not None and out.exists()
    with np.load(out) as z:
        # 3 batches x 2 rows each = 6 frames, not 3 and not 2
        assert z["pred_T"].shape == (6, 1, 2, 2, 2)


def test_write_noop_when_empty(tmp_path):
    """Nothing accumulated must not leave a misleading empty archive."""
    assert write_frame_dump({}, tmp_path) is None
    assert not list(tmp_path.glob("*.npz"))


def test_accumulate_detaches_torch(tmp_path):
    """preds are torch on the plain path and numpy after a frame transfer."""
    torch = pytest.importorskip("torch")
    acc: dict[str, list] = {}
    accumulate_frame_dump(
        acc,
        {"T": torch.zeros((1, 1, 2, 2, 2), requires_grad=True)},
        {"T": torch.ones((1, 1, 2, 2, 2))},
    )
    assert isinstance(acc["pred_T"][0], np.ndarray)
    out = write_frame_dump(acc, tmp_path)
    assert out is not None and out.exists()


# --- striding (PINO_DUMP_FRAMES=N) ----------------------------------------- #

@pytest.mark.parametrize(
    "value,expected",
    [("1", 1), ("2", 2), ("10", 10), ("true", 1), ("0", 0), ("", 0), ("no", 0), ("banana", 1)],
)
def test_stride_parsing(monkeypatch, value, expected):
    """A typo must fall back to keeping every frame, not to silently disabling."""
    monkeypatch.setenv("PINO_DUMP_FRAMES", value)
    assert frame_dump_stride() == expected


def test_stride_keeps_every_nth_batch():
    acc: dict[str, list] = {}
    for i in range(10):
        vol = np.full((1, 1, 2, 2, 2), float(i))
        accumulate_frame_dump(acc, {"T": vol}, {"T": vol}, stride=3, index=i)
    # indices 0,3,6,9
    assert [float(a.flat[0]) for a in acc["pred_T"]] == [0.0, 3.0, 6.0, 9.0]


def test_stride_index_is_required():
    """Deriving the index from len(acc) breaks when a field is missing; refuse it."""
    with pytest.raises(ValueError):
        accumulate_frame_dump({}, {"T": np.zeros((1, 1, 2, 2, 2))},
                              {"T": np.zeros((1, 1, 2, 2, 2))}, stride=2)


def test_stride_one_keeps_everything():
    acc: dict[str, list] = {}
    for i in range(4):
        vol = np.full((1, 1, 2, 2, 2), float(i))
        accumulate_frame_dump(acc, {"T": vol}, {"T": vol}, stride=1, index=i)
    assert len(acc["pred_T"]) == 4


# --- case boundaries are DETECTED, not assumed ------------------------------- #

def _series(*lengths, base=100.0, jump=500.0):
    """Concatenated fake cases: smooth within a case, a jump between them."""
    out, level = [], base
    for n in lengths:
        out.append(np.linspace(level, level + 5.0, n).reshape(n, 1, 1, 1, 1))
        level += jump
    return np.concatenate(out, axis=0)


def test_detects_boundaries_of_concatenated_cases():
    from lpno.models.pipeline.postprocessing import _FRAME_DUMP_KEYS  # noqa: F401
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "rcv", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "render_case_video.py")
    rcv = importlib.util.module_from_spec(spec); spec.loader.exec_module(rcv)

    series = _series(191, 90, 125, 168)
    assert rcv.detect_case_boundaries(series) == [191, 281, 406]
    # matching counts pass
    rcv.check_segments(series, [191, 90, 125, 168])
    # the real defect: counts in the WRONG order must be refused, not rendered
    with pytest.raises(SystemExit):
        rcv.check_segments(series, [191, 167, 125, 91])
