"""eval_frame_transfer='steady': average per-case windowed predictions, score once.

The three frame treatments are comparable only in a shared frame. The steady frame's
shared support is the beam-following window; a lab or moving-ref model's predictions are
pushed into it by (1) cropping each frame to the window (the existing moving_ref
transfer) and (2) averaging the last `eval_frame_avg` frames of each case -- the same
construction as the steady dataset, applied to predictions. Truth is averaged the same
way, so the metric compares average against average.

'fixed_window' is the steady-NATIVE side of the same table: a steady-trained model
already predicts one averaged frame on the legacy 61-wide window; cropping it to the
same 32+16 window (a pure x-slice, beam pinned by construction) puts all three columns
on one support.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lpno.models.pipeline.frame_transfer import (
    fixed_window_slice,
    steady_entry_selection,
)
from lpno.models.pipeline.postprocessing import run_lazy_dataset_metrics


# --------------------------------------------------------------------------- #
# the pure helpers
# --------------------------------------------------------------------------- #
def test_steady_entry_selection_takes_the_last_window_per_case():
    entries = [(0, t) for t in range(5)] + [(1, t) for t in range(4)]
    idx, cases = steady_entry_selection(entries, avg_window=3)
    picked = [entries[i] for i in idx]
    assert picked == [(0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (1, 3)]
    assert cases == [0, 0, 0, 1, 1, 1]


def test_steady_entry_selection_refuses_a_short_case():
    entries = [(0, 0), (0, 1), (1, 0)]
    with pytest.raises(ValueError, match="avg_window"):
        steady_entry_selection(entries, avg_window=2)


def test_steady_entry_selection_refuses_full_case_entries():
    with pytest.raises(ValueError, match="per-frame"):
        steady_entry_selection([(0, None), (1, None)], avg_window=2)


def test_fixed_window_slice_matches_the_legacy_split():
    # 61-wide legacy window splits 41 behind / 20 ahead; a 32+16 crop starts at 41-32.
    sl = fixed_window_slice(nx=61, behind=32, ahead=16)
    assert (sl.start, sl.stop) == (9, 57)


def test_fixed_window_slice_refuses_a_window_wider_than_the_grid():
    with pytest.raises(ValueError, match="width"):
        fixed_window_slice(nx=40, behind=32, ahead=16)


# --------------------------------------------------------------------------- #
# the eval loop, end to end on a stub
# --------------------------------------------------------------------------- #
class _IdentityT(torch.nn.Module):
    """Predicts T = the frame index channel, so the average is checkable exactly."""

    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, invar):
        return {"T": invar["frame_val"].clone()}


class _CaseDS(torch.utils.data.Dataset):
    """Two cases x 4 frames on a 12-cell x line; beam fixed mid-domain.

    truth T = frame index + 10*case; the model predicts frame index + 10*case + 1,
    so after any amount of averaging the error is exactly 1 everywhere.
    """

    def __init__(self):
        self.entries = [(c, t) for c in range(2) for t in range(4)]
        x1d = np.linspace(0.0, 11.0, 12, dtype=np.float32)
        self.x = np.broadcast_to(x1d.reshape(1, 12, 1, 1), (1, 12, 2, 2)).astype(np.float32)
        z1d = np.array([-1.0, 0.5], dtype=np.float32)
        self.z = np.broadcast_to(z1d, (1, 12, 2, z1d.size)).astype(np.float32)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        case, t = self.entries[i]
        val = np.full((1, 12, 2, 2), t + 10.0 * case, dtype=np.float32)
        invar = {
            "x": torch.from_numpy(self.x.copy()),
            "z": torch.from_numpy(self.z.copy()),
            "beam_x": torch.from_numpy(np.full((1, 12, 2, 2), 5.0, dtype=np.float32)),
            "frame_val": torch.from_numpy(val + 1.0),
        }
        return invar, {"T": torch.from_numpy(val)}, 0


def _run_cfg(**over):
    base = dict(
        inference_batch_size=3,     # deliberately NOT a divisor of the window
        material_params={"T_s": 1.0, "T_l": 2.0},
        ref_scales={"T": 1.0},
        alpha_masking=False,
        eval_frame_transfer="steady",
        eval_frame_behind=4,
        eval_frame_ahead=2,
        eval_frame_avg=2,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_steady_transfer_scores_one_averaged_frame_per_case(tmp_path, monkeypatch):
    monkeypatch.delenv("PINO_EVAL_FRAME_STRIDE", raising=False)
    m = run_lazy_dataset_metrics(_IdentityT(), _CaseDS(), _run_cfg(), tmp_path, ["T"])
    stamp = m.errors["eval_frame_transfer"]
    assert stamp["target"] == "steady"
    assert stamp["cases_scored"] == 2.0
    assert stamp["avg_window"] == 2.0
    # identity + constant offset: the averaged error is exactly 1 everywhere
    assert m.errors["T"]["abs_err_mean"] == pytest.approx(1.0, abs=1e-6)


def test_steady_transfer_crops_to_the_window(tmp_path, monkeypatch):
    """A huge error OUTSIDE the window must not reach the metric."""
    monkeypatch.delenv("PINO_EVAL_FRAME_STRIDE", raising=False)

    class _EdgePoison(_CaseDS):
        def __getitem__(self, i):
            invar, out, w = super().__getitem__(i)
            t = out["T"].clone()
            t[..., 0, :, :] = 1.0e6      # x=0: outside the window around beam cell 5
            return invar, {"T": t}, w

    m = run_lazy_dataset_metrics(_IdentityT(), _EdgePoison(), _run_cfg(), tmp_path, ["T"])
    # inside the 4+2 window the error is exactly 1; the poisoned column is cropped away
    assert m.errors["T"]["abs_err_mean"] == pytest.approx(1.0, abs=1e-6)


def test_steady_transfer_refuses_a_frame_stride(tmp_path, monkeypatch):
    monkeypatch.setenv("PINO_EVAL_FRAME_STRIDE", "3")
    with pytest.raises(ValueError, match="stride"):
        run_lazy_dataset_metrics(_IdentityT(), _CaseDS(), _run_cfg(), tmp_path, ["T"])


def test_steady_transfer_needs_case_entries(tmp_path, monkeypatch):
    monkeypatch.delenv("PINO_EVAL_FRAME_STRIDE", raising=False)

    class _Flat(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            raise AssertionError("should refuse before reading")

    with pytest.raises(ValueError, match="entries"):
        run_lazy_dataset_metrics(_IdentityT(), _Flat(), _run_cfg(), tmp_path, ["T"])


def test_steady_transfer_skips_the_crop_when_already_windowed(tmp_path, monkeypatch):
    """ref -> steady: a moving-ref-native eval serves ALREADY windowed batches.

    The loader emitted the beam window (x beam-relative, width = behind+ahead), so
    cropping again -- with absolute beam_x against relative x -- would compute a
    nonsense slice. When the batch width already equals the window, average only.
    """
    monkeypatch.delenv("PINO_EVAL_FRAME_STRIDE", raising=False)

    class _WindowedDS(_CaseDS):
        """Same two cases, but pre-cropped to the 6-cell window with relative x."""

        def __init__(self):
            super().__init__()
            x1d = np.linspace(-4.0, 1.0, 6, dtype=np.float32)
            self.x = np.broadcast_to(x1d.reshape(1, 6, 1, 1), (1, 6, 2, 2)).astype(np.float32)
            z1d = np.array([-1.0, 0.5], dtype=np.float32)
            self.z = np.broadcast_to(z1d, (1, 6, 2, z1d.size)).astype(np.float32)

        def __getitem__(self, i):
            case, t = self.entries[i]
            val = np.full((1, 6, 2, 2), t + 10.0 * case, dtype=np.float32)
            invar = {
                "x": torch.from_numpy(self.x.copy()),
                "z": torch.from_numpy(self.z.copy()),
                "beam_x": torch.from_numpy(np.full((1, 6, 2, 2), 5.0, dtype=np.float32)),
                "frame_val": torch.from_numpy(val + 1.0),
            }
            return invar, {"T": torch.from_numpy(val)}, 0

    m = run_lazy_dataset_metrics(_IdentityT(), _WindowedDS(), _run_cfg(), tmp_path, ["T"])
    stamp = m.errors["eval_frame_transfer"]
    assert stamp["cases_scored"] == 2.0
    assert m.errors["T"]["abs_err_mean"] == pytest.approx(1.0, abs=1e-6)
