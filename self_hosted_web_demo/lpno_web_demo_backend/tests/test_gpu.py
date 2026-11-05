from __future__ import annotations

import os

import pytest

from app import gpu


class _FakeCuda:
    def __init__(self, available: bool = True, count: int = 2) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


class _FakeTorch:
    def __init__(self, available: bool = True, count: int = 2) -> None:
        self.cuda = _FakeCuda(available, count)

    @staticmethod
    def device(spec: str) -> str:
        return f"device({spec})"


def test_auto_picks_gpu_with_most_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "_free_mem_per_gpu", lambda: [1000, 8000])  # gpu0 fuller
    assert gpu.pick_device(_FakeTorch(count=2), "auto") == "device(cuda:1)"


def test_auto_picks_gpu0_when_freer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "_free_mem_per_gpu", lambda: [9000, 500])
    assert gpu.pick_device(_FakeTorch(count=2), "auto") == "device(cuda:0)"


def test_explicit_device_forces_it() -> None:
    assert gpu.pick_device(_FakeTorch(count=2), "1") == "device(cuda:1)"
    assert gpu.pick_device(_FakeTorch(count=2), "cuda:0") == "device(cuda:0)"


def test_cpu_and_no_cuda_give_cpu() -> None:
    assert gpu.pick_device(_FakeTorch(count=2), "cpu") == "device(cpu)"
    assert gpu.pick_device(_FakeTorch(available=False), "auto") == "device(cpu)"


def test_falls_back_when_smi_count_mismatches_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    # nvidia-smi sees 4 GPUs but torch sees 2 (CUDA_VISIBLE_DEVICES) -> don't guess.
    monkeypatch.setattr(gpu, "_free_mem_per_gpu", lambda: [1, 2, 3, 4])
    assert gpu.pick_device(_FakeTorch(count=2), "auto") == "device(cuda)"


def test_single_gpu_uses_plain_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "_free_mem_per_gpu", lambda: [8000])
    assert gpu.pick_device(_FakeTorch(count=1), "auto") == "device(cuda)"


def test_falls_back_when_smi_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "_free_mem_per_gpu", lambda: None)
    assert gpu.pick_device(_FakeTorch(count=2), "auto") == "device(cuda)"


def test_bootstrap_auto_pins_freest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpu, "_free_mem_per_gpu", lambda: [1000, 8000])
    assert gpu.bootstrap_cuda_visible_devices("auto") == "1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_bootstrap_respects_existing_cvd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert gpu.bootstrap_cuda_visible_devices("auto") is None
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"  # left untouched


def test_bootstrap_explicit_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert gpu.bootstrap_cuda_visible_devices("cuda:1") == "1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_bootstrap_cpu_hides_all(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert gpu.bootstrap_cuda_visible_devices("cpu") == ""
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_explicit_index_is_logical_after_bootstrap_pinned_it() -> None:
    # bootstrap_cuda_visible_devices("1") pins CUDA_VISIBLE_DEVICES=1, so torch
    # sees ONE gpu and numbers it 0. Asking for the *physical* index again is an
    # invalid ordinal, which killed startup with
    # "torch.AcceleratorError: CUDA error: invalid device ordinal".
    # Physical vs logical is the split lpno keeps as
    # cuda_device / cuda_local_device (config_loading.py BUG-10).
    assert gpu.pick_device(_FakeTorch(count=1), "1") == "device(cuda:0)"
    assert gpu.pick_device(_FakeTorch(count=1), "cuda:3") == "device(cuda:0)"


def test_explicit_index_still_honoured_when_in_range() -> None:
    # An externally-set CUDA_VISIBLE_DEVICES=0,1 leaves both visible: cuda:1 is
    # a real ordinal and must not be rewritten to 0.
    assert gpu.pick_device(_FakeTorch(count=2), "1") == "device(cuda:1)"
    assert gpu.pick_device(_FakeTorch(count=4), "cuda:3") == "device(cuda:3)"
