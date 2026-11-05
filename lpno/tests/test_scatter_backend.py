"""GAOT must not select a CPU-only torch_scatter build and die on the first forward.

A CPU-only `torch-scatter` still exposes `torch_scatter.scatter`, imports cleanly,
reports the correct version and passes `pip check`. The old guard was
`hasattr(torch_scatter, 'scatter')`, which such a build satisfies, so GAOT took the
torch_scatter path, printed "Using torch_scatter.scatter for aggregation", and died at
`reduce="max"` with "Not compiled with CUDA support". RIGNO needs only scatter_add,
which falls through to native torch, so it trained perfectly on the same install --
a box that is half working, which is far slower to diagnose than a dead one.

torch-scatter is sdist-only; its setup.py compiles the CUDA extensions only when
CUDA_HOME resolves at build time and silently takes the CPU branch otherwise, so a
fresh box hits this easily (the tell is four `.so` in torch_scatter/ instead of eight).

Found while commissioning a new compute box, 2026-08-19.
"""
from __future__ import annotations

import pytest
import torch

from lpno.models.gaot_layers._scatter_backend import resolve_scatter
from lpno.models.gaot_layers.scatter_native import scatter_native


class _FakeModule:
    """Stands in for torch_scatter; `broken_on_cuda` mimics the CPU-only build."""

    def __init__(self, broken_on_cuda=False, has_scatter=True):
        self.broken_on_cuda = broken_on_cuda
        self.calls = 0
        if has_scatter:
            self.scatter = self._scatter

    def _scatter(self, src, index, dim=0, dim_size=None, reduce="sum"):
        self.calls += 1
        if self.broken_on_cuda:
            raise RuntimeError("Not compiled with CUDA support")
        return scatter_native(src, index, dim=dim, dim_size=dim_size, reduce=reduce)


def test_missing_module_uses_native():
    fn, backend = resolve_scatter(None, cuda_available=True)
    assert backend == "native" and fn is scatter_native


def test_module_without_scatter_uses_native():
    fn, backend = resolve_scatter(_FakeModule(has_scatter=False), cuda_available=True)
    assert backend == "native" and fn is scatter_native


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a CUDA device: the default probe runs a CUDA segment max, so without "
    "CUDA it cannot reach the module and the backend correctly falls back to native",
)
def test_working_cuda_build_is_used():
    mod = _FakeModule()
    fn, backend = resolve_scatter(mod, cuda_available=True)
    assert backend == "torch_scatter" and fn is mod.scatter
    assert mod.calls == 1, "the CUDA op must actually be exercised, not just hasattr'd"


def test_cpu_only_build_falls_back_instead_of_dying(capsys):
    """The defect: hasattr passes, the CUDA op does not."""
    mod = _FakeModule(broken_on_cuda=True)
    fn, backend = resolve_scatter(mod, cuda_available=True)
    assert backend == "native" and fn is scatter_native
    assert "CPU-only" in capsys.readouterr().out


def test_no_cuda_present_does_not_probe(capsys):
    """CPU-only hosts (the website stub runtime) must not be probed or warned at."""
    mod = _FakeModule(broken_on_cuda=True)   # would raise if probed
    fn, backend = resolve_scatter(mod, cuda_available=False)
    assert backend == "torch_scatter" and mod.calls == 0
    assert "CPU-only" not in capsys.readouterr().out


# --- the fallback must genuinely implement max, or falling back would be worse ---

def test_native_max_is_exact_not_a_sum_placeholder():
    src = torch.tensor([[1.0], [5.0], [2.0], [-3.0], [-7.0]])
    index = torch.tensor([0, 0, 1, 2, 2])
    out = scatter_native(src, index, dim=0, dim_size=3, reduce="max")
    assert torch.allclose(out, torch.tensor([[5.0], [2.0], [-3.0]])), out
    # a sum placeholder would give [[6.0], [2.0], [-10.0]]


def test_native_max_handles_all_negative_segments():
    """include_self=False matters: a zero-init out must not clamp the max at 0."""
    src = torch.tensor([[-4.0], [-9.0]])
    index = torch.tensor([0, 0])
    out = scatter_native(src, index, dim=0, dim_size=1, reduce="max")
    assert torch.allclose(out, torch.tensor([[-4.0]])), out


def test_native_sum_matches_torch():
    src = torch.arange(6, dtype=torch.float32).view(6, 1)
    index = torch.tensor([0, 0, 1, 1, 2, 2])
    out = scatter_native(src, index, dim=0, dim_size=3, reduce="sum")
    assert torch.allclose(out, torch.tensor([[1.0], [5.0], [9.0]]))


def test_force_native_override(monkeypatch, capsys):
    """A healthy box must be able to exercise the fallback without being broken."""
    monkeypatch.setenv("PINO_FORCE_NATIVE_SCATTER", "1")
    mod = _FakeModule()
    fn, backend = resolve_scatter(mod, cuda_available=True)
    assert backend == "native" and fn is scatter_native
    assert mod.calls == 0, "must not even probe when the override is set"
    assert "PINO_FORCE_NATIVE_SCATTER" in capsys.readouterr().out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_real_attention_path_runs_on_cuda_through_native_scatter(monkeypatch):
    """The half nobody had watched: GAOT's segment softmax on CUDA via scatter_native.

    `reduce="max"` on CUDA tensors is the exact call a CPU-only torch-scatter build
    dies on. This drives the real `IntegralTransform._segment_softmax_pyg` with the
    native backend on GPU tensors, and checks the softmax is valid and differentiable
    -- so the fallback is proven to carry the attention path, not merely to be
    selected.
    """
    from lpno.models.gaot_layers import integral_transform as it

    monkeypatch.setattr(it, "scatter", scatter_native)
    layer = it.IntegralTransform(
        channel_mlp_layers=[8, 8], use_attn=True, coord_dim=3,
        attention_type="dot_product",
    ).cuda()

    scores = torch.randn(12, 1, device="cuda", requires_grad=True)
    index = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3], device="cuda")
    weights = layer._segment_softmax_pyg(scores, index, dim_size=4)

    assert weights.shape == scores.shape
    assert torch.isfinite(weights).all()
    seg_sums = scatter_native(weights, index, dim=0, dim_size=4, reduce="sum")
    assert torch.allclose(seg_sums, torch.ones_like(seg_sums), atol=1e-5), seg_sums
    weights.sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
