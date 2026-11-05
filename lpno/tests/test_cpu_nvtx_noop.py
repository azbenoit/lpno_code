"""A training allowed onto the CPU must not die on physicsnemo's NVTX ranges.

physicsnemo-sym 2.2 calls `torch.cuda.nvtx.range_push`/`range_pop` on every training
step and every graph forward. On a CPU-only torch those raise "NVTX functions not
installed", so `PINO_ALLOW_CPU=1` let a training start and then killed it at step 0
(found writing reproducibility/train_evaluate.ipynb, which shipped a sitecustomize shim
for it). `configure_device` now makes them no-ops on such a build, and only there.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from lpno.models.pipeline import nvtx_guard
from lpno.models.pipeline.config_loading import configure_device


def _cfg():
    return OmegaConf.create({"custom": {"cuda_device": 0, "cuda_visible_devices": None}})


def _raise(*args, **kwargs):
    raise RuntimeError("NVTX functions not installed. Are you sure you have a CUDA build?")


@pytest.fixture
def cpu_run(monkeypatch):
    """A run with no CUDA device, allowed onto the CPU, whose torch NVTX is the stub."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("PINO_ALLOW_CPU", "1")
    # Recorded by monkeypatch first, so teardown restores the real functions whatever
    # configure_device installs over them.
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", _raise)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", _raise)
    return monkeypatch


def test_cpu_only_build_gets_noop_nvtx(cpu_run):
    cpu_run.setattr(nvtx_guard, "nvtx_unavailable", lambda: True)
    configure_device(_cfg(), set_device=True)
    assert torch.cuda.nvtx.range_push("Training iteration") == 0
    assert torch.cuda.nvtx.range_pop() == 0


def test_a_build_with_nvtx_is_left_alone(cpu_run):
    cpu_run.setattr(nvtx_guard, "nvtx_unavailable", lambda: False)
    configure_device(_cfg(), set_device=True)
    assert torch.cuda.nvtx.range_push is _raise


def test_the_condition_is_the_torch_build():
    """No CUDA build and no NVTX extension: exactly the stub case."""
    expected = torch.version.cuda is None and not hasattr(torch._C, "_nvtx")
    assert nvtx_guard.nvtx_unavailable() is expected


@pytest.mark.skipif(torch.version.cuda is not None, reason="needs a CPU-only torch build")
def test_physicsnemo_graph_forward_runs_on_a_cpu_build(monkeypatch):
    """The real call site: physicsnemo's Graph.forward wraps every node in an NVTX range."""
    from physicsnemo.sym.graph import Graph
    from physicsnemo.sym.key import Key
    from physicsnemo.sym.node import Node

    # The stub behaviour, explicitly: an earlier test in this process may already have
    # run configure_device, which patches the module for the rest of the process.
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", _raise)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", _raise)

    class _Twice(torch.nn.Module):
        def forward(self, invar):
            return {"y": 2 * invar["x"]}

    node = Node(["x"], ["y"], _Twice(), name="twice")
    graph = Graph([node], [Key("x")], [Key("y")])
    with pytest.raises(RuntimeError, match="NVTX"):
        graph({"x": torch.ones(2, 1)})
    assert nvtx_guard.make_nvtx_noop_on_cpu_build()
    assert torch.equal(graph({"x": torch.ones(2, 1)})["y"], 2 * torch.ones(2, 1))
