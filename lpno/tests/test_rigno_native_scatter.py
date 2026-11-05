"""RIGNO trains without torch_scatter, and its native scatter_add equals the kernel.

`rigno_arch.py` imported `torch_scatter.scatter_add` unconditionally, so on an install
without torch-scatter (a CPU-only environment, a fresh box) RIGNO could not even be
imported, though its only scatter op is a sum that native torch covers. It now falls
back to `Tensor.scatter_add_`, as GAOT already does (`gaot_layers/_scatter_backend.py`).
Found writing reproducibility/train_evaluate.ipynb.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import lpno.models.rigno_arch as ra

SRC = Path(__file__).resolve().parents[1] / "src"


def _case(B=2, E=37, L=5, n_recv=9, seed=0, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    receivers = torch.randint(0, n_recv - 2, (E,), generator=gen)   # the last two get no edge
    msg = torch.randn(B, E, L, generator=gen, dtype=dtype)
    idx = receivers.view(1, -1, 1).expand(B, -1, L)
    return msg, idx, receivers, n_recv


def test_native_scatter_add_is_the_per_receiver_sum():
    msg, idx, receivers, n_recv = _case(dtype=torch.float64)
    out = ra._scatter_add_native(msg, idx, dim=1, dim_size=n_recv)
    expected = torch.zeros(msg.shape[0], n_recv, msg.shape[2], dtype=msg.dtype)
    for e, r in enumerate(receivers.tolist()):
        expected[:, r] += msg[:, e]
    assert torch.allclose(out, expected, rtol=0, atol=1e-12)
    assert (out[:, -2:] == 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_matches_torch_scatter_exactly(dtype):
    torch_scatter = pytest.importorskip("torch_scatter")
    msg, idx, _, n_recv = _case(dtype=dtype, E=4000, L=16, n_recv=300)
    ours = ra._scatter_add_native(msg, idx, dim=1, dim_size=n_recv)
    theirs = torch_scatter.scatter_add(msg, idx, dim=1, dim_size=n_recv)
    assert torch.equal(ours, theirs)


def test_an_index_not_shaped_like_src_is_refused():
    msg, idx, _, n_recv = _case()
    with pytest.raises(ValueError, match="shaped like src"):
        ra._scatter_add_native(msg, idx[:, :, :1], dim=1, dim_size=n_recv)


def test_interaction_step_is_identical_on_both_backends(monkeypatch):
    torch_scatter = pytest.importorskip("torch_scatter")
    torch.manual_seed(0)
    step = ra.InteractionStep(latent=8, hidden_layers=1, update_sender=True)
    senders = torch.randint(0, 6, (40,))
    receivers = torch.randint(0, 5, (40,))
    x_send, x_recv, edge = torch.randn(2, 6, 8), torch.randn(2, 5, 8), torch.randn(2, 40, 8)
    outs = []
    for fn in (torch_scatter.scatter_add, ra._scatter_add_native):
        monkeypatch.setattr(ra, "scatter_add", fn)
        outs.append(step(x_send, x_recv, edge, senders, receivers, 5))
    for a, b in zip(outs[0], outs[1]):
        assert torch.equal(a, b)


def test_a_whole_rigno_forward_and_backward_agree_on_both_backends(monkeypatch):
    """Forward outputs and every parameter gradient, on a 8x6x5 grid."""
    torch_scatter = pytest.importorskip("torch_scatter")
    from conftest import GRAPH_CACHE_DIR
    from physicsnemo.sym.key import Key

    torch.manual_seed(0)
    model = ra.RIGNOArch(
        input_keys=[Key(k) for k in ["V_scan", "P", "H_norm", "x", "y", "z"]],
        output_keys=[Key("T"), Key("alpha")], dimension=3, node_latent_size=16,
        edge_latent_size=16, processor_steps=2, subsample_factor=6.0, k_p2r=3, k_r2p=3,
        p_edge_masking=0.0, cache_dir=GRAPH_CACHE_DIR)
    inputs = {k: torch.randn(2, 1, 8, 6, 5) for k in ["V_scan", "P", "H_norm", "x", "y", "z"]}
    results = []
    for fn in (torch_scatter.scatter_add, ra._scatter_add_native):
        monkeypatch.setattr(ra, "scatter_add", fn)
        model.zero_grad()
        out = model(inputs)
        (out["T"].square().mean() + out["alpha"].mean()).backward()
        results.append(({k: v.detach().clone() for k, v in out.items()},
                        [p.grad.clone() for p in model.parameters() if p.grad is not None]))
    (out_a, grads_a), (out_b, grads_b) = results
    for key in out_a:
        assert torch.equal(out_a[key], out_b[key]), key
    assert grads_a and len(grads_a) == len(grads_b)
    for a, b in zip(grads_a, grads_b):
        assert torch.allclose(a, b, rtol=0, atol=1e-6)


def _backend_in_fresh_process(prelude: str, env_extra: dict | None = None) -> str:
    # A fresh interpreter, so this module's RIGNOArch is never re-created under other tests.
    code = (prelude + "\nimport lpno.models.rigno_arch as ra\n"
            "assert (ra.scatter_add is ra._scatter_add_native) == (ra.SCATTER_BACKEND == 'native')\n"
            "print(ra.SCATTER_BACKEND)")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(SRC)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]),
        **(env_extra or {})}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout.strip().splitlines()[-1]


def test_rigno_imports_and_falls_back_without_torch_scatter():
    """With torch_scatter unimportable the module still loads, on the native backend."""
    assert _backend_in_fresh_process("import sys; sys.modules['torch_scatter'] = None") == "native"


def test_the_native_backend_can_be_forced():
    assert _backend_in_fresh_process("", {"PINO_FORCE_NATIVE_SCATTER": "1"}) == "native"
