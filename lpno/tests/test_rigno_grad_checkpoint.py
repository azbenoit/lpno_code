"""Gradient-checkpointing equivalence for the RIGNO processor.

CPU-only and license-free. ``grad_checkpointing=True`` recomputes each processor
step in backward instead of storing its activations, to fit multi-step
autoregressive unrolling (k>=3) in 24 GB. The recompute must be numerically
identical to the plain forward/backward, otherwise it silently changes training.
"""
import importlib
import sys

import pytest
import torch
from conftest import GRAPH_CACHE_DIR
from physicsnemo.sym.key import Key

import lpno.models.rigno_arch as rigno_arch
from lpno.models.rigno_arch import RIGNOArch

# Sibling test modules install bare `torch` ModuleType stubs at run time (e.g.
# tests/test_pointwise_grid_restore.py) and don't restore them. Our tests call
# torch.utils.checkpoint.checkpoint(), which lazily `import torch._dynamo` at call
# time and would resolve against that stub ("'torch' is not a package"). Snapshot
# the real torch modules now — collection time, before any test runs, so torch is
# still real — and restore them before each test.
import torch.utils.checkpoint  # noqa: E402,F401  (cache the real submodule)
importlib.import_module("torch._dynamo")  # cache the real lazy dep too
_REAL_TORCH_MODULES = {
    name: mod for name, mod in list(sys.modules.items())
    if name == "torch" or name.startswith("torch.")
}


@pytest.fixture(autouse=True)
def _restore_real_torch():
    for name, mod in _REAL_TORCH_MODULES.items():
        sys.modules[name] = mod
    yield


def _model(**kw) -> RIGNOArch:
    ik = [Key(k) for k in ["V_scan", "P", "H_norm", "x", "y", "z"]]
    ok = [Key("T"), Key("alpha")]
    # p_edge_masking=0.0 -> training-mode forwards are deterministic, so the only
    # difference between the two runs below is checkpointing itself.
    defaults = dict(dimension=3, node_latent_size=16, edge_latent_size=16,
                    processor_steps=3, subsample_factor=6.0, k_p2r=3, k_r2p=3,
                    p_edge_masking=0.0, cache_dir=GRAPH_CACHE_DIR)
    defaults.update(kw)
    return RIGNOArch(input_keys=ik, output_keys=ok, **defaults)


def _inputs(grid, B):
    nx, ny, nz = grid
    return {k: torch.randn(B, 1, nx, ny, nz) for k in ["V_scan", "P", "H_norm", "x", "y", "z"]}


def _forward_backward(m, x):
    m.zero_grad(set_to_none=True)
    y = m(x)
    loss = sum(v.pow(2).mean() for v in y.values())
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
    return {k: v.detach().clone() for k, v in y.items()}, grads


def test_checkpoint_default_off():
    assert _model()._grad_checkpoint is False


def test_checkpoint_is_actually_invoked(monkeypatch):
    # Guard against the flag silently becoming a no-op: torch's checkpoint must be
    # called once per processor step PLUS once each for the encoder (p2r) and
    # decoder (r2p), and never when off. The encoder/decoder are where the memory
    # actually is -- they message-pass over the physical grid (600k nodes at the
    # flow3D resolution), while the processor runs on the subsampled regional mesh.
    # Checkpointing only the processor left a 24 GB card OOMing inside dec_step.
    calls = {"n": 0}
    real = rigno_arch.checkpoint

    def _spy(fn, *args, **kw):
        calls["n"] += 1
        return real(fn, *args, **kw)

    monkeypatch.setattr(rigno_arch, "checkpoint", _spy)
    x = _inputs((8, 6, 5), 1)

    m_off = _model(grad_checkpointing=False, processor_steps=3).train()
    _forward_backward(m_off, x)
    assert calls["n"] == 0

    m_on = _model(grad_checkpointing=True, processor_steps=3).train()
    _forward_backward(m_on, x)
    assert calls["n"] == 3 + 2  # processor_steps + encoder + decoder


_FILM_KEYS = ["V_scan", "P", "H_norm"]


def _count_film_calls(m, calls):
    """Wrap every per-step FiLM module so we can count its forward passes."""
    for film in m.film_steps:
        real = film.forward

        def _spy(*a, _f=real, **kw):
            calls["n"] += 1
            return _f(*a, **kw)

        film.forward = _spy


def test_film_runs_inside_the_checkpointed_body():
    """FiLM must be recomputed in backward, not stored.

    Applied outside the checkpointed body, the FiLM output is a *saved input* of
    the checkpoint, so one extra [B, n_rnodes, latent] tensor per processor step
    stays resident and checkpointing reclaims almost nothing -- measured on the
    1601 random-path data: 23.37 GiB off -> 22.99 GiB on, still OOM at full
    resolution. Inside the body it is recomputed, which is what makes
    full-resolution training fit. Recompute is exact: FiLM is Linear/Sigmoid with
    no randomness, and edge masking already happens once per forward above.

    Two calls per step (forward + backward recompute) is the observable proof.
    """
    m = _model(grad_checkpointing=True, processor_steps=3, film_keys=_FILM_KEYS).train()
    calls = {"n": 0}
    _count_film_calls(m, calls)
    _forward_backward(m, _inputs((8, 6, 5), 1))
    assert calls["n"] == 2 * 3, (
        f"expected 2 FiLM calls per step (forward + recompute), got {calls['n']}; "
        "FiLM is outside the checkpointed body and its output is being stored")


def test_film_runs_once_per_step_without_checkpointing():
    """Guard the counter itself: with checkpointing off there is no recompute."""
    m = _model(grad_checkpointing=False, processor_steps=3, film_keys=_FILM_KEYS).train()
    calls = {"n": 0}
    _count_film_calls(m, calls)
    _forward_backward(m, _inputs((8, 6, 5), 1))
    assert calls["n"] == 3


def test_checkpoint_matches_plain_forward_and_backward_with_film():
    """Folding FiLM into the checkpoint must not change the math.

    The residual is taken on the *post*-FiLM latent, so the checkpointed body has
    to return `film_out + new_r`; returning `latent_r + new_r` would silently
    train a different model.
    """
    torch.manual_seed(0)
    m = _model(grad_checkpointing=False, film_keys=_FILM_KEYS).train()
    x = _inputs((8, 6, 5), 1)

    y0, g0 = _forward_backward(m, x)
    m._grad_checkpoint = True
    y1, g1 = _forward_backward(m, x)

    for key in y0:
        assert torch.allclose(y0[key], y1[key], atol=1e-6, rtol=1e-5), f"output {key} differs"
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable <= set(g1), f"missing grads: {trainable - set(g1)}"
    for name in g0:
        assert torch.allclose(g0[name], g1[name], atol=1e-5, rtol=1e-4), f"grad {name} differs"


def test_checkpoint_matches_plain_forward_and_backward():
    torch.manual_seed(0)
    m = _model(grad_checkpointing=False).train()
    x = _inputs((8, 6, 5), 1)

    # Same model instance, same weights, same lazily-built graph: flipping the
    # flag is the only change, so outputs and gradients must match.
    y0, g0 = _forward_backward(m, x)
    m._grad_checkpoint = True
    y1, g1 = _forward_backward(m, x)

    for key in y0:
        assert torch.allclose(y0[key], y1[key], atol=1e-6, rtol=1e-5), f"output {key} differs"
    assert set(g0) == set(g1)
    # backward must still reach every trainable parameter under checkpointing
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable <= set(g1), f"missing grads: {trainable - set(g1)}"
    for name in g0:
        assert torch.allclose(g0[name], g1[name], atol=1e-5, rtol=1e-4), f"grad {name} differs"
