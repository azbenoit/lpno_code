"""The hand-rolled mean aggregation must equal torch_scatter's scatter_mean.

`InteractionStep` used to call ``scatter_mean``, which allocates a full
(B, E, L) tensor of ones purely to count contributions per receiver -- 1.32 GiB
for the r2p edge set at the flow3D grid (3.6M edges x 96 channels), on the
autograd tape as well. We divide a scatter_add by an (n_recv,) degree vector
instead. That is only safe if it is numerically identical, including for
receivers with no incoming edges (scatter_mean clamps the count at 1).
"""
import pytest
import torch

# The reference is torch_scatter's own scatter_mean. rigno_arch no longer needs the
# package (test_rigno_native_scatter.py), so without it this file skips.
torch_scatter = pytest.importorskip("torch_scatter")
scatter_mean = torch_scatter.scatter_mean

from lpno.models.rigno_arch import _degree  # noqa: E402


def _reference(msg, idx, n_recv):
    return scatter_mean(msg, idx, dim=1, dim_size=n_recv)


def _ours(msg, idx, receivers, n_recv):
    scatter_add = torch_scatter.scatter_add
    return scatter_add(msg, idx, dim=1, dim_size=n_recv) / _degree(
        receivers, n_recv, msg.dtype).view(1, -1, 1)


def _case(receivers, n_recv, B=2, L=5, seed=0):
    torch.manual_seed(seed)
    msg = torch.randn(B, receivers.numel(), L, dtype=torch.float64)
    idx = receivers.view(1, -1, 1).expand(B, -1, L)
    return msg, idx


def test_matches_scatter_mean_uneven_degrees():
    # receiver 0 gets 3 edges, 1 gets 1, 2 gets 2 -> exercises the division
    receivers = torch.tensor([0, 0, 0, 1, 2, 2])
    msg, idx = _case(receivers, 3)
    assert torch.allclose(_reference(msg, idx, 3), _ours(msg, idx, receivers, 3),
                          atol=0, rtol=0)


def test_matches_scatter_mean_with_isolated_receiver():
    # receiver 1 has NO incoming edges: scatter_mean clamps the count to 1 and
    # returns 0, so the degree vector must clamp identically (not divide by 0).
    receivers = torch.tensor([0, 0, 2])
    msg, idx = _case(receivers, 4)
    ours = _ours(msg, idx, receivers, 4)
    assert torch.allclose(_reference(msg, idx, 4), ours, atol=0, rtol=0)
    assert torch.isfinite(ours).all(), "empty receiver produced inf/nan"
    assert (ours[:, 1] == 0).all()


def test_degree_follows_resampled_receivers():
    # Edge masking resamples `receivers` per forward; the divisor must track it
    # rather than be cached from a previous call.
    assert _degree(torch.tensor([0, 0, 1]), 2, torch.float64).tolist() == [2.0, 1.0]
    assert _degree(torch.tensor([0, 1, 1]), 2, torch.float64).tolist() == [1.0, 2.0]


def test_aggregation_runs_in_fp32_under_autocast():
    """BUG-43: the message sum must not run in fp16.

    Autocast promotes `torch.sum` and `LayerNorm` to fp32 but has no rule for
    `torch_scatter.scatter_add`, so the model's only reduction -- summing millions
    of edge messages, fp16 max 65504 -- was the one op left unprotected. Repro
    16679/16681 collapsed the grad scaler through 20 consecutive halvings
    (65536 -> 0.125) with edge masking both on and off, so this is not about
    masking. Guard the promotion, not the symptom.
    """
    import lpno.models.rigno_arch as ra

    step = ra.InteractionStep(latent=8, hidden_layers=1, update_sender=False)
    senders = torch.tensor([0, 1, 2, 3])
    receivers = torch.tensor([0, 0, 1, 1])
    captured = {}
    real = ra.scatter_add

    def spy(src, *a, **kw):
        captured["dtype"] = src.dtype
        return real(src, *a, **kw)

    ra.scatter_add = spy
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev == "cuda" else torch.bfloat16
        step = step.to(dev)
        x = torch.randn(1, 4, 8, device=dev)
        with torch.autocast(dev, dtype=dtype):
            step(x, x, torch.randn(1, 4, 8, device=dev), senders.to(dev),
                 receivers.to(dev), 4)
    finally:
        ra.scatter_add = real
    assert captured["dtype"] is torch.float32, (
        f"scatter_add ran in {captured['dtype']} under autocast; the edge-message "
        "sum must be fp32 (BUG-43)")
