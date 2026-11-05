"""Regression tests for the U-Net refinement head (models/rigno_arch.py).

CPU-only, license-free. The U-Net corrector is the ripple-free alternative to
the spectral FNO head: local convs + pooling give a large receptive field with
NO FFT (no Gibbs ringing in the late-frame wake). These tests pin the two
properties the warm-start training run relies on: identity-at-init (zero-init
final conv -> refined == base at step 0, so a warm-started base starts as pure
rp3) and odd-dimension safety (the up-path must restore the exact encoder sizes
for the skip concat on the real [75,50,40] grid, whose halved sizes are odd).
"""
import torch
from conftest import GRAPH_CACHE_DIR
from physicsnemo.sym.key import Key

from lpno.models.rigno_arch import (
    _FNORefineHead,
    _UNetRefineHead,
    RIGNORefinedArch,
)


def test_fno_head_padding_shape_and_identity():
    # Nyquist modes + FFT domain padding (BUG-41 rung-2): the crop must restore
    # the physical size, and zero-init proj keeps it identity at init.
    head = _FNORefineHead(channels=2, width=6, modes=(38, 25, 21), layers=2, padding=8)
    x = torch.randn(1, 2, 75, 50, 40)
    y = head(x)
    assert y.shape == x.shape
    assert y.abs().max().item() == 0.0


def test_arch_routes_head_padding():
    ik = [Key(k) for k in ["V_scan", "P", "H_norm", "x", "y", "z"]]
    ok = [Key("T"), Key("alpha")]
    m = RIGNORefinedArch(input_keys=ik, output_keys=ok, head_modes=[38, 25, 21],
                         head_padding=8, node_latent_size=16, edge_latent_size=16,
                         processor_steps=2, cache_dir=GRAPH_CACHE_DIR)
    assert isinstance(m.head, _FNORefineHead)
    assert m.head.padding == 8


def test_unet_head_shape_and_identity_at_init():
    head = _UNetRefineHead(channels=2, width=8, levels=3)
    x = torch.randn(1, 2, 75, 50, 40)
    y = head(x)
    assert y.shape == x.shape
    # zero-init final conv -> residual is exactly zero at init (pure base output)
    assert y.abs().max().item() == 0.0


def test_unet_head_learns_nonzero_residual():
    head = _UNetRefineHead(channels=2, width=8, levels=2)
    with torch.no_grad():
        head.proj.weight.add_(0.05)
        head.proj.bias.add_(0.05)
    assert head(torch.randn(1, 2, 40, 30, 24)).abs().max().item() > 0.0


def test_unet_head_odd_dims():
    # halving [75,50,40] gives odd sizes; the trilinear up-path must restore the
    # exact encoder shape or the skip concat raises. Use deliberately odd dims.
    head = _UNetRefineHead(channels=3, width=8, levels=3)
    x = torch.randn(2, 3, 37, 51, 19)
    assert head(x).shape == x.shape


def _keys():
    ik = [Key(k) for k in ["V_scan", "P", "H_norm", "x", "y", "z"]]
    ok = [Key("T"), Key("alpha")]
    return ik, ok


def _small_rigno_kwargs():
    return dict(node_latent_size=16, edge_latent_size=16, processor_steps=2,
                cache_dir=GRAPH_CACHE_DIR)


def test_arch_routes_head_type_unet():
    ik, ok = _keys()
    m = RIGNORefinedArch(input_keys=ik, output_keys=ok, head_type="unet",
                         head_width=8, head_layers=2, **_small_rigno_kwargs())
    assert isinstance(m.head, _UNetRefineHead)


def test_arch_head_type_defaults_to_fno():
    # backward-compat: the live 36k demo model builds with no head_type set
    ik, ok = _keys()
    m = RIGNORefinedArch(input_keys=ik, output_keys=ok, **_small_rigno_kwargs())
    assert isinstance(m.head, _FNORefineHead)
