"""The decoder kernel must be able to vary WITHIN a latent cell.

Mechanism this guards, located 2026-07-30: on a regular lattice with k_decoder=8 every
physical node inside a latent cell selects exactly the 8 corners of that cell
(`gaot_arch._knn_edges`). The cosine attention scores ABSOLUTE normalised coordinates
(`integral_transform.py`), which are nearly identical across a small cell, and the kernel
MLP is fed `cat([source_pos, query_pos])` -- absolute again, never the offset. The decoder
is then close to a box filter over each cell.

Rendered cross-sections of model 22022 (frame idx360) show the consequence: the predicted
temperature field is piecewise-constant in ~3x3x2 tiles with a regular +/-250 K
checkerboard error, which displaces the 1898 K melt isotherm and under-predicts pool depth
(212 predicted melt cells against 291 in ground truth).
"""
import torch
from physicsnemo.sym.key import Key

from lpno.models.gaot_arch import GAOTArch
from lpno.models.gaot_layers.integral_transform import IntegralTransform

IK = [Key("x"), Key("y"), Key("z"), Key("P")]
OK = [Key("T"), Key("alpha")]


def _model(**kw):
    return GAOTArch(input_keys=IK, output_keys=OK, dimension=3, latent_tokens=(4, 4, 4),
                    patch_size=2, lifting_channels=8, hidden_size=16, ffn_hidden_size=32,
                    num_layers=1, num_heads=2, num_kv_heads=2, **kw)


def test_flag_is_off_by_default():
    """Existing checkpoints must keep loading: the widths change when it is on."""
    assert _model().kernel_use_displacement is False


def test_enabling_it_widens_the_kernel_input_by_one_coordinate_set():
    """2*dim -> 3*dim: (source, query) becomes (source, query, query-source)."""
    off, on = _model(), _model(kernel_use_displacement=True)
    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    assert n_on > n_off, "displacement must add kernel input channels"


def test_displacement_is_the_offset_not_another_absolute_copy():
    it = IntegralTransform(channel_mlp_layers=[9, 8, 8], coord_dim=3, use_attn=True,
                           attention_type="cosine", use_displacement=True)
    assert it.use_displacement is True
    src = (__import__("pathlib").Path(__file__).resolve().parents[1] / "src"
           / "lpno" / "models" / "gaot_layers" / "integral_transform.py").read_text()
    # the offset must be query MINUS source, in that order
    assert "self_features_pos[:, : self.coord_dim] - rep_features_pos[:, : self.coord_dim]" in src


def test_both_variants_produce_the_right_shape():
    x = {k.name: torch.randn(1, 1, 6, 5, 4) for k in IK}
    for flag in (False, True):
        y = _model(kernel_use_displacement=flag)(x)
        assert y["T"].shape == (1, 1, 6, 5, 4)
        assert y["alpha"].shape == (1, 1, 6, 5, 4)


def test_gradient_reaches_the_kernel_when_displacement_is_on():
    m = _model(kernel_use_displacement=True)
    y = m({k.name: torch.randn(1, 1, 6, 5, 4) for k in IK})
    y["T"].square().mean().backward()
    grads = [p.grad for p in m.dec_agno.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
