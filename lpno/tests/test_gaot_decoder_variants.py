"""GAOT decoder and neighbourhood variants added for the lattice-ripple study (2026-11-03).

`decoder_stencil="trilinear"` reads, for every physical node, the 8 corner tokens of the latent cell that contains it,
with trilinear weights (a partition of unity) in place of the softmax attention over its k nearest tokens. The
neighbour set then changes only where the weight of the swapped tokens is zero, so the output is continuous across
latent cell walls. `neighbor_strategy="bidirectional"` is upstream GAOT-3D's union of kNN and radius edges. Both are
off by default, so every stored GAOT checkpoint rebuilds unchanged.
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from physicsnemo.sym.key import Key

from lpno.models.gaot_arch import (GAOTArch, _grid_coords, _knn_edges, _knn_edges_source_centric, _radius_edges,
                                   _trilinear_edges)
from lpno.models.gaot_layers.integral_transform import IntegralTransform

CACHE_DIR = os.path.join(tempfile.gettempdir(), f"gaot_test_graphs_{os.getuid()}")
IN_KEYS = ["V_scan", "P", "H_norm", "x", "y", "z"]
CONF_DIR = Path(__file__).resolve().parents[1] / "src" / "lpno" / "conf" / "hydra"


def _model(**kw):
    d = dict(dimension=3, latent_tokens=(6, 4, 4), patch_size=2, lifting_channels=8, hidden_size=32,
             ffn_hidden_size=64, num_layers=3, num_heads=4, num_kv_heads=4, mlp_hidden_layers=(16, 16),
             projection_channels=16, k_encoder=4, k_decoder=4, cache_dir=CACHE_DIR)
    d.update(kw)
    return GAOTArch(input_keys=[Key(k) for k in IN_KEYS], output_keys=[Key("T"), Key("alpha")], **d)


def _inputs(grid, B=1):
    return {k: torch.randn(B, 1, *grid) for k in IN_KEYS}


def _pairs(edges):
    return {(int(a), int(b)) for a, b in edges.T}


def test_trilinear_edges_are_a_partition_of_unity_that_reproduces_linear_fields():
    lat, phys = (6, 4, 5), (11, 7, 9)
    edges, w = _trilinear_edges(lat, phys)
    n = int(np.prod(phys))
    assert edges.shape == (2, 8 * n) and w.shape == (8 * n,)
    assert edges[0].min() >= 0 and edges[0].max() < int(np.prod(lat))
    assert np.array_equal(np.bincount(edges[1], minlength=n), np.full(n, 8))
    assert np.all(w >= -1e-12)
    np.testing.assert_allclose(np.bincount(edges[1], weights=w, minlength=n), 1.0, atol=1e-12)
    # trilinear interpolation is exact on affine fields
    a, b = np.array([0.7, -1.3, 2.1]), 0.4
    f_lat = _grid_coords(lat) @ a + b
    rebuilt = np.bincount(edges[1], weights=w * f_lat[edges[0]], minlength=n)
    np.testing.assert_allclose(rebuilt, _grid_coords(phys) @ a + b, atol=1e-12)


def test_trilinear_edges_on_a_node_that_sits_on_a_token():
    # a physical node exactly on a token takes that token with weight 1
    lat, phys = (5, 3, 3), (9, 5, 5)
    edges, w = _trilinear_edges(lat, phys)
    node = np.ravel_multi_index((4, 2, 2), phys)       # (0, 0, 0) in [-1, 1]^3 coordinates
    token = np.ravel_multi_index((2, 1, 1), lat)
    sel = edges[1] == node
    np.testing.assert_allclose(w[sel][edges[0][sel] == token].sum(), 1.0, atol=1e-12)


def test_supplied_edge_weights_replace_the_attention_softmax():
    torch.manual_seed(0)
    it = IntegralTransform(channel_mlp_layers=[6, 8, 4], transform_type="linear", use_attn=True, coord_dim=3,
                           attention_type="cosine")
    y_pos, x_pos = torch.randn(5, 3), torch.randn(3, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 0], [0, 0, 1, 1, 2, 2]])
    f_y = torch.randn(5, 4)
    w = torch.tensor([0.25, 0.75, 0.5, 0.5, 0.9, 0.1])
    out = it(y_pos=y_pos, x_pos=x_pos, edge_index=edge_index, f_y=f_y, weights=w)
    agg = torch.cat([y_pos[edge_index[0]], x_pos[edge_index[1]]], dim=-1)
    msg = it.channel_mlp(agg) * f_y[edge_index[0]] * w[:, None]
    ref = torch.zeros(3, 4).index_add_(0, edge_index[1], msg)
    torch.testing.assert_close(out, ref)
    # and without weights the transform is what it was (softmax-weighted sum)
    assert not torch.allclose(it(y_pos=y_pos, x_pos=x_pos, edge_index=edge_index, f_y=f_y), out)


def test_trilinear_decoder_trains_and_holds_its_weights():
    m = _model(decoder_stencil="trilinear", kernel_use_displacement=True)
    out = m(_inputs((9, 7, 5), B=2))
    assert out["T"].shape == (2, 1, 9, 7, 5)
    assert m.dec_weights is not None and m.dec_weights.shape == (m.dec_edges.shape[1],)
    sums = torch.zeros(m.n_phys, dtype=torch.float64).index_add_(0, m.dec_edges[1], m.dec_weights.double())
    torch.testing.assert_close(sums, torch.ones(m.n_phys, dtype=torch.float64))
    (out["T"].sum() + out["alpha"].sum()).backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_knn_decoder_is_the_default_and_carries_no_weights():
    m = _model()
    m(_inputs((9, 7, 5)))
    assert m.dec_weights is None


def test_unknown_decoder_stencil_is_rejected():
    with pytest.raises(ValueError, match="decoder_stencil"):
        _model(decoder_stencil="nearest")


def test_bidirectional_is_the_union_of_knn_and_radius_edges():
    grid, r = (9, 7, 5), 0.5
    m = _model(neighbor_strategy="bidirectional", k_encoder=1, k_decoder=1, gno_radius=r)
    m(_inputs(grid))
    phys, lat = _grid_coords(grid), _grid_coords(m.latent_shape)
    enc = _pairs(_knn_edges_source_centric(phys, lat, 1)) | _pairs(_radius_edges(phys, lat, r, 0))
    dec = _pairs(_knn_edges(lat, phys, 1)) | _pairs(_radius_edges(lat, phys, r, 0))
    got_enc, got_dec = m.enc_edges.numpy(), m.dec_edges.numpy()
    assert _pairs(got_enc) == enc and got_enc.shape[1] == len(enc)     # no duplicate edges
    assert _pairs(got_dec) == dec and got_dec.shape[1] == len(dec)


def test_patch_size_one_runs():
    m = _model(latent_tokens=(3, 2, 2), patch_size=1)
    out = m(_inputs((9, 7, 5)))
    assert out["T"].shape == (1, 1, 9, 7, 5)


def test_gaot_config_declares_the_decoder_stencil_with_the_old_default():
    cfg = yaml.safe_load((CONF_DIR / "config_flow3D_GAOT.yaml").read_text())
    assert cfg["arch"]["gaot"]["decoder_stencil"] == "knn"


# --------------------------------------------------------------------------- #
# Position-free decoder, search metric, chunked eigenvalues (second round)
# --------------------------------------------------------------------------- #
def test_position_free_decoder_is_trilinear_interpolation_of_the_tokens():
    m = _model(decoder_stencil="trilinear", decoder_kernel="none")
    m(_inputs((9, 7, 5)))                                  # builds the graph
    A = torch.randn(3, m.latent_channels)
    z = (m.latent_pos @ A).unsqueeze(0)                    # an affine token field
    torch.testing.assert_close(m._decode(z)[0], m.phys_pos @ A, atol=1e-5, rtol=1e-5)
    assert m.dec_agno is None


def test_position_free_decoder_trains_every_parameter():
    m = _model(decoder_stencil="trilinear", decoder_kernel="none", use_pointwise_skip=True)
    out = m(_inputs((9, 7, 5), B=2))
    (out["T"].sum() + out["alpha"].sum()).backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_position_free_decoder_needs_the_trilinear_stencil():
    with pytest.raises(ValueError, match="decoder_kernel"):
        _model(decoder_kernel="none")


def test_neighbor_metric_scales_the_searches_only():
    grid, sc = (9, 7, 5), (1.0, 0.5, 0.25)
    m = _model(neighbor_metric=sc, k_encoder=4, k_decoder=4)
    m(_inputs(grid))
    phys, lat = _grid_coords(grid), _grid_coords(m.latent_shape)
    s = np.asarray(sc)
    assert _pairs(m.enc_edges.numpy()) == _pairs(_knn_edges_source_centric(phys * s, lat * s, 4))
    assert _pairs(m.dec_edges.numpy()) == _pairs(_knn_edges(lat * s, phys * s, 4))
    np.testing.assert_allclose(m.phys_pos.numpy(), phys, atol=1e-6)     # the kernel keeps the plain coordinates
    np.testing.assert_allclose(m.latent_pos.numpy(), lat, atol=1e-6)
    d = _model()
    d(_inputs(grid))
    assert _pairs(d.dec_edges.numpy()) == _pairs(_knn_edges(lat, phys, 4))  # off by default


def test_neighbor_metric_rejects_bad_scales():
    with pytest.raises(ValueError, match="neighbor_metric"):
        _model(neighbor_metric=(1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="neighbor_metric"):
        _model(neighbor_metric=(1.0, 0.5))


def test_chunked_eigenvalues_match_one_batched_call(monkeypatch):
    import lpno.models.gaot_layers.geoembed as ge_mod
    from lpno.models.gaot_layers.geoembed import GeometricEmbedding

    torch.manual_seed(0)
    src, qry = torch.rand(300, 3), torch.rand(50, 3)
    edges = torch.as_tensor(_knn_edges(src.numpy(), qry.numpy(), 6))
    ge = GeometricEmbedding(input_dim=3, output_dim=8)
    monkeypatch.setattr(ge_mod, "_EIGVALSH_CHUNK", 10 ** 9)
    whole = ge._compute_statistical_features_pyg(src, qry, edges)
    monkeypatch.setattr(ge_mod, "_EIGVALSH_CHUNK", 7)
    chunked = ge._compute_statistical_features_pyg(src, qry, edges)
    torch.testing.assert_close(chunked, whole)


def test_gaot_configs_declare_the_new_keys_at_the_old_behaviour():
    for name in ("config_flow3D_GAOT", "config_flow3D_GAOT_onecycle", "config_flow3D_GAOT_randompath"):
        g = yaml.safe_load((CONF_DIR / f"{name}.yaml").read_text())["arch"]["gaot"]
        assert (g["decoder_stencil"], g["decoder_kernel"], g["neighbor_metric"], g["use_pointwise_skip"]) == \
            ("knn", "agno", None, False), name
