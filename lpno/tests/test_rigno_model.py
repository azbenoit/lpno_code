"""Regression tests for the RIGNO model family (models/rigno_arch.py).

CPU-only and license-free: a tiny grid keeps the graph build and forward fast.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lpno.models.pipeline.model_domain as model_domain
from conftest import GRAPH_CACHE_DIR
from physicsnemo.sym.key import Key
from lpno.models.rigno_arch import RIGNOArch, build_region_interaction_graph


def _load_ensemble_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rigno_ensemble_eval.py"
    spec = importlib.util.spec_from_file_location("rigno_ensemble_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _small_model(**kw) -> RIGNOArch:
    ik = [Key(k) for k in ["V_scan", "P", "H_norm", "x", "y", "z"]]
    ok = [Key("T"), Key("alpha")]
    defaults = dict(dimension=3, node_latent_size=16, edge_latent_size=16,
                    processor_steps=2, subsample_factor=6.0, k_p2r=3, k_r2p=3,
                    p_edge_masking=0.5, cache_dir=GRAPH_CACHE_DIR)
    defaults.update(kw)
    return RIGNOArch(input_keys=ik, output_keys=ok, **defaults)


def _inputs(grid, B):
    nx, ny, nz = grid
    return {k: torch.randn(B, 1, nx, ny, nz) for k in ["V_scan", "P", "H_norm", "x", "y", "z"]}


def test_graph_builder_shapes_and_degree():
    coords = np.stack(np.meshgrid(*[np.linspace(-1, 1, n) for n in (8, 6, 5)],
                                  indexing="ij"), axis=-1).reshape(-1, 3)
    g = build_region_interaction_graph(coords, subsample_factor=5.0, rmesh_levels=3,
                                       k_p2r=3, k_r2p=4, seed=0)
    n = coords.shape[0]
    # kNN degree is exact: E = N * k
    assert g["p2r_senders"].shape[0] == n * 3
    assert g["r2p_receivers"].shape[0] == n * 4
    # edge features are [rel_pos (d), dist (1)]
    assert g["p2r_edge_feat"].shape[1] == 4
    # senders/receivers index valid node ranges
    assert g["p2r_receivers"].max() < g["x_rnodes"].shape[0]
    assert g["r2r_senders"].max() < g["x_rnodes"].shape[0]


def test_rigno_forward_output_keys_and_shapes():
    m = _small_model()
    y = m(_inputs((8, 6, 5), 2))
    assert set(y.keys()) == {"T", "alpha"}
    assert tuple(y["T"].shape) == (2, 1, 8, 6, 5)
    assert tuple(y["alpha"].shape) == (2, 1, 8, 6, 5)
    assert torch.isfinite(y["T"]).all()


def test_rigno_backward_reaches_all_parameters():
    m = _small_model()
    y = m(_inputs((8, 6, 5), 1))
    loss = sum(v.pow(2).mean() for v in y.values())
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"parameters without gradient: {missing}"


def test_rigno_graph_shape_matches_true_input_not_config():
    # Graph is built lazily from the actual (possibly cropped) grid, not a config value.
    m = _small_model()
    m(_inputs((7, 6, 5), 1))
    assert m.grid_shape == (7, 6, 5)
    assert m.n_pnodes == 7 * 6 * 5


def test_rigno_eval_mode_is_deterministic():
    # Edge masking is disabled in eval -> repeated forwards match exactly.
    m = _small_model().eval()
    x = _inputs((8, 6, 5), 1)
    with torch.no_grad():
        a = m(x)["T"]
        b = m(x)["T"]
    assert torch.allclose(a, b)


def test_build_model_and_nodes_dispatches_rigno(monkeypatch):
    monkeypatch.setattr(model_domain, "fl_Node", lambda **kwargs: object())
    monkeypatch.setattr(model_domain, "Node", lambda **kwargs: SimpleNamespace(**kwargs))

    cfg = SimpleNamespace(arch=SimpleNamespace(rigno=SimpleNamespace(
        dimension=3, node_latent_size=16, edge_latent_size=16, processor_steps=2,
        mlp_hidden_layers=1, subsample_factor=6.0, rmesh_levels=3, k_p2r=3, k_r2p=3,
        p_edge_masking=0.5, graph_seed=0, cache_dir=GRAPH_CACHE_DIR)))
    key_cfg = SimpleNamespace(
        fno_input_keys=[Key(k) for k in ["V_scan", "P", "H_norm", "x", "y", "z"]],
        fno_output_keys=[Key("T"), Key("alpha")],
        str_fno_input_keys=["V_scan", "P", "H_norm", "x", "y", "z"],
        str_fno_output_keys=["T", "alpha"],
    )

    bundle = model_domain.build_model_and_nodes(
        cfg=cfg, key_cfg=key_cfg,
        material_params={"T_s": 1873.0, "T_l": 1923.0}, laser_params={},
        ref_scales={"T": 3000.0}, alpha_masking=False, eval_only=False,
        model_family="rigno", deeponet_branch_keys=[], deeponet_trunk_keys=[],
        pde_loss=False, vof_pde_loss=False, diff_loss=False,
        dx=1.0, dy=1.0, dz=1.0, diff_weight=1.0,
    )

    assert isinstance(bundle.model, RIGNOArch)
    assert bundle.nn_nodes  # at least the u_network node
    # output channel count matches T + alpha
    assert bundle.model.num_outputs == 2


def test_graph_builder_rnode_count_override():
    # rnode_count pins the regional-node count regardless of subsample_factor,
    # so eval at a different grid resolution can keep the trained rnode budget.
    coords = np.stack(np.meshgrid(*[np.linspace(-1, 1, n) for n in (8, 6, 5)],
                                  indexing="ij"), axis=-1).reshape(-1, 3)
    g = build_region_interaction_graph(coords, subsample_factor=5.0, rmesh_levels=3,
                                       k_p2r=3, k_r2p=4, seed=0, rnode_count=37)
    assert g["x_rnodes"].shape[0] == 37


def test_num_graph_variants_distinct_and_selectable():
    m = _small_model(num_graph_variants=3, p_edge_masking=0.0)
    x = _inputs((8, 6, 5), 1)
    m(x)  # lazy build of all variants
    assert m._n_variants == 3
    # Variants are genuinely different graphs (different rnode subsets).
    # Access path updated 2026-08-17: graphs moved from module buffers to the
    # per-shape store (graph-as-data, as the reference), same assertion intent.
    g = m._graphs[(8, 6, 5)]["tensors"]
    assert not torch.equal(g["rnode_struct"], g["rnode_struct_v1"])
    m.eval()
    with torch.no_grad():
        a = m(x)["T"]
        b = m(x)["T"]
        m._forced_variant = 1
        c = m(x)["T"]
        m._forced_variant = None
    # Eval defaults to variant 0 -> deterministic; a different variant changes output.
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)
    # Training forwards rotate variants and stay finite.
    m.train()
    y = m(x)["T"]
    assert torch.isfinite(y).all()


def test_num_graph_variants_default_single():
    m = _small_model()
    m(_inputs((8, 6, 5), 1))
    assert m._n_variants == 1
    assert not hasattr(m, "rnode_struct_v1")


def test_film_conditioning_moves_scalars_out_of_node_features():
    m = _small_model(film_keys=["P", "H_norm"], p_edge_masking=0.0).eval()
    x = _inputs((8, 6, 5), 1)
    x["P"] = torch.full_like(x["P"], 0.3)
    x["H_norm"] = torch.full_like(x["H_norm"], 0.7)
    with torch.no_grad():
        a = m(x)["T"]
        x2 = dict(x)
        x2["P"] = torch.full_like(x["P"], 0.9)
        b = m(x2)["T"]
    # The scalar still reaches the model (via FiLM), so changing it changes output.
    assert not torch.allclose(a, b)
    # And the pnode embedder no longer sees the two film channels.
    expected_in = (6 - 2) + 3  # non-film channels + xyz coords
    assert m.enc_pnode_embed.layers[0].in_features == expected_in
    # Default arch carries no film parameters (state_dict compatible).
    m2 = _small_model()
    assert not any("film" in n for n, _ in m2.named_parameters())


def test_inference_edge_masking_gives_stochastic_eval():
    m = _small_model(p_edge_masking=0.0, inference_edge_masking=0.5).eval()
    x = _inputs((8, 6, 5), 1)
    torch.manual_seed(0)
    with torch.no_grad():
        a = m(x)["T"]
        b = m(x)["T"]
    assert not torch.allclose(a, b)


def test_ensemble_metrics_averages_members(tmp_path):
    ens = _load_ensemble_module()
    shape = (2, 1, 3, 2, 2)
    truth = {k: np.ones(shape, dtype=np.float32) for k in ("T", "alpha", "fl")}
    # Two members bracketing the truth -> their average equals the truth exactly.
    for name, delta in (("run_a", +0.2), ("run_b", -0.2)):
        d = tmp_path / name / "fno_eval"
        d.mkdir(parents=True)
        np.savez(d / "predictions.npz", **{k: truth[k] + delta for k in truth})
        np.savez(d / "ground_truth.npz", **truth)

    err = ens.ensemble_metrics(str(tmp_path), ["run_a", "run_b"])
    # Averaged prediction is exact -> zero error on every field.
    for key in ("T", "alpha", "fl"):
        assert err[key]["abs_err_mean"] == pytest.approx(0.0, abs=1e-6)
    assert err["fl"]["iou_mean"] == pytest.approx(1.0)
