"""Regression tests for the GAOT model family (models/gaot_arch.py).

CPU-only and license-free: a tiny grid and latent lattice keep the graph build
and forward fast. Mirrors tests/test_rigno_model.py, since GAOT is added as a
drop-in alternative on the same grid-in/grid-out contract.
"""
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import lpno.models.pipeline.model_domain as model_domain
from physicsnemo.sym.key import Key
from lpno.models.gaot_arch import (
    GAOTArch, _grid_coords, _knn_edges, _knn_edges_source_centric,
)

# Same per-uid scoping as conftest.GRAPH_CACHE_DIR: CI runs the suite as a
# different user and a fixed /tmp name would collide.
CACHE_DIR = os.path.join(tempfile.gettempdir(), f"gaot_test_graphs_{os.getuid()}")

IN_KEYS = ["V_scan", "P", "H_norm", "x", "y", "z"]
CONF_DIR = Path(__file__).resolve().parents[1] / "src" / "lpno" / "conf" / "hydra"


def _small_model(**kw) -> GAOTArch:
    defaults = dict(
        dimension=3, latent_tokens=(6, 4, 4), patch_size=2, lifting_channels=8,
        hidden_size=32, ffn_hidden_size=64, num_layers=3, num_heads=4,
        num_kv_heads=4, mlp_hidden_layers=(16, 16), projection_channels=16,
        k_encoder=4, k_decoder=4, cache_dir=CACHE_DIR,
    )
    defaults.update(kw)
    return GAOTArch(
        input_keys=[Key(k) for k in IN_KEYS],
        output_keys=[Key("T"), Key("alpha")],
        **defaults,
    )


def _inputs(grid, B):
    nx, ny, nz = grid
    return {k: torch.randn(B, 1, nx, ny, nz) for k in IN_KEYS}


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def test_grid_coords_are_per_dimension_normalised_and_row_major():
    # Per-dim [-1, 1] (not a single global min/max as upstream GAOT-3D's
    # rescale() does) so an anisotropic box is not squashed; row-major so node i
    # matches element i of x.reshape(B, C, -1).
    c = _grid_coords((4, 3, 2))
    assert c.shape == (24, 3)
    for d in range(3):
        assert c[:, d].min() == pytest.approx(-1.0)
        assert c[:, d].max() == pytest.approx(1.0)
    expected = np.stack(np.meshgrid(*[np.linspace(-1, 1, n) for n in (4, 3, 2)],
                                    indexing="ij"), axis=-1).reshape(-1, 3)
    assert np.allclose(c, expected)


def test_knn_edges_orientation_and_degree():
    src = _grid_coords((6, 4, 4))
    qry = _grid_coords((3, 2, 2))
    e = _knn_edges(src, qry, k=5)
    # Upstream convention: row 0 indexes the source, row 1 the query.
    assert e.shape == (2, qry.shape[0] * 5)
    assert e[0].max() < src.shape[0]
    assert e[1].max() < qry.shape[0]
    # kNN degree is exact: every query gets exactly k edges.
    assert np.bincount(e[1], minlength=qry.shape[0]).tolist() == [5] * qry.shape[0]


def test_source_centric_knn_covers_every_source():
    src = _grid_coords((6, 4, 4))
    qry = _grid_coords((3, 2, 2))
    e = _knn_edges_source_centric(src, qry, k=3)
    assert e.shape == (2, src.shape[0] * 3)
    assert e[0].max() < src.shape[0] and e[1].max() < qry.shape[0]
    # Every source emits exactly k edges -- that is the whole point of this
    # direction, versus the query-centric build which bounds the other side.
    assert np.bincount(e[0], minlength=src.shape[0]).tolist() == [3] * src.shape[0]


def test_encoder_reads_every_physical_node():
    """The encoder must not silently drop part of the input field.

    Regression test for a real bug in the first version of this port: the
    encoder used the query-centric kNN, which bounds the LATENT side, so
    physical nodes that were nobody's nearest neighbour were never read. On the
    production 61x40x40 / 32x20x20 pairing that lost 19.4% of the input.
    """
    m = _small_model()
    m(_inputs((8, 6, 5), 1))
    read = torch.unique(m.enc_edges[0]).numel()
    assert read == m.n_phys, f"encoder reads only {read}/{m.n_phys} physical nodes"
    # And every physical node must receive an output from the decoder.
    produced = torch.unique(m.dec_edges[1]).numel()
    assert produced == m.n_phys, f"decoder produces only {produced}/{m.n_phys} nodes"


def test_encoder_touches_every_latent_token():
    # A latent token with no incoming edge contributes nothing but still costs a
    # transformer slot; on a uniform grid every token should be fed.
    m = _small_model()
    m(_inputs((8, 6, 5), 1))
    assert torch.unique(m.enc_edges[1]).numel() == m.num_latent


def test_graph_shape_matches_true_input_not_config():
    # Built lazily from the actual (possibly cropped) grid, like RIGNO.
    m = _small_model()
    m(_inputs((7, 6, 5), 1))
    assert m.grid_shape == (7, 6, 5)
    assert m.n_phys == 7 * 6 * 5
    assert m.num_latent == 6 * 4 * 4
    assert tuple(m.enc_edges.shape) == (2, m.n_phys * 4)
    assert tuple(m.dec_edges.shape) == (2, m.n_phys * 4)


# --------------------------------------------------------------------------- #
# Forward / backward
# --------------------------------------------------------------------------- #
def test_gaot_forward_output_keys_and_shapes():
    m = _small_model()
    y = m(_inputs((8, 6, 5), 2))
    assert set(y.keys()) == {"T", "alpha"}
    assert tuple(y["T"].shape) == (2, 1, 8, 6, 5)
    assert tuple(y["alpha"].shape) == (2, 1, 8, 6, 5)
    assert torch.isfinite(y["T"]).all()


def test_gaot_backward_reaches_all_parameters():
    m = _small_model()
    y = m(_inputs((8, 6, 5), 1))
    loss = sum(v.pow(2).mean() for v in y.values())
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"parameters without gradient: {missing}"


def test_gaot_eval_mode_is_deterministic():
    m = _small_model(atten_dropout=0.0).eval()
    x = _inputs((8, 6, 5), 1)
    with torch.no_grad():
        assert torch.allclose(m(x)["T"], m(x)["T"])


def test_batch_elements_are_independent():
    # The encoder/decoder loop over the batch; a sample must not leak into another.
    m = _small_model().eval()
    x = _inputs((8, 6, 5), 2)
    with torch.no_grad():
        both = m(x)["T"]
        first = m({k: v[:1] for k, v in x.items()})["T"]
    assert torch.allclose(both[:1], first, atol=1e-5)


def test_every_input_channel_reaches_the_output():
    # Guards against a channel being silently dropped by the lifting/flatten path.
    m = _small_model().eval()
    x = _inputs((8, 6, 5), 1)
    with torch.no_grad():
        base = m(x)["T"]
        for k in IN_KEYS:
            perturbed = dict(x)
            perturbed[k] = x[k] + 1.0
            assert not torch.allclose(base, m(perturbed)["T"]), f"input {k} is ignored"


def test_pointwise_skip_is_off_by_default():
    assert _small_model().pointwise_skip is None


def test_pointwise_skip_starts_as_an_exact_no_op():
    """Zero-init the last layer so the ablation measures what the skip ADDS.

    If the skip contributed noise at step 0, the arm could lose to a bad init rather
    than to the idea, and the comparison against the GAOT baseline would say nothing.
    """
    torch.manual_seed(0)
    base = _small_model()
    torch.manual_seed(0)
    skip = _small_model(use_pointwise_skip=True)
    assert skip.pointwise_skip is not None

    x = _inputs((8, 6, 5), 2)
    base.eval(); skip.eval()
    with torch.no_grad():
        a, b = base(x), skip(x)
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), f"{k} differs at init"


def test_pointwise_skip_learns_and_reaches_full_resolution():
    """The skip must receive gradient, and must be pointwise at the FULL grid --
    no lattice in between. Distinct per-axis sizes catch a reshape that silently
    transposes."""
    m = _small_model(use_pointwise_skip=True)
    y = m(_inputs((9, 7, 5), 1))
    y["T"].square().mean().backward()
    grads = [p.grad for p in m.pointwise_skip.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), "skip got no gradient"


def test_pointwise_skip_adds_no_edges():
    """The cost argument: O(N x C), zero extra graph edges, latent lattice untouched."""
    torch.manual_seed(0)
    base = _small_model()
    torch.manual_seed(0)
    skip = _small_model(use_pointwise_skip=True)
    x = _inputs((8, 6, 5), 1)
    with torch.no_grad():
        base(x); skip(x)
    assert base.enc_edges.shape == skip.enc_edges.shape
    assert base.dec_edges.shape == skip.dec_edges.shape
    assert base.latent_shape == skip.latent_shape


def test_fraction_outputs_are_clamped_to_physical_range():
    """`alpha` is a volume fraction; the model must not emit values outside [0, 1].

    Nothing else in the pipeline enforces this -- the final projection is an
    unbounded linear layer and the alpha-masking node is a fill, not a range
    constraint. Measured 2026-07-30 on the material sweep: eval reported
    rmse_alpha = 1.425 on a [0,1] quantity (the eval path masks every key with the
    PREDICTED alpha, so an out-of-range alpha saturates its own mask and survives),
    while in-training validation of the same model read 0.016.
    """
    m = _small_model().eval()
    with torch.no_grad():
        y = m(_inputs((8, 6, 5), 2))
    a = y["alpha"]
    assert a.min() >= 0.0 and a.max() <= 1.0, f"alpha out of range: [{a.min()}, {a.max()}]"
    # T is unbounded and must NOT be clamped.
    assert "T" in y


def test_clamp_keeps_gradients_flowing_out_of_range():
    """A plain clamp has zero gradient outside the range, which would hide drift
    instead of correcting it. The straight-through form must pass gradient."""
    from lpno.models.gaot_arch import _clamp01_straight_through

    x = torch.tensor([-5.0, 0.3, 9.0], requires_grad=True)
    y = _clamp01_straight_through(x)
    assert y.detach().tolist() == pytest.approx([0.0, 0.3, 1.0])
    y.sum().backward()
    assert x.grad.tolist() == [1.0, 1.0, 1.0], "out-of-range predictions must still get signal"


def test_clamp_can_be_disabled():
    m = _small_model(clamp_fraction_outputs=False)
    assert m._fraction_out_keys == ()


def test_geoembed_decoder_off_by_default_and_addable():
    assert _small_model().geo_dec is None
    m = _small_model(use_geoembed_decoder=True)
    assert m.geo_dec is not None
    assert torch.isfinite(m(_inputs((8, 6, 5), 1))["T"]).all()


def test_radius_strategy_runs():
    m = _small_model(neighbor_strategy="radius", gno_radius=0.9, max_neighbors=6)
    assert torch.isfinite(m(_inputs((8, 6, 5), 1))["T"]).all()


def test_nonlinear_transform_types_run():
    """transform_type='nonlinear' feeds f_y into the kernel MLP.

    Regression test: the kernel input must be sized with the LIFTED channel
    width, not the raw input channel count. Upstream sizes it with the raw count
    (GAOT-3D magno.py:404) and dies on a shape mismatch, so this would break the
    moment an ablation tried the knob.
    """
    for tt in ("nonlinear", "nonlinear_kernelonly"):
        m = _small_model(transform_type=tt)
        y = m(_inputs((8, 6, 5), 1))
        assert torch.isfinite(y["T"]).all(), tt
        loss = sum(v.pow(2).mean() for v in y.values())
        loss.backward()
        missing = [n for n, p_ in m.named_parameters() if p_.requires_grad and p_.grad is None]
        assert not missing, f"{tt}: parameters without gradient: {missing}"


def test_rejects_latent_lattice_not_divisible_by_patch_size():
    with pytest.raises(ValueError, match="divisible by patch_size"):
        _small_model(latent_tokens=(6, 5, 4), patch_size=2)


def test_rejects_latent_lattice_rank_mismatch():
    with pytest.raises(ValueError, match="dimension"):
        _small_model(latent_tokens=(6, 4), dimension=3)


def test_edge_cache_is_reused_across_models():
    grid = (8, 6, 5)
    _small_model()(_inputs(grid, 1))
    cached = sorted(Path(CACHE_DIR).glob("gaot_edges_*.npy"))
    assert cached, "expected the edge build to be cached to disk"
    m2 = _small_model()
    m2(_inputs(grid, 1))  # must hit the cache, not rebuild
    assert sorted(Path(CACHE_DIR).glob("gaot_edges_*.npy")) == cached


# --------------------------------------------------------------------------- #
# Pipeline wiring
# --------------------------------------------------------------------------- #
def test_build_model_and_nodes_dispatches_gaot(monkeypatch):
    monkeypatch.setattr(model_domain, "fl_Node", lambda **kwargs: object())
    monkeypatch.setattr(model_domain, "Node", lambda **kwargs: SimpleNamespace(**kwargs))

    cfg = SimpleNamespace(arch=SimpleNamespace(gaot=SimpleNamespace(
        dimension=3, latent_tokens=[6, 4, 4], patch_size=2, lifting_channels=8,
        hidden_size=32, ffn_hidden_size=64, num_layers=3, num_heads=4,
        num_kv_heads=4, mlp_hidden_layers=[16, 16], projection_channels=16,
        k_encoder=4, k_decoder=4, cache_dir=CACHE_DIR)))
    key_cfg = SimpleNamespace(
        fno_input_keys=[Key(k) for k in IN_KEYS],
        fno_output_keys=[Key("T"), Key("alpha")],
        str_fno_input_keys=list(IN_KEYS),
        str_fno_output_keys=["T", "alpha"],
    )

    bundle = model_domain.build_model_and_nodes(
        cfg=cfg, key_cfg=key_cfg,
        material_params={"T_s": 1873.0, "T_l": 1923.0}, laser_params={},
        ref_scales={"T": 3000.0}, alpha_masking=False, eval_only=False,
        model_family="gaot", deeponet_branch_keys=[], deeponet_trunk_keys=[],
        pde_loss=False, vof_pde_loss=False, diff_loss=False,
        dx=1.0, dy=1.0, dz=1.0, diff_weight=1.0,
    )
    assert isinstance(bundle.model, GAOTArch)
    assert bundle.nn_nodes
    assert bundle.model.num_outputs == 2


def test_gaot_config_is_a_pure_architecture_swap_of_config_m():
    """config_flow3D_GAOT must differ from config M only in arch + model_family.

    The comparison against the published RIGNO numbers is only an architecture
    measurement if nothing else moved, so pin that here rather than trusting a
    reader to diff two YAMLs.
    """
    gaot = yaml.safe_load((CONF_DIR / "config_flow3D_GAOT.yaml").read_text())
    assert gaot["defaults"][0] == "config_flow3D_RIGNO", (
        "GAOT config must inherit config M rather than copy it, so the two "
        "cannot drift apart"
    )
    assert set(gaot) <= {"defaults", "arch", "custom"}, (
        f"unexpected top-level overrides: {sorted(set(gaot) - {'defaults', 'arch', 'custom'})}"
    )
    assert set(gaot["arch"]) == {"gaot"}, "must not touch another family's arch block"
    assert set(gaot["custom"]) == {"name", "model_family"}, (
        f"custom must only rename the run and switch family, got {sorted(gaot['custom'])}. "
        "Field trimming is automatic (config_loading._prune_unused_vtk_fields) -- do "
        "not hand-trim vtk_fields per config."
    )
    assert gaot["custom"]["model_family"] == "gaot"

def test_checkpoint_roundtrips_before_the_first_forward(tmp_path):
    """Eval must be able to load a checkpoint into a model that has never run.

    `Arch.load` does a STRICT `load_state_dict` before any forward, while the
    graph buffers (phys_pos, latent_pos, enc_edges, dec_edges, patch_positions)
    are only created lazily on the first forward. They are all registered
    persistent=False so they stay out of the state_dict; if any of them became
    persistent, training would still pass and only eval would break.
    """
    trained = _small_model()
    trained(_inputs((8, 6, 5), 1))            # build the graph, as training does
    sd = trained.state_dict()
    assert not any(
        k.endswith(("phys_pos", "latent_pos", "enc_edges", "dec_edges", "patch_positions"))
        for k in sd
    ), f"lazily built buffers leaked into the state_dict: {sorted(sd)}"

    fresh = _small_model()                     # never forwarded -> no graph yet
    fresh.load_state_dict(sd)                  # strict by default
    fresh.eval(), trained.eval()
    x = _inputs((8, 6, 5), 1)
    with torch.no_grad():
        assert torch.allclose(trained(x)["T"], fresh(x)["T"], atol=1e-6)


def test_gaot_onecycle_config_swaps_only_the_architecture():
    """The one-cycle arm must inherit RIGNO's one-cycle config, not config M.

    config M carries an inline `scheduler:` block whose decay keys survive a
    `scheduler=one_cycle` group swap and then fail against `OneCycleLRConf`
    (recorded in config_flow3D_RIGNO_onecycle's own header). Inheriting the
    RIGNO one-cycle config sidesteps that and keeps every non-scheduler
    hyperparameter at config M's value, so GAOT-with-one-cycle is comparable to
    RIGNO-with-one-cycle.
    """
    cfg = yaml.safe_load((CONF_DIR / "config_flow3D_GAOT_onecycle.yaml").read_text())
    assert cfg["defaults"][0] == "config_flow3D_RIGNO_onecycle"
    assert set(cfg) <= {"defaults", "arch", "custom"}
    assert set(cfg["arch"]) == {"gaot"}
    assert set(cfg["custom"]) == {"name", "model_family"}
    assert cfg["custom"]["model_family"] == "gaot"
    # The two GAOT configs must describe the SAME architecture, or the scheduler
    # arm silently ablates two things at once.
    base = yaml.safe_load((CONF_DIR / "config_flow3D_GAOT.yaml").read_text())
    assert cfg["arch"]["gaot"] == base["arch"]["gaot"], (
        "one-cycle arm's arch block drifted from config_flow3D_GAOT"
    )
    # And it must not reintroduce the decay keys that break OneCycleLRConf.
    assert "scheduler" not in cfg


def test_config_loading_accepts_gaot_family_on_the_grid_layout():
    src = (CONF_DIR.parents[1] / "models" / "pipeline" / "config_loading.py").read_text()
    # The family whitelist and the grid/pointwise split both have to know "gaot",
    # otherwise a run fails at config load or silently takes the pointwise path.
    whitelist = src.split('if model_family not in ("fno", "unet", "fully_connected"')[1]
    assert '"gaot"' in whitelist.split("raise ValueError")[0]
    layout = src.split('data_layout = "grid" if model_family in')[1].split("\n")[0]
    assert '"gaot"' in layout


def test_output_bounds_are_off_by_default():
    """The ceiling is a physics choice belonging to the config, not the layer."""
    assert _small_model()._output_bounds == {}


def test_t_bound_clamps_forward_but_still_backpropagates():
    """T is unbounded in the model (plain linear projection) and alpha masking does
    NOT protect it: the mask is a sigmoid blend, so wherever alpha > 0.5 -- the metal,
    ~75% of the domain -- the raw prediction passes through untouched by design.

    Measured 2026-07-30 on gaot_mat_steady_20k: T MAE 17.75 K against T RMSE 6279 K.
    """
    from lpno.models.gaot_arch import _clamp_straight_through

    x = torch.tensor([-2.0, 0.5, 740.0], requires_grad=True)
    y = _clamp_straight_through(x, 0.0, 1.25)
    assert y.detach().tolist() == pytest.approx([0.0, 0.5, 1.25])
    y.sum().backward()
    # A plain clamp would give [0, 1, 0] here and the blown-up cell would never
    # be corrected; straight-through keeps the signal alive.
    assert x.grad.tolist() == [1.0, 1.0, 1.0]


def test_bounded_model_cannot_emit_out_of_range_T():
    m = _small_model(output_bounds={"T": [0.0, 1.25]}).eval()
    with torch.no_grad():
        y = m(_inputs((8, 6, 5), 2))
    assert y["T"].min() >= 0.0 and y["T"].max() <= 1.25


def test_one_sided_bound_is_allowed():
    """A floor with no ceiling (or vice versa) must work -- absolute zero is a hard
    physical bound even where a ceiling is a judgement call."""
    from lpno.models.gaot_arch import _clamp_straight_through

    x = torch.tensor([-3.0, 5.0])
    assert _clamp_straight_through(x, 0.0, None).tolist() == [0.0, 5.0]
    assert _clamp_straight_through(x, None, 1.0).tolist() == [-3.0, 1.0]


def test_bounds_for_unknown_keys_are_ignored():
    m = _small_model(output_bounds={"not_an_output": [0.0, 1.0], "T": [0.0, 2.0]})
    assert set(m._output_bounds) == {"T"}
