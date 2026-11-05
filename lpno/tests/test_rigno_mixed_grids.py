"""RIGNO must accept per-forward grid shapes, as the reference does.

The paper (arXiv 2501.19205) claims robust generalization across resolutions on
point clouds in arbitrary domains, and the reference implementation delivers it
by making the graph DATA, not model state: `Batch.g` carries a
`RegionInteractionGraphSet` per batch, `graph_builder.build_graphs(batch.g)` is
called for every batch (train.py:278), and `_encode_process_decode(graphs, ...)`
takes the graph as an argument. The learned parameters are feature-dim MLPs --
nothing depends on node count.

Our port cached the first graph as module buffers behind a `_graph_built` guard,
which silently bound a model to the first grid it ever saw. Measured 2026-08-16
(workchain 50818, `ti64_geom40`): the first differently-shaped case died with
`RuntimeError: Sizes of tensors must match except in dimension 2` at
rigno_arch.py:484 -- at step 0 from the train/val stores' differing extents, so
`batch_size_grid=1` could not avoid it. That failure is the port's artifact, not
the architecture's limit; these tests pin the reference behaviour.
"""
import torch

from conftest import GRAPH_CACHE_DIR
from physicsnemo.sym.key import Key
from lpno.models.rigno_arch import RIGNOArch


def _small_model(**kw) -> RIGNOArch:
    ik = [Key(k) for k in ["V_scan", "P", "H_norm", "x", "y", "z"]]
    ok = [Key("T"), Key("alpha")]
    defaults = dict(dimension=3, node_latent_size=16, edge_latent_size=16,
                    processor_steps=2, subsample_factor=6.0, k_p2r=3, k_r2p=3,
                    p_edge_masking=0.5, cache_dir=GRAPH_CACHE_DIR)
    defaults.update(kw)
    return RIGNOArch(input_keys=ik, output_keys=ok, **defaults)


def _inputs(grid, B=1):
    nx, ny, nz = grid
    return {k: torch.randn(B, 1, nx, ny, nz) for k in ["V_scan", "P", "H_norm", "x", "y", "z"]}


def test_forward_accepts_a_second_grid_shape():
    """The geom40 failure: same model, two extents, both must run."""
    m = _small_model().eval()
    with torch.no_grad():
        a = m(_inputs((7, 6, 5)))
        b = m(_inputs((9, 8, 5)))     # different extent -- crashed pre-fix
    assert tuple(a["T"].shape) == (1, 1, 7, 6, 5)
    assert tuple(b["T"].shape) == (1, 1, 9, 8, 5)
    assert torch.isfinite(b["T"]).all()


def test_val_shaped_like_neither_train_case():
    """Alternating shapes, as a mixed-grid dataset serves them."""
    m = _small_model().eval()
    with torch.no_grad():
        for grid in [(7, 6, 5), (9, 8, 5), (7, 6, 5), (8, 5, 4)]:
            y = m(_inputs(grid))
            assert tuple(y["T"].shape) == (1, 1, *grid)


def test_each_shape_builds_its_graph_once(monkeypatch):
    """Revisiting a shape reuses the in-memory graph -- no rebuild per forward."""
    import lpno.models.rigno_arch as ra
    calls = []
    orig = ra._cached_graph

    def counting(coords, *a, **kw):
        calls.append(coords.shape[0])
        return orig(coords, *a, **kw)

    monkeypatch.setattr(ra, "_cached_graph", counting)
    m = _small_model(num_graph_variants=1).eval()
    with torch.no_grad():
        m(_inputs((7, 6, 5)))
        m(_inputs((9, 8, 5)))
        m(_inputs((7, 6, 5)))   # revisit: must NOT rebuild
    assert calls == [7 * 6 * 5, 9 * 8 * 5]


def test_mixed_shapes_backward_and_variants():
    """Training mode: gradients flow at both shapes, variant rotation intact."""
    m = _small_model(num_graph_variants=2)
    for grid in [(7, 6, 5), (9, 8, 5)]:
        y = m(_inputs(grid))
        loss = sum(v.pow(2).mean() for v in y.values())
        loss.backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"parameters without gradient: {missing}"


def test_single_grid_behaviour_unchanged():
    """A one-grid model keeps its public surface: grid_shape, n_pnodes, determinism."""
    m = _small_model().eval()
    x = _inputs((7, 6, 5))
    with torch.no_grad():
        a = m(x)["T"]
        b = m(x)["T"]
    assert m.grid_shape == (7, 6, 5)
    assert m.n_pnodes == 7 * 6 * 5
    assert torch.equal(a, b)
