"""GAOT must accept a heterogeneous set — geometry independence is the whole point.

The bipartite graph used to be built from the FIRST batch's grid and pinned forever
(`if not self._graph_built`), so a second grid reused edges for a geometry it did not have
and `y.reshape(B, num_outputs, *self.grid_shape)` reshaped to the first grid regardless.
That produced "GAOT cannot train on the heterogeneous composed set", which described this
cache rather than the architecture: the latent lattice is fixed independently of the
physical grid, and `_build_graph` was already parameterised by shape.
"""
import torch
from physicsnemo.sym.key import Key

from lpno.models.gaot_arch import GAOTArch

IK = [Key("P")]
OK = [Key("T")]


def _mk(tmp_path):
    return GAOTArch(input_keys=IK, output_keys=OK, dimension=3, latent_tokens=(4, 4, 4),
                    patch_size=2, lifting_channels=8, hidden_size=16, ffn_hidden_size=32,
                    num_layers=1, num_heads=2, num_kv_heads=2, k_encoder=3, k_decoder=3,
                    cache_dir=str(tmp_path))


def _fwd(model, shape):
    return model({"P": torch.zeros(1, 1, *shape)})["T"]


def test_two_grids_in_one_model(tmp_path):
    """The second grid must produce output at ITS OWN shape, not the first's."""
    m = _mk(tmp_path)
    a, b = (8, 6, 5), (6, 5, 4)
    ya = _fwd(m, a)
    yb = _fwd(m, b)
    assert ya.shape[2:] == a, f"first grid: {ya.shape[2:]} != {a}"
    assert yb.shape[2:] == b, f"second grid reshaped to the first: {yb.shape[2:]} != {b}"


def test_returning_to_a_seen_grid_reuses_its_graph(tmp_path):
    m = _mk(tmp_path)
    a, b = (8, 6, 5), (6, 5, 4)
    _fwd(m, a); _fwd(m, b)
    assert set(m._graphs) == {a, b}, f"expected both graphs cached, got {set(m._graphs)}"
    ya2 = _fwd(m, a)
    assert ya2.shape[2:] == a
    assert set(m._graphs) == {a, b}, "revisiting a grid must not rebuild"


def test_edge_sets_actually_differ_per_grid(tmp_path):
    """Guards against 'accepted the shape but kept the old edges'."""
    m = _mk(tmp_path)
    a, b = (8, 6, 5), (6, 5, 4)
    _fwd(m, a); _fwd(m, b)
    assert m._graphs[a]["n_phys"] == 8 * 6 * 5
    assert m._graphs[b]["n_phys"] == 6 * 5 * 4
    assert m._graphs[a]["enc_edges"].shape != m._graphs[b]["enc_edges"].shape
