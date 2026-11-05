# pyright: reportMissingImports=false
#
# This file is a port of RIGNO (https://github.com/camlab-ethz/rigno), which is
# distributed under the MIT License reproduced here:
#
#   MIT License
#
#   Copyright (c) 2025 Computational and Applied Mathematics Laboratory @ ETH Zurich
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#
# The changes made in porting it to PyTorch and to this pipeline are licensed
# under the PolyForm Noncommercial License 1.0.0, with the rest of this repository.
"""RIGNO (Region Interaction Graph Neural Operator) as a PhysicsNeMo ``Arch``.

Faithful PyTorch port of the JAX/Flax reference implementation (Mousavi et al.,
"RIGNO", arXiv:2501.19205), adapted to the lpno pipeline as a drop-in alternative to the FNO on
the *grid* data layout. The design choices are labelled C1..C8 below.

Only the time-independent, non-periodic, direct-operator mode is implemented
(C1/C3): the model maps the concatenated input channels on a structured grid to
the output channels, with no lead-time/stepper machinery. As in the reference,
the graph is data: it is built (and cached, in memory and on disk) per grid
shape as forwards arrive, so one model serves mixed-extent datasets (C5,
revised 2026-08-17 -- the earlier single-graph buffer bound a model to the
first grid it saw and crashed ti64_geom40 at step 0).
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from scipy.spatial import Delaunay, cKDTree
from torch import Tensor

from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Arch


def _scatter_add_native(src: Tensor, index: Tensor, dim: int = -1,
                        dim_size: int | None = None) -> Tensor:
    """`torch_scatter.scatter_add` for an index already expanded to `src`'s shape.

    torch_scatter's `scatter_add` is `scatter_sum`, which broadcasts the index and then
    calls this same `Tensor.scatter_add_` (no compiled kernel), so the two agree exactly.
    RIGNO's only call (`InteractionStep`) passes a fully expanded index.
    """
    if index.shape != src.shape:
        raise ValueError(f"native scatter_add needs an index shaped like src: {tuple(index.shape)} "
                         f"vs {tuple(src.shape)}")
    shape = list(src.shape)
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() else 0
    shape[dim] = dim_size
    return src.new_zeros(shape).scatter_add_(dim, index, src)


# Optional, like GAOT's (gaot_layers/_scatter_backend.py): RIGNO needs only scatter_add,
# which native torch covers exactly, so a missing torch_scatter no longer stops RIGNO.
# PINO_FORCE_NATIVE_SCATTER forces the native path on a box that has torch_scatter.
try:
    import torch_scatter as _torch_scatter
except ImportError:  # pragma: no cover - depends on the environment
    _torch_scatter = None
if _torch_scatter is not None and not os.environ.get("PINO_FORCE_NATIVE_SCATTER", "").strip():
    scatter_add = _torch_scatter.scatter_add
    SCATTER_BACKEND = "torch_scatter"
else:
    scatter_add = _scatter_add_native
    SCATTER_BACKEND = "native"


# --------------------------------------------------------------------------- #
# Graph construction (numpy/scipy, run once per grid, cached to disk).
# --------------------------------------------------------------------------- #
def _subsample_pointset(coords: np.ndarray, factor: float, rng: np.random.Generator,
                        rnode_count: int = 0) -> np.ndarray:
    """Randomly subsample a point cloud by ``factor`` (RIGNO ``_subsample_pointset``).

    ``rnode_count > 0`` pins the number of kept nodes instead, so evaluating at a
    different grid resolution can keep the rnode budget the model was trained with
    (the reference's ``rmesh_correction_dsf`` serves the same purpose).
    """
    n = coords.shape[0]
    perm = rng.permutation(n)
    keep = min(n, rnode_count) if rnode_count > 0 else max(4, int(n / factor))
    return coords[perm[:keep]]


def _edges_from_triangulation(tri: Delaunay) -> np.ndarray:
    """Undirected edges of a Delaunay triangulation, both directions (ND)."""
    indptr, cols = tri.vertex_neighbor_vertices
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    edges = np.stack([rows, cols], axis=-1)
    return np.concatenate([edges, np.flip(edges, axis=-1)], axis=0)


def _knn_edges(centers: np.ndarray, points: np.ndarray, k: int) -> np.ndarray:
    """Edges (point_idx, center_idx): each point to its ``k`` nearest centers (C4).

    kNN support instead of the reference's 2D Delaunay-median radius. It bounds
    the degree exactly (E = N_points * k), which keeps the fine 3D grid tractable,
    and guarantees every point is connected. Radius-based support would be
    the alternative.
    """
    k = min(k, centers.shape[0])
    tree = cKDTree(centers)
    _, idx = tree.query(points, k=k)  # (N, k)
    idx = idx.reshape(points.shape[0], k)
    pts = np.repeat(np.arange(points.shape[0]), k)
    return np.stack([pts, idx.reshape(-1)], axis=-1)


def _nn_scale(x: np.ndarray) -> np.ndarray:
    """Per-node characteristic length = distance to nearest other node."""
    tree = cKDTree(x)
    d, _ = tree.query(x, k=2)
    return d[:, 1]


def _r2r_edges(x_rnodes: np.ndarray, rmesh_levels: int, subsample_factor: float,
               rng: np.random.Generator) -> np.ndarray:
    """Multi-scale regional-mesh edges via Delaunay at successively coarser levels."""
    edges = []
    for level in range(rmesh_levels):
        size = int(x_rnodes.shape[0] / (subsample_factor ** level))
        if size < (x_rnodes.shape[1] + 2):
            continue
        sub = x_rnodes[:size]
        try:
            tri = Delaunay(sub)
        except Exception:
            continue
        edges.append(_edges_from_triangulation(tri))
    if not edges:
        # degenerate fallback: connect each node to its nearest neighbour
        d = np.linalg.norm(x_rnodes[:, None, :] - x_rnodes[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        nn_idx = d.argmin(axis=1)
        e = np.stack([np.arange(x_rnodes.shape[0]), nn_idx], axis=-1)
        return np.concatenate([e, np.flip(e, axis=-1)], axis=0)
    edges = np.concatenate(edges, axis=0)
    return np.unique(edges, axis=0)


def _edge_features(x_send: np.ndarray, x_recv: np.ndarray, senders: np.ndarray,
                   receivers: np.ndarray, max_edge_length: float) -> np.ndarray:
    """RIGNO structural edge features: [rel_pos / L, |rel_pos| / L]  (dim = d + 1)."""
    z = x_send[senders] - x_recv[receivers]
    d = np.linalg.norm(z, axis=-1, keepdims=True)
    return np.concatenate([z / max_edge_length, d / max_edge_length], axis=-1).astype(np.float32)


def build_region_interaction_graph(coords: np.ndarray, subsample_factor: float,
                                   rmesh_levels: int, k_p2r: int, k_r2p: int,
                                   seed: int, rnode_count: int = 0) -> Dict[str, np.ndarray]:
    """Build all RIGNO graph tensors from normalised physical-node coords in [-1, 1]."""
    rng = np.random.default_rng(seed)
    d = coords.shape[1]
    max_edge_length = 2.0 * np.sqrt(d)

    x_rnodes = _subsample_pointset(coords, subsample_factor, rng, rnode_count)
    scale = _nn_scale(x_rnodes)  # per-rnode characteristic length (structural feature)

    p2r = _knn_edges(x_rnodes, coords, k_p2r)      # (pnode, rnode)
    r2p_src = _knn_edges(x_rnodes, coords, k_r2p)  # (pnode, rnode) -> flipped below
    r2r = _r2r_edges(x_rnodes, rmesh_levels, subsample_factor, rng)

    p2r_send, p2r_recv = p2r[:, 0], p2r[:, 1]
    r2p_send, r2p_recv = r2p_src[:, 1], r2p_src[:, 0]  # rnode -> pnode
    r2r_send, r2r_recv = r2r[:, 0], r2r[:, 1]

    return {
        "x_rnodes": x_rnodes.astype(np.float32),
        "pnode_struct": coords.astype(np.float32),                       # (N, d)
        "rnode_struct": np.concatenate(                                  # (Nr, d+1)
            [x_rnodes, scale[:, None]], axis=-1).astype(np.float32),
        "p2r_senders": p2r_send.astype(np.int64), "p2r_receivers": p2r_recv.astype(np.int64),
        "p2r_edge_feat": _edge_features(coords, x_rnodes, p2r_send, p2r_recv, max_edge_length),
        "r2r_senders": r2r_send.astype(np.int64), "r2r_receivers": r2r_recv.astype(np.int64),
        "r2r_edge_feat": _edge_features(x_rnodes, x_rnodes, r2r_send, r2r_recv, max_edge_length),
        "r2p_senders": r2p_send.astype(np.int64), "r2p_receivers": r2p_recv.astype(np.int64),
        "r2p_edge_feat": _edge_features(x_rnodes, coords, r2p_send, r2p_recv, max_edge_length),
    }


def _cached_graph(coords: np.ndarray, subsample_factor: float, rmesh_levels: int,
                  k_p2r: int, k_r2p: int, seed: int, cache_dir: str,
                  rnode_count: int = 0) -> Dict[str, np.ndarray]:
    key = hashlib.md5(
        coords.tobytes()
        + np.array([subsample_factor, rmesh_levels, k_p2r, k_r2p, seed, rnode_count],
                   dtype=np.float64).tobytes()
    ).hexdigest()[:16]
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"rigno_graph_{key}.npz")
    if os.path.exists(path):
        data = np.load(path)
        return {k: data[k] for k in data.files}
    graph = build_region_interaction_graph(
        coords, subsample_factor, rmesh_levels, k_p2r, k_r2p, seed, rnode_count)
    np.savez(path, **graph)
    return graph


# --------------------------------------------------------------------------- #
# Network blocks.
# --------------------------------------------------------------------------- #
class FeedForwardBlock(nn.Module):
    """MLP with swish on hidden layers and optional LayerNorm on the output.

    Matches the reference ``FeedForwardBlock`` for ``mlp_hidden_layers`` hidden
    layers: sizes = [width] * hidden + [out]; activation on all but the last.
    """

    def __init__(self, in_dim: int, width: int, out_dim: int, hidden_layers: int = 1,
                 layer_norm: bool = True):
        super().__init__()
        sizes = [width] * hidden_layers + [out_dim]
        layers = []
        prev = in_dim
        for s in sizes:
            layers.append(nn.Linear(prev, s))
            prev = s
        self.layers = nn.ModuleList(layers)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(out_dim) if layer_norm else None

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class _FiLM(nn.Module):
    """Per-sample scalar conditioning: x * (1 + scale(c)) + bias(c).

    Adapted from the reference ConditionedNorm (models/utils.py:64-101): same
    Dense(hidden)->sigmoid->Dense(latent) heads with small-normal init so it
    starts near-identity. The reference multiplies the correction by c so it
    vanishes at tau=0; our conditioners (P, V_scan, H_norm) carry no such
    zero-lead-time semantics, so that factor is dropped.
    """

    def __init__(self, n_cond: int, latent: int, hidden: int = 16):
        super().__init__()
        self.scale = nn.Sequential(nn.Linear(n_cond, hidden), nn.Sigmoid(),
                                   nn.Linear(hidden, latent))
        self.bias = nn.Sequential(nn.Linear(n_cond, hidden), nn.Sigmoid(),
                                  nn.Linear(hidden, latent))
        for head in (self.scale, self.bias):
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.zeros_(head[-1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        # x: (B, N, L); c: (B, k)
        return x * (1.0 + self.scale(c)).unsqueeze(1) + self.bias(c).unsqueeze(1)


def _mask_edges(senders: Tensor, receivers: Tensor, feat: Tensor, p: float,
                training: bool) -> tuple[Tensor, Tensor, Tensor]:
    """Randomly drop a fraction ``p`` of edges during training (regularisation)."""
    if not training or p <= 0.0:
        return senders, receivers, feat
    n = senders.shape[0]
    keep = int((1.0 - p) * n)
    idx = torch.randperm(n, device=senders.device)[:keep]
    return senders[idx], receivers[idx], feat[idx]


def _step_no_msg(step, *args):
    """Run a message-passing step, dropping its edge output.

    Only the processor consumes `msg` (as the edge residual); the encoder and
    decoder discard it. That matters under checkpointing: a tensor *returned*
    from a checkpointed function is stored rather than freed and recomputed, and
    the decoder's msg is (B, 3.6M, 96) = 1.29 GiB at the flow3D grid. Dropping it
    inside the checkpoint boundary keeps it off the stored set.
    """
    new_send, new_recv, _ = step(*args)
    return new_send, new_recv


def _film_step(film, step, latent_r, cond, edge, s, rec, n_rnodes):
    """One processor step plus its FiLM modulation, as a single checkpointable unit.

    FiLM belongs *inside* the boundary: applied outside, its output is an argument
    of the checkpointed call and is therefore stored, costing an extra
    (B, n_rnodes, latent) per processor step on top of the pre-FiLM tensor. That
    is what defeats checkpointing at full resolution -- measured on the 1601
    random-path data, `grad_checkpointing` moved a FiLM run only 23.37 -> 22.99 GiB
    and still OOMed. Inside, only the pre-FiLM input is stored and the modulation
    is recomputed in backward.

    Recompute is exact: FiLM is Linear/Sigmoid with no randomness, and edge masking
    already ran once per forward before the loop.

    Returns the post-FiLM residual, so the caller's latent matches the unfused
    path (`film_out + new_r`, not `latent_r + new_r`).
    """
    if film is not None:
        latent_r = film(latent_r, cond)
    _, new_r, new_e = step(latent_r, latent_r, edge, s, rec, n_rnodes)
    return latent_r + new_r, new_e


def _degree(receivers: Tensor, n_recv: int, dtype: torch.dtype) -> Tensor:
    """Per-receiver edge count, clamped at 1 -- scatter_mean's divisor.

    Recomputed per call rather than cached: it is one bincount over the edge
    index (microseconds next to the MLPs) and it stays correct when edge masking
    resamples `receivers`, which an identity-keyed cache would not.
    """
    return torch.bincount(receivers, minlength=n_recv).clamp(min=1).to(dtype)


class InteractionStep(nn.Module):
    """One InteractionNetwork message-passing step over a single edge set.

    Edge update  : e' = MLP_e([e, x_send[send], x_recv[recv]])
    Recv update  : r' = MLP_r([x_recv, mean_recv(e')])
    Sender update: s' = MLP_s([x_send])            (senders receive no messages)
    Residuals are added by the caller when appropriate.
    """

    def __init__(self, latent: int, hidden_layers: int, update_sender: bool):
        super().__init__()
        self.edge_mlp = FeedForwardBlock(3 * latent, latent, latent, hidden_layers)
        self.recv_mlp = FeedForwardBlock(2 * latent, latent, latent, hidden_layers)
        self.sender_mlp = (FeedForwardBlock(latent, latent, latent, hidden_layers)
                           if update_sender else None)

    def forward(self, x_send: Tensor, x_recv: Tensor, edge: Tensor,
                senders: Tensor, receivers: Tensor, n_recv: int):
        # x_*: (B, N, L); edge: (B, E, L)
        msg = self.edge_mlp(torch.cat(
            [edge, x_send[:, senders], x_recv[:, receivers]], dim=-1))
        idx = receivers.view(1, -1, 1).expand(msg.shape[0], -1, msg.shape[-1])
        # scatter_mean would allocate a full (B, E, L) tensor of ones just to
        # count contributions per receiver -- 1.32 GiB for the r2p edge set at
        # the flow3D grid (3.6M edges x 96 channels), and it lands on the
        # autograd tape too. The divisor only depends on `receivers`, which is a
        # fixed graph, so count it once as an (n_recv,) degree vector instead.
        # Same sum, same clamp(min=1) divisor as scatter_mean: bitwise identical.
        # ...and the accumulation itself runs in fp32 even under autocast
        # (BUG-43). Autocast promotes `torch.sum` and `LayerNorm` to fp32 but has
        # no rule for `torch_scatter.scatter_add`, a third-party op, so this --
        # the model's only reduction, summing millions of edge messages -- was the
        # one left in fp16 (max 65504). The matmuls above stay fp16, where the
        # speed is; only the sum is protected.
        with torch.autocast(msg.device.type, enabled=False):
            agg = scatter_add(msg.float(), idx, dim=1, dim_size=n_recv) / _degree(
                receivers, n_recv, torch.float32).view(1, -1, 1)
        agg = agg.to(msg.dtype)
        new_recv = self.recv_mlp(torch.cat([x_recv, agg], dim=-1))
        new_send = self.sender_mlp(x_send) if self.sender_mlp is not None else None
        return new_send, new_recv, msg


# --------------------------------------------------------------------------- #
# The Arch.
# --------------------------------------------------------------------------- #
class RIGNOArch(Arch):
    """Region Interaction Graph Neural Operator on a fixed structured grid."""

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        dimension: int = 3,
        detach_keys: List[Key] = [],
        node_latent_size: int = 64,
        edge_latent_size: int = 64,
        processor_steps: int = 6,
        mlp_hidden_layers: int = 1,
        subsample_factor: float = 8.0,
        rmesh_levels: int = 4,
        k_p2r: int = 4,
        k_r2p: int = 4,
        p_edge_masking: float = 0.5,
        graph_seed: int = 0,
        cache_dir: str = "outputs/rigno_graphs",
        grad_checkpointing: bool = False,
        num_graph_variants: int = 1,
        inference_edge_masking: float = 0.0,
        rnode_count: int = 0,
        film_keys: List[str] = [],
    ) -> None:
        super().__init__(input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys)
        assert node_latent_size == edge_latent_size, "port assumes node_latent == edge_latent"
        self.latent = node_latent_size
        self.hidden_layers = mlp_hidden_layers
        self.p_edge_masking = p_edge_masking
        # The reference resamples the regional mesh every epoch (train.py:883-887);
        # we pre-build `num_graph_variants` graphs (seeds graph_seed..+N-1) and
        # rotate per training forward. Eval always uses variant 0.
        self._n_variants = max(1, int(num_graph_variants))
        self._forced_variant = None  # test/debug hook: pin a specific variant
        # Reference-style inference-time edge masking (test.py:791-913): repeated
        # stochastic eval forwards give a near-free predictive ensemble. 0 = off.
        self.inference_edge_masking = float(inference_edge_masking)
        # Recompute the encoder (p2r), every processor step (r2r) and the decoder
        # (r2p) in backward instead of storing their activations (trades ~1 extra
        # forward for far less activation memory). Off by default so trained
        # baselines are byte-for-byte unchanged; needed to fit multi-step
        # autoregressive unrolling (k>=3) in 24 GB. The encoder/decoder matter most:
        # they message-pass over the physical grid (600k nodes / ~3.6M r2p edges at
        # the flow3D resolution) while the processor runs on the subsampled regional
        # mesh, so checkpointing only the processor still OOMed inside dec_step.
        self._grad_checkpoint = bool(grad_checkpointing)
        self.dim = int(dimension)
        self.num_outputs = sum(self.output_key_dict.values())
        # FiLM conditioning: broadcast-constant scalar channels (P, V_scan,
        # H_norm, mat_*) leave the pnode features and modulate the processor
        # latents instead — cheaper than carrying k constants on every pnode.
        self.film_keys = [str(k) for k in (film_keys or [])]
        input_names = list(self.input_key_dict.keys())
        unknown_film = [k for k in self.film_keys if k not in input_names]
        assert not unknown_film, f"film_keys not in input keys: {unknown_film}"
        self._film_idx = [input_names.index(k) for k in self.film_keys]
        self._keep_idx = [i for i, k in enumerate(input_names) if k not in self.film_keys]
        in_channels = sum(self.input_key_dict.values()) - len(self.film_keys)

        # Graph-build config; graphs are built lazily per grid shape as forwards
        # arrive. The reference treats the graph as DATA (built per batch from
        # `Batch.g`, train.py:278) and its weights are feature-dim MLPs, so one
        # model serves any point cloud; caching per shape is our optimization on
        # top, keyed the same way as the on-disk `_cached_graph` (by coords).
        # An earlier port cached ONE graph behind a `_graph_built` guard, which
        # bound the model to the first grid it ever saw and crashed geom40 at
        # step 0 (2026-08-16, workchain 50818).
        self._graph_cfg = dict(subsample_factor=subsample_factor, rmesh_levels=rmesh_levels,
                               k_p2r=k_p2r, k_r2p=k_r2p, graph_seed=graph_seed,
                               cache_dir=cache_dir, rnode_count=int(rnode_count))
        self.grid_shape = None          # shape of the LAST forward (kept: tests, output reshape)
        # CPU-resident store, one shape's tensors on the compute device at a
        # time. All-on-GPU measured ~14 GB for geom40's 40 shapes x 8 variants
        # -- an instant OOM next to ~10 GB of activations -- while one active
        # shape is <700 MB and a shape SWITCH is a single H2D copy (~40 ms,
        # rare under case-coherent sampling; single-grid datasets copy once
        # ever). ponytail: unbounded CPU store (~14 GB for geom40); spill to
        # the on-disk npz cache via an LRU if a sweep with hundreds of
        # distinct extents ever outgrows host RAM.
        self._graphs: "OrderedDict[tuple, dict]" = OrderedDict()
        self._active_graph: tuple | None = None   # (shape, device) currently on device

        pnode_in = in_channels + self.dim          # field channels + coords
        rnode_in = in_channels + self.dim + 1      # dummy field zeros + coords + scale
        edge_in = self.dim + 1                      # rel_pos + dist

        L, H = self.latent, self.hidden_layers
        # Encoder (p2r): embed nodes + edges, then one interaction step (updates both sides).
        self.enc_pnode_embed = FeedForwardBlock(pnode_in, L, L, H)
        self.enc_rnode_embed = FeedForwardBlock(rnode_in, L, L, H)
        self.enc_edge_embed = FeedForwardBlock(edge_in, L, L, H)
        self.enc_step = InteractionStep(L, H, update_sender=True)

        # Processor (r2r): embed edges once, then `processor_steps` steps with residuals.
        self.proc_edge_embed = FeedForwardBlock(edge_in, L, L, H)
        self.proc_steps = nn.ModuleList(
            [InteractionStep(L, H, update_sender=False) for _ in range(processor_steps)])
        # One FiLM per processor step (unshared, like the reference's per-step
        # ConditionedNorm), applied to the regional latents before each step.
        self.film_steps = (nn.ModuleList(
            [_FiLM(len(self.film_keys), L) for _ in range(processor_steps)])
            if self.film_keys else None)

        # Decoder (r2p): embed edges, one step (updates pnodes), then per-node output MLP.
        self.dec_edge_embed = FeedForwardBlock(edge_in, L, L, H)
        self.dec_step = InteractionStep(L, H, update_sender=False)
        self.dec_out = FeedForwardBlock(L, self.num_outputs, self.num_outputs, H, layer_norm=False)

    def _graph_for(self, spatial_shape: tuple, device) -> dict:
        """The region-interaction graph for this grid shape, built on first sight.

        One entry per distinct shape, matching the reference's graph-as-data
        design; revisiting a shape is a dict hit, not a rebuild (the disk layer
        under `_cached_graph` already keys by the coordinate bytes).
        """
        shape = tuple(int(s) for s in spatial_shape)
        assert len(shape) == self.dim, (
            f"RIGNO built for dimension={self.dim} but got grid {shape}")
        entry = self._graphs.get(shape)
        if entry is None:
            entry = self._build_graph(shape, torch.device("cpu"))
            self._graphs[shape] = entry
        if device.type == "cpu":
            return entry
        if self._active_graph != (shape, device):
            self._device_entry = dict(entry)
            self._device_entry["tensors"] = {
                k: t.to(device, non_blocking=True) for k, t in entry["tensors"].items()}
            self._device_entry["device"] = device
            self._active_graph = (shape, device)
        return self._device_entry

    def _build_graph(self, shape: tuple, device) -> dict:
        """Build every variant of the graph for one grid shape."""
        axes = [np.linspace(-1.0, 1.0, n, dtype=np.float64) for n in shape]
        mesh = np.meshgrid(*axes, indexing="ij")
        coords = np.stack([m.reshape(-1) for m in mesh], axis=-1)  # (N, d), row-major
        long_names = ("p2r_senders", "p2r_receivers", "r2r_senders", "r2r_receivers",
                      "r2p_senders", "r2p_receivers")
        float_names = ("p2r_edge_feat", "r2r_edge_feat", "r2p_edge_feat",
                       "pnode_struct", "rnode_struct")
        entry: dict = {"n_pnodes": coords.shape[0], "device": device, "tensors": {}}
        for v in range(self._n_variants):
            graph = _cached_graph(coords, self._graph_cfg["subsample_factor"],
                                  self._graph_cfg["rmesh_levels"], self._graph_cfg["k_p2r"],
                                  self._graph_cfg["k_r2p"], self._graph_cfg["graph_seed"] + v,
                                  self._graph_cfg["cache_dir"], self._graph_cfg["rnode_count"])
            if v == 0:
                entry["n_rnodes"] = int(graph["x_rnodes"].shape[0])
            else:
                # `_subsample_pointset` keeps a count that depends only on N, so
                # every variant has the same rnode budget.
                assert int(graph["x_rnodes"].shape[0]) == entry["n_rnodes"]
            suffix = "" if v == 0 else f"_v{v}"
            for name in long_names:
                entry["tensors"][name + suffix] = torch.as_tensor(
                    graph[name], dtype=torch.long, device=device)
            for name in float_names:
                if name == "pnode_struct" and v > 0:
                    continue  # identical across variants (same physical grid)
                entry["tensors"][name + suffix] = torch.as_tensor(
                    graph[name], dtype=torch.float32, device=device)
        return entry

    def _encode_process_decode(self, pnode_feat: Tensor, cond: Tensor | None,
                               graph: dict) -> Tensor:
        def _g(name: str, variant: int) -> Tensor:
            if variant == 0 or name == "pnode_struct":
                return graph["tensors"][name]
            return graph["tensors"][f"{name}_v{variant}"]

        n_rnodes, n_pnodes = graph["n_rnodes"], graph["n_pnodes"]
        B = pnode_feat.shape[0]
        # Variant rotation: random per training forward (approximates the
        # reference's per-epoch rmesh resampling), variant 0 in eval.
        if self._forced_variant is not None:
            v = int(self._forced_variant)
        elif self.training and self._n_variants > 1:
            v = int(torch.randint(self._n_variants, (1,)).item())
        else:
            v = 0
        # Edge masking: training uses p_edge_masking; eval uses
        # inference_edge_masking (0 = deterministic eval, unchanged behaviour).
        p_mask = self.p_edge_masking if self.training else self.inference_edge_masking
        struct_p = _g("pnode_struct", v).unsqueeze(0).expand(B, -1, -1)
        struct_r = _g("rnode_struct", v).unsqueeze(0).expand(B, -1, -1)
        dummy = pnode_feat.new_zeros(B, n_rnodes, pnode_feat.shape[-1])

        # Edge masking happens outside every checkpointed body (once per forward),
        # so each recompute is deterministic.
        ckpt = self._grad_checkpoint and self.training and torch.is_grad_enabled()

        # ---- Encoder (p2r) ----
        p = self.enc_pnode_embed(torch.cat([pnode_feat, struct_p], dim=-1))
        r = self.enc_rnode_embed(torch.cat([dummy, struct_r], dim=-1))
        s, rec, feat = _mask_edges(_g("p2r_senders", v), _g("p2r_receivers", v),
                                   self.enc_edge_embed(_g("p2r_edge_feat", v)),
                                   p_mask, p_mask > 0)
        feat = feat.unsqueeze(0).expand(B, -1, -1)
        if ckpt:
            new_p, new_r = checkpoint(
                _step_no_msg, self.enc_step, p, r, feat, s, rec, n_rnodes,
                use_reentrant=False)
        else:
            new_p, new_r, _ = self.enc_step(p, r, feat, s, rec, n_rnodes)
        latent_p = p + new_p          # senders (physical)
        latent_r = r + new_r          # receivers (regional)

        # ---- Processor (r2r) ----
        # Mask the regional edge set once per forward, then run all steps on it,
        # carrying node + edge residuals across steps (matches the reference).
        s, rec, edge = _mask_edges(_g("r2r_senders", v), _g("r2r_receivers", v),
                                   self.proc_edge_embed(_g("r2r_edge_feat", v)),
                                   p_mask, p_mask > 0)
        edge = edge.unsqueeze(0).expand(B, -1, -1)
        for step_idx, step in enumerate(self.proc_steps):
            # Per-step unshared FiLM parameters, passed in rather than closed over
            # so the checkpointed and plain paths run the *same* function body.
            film = None if self.film_steps is None else self.film_steps[step_idx]
            if ckpt:
                # Recompute this step in backward; edge masking already happened
                # above (once per forward), so the recompute is deterministic.
                latent_r, new_e = checkpoint(
                    _film_step, film, step, latent_r, cond, edge, s, rec,
                    n_rnodes, use_reentrant=False)
            else:
                latent_r, new_e = _film_step(film, step, latent_r, cond, edge, s,
                                             rec, n_rnodes)
            edge = edge + new_e

        # ---- Decoder (r2p) ----
        s, rec, feat = _mask_edges(_g("r2p_senders", v), _g("r2p_receivers", v),
                                   self.dec_edge_embed(_g("r2p_edge_feat", v)),
                                   p_mask, p_mask > 0)
        feat = feat.unsqueeze(0).expand(B, -1, -1)
        if ckpt:
            _, new_p = checkpoint(
                _step_no_msg, self.dec_step, latent_r, latent_p, feat, s, rec,
                n_pnodes, use_reentrant=False)
        else:
            _, new_p, _ = self.dec_step(latent_r, latent_p, feat, s, rec, n_pnodes)
        latent_p = latent_p + new_p
        return self.dec_out(latent_p)  # (B, N, num_outputs)

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.prepare_input(
            in_vars, self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict, dim=1, input_scales=self.input_scales,
        )  # (B, C_in, nx, ny, nz)
        cond = None
        if self.film_keys:
            # Broadcast-constant channels -> per-sample scalars (mean over space).
            cond = x[:, self._film_idx].mean(dim=tuple(range(2, x.ndim)))
            x = x[:, self._keep_idx]
        B, C = x.shape[0], x.shape[1]
        graph = self._graph_for(x.shape[2:], x.device)
        # Last-forward view, kept as public surface (tests, external inspection).
        self.grid_shape = tuple(int(n) for n in x.shape[2:])
        self.n_pnodes, self.n_rnodes = graph["n_pnodes"], graph["n_rnodes"]
        pnode_feat = x.reshape(B, C, -1).permute(0, 2, 1).contiguous()  # (B, N, C_in)
        out = self._encode_process_decode(pnode_feat, cond, graph)      # (B, N, C_out)
        y = out.permute(0, 2, 1).reshape(B, self.num_outputs, *self.grid_shape)
        return self.prepare_output(y, self.output_key_dict, dim=1, output_scales=self.output_scales)


# --------------------------------------------------------------------------- #
# Autoregressive delta / residual-in-time reparametrization.
# --------------------------------------------------------------------------- #
class DeltaArch(Arch):
    """Wrap any base ``Arch`` so it predicts ``u_{t+1} = u_t + f(x)``.

    Family-agnostic (FNO/RIGNO/UNet/...). For each fed-back state key ``k``, the
    previous field ``k{prev_suffix}`` — already a network input channel in
    autoregressive mode (config_loading appends the ``*_prev`` inputs) — is ADDED
    to the base network output. The network then only has to predict the per-step
    increment, which is small for smoothly-evolving fields. The hypothesis is that
    carrying the base level exactly through the skip improves one-step accuracy and
    the smooth wake; note it does not by itself stop rollout error accumulation
    (a biased increment integrates into drift), so it is meant to be trained under
    the multi-step unroll regime and compared there.

    Requires autoregressive mode so the ``*_prev`` inputs exist; keys without a
    matching ``*_prev`` input are left as absolute outputs (e.g. an aux ``T_dot``).

    Scales: input/output scales are identity in this pipeline (normalization is at
    the data level via ``ref_scales``), so ``out[k]`` and ``in[k_prev]`` share
    units and the add is well-posed. Revisit if per-key output_scales are ever set.
    """

    def __init__(self, base: Arch, state_keys: List[str], state_prev_suffix: str = "_prev") -> None:
        in_keys = [Key(name, size=size) for name, size in base.input_key_dict.items()]
        out_keys = [Key(name, size=size) for name, size in base.output_key_dict.items()]
        detach_keys = [Key(name) for name in getattr(base, "detach_key_dict", {}).keys()]
        super().__init__(input_keys=in_keys, output_keys=out_keys, detach_keys=detach_keys)
        self.base = base
        self._prev_suffix = str(state_prev_suffix)
        in_names = set(base.input_key_dict.keys())
        self._delta_keys = [str(k) for k in state_keys if f"{k}{self._prev_suffix}" in in_names]
        if not self._delta_keys:
            raise ValueError(
                "DeltaArch: no state key has a matching "
                f"'*{self._prev_suffix}' input (state_keys={list(state_keys)}, "
                f"inputs={sorted(in_names)}). Enable autoregressive so the *_prev "
                "inputs are added, or turn off predict_delta."
            )

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        out = self.base(in_vars)
        for key in self._delta_keys:
            prev_key = f"{key}{self._prev_suffix}"
            if key in out and prev_key in in_vars:
                out[key] = out[key] + in_vars[prev_key]
        return out


# --------------------------------------------------------------------------- #
# End-to-end refinement: RIGNO base + residual 3D-FNO head, trained jointly.
# The diagnostic corrector (off-pipeline) showed a SPATIAL FNO reduces rp3's
# smooth-bias error ~31% on held-out test where a pointwise MLP only reaches
# ~5%. This wires that refinement in-model so the base RIGNO co-adapts (no
# post-hoc leakage): refined = rigno(x) + head(rigno(x)); head zero-init so the
# model starts as pure RIGNO and learns the correction from there.
# --------------------------------------------------------------------------- #
class _SpectralConv3d(nn.Module):
    # Weights stored as REAL tensors (real/imag at index 0/1) rather than
    # torch.cfloat, so real-only optimizers work (Lion calls torch.sign, which
    # rejects complex params).
    def __init__(self, cin: int, cout: int, m1: int, m2: int, m3: int):
        super().__init__()
        self.m1, self.m2, self.m3 = m1, m2, m3
        scale = 1.0 / (cin * cout)
        self.w = nn.Parameter(scale * torch.rand(2, cin, cout, m1, m2, m3))

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        xf = torch.fft.rfftn(x, dim=[-3, -2, -1])
        m1 = min(self.m1, x.shape[-3]); m2 = min(self.m2, x.shape[-2]); m3 = min(self.m3, x.shape[-1] // 2 + 1)
        a, b = xf[..., :m1, :m2, :m3].real, xf[..., :m1, :m2, :m3].imag
        c, d = self.w[0, ..., :m1, :m2, :m3], self.w[1, ..., :m1, :m2, :m3]
        # (a+ib)(c+id) = (ac-bd) + i(ad+bc)
        real = torch.einsum("bixyz,ioxyz->boxyz", a, c) - torch.einsum("bixyz,ioxyz->boxyz", b, d)
        imag = torch.einsum("bixyz,ioxyz->boxyz", a, d) + torch.einsum("bixyz,ioxyz->boxyz", b, c)
        out = torch.zeros(B, self.w.shape[2], x.shape[-3], x.shape[-2], x.shape[-1] // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        out[..., :m1, :m2, :m3] = torch.complex(real, imag)
        return torch.fft.irfftn(out, s=x.shape[-3:], dim=[-3, -2, -1])


class _FNORefineHead(nn.Module):
    """Small 3D FNO producing a residual on the base output field."""

    def __init__(self, channels: int, width: int = 24, modes=(12, 12, 8), layers: int = 4,
                 padding: int = 0):
        super().__init__()
        self.padding = int(padding)          # zero-pad each spatial dim before rFFT
        m = tuple(int(v) for v in modes)
        self.lift = nn.Conv3d(channels, width, 1)
        self.specs = nn.ModuleList(_SpectralConv3d(width, width, *m) for _ in range(layers))
        self.ws = nn.ModuleList(nn.Conv3d(width, width, 1) for _ in range(layers))
        self.proj = nn.Sequential(nn.Conv3d(width, 64, 1), nn.GELU(), nn.Conv3d(64, channels, 1))
        nn.init.zeros_(self.proj[-1].weight); nn.init.zeros_(self.proj[-1].bias)  # identity at init

    def forward(self, x: Tensor) -> Tensor:
        h = self.lift(x)
        p = self.padding
        if p:
            # Zero-pad the high side of D,H,W so the FFT's periodic wraparound sees
            # the padded (empty) region, not the hot-pool/cold-edge discontinuity —
            # kills the non-periodic boundary ringing (BUG-41). Cropped back after.
            h = torch.nn.functional.pad(h, (0, p, 0, p, 0, p))
        for s, w in zip(self.specs, self.ws):
            h = torch.nn.functional.gelu(s(h) + w(h))
        if p:
            h = h[..., :x.shape[-3], :x.shape[-2], :x.shape[-1]]
        return self.proj(h)


class _ConvBlock3d(nn.Module):
    """Two padded 3x3x3 convs + GELU; preserves spatial size."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1), nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1), nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class _UNetRefineHead(nn.Module):
    """Small 3D U-Net producing a residual on the base output field.

    The ripple-free alternative to _FNORefineHead: local convs + pooling give a
    large (near-global) receptive field with NO FFT, so it can sharpen the melt
    pool without the global spectral (Gibbs) ringing the FNO head injects into
    the late-frame wake. `levels` down/up-sampling stages
    (avg-pool down, trilinear up to the exact encoder size so odd halved dims are
    safe). Final conv zero-init -> identity at init, so a warm-started base
    starts as exactly the pre-trained RIGNO and learns the correction.
    """

    def __init__(self, channels: int, width: int = 24, levels: int = 3):
        super().__init__()
        self.enc = nn.ModuleList()
        self.enc_widths = []
        c, w = channels, width
        for _ in range(levels):
            self.enc.append(_ConvBlock3d(c, w))
            self.enc_widths.append(w)
            c, w = w, w * 2
        self.bottleneck = _ConvBlock3d(c, w)      # c = enc_widths[-1], w = 2*that
        self.reduce = nn.ModuleList()             # 1x1 channel-match before concat
        self.dec = nn.ModuleList()
        c_up = w
        for skip_w in reversed(self.enc_widths):
            self.reduce.append(nn.Conv3d(c_up, skip_w, 1))
            self.dec.append(_ConvBlock3d(2 * skip_w, skip_w))
            c_up = skip_w
        self.proj = nn.Conv3d(c_up, channels, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)  # identity at init

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        h = x
        for enc in self.enc:
            h = enc(h)
            skips.append(h)
            h = torch.nn.functional.avg_pool3d(h, 2, ceil_mode=True)
        h = self.bottleneck(h)
        for reduce, dec, skip in zip(self.reduce, self.dec, reversed(skips)):
            h = reduce(h)
            h = torch.nn.functional.interpolate(h, size=skip.shape[-3:],
                                                mode="trilinear", align_corners=False)
            h = dec(torch.cat([h, skip], dim=1))
        return self.proj(h)


class RIGNORefinedArch(Arch):
    """RIGNO base with a residual 3D-FNO refinement head, trained end-to-end."""

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        head_width: int = 24,
        head_modes=(12, 12, 8),
        head_layers: int = 4,
        head_type: str = "fno",
        head_padding: int = 0,
        base_init_ckpt: str = None,
        detach_keys: List[Key] = [],
        **rigno_kwargs,
    ) -> None:
        super().__init__(input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys)
        assert not rigno_kwargs.get("film_keys"), \
            "film_keys is not supported for rigno_refined (its forward bypasses the film split)"
        self.rigno = RIGNOArch(input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys, **rigno_kwargs)
        # head_type "unet": local-conv U-Net corrector (no FFT, no Gibbs ringing);
        # else the spectral FNO head (head_padding zero-pads the FFT for non-periodic
        # domains). `head_layers` = U-Net levels when type=unet.
        if str(head_type) == "unet":
            self.head = _UNetRefineHead(self.rigno.num_outputs, head_width, levels=head_layers)
        else:
            self.head = _FNORefineHead(self.rigno.num_outputs, head_width, head_modes, head_layers,
                                       padding=head_padding)
        if base_init_ckpt:
            # Warm-start the base from a trained RIGNO (e.g. rp3): load its
            # learnable weights (strict=False skips lazily-built graph buffers,
            # which are rebuilt from cache). Head stays zero-init -> the model
            # starts as exactly the pre-trained RIGNO, then co-adapts.
            sd = torch.load(base_init_ckpt, map_location="cpu")
            miss, unexp = self.rigno.load_state_dict(sd, strict=False)
            print(f"[rigno_refined] warm-started base from {base_init_ckpt} "
                  f"({len(sd)} tensors, missing={len(miss)}, unexpected={len(unexp)})")

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        r = self.rigno
        x = r.prepare_input(
            in_vars, r.input_key_dict.keys(), detach_dict=r.detach_key_dict, dim=1, input_scales=r.input_scales,
        )
        B, C = x.shape[0], x.shape[1]
        if not r._graph_built:
            r._build_graph(x.shape[2:], x.device)
        pnode_feat = x.reshape(B, C, -1).permute(0, 2, 1).contiguous()
        out = r._encode_process_decode(pnode_feat)
        y = out.permute(0, 2, 1).reshape(B, r.num_outputs, *r.grid_shape)
        y = y + self.head(y)                        # residual refinement (identity at init)
        return r.prepare_output(y, r.output_key_dict, dim=1, output_scales=r.output_scales)
