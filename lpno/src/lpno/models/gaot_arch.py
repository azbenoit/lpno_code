# pyright: reportMissingImports=false
#
# This file is a port of GAOT (https://github.com/camlab-ethz/GAOT, and GAOT-3D,
# https://github.com/Shizheng-Wen/GAOT-3D), which is distributed under the MIT License
# reproduced in gaot_layers/LICENSE. The changes made in porting it to this pipeline are
# licensed under the PolyForm Noncommercial License 1.0.0, with the rest of this repository.
"""GAOT (Geometry-Aware Operator Transformer) as a PhysicsNeMo ``Arch``.

Encoder/processor/decoder port of GAOT-3D (Wen et al., "Geometry-Aware
Operator Transformer", NeurIPS 2025, arXiv:2505.18781), adapted to the lpno
pipeline as a drop-in alternative to RIGNO on the *grid* data layout — the same
contract ``rigno_arch.RIGNOArch`` implements: concatenated input channels on a
fixed structured grid in, output channels on that grid out.

Only the time-independent, direct-operator mode exists here -- the same mode
``RIGNOArch`` implements. There is no lead-time or stepper machinery, so the
architecture is agnostic to which temporal layout the pipeline feeds it:

* ``time_as_batch=false`` (what `config_flow3D_GAOT` inherits from config M):
  each sample is a **time-averaged, moving-reference-frame** field over
  ``time_avg_window`` steps. This is the layout the early GAOT-vs-RIGNO
  comparisons were measured on.
* ``time_as_batch=true``: each stored timestep is its own sample and time enters
  through the ``t_hat``/``beam_*``/``path_dose*`` input channels. Supported --
  those are ordinary input channels -- but not yet benchmarked for this family.

Differences from upstream GAOT-3D, all forced by "our geometry is one fixed
Cartesian grid" rather than a per-sample point cloud:

* Upstream drives everything through ``torch_geometric`` ``Batch`` objects and
  builds edges per sample with ``torch_cluster``. The grid here never changes,
  so edges are built **once** with ``scipy.spatial.cKDTree`` and carried as
  buffers — the same approach (and disk cache) ``RIGNOArch`` already uses. That
  drops the ``torch_geometric``/``torch_cluster`` dependency entirely; the
  vendored layers under ``gaot_layers/`` need only ``torch`` + ``torch_scatter``
  (already a RIGNO dependency).
* Coordinates are normalised **per dimension** to [-1, 1], matching
  ``RIGNOArch._build_graph``. Upstream's ``utils/scale.py:rescale`` takes a
  single global min/max over all axes, which would squash our anisotropic
  900 x 400 x 300 um box.

The vendored layers (``gaot_layers/``) are copied verbatim from upstream apart
from import paths and an optional-dependency guard on ``rotary_embedding_torch``
(only needed for ``positional_embedding="rope"``; the default here is
``"absolute"``, which needs no extra package).
"""

from __future__ import annotations

import hashlib
import itertools
import os
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from torch import Tensor

from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Arch

from .gaot_layers.attn import AttentionConfig, FFNConfig, Transformer, TransformerConfig
from .gaot_layers.geoembed import GeometricEmbedding
from .gaot_layers.integral_transform import IntegralTransform
from .gaot_layers.mlp import ChannelMLP


# --------------------------------------------------------------------------- #
# Graph construction (numpy/scipy, run once per grid, cached to disk).
# --------------------------------------------------------------------------- #
def _grid_coords(shape: Sequence[int]) -> np.ndarray:
    """Cell-centre coordinates of a structured grid, per-axis normalised to [-1, 1].

    Row-major (``indexing="ij"`` + C-order reshape) so node ``i`` corresponds to
    element ``i`` of ``x.reshape(B, C, -1)`` — the same flattening the forward
    pass uses, and the same convention as ``RIGNOArch``.
    """
    axes = [np.linspace(-1.0, 1.0, n, dtype=np.float64) for n in shape]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.reshape(-1) for m in mesh], axis=-1)


def _knn_edges(source: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Query-centric kNN: every QUERY point pulls its ``k`` nearest sources.

    ``[2, E]`` with row 0 indexing ``source`` and row 1 indexing ``query`` --
    upstream's convention (``integral_transform.py``: ``edge_index[0]`` into
    ``y_pos``/source, ``edge_index[1]`` into ``x_pos``/query), which
    ``GeometricEmbedding`` assumes too.

    Every query is guaranteed exactly ``k`` incoming edges, so this is the right
    direction for the DECODER, where the queries are the physical nodes we must
    produce an output for. It is the WRONG direction for the encoder: it bounds
    the query side, not the source side, so physical nodes that are nobody's
    nearest neighbour are simply never read (measured on the 61x40x40 grid
    against a 32x20x20 lattice: k=8 reads only 80.6% of the input field).
    """
    k = int(min(k, source.shape[0]))
    _, idx = cKDTree(source).query(query, k=k, workers=-1)
    idx = np.asarray(idx).reshape(query.shape[0], k)
    return np.stack(
        [idx.reshape(-1), np.repeat(np.arange(query.shape[0]), k)]
    ).astype(np.int64)


def _knn_edges_source_centric(source: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Source-centric kNN: every SOURCE point pushes to its ``k`` nearest queries.

    Same ``[2, E]`` layout (row 0 source, row 1 query). This is upstream's
    encoder ``knn`` strategy -- GAOT-3D's README states it as "each phys point ->
    k nearest latent tokens, direction phys -> latent" -- and it is what the
    ENCODER needs: it guarantees every physical node contributes to the latent
    representation, so no part of the input field is silently dropped.
    """
    k = int(min(k, query.shape[0]))
    _, idx = cKDTree(query).query(source, k=k, workers=-1)
    idx = np.asarray(idx).reshape(source.shape[0], k)
    return np.stack(
        [np.repeat(np.arange(source.shape[0]), k), idx.reshape(-1)]
    ).astype(np.int64)


def _radius_edges(
    source: np.ndarray, query: np.ndarray, radius: float, max_neighbors: int
) -> np.ndarray:
    """Radius-ball edges, optionally capped at ``max_neighbors`` nearest per query.

    A query with no source inside ``radius`` contributes no edge; the vendored
    layers already handle empty neighbourhoods (zeroed geometry features, zero
    integral contribution). ``knn`` avoids that failure mode entirely, which is
    why it is the default here.
    """
    nbrs = cKDTree(source).query_ball_point(query, r=radius, workers=-1)
    src, dst = [], []
    for q, ns in enumerate(nbrs):
        if not ns:
            continue
        if max_neighbors and len(ns) > max_neighbors:
            d = np.linalg.norm(source[ns] - query[q], axis=1)
            ns = list(np.asarray(ns)[np.argsort(d)[:max_neighbors]])
        src.extend(ns)
        dst.extend([q] * len(ns))
    if not src:
        raise ValueError(
            f"radius={radius} produced no edges for {query.shape[0]} query points; "
            "increase gno_radius or use neighbor_strategy='knn'."
        )
    return np.stack([np.asarray(src), np.asarray(dst)]).astype(np.int64)


def _trilinear_edges(latent_shape: Sequence[int], phys_shape: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Decoder edges and weights of trilinear interpolation from the latent lattice onto the physical grid.

    ``([2, 8 N], [8 N])``, row 0 indexing the latent tokens and row 1 the physical nodes (the ``_knn_edges``
    layout). Every node takes the 2^D corner tokens of the latent cell that contains it, in the per-axis [-1, 1]
    coordinates of ``_grid_coords``, with weights that sum to one. A k-nearest set on two regular lattices changes
    membership at the walls between tokens, so a decoder that sums over it jumps there and imprints the lattice on
    the output (measured 2026-09-07); here a token leaves the set only where its weight has fallen to zero, so the
    decoded field is continuous. Exact on affine fields.
    """
    lat = tuple(int(n) for n in latent_shape)
    phys = _grid_coords(phys_shape)
    n, dim = phys.shape
    lo, frac = [], []
    for a, nt in enumerate(lat):
        if nt == 1:
            lo.append(np.zeros(n, np.int64))
            frac.append(np.zeros(n))
            continue
        u = (phys[:, a] + 1.0) * 0.5 * (nt - 1)
        i0 = np.clip(np.floor(u), 0, nt - 2).astype(np.int64)
        lo.append(i0)
        frac.append(np.clip(u - i0, 0.0, 1.0))
    src, wts = [], []
    for corner in itertools.product((0, 1), repeat=dim):
        idx = [np.minimum(lo[a] + c, lat[a] - 1) for a, c in enumerate(corner)]
        w = np.ones(n)
        for a, c in enumerate(corner):
            w = w * (frac[a] if c else 1.0 - frac[a])
        src.append(np.ravel_multi_index(idx, lat))
        wts.append(w)
    edges = np.stack([np.stack(src, 1).reshape(-1), np.repeat(np.arange(n), 2 ** dim)]).astype(np.int64)
    return edges, np.stack(wts, 1).reshape(-1)


def _union_edges(*edge_sets: np.ndarray) -> np.ndarray:
    """Union of edge sets in the ``[2, E]`` layout, each (source, query) pair once (upstream's ``coalesce``)."""
    return np.unique(np.concatenate(edge_sets, axis=1), axis=1).astype(np.int64)


def _cached_edges(cache_dir: str, token: str, build) -> np.ndarray:
    """Memoise an edge build on disk. Keyed on every parameter that shapes it."""
    if not cache_dir:
        return build()
    key = hashlib.sha1(token.encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"gaot_edges_{key}.npy")
    if os.path.exists(path):
        try:
            return np.load(path)
        except (ValueError, OSError):  # truncated/corrupt cache: rebuild
            pass
    edges = build()
    os.makedirs(cache_dir, exist_ok=True)
    # np.save appends ".npy" unless the name already ends with it, so the temp
    # name must too or os.replace below has nothing to rename.
    tmp = f"{path[:-4]}.tmp{os.getpid()}.npy"
    np.save(tmp, edges)
    os.replace(tmp, path)
    return edges


# Output keys that are physically volume fractions and cannot lie outside [0, 1].
# Nothing in the pipeline enforces that: the final projection is an unbounded
# linear layer, and the alpha-masking `MaskNode`/`fill_where_mask_smooth_np` is a
# FILL, not a range constraint. Worse, the eval path masks every key using the
# *predicted* alpha as the mask (postprocessing.py:882-896), so an out-of-range
# alpha saturates its own sigmoid mask, survives, and lands in the global metric:
# measured 2026-07-30 on the material sweep, eval rmse_alpha 1.425 on a [0,1]
# quantity while in-training validation of the same model read 0.016.
_FRACTION_OUTPUT_KEYS = frozenset({"alpha", "fl", "VOF"})


def _clamp_straight_through(x: Tensor, lo: float | None, hi: float | None) -> Tensor:
    """Clamp to [lo, hi] in the forward pass, pass the gradient through unchanged.

    Same estimator as `_clamp01_straight_through`, but for quantities whose physical
    range is not [0, 1] -- notably `T`, which is unbounded in the model (its final
    projection is a plain linear layer) but has a physical ceiling.

    Why a bound on T is needed at all: `alpha` masking does NOT protect it.
    `fill_where_mask_smooth_np` is a sigmoid blend, `pred*s + d*(1-s)` with
    `s = sigmoid(k*(alpha-0.5))`, so wherever `alpha > 0.5` -- the metal, ~75% of the
    domain -- `s ~ 1` and the raw prediction passes through untouched by design, since
    that is exactly where T is meant to be predicted. `boiling_fill_value` is a fill
    TARGET, not a ceiling, and nothing bounds T from above anywhere in the pipeline.

    Two properties matter here, and both come from the straight-through form:
      1. A drifted cell still receives corrective gradient. With a plain clamp the
         derivative outside the range is exactly zero, so the weights that produced the
         bad value would get no signal and the clamp would hide the drift.
      2. The gradient is computed from the CLAMPED error, so one wild cell cannot
         dominate the batch gradient. An unclamped 2.2e6 K error would swamp every
         other cell's contribution -- which is itself a plausible way for the drift to
         become self-sustaining.
    """
    y = x.clamp(min=lo, max=hi)
    return x + (y - x).detach()


def _clamp01_straight_through(x: Tensor) -> Tensor:
    """Clamp to [0, 1] in the forward pass, pass the gradient through unchanged.

    A plain ``clamp`` has zero gradient outside the range, so a prediction that
    has drifted to alpha=50 would receive no signal to come back -- the clamp
    would hide the drift instead of correcting it. The straight-through form
    reports the physical value while still letting the loss pull the raw output
    back into range.
    """
    return x + (x.clamp(0.0, 1.0) - x).detach()


# --------------------------------------------------------------------------- #
# The Arch.
# --------------------------------------------------------------------------- #
class GAOTArch(Arch):
    """Geometry-Aware Operator Transformer on a fixed structured grid.

    MAGNO encoder (attentional graph neural operator + statistical geometry
    embedding) -> UViT processor on a fixed latent lattice -> MAGNO decoder.
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        dimension: int = 3,
        detach_keys: List[Key] = [],
        # --- latent lattice + processor ---
        latent_tokens: Sequence[int] = (48, 24, 16),
        patch_size: int = 2,
        lifting_channels: int = 32,
        hidden_size: int = 256,
        ffn_hidden_size: int = 1024,
        num_layers: int = 5,
        num_heads: int = 8,
        num_kv_heads: int = 8,
        atten_dropout: float = 0.0,
        positional_embedding: str = "absolute",
        use_long_range_skip: bool = True,
        # --- MAGNO encoder/decoder ---
        mlp_hidden_layers: Sequence[int] = (64, 64, 64),
        projection_channels: int = 256,
        neighbor_strategy: str = "knn",
        k_encoder: int = 4,
        k_decoder: int = 8,
        gno_radius: float = 0.15,
        max_neighbors: int = 0,
        decoder_stencil: str = "knn",
        decoder_kernel: str = "agno",
        neighbor_metric: Sequence[float] | None = None,
        use_attention: bool = True,
        attention_type: str = "cosine",
        transform_type: str = "linear",
        use_geoembed_encoder: bool = False,
        use_geoembed_decoder: bool = False,
        kernel_use_displacement: bool = False,
        use_pointwise_skip: bool = False,
        pointwise_skip_channels: int = 64,
        output_bounds: Mapping[str, Sequence[float | None]] | None = None,
        embedding_method: str = "statistical",
        clamp_fraction_outputs: bool = True,
        cache_dir: str = "outputs/gaot_graphs",
    ) -> None:
        super().__init__(
            input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys
        )
        self.dim = int(dimension)
        if self.dim != len(latent_tokens):
            raise ValueError(
                f"latent_tokens {tuple(latent_tokens)} has {len(latent_tokens)} axes "
                f"but dimension={self.dim}"
            )
        P = int(patch_size)
        bad = [n for n in latent_tokens if n % P]
        if bad:
            raise ValueError(
                f"every latent_tokens axis must be divisible by patch_size={P}; "
                f"got {tuple(latent_tokens)}"
            )
        if neighbor_strategy not in ("knn", "radius", "bidirectional"):
            raise ValueError(f"neighbor_strategy must be knn|radius|bidirectional, got {neighbor_strategy}")
        # "knn": the decoder reads the neighbour set of `neighbor_strategy` with attention (every model before
        # 2026-11-03). "trilinear": the 2^D corners of each node's latent cell with trilinear weights, which
        # replace the attention softmax (`_trilinear_edges`).
        if decoder_stencil not in ("knn", "trilinear"):
            raise ValueError(f"decoder_stencil must be knn|trilinear, got {decoder_stencil}")
        # "agno": the decoder's kernel integral (every model before 2026-11-03). "none": no kernel, the decoded
        # features are the trilinear interpolation of the token features (needs decoder_stencil="trilinear"), so
        # the decoder cannot itself put structure inside a latent cell.
        if decoder_kernel not in ("agno", "none"):
            raise ValueError(f"decoder_kernel must be agno|none, got {decoder_kernel}")
        if decoder_kernel == "none" and decoder_stencil != "trilinear":
            raise ValueError("decoder_kernel='none' needs decoder_stencil='trilinear'")
        # Per-axis scale factors applied to the coordinates of the kNN / radius SEARCHES only (the kernel and the
        # attention keep the per-axis [-1, 1] coordinates). The physical half-extents over the largest one make
        # the searches isotropic in physical distance, as upstream's single global normalisation is. None: off.
        if neighbor_metric is not None:
            neighbor_metric = tuple(float(v) for v in neighbor_metric)
            if len(neighbor_metric) != len(latent_tokens) or min(neighbor_metric) <= 0:
                raise ValueError(f"neighbor_metric needs {len(latent_tokens)} positive scales, got {neighbor_metric}")
        self.decoder_kernel = str(decoder_kernel)

        self.latent_shape = tuple(int(n) for n in latent_tokens)
        self.patch_size = P
        self.num_latent = int(np.prod(self.latent_shape))
        self.latent_channels = int(lifting_channels)
        self.num_inputs = sum(self.input_key_dict.values())
        self.num_outputs = sum(self.output_key_dict.values())
        self.clamp_fraction_outputs = bool(clamp_fraction_outputs)
        self._fraction_out_keys = tuple(
            k for k in self.output_key_dict if k in _FRACTION_OUTPUT_KEYS
        ) if self.clamp_fraction_outputs else ()
        # Explicit per-key physical bounds, e.g. {"T": [0.0, 1.25]} in NON-DIMENSIONAL
        # units (divide the physical ceiling by ref_scales["T"]). Off by default: the
        # ceiling is a physics choice that belongs to the config, not to the layer.
        self._output_bounds = {
            str(k): (
                None if v[0] is None else float(v[0]),
                None if len(v) < 2 or v[1] is None else float(v[1]),
            )
            for k, v in (output_bounds or {}).items()
            if str(k) in self.output_key_dict
        }
        in_channels = sum(self.input_key_dict.values())

        self._edge_cfg = dict(
            strategy=neighbor_strategy,
            k_encoder=int(k_encoder),
            k_decoder=int(k_decoder),
            radius=float(gno_radius),
            max_neighbors=int(max_neighbors),
            decoder_stencil=str(decoder_stencil),
            neighbor_metric=neighbor_metric,
            cache_dir=cache_dir,
        )
        self.grid_shape = None
        self._graph_built = False
        # Graphs keyed by physical grid shape. GAOT's premise is geometry
        # independence -- the latent lattice is fixed and the bipartite graph is
        # rebuilt per grid -- so a heterogeneous set is a matter of holding more
        # than one edge set, not a limitation of the method. Building once and
        # pinning `grid_shape` forever made a mixed-resolution set fail at the
        # output reshape (line ~623), which read as "GAOT cannot train on the
        # composed set" when it was this cache.
        self._graphs: dict[tuple, dict] = {}

        L = self.latent_channels
        # Kernel MLP sees [source_coord, query_coord] (+ features when nonlinear).
        # For the nonlinear transforms the kernel MLP is fed
        # [source_coord, query_coord, f_y]. f_y here is the LIFTED feature vector
        # (L channels), not the raw input, because `lifting` runs before the
        # encoder AGNO -- sizing this with in_channels is upstream's bug
        # (GAOT-3D magno.py:404) and makes transform_type="nonlinear" die on a
        # shape mismatch.
        # 3*dim when the kernel is given the displacement (source, query, query-source);
        # 2*dim otherwise. Without the displacement the decoder is close to a box filter
        # over each latent cell -- see integral_transform.use_displacement.
        self.kernel_use_displacement = bool(kernel_use_displacement)
        _coord_in = (3 if self.kernel_use_displacement else 2) * self.dim
        kernel_in = _coord_in + (L if transform_type.startswith("nonlinear") else 0)
        enc_mlp = [kernel_in, *[int(h) for h in mlp_hidden_layers], L]
        dec_mlp = [_coord_in, *[int(h) for h in mlp_hidden_layers], L]

        # --- Encoder: lift -> AGNO(phys -> latent) -> optional geometry embedding ---
        self.lifting = ChannelMLP(
            in_channels=in_channels, hidden_channels=L, out_channels=L, n_layers=1
        )
        self.enc_agno = IntegralTransform(
            channel_mlp_layers=enc_mlp,
            transform_type=transform_type,
            use_attn=use_attention,
            attention_type=attention_type,
            coord_dim=self.dim,
            use_displacement=self.kernel_use_displacement,
        )
        self.geo_enc = (
            GeometricEmbedding(
                input_dim=self.dim, output_dim=L, method=embedding_method
            )
            if use_geoembed_encoder
            else None
        )
        self.enc_recovery = (
            ChannelMLP(in_channels=2 * L, out_channels=L, n_layers=1)
            if use_geoembed_encoder
            else None
        )

        # --- Processor: UViT over patched latent lattice ---
        patch_volume = P**self.dim
        self.patch_linear = nn.Linear(patch_volume * L, patch_volume * L)
        self.positional_embedding_name = positional_embedding
        self.register_buffer(
            "patch_positions", self._patch_positions(), persistent=False
        )
        self.processor = Transformer(
            input_size=patch_volume * L,
            output_size=patch_volume * L,
            config=TransformerConfig(
                patch_size=P,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                positional_embedding=positional_embedding,
                use_long_range_skip=bool(use_long_range_skip),
                attn_config=AttentionConfig(
                    hidden_size=int(hidden_size),
                    num_heads=int(num_heads),
                    num_kv_heads=int(num_kv_heads),
                    atten_dropout=float(atten_dropout),
                    positional_embedding=positional_embedding,
                ),
                ffn_config=FFNConfig(hidden_size=int(ffn_hidden_size)),
            ),
        )

        # --- Decoder: AGNO(latent -> phys) -> optional geometry embedding -> project ---
        self.dec_agno = None if self.decoder_kernel == "none" else IntegralTransform(
            channel_mlp_layers=dec_mlp,
            transform_type="linear",
            use_attn=use_attention,
            attention_type=attention_type,
            coord_dim=self.dim,
            use_displacement=self.kernel_use_displacement,
        )
        self.geo_dec = (
            GeometricEmbedding(
                input_dim=self.dim, output_dim=L, method=embedding_method
            )
            if use_geoembed_decoder
            else None
        )
        self.dec_recovery = (
            ChannelMLP(in_channels=2 * L, out_channels=L, n_layers=1)
            if use_geoembed_decoder
            else None
        )
        self.projection = ChannelMLP(
            in_channels=L,
            hidden_channels=int(projection_channels),
            out_channels=self.num_outputs,
            n_layers=2,
        )

        # --- Full-resolution pointwise skip ---
        # Everything above routes through the latent lattice, so the only path from
        # input to output is a bottleneck: the MAGNO decoder interpolates from
        # ~12.8k-34k latent tokens back to 100k-600k physical nodes, which makes it a
        # smoothing operator. That matches the measured weakness -- GAOT ties RIGNO on
        # bulk T MAE but loses on interface/tail metrics (fl IoU 0.56-0.61 vs RIGNO
        # 0.68-0.76), and the kdec12 ablation made surface MAE 43% WORSE, i.e. more
        # decoder smoothing hurts exactly where it should.
        #
        # This adds a second, non-bottlenecked path: a 1x1 channel MLP on the raw input
        # channels at full resolution, added to the decoder output. It is what a U-Net's
        # skip connections provide and GAOT has none of. Cost is O(N x C) with ZERO
        # extra edges and no change to the latent lattice, so it does not touch the
        # speed/memory advantage that motivates GAOT in the first place.
        #
        # ChannelMLP is reused deliberately: it is 1d convolutions over channels and is
        # explicitly resolution-invariant, which is the whole requirement here.
        #
        # The final layer is ZERO-initialised, so at step 0 the skip contributes exactly
        # nothing and the model is identical to the baseline. The ablation therefore
        # measures what the skip adds, and cannot lose to an unlucky init.
        self.pointwise_skip = None
        if use_pointwise_skip:
            self.pointwise_skip = ChannelMLP(
                in_channels=self.num_inputs,
                hidden_channels=int(pointwise_skip_channels),
                out_channels=self.num_outputs,
                n_layers=2,
            )
            last_conv = [m for m in self.pointwise_skip.modules()
                         if isinstance(m, nn.Conv1d)][-1]
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)

    # ------------------------------------------------------------------ #
    # Graph
    # ------------------------------------------------------------------ #
    def _patch_positions(self) -> Tensor:
        counts = [n // self.patch_size for n in self.latent_shape]
        mesh = torch.meshgrid(
            *[torch.arange(c, dtype=torch.float32) for c in counts], indexing="ij"
        )
        return torch.stack(mesh, dim=-1).reshape(-1, self.dim)

    def _activate_graph(self, shape: tuple, device) -> None:
        """Point the live buffers at the graph for ``shape`` (already built)."""
        g = self._graphs[shape]
        self.register_buffer("phys_pos", g["phys_pos"].to(device), persistent=False)
        self.register_buffer("enc_edges", g["enc_edges"].to(device), persistent=False)
        self.register_buffer("dec_edges", g["dec_edges"].to(device), persistent=False)
        dw = g.get("dec_weights")
        self.register_buffer("dec_weights", None if dw is None else dw.to(device), persistent=False)
        self.n_phys = g["n_phys"]
        self.grid_shape = shape
        self._graph_built = True

    def _build_graph(self, spatial_shape: tuple, device) -> None:
        """Activate the phys<->latent graphs for this grid, building them once per shape.

        Re-entrant across shapes: a batch whose grid differs from the last one swaps the
        active edge buffers instead of reusing a graph built for a different geometry.
        """
        shape = tuple(int(s) for s in spatial_shape)
        if len(shape) != self.dim:
            raise ValueError(
                f"GAOT built for dimension={self.dim} but got grid {shape}"
            )
        if shape in self._graphs:
            self._activate_graph(shape, device)
            return
        self.grid_shape = shape
        cfg = self._edge_cfg
        phys = _grid_coords(shape)
        latent = _grid_coords(self.latent_shape)
        if cfg["neighbor_metric"] is None:
            phys_s, latent_s, mtag = phys, latent, ""
        else:
            sc = np.asarray(cfg["neighbor_metric"], dtype=np.float64)
            phys_s, latent_s = phys * sc, latent * sc
            mtag = "_metric" + "x".join(f"{v:g}" for v in sc)
        base = f"{shape}_{self.latent_shape}_{cfg['strategy']}{mtag}"

        if cfg["strategy"] == "knn":
            # Encoder is SOURCE-centric (every physical node pushes to its k
            # nearest latent tokens) and the decoder QUERY-centric (every
            # physical node pulls from its k nearest latent tokens). Both
            # directions bound the physical side, which is the one that must be
            # covered: all of the input is read, all of the output is produced.
            enc = _cached_edges(
                cfg["cache_dir"], f"enc_{base}_sc_k{cfg['k_encoder']}",
                lambda: _knn_edges_source_centric(phys_s, latent_s, cfg["k_encoder"]),
            )
            dec = _cached_edges(
                cfg["cache_dir"], f"dec_{base}_k{cfg['k_decoder']}",
                lambda: _knn_edges(latent_s, phys_s, cfg["k_decoder"]),
            )
        elif cfg["strategy"] == "radius":
            r, mn = cfg["radius"], cfg["max_neighbors"]
            enc = _cached_edges(
                cfg["cache_dir"], f"enc_{base}_r{r}_m{mn}",
                lambda: _radius_edges(phys_s, latent_s, r, mn),
            )
            dec = _cached_edges(
                cfg["cache_dir"], f"dec_{base}_r{r}_m{mn}",
                lambda: _radius_edges(latent_s, phys_s, r, mn),
            )
        else:
            # "bidirectional", upstream GAOT-3D's reference strategy (magno.py `_get_encoder_strategy`,
            # `_get_decoder_strategy`): the kNN edges of the "knn" branch above united with the radius edges of the
            # "radius" branch, each (source, query) pair once.
            r, mn = cfg["radius"], cfg["max_neighbors"]
            ke, kd = cfg["k_encoder"], cfg["k_decoder"]
            enc = _cached_edges(
                cfg["cache_dir"], f"enc_{base}_sc_k{ke}_r{r}_m{mn}",
                lambda: _union_edges(_knn_edges_source_centric(phys_s, latent_s, ke), _radius_edges(phys_s, latent_s, r, mn)),
            )
            dec = _cached_edges(
                cfg["cache_dir"], f"dec_{base}_k{kd}_r{r}_m{mn}",
                lambda: _union_edges(_knn_edges(latent_s, phys_s, kd), _radius_edges(latent_s, phys_s, r, mn)),
            )
        dec_w = None
        if cfg["decoder_stencil"] == "trilinear":
            dec, dec_w = _trilinear_edges(self.latent_shape, shape)

        f32 = torch.float32
        self.register_buffer("latent_pos", torch.as_tensor(latent, dtype=f32, device=device), persistent=False)
        self._graphs[shape] = {
            "phys_pos": torch.as_tensor(phys, dtype=f32, device=device),
            "enc_edges": torch.as_tensor(enc, dtype=torch.long, device=device),
            "dec_edges": torch.as_tensor(dec, dtype=torch.long, device=device),
            "dec_weights": None if dec_w is None else torch.as_tensor(dec_w, dtype=f32, device=device),
            "n_phys": int(phys.shape[0]),
        }
        self._activate_graph(shape, device)
        print(
            f"GAOT graph: grid {self.grid_shape} ({self.n_phys} nodes) <-> latent "
            f"{self.latent_shape} ({self.num_latent} tokens); "
            f"encoder edges {enc.shape[1]}, decoder edges {dec.shape[1]}"
        )

    # ------------------------------------------------------------------ #
    # Encode / process / decode
    # ------------------------------------------------------------------ #
    def _encode(self, f: Tensor) -> Tensor:
        """(B, N, L) physical features -> (B, M, L) latent tokens."""
        # ponytail: the integral transform takes one sample at a time (upstream
        # flattens the batch into the node dim instead). The shipped configs run
        # batch_size.grid=1, so this loop runs once; batch the edge set with an
        # index offset if a larger batch ever becomes the bottleneck.
        z = torch.stack(
            [
                self.enc_agno(
                    y_pos=self.phys_pos, x_pos=self.latent_pos,
                    edge_index=self.enc_edges, f_y=f[b],
                )
                for b in range(f.shape[0])
            ],
            dim=0,
        )
        if self.geo_enc is not None:
            # ponytail: the geometry statistics are a pure function of the (fixed)
            # grid, so this is constant across samples and steps apart from the
            # MLP weights; recomputed each forward for fidelity to upstream.
            # Cache the pre-MLP statistics if profiling ever shows it matters.
            g = self.geo_enc(
                source_pos=self.phys_pos, query_pos=self.latent_pos,
                edge_index=self.enc_edges,
            )
            z = torch.cat([z, g.unsqueeze(0).expand(z.shape[0], -1, -1)], dim=-1)
            z = self.enc_recovery(z.permute(0, 2, 1)).permute(0, 2, 1)
        return z

    def _process(self, z: Tensor) -> Tensor:
        """(B, M, L) -> (B, M, L) through the patched UViT."""
        B, M, C = z.shape
        P = self.patch_size
        counts = [n // P for n in self.latent_shape]
        D, H, W = self.latent_shape
        nD, nH, nW = counts

        z = z.view(B, D, H, W, C).view(B, nD, P, nH, P, nW, P, C)
        z = z.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        z = z.view(B, nD * nH * nW, P * P * P * C)

        z = self.patch_linear(z)
        pos = self.patch_positions.to(z.device)
        relative_positions = None
        if self.positional_embedding_name == "absolute":
            z = z + self._absolute_embeddings(pos, P**self.dim * C).to(z.dtype)
        elif self.positional_embedding_name == "rope":
            relative_positions = pos
        z = self.processor(z, condition=None, relative_positions=relative_positions)

        z = z.view(B, nD, nH, nW, P, P, P, C)
        z = z.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return z.view(B, M, C)

    @staticmethod
    def _absolute_embeddings(positions: Tensor, embed_dim: int) -> Tensor:
        """Sinusoidal patch-position embedding (upstream ``gaot_3d.py``)."""
        freq = 1.0 / 10000 ** (
            2 * torch.arange(0, embed_dim // 2, dtype=torch.float32, device=positions.device)
            / embed_dim
        )
        angles = positions[:, :, None] * freq[None, None, :]
        pe = torch.zeros(positions.shape[0], embed_dim, device=positions.device)
        pe[:, 0::2] = torch.sin(angles).sum(dim=1)
        pe[:, 1::2] = torch.cos(angles).sum(dim=1)
        return pe

    def _decode(self, z: Tensor) -> Tensor:
        """(B, M, L) latent tokens -> (B, N, L) physical features."""
        if self.dec_agno is None:
            # decoder_kernel="none": trilinear interpolation of the token features (partition-of-unity weights)
            src, dst = self.dec_edges[0], self.dec_edges[1]
            w = self.dec_weights.to(z.dtype).unsqueeze(-1)
            y = torch.stack([torch.zeros(self.n_phys, z.shape[-1], device=z.device, dtype=z.dtype)
                             .index_add_(0, dst, z[b][src] * w) for b in range(z.shape[0])], dim=0)
        else:
            y = torch.stack(
                    [
                        self.dec_agno(
                            y_pos=self.latent_pos, x_pos=self.phys_pos,
                            edge_index=self.dec_edges, f_y=z[b], weights=self.dec_weights,
                        )
                        for b in range(z.shape[0])
                    ],
                    dim=0,
                )
        if self.geo_dec is not None:
            g = self.geo_dec(
                source_pos=self.latent_pos, query_pos=self.phys_pos,
                edge_index=self.dec_edges,
            )
            y = torch.cat([y, g.unsqueeze(0).expand(y.shape[0], -1, -1)], dim=-1)
            y = self.dec_recovery(y.permute(0, 2, 1)).permute(0, 2, 1)
        return y

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.prepare_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=1,
            input_scales=self.input_scales,
        )  # (B, C_in, nx, ny, nz)
        B, C = x.shape[0], x.shape[1]
        # Every batch, not only the first: a heterogeneous set changes grid between
        # batches and the cached graph must follow. `_build_graph` is a no-op re-activation
        # when the shape is unchanged, so the steady-state cost is a tuple compare.
        if not self._graph_built or tuple(int(v) for v in x.shape[2:]) != self.grid_shape:
            self._build_graph(x.shape[2:], x.device)

        f = self.lifting(x.reshape(B, C, -1)).permute(0, 2, 1).contiguous()  # (B, N, L)
        z = self._encode(f)                                                  # (B, M, L)
        z = self._process(z)                                                 # (B, M, L)
        y = self._decode(z)                                                  # (B, N, L)
        y = self.projection(y.permute(0, 2, 1))                              # (B, C_out, N)
        if self.pointwise_skip is not None:
            # Raw inputs at full resolution, never routed through the lattice.
            y = y + self.pointwise_skip(x.reshape(B, C, -1))                  # (B, C_out, N)
        y = y.reshape(B, self.num_outputs, *self.grid_shape)
        out = self.prepare_output(
            y, self.output_key_dict, dim=1, output_scales=self.output_scales
        )
        for key in self._fraction_out_keys:
            out[key] = _clamp01_straight_through(out[key])
        for key, (lo, hi) in self._output_bounds.items():
            out[key] = _clamp_straight_through(out[key], lo, hi)
        return out
