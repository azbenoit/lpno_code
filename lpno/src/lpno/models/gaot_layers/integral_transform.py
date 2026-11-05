# Vendored from GAOT-3D (https://github.com/Shizheng-Wen/GAOT-3D), MIT License: see LICENSE in this folder.
# src/model/layers/integral_transform.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple
import importlib

from .mlp import LinearChannelMLP
try:
    import torch_scatter as _torch_scatter
except ImportError:
    _torch_scatter = None

from ._scatter_backend import resolve_scatter

scatter, SCATTER_BACKEND = resolve_scatter(
    _torch_scatter, cuda_available=torch.cuda.is_available()
)
HAS_TORCH_SCATTER = SCATTER_BACKEND == "torch_scatter"
print(f"GAOT aggregation backend: {SCATTER_BACKEND}", flush=True)

"""
Reference: https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/integral_transform.py
"""

class IntegralTransform(nn.Module):
    def __init__(
        self,
        channel_mlp=None,
        channel_mlp_layers=None,
        channel_mlp_non_linearity=F.gelu,
        transform_type="linear",
        use_attn=None,
        coord_dim=None,
        attention_type='cosine',
        use_displacement=False,
    ):
        super().__init__()
        # parameters for attentional integral transform
        self.transform_type = transform_type
        self.use_attn = use_attn
        self.coord_dim = coord_dim  
        self.use_displacement = bool(use_displacement)
        self.attention_type = attention_type
        # Init MLP based on channel_mlp or channel_mlp_layers
        if channel_mlp is None:
             if channel_mlp_layers is None: raise ValueError("Need channel_mlp or layers")
             self.channel_mlp = LinearChannelMLP(layers=channel_mlp_layers, non_linearity=channel_mlp_non_linearity)
        else:
            self.channel_mlp = channel_mlp
        
        if self.use_attn:
            if coord_dim is None:
                raise ValueError("coord_dim must be specified when use_attn is True")

            if self.attention_type == 'dot_product':
                attention_dim = 64 
                self.query_proj = nn.Linear(self.coord_dim, attention_dim)
                self.key_proj = nn.Linear(self.coord_dim, attention_dim)
                self.scaling_factor = 1.0 / (attention_dim ** 0.5)
            elif self.attention_type == 'cosine':
                pass
            else:
                raise ValueError(f"Invalid attention_type: {self.attention_type}. Must be 'cosine' or 'dot_product'.")

    def _segment_softmax_pyg(self, scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
        """Applies softmax per segment based on index using torch_scatter."""
        scores_max = scatter(scores, index, dim=0, dim_size=dim_size, reduce="max")
        scores_max_expanded = scores_max[index]
        scores = scores - scores_max_expanded # Stable softmax
        exp_scores = torch.exp(scores)
        exp_sum = scatter(exp_scores, index, dim=0, dim_size=dim_size, reduce="sum")
        exp_sum_clamped = torch.clamp(exp_sum, min=torch.finfo(exp_sum.dtype).tiny) # Clamp sum to avoid division by zero
        exp_sum_expanded = exp_sum_clamped[index]
        attention_weights = exp_scores / exp_sum_expanded
        return attention_weights
    
    def forward(self, 
                y_pos: torch.Tensor, 
                x_pos: torch.Tensor, 
                edge_index: torch.Tensor, 
                f_y: Optional[torch.Tensor] = None, 
                weights: Optional[torch.Tensor] = None, 
                batch_y=None, 
                batch_x=None):
        """
        Compute kernel integral transform using PyG edge_index.

        Args:
            y_pos (Tensor): Source node coordinates [N_y, D] (or [TotalNodes_y, D] if batched).
            x_pos (Tensor): Query node coordinates [N_x, D] (or [TotalNodes_x, D] if batched).
            edge_index (Tensor): Edge index [2, NumEdges] where edge_index[1] indexes into x_pos (query)
                                 and edge_index[0] indexes into y_pos (source).
            f_y (Tensor, optional): Source node features [N_y, C_in] (or [TotalNodes_y, C_in] if batched).
            weights (Tensor, optional): Edge weights [NumEdges,]. Not typically volume weights here.
            batch_y (Tensor, optional): Batch assignment for y_pos nodes [TotalNodes_y].
            batch_x (Tensor, optional): Batch assignment for x_pos nodes [TotalNodes_x]. Required if using batching.
        """
        device = x_pos.device
        num_query_nodes = x_pos.shape[0]

        sampled_edge_index = edge_index.to(device)

        num_sampled_edges = sampled_edge_index.shape[1]
        if num_sampled_edges == 0:
            output_channels = self.channel_mlp.fcs[-1].out_features
            output_shape = [num_query_nodes, output_channels] 
            if batch_x is not None and f_y is not None and f_y.ndim == 3: 
                pass 
            return torch.zeros(output_shape, device=device, dtype=self.channel_mlp.fcs[-1].weight.dtype)

        query_idx = sampled_edge_index[1].long()
        source_idx = sampled_edge_index[0].long()

        rep_features_pos = y_pos[source_idx]    # Source node coords [NumSampledEdges, D]
        self_features_pos = x_pos[query_idx]    # Query node coords [NumSampledEdges, D]

        in_features = None
        if f_y is not None:
             # Assume f_y is [TotalNodes_y, C_in]
             in_features = f_y[source_idx]      # Source node features [NumSampledEdges, C_in]


        # --- Attention Logic ---
        attention_weights = None
        # Supplied edge weights (lpno addition, 2026-11-03; upstream accepts `weights` and ignores it) replace the
        # softmax: GAOTArch's trilinear decoder passes partition-of-unity weights here.
        if self.use_attn and weights is None:
            query_coords = self_features_pos[:, :self.coord_dim]
            key_coords = rep_features_pos[:, :self.coord_dim]
            if self.attention_type == 'dot_product':
                 query = self.query_proj(query_coords)
                 key = self.key_proj(key_coords)
                 attention_scores = torch.sum(query * key, dim=-1) * self.scaling_factor
            elif self.attention_type == 'cosine':
                 query_norm = F.normalize(query_coords, p=2, dim=-1)
                 key_norm = F.normalize(key_coords, p=2, dim=-1)
                 attention_scores = torch.sum(query_norm * key_norm, dim=-1)
            else:
                 raise ValueError(f"Invalid attention_type: {self.attention_type}")
            attention_weights = self._segment_softmax_pyg(attention_scores, query_idx, num_query_nodes)
        # --- End Attention Logic ---


        # Create aggregated features for MLP input
        if self.use_displacement:
            # Give the kernel the DISPLACEMENT, not just the two absolute positions.
            #
            # Without it the decoder is close to a box filter. On a regular lattice with
            # k_decoder=8 every physical node inside a latent cell selects exactly the 8
            # corners of that cell, and the cosine attention above scores ABSOLUTE
            # normalised coordinates, which are nearly identical across a small cell -- so
            # the weights barely discriminate and the kernel is never told where inside
            # the cell the query sits. Rendered cross-sections of model 22022 show the
            # consequence directly: the predicted temperature field is piecewise-constant
            # in ~3x3x2 tiles with a regular +/-250 K checkerboard error, which displaces
            # the 1898 K melt isotherm and under-predicts pool depth.
            #
            # query - source is the standard GNO kernel argument and is what lets the
            # output vary continuously within a cell.
            disp = self_features_pos[:, : self.coord_dim] - rep_features_pos[:, : self.coord_dim]
            agg_features = torch.cat([rep_features_pos, self_features_pos, disp], dim=-1)
        else:
            agg_features = torch.cat([rep_features_pos, self_features_pos], dim=-1) # [NumSampledEdges, 2*D]

        if in_features is not None and (
            self.transform_type == "nonlinear_kernelonly"
            or self.transform_type == "nonlinear"
        ):
            agg_features = torch.cat([agg_features, in_features], dim=-1)

        rep_features_transformed = self.channel_mlp(agg_features) # [NumSampledEdges, C_mlp_out]

        if in_features is not None and self.transform_type != "nonlinear_kernelonly":
            rep_features_transformed = rep_features_transformed * in_features

        if attention_weights is not None:
            rep_features_transformed = rep_features_transformed * attention_weights.unsqueeze(-1)
        if weights is not None:
            rep_features_transformed = rep_features_transformed * weights.to(
                device=rep_features_transformed.device, dtype=rep_features_transformed.dtype).unsqueeze(-1)

        # Apply edge weights if provided (e.g., distances, kernel values not from MLP)
        reduction = "sum" if (weights is not None or (self.use_attn and attention_weights is not None)) else "mean"
       
        out_features = scatter(
            src=rep_features_transformed,
            index=query_idx.long(),
            dim=0,                  
            dim_size=num_query_nodes,
            reduce=reduction
        )
        if HAS_TORCH_SCATTER and (reduction == "max" or reduction=="min"):
             out_features = out_features[0] 

        return out_features



