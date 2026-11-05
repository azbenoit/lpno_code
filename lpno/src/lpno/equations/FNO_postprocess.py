import numpy as np
import torch
import torch.nn.functional as F
from physicsnemo.sym.key import Key
from ..custom_utils.utils import fill_where_mask_smooth_torch, smooth_liquid_fraction_torch

class fl_Node(torch.nn.Module):
    """
    PhysicsNeMo Node that computes liquid fraction from temperature field T.

    Inputs:
        T: Temperature field (B, 1, Nz, Ny, Nx)
        T_s: Solidus temperature (scalar)
        T_l: Liquidus temperature (scalar)
    Outputs:
        fl: Liquid fraction field (B, 1, Nz, Ny, Nx)
    """

    def __init__(self, T_s, T_l, ts_scale=None, tl_scale=None):
        super().__init__()
        self.T_s = T_s
        self.T_l = T_l
        # Ref-scale ratios that put mat_Ts/mat_Tl back on T's scale. The channels arrive
        # divided by their OWN ref_scale (mat_Ts: 1800), while T is divided by ref_scales
        # ["T"] (3000), so a raw comparison would be wrong by 1800/3000.
        self.ts_scale = ts_scale
        self.tl_scale = tl_scale

    def forward(self, outvar, ts_scale=None, tl_scale=None):
        """Per-case melt window when `mat_Ts`/`mat_Tl` ride in the batch.

        This node sits in EVERY run's graph, so it is a third place the melt window is
        chosen -- alongside the target in `preprocessing_datasets` and the eval metric in
        `postprocessing.derive_fl_masked`. Leaving it on a fixed Ti64 pair while those two
        went per-case would have the graph's predicted fl answer a different question from
        the truth it is scored against, on every non-Ti64 material.
        """
        T = outvar["T"]
        ts, tl = outvar.get("mat_Ts"), outvar.get("mat_Tl")
        if ts is None or tl is None:
            return {"fl": smooth_liquid_fraction_torch(T, self.T_s, self.T_l)}
        s_scale = self.ts_scale if ts_scale is None else ts_scale
        l_scale = self.tl_scale if tl_scale is None else tl_scale
        if s_scale is not None:
            ts = ts * float(s_scale)
        if l_scale is not None:
            tl = tl * float(l_scale)
        return {"fl": smooth_liquid_fraction_torch(T, ts, tl)}
    
class MaskNode(torch.nn.Module):
    """
    PhysicsNeMo Node that masks fields based on a specified mask key.
    Inputs:
        mask_key: Key in outvar to use as the mask (e.g., "alpha")
        threshold: Threshold value to determine solid/liquid
        d: Value to fill where mask condition is met
        channel_axis: Axis of channels in input tensors
        inplace: Whether to modify input tensors in place
    Outputs:
        masked fields in outvar dictionary
    """
    def __init__(self, keys, threshold=0.5, d=0.0, k=20.0, mask_str_key ="alpha", t_fill=None,
                 t_fill_key=None, t_fill_scale=1.0):
        super().__init__()
        self.keys = keys
        self.threshold = threshold
        self.d = d
        self.k = k
        self.mask_str_key = mask_str_key
        # Per-sample fill: `t_fill_key` names an input the graph hands this node (the
        # `mat_Tv` channel), and `t_fill_scale` is `ref_scales['mat_Tv'] / ref_scales['T']`
        # -- the same formula as `utils.boiling_fill_from_channel`, so training and eval
        # fill identically. `t_fill` stays the scalar fallback when the key is absent.
        self.t_fill_key = t_fill_key
        self.t_fill_scale = float(t_fill_scale)
        # T's off-metal fill is the boiling point over the configured T
        # reference, which only the caller knows. A literal here agreed with the
        # training target for exactly one value of ref_scales.T.
        if t_fill is None and any(self._normalize_key(key) == "T" for key in keys):
            raise ValueError(
                "MaskNode masks 'T' but got no t_fill; pass "
                "boiling_fill_value(material_params, ref_scales)."
            )
        self.t_fill = t_fill

    @staticmethod
    def _normalize_key(key):
        if isinstance(key, Key):
            return key.name
        return key

    def forward(self, outvar):
        mask_key = self._normalize_key(self.mask_str_key)
        mask = outvar[mask_key]
        for key in self.keys:
            key_name = self._normalize_key(key)
            if key_name == mask_key:
                # Never gate the mask by itself: `alpha_out = gate(alpha)*alpha`
                # is circular and self-squashing — below threshold the product
                # drives alpha down, which lowers the gate, which lowers alpha.
                # With the gate saturating that became an absorbing state a run
                # could not leave (latent-128 RIGNO frozen from step 500 at two
                # learning rates, alpha collapsed to exactly 0, 2026-07-24).
                # It is also redundant: for a converged model alpha is ~0 or ~1,
                # where gate(alpha)*alpha == alpha, so this is a no-op in the
                # healthy regime and only removes the pathological one.
                continue
            data = outvar[key_name]
            if key_name == "T":
                # For temperature, fill with boiling temp -- per sample when the channel
                # was declared and delivered, else the scalar.
                fill = self.t_fill
                if self.t_fill_key is not None and self.t_fill_key in outvar:
                    fill = outvar[self.t_fill_key] * self.t_fill_scale
                outvar[key_name] = fill_where_mask_smooth_torch(data, mask, threshold=self.threshold, d=fill, channel_axis=1, k=self.k, inplace=False)
            else:
                # For other fields, fill with d
                outvar[key_name] = fill_where_mask_smooth_torch(data, mask, threshold=self.threshold, d=self.d, channel_axis=1, k=self.k, inplace=False)
            
        return outvar

#: The trainer's authoritative training step, published once per optimizer step
#: by the constraint-loss patch in `models/pipeline/model_domain.py`. `None`
#: means nobody is publishing (standalone use), and WarmupNode falls back to
#: counting its own calls.
_TRAIN_STEP: dict[str, int | None] = {"value": None}


def set_train_step(step: int) -> None:
    """Publish the trainer's step so WarmupNode can gate on it.

    WarmupNode cannot count its own forwards to measure training progress: the
    node is invoked several times per optimizer step (and, before the
    `self.training` guard, during validation too), so a call counter runs ahead
    of the real step by an unknown factor. Measured 2026-07-28: with
    `mask_warmup_steps=4000` the mask engaged around training step 2000 and
    collapsed the outputs to the constant fill.
    """
    _TRAIN_STEP["value"] = int(step)


class WarmupNode(torch.nn.Module):
    def __init__(self, inner, start_step):
        super().__init__()
        self.inner = inner
        self.start_step = start_step
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    def forward(self, outvar):
        # Count TRAINING steps only. In-training validation runs this node too
        # (many batches per validation), so counting every forward expired the
        # warmup far earlier than configured: with mask_warmup_steps=4000 the
        # mask engaged before training step 2000 and collapsed the outputs to the
        # constant fill (val mae_T 1076.4 K -> 2093.1 K, l2_relative_error_fl
        # 1.0009 = predicting zeros). Branching on `self.training` is a Python
        # bool, not a step-buffer read, so it costs no CPU sync and train/eval
        # capture separate graphs anyway (capture safety is about the CPU-sync read).
        #
        # Prefer the trainer's published step: the node runs several times per
        # optimizer step, so counting calls overshoots by an unknown factor.
        # `fill_` writes a Python int into the existing buffer -- device-side,
        # no sync, and the graph stays static. Fall back to counting when
        # nothing publishes a step (standalone use outside the sym trainer).
        if self.training:
            published = _TRAIN_STEP["value"]
            if published is None:
                self.step += 1
            else:
                self.step.fill_(published)
        # Capture-safe warmup gate: no CPU-sync read of the step buffer
        # and no Python branch, so the graph is static under cuda_graphs capture.
        # Always run the inner node and blend with the pre-inner values; the
        # shipped inners (MaskNode) only overwrite existing keys, so passing the
        # originals through equals the old pre-warmup "return {}".
        gate = self.step >= self.start_step
        pre = dict(outvar)
        inner_out = self.inner(outvar)  # inner should return a dict keyed like `outputs`
        return {
            key: torch.where(gate, value, pre[key]) if key in pre else value
            for key, value in inner_out.items()
        }


class SpatialAlignNode(torch.nn.Module):
    """Align prediction tensors to the reference grid shape by symmetric crop/pad."""

    def __init__(self, keys, ref_key):
        super().__init__()
        self.keys = [self._normalize_key(key) for key in keys]
        self.ref_key = self._normalize_key(ref_key)

    @staticmethod
    def _normalize_key(key):
        if isinstance(key, Key):
            return key.name
        return key

    @staticmethod
    def _center_crop_or_pad(data: torch.Tensor, target_spatial_shape: tuple[int, ...]) -> torch.Tensor:
        aligned = data
        current_spatial_shape = aligned.shape[-len(target_spatial_shape) :]

        if current_spatial_shape == target_spatial_shape:
            return aligned

        slices = [slice(None)] * aligned.ndim
        for dim_idx, (current_dim, target_dim) in enumerate(zip(current_spatial_shape, target_spatial_shape, strict=True)):
            axis = aligned.ndim - len(target_spatial_shape) + dim_idx
            if current_dim > target_dim:
                crop_start = (current_dim - target_dim) // 2
                slices[axis] = slice(crop_start, crop_start + target_dim)
        aligned = aligned[tuple(slices)]

        cropped_spatial_shape = aligned.shape[-len(target_spatial_shape) :]
        if cropped_spatial_shape == target_spatial_shape:
            return aligned

        pad_widths = []
        for current_dim, target_dim in zip(reversed(cropped_spatial_shape), reversed(target_spatial_shape), strict=True):
            if current_dim > target_dim:
                raise ValueError(
                    f"Spatial alignment failed: cropped dim {current_dim} still exceeds target {target_dim}."
                )
            pad_total = target_dim - current_dim
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            pad_widths.extend([pad_before, pad_after])

        if any(pad_widths):
            aligned = F.pad(aligned, pad_widths, mode="replicate")
        return aligned

    def forward(self, outvar):
        reference = outvar[self.ref_key]
        target_spatial_shape = tuple(reference.shape[2:])
        return {
            key: self._center_crop_or_pad(outvar[key], target_spatial_shape)
            for key in self.keys
        }


class dirichlet_boundary_Node(torch.nn.Module):
    """
    PhysicsNeMo Node that applies Dirichlet boundary conditions to specified fields.
    Inputs:
        keys: List of keys in outvar to apply BCs to
        boundary_slices: Dictionary mapping keys to boundary slices (tuples)
        boundary_values: Dictionary mapping keys to boundary values (scalars or tensors)
    Outputs:
        outvar with Dirichlet BCs applied
    """
    def __init__(self, keys, boundary_slices, boundary_values):
        super().__init__()
        self.keys = keys
        self.boundary_slices = boundary_slices
        self.boundary_values = boundary_values

    def forward(self, outvar):
        for key in self.keys:
            b_slice = self.boundary_slices[key]
            b_value = self.boundary_values[key]
            outvar[key][b_slice] = b_value
        return outvar
