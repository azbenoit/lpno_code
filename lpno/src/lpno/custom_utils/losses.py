"""Custom physicsnemo loss functions."""

from typing import Dict, Optional

import torch
from torch import Tensor

from physicsnemo.sym.loss import Loss

from .utils import smooth_liquid_fraction_torch


def topk_tail_loss(pred, true, mask, k: int = 256):
    """Match the mean of the k hottest MASKED cells between prediction and truth.

    The measured failure (2026-08-04): melt is ~1% of the substrate, so a model
    can clip its temperature tail 267 K below the melt threshold, hold T RMSE at
    84 K, and produce zero melt (model 29125, peak 1631 K vs an 1898 K
    threshold). L2 is indifferent to the tail; every thresholded quantity then
    collapses.

    Why a tail statistic and not a mask-space / boundary loss: `fl` is derived
    from T through a tanh centred on 1898 K, and in float32 `tanh` saturates to
    exactly -1.0 below 1717.8 K, so `d(fl)/dT` is BIT-ZERO for the very model
    that needs rescuing. There is no tanh in this path -- `d/dT` is `1/k`,
    scale-free -- so it produces gradient exactly where the deficit is.

    Why this is not `melt_weight` (a recorded regression, 2026-07-24): that
    reweighted the pointwise L2 by `1 + w*fl_true`, so it was zero wherever the
    truth had no melt and only amplified an error L2 already saw. This is a
    statistic mismatch on the tail, active regardless.

    ``mask`` must select substrate METAL: with alpha masking the gas carries a
    ~3560 K boiling fill that would otherwise own the entire top-k.
    """

    import torch as _t

    if mask is None or not bool(mask.any()):
        return _t.zeros((), dtype=pred.dtype, device=pred.device)
    p = pred[mask].reshape(-1)
    t = true[mask].reshape(-1)
    n = int(min(k, p.numel()))
    if n <= 0:
        return _t.zeros((), dtype=pred.dtype, device=pred.device)
    # k > 1 so the gradient is spread over the tail rather than one voxel.
    return (p.topk(n).values.mean() - t.topk(n).values.mean()).pow(2)


class GridGradientLoss(Loss):
    """Pointwise L2 loss plus a spatial-gradient (H1 seminorm) mismatch term.

    The gradient term penalizes the finite-difference mismatch between
    prediction and target along the trailing ``spatial_dims`` dims. It targets
    patchy decode artifacts (spurious gradients at patch seams) and missing
    conduction halos (absent gradient structure) without blurring true fronts,
    because it matches gradients rather than penalizing their magnitude.

    Drop-in for ``SupervisedGridConstraint(loss=...)``; expects grid tensors
    shaped ``(batch, channels, x, y, z)``. With ``gradient_weight=0`` and
    ``edge_downweight=0`` it reproduces physicsnemo's ``PointwiseLossNorm(ord=2)``.

    ``edge_downweight`` (beta) applies a per-voxel spatial weight
    ``w = 1 / (1 + beta * |grad T_true| / mean|grad T_true|)`` to BOTH the
    pointwise and the gradient terms: it down-weights sharp-gradient voxels (the
    keyhole/melt-pool edge the RIGNO base already fits well) and leaves the smooth
    global wake at full weight, so a refinement head spends its capacity — and
    stops ringing — on the global field instead of over-correcting the edge.
    beta=0 disables it (uniform weight, identical to before).

    Two melt-focus terms make the loss care about the *pool*, not just the
    voxels, and both are gated by the same step warmup:

    - ``melt_dice_weight`` adds a soft-Dice region-overlap term. The melt pool
      comes from ``T`` through the same tanh liquid fraction the pipeline uses
      elsewhere, and the recoil cavity from ``sigmoid((0.5 - alpha)/tau)``, so
      both masks are differentiable -- unlike the thresholded depth/volume
      scalars the metrics report, whose gradient is zero almost everywhere.
    - ``melt_weight`` upweights every voxel inside the *ground-truth* melt
      region by ``1 + melt_weight * fl_true``. The weight is taken from true
      ``T`` once and applied to every output key, so an ``alpha`` error inside
      the pool costs more than one in the far field. It is the mirror image of
      ``edge_downweight``: attention onto the pool rather than off its edge.

    ``warmup_steps`` keeps both terms at exactly zero until the model already
    predicts a rough pool (a region term has nothing to pull on before that),
    then ``ramp_steps`` ramps the weight linearly to full. Both default to
    zero, and with both melt weights at zero this class behaves exactly as it
    did before they existed.

    ``melt_dice_weight`` is in units of the *summed* pointwise loss (the
    physicsnemo convention), so its scale depends on the grid size: pick it
    against the data-loss magnitude the run actually reports at the warmup
    step, not from an intuition about a [0, 1] Dice value.
    """

    #: Output keys whose soft region mask is defined, and how.
    MELT_KEY = "T"
    CAVITY_KEY = "alpha"
    #: Velocity outputs: scored only inside the melt pool (see forward).
    VELOCITY_KEYS = ("u", "v", "w")

    def __init__(
        self,
        gradient_weight: float = 1.0,
        spatial_dims: int = 3,
        edge_downweight: float = 0.0,
        melt_dice_weight: float = 0.0,
        melt_weight: float = 0.0,
        tail_weight: float = 0.0,
        tail_k: int = 256,
        melt_tau: float = 0.05,
        warmup_steps: int = 0,
        ramp_steps: int = 0,
        t_solidus: Optional[float] = None,
        t_liquidus: Optional[float] = None,
        vel_warmup_steps: int = 0,
        vel_ramp_steps: int = 0,
        vel_tau: float = 0.05,
    ):
        super().__init__()
        self.gradient_weight = float(gradient_weight)
        self.spatial_dims = int(spatial_dims)
        self.edge_downweight = float(edge_downweight)
        self.melt_dice_weight = float(melt_dice_weight)
        self.melt_weight = float(melt_weight)
        self.tail_weight = float(tail_weight)
        self.tail_k = int(tail_k)
        self._tail_warned = False
        self.melt_tau = float(melt_tau)
        self.warmup_steps = int(warmup_steps)
        self.ramp_steps = int(ramp_steps)
        self.t_solidus = None if t_solidus is None else float(t_solidus)
        self.t_liquidus = None if t_liquidus is None else float(t_liquidus)
        self.vel_warmup_steps = int(vel_warmup_steps)
        self.vel_ramp_steps = int(vel_ramp_steps)
        self.vel_tau = float(vel_tau)

    def _vel_scale(self, step: int) -> float:
        """Warmup/ramp factor in [0, 1] for the velocity gradient (not value)."""
        if step < self.vel_warmup_steps:
            return 0.0
        if self.vel_ramp_steps <= 0:
            return 1.0
        return min(1.0, (step - self.vel_warmup_steps) / self.vel_ramp_steps)

    def _pool_mask(self, true_outvar: Dict[str, Tensor]) -> Tensor:
        """Smooth melt-pool mask from TRUE fields: fl(T) x metal(alpha), detached.

        The alpha factor is not optional: with alpha masking the gas T fill is
        ~boiling, so ``fl(T_fill) = 1`` and a T-only mask scores the entire gas
        region (the same trap the tail term refuses to fall into).
        """
        true_t = true_outvar.get(self.MELT_KEY)
        alpha_true = true_outvar.get(self.CAVITY_KEY)
        if true_t is None or alpha_true is None:
            raise ValueError(
                "GridGradientLoss: velocity outputs need true 'T' and 'alpha' in the "
                "same constraint to build the melt-pool mask; got "
                f"T={'yes' if true_t is not None else 'no'}, "
                f"alpha={'yes' if alpha_true is not None else 'no'}. Velocities are "
                "only defined inside the pool -- refusing to score the whole domain."
            )
        if self.t_solidus is None or self.t_liquidus is None:
            raise ValueError(
                "GridGradientLoss: velocity outputs need t_solidus/t_liquidus for the "
                "liquid-fraction mask."
            )
        fl = smooth_liquid_fraction_torch(true_t.detach(), self.t_solidus, self.t_liquidus)
        metal = torch.sigmoid((alpha_true.detach() - 0.5) / self.vel_tau)
        return fl * metal

    def _melt_scale(self, step: int) -> float:
        """Warmup/ramp factor in [0, 1] for the melt-focus terms."""
        if self.melt_dice_weight <= 0.0 and self.melt_weight <= 0.0 and self.tail_weight <= 0.0:
            return 0.0
        if step < self.warmup_steps:
            return 0.0
        if self.ramp_steps <= 0:
            return 1.0
        return min(1.0, (step - self.warmup_steps) / self.ramp_steps)

    def _soft_mask(self, key: str, value: Tensor):
        """Differentiable melt (from T) or cavity (from alpha) mask; None if undefined."""
        if key == self.MELT_KEY:
            if self.t_solidus is None or self.t_liquidus is None:
                return None
            return smooth_liquid_fraction_torch(value, self.t_solidus, self.t_liquidus)
        if key == self.CAVITY_KEY:
            return torch.sigmoid((0.5 - value) / self.melt_tau)
        return None

    def _melt_region_weight(self, true_outvar: Dict[str, Tensor], scale: float):
        """``1 + melt_weight * fl(true T)``, or None when off/underivable."""
        if scale <= 0.0 or self.melt_weight <= 0.0:
            return None
        true_t = true_outvar.get(self.MELT_KEY)
        if true_t is None:
            return None
        mask = self._soft_mask(self.MELT_KEY, true_t.detach())
        if mask is None:
            return None
        return 1.0 + self.melt_weight * scale * mask

    def _edge_weight(self, true: Tensor):
        """Per-voxel weight in (0, 1]; low where |grad true| is large. None if off."""
        if self.edge_downweight <= 0.0:
            return None
        dims = tuple(range(true.ndim - self.spatial_dims, true.ndim))
        grads = torch.gradient(true, dim=dims)  # same-shape central differences
        g = torch.sqrt(sum(gi.pow(2) for gi in grads) + 1e-12)
        gn = g / (g.mean() + 1e-8)  # scale-free: beta~1 halves an average-gradient voxel
        return 1.0 / (1.0 + self.edge_downweight * gn)

    def forward(
        self,
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
    ) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}
        scale = self._melt_scale(step)
        mw = self._melt_region_weight(true_outvar, scale)
        pool = None
        if any(k in pred_outvar for k in self.VELOCITY_KEYS):
            pool = self._pool_mask(true_outvar)
        for key, pred in pred_outvar.items():
            true = true_outvar[key]
            w = lambda_weighting[key]
            if key in self.VELOCITY_KEYS:
                # Velocities exist only in the melt pool: score mask*(pred-true)
                # (identical to masking both sides), no H1 term -- the mask edge
                # would manufacture gradient mismatch neither field has. The
                # warmup detaches the MODEL gradient but keeps the value, so a
                # homoscedastic aggregator calibrates this key's sigma from step
                # 0 instead of collapsing it on 4k steps of literal zero.
                vel_pointwise = w * (pool * (pred - true)).pow(2)
                if "area" in invar:
                    vel_pointwise = vel_pointwise * invar["area"]
                vel_loss = vel_pointwise.sum()
                s = self._vel_scale(step)
                losses[key] = s * vel_loss + (1.0 - s) * vel_loss.detach()
                continue
            ew = self._edge_weight(true)
            if ew is not None:
                w = w * ew
            if mw is not None and mw.shape == pred.shape:
                w = w * mw
            pointwise = w * (pred - true).pow(2)
            if "area" in invar:
                pointwise = pointwise * invar["area"]
            loss = pointwise.sum()
            if self.gradient_weight > 0.0:
                for dim in range(pred.ndim - self.spatial_dims, pred.ndim):
                    dp = torch.diff(pred, dim=dim)
                    dt = torch.diff(true, dim=dim)
                    term = (dp - dt).pow(2)
                    if ew is not None:
                        # diff shortens this dim by 1; weight by the mean of the
                        # two endpoint edge-weights so the grad term is masked too.
                        n = ew.shape[dim]
                        wd = 0.5 * (ew.narrow(dim, 0, n - 1) + ew.narrow(dim, 1, n - 1))
                        term = term * wd
                    loss = loss + self.gradient_weight * term.sum()
            # Tail term on T: the melt threshold is a hard cut at 1898 K, and an
            # L2 objective is indifferent to the ~1% of cells above it -- a model
            # can clip its peak 267 K below and lose ALL melt while T RMSE stays
            # 84 K (model 29125). No tanh in this path, so it produces gradient
            # even when the model predicts no melt at all, which a mask-space
            # term cannot (float32 tanh is bit-zero below 1717.8 K).
            if self.tail_weight > 0.0 and key == self.MELT_KEY:
                alpha_true = true_outvar.get(self.CAVITY_KEY)
                if alpha_true is None or alpha_true.shape != true.shape:
                    # Do NOT fall back to the whole domain. With alpha masking the
                    # gas carries a ~3560 K boiling fill in BOTH prediction and
                    # truth, so it would own the top-k on both sides, the
                    # difference would be ~0, and the term would silently do
                    # nothing for the whole run. Fail loudly instead.
                    if not self._tail_warned:
                        print(
                            "GridGradientLoss: tail_weight>0 but true alpha is unavailable "
                            f"(key={self.CAVITY_KEY!r}); the tail term is DISABLED for this run "
                            "rather than being computed over the gas fill.",
                            flush=True,
                        )
                        self._tail_warned = True
                else:
                    metal = alpha_true.detach() > 0.5
                    loss = loss + self.tail_weight * topk_tail_loss(
                        pred, true, metal, k=self.tail_k
                    )
            if scale > 0.0 and self.melt_dice_weight > 0.0:
                mask_pred = self._soft_mask(key, pred)
                mask_true = self._soft_mask(key, true)
                if mask_pred is not None:
                    # Soft Dice with a squared denominator (Milletari): with the
                    # plain |A|+|B| form a soft mask does not score 0 against
                    # itself, because sum(m*m) != sum(m) off the 0/1 corners.
                    # eps in both places so a frame with no melt scores 0, not 1.
                    overlap = (mask_pred * mask_true).sum()
                    total = mask_pred.pow(2).sum() + mask_true.pow(2).sum()
                    dice = 1.0 - (2.0 * overlap + 1e-6) / (total + 1e-6)
                    loss = loss + self.melt_dice_weight * scale * dice
            losses[key] = loss
        return losses
