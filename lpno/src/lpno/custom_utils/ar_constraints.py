"""Autoregressive training constraints (multi-step unrolled loss, mitigation #4).

``UnrolledGridConstraint`` trains a grid model (FNO/RIGNO) on its own rollout: it
unrolls ``k`` steps, feeding each prediction back into the ``*_prev`` inputs, and
averages the base per-step loss over the k steps. This exposes the model at
training time to the off-distribution state it faces during the autoregressive
rollout eval, constraining the input->output gain that plain one-step training
leaves free (docs/training_and_evaluation.md, "Roadmap").

The dataset is built by ``_build_autoregressive_window_pairs``: step-0 context and
targets are the base keys (which drive the physicsnemo Graph); per-step future
context (``c__ar{i}``) and future targets (``s__tgt_ar{i}``) ride along as extra
ds_input keys that the Graph ignores and this constraint slices per step.
"""
from typing import Dict, List

import torch
from physicsnemo.sym.domain.constraint import SupervisedGridConstraint
from physicsnemo.sym.key import Key


class TemporalConsistencyGridConstraint(SupervisedGridConstraint):
    """Non-autoregressive temporal-consistency constraint (causality lever 2).

    Queries the DIRECT per-frame map twice: once at the sample's time t, and once
    at t+delta using time-advanced inputs (``*__adv`` ride-along keys: t_hat, beam
    position, dose) — NOT fed-back predictions. Adds a ``temporal_consistency`` loss
    term penalizing the model's predicted change ``T(t+delta)-T(t)`` against the GT
    change. Because the second pass uses GT-derived advanced inputs (not the model's
    own output), there is no rollout and no error accumulation; it just trains the
    direct map to respect the temporal evolution between two real frames.

    Fail-safe: if the ``*__adv``/``T__adv_tgt`` ride-along keys are absent (mis-wired
    dataset), it silently reduces to the plain supervised loss.
    """

    def __init__(self, *args, advance_keys: List[str], tc_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._advance_keys = list(advance_keys)
        self._tc_weight = float(tc_weight)
        self._pred_t: Dict[str, torch.Tensor] | None = None
        self._pred_dt: Dict[str, torch.Tensor] | None = None

    def forward(self):
        self._pred_t = self.model(self._input_vars)
        self._output_vars = self._pred_t  # keep base-class compatibility
        if not any(f"{k}__adv" in self._input_vars for k in self._advance_keys):
            self._pred_dt = None
            return
        adv = dict(self._input_vars)
        for k in self._advance_keys:
            akey = f"{k}__adv"
            if akey in self._input_vars:
                adv[k] = self._input_vars[akey]
        self._pred_dt = self.model(adv)

    @property
    def output_names(self) -> List[Key]:
        # Count temporal_consistency as a loss term so fixed-size adaptive aggregators
        # (homoscedastic/grad_norm/...) size their per-loss weights to include it
        # (domain.get_num_losses counts unique output_names). The plain `sum`
        # aggregator ignores the count, so this is harmless there.
        return list(self._output_names) + [Key("temporal_consistency")]

    def loss(self, step: int) -> Dict[str, torch.Tensor]:
        losses = self._loss(
            self._input_vars, self._pred_t, self._target_vars, self._lambda_weighting, step
        )
        # Always emit the key (0 when the ride-along frame is absent) so the loss-dict
        # size matches num_losses every step, which the adaptive aggregators require.
        tc = torch.zeros((), device=self._pred_t["T"].device, dtype=self._pred_t["T"].dtype)
        if self._pred_dt is not None and "T__adv_tgt" in self._input_vars:
            pred_change = self._pred_dt["T"] - self._pred_t["T"]
            gt_change = self._input_vars["T__adv_tgt"] - self._target_vars["T"]
            tc = self._tc_weight * (pred_change - gt_change).pow(2).sum()
        losses["temporal_consistency"] = tc
        return losses


class UnrolledGridConstraint(SupervisedGridConstraint):
    """Grid constraint that unrolls the model ``unroll_steps`` times per step.

    Parameters mirror ``SupervisedGridConstraint`` plus:
      context_keys / state_keys : the exogenous context and fed-back state keys.
      state_prev_suffix         : suffix of the ``*_prev`` feedback inputs.
      unroll_steps              : k (>= 2); k=1 should use the plain constraint.
      detach_grad               : truncate the gradient between steps (pushforward
                                  / no-BPTT). False = full backprop-through-time.
    """

    def __init__(
        self,
        *args,
        context_keys: List[str],
        state_keys: List[str],
        state_prev_suffix: str,
        unroll_steps: int,
        detach_grad: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._ctx_keys = list(context_keys)
        self._state_keys = list(state_keys)
        self._prev_suffix = str(state_prev_suffix)
        self._k = int(unroll_steps)
        self._detach = bool(detach_grad)
        if self._k < 2:
            raise ValueError(f"UnrolledGridConstraint needs unroll_steps >= 2; got {self._k}.")
        self._unroll_preds: List[Dict[str, torch.Tensor]] | None = None
        self._step_invars: List[Dict[str, torch.Tensor]] | None = None

    def _step_input(self, i: int, running_prev: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        invar: Dict[str, torch.Tensor] = {}
        for key in self._ctx_keys:
            invar[key] = self._input_vars[key] if i == 0 else self._input_vars[f"{key}__ar{i}"]
        for key in self._state_keys:
            invar[f"{key}{self._prev_suffix}"] = running_prev[key]
        return invar

    def forward(self):
        running_prev = {key: self._input_vars[f"{key}{self._prev_suffix}"] for key in self._state_keys}
        preds: List[Dict[str, torch.Tensor]] = []
        invars: List[Dict[str, torch.Tensor]] = []
        for i in range(self._k):
            invar_i = self._step_input(i, running_prev)
            pred_i = self.model(invar_i)
            preds.append(pred_i)
            invars.append(invar_i)
            running_prev = {
                key: (pred_i[key].detach() if self._detach else pred_i[key]) for key in self._state_keys
            }
        self._unroll_preds = preds
        self._step_invars = invars
        self._output_vars = preds[-1]  # keep base-class compatibility

    def loss(self, step: int) -> Dict[str, torch.Tensor]:
        if self._unroll_preds is None:
            return super().loss(step)
        total: Dict[str, torch.Tensor] = {}
        for i, pred_i in enumerate(self._unroll_preds):
            if i == 0:
                true_i = {key: self._target_vars[key] for key in pred_i.keys()}
            else:
                true_i = {key: self._input_vars[f"{key}__tgt_ar{i}"].detach() for key in pred_i.keys()}
            lambda_i = {key: torch.ones_like(pred_i[key]) for key in pred_i.keys()}
            step_losses = self._loss(self._step_invars[i], pred_i, true_i, lambda_i, step)
            for loss_key, value in step_losses.items():
                total[loss_key] = total.get(loss_key, 0.0) + value / self._k
        return total
