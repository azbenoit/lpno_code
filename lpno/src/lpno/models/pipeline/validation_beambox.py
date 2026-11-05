"""Report beam-box error during training validation, not only at eval time.

Whole-domain validation MAE describes the quiescent ~99% of the substrate. The metric that
tracks the melt region is the beam-box error -- restricted to within 2 beam radii of the
laser, inside the metal -- and it existed only on the eval path, so an arm's melt-region
behaviour was invisible for the whole run.

Getting it into validation needs two hooks, because neither alone has the pieces:

  * `_l2_relative_error(true_var, pred_var)` receives the CONCATENATED outputs but, by
    signature, never the inputs.
  * `save_results` does see the inputs -- it builds `invar_cpu` from
    `self.dataset.invar_keys` -- but assembles them locally and returns only `losses`.

So `self.forward` is wrapped to stash `beam_dist` and `z` as each minibatch passes, in the
same order the outputs are concatenated, and the metric is computed in the
`_l2_relative_error` wrapper where the concatenated true/pred are in hand. This adds **no
extra forward passes and no extra data loading** -- it keeps references to tensors the
validator already computed. Re-iterating the dataloader instead would double validation
cost on a lazy NPZ store, where a cache miss decompresses an entire case.

The metric itself is a masked mean over the beam box, a small fraction of the domain,
against an MAE/RMSE pass over all of it.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: The one channel the box cannot be built without. `beam_dist` used to be required
#: alongside it, which is what kept this metric off every key set that omits that channel
#: -- run 82336 (`FNO mref-tab CROP10 v2`) trained 32k steps with no `beambox_*` column on
#: the 8-channel set `[V_scan,P,H_norm,x,y,z,beam_x,beam_y]`, which states the beam centre
#: outright. `beam_box_from_input` had already been taught to synthesise the footprint from
#: `beam_x`/`beam_y` (2026-08-22: "beam box should not need beam dist, fix it"); the
#: fix landed on the eval path only and the two validation gates kept the old precondition.
REQUIRED_INPUTS = ("z",)

#: Stashed when present, because the eval-path box uses them: `x`/`y` are the grids the
#: footprint is drawn on and what makes the beam sigma (and therefore the DEPTH bound)
#: recoverable, `beam_x`/`beam_y` carry the beam centre when `beam_dist` is absent,
#: `beam_radius` sizes a synthesised footprint exactly (already in grid units --
#: `_preprocess_var` divides it by ref_scales["L"]), and `beam_shape_uniform` selects the
#: top-hat threshold. Absent, the box degrades to the eval's own behaviour rather than to a
#: different definition.
OPTIONAL_INPUTS = (
    "beam_dist", "x", "y", "beam_x", "beam_y", "beam_shape_uniform", "beam_radius",
)


def beam_box_inputs_available(has, at_origin: bool | None = None) -> bool:
    """Whether a beam box is buildable from these channels. ``has(key) -> bool``.

    Shared by the two validation gates so the precondition cannot drift from the builder
    the way the box predicate itself did twice (see `_beam_box_from_stash`). Mirrors
    `beam_box_from_input`: a real `beam_dist` footprint is enough on its own; without it
    the footprint is synthesised, which needs the centre AND both grids it is drawn on.
    Checking the grids here rather than letting the synthesiser raise keeps a legitimate
    config (no trajectory channel at all) silent instead of logging a warning per step.
    """
    if at_origin is None:
        at_origin = _FRAME["at_origin"]
    if not has("z"):
        return False
    if has("beam_dist"):
        return True
    # Beam-following frame: the centre is the origin by construction, so only the grids are
    # needed. The SIZE comes from the `beam_radius` channel when the key set carries one
    # (82820/82838 do) and otherwise from the announced fixed default.
    if at_origin and has("x") and has("y"):
        return True
    return all(has(k) for k in ("beam_x", "beam_y", "x", "y"))

#: Set by the forward wrapper, read by the `_l2_relative_error` wrapper in the same call.
_STASH: dict[str, list] = {}

#: Whether the enclosed run is in a BEAM-FOLLOWING frame, set by `beam_box_validation` from
#: `custom.moving_ref_frame`. There the beam sits at the grid origin by construction, so the
#: box is determined without any beam channel at all -- which is how the eval path has always
#: built it (`beam_at_origin=bool(run_cfg.moving_ref_frame)`, three sites in postprocessing).
#:
#: Validation used to refuse this deliberately, so a steady run reported no box while its own
#: eval reported one. The refusal existed because `best_checkpoint._METRIC_PREFERENCE` ranks
#: `beambox_mae_T` above `mae_T` and takes the first AVAILABLE metric, so emitting it changes
#: which checkpoint a beam-following run saves. User decision 2026-09-01, on runs 82820/82838
#: (`*_anchorbox_steady_32k`): report it and let it select -- weights chosen on the melt
#: region beat weights chosen on the quiescent ~99% substrate, and training then agrees with
#: eval. NEVER inferred: it follows the config flag, so a lab-frame run cannot acquire an
#: origin box by accident, which is the failure this metric has produced three times.
_FRAME: dict[str, bool] = {"at_origin": False}


def _beam_box_from_stash(beam, z, x=None, shape=None, n_radii: float = 2.0,
                         y=None, beam_x=None, beam_y=None, radius=None,
                         at_origin: bool | None = None):
    """Cells within ``n_radii`` beam radii of the beam AND inside the metal.

    Delegates to `postprocessing.beam_box_from_input`, the eval path's own
    builder, so there is exactly ONE definition of the box. Twice now this
    function held a second copy of it and the copies diverged:

    * 2026-08-10: `beam.abs() <= n_radii` was true for EVERY voxel (`beam_dist`
      is a peak-1 footprint, not a distance), so `beambox_*` was the substrate
      under a beam-box name -- val beambox_mae_T 0.00478 on eval
      substrate_abs_err_mean 0.00439, 8x below the true eval box 0.0402.
    * 2026-08-15: the radial predicate was fixed but the DEPTH bound
      (`z >= -n_radii*sigma`) and the uniform-footprint threshold were not
      carried across, so the box stayed a full-depth column of quiescent metal
      -- `rigno_e8_varspeed_35k` val beambox_mae_T 0.005155 against the eval's
      0.049499 on the same model, 9.6x apart and again sitting on the eval's
      substrate_abs_err_mean 0.006176.

    Torch in, torch out; the shared builder is numpy, and `np.asarray` on a CPU
    tensor is a view, so the round trip copies nothing.
    """
    import numpy as np
    import torch

    from lpno.models.pipeline.postprocessing import beam_box_from_input

    def _host(value):
        """`np.asarray` raises on a device-resident tensor, and the stash holds
        references to whatever the validator computed on -- the GPU, in training.
        Detach-and-move only when the object offers it; a CPU tensor still
        round-trips as a view, so the zero-copy claim above is unaffected.
        """
        detach = getattr(value, "detach", None)
        if detach is not None:
            value = detach()
            cpu = getattr(value, "cpu", None)
            if cpu is not None:
                value = cpu()
        return np.asarray(value)

    # `beam` is optional: with the centre channels present the shared builder synthesises
    # the footprint instead, so a key set without `beam_dist` still gets a real box.
    ds_input = {}
    for key, value in (
        ("beam_dist", beam), ("x", x), ("y", y), ("beam_x", beam_x), ("beam_y", beam_y),
        ("beam_shape_uniform", shape), ("beam_radius", radius),
    ):
        if value is not None:
            ds_input[key] = _host(value)
    box = beam_box_from_input(
        ds_input, _host(z), n_radii=n_radii,
        beam_at_origin=(_FRAME["at_origin"] if at_origin is None else bool(at_origin)),
    )
    if box is None:
        return None
    return torch.as_tensor(np.ascontiguousarray(box), dtype=torch.bool)


def gas_alpha_validation_metrics(true_var: dict, pred_var: dict, threshold: float = 0.5) -> dict:
    """How much of the true GAS region the model's own alpha actually calls gas.

    `mae_alpha` cannot see this: it is a domain mean and the gas is ~0.4 % of cells, so a
    head that never predicts a single gas cell scored 0.00421 against a healthy baseline's
    0.00914 (models 70965 vs the expLR baseline, 2026-08-27). It matters because the
    prediction-side boiling fill is gated on PREDICTED alpha, so a collapsed head silently
    skips the fill and leaves the gas ~1257 K cold.

    Returns `{}` when alpha is absent or the truth contains no gas, so a batch that simply
    has no gas cannot look like a failure.
    """
    import torch

    t = true_var.get("alpha")
    p = pred_var.get("alpha")
    if t is None or p is None or t.shape != p.shape:
        return {}
    true_gas = t < threshold
    n_true = int(true_gas.sum().item())
    if n_true == 0:
        return {}
    pred_gas = p.to(true_gas.device) < threshold
    caught = int((pred_gas & true_gas).sum().item())
    n_pred = int(pred_gas.sum().item())
    return {
        "gas_true_n": torch.tensor(float(n_true)),
        "gas_pred_n": torch.tensor(float(n_pred)),
        "gas_recall_alpha": torch.tensor(float(caught) / float(n_true)),
        # Share of ALL cells the model calls gas -- the all-gas collapse signal. Recall
        # cannot see that mode (predicting gas everywhere gives recall 1.0).
        "gas_pred_fraction_alpha": torch.tensor(float(n_pred) / float(t.numel())),
    }


#: True once the "no beam box here" notice has been printed for this process.
_UNAVAILABLE_ANNOUNCED: set = set()


def _announce_unavailable(has) -> None:
    """Say, once, that no beam box can be built here and which channel is why.

    The degradation itself is deliberate -- a config with no laser-trajectory channel is
    legitimate -- but it was INAUDIBLE, and it does not merely drop a number: with
    `beambox_mae_T` absent, `best_checkpoint._METRIC_PREFERENCE` falls through to `mae_T`,
    so the run selects its checkpoint on whole-domain substrate error while sibling arms
    select on the melt region. GAOT AR k=8 (model 86604, run 85613) finished 32k steps that
    way and read as a comparable arm; its config inherits `config_flow3D_RIGNO`, whose
    input keys carry no `beam_dist`, while the arms it was compared against used the
    `*_randompath` configs that do (session 067cdc63, 2026-09-02).
    """
    if _UNAVAILABLE_ANNOUNCED:
        return
    _UNAVAILABLE_ANNOUNCED.add(True)
    if not has("z"):
        why = "no 'z' channel"
    else:
        why = ("no 'beam_dist', and no stated centre either ('beam_x'/'beam_y' plus the "
               "'x'/'y' grids); a beam-following run would also need custom.moving_ref_frame")
    print(
        f"[beam-box] beambox_* metrics are NOT available for this run: {why}. This is "
        "legitimate for a config without a laser-trajectory channel, but it CHANGES "
        "CHECKPOINT SELECTION: best_checkpoint prefers beambox_mae_T and will fall back to "
        "mae_T (whole-domain substrate error), so this arm is not ranked like one that has "
        "the beam box. Add beam_dist -- or beam_x/beam_y with x/y -- to the input keys to "
        "restore it.",
        flush=True,
    )


def beam_box_validation_metrics(true_var: dict, pred_var: dict, n_radii: float = 2.0) -> dict:
    """`beambox_*` metrics from the stashed inputs. Empty dict when not computable."""
    import torch

    if not beam_box_inputs_available(lambda k: bool(_STASH.get(k))):
        _announce_unavailable(lambda k: bool(_STASH.get(k)))
        return {}
    def _cat(key):
        """Concatenated stash entry, or None when the channel was not captured."""
        parts = _STASH.get(key)
        return torch.cat(parts, dim=0) if parts else None

    try:
        beam = _cat("beam_dist")
        z = torch.cat(_STASH["z"], dim=0)
        x = _cat("x")
        shape = _cat("beam_shape_uniform")
        y = _cat("y")
        beam_x = _cat("beam_x")
        beam_y = _cat("beam_y")
        radius = _cat("beam_radius")
    except Exception:
        return {}

    # n_radii by KEYWORD: it used to be the third positional argument, which is
    # now `x`, so a positional call would silently pass 2.0 as the x channel.
    box = _beam_box_from_stash(beam, z, x=x, shape=shape, n_radii=n_radii,
                               y=y, beam_x=beam_x, beam_y=beam_y, radius=radius)
    if box is None:
        return {}
    n = int(box.sum().item())
    if n == 0:
        return {}

    out: dict[str, Any] = {}
    for key in true_var:
        if key not in pred_var:
            continue
        t, p = true_var[key], pred_var[key]
        if t.shape != box.shape or p.shape != box.shape:
            continue
        err = (t.to(box.device) - p.to(box.device))[box]
        if err.numel() == 0:
            continue
        out[f"beambox_mae_{key}"] = torch.mean(torch.abs(err))
        out[f"beambox_rmse_{key}"] = torch.sqrt(torch.mean(torch.square(err)))
    if out:
        out["beambox_n"] = torch.tensor(float(n))
    return out


@contextlib.contextmanager
def beam_box_validation(enabled: bool = True, at_origin: bool = False):
    """Add `beambox_*` to every validator's reported metrics, for the enclosed block.

    `at_origin` marks a beam-following frame (`custom.moving_ref_frame`), where the box is
    determined with no beam channel at all. It is passed explicitly rather than sniffed from
    the data, so a lab-frame run cannot acquire an origin box by accident.
    """
    if not enabled:
        yield
        return
    _FRAME["at_origin"] = bool(at_origin)
    _UNAVAILABLE_ANNOUNCED.clear()

    from physicsnemo.sym.domain.validator import GridValidator, PointwiseValidator

    patched: list[tuple[type, str, Any]] = []

    def _wrap_save(original):
        def save_results(self, name, results_dir, writer, save_filetypes, step):
            _STASH.clear()
            original_forward = self.forward

            def capturing_forward(invar, *a, **kw):
                for key in REQUIRED_INPUTS + OPTIONAL_INPUTS:
                    if key in invar:
                        _STASH.setdefault(key, []).append(invar[key].detach().cpu())
                return original_forward(invar, *a, **kw)

            self.forward = capturing_forward
            try:
                return original(self, name, results_dir, writer, save_filetypes, step)
            finally:
                self.forward = original_forward
                _STASH.clear()

        return save_results

    def _wrap_l2(original):
        def wrapped(true_var, pred_var, _orig=original):
            out = dict(_orig(true_var, pred_var))
            try:
                out.update(beam_box_validation_metrics(true_var, pred_var))
            except Exception as exc:  # never let a metric kill a training run
                logger.warning("beam-box validation metric failed: %s", exc)
            try:
                out.update(gas_alpha_validation_metrics(true_var, pred_var))
            except Exception as exc:  # same rule: a metric must not kill a run
                logger.warning("gas-alpha validation metric failed: %s", exc)
            return out

        return wrapped

    for cls in (GridValidator, PointwiseValidator):
        if "save_results" in cls.__dict__ and not getattr(
            cls.save_results, "_beambox_patched", False
        ):
            patched.append((cls, "save_results", cls.save_results))
            w = _wrap_save(cls.__dict__["save_results"])
            w._beambox_patched = True
            cls.save_results = w
        if not getattr(cls._l2_relative_error, "_beambox_patched", False):
            patched.append((cls, "_l2_relative_error", cls._l2_relative_error))
            w2 = _wrap_l2(cls._l2_relative_error)
            w2._beambox_patched = True
            cls._l2_relative_error = staticmethod(w2)

    try:
        yield
    finally:
        for cls, attr, original in patched:
            setattr(cls, attr, original)
        _STASH.clear()
        _FRAME["at_origin"] = False
