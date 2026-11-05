# pyright: reportAttributeAccessIssue=false
# pyright: reportUnboundVariable=false
# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportOptionalSubscript=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportOptionalMemberAccess=false
"""Model/node construction and PhysicsNeMo domain assembly helpers."""

import csv
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from physicsnemo.sym.constants import TF_SUMMARY
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import Constraint, SupervisedGridConstraint
from physicsnemo.sym.utils.io.vtk import grid_to_vtk
from physicsnemo.sym.domain.constraint.continuous import DeepONetConstraint, PointwiseConstraint
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.validator import GridValidator
from physicsnemo.sym.hydra import PhysicsNeMoConfig, instantiate_arch
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.solver import Solver

from ...equations import MaskNode, SpatialAlignNode, WarmupNode, fl_Node
from ...equations.FNO_postprocess import set_train_step
# Physics-loss PDE residuals (EnergyConservation, VOF, EnergyDiffusionTerm) are
# deprecated; imported lazily from `deprecated.equations` in the pde branches.
from ...custom_utils.utils import boiling_fill_value
from ...plotting.validators import CustomSlicePlotter, TemporalPatchPlotter

from .dataclasses import DatasetBundle, KeyConfig, ModelBundle, RunConfig
from .npz_lazy_store import dataloader_workers


def _augment_validator_metrics() -> None:
    """Record absolute MAE and RMSE alongside physicsnemo's l2_relative_error.

    l2_relative_error (a variance-normalized RMSE) dissociates from the absolute
    MAE the region eval judges on, so surface both during validation. Patched on
    the class because save_results calls the staticmethod via the explicit class
    name (`GridValidator._l2_relative_error`), so a subclass override is ignored.
    Metrics stay in the model's normalized T units (scale by REF_T for Kelvin).
    """
    for cls in (GridValidator, PointwiseValidator):
        if getattr(cls._l2_relative_error, "_mae_augmented", False):
            continue
        orig = cls._l2_relative_error

        def wrapped(true_var, pred_var, _orig=orig):
            out = dict(_orig(true_var, pred_var))
            for key in true_var:
                tv = true_var[key].reshape(-1)
                pv = pred_var[key].reshape(-1)
                err = tv - pv
                out["mae_" + str(key)] = torch.mean(torch.abs(err))
                out["rmse_" + str(key)] = torch.sqrt(torch.mean(torch.square(err)))
                # Tail deficit: how much colder the prediction's hot end is than truth.
                #
                # MAE and RMSE cannot see a cold tail. ~99% of the substrate is quiescent
                # metal, so they describe the easy part: `gaot_cond_disp_deep_v2` finished
                # only 45% behind on val MAE while running 151 K colder at the tail
                # (p99.9 1771 K against truth 1922 K), which is what actually cost it the
                # wake melt -- the melt threshold is 1898 K, so a tail below it forms no
                # melt however healthy the peak looks. `thermal_headroom_K` read +837 K on
                # that same model and is deliberately NOT surfaced here: it is a floor
                # check ("can this melt at all"), not a fidelity measure.
                #
                # Peak-based rather than p99.9 on purpose. This hook runs on the whole
                # concatenated validation tensor, and measured on 60M elements a topk
                # p99.9 costs ~6x the entire existing metric computation while max costs
                # ~10% of it. The lazy eval path already falls back to the same peak-based
                # variant for the same reason, so the two agree.
                if str(key) == "T":
                    out["tail_deficit_max_" + str(key)] = tv.max() - pv.max()
            return out

        wrapped._mae_augmented = True
        cls._l2_relative_error = staticmethod(wrapped)


_augment_validator_metrics()


class AlphaCollapseError(RuntimeError):
    """Raised to stop a run whose alpha head no longer predicts the gas region."""


_ALPHA_GUARD = None


def record_vram_peak(network_dir) -> int | None:
    """Write the run's measured CUDA peak to `<network_dir>/vram_peak.json`, return MiB.

    Exists because every path donor records `gpu_vram_peak_mb = None`, so bookings have
    been folklore: 2026-08-28 alone produced over-booking that idled three GPUs behind a
    full ledger (12000 booked, 7.4-7.7 GiB used) AND an inherited under-booking OOM
    (74176: 11218 booked). `max_memory_reserved` is the number the allocator actually
    holds and is what the next booking should start from; `max_memory_allocated` is
    recorded beside it for context. Never raises -- bookkeeping must not kill a run.
    """
    try:
        import json as _json
        from pathlib import Path as _Path

        if not torch.cuda.is_available():
            return None
        reserved = int(torch.cuda.max_memory_reserved() // 2**20)
        allocated = int(torch.cuda.max_memory_allocated() // 2**20)
        payload = {"gpu_vram_peak_mb": reserved, "gpu_vram_allocated_peak_mb": allocated}
        _Path(_Path(network_dir) / "vram_peak.json").write_text(_json.dumps(payload) + "\n")
        print(f"[vram] peak reserved {reserved} MiB, allocated {allocated} MiB "
              f"-- book from THIS, not the donor's guess", flush=True)
        return reserved
    except Exception:  # noqa: BLE001 - a metric must not kill a finished run
        return None


def _check_alpha_collapse(losses: dict, step: int) -> None:
    """Stop the run when `gas_recall_alpha` has been zero for several validations.

    Deliberately an abort rather than a `stop_criterion` metric: a collapsed alpha head sits
    at exactly zero forever while every temperature metric looks healthy, so a
    plateau-with-patience rule is the wrong instrument (see `alpha_guard` for the measured
    case). Disabled by setting PINO_ALPHA_GUARD=0.
    """
    import os

    global _ALPHA_GUARD
    if os.environ.get("PINO_ALPHA_GUARD", "").strip() in ("0", "false", "no"):
        return
    def _scalar(key):
        raw = losses.get(key)
        if raw is None:
            return None
        try:
            return float(raw.item() if hasattr(raw, "item") else raw)
        except Exception:  # noqa: BLE001 - a bad metric must not kill a run by itself
            return None

    recall = _scalar("gas_recall_alpha")
    pred_fraction = _scalar("gas_pred_fraction_alpha")
    if _ALPHA_GUARD is None:
        from lpno.models.pipeline.alpha_guard import AlphaCollapseGuard

        _ALPHA_GUARD = AlphaCollapseGuard(
            warmup_steps=int(os.environ.get("PINO_ALPHA_GUARD_WARMUP", "4000")),
            patience=int(os.environ.get("PINO_ALPHA_GUARD_PATIENCE", "3")),
        )
    reason = _ALPHA_GUARD.update(int(step), recall, pred_fraction=pred_fraction)
    if reason:
        print(f"[alpha-guard] STOPPING: {reason}", flush=True)
        raise AlphaCollapseError(reason)


def _append_validation_metrics_csv(results_dir: str, name: str, losses: dict[str, torch.Tensor], step: int) -> None:
    """Persist validator scalars even when the active writer drops TensorBoard tags."""
    try:
        out = Path(results_dir) / "validation_metrics.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        exists = out.is_file()
        with out.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["step", "validator", "metric", "value"])
            if not exists:
                writer.writeheader()
            for metric, value in sorted(losses.items()):
                writer.writerow({
                    "step": int(step),
                    "validator": str(name),
                    "metric": str(metric),
                    "value": float(value.detach().cpu()),
                })
    except Exception as exc:  # noqa: BLE001 - validation must not fail because metric export did
        print(f"[validation] WARNING: could not append validation_metrics.csv: {exc}", flush=True)


def _publish_train_step_to_warmup_nodes() -> None:
    """Feed the trainer's step to `WarmupNode` via `Constraint.loss(step)`.

    `WarmupNode` gates hard alpha-masking on a step count, but it cannot measure
    training progress by counting its own forwards -- the node runs several times
    per optimizer step, so a call counter overshoots by an unknown factor and the
    mask engages early (2026-07-28: `mask_warmup_steps=4000` fired near training
    step 2000 and collapsed the outputs to the constant fill). `Constraint.loss`
    is the one hook that receives the trainer's authoritative step and runs once
    per optimizer step. Patched on the classes for the same reason as the
    validator metrics above: the trainer calls these directly.
    """
    for cls in (SupervisedGridConstraint, DeepONetConstraint, PointwiseConstraint):
        if getattr(cls.loss, "_publishes_train_step", False):
            continue
        orig = cls.loss

        def wrapped(self, step, _orig=orig):
            set_train_step(step)
            return _orig(self, step)

        wrapped._publishes_train_step = True
        cls.loss = wrapped


_publish_train_step_to_warmup_nodes()


class StreamingGridValidator(GridValidator):
    """GridValidator whose save_results streams batches instead of hoarding them (A35).

    Upstream ``GridValidator.save_results`` appends every validation batch's
    invar/true/pred to host-RAM lists before computing metrics, which spikes RSS
    with the whole val split (~45 GB on ti64_random150) and gets the run
    SIGKILLed. This override computes the same per-key metrics
    (``l2_relative_error = sqrt(mse / var(true))`` with unbiased variance, plus
    the repo's ``mae``/``rmse``) from running scalar sums, and bounds the
    np/vtk dump and plotter figures to the first batch.
    """

    #: Class-level default so instances built without __init__ (tests, pickles)
    #: still resolve it -- and so the safe answer is the one you get by omission.
    dump_files = False

    def __init__(self, *args, dump_files: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # Off by default: the npz/vtk dump is a debug artifact, and it grew to
        # 19 GB per run (229 GB across outputs/, deleted 2026-07-21). Metrics
        # reach tensorboard/wandb either way. Opt in with custom.validator_dumps.
        self.dump_files = bool(dump_files)

    def save_results(self, name, results_dir, writer, save_filetypes, step):
        stats: dict[str, dict[str, float]] = {}
        first = None
        for i, (invar0, true_outvar0, lambda_weighting) in enumerate(self.dataloader):
            invar = Constraint._set_device(invar0, device=self.device, requires_grad=self.requires_grad)
            true_outvar = Constraint._set_device(true_outvar0, device=self.device, requires_grad=self.requires_grad)
            pred_outvar = self.forward(invar)
            # Beam box: within n_radii of the beam axis, inside the metal. Built per
            # batch because `invar` is only available here -- the upstream
            # `_l2_relative_error` hook never sees the inputs.
            box = None
            from .validation_beambox import beam_box_inputs_available
            # Was `"beam_dist" in invar and "z" in invar`, which is why the 8-channel set
            # (beam_x/beam_y, no beam_dist) trained with an empty beambox column -- this is
            # the copy that reaches the CSV. The precondition is shared with the validator
            # gate so the two cannot drift.
            if beam_box_inputs_available(lambda k: k in invar):
                # ONE definition of the box, shared with the eval path. This held a
                # hand-rolled third copy until 2026-08-26:
                #     (beam_dist.abs() <= 2.0) & (z <= 0)
                # `beam_dist` is a peak-1 FOOTPRINT, not a distance -- measured on
                # real data it spans [0.000000, 0.995010], so `abs() <= 2.0` admitted
                # 100.000% of voxels and the "beam box" was exactly the substrate.
                # `beambox_mae_T` was a substrate MAE under a beam-box name, and it
                # is the copy that reaches the CSV (see the direct
                # `losses["beambox_mae_" + key]` write below). The same bug had
                # already been fixed twice in `validation_beambox`, whose docstring
                # records both; a corrected second copy is what produced instances
                # two and three, so this now delegates rather than re-deriving.
                from .validation_beambox import _beam_box_from_stash

                def _ch(key):
                    return invar[key].detach() if key in invar else None

                box = _beam_box_from_stash(
                    _ch("beam_dist"),
                    invar["z"].detach(),
                    x=_ch("x"),
                    shape=_ch("beam_shape_uniform"),
                    y=_ch("y"),
                    beam_x=_ch("beam_x"),
                    beam_y=_ch("beam_y"),
                    radius=_ch("beam_radius"),
                )
            for key, true_t in true_outvar.items():
                err = (true_t - pred_outvar[key]).detach().double()
                true_d = true_t.detach().double()
                s = stats.setdefault(
                    key, {"n": 0.0, "err_sq": 0.0, "abs_err": 0.0, "true": 0.0, "true_sq": 0.0,
                          "true_max": -float("inf"), "pred_max": -float("inf"),
                          "box_n": 0.0, "box_abs": 0.0, "box_sq": 0.0}
                )
                s["n"] += float(err.numel())
                s["err_sq"] += float(torch.sum(err * err))
                s["abs_err"] += float(torch.sum(torch.abs(err)))
                s["true"] += float(torch.sum(true_d))
                s["true_sq"] += float(torch.sum(true_d * true_d))
                # Running peaks for the tail deficit. MAE cannot see a cold tail:
                # ~99% of the substrate is quiescent, so a model can sit 150-500 K
                # low at the top of the distribution while its mean looks healthy,
                # and every thresholded quantity (melt, keyhole) depends on the top.
                s["true_max"] = max(s["true_max"], float(true_d.max()))
                s["pred_max"] = max(s["pred_max"], float(pred_outvar[key].detach().double().max()))
                if box is not None and box.shape == err.shape:
                    be = err[box]
                    if be.numel():
                        s["box_n"] += float(be.numel())
                        s["box_abs"] += float(torch.sum(torch.abs(be)))
                        s["box_sq"] += float(torch.sum(be * be))
            if i == 0:
                # Only the first batch is materialized for the file dump/plots.
                first = (
                    {k: v.cpu().detach().numpy() for k, v in invar.items()},
                    {"true_" + k: v.cpu().detach().numpy() for k, v in true_outvar.items()},
                    {"pred_" + k: v.cpu().detach().numpy() for k, v in pred_outvar.items()},
                )

        losses = {}
        for key, s in stats.items():
            mse = s["err_sq"] / s["n"]
            mean = s["true"] / s["n"]
            # Unbiased variance (matches torch.var on the concatenated tensor).
            var = (s["true_sq"] - s["n"] * mean * mean) / max(s["n"] - 1.0, 1.0)
            losses["l2_relative_error_" + key] = torch.tensor(
                math.sqrt(mse / var) if var > 0 else float("inf")
            )
            losses["mae_" + key] = torch.tensor(s["abs_err"] / s["n"])
            losses["rmse_" + key] = torch.tensor(math.sqrt(mse))
            # Positive = prediction is COLDER than truth at the peak. Sign matters:
            # the unmodified GAOT baseline reads NEGATIVE (over-predicts the peak by
            # ~353 K) while the displacement arm reads +449 K, so "runs cold" is not
            # a property of the family. thermal_headroom_K is deliberately not
            # reported here -- it read +837 K on a model that was 151 K cold.
            if math.isfinite(s["true_max"]) and math.isfinite(s["pred_max"]):
                losses["tail_deficit_max_" + key] = torch.tensor(s["true_max"] - s["pred_max"])
            if s["box_n"] > 0:
                losses["beambox_mae_" + key] = torch.tensor(s["box_abs"] / s["box_n"])
                losses["beambox_rmse_" + key] = torch.tensor(math.sqrt(s["box_sq"] / s["box_n"]))

        if first is not None:
            invar_np, named_true, named_pred = first
            if self.dump_files and "np" in save_filetypes:
                np.savez(results_dir + name, {**invar_np, **named_true, **named_pred})
            if self.dump_files and "vtk" in save_filetypes:
                grid_to_vtk({**invar_np, **named_true, **named_pred}, results_dir + name)
            if self.plotter is not None:
                self.plotter._add_figures(
                    "Validators",
                    name,
                    results_dir,
                    writer,
                    step,
                    invar_np,
                    {k[len("true_") :]: v for k, v in named_true.items()},
                    {k[len("pred_") :]: v for k, v in named_pred.items()},
                )

        for k, loss in losses.items():
            if TF_SUMMARY:
                writer.add_scalar("val/" + name + "/" + k, loss, step, new_style=True)
            else:
                writer.add_scalar("Validators/" + name + "/" + k, loss, step, new_style=True)
        _append_validation_metrics_csv(results_dir, name, losses, step)
        _check_alpha_collapse(losses, step)
        return losses


def _validator_dumps_enabled(cfg) -> bool:
    """Whether validators write their invar/true/pred npz+vtk dump.

    Default off: these are debug artifacts that dominated disk (19 GB per run,
    229 GB across outputs/ before the 2026-07-21 cleanup), while the validation
    metrics that drive the run reach tensorboard/wandb regardless.
    """
    return bool(OmegaConf.select(cfg, "custom.validator_dumps", default=False) or False)


def _validator_plotter(cfg, factory):
    """Return factory() only when `custom.validator_plots` is enabled.

    Validator scalar errors (Validators/<name>/l2_relative_error_*) reach
    tensorboard/wandb regardless of the plotter; the plotter only adds slice
    figures, which cost time every rec_validation_freq steps.
    """
    if bool(OmegaConf.select(cfg, "custom.validator_plots", default=False) or False):
        return factory()
    return None


DEFAULT_MASK_WARMUP_STEPS = 500


def mask_warmup_start_step(*, eval_only: bool, initialization_network_dir: Any,
                           configured: Any = None) -> int:
    """Step at which hard alpha-masking switches on.

    The warmup exists to avoid hard masking too early *when training from
    scratch*. A run that loads pre-trained weights (physicsnemo's
    `initialization_network_dir`) is not from scratch, and re-running the warmup
    puts its sharp outputs through an unmasked head: the loss jumps to ~3e+04,
    fp16 AMP gradients overflow and the scaler collapses within seconds, and the
    step-0 validation looks like a failed checkpoint load. Eval already skips it.

    `configured` (`custom.mask_warmup_steps`) moves the cold-start step, because
    the shipped 500 assumes an LR schedule already at speed by then. One-cycle is
    not: it starts at `max_lr/div_factor` and peaks only at
    `pct_start * total_steps`. Masking a model that has not begun to learn alpha
    collapses its outputs to the constant fill and zeroes the gradients -- run
    `rigno_rp_all3_2ep_s221` (2026-07-27) masked at step 500 with lr=1.44e-05 and
    trained nothing for 30k steps. **Set it past the point where the LR is at
    full magnitude and validation has started moving.** Warm starts and eval
    still skip the warmup entirely; a configured value cannot re-enable it.
    """
    if eval_only:
        return 0
    init = str(initialization_network_dir or "").strip()
    if init:
        return 0
    if configured is None or not str(configured).strip():
        return DEFAULT_MASK_WARMUP_STEPS
    return max(0, int(configured))


def build_model_and_nodes(
    cfg: PhysicsNeMoConfig,
    key_cfg: KeyConfig,
    material_params: dict[str, Any],
    laser_params: dict[str, Any],
    ref_scales: dict[str, Any],
    alpha_masking: bool,
    eval_only: bool,
    model_family: str,
    deeponet_branch_keys: list[str],
    deeponet_trunk_keys: list[str],
    pde_loss: bool,
    vof_pde_loss: bool,
    diff_loss: bool,
    dx: float,
    dy: float,
    dz: float,
    diff_weight: float,
    predict_delta: bool = False,
    state_prev_suffix: str = "_prev",
) -> ModelBundle:
    """Instantiate the configured model family and optional PDE node stacks."""

    if model_family == "fno":
        decoder_net = instantiate_arch(
            cfg=cfg.arch.decoder,
            output_keys=key_cfg.fno_output_keys,
        )
        model = instantiate_arch(
            cfg=cfg.arch.fno,
            decoder_net=decoder_net,
            input_keys=key_cfg.fno_input_keys,
            output_keys=key_cfg.fno_output_keys,
        )
    elif model_family == "unet":
        model = instantiate_arch(
            cfg=cfg.arch.pix2pix,
            input_keys=key_cfg.fno_input_keys,
            output_keys=key_cfg.fno_output_keys,
            dimension=int(getattr(cfg.arch.pix2pix, "dimension", 3)),
        )
    elif model_family == "fully_connected":
        model = instantiate_arch(
            cfg=cfg.arch.fully_connected,
            input_keys=key_cfg.fno_input_keys,
            output_keys=key_cfg.fno_output_keys,
        )
    elif model_family == "deeponet":
        branch_dim = int(getattr(cfg.arch.deeponet, "branch_dim", None) or getattr(cfg.arch.branch, "layer_size", 128))
        trunk_dim = int(getattr(cfg.arch.deeponet, "trunk_dim", None) or getattr(cfg.arch.trunk, "layer_size", 128))
        branch_net = instantiate_arch(
            cfg=cfg.arch.branch,
            input_keys=[Key(key) for key in deeponet_branch_keys],
            output_keys=[Key("branch", branch_dim)],
        )
        trunk_net = instantiate_arch(
            cfg=cfg.arch.trunk,
            input_keys=[Key(key) for key in deeponet_trunk_keys],
            output_keys=[Key("trunk", trunk_dim)],
        )
        model = instantiate_arch(
            cfg=cfg.arch.deeponet,
            branch_net=branch_net,
            trunk_net=trunk_net,
            output_keys=key_cfg.fno_output_keys,
            branch_dim=branch_dim,
            trunk_dim=trunk_dim,
        )
    elif model_family == "rigno":
        # Custom in-repo Arch (not a physicsnemo arch group): instantiate directly,
        # reading hyperparameters from cfg.arch.rigno with sensible defaults.
        from ..rigno_arch import RIGNOArch

        rc = cfg.arch.rigno
        model = RIGNOArch(
            input_keys=key_cfg.fno_input_keys,
            output_keys=key_cfg.fno_output_keys,
            dimension=int(getattr(rc, "dimension", 3)),
            node_latent_size=int(getattr(rc, "node_latent_size", 64)),
            edge_latent_size=int(getattr(rc, "edge_latent_size", 64)),
            processor_steps=int(getattr(rc, "processor_steps", 6)),
            mlp_hidden_layers=int(getattr(rc, "mlp_hidden_layers", 1)),
            subsample_factor=float(getattr(rc, "subsample_factor", 8.0)),
            rmesh_levels=int(getattr(rc, "rmesh_levels", 4)),
            k_p2r=int(getattr(rc, "k_p2r", 4)),
            k_r2p=int(getattr(rc, "k_r2p", 4)),
            p_edge_masking=float(getattr(rc, "p_edge_masking", 0.5)),
            graph_seed=int(getattr(rc, "graph_seed", 0)),
            cache_dir=str(getattr(rc, "cache_dir", "outputs/rigno_graphs")),
            grad_checkpointing=bool(getattr(rc, "grad_checkpointing", False)),
            num_graph_variants=int(getattr(rc, "num_graph_variants", 1)),
            inference_edge_masking=float(getattr(rc, "inference_edge_masking", 0.0)),
            rnode_count=int(getattr(rc, "rnode_count", 0)),
            film_keys=list(getattr(rc, "film_keys", []) or []),
        )
    elif model_family == "rigno_refined":
        # RIGNO base + residual 3D-FNO refinement head, trained end-to-end (the
        # leakage-free version of the spatial corrector). Head params from
        # cfg.arch.refine_head; RIGNO params from cfg.arch.rigno as above.
        from ..rigno_arch import RIGNORefinedArch

        rc = cfg.arch.rigno
        hc = getattr(cfg.arch, "refine_head", None)
        base_ckpt = OmegaConf.select(cfg, "custom.base_init_ckpt", default=None) or None
        model = RIGNORefinedArch(
            input_keys=key_cfg.fno_input_keys,
            output_keys=key_cfg.fno_output_keys,
            head_width=int(getattr(hc, "width", 24)) if hc is not None else 24,
            head_modes=list(getattr(hc, "modes", [12, 12, 8])) if hc is not None else [12, 12, 8],
            head_layers=int(getattr(hc, "layers", 4)) if hc is not None else 4,
            head_type=str(getattr(hc, "type", "fno")) if hc is not None else "fno",
            head_padding=int(getattr(hc, "padding", 0)) if hc is not None else 0,
            base_init_ckpt=(str(base_ckpt) if base_ckpt else None) if not eval_only else None,
            dimension=int(getattr(rc, "dimension", 3)),
            node_latent_size=int(getattr(rc, "node_latent_size", 64)),
            edge_latent_size=int(getattr(rc, "edge_latent_size", 64)),
            processor_steps=int(getattr(rc, "processor_steps", 6)),
            mlp_hidden_layers=int(getattr(rc, "mlp_hidden_layers", 1)),
            subsample_factor=float(getattr(rc, "subsample_factor", 8.0)),
            rmesh_levels=int(getattr(rc, "rmesh_levels", 4)),
            k_p2r=int(getattr(rc, "k_p2r", 4)),
            k_r2p=int(getattr(rc, "k_r2p", 4)),
            p_edge_masking=float(getattr(rc, "p_edge_masking", 0.5)),
            graph_seed=int(getattr(rc, "graph_seed", 0)),
            cache_dir=str(getattr(rc, "cache_dir", "outputs/rigno_graphs")),
            num_graph_variants=int(getattr(rc, "num_graph_variants", 1)),
            inference_edge_masking=float(getattr(rc, "inference_edge_masking", 0.0)),
            rnode_count=int(getattr(rc, "rnode_count", 0)),
        )
    elif model_family == "gaot":
        # Geometry-Aware Operator Transformer (NeurIPS 2025), ported from GAOT-3D
        # onto the fixed structured grid: MAGNO encoder -> UViT -> MAGNO decoder.
        # Same grid-in/grid-out contract as rigno, so every downstream node (alpha
        # masking, spatial align, validators) is unchanged. Params from cfg.arch.gaot.
        from ..gaot_arch import GAOTArch

        gc = getattr(cfg.arch, "gaot", None)
        g = lambda name, default: default if gc is None else getattr(gc, name, default)
        model = GAOTArch(
            input_keys=key_cfg.fno_input_keys,
            output_keys=key_cfg.fno_output_keys,
            dimension=int(g("dimension", 3)),
            latent_tokens=[int(v) for v in g("latent_tokens", [48, 24, 16])],
            patch_size=int(g("patch_size", 2)),
            lifting_channels=int(g("lifting_channels", 32)),
            hidden_size=int(g("hidden_size", 256)),
            ffn_hidden_size=int(g("ffn_hidden_size", 1024)),
            num_layers=int(g("num_layers", 5)),
            num_heads=int(g("num_heads", 8)),
            num_kv_heads=int(g("num_kv_heads", 8)),
            atten_dropout=float(g("atten_dropout", 0.0)),
            positional_embedding=str(g("positional_embedding", "absolute")),
            use_long_range_skip=bool(g("use_long_range_skip", True)),
            mlp_hidden_layers=[int(v) for v in g("mlp_hidden_layers", [64, 64, 64])],
            projection_channels=int(g("projection_channels", 256)),
            neighbor_strategy=str(g("neighbor_strategy", "knn")),
            k_encoder=int(g("k_encoder", 8)),
            k_decoder=int(g("k_decoder", 8)),
            gno_radius=float(g("gno_radius", 0.15)),
            max_neighbors=int(g("max_neighbors", 0)),
            decoder_stencil=str(g("decoder_stencil", "knn")),
            decoder_kernel=str(g("decoder_kernel", "agno")),
            neighbor_metric=(None if g("neighbor_metric", None) is None
                             else [float(v) for v in g("neighbor_metric", None)]),
            use_attention=bool(g("use_attention", True)),
            attention_type=str(g("attention_type", "cosine")),
            transform_type=str(g("transform_type", "linear")),
            use_geoembed_encoder=bool(g("use_geoembed_encoder", False)),
            use_geoembed_decoder=bool(g("use_geoembed_decoder", False)),
            kernel_use_displacement=bool(g("kernel_use_displacement", False)),
            use_pointwise_skip=bool(g("use_pointwise_skip", False)),
            # Physical output bounds, e.g. {"T": [0.0, 1.3]}. OmegaConf hands these back
            # as ListConfig, which GAOTArch would store without converting, so coerce
            # to plain floats here.
            output_bounds={
                str(k): [None if b is None else float(b) for b in v]
                for k, v in dict(g("output_bounds", {}) or {}).items()
            },
            clamp_fraction_outputs=bool(g("clamp_fraction_outputs", True)),
            pointwise_skip_channels=int(g("pointwise_skip_channels", 64)),
            embedding_method=str(g("embedding_method", "statistical")),
            cache_dir=str(g("cache_dir", "outputs/gaot_graphs")),
        )
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    if predict_delta:
        # Reparametrize as u_{t+1} = u_t + f(x): add the fed-back *_prev state to
        # the base output so the net predicts only the increment. Family-agnostic
        # wrapper; downstream nodes still see the absolute [T, alpha] outputs.
        from ..rigno_arch import DeltaArch

        model = DeltaArch(
            model,
            state_keys=list(key_cfg.str_fno_output_keys),
            state_prev_suffix=state_prev_suffix,
        )
        print(f"predict_delta: wrapped {model_family} in DeltaArch (u_t+1 = u_t + f) for {model._delta_keys}")

    print(
        f"Instatiating the Architecture with input keys: {key_cfg.fno_input_keys} "
        f"and output keys: {key_cfg.fno_output_keys}"
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total {model_family} parameters: {total_params:,}, trainable: {trainable_params:,}")
    if eval_only:
        model.eval()
    print(f"Network:\n {model}")

    # Per-case melt window when the batch carries mat_Ts/mat_Tl, matching the target
    # (preprocessing_datasets) and the eval metric (postprocessing.derive_fl_masked).
    # Declared as Node INPUTS only when present: naming a key the graph cannot supply
    # fails the whole run, and single-material configs have neither channel.
    _fl_keys = [Key("T")]
    _ts_scale = _tl_scale = None
    _declared = set(getattr(key_cfg, "str_fno_input_keys", []) or [])
    if {"mat_Ts", "mat_Tl"} <= _declared:
        _fl_keys += [Key("mat_Ts"), Key("mat_Tl")]
        _ts_scale = float(ref_scales.get("mat_Ts", 1.0)) / float(ref_scales["T"])
        _tl_scale = float(ref_scales.get("mat_Tl", 1.0)) / float(ref_scales["T"])
        print("[fl] graph node: PER-CASE melt window from mat_Ts/mat_Tl")
    fl_module = fl_Node(
        T_s=material_params["T_s"] / ref_scales["T"],
        T_l=material_params["T_l"] / ref_scales["T"],
        ts_scale=_ts_scale, tl_scale=_tl_scale,
    )
    fl_node = Node(
        inputs=_fl_keys,
        outputs=[Key("fl")],
        evaluate=fl_module,
        name="liquid_fraction",
    )

    align_node = None
    if model_family == "unet":
        ref_key = key_cfg.fno_input_keys[0]
        for candidate in ("x", "y", "z"):
            if candidate in key_cfg.str_fno_input_keys:
                ref_key = Key(candidate)
                break
        align_node = Node(
            inputs=key_cfg.fno_output_keys + [ref_key],
            outputs=key_cfg.fno_output_keys,
            evaluate=SpatialAlignNode(keys=key_cfg.fno_output_keys, ref_key=ref_key),
            name="unet_grid_align",
        )

    if alpha_masking:
        # T_dot (dT/dt aux output) must NOT be alpha-masked: dT/dt is nonzero in the
        # heating substrate outside the melt. Masking is a no-op filter when absent.
        mask_str = [k for k in key_cfg.str_fno_output_keys if k != "T_dot"]
        mask_keys = [Key(k) for k in mask_str]
        # Per-case fill (custom.per_case_t_boil): the node declares `mat_Tv` as an input so
        # the graph hands it the channel from the dataset's invar -- the FNO's own inputs
        # are untouched, so a material-blind model never sees it. Off by default: the
        # fill is part of the target (see utils.t_fill_field).
        _per_case_fill = bool(OmegaConf.select(cfg, "custom.per_case_t_boil", default=False))
        _mask_extra_inputs = [Key("mat_Tv")] if _per_case_fill else []
        if _per_case_fill:
            print("[t_boil] graph node: PER-CASE mask fill from mat_Tv", flush=True)
        alpha_mask_module = MaskNode(
            keys=mask_str + ["fl"],
            threshold=0.5,
            d=0.0,
            k=20,
            t_fill=boiling_fill_value(material_params, ref_scales),
            t_fill_key="mat_Tv" if _per_case_fill else None,
            t_fill_scale=float(ref_scales.get("mat_Tv", 1.0)) / float(ref_scales["T"]),
        )
        _mask_warmup_step = mask_warmup_start_step(
            eval_only=eval_only,
            initialization_network_dir=OmegaConf.select(cfg, "initialization_network_dir", default=""),
            configured=OmegaConf.select(cfg, "custom.mask_warmup_steps", default=None),
        )
        alpha_mask_node = Node(
            inputs=mask_keys + [Key("fl")] + _mask_extra_inputs,
            outputs=mask_keys + [Key("fl")],
            # Warmup avoids hard masking too early when training from scratch;
            # a warm start is not from scratch, so it skips the warmup.
            evaluate=WarmupNode(alpha_mask_module, start_step=_mask_warmup_step),
            name="alpha_mask",
        )
        nn_nodes = [model.make_node("u_network")]
        if align_node is not None:
            nn_nodes.append(align_node)
        nn_nodes.extend([fl_node, alpha_mask_node])
        if Key("u") in key_cfg.fno_output_keys and Key("v") in key_cfg.fno_output_keys and Key("w") in key_cfg.fno_output_keys:
            vel_fl_mask = MaskNode(keys=["u", "v", "w"], threshold=0.5, d=0.0, k=20, mask_str_key="fl")
            vel_fl_mask_node = Node(
                inputs=[Key("u"), Key("v"), Key("w"), Key("fl")],
                outputs=[Key("u"), Key("v"), Key("w")],
                evaluate=WarmupNode(vel_fl_mask, start_step=_mask_warmup_step),
                name="velocity_fl_mask",
            )
            nn_nodes.append(vel_fl_mask_node)
    else:
        nn_nodes = [model.make_node("u_network")]
        if align_node is not None:
            nn_nodes.append(align_node)
        nn_nodes.append(fl_node)
    print(nn_nodes)

    energy_nodes = None
    if pde_loss:
        print("Using PDE loss")
        from deprecated.equations import EnergyConservation  # deprecated physics loss
        energy_eq = EnergyConservation(
            material_params=material_params,
            laser_params=laser_params,
            ref_scales=ref_scales,
            dim=3,
            laser_frame=False,
            laser_vol_source=True,
            liquid_fraction=True,
            solid_fraction=True,
        )
        energy_nodes = nn_nodes + energy_eq.make_nodes()

    vof_nodes = None
    if vof_pde_loss:
        print("Using VOF PDE loss")
        from deprecated.equations import VOF  # deprecated physics loss
        vof_nodes = nn_nodes + VOF(weight=1.0).make_nodes()

    diff_nodes = None
    if diff_loss:
        print("Using PDE loss with diffusion node")
        from deprecated.equations import EnergyDiffusionTerm  # deprecated physics loss
        a, b = material_params["K"]["coeffs"]
        diffusion_term = EnergyDiffusionTerm(
            dx=dx,
            dy=dy,
            dz=dz,
            a=float(a),
            b=float(b),
            kappa_g=float(material_params["K_g"]),
            Diff_weight=diff_weight,
        )
        diff_node = Node(
            inputs=("T", "alpha"),
            outputs=("T_diffTerm",),
            evaluate=diffusion_term,
            name="energy_diffusion",
        )
        diff_nodes = nn_nodes + [diff_node]
    if not pde_loss and not vof_pde_loss and not diff_loss:
        print(f"Not using PDE loss, data-only {model_family} run")

    return ModelBundle(model=model, nn_nodes=nn_nodes, energy_nodes=energy_nodes, vof_nodes=vof_nodes, diff_nodes=diff_nodes)


def build_domain(
    cfg: PhysicsNeMoConfig,
    model_bundle: ModelBundle,
    datasets: DatasetBundle,
    run_cfg: RunConfig,
) -> Domain:
    """Build the training/validation domain for active constraints."""

    domain = Domain()
    if run_cfg.data_layout == "grid":
        print("Using structured grid constraint")
        constraint_kwargs: dict[str, Any] = {}
        # SupervisedGridConstraint defaults to num_workers=0, which decompresses
        # one case per step on the main process while the GPU waits. Workers
        # overlap that with training and keep global shuffling intact.
        loader_workers = dataloader_workers(run_cfg)
        if loader_workers > 0:
            print(f"Using {loader_workers} dataloader workers for the training constraint")
            constraint_kwargs["num_workers"] = loader_workers
        grad_w = float(OmegaConf.select(cfg, "custom.gradient_loss_weight", default=0.0) or 0.0)
        edge_dw = float(OmegaConf.select(cfg, "custom.edge_downweight", default=0.0) or 0.0)
        melt_dice_w = float(OmegaConf.select(cfg, "custom.melt_dice_weight", default=0.0) or 0.0)
        melt_w = float(OmegaConf.select(cfg, "custom.melt_weight", default=0.0) or 0.0)
        melt_tau = float(OmegaConf.select(cfg, "custom.melt_tau", default=0.05) or 0.05)
        melt_warmup = int(OmegaConf.select(cfg, "custom.melt_loss_warmup_steps", default=0) or 0)
        melt_ramp = int(OmegaConf.select(cfg, "custom.melt_loss_ramp_steps", default=0) or 0)
        tail_w = float(OmegaConf.select(cfg, "custom.tail_weight", default=0.0) or 0.0)
        tail_k = int(OmegaConf.select(cfg, "custom.tail_k", default=256) or 256)
        # Velocity outputs (custom.infer_velocities) are scored pool-masked with a
        # detached warmup; defaults per the 2026-08-09 design: engage at 4k, ramp 1k.
        infer_vel = bool(OmegaConf.select(cfg, "custom.infer_velocities", default=False))
        vel_warmup = int(OmegaConf.select(cfg, "custom.vel_warmup_steps", default=4000) or 0)
        vel_ramp = int(OmegaConf.select(cfg, "custom.vel_ramp_steps", default=1000) or 0)
        vel_tau = float(OmegaConf.select(cfg, "custom.vel_tau", default=0.05) or 0.05)
        # edge_downweight and the melt-focus terms need GridGradientLoss too (the
        # default PointwiseLossNorm ignores any per-voxel weight), so force it on
        # whenever any of them is set -- and whenever velocities are predicted,
        # because only GridGradientLoss knows to pool-mask them.
        if grad_w > 0.0 or edge_dw > 0.0 or melt_dice_w > 0.0 or melt_w > 0.0 or tail_w > 0.0 or infer_vel:
            from ...custom_utils.losses import GridGradientLoss

            print(
                f"Using GridGradientLoss with gradient_weight={grad_w} edge_downweight={edge_dw} "
                f"melt_dice_weight={melt_dice_w} melt_weight={melt_w} "
                f"tail_weight={tail_w} tail_k={tail_k} "
                f"warmup={melt_warmup} ramp={melt_ramp}"
                + (f" vel_warmup={vel_warmup} vel_ramp={vel_ramp} vel_tau={vel_tau}" if infer_vel else "")
            )
            constraint_kwargs["loss"] = GridGradientLoss(
                gradient_weight=grad_w,
                edge_downweight=edge_dw,
                melt_dice_weight=melt_dice_w,
                melt_weight=melt_w,
                tail_weight=tail_w,
                tail_k=tail_k,
                melt_tau=melt_tau,
                warmup_steps=melt_warmup,
                ramp_steps=melt_ramp,
                t_solidus=run_cfg.material_params["T_s"] / run_cfg.ref_scales["T"],
                t_liquidus=run_cfg.material_params["T_l"] / run_cfg.ref_scales["T"],
                vel_warmup_steps=vel_warmup,
                vel_ramp_steps=vel_ramp,
                vel_tau=vel_tau,
            )
        unroll_steps = int(getattr(run_cfg, "autoregressive_unroll_steps", 1))
        if getattr(run_cfg, "autoregressive_enabled", False) and unroll_steps > 1:
            from ...custom_utils.ar_constraints import UnrolledGridConstraint

            print(
                f"Using UnrolledGridConstraint (unroll_steps={unroll_steps}, "
                f"detach_grad={run_cfg.autoregressive_unroll_detach_grad})"
            )
            data_constraint = UnrolledGridConstraint(
                nodes=model_bundle.nn_nodes,
                dataset=datasets.data_ds,
                batch_size=cfg.batch_size.grid,
                context_keys=run_cfg.autoregressive_context_keys,
                state_keys=run_cfg.autoregressive_state_keys,
                state_prev_suffix=run_cfg.autoregressive_state_prev_suffix,
                unroll_steps=unroll_steps,
                detach_grad=run_cfg.autoregressive_unroll_detach_grad,
                **constraint_kwargs,
            )
        else:
            try:
                tc_w = max(0.0, float(os.environ.get("PINO_TC_WEIGHT", "0")))
            except ValueError:
                tc_w = 0.0
            if tc_w > 0.0:
                from ...custom_utils.ar_constraints import TemporalConsistencyGridConstraint
                from .preprocessing_datasets import tc_advance_keys

                print(f"Using TemporalConsistencyGridConstraint (tc_weight={tc_w})")
                data_constraint = TemporalConsistencyGridConstraint(
                    nodes=model_bundle.nn_nodes,
                    dataset=datasets.data_ds,
                    batch_size=cfg.batch_size.grid,
                    advance_keys=tc_advance_keys(run_cfg.keys.str_fno_input_keys),
                    tc_weight=tc_w,
                    **constraint_kwargs,
                )
            else:
                data_constraint = SupervisedGridConstraint(
                    nodes=model_bundle.nn_nodes,
                    dataset=datasets.data_ds,
                    batch_size=cfg.batch_size.grid,
                    **constraint_kwargs,
                )
        domain.add_constraint(data_constraint, name="data")
    else:
        print("Using pointwise data constraint")
        if datasets.ds_input is None or datasets.ds_output is None:
            raise ValueError("Pointwise training requires eager ds_input and ds_output dictionaries.")
        if run_cfg.model_family == "deeponet":
            data_constraint = DeepONetConstraint.from_numpy(
                nodes=model_bundle.nn_nodes,
                invar=datasets.ds_input,
                outvar=datasets.ds_output,
                batch_size=cfg.batch_size.grid,
            )
        else:
            data_constraint = PointwiseConstraint.from_numpy(
                nodes=model_bundle.nn_nodes,
                invar=datasets.ds_input,
                outvar=datasets.ds_output,
                batch_size=cfg.batch_size.grid,
            )
            # PointwiseConstraint.save_batch writes VTK unconditionally (ignores
            # save_filetypes) via var_to_polyvtk, which crashes on grid-derived
            # pointwise data (e.g. fully_connected). Constraint recording is
            # debug-only, so disable it when file output is off.
            if not str(OmegaConf.select(cfg, "save_filetypes", default="") or ""):
                data_constraint.save_batch = lambda *a, **k: None
        domain.add_constraint(data_constraint, name="data")

    if run_cfg.time_as_batch:
        validation_examples = 2
    else:
        validation_examples = cfg.batch_size.validation

    if run_cfg.pde_loss and model_bundle.energy_nodes is not None and datasets.energy_ds is not None and datasets.val_energy_ds is not None:
        energy_constraint = SupervisedGridConstraint(
            nodes=model_bundle.energy_nodes,
            dataset=datasets.energy_ds,
            batch_size=cfg.batch_size.grid,
        )
        domain.add_constraint(energy_constraint, name="energy")
        if not run_cfg.fast_training:
            energy_val = StreamingGridValidator(
                nodes=model_bundle.energy_nodes,
                dataset=datasets.val_energy_ds,
                batch_size=cfg.batch_size.validation,
                plotter=_validator_plotter(cfg, lambda: CustomSlicePlotter(
                    xy_idx=run_cfg.nz_effective - 3,
                    yz_idx=run_cfg.nx_effective // 2,
                    xz_idx=run_cfg.ny_effective // 2,
                    variable="energy",
                    n_examples=validation_examples,
                    ref_scales=run_cfg.ref_scales,
                    debug=False,
                )),
                requires_grad=True,
                dump_files=_validator_dumps_enabled(cfg),
            )
            domain.add_validator(energy_val, "energy")

    if run_cfg.vof_pde_loss and model_bundle.vof_nodes is not None and datasets.vof_ds is not None and datasets.val_vof_ds is not None:
        vof_constraint = SupervisedGridConstraint(
            nodes=model_bundle.vof_nodes,
            dataset=datasets.vof_ds,
            batch_size=cfg.batch_size.grid,
        )
        domain.add_constraint(vof_constraint, name="VOF")
        if not run_cfg.fast_training:
            vof_val = StreamingGridValidator(
                nodes=model_bundle.vof_nodes,
                dataset=datasets.val_vof_ds,
                batch_size=cfg.batch_size.validation,
                plotter=_validator_plotter(cfg, lambda: CustomSlicePlotter(
                    xy_idx=run_cfg.nz_effective - 3,
                    yz_idx=run_cfg.nx_effective // 2,
                    xz_idx=run_cfg.ny_effective // 2,
                    variable="VOF",
                    n_examples=validation_examples,
                    ref_scales=run_cfg.ref_scales,
                    debug=False,
                )),
                requires_grad=True,
                dump_files=_validator_dumps_enabled(cfg),
            )
            domain.add_validator(vof_val, "VOF")

    if run_cfg.diff_loss and model_bundle.diff_nodes is not None and datasets.diff_ds is not None and datasets.val_diff_ds is not None:
        diff_constraint = SupervisedGridConstraint(
            nodes=model_bundle.diff_nodes,
            dataset=datasets.diff_ds,
            batch_size=cfg.batch_size.grid,
        )
        domain.add_constraint(diff_constraint, name="diffusion")
        if not run_cfg.fast_training:
            diff_val = StreamingGridValidator(
                nodes=model_bundle.diff_nodes,
                dataset=datasets.val_diff_ds,
                batch_size=cfg.batch_size.validation,
                plotter=_validator_plotter(cfg, lambda: CustomSlicePlotter(
                    xy_idx=run_cfg.nz_effective - 3,
                    yz_idx=run_cfg.nx_effective // 2,
                    xz_idx=run_cfg.ny_effective // 2,
                    variable="T_diffTerm",
                    n_examples=validation_examples,
                    ref_scales=run_cfg.ref_scales,
                    debug=False,
                )),
                requires_grad=True,
                dump_files=_validator_dumps_enabled(cfg),
            )
            domain.add_validator(diff_val, "diffusion")

    if not run_cfg.fast_training:
        if run_cfg.autoregressive_enabled and run_cfg.autoregressive_mode == "masked_patch":
            plotter = _validator_plotter(cfg, lambda: TemporalPatchPlotter(
                variables=run_cfg.keys.str_fno_output_keys,
                state_prev_suffix=run_cfg.autoregressive_state_prev_suffix,
                n_examples=validation_examples,
            ))
            data_val = StreamingGridValidator(
                nodes=model_bundle.nn_nodes,
                dataset=datasets.val_data_ds,
                batch_size=cfg.batch_size.validation,
                plotter=plotter,
                requires_grad=False,
                dump_files=_validator_dumps_enabled(cfg),
            )
        elif run_cfg.data_layout == "grid":
            plotter = _validator_plotter(cfg, lambda: CustomSlicePlotter(
                xy_idx=run_cfg.nz_effective - 3,
                yz_idx=run_cfg.nx_effective // 2,
                xz_idx=run_cfg.ny_effective // 2,
                variables=run_cfg.keys.str_fno_output_keys,
                contours={"T": run_cfg.material_params["T_s"] / run_cfg.ref_scales["T"], "alpha": 0.5},
                n_examples=validation_examples,
                ref_scales=run_cfg.ref_scales,
                debug=False,
            ))
            data_val = StreamingGridValidator(
                nodes=model_bundle.nn_nodes,
                dataset=datasets.val_data_ds,
                batch_size=cfg.batch_size.validation,
                plotter=plotter,
                requires_grad=False,
                dump_files=_validator_dumps_enabled(cfg),
            )
        else:
            if datasets.ds_input_val is None or datasets.ds_output_val is None:
                raise ValueError("Pointwise validation requires eager ds_input_val and ds_output_val dictionaries.")
            data_val = PointwiseValidator(
                nodes=model_bundle.nn_nodes,
                invar=datasets.ds_input_val,
                true_outvar=datasets.ds_output_val,
                batch_size=cfg.batch_size.validation,
                plotter=None,
                requires_grad=False,
            )
        domain.add_validator(data_val, "data")

    return domain


def _maybe_enable_separate_head_lr(cfg: PhysicsNeMoConfig) -> None:
    """When `custom.head_lr` is set, give the refinement head its own optimizer
    LR via param groups (base params at optimizer.lr, head params at head_lr).

    Lion is sign-based, so per-parameter LR cannot be faked with gradient scaling
    — it needs real param groups. physicsnemo builds the optimizer from a flat
    `model.parameters()`, so we intercept `instantiate_optim` (used once at
    solve() setup) and split by parameter name.
    """
    head_lr = OmegaConf.select(cfg, "custom.head_lr", default=None)
    if head_lr is None:
        return
    head_lr = float(head_lr)
    import copy
    import hydra
    import physicsnemo.sym.trainer as _pnt

    def _grouped_instantiate_optim(cfg_, model, verbose=False):
        optim_cfg = copy.deepcopy(cfg_.optimizer)
        del optim_cfg._params_
        head, base = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (head if ".head." in name or name.startswith("head.") else base).append(p)
        groups = [{"params": base}, {"params": head, "lr": head_lr}]
        # Instantiate the optimizer class directly: hydra.utils.instantiate would
        # convert the param-group dicts into DictConfigs, which the optimizer rejects.
        opt_kwargs = OmegaConf.to_container(optim_cfg, resolve=True)
        opt_cls = hydra.utils.get_class(opt_kwargs.pop("_target_"))
        print(f"[rigno_refined] separate head LR: base lr={opt_kwargs.get('lr')} "
              f"({len(base)} tensors), head lr={head_lr} ({len(head)} tensors)")
        return opt_cls(groups, **opt_kwargs)

    _pnt.instantiate_optim = _grouped_instantiate_optim


def solve_domain(
    cfg: PhysicsNeMoConfig, domain: Domain, debug: bool, model=None
) -> None:
    """Run solver execution with optional anomaly detection enabled.

    `model`, when given, enables the best-validation checkpoint keeper: physicsnemo
    overwrites `u_network.0.pth` every save, so a run that trains well and then diverges
    ends with the wreckage. See best_checkpoint.py for the run that motivated it.
    """

    from .amp_guard import amp_collapse_guard
    from .best_checkpoint import best_val_checkpoint
    from .validation_beambox import beam_box_validation

    _maybe_enable_separate_head_lr(cfg)
    solver = Solver(cfg, domain)
    # An aggregator with parameters (homoscedastic log_var) lands in a param
    # group ADDED AFTER the scheduler was built; OneCycleLR then raises
    # KeyError('initial_lr') on its first step (killed both gaot_spcal arms at
    # step 0, 2026-08-10). The optimizer only exists inside _train_loop, so the
    # repair hooks load_network -- the seam between add_param_group and the
    # first scheduler step (a direct solver.optimizer read here crashed the
    # first resubmit with AttributeError).
    from ...custom_utils.utils import attach_late_param_group_repair

    attach_late_param_group_repair(solver)
    # BUG-43: a collapsed AMP scaler skips every optimizer step, and the run
    # would otherwise finish and register a model holding its init weights.
    network_dir = OmegaConf.select(cfg, "network_dir", default=".")
    # Beam-box error during validation: whole-domain MAE describes the quiescent ~99% of
    # the substrate, so an arm's melt-region behaviour is otherwise invisible until eval.
    beambox_on = bool(OmegaConf.select(cfg, "custom.validation_beambox", default=True))
    with amp_collapse_guard(cfg), best_val_checkpoint(cfg, model, network_dir), \
            beam_box_validation(
                beambox_on,
                at_origin=bool(OmegaConf.select(cfg, "custom.moving_ref_frame", default=False)),
            ), \
            torch.autograd.set_detect_anomaly(debug):
        solver.solve()
    # Measured peak beside the checkpoints: the next submission's booking starts here.
    record_vram_peak(network_dir)
