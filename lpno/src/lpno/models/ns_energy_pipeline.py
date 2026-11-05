"""High-level training/evaluation orchestration facade for NS/Energy PINO."""

import os
from pathlib import Path
from typing import Any

import wandb

try:
    # Registers torch.ops.physicsnemo_ext.* (the custom AMP GradScaler update
    # kernel). physicsnemo.sym ships the .so but never imports it, so fp16
    # training crashes at the first scaler.update() otherwise. This must stay
    # OUT of models/__init__.py: the entrypoints import that package before
    # the CUDA_VISIBLE_DEVICES bootstrap, and pulling torch in early breaks
    # custom.cuda_device routing (see cuda_bootstrap.py).
    import physicsnemo.sym.physicsnemo_ext  # noqa: F401
except ImportError:
    # amp.enabled=false paths work without the extension.
    pass

from ..custom_utils.utils import aggregate_fold_stats
from .pipeline.h_norm_record import write_training_record as write_h_norm_record
from .pipeline.input_fields_record import write_input_fields_record

from .pipeline import ( # pyright: ignore[reportMissingImports]
    DatasetBundle,
    InferenceMetrics,
    KeyConfig,
    ModelBundle,
    RunConfig,
    SplitSpec,
    _collect_sim_param_overrides,
    build_datasets,
    build_domain,
    build_model_and_nodes,
    configure_device,
    get_splits,
    load_and_preprocess_data,
    load_cfg_and_paths,
    print_dict_shapes,
    run_lazy_dataset_metrics,
    run_masked_patch_metrics,
    run_autoregressive_rollout_and_metrics,
    run_lazy_autoregressive_rollout_and_metrics,
    run_inference_and_metrics,
    run_pointwise_inference_and_metrics,
    sample_count,
    seed_all,
    setup_wandb_run,
    solve_domain,
    write_val_error_csv,
)

# pyright: reportAttributeAccessIssue=false
# pyright: reportUnboundVariable=false
# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportOptionalSubscript=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportOptionalMemberAccess=false


def assert_init_checkpoint_loadable(model: Any, init_dir: str | Path | None) -> None:
    """Refuse to start a warm start that cannot actually load its donor.

    physicsnemo wraps the ``initialization_network_dir`` load in a bare
    ``except`` (`sym/trainer.py`): on a state-dict mismatch it logs one red
    "Fail loading model:" line, resets the step counter, and **continues with
    randomly initialised weights**. The run then trains to completion and its
    loss curve looks entirely normal, so the failure surfaces only if somebody
    thinks to compare step-0 metrics against a known warm start.

    Measured 2026-08-24 (arm 65831): a donor trained when
    ``config_flow3D_RIGNO_randompath_onecycle`` defaulted to
    ``num_graph_variants: 1``, loaded into a run that resolved the config after
    that default moved to 8. Step-0 beam-box MAE 1592 K where a working warm
    start begins at 42-55 K. 6000 steps spent answering nothing.

    A no-op when no warm start was requested, so ordinary runs are unaffected.
    """
    if init_dir is None or not str(init_dir).strip():
        return

    ckpt_dir = Path(str(init_dir).strip())
    ckpt_name = getattr(model, "checkpoint_filename", None)
    if not ckpt_name:
        raise RuntimeError(
            "Model checkpoint filename is not initialized, so the warm-start "
            "check would verify nothing. Ensure model.make_node(...) runs first."
        )

    ckpt_path = ckpt_dir / str(ckpt_name)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Warm start requested but the donor checkpoint is missing: {ckpt_path}. "
            "physicsnemo would warn and train from RANDOM weights instead. Check "
            "that the init model's artifacts were staged."
        )

    try:
        model.load(str(ckpt_dir), map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - the underlying error is the useful part
        raise RuntimeError(
            f"Warm start cannot load '{ckpt_path}': the donor's architecture differs "
            "from this run's. physicsnemo would log 'Fail loading model' and train "
            "from RANDOM weights, producing a normal-looking curve that answers "
            "nothing.\n"
            "Pin every architecture key to the DONOR's values -- the donor is what "
            "constrains a warm start, not whatever you are comparing against. A "
            "config default that moved between the donor's run and this one is the "
            "usual cause, and a queued job resolves its config at RUN time, not at "
            "submit time.\n"
            f"Underlying error: {exc}"
        ) from exc


def _use_best_checkpoint(cfg: Any) -> bool:
    """Read `custom.use_best_checkpoint` from a cfg that may not be a DictConfig.

    Plain attribute access rather than `OmegaConf.select`: the pipeline dispatch
    tests pass a `types.SimpleNamespace`, which OmegaConf refuses.
    """
    custom = getattr(cfg, "custom", None)
    return bool(getattr(custom, "use_best_checkpoint", False)) if custom is not None else False


def _describe_archived_best(ckpt_dir: Path) -> str:
    """`best_checkpoint.json`'s step and metric as a phrase, or '' if absent."""
    meta = ckpt_dir / "best_checkpoint.json"
    if not meta.is_file():
        return ""
    try:
        import json

        payload = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a broken sidecar must not stop an eval
        return ""
    step = payload.get("step")
    key = payload.get("metric", "metric")     # the NAME; see best_checkpoint.py
    value = payload.get("value")              # the number
    return f"{key}={value} at step {step}"


def _assert_eval_checkpoint_loadable(
    model: Any, checkpoint_dir: str | Path, use_best_checkpoint: bool = False
) -> None:
    """Fail-fast in eval mode when model checkpoint is missing or incompatible.

    `best_checkpoint.py` archives `u_network.best.pth` every run because
    physicsnemo's `u_network.0.pth` holds the LAST weights, and a run that trains
    well then diverges ends with the wreckage. Until 2026-08-25 nothing read it:
    no eval path and no evaluation_profile key, so the safety net was write-only.
    Model 68163 records mae_T 0.012049 (36.1 K) at step 26000 and scored 558 K
    from its last checkpoint -- exit 0, populated metrics.json, nothing flagged.

    `use_best_checkpoint` is opt-in rather than default-true on purpose: flipping
    it silently changes every historical number, which is a campaign decision and
    not this function's to make. What is NOT optional is saying so -- an eval that
    ignores an archived better checkpoint now announces it, with the recorded
    metric, since an eval far worse than the model's own best is the tell.

    Reaching it from aiida-pino (2026-08-25): there is no `use_best_checkpoint`
    evaluation-profile key -- aiida-pino refuses unknown keys at submit time, so
    asking for one is a refusal rather than a no-op. Pass it as
    `hydra_overrides=["++custom.use_best_checkpoint=true"]`. The `++` is required:
    the key is declared in no config, so a plain `custom.use_best_checkpoint=true`
    fails struct validation, and `remap_plus_args_to_custom` leaves an
    already-`custom.`-prefixed argument alone rather than doubling it.
    """

    ckpt_dir = Path(checkpoint_dir)
    ckpt_name = getattr(model, "checkpoint_filename", None)
    if not ckpt_name:
        raise RuntimeError(
            "Model checkpoint filename is not initialized. Ensure model.make_node(...) is called before eval load checks."
        )

    best_path = ckpt_dir / "u_network.best.pth"
    if use_best_checkpoint:
        if best_path.is_file():
            ckpt_name = best_path.name
        else:
            print(
                f"WARNING: use_best_checkpoint=true but '{best_path.name}' is not in "
                f"{ckpt_dir}; scoring '{ckpt_name}' (the last weights) instead. Runs "
                "trained before best-checkpoint keeping, or with "
                "custom.best_checkpoint=false, archive none.",
                flush=True,
            )
    elif best_path.is_file():
        described = _describe_archived_best(ckpt_dir)
        print(
            f"WARNING: this model archived '{best_path.name}'"
            + (f" ({described})" if described else "")
            + f", but the eval is scoring '{ckpt_name}', which holds the LAST weights, "
            "not the best. A run that diverged after its best step is scored as the "
            "wreckage -- for example model 68163, a DIFFERENT model quoted here as "
            "an illustration, which recorded 36.1 K and scored 558 K at exit 0. To "
            "score the best weights instead, pass "
            "hydra_overrides=['++custom.use_best_checkpoint=true'] -- NOT an "
            "evaluation-profile key, which aiida-pino refuses at submit time, and "
            "NOT a plain 'custom.use_best_checkpoint=true', which fails struct "
            "validation because the key is not declared in any config.",
            flush=True,
        )

    ckpt_path = ckpt_dir / str(ckpt_name)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Evaluation checkpoint not found: {ckpt_path}. "
            "Verify network_dir points to the trained run output."
        )

    try:
        # `model.load` reads `self.checkpoint_filename`, so point it at the file we
        # resolved above and put it back afterwards -- the model object outlives
        # this call and a mutated filename would follow it.
        original_name = model.checkpoint_filename
        try:
            model.checkpoint_filename = str(ckpt_name)
            model.load(str(ckpt_dir), map_location="cpu")
        finally:
            model.checkpoint_filename = original_name
    except Exception as exc:  # noqa: BLE001 - preserve exact load failure details for user.
        raise RuntimeError(
            f"Failed to load evaluation checkpoint '{ckpt_path}'. "
            "Current model architecture/config likely differs from the training run."
        ) from exc


def _run_standard(cfg, run_cfg: RunConfig) -> None:
    """Execute the standard single-split train/eval flow."""

    setup_wandb_run(cfg, run_cfg.wb_project, run_cfg.run_name)

    var = load_and_preprocess_data(
        run_cfg,
        input_dir=run_cfg.data_directory_path,
        h_norm_mode="max",
        set_runtime_params=True,
    )
    if isinstance(var, dict) and not run_cfg.eval_only:
        # Eager loader: the train pass just fitted (or took) the H_norm denominator and
        # stashed it on laser_params. Record it beside the weights for the evaluation
        # (the lazy path records its own in `_build_lazy_datasets`).
        write_h_norm_record(
            run_cfg,
            getattr(run_cfg, "h_norm_denom", None)
            or (getattr(run_cfg, "laser_params", None) or {}).get("h_norm_denom"),
            network_dir=getattr(cfg, "network_dir", None),
        )
        write_input_fields_record(run_cfg, network_dir=getattr(cfg, "network_dir", None))
    var_val_source = load_and_preprocess_data(
        run_cfg,
        input_dir=run_cfg.val_directory_path,
        h_norm_mode="max",
        set_runtime_params=False,
    )
    print_dict_shapes(var)

    n_samples = sample_count(var)
    n_val_samples = sample_count(var_val_source)
    splits = get_splits(
        mode="standard",
        n_samples=n_samples,
        seed=run_cfg.seed,
        val_n_samples=n_val_samples,
    )
    split = splits[0]

    datasets = build_datasets(
        var=var,
        key_cfg=run_cfg.keys,
        split=split,
        pde_loss=run_cfg.pde_loss,
        vof_pde_loss=run_cfg.vof_pde_loss,
        diff_loss=run_cfg.diff_loss,
        var_val_source=var_val_source,
        run_cfg=run_cfg,
    )
    model_bundle = build_model_and_nodes(
        cfg=cfg,
        key_cfg=run_cfg.keys,
        material_params=run_cfg.material_params,
        laser_params=run_cfg.laser_params,
        ref_scales=run_cfg.ref_scales,
        alpha_masking=run_cfg.alpha_masking,
        eval_only=run_cfg.eval_only,
        model_family=getattr(run_cfg, "model_family", "fno"),
        deeponet_branch_keys=getattr(run_cfg, "deeponet_branch_keys", []),
        deeponet_trunk_keys=getattr(run_cfg, "deeponet_trunk_keys", []),
        pde_loss=run_cfg.pde_loss,
        vof_pde_loss=run_cfg.vof_pde_loss,
        diff_loss=run_cfg.diff_loss,
        dx=run_cfg.dx,
        dy=run_cfg.dy,
        dz=run_cfg.dz,
        diff_weight=run_cfg.diff_weight,
        predict_delta=getattr(run_cfg, "autoregressive_predict_delta", False),
        state_prev_suffix=getattr(run_cfg, "autoregressive_state_prev_suffix", "_prev"),
    )
    if run_cfg.eval_only:
        # Weights are loaded here (model.load inside the assert). Skip the
        # physicsnemo Solver for eval: solver.solve() eager-loads the whole
        # data constraint into host RAM (~0.75 GB per full-frame case, so the
        # 100-case test set alone is ~75 GB) and OOMs the box. The metric
        # helpers below stream the lazy dataset through their own DataLoader and
        # only need the loaded model on-device, so build_domain/solve_domain are
        # pure overhead in eval mode.
        _assert_eval_checkpoint_loadable(
            model_bundle.model, cfg.network_dir,
            use_best_checkpoint=_use_best_checkpoint(cfg),
        )
        # An FNO scored on a finer grid than it trained on (fno_superres.py): rescale its padding and remap its spectral
        # modes onto the frequencies its training grid held (on by default, PINO_FNO_SUPERRES=0 turns it off);
        # PINO_DEBUG_FNO_DUMP=<file.npz> saves the first sample's encoder features stage by stage. No-op for other
        # families and for matching grids.
        _store = getattr(getattr(datasets, "val_data_ds", None), "store", None)
        if _store is not None and hasattr(_store, "spatial_shape"):
            from .pipeline.fno_superres import maybe_apply_fno_superres

            maybe_apply_fno_superres(model_bundle.model, cfg.network_dir, tuple(_store.spatial_shape())[:3],
                                     run_cfg=run_cfg, output_dir=Path("fno_eval"))
        if os.environ.get("PINO_DEBUG_FNO_DUMP", "").strip():
            from .pipeline.fno_superres import attach_fno_feature_dump

            attach_fno_feature_dump(model_bundle.model, os.environ["PINO_DEBUG_FNO_DUMP"].strip())
        import torch  # local: a module-level torch import breaks cuda_device routing (see header note)

        if torch.cuda.is_available():
            model_bundle.model.cuda()
    else:
        # Before any GPU time: a warm start that cannot load its donor would
        # otherwise train to completion from random weights (see the function).
        assert_init_checkpoint_loadable(
            model_bundle.model, getattr(cfg, "initialization_network_dir", None)
        )
        domain = build_domain(cfg=cfg, model_bundle=model_bundle, datasets=datasets, run_cfg=run_cfg)
        solve_domain(cfg=cfg, domain=domain, debug=run_cfg.debug,
                     model=getattr(model_bundle, "model", None))

    if run_cfg.eval_only:
        if getattr(run_cfg, "autoregressive_enabled", False):
            if run_cfg.autoregressive_mode == "masked_patch":
                run_masked_patch_metrics(
                    model=model_bundle.model,
                    dataset=datasets.val_data_ds,
                    run_cfg=run_cfg,
                    output_dir=Path("fno_eval"),
                )
            elif datasets.var_train is not None:
                run_autoregressive_rollout_and_metrics(
                    model=model_bundle.model,
                    trajectory_var=datasets.var_train,
                    run_cfg=run_cfg,
                    output_dir=Path("fno_eval"),
                )
            else:
                # Lazy dataset path: no eager trajectories exist. Stream the
                # rollout one case at a time (the AiiDA eval lane stages the
                # lazy arraydir, which crashed here with trajectory_var=None).
                if datasets.val_frame_reader is None:
                    raise ValueError(
                        "AR rollout eval on the lazy path needs the dataset bundle's "
                        "val_frame_reader (built by the lazy AR-rollout branch)."
                    )
                run_lazy_autoregressive_rollout_and_metrics(
                    model=model_bundle.model,
                    frame_reader=datasets.val_frame_reader,
                    case_indices=datasets.val_case_indices,
                    case_lengths=datasets.val_case_lengths,
                    run_cfg=run_cfg,
                    output_dir=Path("fno_eval"),
                )
        else:
            if getattr(run_cfg, "data_layout", "grid") == "pointwise":
                if datasets.ds_input_val is None or datasets.ds_output_val is None:
                    raise ValueError("Pointwise eval requires eager validation tensors.")
                run_pointwise_inference_and_metrics(
                    model=model_bundle.model,
                    ds_input=datasets.ds_input_val,
                    ds_output=datasets.ds_output_val,
                    run_cfg=run_cfg,
                    output_dir=Path("fno_eval"),
                )
            elif datasets.ds_input is None or datasets.ds_output is None:
                run_lazy_dataset_metrics(
                    model=model_bundle.model,
                    dataset=datasets.val_data_ds,
                    run_cfg=run_cfg,
                    output_dir=Path("fno_eval"),
                    metric_keys=list(run_cfg.keys.str_fno_output_keys) + ["fl"],
                )
            else:
                run_inference_and_metrics(
                    model=model_bundle.model,
                    ds_input=datasets.ds_input,
                    ds_output=datasets.ds_output,
                    run_cfg=run_cfg,
                    output_dir=Path("fno_eval"),
                )


def _run_kfold(cfg, run_cfg: RunConfig) -> None:
    """Execute k-fold training/evaluation and aggregate fold metrics."""

    var = load_and_preprocess_data(run_cfg, input_dir=run_cfg.data_directory_path, h_norm_mode="ref_scale")
    print_dict_shapes(var)

    n_samples = sample_count(var)
    splits = get_splits(
        mode="kfold",
        n_samples=n_samples,
        k_folds=run_cfg.k_folds,
        fold_index=run_cfg.fold_index,
        seed=run_cfg.seed,
    )
    if not splits:
        raise ValueError("No folds selected. Check k_folds and fold_index.")

    base_dir = run_cfg.network_base_dir
    base_name = run_cfg.run_name
    all_stats: list[dict[str, dict[str, float]]] = []
    fold_counts: list[int] = []
    val_error_rows: list[dict[str, object]] = []

    for split in splits:
        fold = split.fold
        cfg.network_dir = f"{base_dir}/fold_{fold}"
        fold_run_name = f"{base_name}_fold{fold}"

        setup_wandb_run(cfg, run_cfg.wb_project, fold_run_name)

        datasets = build_datasets(
            var=var,
            key_cfg=run_cfg.keys,
            split=split,
            pde_loss=run_cfg.pde_loss,
            vof_pde_loss=run_cfg.vof_pde_loss,
            diff_loss=run_cfg.diff_loss,
            run_cfg=run_cfg,
        )
        model_bundle = build_model_and_nodes(
            cfg=cfg,
            key_cfg=run_cfg.keys,
            material_params=run_cfg.material_params,
            laser_params=run_cfg.laser_params,
        ref_scales=run_cfg.ref_scales,
        alpha_masking=run_cfg.alpha_masking,
        eval_only=run_cfg.eval_only,
        model_family=getattr(run_cfg, "model_family", "fno"),
        deeponet_branch_keys=getattr(run_cfg, "deeponet_branch_keys", []),
        deeponet_trunk_keys=getattr(run_cfg, "deeponet_trunk_keys", []),
        pde_loss=run_cfg.pde_loss,
        vof_pde_loss=run_cfg.vof_pde_loss,
        diff_loss=run_cfg.diff_loss,
            dx=run_cfg.dx,
            dy=run_cfg.dy,
            dz=run_cfg.dz,
            diff_weight=run_cfg.diff_weight,
            predict_delta=getattr(run_cfg, "autoregressive_predict_delta", False),
            state_prev_suffix=getattr(run_cfg, "autoregressive_state_prev_suffix", "_prev"),
        )
        if run_cfg.eval_only:
            _assert_eval_checkpoint_loadable(
            model_bundle.model, cfg.network_dir,
            use_best_checkpoint=_use_best_checkpoint(cfg),
        )
        domain = build_domain(cfg=cfg, model_bundle=model_bundle, datasets=datasets, run_cfg=run_cfg)
        solve_domain(cfg=cfg, domain=domain, debug=run_cfg.debug,
                     model=getattr(model_bundle, "model", None))

        if getattr(run_cfg, "autoregressive_enabled", False):
            if run_cfg.autoregressive_mode == "masked_patch":
                metrics = run_masked_patch_metrics(
                    model=model_bundle.model,
                    dataset=datasets.val_data_ds,
                    run_cfg=run_cfg,
                    output_dir=Path(f"fold_{fold}"),
                )
            else:
                metrics = run_autoregressive_rollout_and_metrics(
                    model=model_bundle.model,
                    trajectory_var=datasets.var_val,
                    run_cfg=run_cfg,
                    output_dir=Path(f"fold_{fold}"),
                )
        else:
            if getattr(run_cfg, "data_layout", "grid") == "pointwise":
                if datasets.ds_input_val is None or datasets.ds_output_val is None:
                    raise ValueError("Pointwise eval requires eager validation tensors.")
                if datasets.var_val is None:
                    raise ValueError("Pointwise k-fold eval requires eager var_val tensors.")
                p_scale = run_cfg.ref_scales.get("P", 1.0)
                v_scale = run_cfg.ref_scales.get("V", 1.0)
                p_per_sample = datasets.var_val["P"][:, 0, ...].reshape(len(split.val_idx), -1)[:, 0] * p_scale
                v_per_sample = datasets.var_val["V_scan"][:, 0, ...].reshape(len(split.val_idx), -1)[:, 0] * v_scale
                metrics = run_pointwise_inference_and_metrics(
                    model=model_bundle.model,
                    ds_input=datasets.ds_input_val,
                    ds_output=datasets.ds_output_val,
                    run_cfg=run_cfg,
                    output_dir=Path(f"fold_{fold}"),
                    fold=fold,
                    val_idx=split.val_idx,
                    p_values=p_per_sample,
                    v_values=v_per_sample,
                )
            elif datasets.var_val is None or datasets.ds_input_val is None or datasets.ds_output_val is None:
                metrics = run_lazy_dataset_metrics(
                    model=model_bundle.model,
                    dataset=datasets.val_data_ds,
                    run_cfg=run_cfg,
                    output_dir=Path(f"fold_{fold}"),
                    metric_keys=list(run_cfg.keys.str_fno_output_keys) + ["fl"],
                )
            else:
                p_scale = run_cfg.ref_scales.get("P", 1.0)
                v_scale = run_cfg.ref_scales.get("V", 1.0)
                p_per_sample = datasets.var_val["P"][:, 0, ...].reshape(len(split.val_idx), -1)[:, 0] * p_scale
                v_per_sample = datasets.var_val["V_scan"][:, 0, ...].reshape(len(split.val_idx), -1)[:, 0] * v_scale
                metrics = run_inference_and_metrics(
                    model=model_bundle.model,
                    ds_input=datasets.ds_input_val,
                    ds_output=datasets.ds_output_val,
                    run_cfg=run_cfg,
                    output_dir=Path(f"fold_{fold}"),
                    relative_reference=var,
                    fold=fold,
                    val_idx=split.val_idx,
                    p_values=p_per_sample,
                    v_values=v_per_sample,
                )
        all_stats.append(metrics.errors)
        fold_counts.append(len(split.val_idx))
        val_error_rows.extend(metrics.val_rows)
        wandb.finish()

    if all_stats and run_cfg.fold_index == -1:
        aggregated_path = Path(base_dir) / "fno_eval" / "aggregated_errors.txt"
        aggregate_fold_stats(all_stats, fold_counts=fold_counts, output_path=aggregated_path)
    if val_error_rows:
        csv_path = Path(base_dir) / "fno_eval" / "val_errors_by_sample.csv"
        write_val_error_csv(val_error_rows, csv_path)


def run_training_pipeline(cfg, default_split_mode: str = "standard") -> None:
    """Main entrypoint used by wrappers; dispatches to standard or k-fold mode."""

    configure_device(cfg, set_device=False)
    run_cfg = load_cfg_and_paths(cfg, default_split_mode=default_split_mode)
    resolved_cfg = getattr(run_cfg, "cfg", cfg)
    seed_all(run_cfg.seed)
    configure_device(resolved_cfg, set_device=True)

    if run_cfg.split_mode == "standard":
        _run_standard(resolved_cfg, run_cfg)
        return
    if run_cfg.split_mode == "kfold":
        _run_kfold(resolved_cfg, run_cfg)
        return

    raise ValueError(f"Unsupported split_mode: {run_cfg.split_mode}")


__all__ = [
    "DatasetBundle",
    "InferenceMetrics",
    "KeyConfig",
    "ModelBundle",
    "RunConfig",
    "SplitSpec",
    "_collect_sim_param_overrides",
    "build_datasets",
    "build_domain",
    "build_model_and_nodes",
    "_assert_eval_checkpoint_loadable",
    "configure_device",
    "get_splits",
    "load_and_preprocess_data",
    "load_cfg_and_paths",
    "print_dict_shapes",
    "run_autoregressive_rollout_and_metrics",
    "run_inference_and_metrics",
    "run_training_pipeline",
    "seed_all",
    "setup_wandb_run",
    "solve_domain",
    "write_val_error_csv",
]
