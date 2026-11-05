# pyright: reportAttributeAccessIssue=false
# pyright: reportUnboundVariable=false
# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportOptionalSubscript=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportOptionalMemberAccess=false
"""Public exports for the modular NS/Energy pipeline implementation.

The facade ``lpno.models.ns_energy_pipeline`` re-exports these
same symbols for top-level pipeline access.
"""

from .config_loading import (
    _collect_sim_param_overrides,
    configure_device,
    load_cfg_and_paths,
    seed_all,
    setup_wandb_run,
)
from .dataclasses import DatasetBundle, InferenceMetrics, KeyConfig, ModelBundle, RunConfig, SplitSpec
from .model_domain import build_domain, build_model_and_nodes, solve_domain
from .postprocessing import (
    run_lazy_dataset_metrics,
    run_masked_patch_metrics,
    run_autoregressive_rollout_and_metrics,
    run_lazy_autoregressive_rollout_and_metrics,
    run_inference_and_metrics,
    run_pointwise_inference_and_metrics,
    write_val_error_csv,
)
from .preprocessing_datasets import build_datasets, get_splits, load_and_preprocess_data, print_dict_shapes, sample_count

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
    "configure_device",
    "get_splits",
    "load_and_preprocess_data",
    "load_cfg_and_paths",
    "print_dict_shapes",
    "run_masked_patch_metrics",
    "run_lazy_dataset_metrics",
    "run_autoregressive_rollout_and_metrics",
    "run_lazy_autoregressive_rollout_and_metrics",
    "run_inference_and_metrics",
    "run_pointwise_inference_and_metrics",
    "sample_count",
    "seed_all",
    "setup_wandb_run",
    "solve_domain",
    "write_val_error_csv",
]
