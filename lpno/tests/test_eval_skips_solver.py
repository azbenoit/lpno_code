"""Eval mode must not run the physicsnemo Solver.

The Solver eager-loads the whole data constraint into host RAM (~0.75 GB per
full-frame case, so a 100-case test set is ~75 GB) and OOMs the box. In eval the
weights are already loaded by the checkpoint assert and the metric helpers stream
their own DataLoader, so `_run_standard` must skip build_domain/solve_domain when
`eval_only` is set -- and still run them for training.
"""

from types import SimpleNamespace

import pytest

import lpno.models.ns_energy_pipeline as pipeline

# pyright: reportArgumentType=false


def _install_stubs(monkeypatch: pytest.MonkeyPatch, calls: list) -> None:
    monkeypatch.setattr(pipeline, "setup_wandb_run", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "load_and_preprocess_data", lambda *a, **k: {"var": 1})
    monkeypatch.setattr(pipeline, "print_dict_shapes", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "sample_count", lambda *a, **k: 4)
    monkeypatch.setattr(pipeline, "get_splits", lambda *a, **k: [SimpleNamespace(train_idx=[0], val_idx=[1])])
    # Lazy bundle: ds_input None routes the eval branch to run_lazy_dataset_metrics.
    monkeypatch.setattr(
        pipeline,
        "build_datasets",
        lambda *a, **k: SimpleNamespace(ds_input=None, ds_output=None, val_data_ds=object()),
    )
    monkeypatch.setattr(
        pipeline,
        "build_model_and_nodes",
        lambda *a, **k: SimpleNamespace(model=SimpleNamespace(cuda=lambda *a, **k: None)),
    )
    monkeypatch.setattr(pipeline, "_assert_eval_checkpoint_loadable", lambda *a, **k: calls.append("assert_ckpt"))
    monkeypatch.setattr(pipeline, "build_domain", lambda *a, **k: calls.append("build_domain"))
    monkeypatch.setattr(pipeline, "solve_domain", lambda *a, **k: calls.append("solve_domain"))
    monkeypatch.setattr(pipeline, "run_lazy_dataset_metrics", lambda *a, **k: calls.append("lazy_metrics"))


def _run_cfg(eval_only: bool) -> SimpleNamespace:
    return SimpleNamespace(
        wb_project="p", run_name="r", seed=0,
        data_directory_path="d", val_directory_path="v",
        keys=SimpleNamespace(str_fno_output_keys=["T", "alpha"]),
        pde_loss=False, vof_pde_loss=False, diff_loss=False,
        material_params={}, laser_params={}, ref_scales={}, alpha_masking=True,
        eval_only=eval_only, dx=1.0, dy=1.0, dz=1.0, diff_weight=1.0, debug=False,
    )


def test_eval_only_skips_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    _install_stubs(monkeypatch, calls)
    pipeline._run_standard(SimpleNamespace(network_dir="ckpt"), _run_cfg(eval_only=True))
    assert "solve_domain" not in calls, "eval must not run the OOM-prone Solver"
    assert "build_domain" not in calls
    assert "lazy_metrics" in calls  # eval still streams its metrics


def test_training_runs_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    _install_stubs(monkeypatch, calls)
    pipeline._run_standard(SimpleNamespace(network_dir="ckpt"), _run_cfg(eval_only=False))
    assert "build_domain" in calls and "solve_domain" in calls
    assert "lazy_metrics" not in calls  # training does not run the eval branch
