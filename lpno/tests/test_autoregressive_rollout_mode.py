"""Tests for the pair-based autoregressive 'rollout' mode across model families.

Covers the eager pair-dataset wiring in build_datasets (grid and pointwise),
the masked_patch validator regression in build_domain, and the
_run_standard eval dispatch into run_autoregressive_rollout_and_metrics.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

import lpno.models.ns_energy_pipeline as pipeline
import lpno.models.pipeline.model_domain as model_domain
from lpno.models.pipeline.dataclasses import DatasetBundle
from lpno.models.pipeline.preprocessing_datasets import (
    DictGridDataset,
    DictPointwiseDataset,
    build_datasets,
)


class _RecordingDomain:
    def __init__(self):
        self.constraints = []
        self.validators = []

    def add_constraint(self, constraint, name):
        self.constraints.append((name, constraint))

    def add_validator(self, validator, name):
        self.validators.append((name, validator))


def _temporal_var() -> dict[str, np.ndarray]:
    shape = (2, 3, 1, 2, 2, 2)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return {
        "P": base + 1.0,
        "V_scan": base + 2.0,
        "x": base + 3.0,
        "T": base + 4.0,
        "alpha": base + 5.0,
        "fl": base + 6.0,
    }


def _key_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        str_fno_input_keys=["P", "V_scan", "x", "T_prev", "alpha_prev"],
        str_fno_output_keys=["T", "alpha"],
        str_energy_input_keys=[],
        str_energy_output_keys=[],
        str_vof_input_keys=[],
        str_vof_output_keys=[],
        str_diff_input_keys=[],
        str_diff_output_keys=[],
    )


def _ar_run_cfg(data_layout: str = "grid", prev_noise_std: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        data_layout=data_layout,
        autoregressive_enabled=True,
        autoregressive_mode="rollout",
        autoregressive_context_keys=["P", "V_scan", "x"],
        autoregressive_state_keys=["T", "alpha"],
        autoregressive_state_prev_suffix="_prev",
        autoregressive_prev_noise_std=prev_noise_std,
        seed=7,
    )


def _split() -> SimpleNamespace:
    return SimpleNamespace(fold=0, train_idx=np.array([0, 1]), val_idx=np.array([0, 1]))


def test_build_datasets_autoregressive_grid_assigns_pair_datasets() -> None:
    var = _temporal_var()

    datasets = build_datasets(
        var=var,
        key_cfg=_key_cfg(),
        split=_split(),
        pde_loss=False,
        vof_pde_loss=False,
        diff_loss=False,
        run_cfg=_ar_run_cfg("grid"),
    )

    assert isinstance(datasets.data_ds, DictGridDataset)
    assert isinstance(datasets.val_data_ds, DictGridDataset)
    assert datasets.ds_input is not None
    assert datasets.ds_input["T_prev"].shape == (4, 1, 2, 2, 2)
    np.testing.assert_allclose(datasets.ds_input["T_prev"], var["T"][:, :-1].reshape(4, 1, 2, 2, 2))
    np.testing.assert_allclose(datasets.ds_input["P"], var["P"][:, 1:].reshape(4, 1, 2, 2, 2))
    np.testing.assert_allclose(datasets.ds_output["T"], var["T"][:, 1:].reshape(4, 1, 2, 2, 2))
    np.testing.assert_allclose(datasets.ds_output["fl"], var["fl"][:, 1:].reshape(4, 1, 2, 2, 2))
    # Trajectories with the time axis intact must survive for the rollout eval.
    assert datasets.var_train["T"].shape == (2, 3, 1, 2, 2, 2)


def test_build_datasets_autoregressive_pointwise_flattens_pairs() -> None:
    var = _temporal_var()

    datasets = build_datasets(
        var=var,
        key_cfg=_key_cfg(),
        split=_split(),
        pde_loss=False,
        vof_pde_loss=False,
        diff_loss=False,
        run_cfg=_ar_run_cfg("pointwise"),
    )

    assert isinstance(datasets.data_ds, DictPointwiseDataset)
    n_rows = 2 * 2 * 8  # cases * pairs-per-case * voxels
    assert datasets.ds_input["T_prev"].shape == (n_rows, 1)
    expected_prev = np.moveaxis(var["T"][:, :-1].reshape(4, 1, 2, 2, 2), 1, -1).reshape(-1, 1)
    np.testing.assert_allclose(datasets.ds_input["T_prev"], expected_prev)
    assert datasets.ds_output["T"].shape == (n_rows, 1)
    assert datasets.ds_output["fl"].shape == (n_rows, 1)
    assert datasets.ds_input_val["T_prev"].shape == (n_rows, 1)


def test_build_datasets_autoregressive_prev_noise_train_only() -> None:
    var = _temporal_var()

    datasets = build_datasets(
        var=var,
        key_cfg=_key_cfg(),
        split=_split(),
        pde_loss=False,
        vof_pde_loss=False,
        diff_loss=False,
        run_cfg=_ar_run_cfg("grid", prev_noise_std=0.5),
    )

    clean_prev = var["T"][:, :-1].reshape(4, 1, 2, 2, 2)
    assert not np.allclose(datasets.ds_input["T_prev"], clean_prev)
    np.testing.assert_allclose(datasets.ds_input_val["T_prev"], clean_prev)
    # Context channels and targets stay noise-free.
    np.testing.assert_allclose(datasets.ds_input["P"], var["P"][:, 1:].reshape(4, 1, 2, 2, 2))
    np.testing.assert_allclose(datasets.ds_output["T"], var["T"][:, 1:].reshape(4, 1, 2, 2, 2))


def _domain_datasets() -> DatasetBundle:
    return DatasetBundle(
        var_train=None,
        var_val=None,
        ds_input=None,
        ds_output=None,
        ds_input_val=None,
        ds_output_val=None,
        data_ds=SimpleNamespace(),
        val_data_ds=SimpleNamespace(),
    )


def _domain_run_cfg(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        data_layout="grid",
        model_family="fno",
        time_as_batch=True,
        pde_loss=False,
        vof_pde_loss=False,
        diff_loss=False,
        fast_training=False,
        autoregressive_enabled=True,
        autoregressive_mode=mode,
        autoregressive_state_prev_suffix="_prev",
        keys=SimpleNamespace(str_fno_output_keys=["T", "alpha"]),
        material_params={"T_s": 1.0},
        ref_scales={"T": 1.0},
        nx_effective=4,
        ny_effective=4,
        nz_effective=4,
    )


def test_build_domain_masked_patch_builds_grid_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_domain, "Domain", _RecordingDomain)
    monkeypatch.setattr(model_domain, "SupervisedGridConstraint", lambda **kwargs: "grid_constraint")
    monkeypatch.setattr(model_domain, "StreamingGridValidator", lambda **kwargs: ("grid_validator", kwargs))
    monkeypatch.setattr(model_domain, "TemporalPatchPlotter", lambda **kwargs: ("temporal_patch_plotter", kwargs))

    datasets = _domain_datasets()
    # validator_plots: plotters are opt-in since the scalar-only recording
    # change; this test asserts the plotter wiring, so switch them on.
    cfg = OmegaConf.create({"batch_size": {"grid": 2, "validation": 1}, "custom": {"validator_plots": True}})
    model_bundle = SimpleNamespace(nn_nodes=["node"], energy_nodes=None, vof_nodes=None, diff_nodes=None)

    domain = model_domain.build_domain(
        cfg=cfg,
        model_bundle=model_bundle,
        datasets=datasets,
        run_cfg=_domain_run_cfg("masked_patch"),
    )

    assert domain.constraints == [("data", "grid_constraint")]
    assert len(domain.validators) == 1
    name, validator = domain.validators[0]
    assert name == "data"
    tag, kwargs = validator
    assert tag == "grid_validator"
    assert kwargs["dataset"] is datasets.val_data_ds
    plotter_tag, plotter_kwargs = kwargs["plotter"]
    assert plotter_tag == "temporal_patch_plotter"
    assert plotter_kwargs["variables"] == ["T", "alpha"]
    assert plotter_kwargs["state_prev_suffix"] == "_prev"


def test_build_domain_rollout_mode_uses_standard_grid_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_domain, "Domain", _RecordingDomain)
    monkeypatch.setattr(model_domain, "SupervisedGridConstraint", lambda **kwargs: "grid_constraint")
    monkeypatch.setattr(model_domain, "StreamingGridValidator", lambda **kwargs: ("grid_validator", kwargs))
    monkeypatch.setattr(model_domain, "CustomSlicePlotter", lambda **kwargs: ("slice_plotter", kwargs))

    datasets = _domain_datasets()
    # validator_plots on: this test asserts the slice-plotter wiring.
    cfg = OmegaConf.create({"batch_size": {"grid": 2, "validation": 1}, "custom": {"validator_plots": True}})
    model_bundle = SimpleNamespace(nn_nodes=["node"], energy_nodes=None, vof_nodes=None, diff_nodes=None)

    domain = model_domain.build_domain(
        cfg=cfg,
        model_bundle=model_bundle,
        datasets=datasets,
        run_cfg=_domain_run_cfg("rollout"),
    )

    assert domain.constraints == [("data", "grid_constraint")]
    assert len(domain.validators) == 1
    tag, kwargs = domain.validators[0][1]
    assert tag == "grid_validator"
    assert kwargs["plotter"][0] == "slice_plotter"


def test_run_standard_eval_dispatches_rollout_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    var_train = {"T": np.zeros((2, 3, 1, 1, 1, 1), dtype=np.float32)}
    run_cfg = SimpleNamespace(
        wb_project="proj",
        run_name="run",
        data_directory_path="train",
        val_directory_path="val",
        seed=3,
        keys=SimpleNamespace(str_fno_input_keys=["x"], str_fno_output_keys=["T"]),
        material_params={"T_s": 1.0},
        laser_params={},
        ref_scales={"T": 1.0},
        alpha_masking=False,
        eval_only=True,
        model_family="fno",
        data_layout="grid",
        deeponet_branch_keys=[],
        deeponet_trunk_keys=[],
        pde_loss=False,
        vof_pde_loss=False,
        diff_loss=False,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        diff_weight=1.0,
        debug=False,
        autoregressive_enabled=True,
        autoregressive_mode="rollout",
    )
    cfg = SimpleNamespace(network_dir="/tmp/fake")

    monkeypatch.setattr(pipeline, "setup_wandb_run", lambda cfg, wb_project, run_name: None)
    monkeypatch.setattr(pipeline, "load_and_preprocess_data", lambda *args, **kwargs: var_train)
    monkeypatch.setattr(pipeline, "print_dict_shapes", lambda var: None)
    monkeypatch.setattr(
        pipeline,
        "get_splits",
        lambda **kwargs: [SimpleNamespace(fold=0, train_idx=np.array([0]), val_idx=np.array([1]))],
    )
    monkeypatch.setattr(
        pipeline,
        "build_datasets",
        lambda **kwargs: SimpleNamespace(
            var_train=var_train,
            var_val=None,
            ds_input={},
            ds_output={},
            ds_input_val={},
            ds_output_val={},
            data_ds=object(),
            val_data_ds=object(),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "build_model_and_nodes",
        # eval_only moves the model to GPU (model.cuda()) when CUDA is present;
        # the mock must accept that call so the test is device-independent.
        lambda **kwargs: SimpleNamespace(
            model=SimpleNamespace(checkpoint_filename="u_network.0.pth", cuda=lambda *a, **k: None)
        ),
    )
    monkeypatch.setattr(pipeline, "_assert_eval_checkpoint_loadable", lambda model, checkpoint_dir, **kw: None)
    monkeypatch.setattr(pipeline, "build_domain", lambda **kwargs: object())
    monkeypatch.setattr(pipeline, "solve_domain", lambda **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "run_autoregressive_rollout_and_metrics",
        lambda **kwargs: calls.append((kwargs["output_dir"], kwargs["trajectory_var"] is var_train)),
    )

    pipeline._run_standard(cfg, run_cfg)

    assert calls == [(Path("fno_eval"), True)]


def test_build_datasets_eager_rejects_masked_patch_mode() -> None:
    var = _temporal_var()
    run_cfg = _ar_run_cfg("grid")
    run_cfg.autoregressive_mode = "masked_patch"

    with pytest.raises(ValueError, match="masked_patch"):
        build_datasets(
            var=var,
            key_cfg=_key_cfg(),
            split=_split(),
            pde_loss=False,
            vof_pde_loss=False,
            diff_loss=False,
            run_cfg=run_cfg,
        )
