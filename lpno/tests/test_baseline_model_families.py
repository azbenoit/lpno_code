from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import lpno.models.ns_energy_pipeline as pipeline
import lpno.models.pipeline.model_domain as model_domain
import lpno.models.pipeline.postprocessing as postprocessing
from lpno.models.pipeline.dataclasses import DatasetBundle
from lpno.models.pipeline.preprocessing_datasets import build_datasets


class _RecordingDomain:
    def __init__(self):
        self.constraints = []
        self.validators = []

    def add_constraint(self, constraint, name):
        self.constraints.append((name, constraint))

    def add_validator(self, validator, name):
        self.validators.append((name, validator))


class _DummyModel:
    def __init__(self):
        self.checkpoint_filename = "u_network.0.pth"

    def parameters(self):
        return [torch.nn.Parameter(torch.ones(1))]

    def make_node(self, name):
        return f"node:{name}"

    def eval(self):
        return self


def _pointwise_var() -> dict[str, np.ndarray]:
    base = np.arange(2 * 1 * 3 * 2 * 2, dtype=np.float32).reshape(2, 1, 3, 2, 2)
    return {
        "V_scan": base + 1.0,
        "P": base + 2.0,
        "H_norm": base + 3.0,
        "x": base + 4.0,
        "y": base + 5.0,
        "z": base + 6.0,
        "T": base + 7.0,
        "alpha": base + 8.0,
        "fl": base + 9.0,
    }


def test_build_datasets_pointwise_flattens_structured_tensors() -> None:
    var = _pointwise_var()
    split = SimpleNamespace(fold=0, train_idx=np.array([0]), val_idx=np.array([1]))
    key_cfg = SimpleNamespace(
        str_fno_input_keys=["V_scan", "P", "H_norm", "x", "y", "z"],
        str_fno_output_keys=["T", "alpha"],
        str_energy_input_keys=[],
        str_energy_output_keys=[],
        str_vof_input_keys=[],
        str_vof_output_keys=[],
        str_diff_input_keys=[],
        str_diff_output_keys=[],
    )
    run_cfg = SimpleNamespace(data_layout="pointwise", autoregressive_enabled=False)

    datasets = build_datasets(
        var=var,
        key_cfg=key_cfg,
        split=split,
        pde_loss=False,
        vof_pde_loss=False,
        diff_loss=False,
        run_cfg=run_cfg,
    )

    assert datasets.ds_input is not None
    assert datasets.ds_output is not None
    assert datasets.ds_input["V_scan"].shape == (12, 1)
    assert datasets.ds_output["T"].shape == (12, 1)
    assert datasets.ds_output["fl"].shape == (12, 1)
    assert datasets.ds_output_val is not None
    assert datasets.ds_output_val["fl"].shape == (12, 1)


def test_build_model_and_nodes_dispatches_deeponet(monkeypatch: pytest.MonkeyPatch) -> None:
    instantiate_calls = []

    def _instantiate_arch(**kwargs):
        instantiate_calls.append(kwargs)
        return _DummyModel()

    monkeypatch.setattr(model_domain, "instantiate_arch", _instantiate_arch)
    monkeypatch.setattr(model_domain, "fl_Node", lambda **kwargs: object())
    monkeypatch.setattr(model_domain, "Node", lambda **kwargs: SimpleNamespace(**kwargs))

    cfg = SimpleNamespace(
        arch=SimpleNamespace(
            branch=SimpleNamespace(layer_size=64),
            trunk=SimpleNamespace(layer_size=96),
            deeponet=SimpleNamespace(branch_dim=None, trunk_dim=None),
        )
    )
    key_cfg = SimpleNamespace(
        fno_input_keys=[],
        fno_output_keys=["T", "alpha"],
        str_fno_output_keys=["T", "alpha"],
    )

    bundle = model_domain.build_model_and_nodes(
        cfg=cfg,
        key_cfg=key_cfg,
        material_params={"T_s": 1.0, "T_l": 2.0},
        laser_params={},
        ref_scales={"T": 1.0},
        alpha_masking=False,
        eval_only=False,
        model_family="deeponet",
        deeponet_branch_keys=["V_scan", "P", "H_norm"],
        deeponet_trunk_keys=["x", "y", "z"],
        pde_loss=False,
        vof_pde_loss=False,
        diff_loss=False,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        diff_weight=1.0,
    )

    assert bundle.model is not None
    assert len(instantiate_calls) == 3
    assert instantiate_calls[0]["output_keys"][0].name == "branch"
    assert instantiate_calls[1]["output_keys"][0].name == "trunk"
    assert instantiate_calls[2]["branch_dim"] == 64
    assert instantiate_calls[2]["trunk_dim"] == 96


def test_build_domain_uses_pointwise_constraint_for_fully_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    pointwise_calls = []
    # build_domain assigns data_constraint.save_batch to suppress VTK writes, so
    # the mocked constraint must be an object (not a bare string).
    fake_constraint = SimpleNamespace()

    monkeypatch.setattr(model_domain, "Domain", _RecordingDomain)
    monkeypatch.setattr(
        model_domain.PointwiseConstraint,
        "from_numpy",
        lambda **kwargs: pointwise_calls.append(kwargs) or fake_constraint,
    )
    monkeypatch.setattr(model_domain, "PointwiseValidator", lambda **kwargs: ("pointwise_validator", kwargs))

    datasets = DatasetBundle(
        var_train=None,
        var_val=None,
        ds_input={"x": np.ones((8, 1), dtype=np.float32)},
        ds_output={"T": np.ones((8, 1), dtype=np.float32), "fl": np.ones((8, 1), dtype=np.float32)},
        ds_input_val={"x": np.ones((4, 1), dtype=np.float32)},
        ds_output_val={"T": np.ones((4, 1), dtype=np.float32), "fl": np.ones((4, 1), dtype=np.float32)},
        data_ds=SimpleNamespace(),
        val_data_ds=SimpleNamespace(),
    )
    model_bundle = SimpleNamespace(nn_nodes=["node"], energy_nodes=None, vof_nodes=None, diff_nodes=None)
    run_cfg = SimpleNamespace(
        data_layout="pointwise",
        model_family="fully_connected",
        time_as_batch=False,
        pde_loss=False,
        vof_pde_loss=False,
        diff_loss=False,
        fast_training=False,
        autoregressive_enabled=False,
        autoregressive_mode="masked_patch",
        keys=SimpleNamespace(str_fno_output_keys=["T"]),
        material_params={"T_s": 1.0},
        ref_scales={"T": 1.0},
        nz_effective=4,
        nx_effective=4,
        ny_effective=4,
    )
    # build_domain reads cfg via OmegaConf.select (e.g. save_filetypes,
    # custom.gradient_loss_weight), so cfg must be a real OmegaConf node, not a
    # SimpleNamespace. run_cfg stays a SimpleNamespace (accessed by attribute).
    cfg = OmegaConf.create({"batch_size": {"grid": 16, "validation": 8}})

    domain = model_domain.build_domain(cfg=cfg, model_bundle=model_bundle, datasets=datasets, run_cfg=run_cfg)

    assert pointwise_calls
    assert "fl" in pointwise_calls[0]["outvar"]
    assert domain.constraints == [("data", fake_constraint)]
    assert domain.validators[0][0] == "data"
    assert "fl" in domain.validators[0][1][1]["true_outvar"]


def test_run_pointwise_inference_and_metrics_writes_expected_files(tmp_path: Path) -> None:
    class _LinearPointwise(torch.nn.Module):
        def forward(self, invar):
            return {
                "T": invar["x"] + invar["P"],
                "alpha": torch.sigmoid(invar["z"]),
            }

    ds_input = {
        "x": np.array([[0.1], [0.2], [0.3]], dtype=np.float32),
        "P": np.array([[0.4], [0.5], [0.6]], dtype=np.float32),
        "z": np.array([[0.0], [0.1], [0.2]], dtype=np.float32),
    }
    ds_output = {
        "T": np.array([[0.5], [0.7], [0.9]], dtype=np.float32),
        "alpha": 1.0 / (1.0 + np.exp(-ds_input["z"])),
    }
    run_cfg = SimpleNamespace(
        inference_batch_size=2,
        keys=SimpleNamespace(str_fno_output_keys=["T", "alpha"]),
        alpha_masking=False,
        material_params={"T_s": 1.0, "T_l": 2.0},
        ref_scales={"T": 1.0},
    )

    metrics = postprocessing.run_pointwise_inference_and_metrics(
        model=_LinearPointwise(),
        ds_input=ds_input,
        ds_output=ds_output,
        run_cfg=run_cfg,
        output_dir=tmp_path / "fno_eval",
    )

    assert "T" in metrics.errors
    assert "alpha" in metrics.errors
    assert "fl" in metrics.errors
    assert (tmp_path / "fno_eval" / "predictions.npz").is_file()
    assert (tmp_path / "fno_eval" / "ground_truth.npz").is_file()
    assert (tmp_path / "fno_eval" / "model_errors.txt").is_file()


def test_run_pointwise_inference_and_metrics_emits_kfold_val_rows(tmp_path: Path) -> None:
    class _IdentityPointwise(torch.nn.Module):
        def forward(self, invar):
            return {
                "T": invar["x"],
                "alpha": invar["alpha_in"],
            }

    def _flatten_structured(arr: np.ndarray) -> np.ndarray:
        moved = np.moveaxis(arr, 1, -1)
        return moved.reshape(-1, arr.shape[1])

    structured_t = np.arange(2 * 1 * 2 * 2 * 2, dtype=np.float32).reshape(2, 1, 2, 2, 2)
    structured_alpha = np.full_like(structured_t, 0.75)
    ds_input = {
        "x": _flatten_structured(structured_t),
        "alpha_in": _flatten_structured(structured_alpha),
    }
    ds_output = {
        "T": _flatten_structured(structured_t),
        "alpha": _flatten_structured(structured_alpha),
        "fl": _flatten_structured(np.zeros_like(structured_t)),
    }
    run_cfg = SimpleNamespace(
        inference_batch_size=16,
        keys=SimpleNamespace(str_fno_output_keys=["T", "alpha"]),
        alpha_masking=False,
        material_params={"T_s": 1.0, "T_l": 2.0},
        ref_scales={"T": 1.0},
        use_2d=False,
        time_as_dim=False,
        nx_effective=2,
        ny_effective=2,
        nz_effective=2,
    )

    metrics = postprocessing.run_pointwise_inference_and_metrics(
        model=_IdentityPointwise(),
        ds_input=ds_input,
        ds_output=ds_output,
        run_cfg=run_cfg,
        output_dir=tmp_path / "fold_0",
        fold=3,
        val_idx=np.array([5, 9]),
        p_values=np.array([100.0, 200.0]),
        v_values=np.array([0.5, 1.5]),
    )

    assert len(metrics.val_rows) == 6
    first_row = metrics.val_rows[0]
    assert first_row["fold"] == 3
    assert first_row["global_index"] == 5
    assert first_row["local_val_index"] == 0
    assert first_row["P"] == 100.0
    assert first_row["V_scan"] == 0.5


def test_run_pointwise_inference_and_metrics_restores_structured_grid_outputs(tmp_path: Path) -> None:
    class _IdentityPointwise(torch.nn.Module):
        def forward(self, invar):
            return {
                "T": invar["x"],
                "alpha": invar["alpha_in"],
            }

    def _flatten_structured(arr: np.ndarray) -> np.ndarray:
        moved = np.moveaxis(arr, 1, -1)
        return moved.reshape(-1, arr.shape[1])

    structured_t = np.arange(2 * 1 * 3 * 2 * 2, dtype=np.float32).reshape(2, 1, 3, 2, 2)
    structured_alpha = np.linspace(0.1, 0.9, structured_t.size, dtype=np.float32).reshape(2, 1, 3, 2, 2)
    ds_input = {
        "x": _flatten_structured(structured_t),
        "alpha_in": _flatten_structured(structured_alpha),
    }
    ds_output = {
        "T": _flatten_structured(structured_t),
        "alpha": _flatten_structured(structured_alpha),
    }
    run_cfg = SimpleNamespace(
        inference_batch_size=7,
        keys=SimpleNamespace(str_fno_output_keys=["T", "alpha"]),
        alpha_masking=False,
        material_params={"T_s": 1.0, "T_l": 2.0},
        ref_scales={"T": 1.0},
        use_2d=False,
        time_as_dim=False,
        nx_effective=3,
        ny_effective=2,
        nz_effective=2,
    )

    metrics = postprocessing.run_pointwise_inference_and_metrics(
        model=_IdentityPointwise(),
        ds_input=ds_input,
        ds_output=ds_output,
        run_cfg=run_cfg,
        output_dir=tmp_path / "fno_eval",
    )

    saved_pred = np.load(tmp_path / "fno_eval" / "predictions.npz")
    saved_truth = np.load(tmp_path / "fno_eval" / "ground_truth.npz")
    assert saved_pred["T"].shape == (2, 1, 3, 2, 2)
    assert saved_pred["alpha"].shape == (2, 1, 3, 2, 2)
    assert saved_truth["T"].shape == (2, 1, 3, 2, 2)
    assert saved_truth["alpha"].shape == (2, 1, 3, 2, 2)
    assert metrics.errors["T"]["abs_err_mean"] == pytest.approx(0.0)
    assert metrics.errors["alpha"]["abs_err_mean"] == pytest.approx(0.0)


def test_run_inference_and_metrics_aligns_unet_grid_predictions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _DummyInferencer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, ds_input, name, step, batch_size):
            return {
                "T": np.ones((2, 1, 6, 4, 4), dtype=np.float32),
                "alpha": np.ones((2, 1, 6, 4, 4), dtype=np.float32),
            }

    monkeypatch.setattr(postprocessing, "FNOGridInferencer", _DummyInferencer)

    metrics = postprocessing.run_inference_and_metrics(
        model=torch.nn.Identity(),
        ds_input={"x": np.zeros((2, 1, 3, 4, 4), dtype=np.float32)},
        ds_output={
            "T": np.zeros((2, 1, 3, 4, 4), dtype=np.float32),
            "alpha": np.ones((2, 1, 3, 4, 4), dtype=np.float32),
            "fl": np.ones((2, 1, 3, 4, 4), dtype=np.float32),
        },
        run_cfg=SimpleNamespace(
            inference_batch_size=1,
            keys=SimpleNamespace(str_fno_input_keys=["x"], str_fno_output_keys=["T", "alpha"]),
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            alpha_masking=False,
            material_params={"T_s": 1.0, "T_l": 2.0},
            ref_scales={"T": 1.0},
        ),
        output_dir=tmp_path / "fno_eval",
    )

    saved = np.load(tmp_path / "fno_eval" / "predictions.npz")
    assert saved["T"].shape == (2, 1, 3, 4, 4)
    assert saved["alpha"].shape == (2, 1, 3, 4, 4)
    assert "T" in metrics.errors


def test_run_standard_eval_dispatches_pointwise_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    var = {"T": np.zeros((2, 1, 1, 1, 1), dtype=np.float32)}
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
        model_family="fully_connected",
        data_layout="pointwise",
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
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        inference_batch_size=1,
    )
    cfg = SimpleNamespace(network_dir="/tmp/fake")

    monkeypatch.setattr(pipeline, "setup_wandb_run", lambda cfg, wb_project, run_name: None)
    monkeypatch.setattr(pipeline, "load_and_preprocess_data", lambda *args, **kwargs: var)
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
            ds_input={},
            ds_output={},
            ds_input_val={"x": np.ones((4, 1), dtype=np.float32)},
            ds_output_val={"T": np.ones((4, 1), dtype=np.float32)},
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
        "run_pointwise_inference_and_metrics",
        lambda **kwargs: calls.append(kwargs["output_dir"]),
    )

    pipeline._run_standard(cfg, run_cfg)

    assert calls == [Path("fno_eval")]


def test_run_kfold_pointwise_collects_val_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    var = {
        "P": np.ones((2, 1, 1, 1, 1), dtype=np.float32),
        "V_scan": np.ones((2, 1, 1, 1, 1), dtype=np.float32),
    }
    run_cfg = SimpleNamespace(
        wb_project="proj",
        run_name="run",
        data_directory_path="train",
        seed=3,
        split_mode="kfold",
        k_folds=2,
        fold_index=-1,
        keys=SimpleNamespace(str_fno_input_keys=["x"], str_fno_output_keys=["T"]),
        material_params={"T_s": 1.0},
        laser_params={},
        ref_scales={"T": 1.0, "P": 10.0, "V": 2.0},
        alpha_masking=False,
        eval_only=False,
        model_family="fully_connected",
        data_layout="pointwise",
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
        network_base_dir="/tmp/base",
    )
    cfg = SimpleNamespace(network_dir="/tmp/fake")

    monkeypatch.setattr(pipeline, "load_and_preprocess_data", lambda *args, **kwargs: var)
    monkeypatch.setattr(pipeline, "print_dict_shapes", lambda var: None)
    monkeypatch.setattr(
        pipeline,
        "get_splits",
        lambda **kwargs: [SimpleNamespace(fold=1, train_idx=np.array([0]), val_idx=np.array([1]))],
    )
    monkeypatch.setattr(pipeline, "setup_wandb_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "build_datasets",
        lambda **kwargs: SimpleNamespace(
            var_val=var,
            ds_input_val={"x": np.ones((4, 1), dtype=np.float32)},
            ds_output_val={"T": np.ones((4, 1), dtype=np.float32)},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "build_model_and_nodes",
        lambda **kwargs: SimpleNamespace(model=SimpleNamespace(checkpoint_filename="u_network.0.pth")),
    )
    monkeypatch.setattr(pipeline, "build_domain", lambda **kwargs: object())
    monkeypatch.setattr(pipeline, "solve_domain", lambda **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "run_pointwise_inference_and_metrics",
        lambda **kwargs: calls.append(kwargs) or SimpleNamespace(errors={"T": {}}, val_rows=[{"row": 1}]),
    )
    monkeypatch.setattr(pipeline, "aggregate_fold_stats", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "write_val_error_csv", lambda rows, csv_path: calls.append(("csv", rows, csv_path)))
    monkeypatch.setattr(pipeline.wandb, "finish", lambda: None)

    pipeline._run_kfold(cfg, run_cfg)

    pointwise_call = calls[0]
    assert pointwise_call["fold"] == 1
    assert np.array_equal(pointwise_call["val_idx"], np.array([1]))
    assert np.array_equal(pointwise_call["p_values"], np.array([10.0]))
    assert np.array_equal(pointwise_call["v_values"], np.array([2.0]))
    assert calls[1][0] == "csv"
