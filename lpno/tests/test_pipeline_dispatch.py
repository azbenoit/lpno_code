from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lpno.models.ns_energy_pipeline as pipeline
# pyright: reportArgumentType=false

def test_run_training_pipeline_dispatches_standard(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []
    run_cfg = SimpleNamespace(seed=17, split_mode="standard")

    monkeypatch.setattr(
        pipeline,
        "load_cfg_and_paths",
        lambda cfg, default_split_mode="standard": run_cfg,
    )
    monkeypatch.setattr(pipeline, "configure_device", lambda cfg, set_device=True: calls.append(("configure", set_device)))
    monkeypatch.setattr(pipeline, "seed_all", lambda seed: calls.append(("seed", seed)))
    monkeypatch.setattr(pipeline, "_run_standard", lambda cfg, rc: calls.append(("standard", rc)))
    monkeypatch.setattr(pipeline, "_run_kfold", lambda cfg, rc: calls.append(("kfold", rc)))

    pipeline.run_training_pipeline(object(), default_split_mode="kfold") 

    assert calls[0] == ("configure", False)
    assert calls[1] == ("seed", 17)
    assert calls[2] == ("configure", True)
    assert calls[3] == ("standard", run_cfg)
    assert all(name != "kfold" for name, _ in calls)


def test_run_training_pipeline_uses_resolved_cfg_from_run_cfg(monkeypatch: pytest.MonkeyPatch) -> None:
    original_cfg = SimpleNamespace(tag="original")
    resolved_cfg = SimpleNamespace(tag="resolved")
    calls: list[tuple[str, object]] = []
    run_cfg = SimpleNamespace(seed=17, split_mode="standard", cfg=resolved_cfg)

    monkeypatch.setattr(
        pipeline,
        "load_cfg_and_paths",
        lambda cfg, default_split_mode="standard": run_cfg,
    )
    monkeypatch.setattr(pipeline, "configure_device", lambda cfg, set_device=True: calls.append(("configure", cfg, set_device)))
    monkeypatch.setattr(pipeline, "seed_all", lambda seed: calls.append(("seed", seed)))
    monkeypatch.setattr(pipeline, "_run_standard", lambda cfg, rc: calls.append(("standard", cfg, rc)))
    monkeypatch.setattr(pipeline, "_run_kfold", lambda cfg, rc: calls.append(("kfold", cfg, rc)))

    pipeline.run_training_pipeline(original_cfg, default_split_mode="standard")

    assert calls[0] == ("configure", original_cfg, False)
    assert calls[2] == ("configure", resolved_cfg, True)
    assert calls[3] == ("standard", resolved_cfg, run_cfg)


def test_run_training_pipeline_dispatches_kfold(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []
    run_cfg = SimpleNamespace(seed=19, split_mode="kfold")

    monkeypatch.setattr(
        pipeline,
        "load_cfg_and_paths",
        lambda cfg, default_split_mode="standard": run_cfg,
    )
    monkeypatch.setattr(pipeline, "configure_device", lambda cfg, set_device=True: None)
    monkeypatch.setattr(pipeline, "seed_all", lambda seed: None)
    monkeypatch.setattr(pipeline, "_run_standard", lambda cfg, rc: calls.append(("standard", rc)))
    monkeypatch.setattr(pipeline, "_run_kfold", lambda cfg, rc: calls.append(("kfold", rc)))

    pipeline.run_training_pipeline(object())

    assert calls == [("kfold", run_cfg)]


def test_run_training_pipeline_invalid_mode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = SimpleNamespace(seed=23, split_mode="unknown")

    monkeypatch.setattr(
        pipeline,
        "load_cfg_and_paths",
        lambda cfg, default_split_mode="standard": run_cfg,
    )
    monkeypatch.setattr(pipeline, "configure_device", lambda cfg, set_device=True: None)
    monkeypatch.setattr(pipeline, "seed_all", lambda seed: None)

    with pytest.raises(ValueError, match="Unsupported split_mode"):
        pipeline.run_training_pipeline(object())


def test_assert_eval_checkpoint_loadable_raises_if_missing(tmp_path: Path) -> None:
    model = SimpleNamespace(checkpoint_filename="u_network.0.pth", load=lambda directory, map_location=None: None)

    with pytest.raises(FileNotFoundError, match="Evaluation checkpoint not found"):
        pipeline._assert_eval_checkpoint_loadable(model, tmp_path)


def test_assert_eval_checkpoint_loadable_raises_if_incompatible(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "u_network.0.pth"
    ckpt_path.write_bytes(b"dummy")

    def _raise_load(directory: str, map_location=None):
        raise RuntimeError("size mismatch for spectral_conv.weight")

    model = SimpleNamespace(checkpoint_filename="u_network.0.pth", load=_raise_load)
    with pytest.raises(RuntimeError, match="Failed to load evaluation checkpoint"):
        pipeline._assert_eval_checkpoint_loadable(model, tmp_path)


def test_assert_eval_checkpoint_loadable_calls_model_load(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "u_network.0.pth"
    ckpt_path.write_bytes(b"dummy")
    calls: list[tuple[str, str | None]] = []

    def _load(directory: str, map_location=None):
        calls.append((directory, map_location))

    model = SimpleNamespace(checkpoint_filename="u_network.0.pth", load=_load)
    pipeline._assert_eval_checkpoint_loadable(model, tmp_path)

    assert calls == [(str(tmp_path), "cpu")]


def test_run_standard_eval_calls_checkpoint_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

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
            data_ds=object(),
            val_data_ds=object(),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "build_model_and_nodes",
        lambda **kwargs: SimpleNamespace(
            model=SimpleNamespace(checkpoint_filename="u_network.0.pth", cuda=lambda *a, **k: None)
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_assert_eval_checkpoint_loadable",
        lambda model, checkpoint_dir, **kw: calls.append(("guard", checkpoint_dir)),
    )
    monkeypatch.setattr(pipeline, "build_domain", lambda **kwargs: object())
    monkeypatch.setattr(pipeline, "solve_domain", lambda **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "run_inference_and_metrics",
        lambda **kwargs: calls.append(("infer", kwargs["output_dir"])),
    )

    pipeline._run_standard(cfg, run_cfg)

    assert ("guard", cfg.network_dir) in calls
