"""Tests for packaged `lpno` adapter behavior."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aiida_pino.scripts import eval_adapter, flow3d_to_vtk_adapter, kfold_adapter, live_tail, train_adapter, vtk_to_dataset_adapter
from aiida_pino.utils import load_training_profile_defaults


def _training_profile(mode: str = "standard") -> dict:
    profile = load_training_profile_defaults("config_NS_energy_PINO_flow3D")
    profile["run_name"] = "run"
    profile["cuda_device"] = "0"
    profile["hydra_overrides"] = ["custom.foo=bar"]
    profile["split_mode"] = mode
    return profile


def test_build_train_command_uses_packaged_module() -> None:
    command = train_adapter.build_train_command(
        _training_profile(),
        Path("/tmp/train"),
        Path("/tmp/val"),
        Path("/tmp/output"),
    )
    assert command[:4] == [sys.executable, "-u", "-m", train_adapter.TRAIN_MODULE]
    assert "custom.split_mode=standard" in command
    assert "custom.cuda_device=0" in command
    assert command[-1] == "custom.foo=bar"


def test_build_commands_do_not_emit_auto_cuda_device() -> None:
    profile = _training_profile()
    profile["cuda_device"] = "auto"
    train_command = train_adapter.build_train_command(
        profile,
        Path("/tmp/train"),
        Path("/tmp/val"),
        Path("/tmp/output"),
    )
    eval_command = eval_adapter.build_eval_command(
        profile,
        Path("/tmp/dataset"),
        Path("/tmp/training"),
    )
    assert "custom.cuda_device=auto" not in train_command
    assert "custom.cuda_device=auto" not in eval_command


def test_build_train_command_scheduler_tokens_default_on_and_null_opt_out() -> None:
    # Default: the packaged decay keys are emitted (unchanged behaviour).
    command = train_adapter.build_train_command(
        _training_profile(), Path("/tmp/train"), Path("/tmp/val"), Path("/tmp/output"))
    assert any(t.startswith("scheduler.decay_rate=") for t in command)
    assert any(t.startswith("scheduler.decay_steps=") for t in command)
    # Explicit None opts out, for configs whose scheduler group has no decay
    # keys (e.g. one_cycle) — hydra struct mode rejects the unknown key.
    profile = _training_profile()
    profile["scheduler_decay_rate"] = None
    profile["scheduler_decay_steps"] = None
    command = train_adapter.build_train_command(
        profile, Path("/tmp/train"), Path("/tmp/val"), Path("/tmp/output"))
    assert not any(t.startswith("scheduler.decay_") for t in command)
    kfold_command = kfold_adapter.build_kfold_command(
        profile, Path("/tmp/dataset"), Path("/tmp/output"))
    assert not any(t.startswith("scheduler.decay_") for t in kfold_command)


def test_merge_training_profile_accepts_null_scheduler_keys() -> None:
    from aiida_pino.utils.profiles import merge_training_profile, validate_training_profile

    base = load_training_profile_defaults("config_NS_energy_PINO_flow3D")
    merged = merge_training_profile(
        {**base, "scheduler_decay_rate": None, "scheduler_decay_steps": None})
    assert merged["scheduler_decay_rate"] is None
    assert merged["scheduler_decay_steps"] is None
    validate_training_profile(merged)  # must not raise


def test_discover_raw_eval_dir_prefers_cwd_then_newest_hydra_run(tmp_path: Path) -> None:
    # Hydra chdirs evals into outputs/<name>_<ts>/, so fno_eval lands there; the
    # adapter must find it or every lazy eval 502s with metrics on disk.
    old = tmp_path / "outputs" / "run_a_0101" / "fno_eval"
    new = tmp_path / "outputs" / "run_b_0202" / "fno_eval"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    os.utime(old, (1, 1))
    assert eval_adapter.discover_raw_eval_dir(tmp_path) == new
    # Legacy layout (fno_eval directly in the job CWD) still wins.
    legacy = tmp_path / "fno_eval"
    legacy.mkdir()
    assert eval_adapter.discover_raw_eval_dir(tmp_path) == legacy
    # Nothing anywhere: returns the (nonexistent) legacy path unchanged.
    empty = tmp_path / "elsewhere"
    empty.mkdir()
    assert eval_adapter.discover_raw_eval_dir(empty) == empty / "fno_eval"


def test_build_eval_command_uses_packaged_module() -> None:
    command = eval_adapter.build_eval_command(
        _training_profile(),
        Path("/tmp/dataset"),
        Path("/tmp/training"),
    )
    assert command[:4] == [sys.executable, "-u", "-m", eval_adapter.EVAL_MODULE]
    assert "custom.name=run_eval" in command
    assert "custom.time_as_batch=false" in command
    assert command[-1] == "custom.foo=bar"


def test_standard_train_and_eval_commands_honor_profile_split_mode() -> None:
    kfold_profile = _training_profile(mode="kfold")

    train_command = train_adapter.build_train_command(
        kfold_profile,
        Path("/tmp/train"),
        Path("/tmp/val"),
        Path("/tmp/output"),
    )
    eval_command = eval_adapter.build_eval_command(
        kfold_profile,
        Path("/tmp/dataset"),
        Path("/tmp/training"),
    )

    assert "custom.split_mode=kfold" in train_command
    assert "custom.split_mode=kfold" in eval_command


def test_build_kfold_command_uses_packaged_module() -> None:
    command = kfold_adapter.build_kfold_command(
        _training_profile(mode="kfold"),
        Path("/tmp/dataset"),
        Path("/tmp/output"),
    )
    assert command[:4] == [sys.executable, "-u", "-m", kfold_adapter.KFOLD_MODULE]
    assert "custom.split_mode=kfold" in command
    assert "custom.k_folds=8" in command
    assert command[-1] == "custom.foo=bar"


def test_eval_adapter_collects_rollout_metric_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    (output_dir / "model_errors.txt").write_text("[T]\nrmse: 0.1\n", encoding="utf-8")
    (output_dir / "model_errors_rollout_full.txt").write_text("[T]\nrmse: 0.2\n", encoding="utf-8")
    (output_dir / "model_errors_rollout_k.txt").write_text("[T]\nrmse: 0.3\n", encoding="utf-8")

    payload = eval_adapter.collect_metric_payload(output_dir)
    assert payload["T"]["rmse"] == 0.1
    assert payload["rollout_full"]["T"]["rmse"] == 0.2
    assert payload["rollout_k"]["T"]["rmse"] == 0.3


def test_eval_adapter_collects_stdout_metrics_when_metric_files_are_absent(tmp_path: Path) -> None:
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    stdout_path = tmp_path / "aiida.out"
    stdout_path.write_text(
        "T: {'rmse': 0.1, 'rel_rmse': 0.2}\nalpha: {'iou_mean': 0.9}\n",
        encoding="utf-8",
    )

    payload = eval_adapter.collect_metric_payload(output_dir, stdout_path)

    assert payload["T"]["rmse"] == 0.1
    assert payload["alpha"]["iou_mean"] == 0.9


def test_kfold_adapter_collects_aggregated_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "kfold"
    (output_dir / "fno_eval").mkdir(parents=True)
    (output_dir / "fold_0").mkdir(parents=True)
    (output_dir / "fno_eval" / "aggregated_errors.txt").write_text("[T]\nrmse: 0.11\n", encoding="utf-8")
    (output_dir / "fold_0" / "model_errors.txt").write_text("[T]\nrmse: 0.12\n", encoding="utf-8")

    payload = kfold_adapter.collect_kfold_metrics(output_dir)
    assert payload["aggregated"]["T"]["rmse"] == 0.11
    assert payload["folds"]["fold_0"]["T"]["rmse"] == 0.12


def test_train_adapter_reports_missing_package(monkeypatch) -> None:
    monkeypatch.setattr(train_adapter.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(SystemExit, match="lpno"):
        train_adapter.ensure_module_available(train_adapter.TRAIN_MODULE)


def test_eval_adapter_reports_missing_package(monkeypatch) -> None:
    monkeypatch.setattr(eval_adapter.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(SystemExit, match="lpno"):
        eval_adapter.ensure_module_available(eval_adapter.EVAL_MODULE)


def test_kfold_adapter_reports_missing_package(monkeypatch) -> None:
    monkeypatch.setattr(kfold_adapter.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(SystemExit, match="lpno"):
        kfold_adapter.ensure_module_available(kfold_adapter.KFOLD_MODULE)


def test_live_tail_success_deletes_temp_file(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    def _main() -> int:
        print("adapter ok")
        return 0

    assert live_tail.run_with_live_tail(_main) == 0
    assert not (tmp_path / live_tail.LIVE_TAIL_FILENAME).exists()
    captured = capsys.readouterr()
    assert "adapter ok" in captured.out


def test_live_tail_failure_keeps_temp_file(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    def _main() -> int:
        print("before failure")
        raise RuntimeError("boom")

    with pytest.raises(SystemExit, match="1"):
        live_tail.run_with_live_tail(_main)

    tail_file = tmp_path / live_tail.LIVE_TAIL_FILENAME
    assert tail_file.exists()
    assert "before failure" in tail_file.read_text(encoding="utf-8")
    assert "RuntimeError: boom" in tail_file.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert "RuntimeError: boom" in captured.err


def test_live_tail_nonzero_return_keeps_temp_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="3"):
        live_tail.run_with_live_tail(lambda: 3)

    assert (tmp_path / live_tail.LIVE_TAIL_FILENAME).exists()


def test_run_streamed_subprocess_streams_merged_output(capsys) -> None:
    live_tail.run_streamed_subprocess(
        [
            sys.executable,
            "-c",
            "import sys; print('stdout line'); print('stderr line', file=sys.stderr)",
        ]
    )

    captured = capsys.readouterr()
    assert "stdout line" in captured.out
    assert "stderr line" in captured.out


def test_flow3d_adapter_resolves_packaged_script_path(monkeypatch) -> None:
    fake_spec = SimpleNamespace(origin="/tmp/site-packages/lpno/custom_utils/Flow3D_flask_to_vtk.py")
    monkeypatch.setattr(flow3d_to_vtk_adapter.importlib.util, "find_spec", lambda _: fake_spec)
    assert flow3d_to_vtk_adapter.resolve_flow3d_script_path() == Path(fake_spec.origin).resolve()


def test_flow3d_adapter_reports_missing_package(monkeypatch) -> None:
    monkeypatch.setattr(flow3d_to_vtk_adapter.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(SystemExit, match="lpno"):
        flow3d_to_vtk_adapter.resolve_flow3d_script_path()


def test_dataset_adapter_imports_packaged_datasource(monkeypatch) -> None:
    fake_module = SimpleNamespace(VTKOpenFOAMDataSource=object)
    monkeypatch.setattr(vtk_to_dataset_adapter.importlib, "import_module", lambda _: fake_module)
    assert vtk_to_dataset_adapter.load_vtk_datasource_class() is object


def test_dataset_adapter_reports_missing_package(monkeypatch) -> None:
    error = ModuleNotFoundError("No module named 'lpno'")
    error.name = "lpno"

    def _raise(_name: str):
        raise error

    monkeypatch.setattr(vtk_to_dataset_adapter.importlib, "import_module", _raise)
    with pytest.raises(SystemExit, match="lpno"):
        vtk_to_dataset_adapter.load_vtk_datasource_class()


def test_dataset_adapter_uses_canonical_base_cache_basename() -> None:
    assert vtk_to_dataset_adapter.PinoDatasetData.DEFAULT_BASE_CACHE_BASENAME == "var_1_time_noref.npz"


def test_dataset_adapter_builds_symlink_mirror(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "a.vtk").write_text("vtk", encoding="utf-8")
    (source_root / "subdir").mkdir()
    (source_root / "subdir" / "b.vtk").write_text("vtk", encoding="utf-8")
    (source_root / ".aiida_pino_cache").mkdir()

    mirror_root = tmp_path / "mirror"
    vtk_to_dataset_adapter.build_symlink_mirror(source_root, mirror_root)

    assert (mirror_root / "a.vtk").is_symlink()
    assert (mirror_root / "subdir").is_symlink()
    assert not (mirror_root / ".aiida_pino_cache").exists()


def test_dataset_adapter_links_or_copies_npz_payload(tmp_path: Path) -> None:
    source = tmp_path / "var_1_time_noref.npz"
    np.savez_compressed(source, Temperature=np.ones((1, 1, 2, 2, 2), dtype=np.float32))
    destination = tmp_path / "dataset_output" / "cache.npz"

    vtk_to_dataset_adapter.link_or_copy_file(source, destination)

    assert destination.exists()
    with np.load(destination) as archive:
        assert archive["Temperature"].shape == (1, 1, 2, 2, 2)
    if os.name != "nt":
        assert source.stat().st_ino == destination.stat().st_ino


def test_dataset_adapter_external_cache_workspace_reuses_fingerprint(tmp_path: Path) -> None:
    source_root = tmp_path / "vtk-source"
    source_root.mkdir()
    (source_root / "sample.vtk").write_text("vtk", encoding="utf-8")

    mirror_root, cache_mode, cache_root = vtk_to_dataset_adapter.prepare_external_cache_workspace(
        source_root,
        "abc123",
        force_recompute=False,
    )
    assert cache_mode == "source_cache"
    assert cache_root == source_root / ".aiida_pino_cache" / "vtk_to_dataset" / "abc123"
    assert (mirror_root / "sample.vtk").is_symlink()


def test_dataset_adapter_external_cache_workspace_force_recompute_clears_cached_files(tmp_path: Path) -> None:
    source_root = tmp_path / "vtk-source"
    source_root.mkdir()
    (source_root / "sample.vtk").write_text("vtk", encoding="utf-8")

    _, _, cache_root = vtk_to_dataset_adapter.prepare_external_cache_workspace(
        source_root,
        "abc123",
        force_recompute=False,
    )
    (cache_root / "var_1_time_noref.npz").write_text("stale", encoding="utf-8")

    _, _, refreshed_cache_root = vtk_to_dataset_adapter.prepare_external_cache_workspace(
        source_root,
        "abc123",
        force_recompute=True,
    )
    assert refreshed_cache_root == cache_root
    assert not (cache_root / "var_1_time_noref.npz").exists()


def test_dataset_adapter_collects_group_metadata(tmp_path: Path) -> None:
    input_dir = tmp_path / "vtk"
    input_dir.mkdir()
    (input_dir / "results_V_scan_0p01_laser_power_5_395.vtk").write_text("vtk", encoding="utf-8")

    class FakeDatasource:
        def _group_files_by_base(self, paths):
            return {"results_V_scan_0p01_laser_power_5": paths}

        def _group_files_by_subdir(self):
            return {}

        def get_file_list(self):
            return ["results_V_scan_0p01_laser_power_5_395"]

        def _resolve_file_path(self, stem):
            return input_dir / f"{stem}.vtk"

        @staticmethod
        def _base_without_time(stem):
            return stem.rsplit("_", 1)[0]

        @staticmethod
        def _parse_params(stem):
            return {"V_scan": 0.01, "laser_power": 5.0, "time_s": 3.95e-4}

    grouping_mode, grouped_metadata, sample_metadata = vtk_to_dataset_adapter.collect_source_metadata(
        FakeDatasource(),
        input_dir,
        time_axis=True,
    )
    assert grouping_mode == "flat_files"
    assert grouped_metadata[0]["group"] == "results_V_scan_0p01_laser_power_5"
    assert sample_metadata[0]["params"]["V_scan"] == 0.01


def test_dataset_adapter_main_defaults_flip_off_for_flow3d_and_records_convention(monkeypatch, tmp_path: Path) -> None:
    """BUG-11: a profile without flip_axes gets the identity convention for flow3d, recorded in metadata."""
    input_dir = tmp_path / "vtk_input"
    input_dir.mkdir()
    (input_dir / "sim_0.vtk").write_text("vtk", encoding="utf-8")
    (input_dir / "var_1_time_noref.npz").write_text("cache", encoding="utf-8")

    captured: dict = {}

    class FakeDatasource:
        extra_fields = ["Temperature"]

        def _group_files_by_base(self, paths):
            return {"sim": list(paths)}

        def parse_directory(self, **kwargs):
            captured.update(kwargs)
            return {"Temperature": np.zeros((1, 1, 2, 2, 2), dtype=np.float32)}

        @staticmethod
        def _parse_params(stem):
            return {"time_s": 0.0}

    monkeypatch.setattr(vtk_to_dataset_adapter, "load_vtk_datasource_class", lambda: FakeDatasource)
    profile = {
        "geometry": {"x_range": [0, 1], "y_range": [0, 1], "z_range": [0, 1], "resolution": [2, 2, 2]},
        "vtk_fields": [],
    }  # deliberately no flip_axes key
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(json.dumps(profile), encoding="utf-8")
    # BUG-01: a flow3d import without laser metadata now fails loudly, so this
    # flip-axes test supplies a minimal case-manifest entry for the `sim` group.
    source_metadata_file = tmp_path / "source_metadata.json"
    source_metadata_file.write_text(
        json.dumps({"flow3d": {"case_manifests": {"sim": {"laser_power": 100.0, "V_scan": 0.5}}}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "dataset_output"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vtk_to_dataset_adapter",
            "--vtk-input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--profile-file",
            str(profile_file),
            "--source-metadata-file",
            str(source_metadata_file),
        ],
    )

    vtk_to_dataset_adapter._main()

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["flip_axes"] is False
    assert metadata["axis_convention"] == "identity"
    assert captured["flip_axes"] is False


def test_dataset_adapter_symlink_mirror_excludes_preexisting_datasource_caches(tmp_path: Path) -> None:
    """BUG-05 (aiida side): a stale var_*.npz in the source tree must not be mirrored and reused."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "a.vtk").write_text("vtk", encoding="utf-8")
    (source_root / "var_1_time_noref.npz").write_text("stale", encoding="utf-8")
    (source_root / "var_2.npz").write_text("stale", encoding="utf-8")

    mirror_root = tmp_path / "mirror"
    vtk_to_dataset_adapter.build_symlink_mirror(source_root, mirror_root)

    assert (mirror_root / "a.vtk").is_symlink()
    assert not (mirror_root / "var_1_time_noref.npz").exists()
    assert not (mirror_root / "var_2.npz").exists()


def test_train_adapter_seed_network_dir(tmp_path: Path) -> None:
    """Warm start: checkpoint files from --init-dir land in network_dir; clutter does not."""
    init_dir = tmp_path / "init_model"
    init_dir.mkdir()
    (init_dir / "model.pth").write_bytes(b"weights")
    (init_dir / "optim_checkpoint.0.pth").write_bytes(b"optim")
    (init_dir / "notes.txt").write_text("not a checkpoint", encoding="utf-8")
    (init_dir / ".hydra").mkdir()
    output_dir = tmp_path / "training_output"
    output_dir.mkdir()

    seeded = train_adapter.seed_network_dir(init_dir, output_dir)

    assert seeded == ["model.pth", "optim_checkpoint.0.pth"]
    assert (output_dir / "model.pth").read_bytes() == b"weights"
    assert not (output_dir / "notes.txt").exists()
    # empty init dir warns but does not fail
    empty = tmp_path / "empty"
    empty.mkdir()
    assert train_adapter.seed_network_dir(empty, output_dir) == []


def test_profile_environment_overlay(monkeypatch) -> None:
    monkeypatch.setenv("EXISTING", "1")
    assert live_tail.profile_environment({}) is None
    assert live_tail.profile_environment({"env": {}}) is None
    env = live_tail.profile_environment({"env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PINO_LAZY_NPZ": 1}})
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert env["PINO_LAZY_NPZ"] == "1"  # values coerced to str
    assert env["EXISTING"] == "1"  # os.environ inherited


def test_run_streamed_subprocess_dumps_tail_to_stderr_on_failure(capsys) -> None:
    import pytest as _pytest
    import subprocess as _sp
    with _pytest.raises(_sp.CalledProcessError):
        live_tail.run_streamed_subprocess(
            [sys.executable, "-c", "print('the real traceback line'); raise SystemExit(2)"]
        )
    captured = capsys.readouterr()
    assert "the real traceback line" in captured.err  # scheduler stderr now carries the tail
