from pathlib import Path

from lpno.models import cuda_bootstrap


def test_bootstrap_uses_custom_cuda_device_cli(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    cuda_bootstrap.bootstrap_cuda_visible_devices_from_argv(["custom.cuda_device=1"])

    assert cuda_bootstrap.os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_bootstrap_prefers_custom_visible_devices_cli(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    cuda_bootstrap.bootstrap_cuda_visible_devices_from_argv(
        ["custom.cuda_device=0", "custom.cuda_visible_devices=1,0"]
    )

    assert cuda_bootstrap.os.environ["CUDA_VISIBLE_DEVICES"] == "1,0"


def test_bootstrap_preserves_scheduler_cuda_visible_devices(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    (config_dir / "config_test.yaml").write_text(
        """
custom:
  cuda_device: 0
  cuda_visible_devices: null
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(cuda_bootstrap, "_CONFIG_DIR", config_dir)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    cuda_bootstrap.bootstrap_cuda_visible_devices_from_argv(["--config-name=config_test"])

    assert cuda_bootstrap.os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_bootstrap_reads_selected_config_default(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    (config_dir / "config_test.yaml").write_text(
        """
defaults:
  - _self_
custom:
  cuda_device: 1
  cuda_visible_devices: null
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(cuda_bootstrap, "_CONFIG_DIR", config_dir)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    cuda_bootstrap.bootstrap_cuda_visible_devices_from_argv(["--config-name=config_test"])

    assert cuda_bootstrap.os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_bootstrap_reads_selected_config_visible_devices(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    (config_dir / "config_test.yaml").write_text(
        """
custom:
  cuda_device: 0
  cuda_visible_devices: "1"
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(cuda_bootstrap, "_CONFIG_DIR", config_dir)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    cuda_bootstrap.bootstrap_cuda_visible_devices_from_argv(["--config-name", "config_test"])

    assert cuda_bootstrap.os.environ["CUDA_VISIBLE_DEVICES"] == "1"
