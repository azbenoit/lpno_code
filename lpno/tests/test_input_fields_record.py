"""input_fields.json must record the input-field settings a training actually used, including code defaults and
environment overrides that leave no trace in the stored config."""
import json
from types import SimpleNamespace

from lpno.models.pipeline.input_fields_record import (INPUT_FIELDS_RECORD_NAME, input_field_settings,
                                                      write_input_fields_record)


def _cfg(**kw):
    base = dict(eval_only=False, str_fno_input_keys=["V_scan", "P", "conduction_field"])
    base.update(kw)
    return SimpleNamespace(**base)


def test_defaults_are_written_out(tmp_path, monkeypatch):
    monkeypatch.delenv("PINO_POWER_WEIGHTED_CONDUCTION", raising=False)
    monkeypatch.delenv("PINO_POWER_WEIGHTED_DOSE", raising=False)
    path = write_input_fields_record(_cfg(), network_dir=tmp_path)
    rec = json.loads(path.read_text())
    assert path.name == INPUT_FIELDS_RECORD_NAME
    assert rec["conduction_field"] is True
    assert rec["conduction_source_sigma_factor"] == 1.0          # the original channel, sigma = r0
    assert rec["power_weighted_conduction"] is False and rec["power_weighted_dose"] is False


def test_explicit_settings_are_recorded(tmp_path, monkeypatch):
    monkeypatch.delenv("PINO_POWER_WEIGHTED_CONDUCTION", raising=False)
    cfg = _cfg(conduction_source_sigma_factor=0.5, power_weighted_conduction=True, power_weighted_dose=True)
    rec = json.loads(write_input_fields_record(cfg, network_dir=tmp_path).read_text())
    assert (rec["conduction_source_sigma_factor"], rec["power_weighted_conduction"], rec["power_weighted_dose"]) == \
        (0.5, True, True)


def test_environment_override_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("PINO_POWER_WEIGHTED_CONDUCTION", "1")
    assert input_field_settings(_cfg())["power_weighted_conduction"] is True


def test_no_record_at_evaluation(tmp_path):
    assert write_input_fields_record(_cfg(eval_only=True), network_dir=tmp_path) is None
    assert not (tmp_path / INPUT_FIELDS_RECORD_NAME).exists()


def test_both_training_paths_write_it():
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "lpno" / "models"
    assert "write_input_fields_record(" in (src / "ns_energy_pipeline.py").read_text()
    assert "write_input_fields_record(" in (src / "pipeline" / "preprocessing_datasets.py").read_text()
