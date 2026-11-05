"""The eval calculation supplies the H_norm denominator a model TRAINED with.

2026-10-30. lpno normalizes H_norm by the training split's maximum and, since
this date, records it beside the weights (`h_norm_denom.json`). Models trained before carry
no record, and their standalone evaluations refitted the denominator on the scored split
(every published single-alloy evaluation: ratios 0.918 to 1.445 on the test splits, up to
3.99 on one-case figure evaluations). `PinoEvalCalculation` now rebuilds the record from the
model's training-source simulations and stages it with the model, where the evaluation reads
it (`aiida_pino.utils.h_norm`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aiida_pino.utils.h_norm import H_NORM_RECORD_NAME, training_split_record

TI64 = {"name": "ti64", "properties": {"density_kg_m3": 4420.0, "specific_heat_j_kg_k": 546.0,
                                       "thermal_conductivity_w_m_k": 7.0, "latent_heat_j_kg": 345000.0,
                                       "liquidus_temperature_k": 1923.15},
        "optical": {"normal_absorptivity": 0.35}}
IN718 = {"name": "Inconel 718", "properties": {"density_kg_m3": 8190.0, "specific_heat_j_kg_k": 435.0,
                                               "thermal_conductivity_w_m_k": 8.9, "latent_heat_j_kg": 210000.0,
                                               "liquidus_temperature_k": 1608.15},
         "optical": {"normal_absorptivity": 0.35}}


def _attrs(power, speed, material=None):
    metadata = {"flow3d_case_manifest": {"input": {}, "material": material}} if material else {}
    return {"laser_power": power, "V_scan": speed, "simulation_metadata": metadata}


@pytest.mark.requires_lpno
def test_record_name_is_the_one_lpno_reads():
    from lpno.models.pipeline.h_norm_record import H_NORM_RECORD_NAME as pipeline_name

    assert H_NORM_RECORD_NAME == pipeline_name


@pytest.mark.requires_lpno
def test_record_is_the_max_over_the_moving_training_cases():
    record = training_split_record([_attrs(100.0, 0.5), _attrs(200.0, 0.5), _attrs(300.0, 0.0)])
    assert record["h_norm_denom"] == pytest.approx(200.0 / np.sqrt(0.5))
    assert record["n_cases"] == 2  # the stationary case is skipped, as compute_h_norm_denom does
    assert record["material_aware"] is False and record["radius_aware"] is False


@pytest.mark.requires_lpno
def test_material_aware_record_equals_the_pipelines_own_fit(tmp_path):
    """Same three cases served through lpno' staged store and its fit."""
    from lpno.models.pipeline.lazy_runtime import (
        RUNTIME_FORMAT_V2,
        ArrayDirCollectionStore,
        compute_h_norm_denom,
    )

    cases = [(200.0, 1.0, IN718), (150.0, 0.5, TI64), (250.0, 0.25, TI64)]
    for i, (power, speed, material) in enumerate(cases):
        member = tmp_path / f"m{i}"
        member.mkdir()
        arrays = {"P": np.full((1, 2, 2, 2), power, np.float32), "V_scan": np.full((1, 2, 2, 2), speed, np.float32)}
        for name, arr in arrays.items():
            np.save(member / f"{name}.npy", arr)
        (member / "manifest.json").write_text(json.dumps({
            "format": RUNTIME_FORMAT_V2, "source_kind": "simulation",
            "simulation_metadata": {"case_name": member.name,
                                    "flow3d_case_manifest": {"input": {}, "material": material}},
            "array_entries": [{"original_name": n, "filename": f"{n}.npy", "shape": list(a.shape)}
                              for n, a in arrays.items()]}))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format": RUNTIME_FORMAT_V2, "source_kind": "dataset_collection",
        "members": [{"relative_path": f"m{i}"} for i in range(len(cases))]}))
    fitted = compute_h_norm_denom(ArrayDirCollectionStore(tmp_path), range(len(cases)),
                                  time_as_batch=False, include_material=True)
    record = training_split_record([_attrs(p, v, m) for p, v, m in cases], include_material=True)
    assert record["h_norm_denom"] == pytest.approx(fitted, rel=1e-6)
    assert record["material_aware"] is True


# ------------------------------------------------------------- the calculation stages it


def _training_profile() -> dict:
    from aiida_pino.utils import load_training_profile_defaults

    profile = load_training_profile_defaults("config_NS_energy_PINO_flow3D")
    profile.update(run_name="run", hydra_overrides=[], split_mode="standard")
    return profile


def _training_source(pino_simulation_factory, cases):
    from aiida_pino.utils import build_collection_metadata_from_simulations, get_or_create_dataset_collection

    sims = [pino_simulation_factory("hnorm-train", group_name=f"hnorm_{i:03d}",
                                    metadata={"laser_power": p, "V_scan": v})
            for i, (p, v) in enumerate(cases)]
    metadata = build_collection_metadata_from_simulations(
        sims, collection_name="hnorm-train", dataset_role="base", composition_kind="import",
        derivation_profile={"include_time": True, "time_as_batch": False, "time_as_dim": True,
                            "moving_ref_frame": False, "moving_frame_steps": 1, "time_avg_window": 1,
                            "super_res_factor": 1})
    node, _ = get_or_create_dataset_collection(sims, metadata=metadata)
    return node


def _model(folder_data_factory, tmp_path, source, *, custom: dict, record: float | None = None):
    from aiida_pino.data import PinoModelData

    root = tmp_path / "artifacts"
    (root / ".hydra").mkdir(parents=True)
    (root / "u_network.0.pth").write_text("checkpoint", encoding="utf-8")
    (root / ".hydra" / "config.yaml").write_text(json.dumps({"custom": custom}), encoding="utf-8")
    if record is not None:
        (root / H_NORM_RECORD_NAME).write_text(json.dumps({"h_norm_denom": record}), encoding="utf-8")
    return PinoModelData.from_training_artifacts(
        folder_data_factory(root),
        metadata={"model_name": "m", "run_name": "m", "config_name": "config_NS_energy_PINO_flow3D",
                  "training_profile": _training_profile(),
                  "derivation_profile": {"include_time": True, "time_as_batch": False, "time_as_dim": True,
                                         "moving_ref_frame": False, "moving_frame_steps": 1,
                                         "time_avg_window": 1, "super_res_factor": 1},
                  "training_source_kind": "dataset", "training_source_pk": source.pk,
                  "training_source_uuid": source.uuid, "training_source_label": source.label,
                  "training_source_summary": {}, "primary_checkpoint": "u_network.0.pth",
                  "training_summary": {"imported": True}},
    ).store()


def _staged_record(local_code_factory, pino_dataset_factory, model):
    from aiida import orm
    from aiida.common.folders import SandboxFolder
    from aiida.engine.utils import instantiate_process
    from aiida.manage import get_manager

    from aiida_pino.calculations.eval import PinoEvalCalculation

    process = instantiate_process(
        get_manager().get_runner(), PinoEvalCalculation, code=local_code_factory("/bin/bash"),
        dataset=pino_dataset_factory("hnorm-eval"), training_artifacts=model,
        training_profile=orm.Dict(dict=_training_profile()))
    with SandboxFolder() as folder:
        process.prepare_for_submission(folder)
        path = Path(folder.get_abs_path("training_input")) / H_NORM_RECORD_NAME
        return json.loads(path.read_text()) if path.is_file() else None


CASES = [(100.0, 0.5), (200.0, 0.5), (150.0, 1.0)]


@pytest.mark.requires_lpno
def test_eval_stages_the_training_max_for_a_model_without_a_record(
        local_code_factory, pino_dataset_factory, pino_simulation_factory, folder_data_factory, tmp_path):
    source = _training_source(pino_simulation_factory, CASES)
    model = _model(folder_data_factory, tmp_path, source, custom={})
    staged = _staged_record(local_code_factory, pino_dataset_factory, model)
    assert staged is not None, "the evaluation would refit H_norm on the scored split"
    assert staged["h_norm_denom"] == pytest.approx(200.0 / np.sqrt(0.5))
    assert staged["n_cases"] == 3 and staged["training_source_uuid"] == source.uuid


@pytest.mark.requires_lpno
def test_eval_leaves_a_models_own_record_and_an_explicit_denominator_alone(
        local_code_factory, pino_dataset_factory, pino_simulation_factory, folder_data_factory, tmp_path):
    source = _training_source(pino_simulation_factory, CASES)
    own = _model(folder_data_factory, tmp_path / "own", source, custom={}, record=123.0)
    assert _staged_record(local_code_factory, pino_dataset_factory, own) == {"h_norm_denom": 123.0}
    explicit = _model(folder_data_factory, tmp_path / "explicit", source, custom={"h_norm_denom": 1.2e-5})
    assert _staged_record(local_code_factory, pino_dataset_factory, explicit) is None
