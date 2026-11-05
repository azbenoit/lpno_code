"""The 2026-08-10 model-import friction list (8 items).

Ranked by the reporter as "actively misleading" first: an import used to record
`training_stride = [1,1,1]` from a DEFAULTED derivation profile, so a reader
concluded full resolution for runs that actually read `random2579_ds_split` --
a 2,2,1 pre-strided copy (75x50x40 of 150x100x40). The stride lived in the
data-directory choice, which no profile field names.
"""

from __future__ import annotations

import json

import pytest

from aiida_pino.cli import parse_profile
from aiida_pino.data.pino_model import queryable_model_extras
from aiida_pino.utils.profiles import (
    IMPORT_PROFILE_REQUIRED_KEYS,
    TRAINING_PROFILE_REQUIRED_KEYS,
    resolve_artifacts_dir,
    validate_training_profile,
)


def test_a_defaulted_layout_records_no_stride(): # item 1
    metadata = {
        "derivation_profile": {"stride": [1, 1, 1]},
        "training_summary": {"imported": True, "derivation_profile_defaulted": True},
        "training_profile": {},
    }
    extras = queryable_model_extras(metadata)
    assert "training_stride" not in extras, "a default must not read as a measurement"
    assert "full_resolution" not in extras

    # An explicitly given layout IS an assertion, and is recorded.
    metadata["training_summary"]["derivation_profile_defaulted"] = False
    extras = queryable_model_extras(metadata)
    assert extras["training_stride"] == [1, 1, 1]
    assert extras["full_resolution"] is True


def test_import_does_not_demand_launch_only_keys():  # item 2
    assert {"cuda_device", "wandb_offline"} <= TRAINING_PROFILE_REQUIRED_KEYS
    assert not ({"cuda_device", "wandb_offline"} & IMPORT_PROFILE_REQUIRED_KEYS)

    finished_run = {"config_name": "config_flow3D_RIGNO", "run_name": "r0", "hydra_overrides": []}
    with pytest.raises(ValueError, match="cuda_device"):
        validate_training_profile(dict(finished_run), expected_split_mode="standard")
    profile = validate_training_profile(
        dict(finished_run), expected_split_mode="standard", for_import=True
    )
    assert profile["run_name"] == "r0"


def test_a_pasted_json_document_is_named_as_a_shape_error():  # item 3
    document = json.dumps({"config_name": "x", "run_name": "y", "hydra_overrides": []})
    with pytest.raises(ValueError) as excinfo:
        parse_profile([document])
    message = str(excinfo.value)
    assert "@file.json" in message and "profile-keys" in message
    assert "config_name" not in message, "must not read as a complaint about the user's key"


def test_profile_keys_is_advertised_where_the_flag_is():  # item 4
    import argparse

    from aiida_pino.cli import build_parser

    def _walk(parser):
        yield parser
        for action in parser._actions:  # noqa: SLF001
            if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
                for sub in action.choices.values():
                    yield from _walk(sub)

    # argparse hard-wraps help text, so compare on collapsed whitespace.
    texts = [" ".join(p.format_help().split()) for p in _walk(build_parser())]
    advertising = [t for t in texts if "profile-keys training" in t]
    assert advertising, "every --training-profile flag must name the key-list command"
    imports = [t for t in texts if "import-model" in t and "--training-profile" in t]
    assert imports and all("profile-keys training" in t for t in imports)


def test_artifacts_path_finds_the_nested_run_and_refuses_an_empty_one(tmp_path):  # item 5
    outer = tmp_path / "outputs" / "run1"
    inner = outer / "outputs" / "run1"
    inner.mkdir(parents=True)
    (inner / "u_network.0.pth").write_bytes(b"ckpt")
    (outer / "wandb").mkdir()

    assert resolve_artifacts_dir(outer) == str(inner), "pointing at the outer dir must descend"
    assert resolve_artifacts_dir(inner) == str(inner)

    empty = tmp_path / "logs_only"
    (empty / "wandb").mkdir(parents=True)
    with pytest.raises(ValueError, match="no checkpoint"):
        resolve_artifacts_dir(empty)


def test_two_runs_below_one_path_is_refused_not_guessed(tmp_path):  # item 5
    root = tmp_path / "outputs"
    for name in ("runA", "runB"):
        d = root / name
        d.mkdir(parents=True)
        (d / "u_network.0.pth").write_bytes(b"ckpt")
    with pytest.raises(ValueError, match="checkpoint directories"):
        resolve_artifacts_dir(root)


def test_import_request_carries_val_source_and_dry_run():  # items 7, 8
    from aiida_pino.console.schemas import ImportModelRequest

    req = ImportModelRequest(
        python_code="c", training_source_pk=1, validation_source_pk=2,
        artifacts_path="/x", dry_run=True,
    )
    assert req.validation_source_pk == 2 and req.dry_run is True
    # defaults keep the old shape working
    plain = ImportModelRequest(python_code="c", training_source_pk=1, artifacts_path="/x")
    assert plain.validation_source_pk is None and plain.dry_run is False


def test_freezing_an_imported_profile_does_not_invent_the_dose_stamp(  # item 9
    pino_dataset_factory,
):
    """A pre-stamp import's `power_weighted_dose: None` must survive registration.

    The import workchain validates with `for_import=True`, which correctly leaves the
    flag None ("inventing True for a run that trained power-blind would be fabricated
    provenance"). `build_model_metadata` then RE-validated the raw profile without the
    flag, so the submission-time None->True default fired at freeze time and the model
    registered as power-weighted. Its evals then emitted
    `++custom.power_weighted_dose=true` and fed the checkpoint a dose channel it never
    trained on -- measured 2026-08-21 on three demo-model imports,
    where every frozen profile read True.
    """
    from aiida import orm

    from aiida_pino.utils.model_registry import build_model_metadata
    from aiida_pino.utils.profiles import validate_training_profile

    profile = validate_training_profile(
        {"config_name": "config_flow3D_RIGNO_randompath", "run_name": "imported",
         "hydra_overrides": []},
        expected_split_mode="standard",
        for_import=True,
    )
    assert profile.get("power_weighted_dose") is None  # precondition: import keeps None

    artifacts = orm.FolderData().store()
    summary = orm.Dict(dict={"imported": True}).store()
    source = pino_dataset_factory("importfreeze")
    metadata = build_model_metadata(
        artifacts, summary, orm.Dict(dict=profile).store(), source,
        {"time_as_batch": True},
        model_name="imported",
    )
    frozen = metadata["training_profile"]
    assert frozen.get("power_weighted_dose") is None, (
        "freeze-time validation re-applied the submission-time None->True default; "
        "the frozen profile must preserve what the import validated"
    )
