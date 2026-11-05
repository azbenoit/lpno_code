"""CalcJob that launches a `flow3D_scripts` FLOW-3D run as a tracked job."""

from __future__ import annotations

import json

from aiida import orm
from aiida.common.folders import Folder

from .base import ScriptCalculation
from aiida_pino.utils import validate_flow3d_run_execution_profile


def _validate_execution_profile_port(node: orm.Dict, _port=None) -> str | None:
    try:
        validate_flow3d_run_execution_profile(node.get_dict())
    except ValueError as exc:
        return str(exc)
    return None


class Flow3DRunCalculation(ScriptCalculation):
    """Run `flow3D_scripts/python/run_rectangular_case.py` through the bundled adapter.

    The ``run_config`` Dict is the full flow3D_scripts config JSON, passed
    through verbatim — every config field the runner supports is available
    without aiida-pino changes. The NPZ export stays on disk under the run's
    ``out_root``; the run summary records its absolute path and only small
    curated files are retrieved.

    The runner deletes the multi-GB ``flsgrf*`` result database itself once the
    NPZ export completes, so it is normally gone by the time this CalcJob
    finishes — nothing here reads it, and the NPZ ingest path does not either.
    Pass ``keep_flsgrf`` in the execution profile when a later re-export needs
    it (the legacy ``flow3d_to_vtk`` pvpython ingest is the one case that does).
    """

    script_name = "flow3d_run_adapter.py"
    parser_entry_point = "pino.flow3d_run"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "run_config",
            valid_type=orm.Dict,
            help="Full flow3D_scripts case config (JSON), passed to the runner verbatim.",
        )
        spec.input(
            "execution_profile",
            valid_type=orm.Dict,
            validator=_validate_execution_profile_port,
            help="Runner options: flow3d_scripts_root, out_root, case_name, cores, "
            "threads_per_rank, prepare_only, overwrite, export, flow3d_env, post_home.",
        )
        spec.output("run_summary", valid_type=orm.Dict)
        spec.output("run_artifacts", valid_type=orm.FolderData, required=False)
        spec.exit_code(520, "ERROR_MISSING_SUMMARY", message="FLOW-3D run did not produce `flow3d_run_summary.json`.")
        spec.exit_code(521, "ERROR_RUN_FAILED", message="FLOW-3D run adapter reported a failure.")
        spec.exit_code(522, "ERROR_MISSING_ARTIFACTS", message="FLOW-3D run did not produce `run_artifacts`.")

    def prepare_for_submission(self, folder: Folder):
        script_name = self.stage_script(folder)
        config_name = "run_config.json"
        with folder.open(config_name, "w", encoding="utf-8") as handle:
            json.dump(self.inputs.run_config.get_dict(), handle, indent=2, sort_keys=True)
        profile_name = "execution_profile.json"
        with folder.open(profile_name, "w", encoding="utf-8") as handle:
            json.dump(self.inputs.execution_profile.get_dict(), handle, indent=2, sort_keys=True)

        codeinfo = self.build_codeinfo(
            [script_name, "--config-file", config_name, "--profile-file", profile_name]
        )
        return self.build_calcinfo(
            codeinfo,
            [self.options.output_filename, "flow3d_run_summary.json", "run_artifacts"],
        )
