"""Build a dataset's local runtime cache as a scheduled job, off the daemon worker.

The export used to run inside `do_upload` -- `prepare_for_submission` is called
synchronously on the daemon worker's event loop -- so for the minutes-a-member duration
of a derived-collection export that worker could service nothing: not the zero-copy jobs
queued behind it (three fine-tunes waited over an hour behind one 419-member export on
2026-09-11), not `verdi process kill` (eight timed out at 20 s; the kill RPC is answered
by the same loop), and a daemon restart, the only remedy, orphaned the export's spawn
pool. Running the same export as its own local CalcJob gives it a process the scheduler
owns: the loop stays free, a kill reaches a process group, and a local job survives a
daemon restart the way every local training already does.

LOCAL ONLY. The script imports `aiida_pino` and loads the profile, which exist on this
workstation and not on a compute box; a remote runtime cache is staged explicitly with
`aiida-pino stage-remote`, and presubmit refuses a remote job whose cache is absent.
"""

from __future__ import annotations

from aiida import orm
from aiida.common import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob
from aiida.engine.processes.calcjobs.calcjob import CalcJobProcessSpec

from aiida_pino.data.pino_dataset_collection import PinoDatasetCollectionData


def warm_script(dataset_uuid: str, runtime_fields: list | None, result_name: str) -> str:
    """The script the job runs, as text.

    Everything lives under `main()` behind a `__main__` guard. The export starts a SPAWN
    pool, and every spawned child re-imports this file as `__mp_main__`: with the work at
    module level each child ran the export again, blocked on the `.materialise.lock` flock
    its parent held, and the parent waited on the children -- a deadlock, reproduced twice
    on 2026-09-14 (self-improve). `tests/test_warm_script_spawn_safe.py` pins it.
    """
    return f'''"""Written by WarmRuntimeCacheCalculation; runs under the local scheduler."""
import json, sys, traceback


def main() -> int:
    from aiida import load_profile, orm
    load_profile()
    from aiida_pino.calculations.base import warm_dataset_runtime_cache
    try:
        path = warm_dataset_runtime_cache(orm.load_node({dataset_uuid!r}),
                                          runtime_fields={runtime_fields!r}, transport_type="core.local")
    except Exception:  # noqa: BLE001 - the marker's absence is the failure signal
        traceback.print_exc()
        return 1
    with open({result_name!r}, "w", encoding="utf-8") as handle:
        json.dump({{"cache": str(path)}}, handle)
    print(f"[warm] runtime cache ready at {{path}}", flush=True)
    return 0


if __name__ == "__main__":  # a spawn-pool child imports this file; it must run nothing
    sys.exit(main())
'''


class WarmRuntimeCacheCalculation(CalcJob):
    """`warm_dataset_runtime_cache(dataset)` in a process of its own."""

    SCRIPT_NAME = "warm_runtime_cache.py"
    RESULT_NAME = "warm_result.json"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec):
        super().define(spec)
        spec.input("dataset", valid_type=PinoDatasetCollectionData, help="collection whose local runtime cache to build")
        spec.input("runtime_fields", valid_type=orm.List, required=False,
                   help="field trimming, exactly as the consuming training passes it -- it is part of the cache name")
        spec.inputs["metadata"]["options"]["resources"].default = {"num_machines": 1, "num_mpiprocs_per_machine": 1}
        spec.inputs["metadata"]["options"]["withmpi"].default = False
        spec.inputs["metadata"]["options"]["output_filename"].default = "aiida.out"
        spec.exit_code(310, "ERROR_NO_CACHE_PUBLISHED",
                       message="The export failed and published no runtime cache; see aiida.out and the scheduler stderr.")

    def parse(self, retrieved_temporary_folder=None, existing_exit_code=None):
        """Fail the job when the export left no result marker.

        Without a parser a job whose export raised still showed `Finished [0]`: the script's
        own exit status never reaches the node. The workchain caught it (exit 307), but the
        job itself looked successful in `verdi process list` (2026-10-07, reproduction test).
        """
        exit_code = super().parse(retrieved_temporary_folder, existing_exit_code)
        if exit_code is not None and exit_code.status:
            return exit_code
        retrieved = self.node.outputs.retrieved if "retrieved" in self.node.outputs else None
        names = retrieved.base.repository.list_object_names() if retrieved is not None else []
        if self.RESULT_NAME not in names:
            return self.exit_codes.ERROR_NO_CACHE_PUBLISHED
        return exit_code

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        fields = list(self.inputs.runtime_fields.get_list()) if "runtime_fields" in self.inputs else None
        script = warm_script(self.inputs.dataset.uuid, fields, self.RESULT_NAME)
        with folder.open(self.SCRIPT_NAME, "w", encoding="utf-8") as handle:
            handle.write(script)
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [self.SCRIPT_NAME]
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.withmpi = False
        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = [self.RESULT_NAME, self.options.output_filename, "_scheduler-stderr.txt"]
        return calcinfo
