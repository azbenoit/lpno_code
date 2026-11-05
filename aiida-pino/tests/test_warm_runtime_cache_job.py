"""The runtime-cache export runs as its OWN scheduled job, not inside the daemon worker.

`do_upload` calls `prepare_for_submission` synchronously on the daemon worker's event
loop. A derived-collection export there is minutes a member, so for its whole duration
that worker cannot service anything: on 2026-09-11 three zero-copy fine-tunes waited more
than an hour behind an unrelated 419-member export, two other workers sat at
0.4 % CPU unable to touch the hoarded tasks, eight `verdi process kill`s timed out at 20 s
because the kill RPC is answered by the same blocked loop, and a daemon restart -- the only
remedy -- orphaned the export's spawn pool.

`TrainModelWorkChain` now submits `WarmRuntimeCacheCalculation` for each LOCAL source
whose cache is not built and awaits it like the smoke job. The export becomes a process
the scheduler owns: the loop stays free, a kill reaches a real process group, and a local
job survives a daemon restart the way every local training already does.
"""

from __future__ import annotations

from types import SimpleNamespace

from aiida import orm
from aiida.common.folders import SandboxFolder
from aiida.engine.utils import instantiate_process
from aiida.manage import get_manager

from aiida_pino.calculations.warm_runtime_cache import WarmRuntimeCacheCalculation
from aiida_pino.workflows.train_model import warm_targets


def test_prepare_writes_a_script_that_warms_the_dataset_and_leaves_a_result_marker(
    local_code_factory, pino_dataset_factory
):
    code = local_code_factory("/bin/bash")
    dataset = pino_dataset_factory()
    process = instantiate_process(
        get_manager().get_runner(), WarmRuntimeCacheCalculation, code=code, dataset=dataset
    )
    with SandboxFolder() as folder:
        calcinfo = process.prepare_for_submission(folder)
        assert WarmRuntimeCacheCalculation.SCRIPT_NAME in folder.get_content_list()
        with folder.open(WarmRuntimeCacheCalculation.SCRIPT_NAME) as handle:
            script = handle.read()
    assert dataset.uuid in script
    assert "warm_dataset_runtime_cache" in script
    assert WarmRuntimeCacheCalculation.RESULT_NAME in calcinfo.retrieve_list
    assert calcinfo.codes_info[0].cmdline_params[0] == WarmRuntimeCacheCalculation.SCRIPT_NAME


class _Collection:
    def __init__(self, pk, profile, built=False):
        self.pk = pk
        self.uuid = f"{pk:08x}-0000-0000-0000-000000000000"
        self._profile = profile
        self._built = built

    def get_dataset_metadata(self):
        return {"derivation_profile": dict(self._profile), "dataset_role": "derived" if self._profile else "base"}

    def iter_member_simulations(self):
        return [SimpleNamespace(get_simulation_metadata=lambda: {"array_shapes": {"T": [2, 1, 2, 2, 2]}})]

    def export_runtime_arraydir(self, target_dir, *, runtime_fields=None, allow_pack_index=True):
        raise AssertionError("selection must not export")


def test_only_local_unbuilt_copying_sources_are_warmed(monkeypatch, tmp_path):
    monkeypatch.setenv("AIIDA_PINO_RUNTIME_CACHE", str(tmp_path))
    base = _Collection(1, {})
    derived = _Collection(2, {"derived_fields": ["conduction_field"]})
    picked = warm_targets({"train": derived, "val": base}, transport_type="core.local")
    assert [c.pk for c in picked.values()] == [2], "the base is zero-copy; only the derived copy needs a warm"


def test_the_same_collection_for_train_and_val_is_warmed_once(monkeypatch, tmp_path):
    monkeypatch.setenv("AIIDA_PINO_RUNTIME_CACHE", str(tmp_path))
    derived = _Collection(2, {"derived_fields": ["conduction_field"]})
    picked = warm_targets({"train": derived, "val": derived}, transport_type="core.local")
    assert len(picked) == 1


def test_a_remote_target_is_never_warmed_here(monkeypatch, tmp_path):
    """Remote caches are staged explicitly (`stage-remote`); presubmit refuses otherwise."""
    monkeypatch.setenv("AIIDA_PINO_RUNTIME_CACHE", str(tmp_path))
    derived = _Collection(2, {"derived_fields": ["conduction_field"]})
    assert warm_targets({"train": derived}, transport_type="core.ssh_async") == {}


def test_the_workchain_warms_before_it_submits_anything_that_stages():
    import inspect

    from aiida_pino.workflows.train_model import TrainModelWorkChain

    source = inspect.getsource(TrainModelWorkChain.define)
    assert source.index("cls.warm_runtime_caches") > source.index("cls.resolve_sources")
    assert source.index("cls.harvest_warm") < source.index("cls.prepare_vram_reservation")


def test_a_job_without_a_result_marker_is_a_failure_not_finished_zero():
    """2026-10-07: a warm job whose export raised showed `Finished [0]`."""
    from aiida_pino.calculations.warm_runtime_cache import WarmRuntimeCacheCalculation

    spec = WarmRuntimeCacheCalculation.spec()
    assert spec.exit_codes.ERROR_NO_CACHE_PUBLISHED.status == 310
    assert "parse" in WarmRuntimeCacheCalculation.__dict__
