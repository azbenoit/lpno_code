"""A submission that will EXPORT a dataset before it can reach the GPU says so, at submit time.

"Waiting for transport task: upload" is not a file copy: `warm_dataset_runtime_cache` documents
that the runtime-cache export runs INSIDE `do_upload`, on the submitting machine, and the training
cannot start until it finishes. On 2026-09-11 three local fine-tunes sat behind a single-core
61-member export for two hours with both GPUs idle, and `aiida-pino submit` had printed nothing
about it. `staging_report` said "local computer: jobs read the runtime
cache directly" -- true only when the cache exists.
"""

from __future__ import annotations

from tempfile import TemporaryDirectory
from types import SimpleNamespace

from aiida_pino.utils.presubmit import local_build_report


class _Collection:
    uuid = "cafe0000-0000-0000-0000-000000000000"
    pk = 7

    def __init__(self, profile, n=3):
        self._profile = profile
        self._n = n

    def get_dataset_metadata(self):
        return {"derivation_profile": dict(self._profile), "dataset_role": "derived"}

    def iter_member_simulations(self):
        return [SimpleNamespace(get_simulation_metadata=lambda: {"array_shapes": {"T": [10, 1, 4, 4, 4]}})
                for _ in range(self._n)]

    def export_runtime_arraydir(self, target_dir, *, runtime_fields=None, allow_pack_index=True):
        raise AssertionError("a report must never export")


def test_a_missing_cache_for_a_copying_derivation_is_announced(monkeypatch):
    with TemporaryDirectory() as tmp:
        monkeypatch.setenv("AIIDA_PINO_RUNTIME_CACHE", tmp)
        report = local_build_report(_Collection({"derived_fields": ["conduction_field"]}))
    assert report["needed"] is True
    assert report["members"] == 3
    assert "upload" in report["detail"] and "GPU" in report["detail"]


def test_an_identity_derivation_needs_no_build(monkeypatch):
    with TemporaryDirectory() as tmp:
        monkeypatch.setenv("AIIDA_PINO_RUNTIME_CACHE", tmp)
        report = local_build_report(_Collection({}))
    assert report["needed"] is False


def test_warm_uses_a_per_attempt_build_dir():
    import inspect

    from aiida_pino.calculations import base

    source = inspect.getsource(base.warm_dataset_runtime_cache)
    assert "_fresh_build_dir" in source, "warm is the second copy of the builder and must not diverge"
    assert ".build-{cache_name}-{os.getpid()}" not in source


def test_the_cli_prints_notes_one_per_line():
    from aiida_pino.cli import _render

    out = _render({"pk": 1, "notes": ["train staging: zero-copy: served through a pack index", "val staging: x"]})
    assert "notes:\n  - train staging: zero-copy" in out
    assert "[" not in out.split("notes:")[1], "not a JSON list"


def test_resubmit_overrides_can_reach_the_derivation_profile():
    """`--override derivation.derived_fields=[]` repairs the field the 09-10 promotion corrupted.

    `resubmit --override` applied to the cloned TRAINING profile only, so the natural
    "rerun it correctly" command could not touch the derivation, and the full `submit train-eval`
    invocation had to be rebuilt by hand (2026-09-11).
    """
    from aiida_pino.console.service import split_resubmit_overrides

    training, derivation = split_resubmit_overrides(
        {"run_name": "x", "derivation.derived_fields": [], "derivation.crop_z_top": 0}
    )
    assert training == {"run_name": "x"}
    assert derivation == {"derived_fields": [], "crop_z_top": 0}
