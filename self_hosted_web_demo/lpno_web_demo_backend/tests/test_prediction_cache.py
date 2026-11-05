from __future__ import annotations

from app.config import Settings
from app.schemas import PathSpec
from app.service import InferenceService
from tests.conftest import needs_served_gt

pytestmark = needs_served_gt(Settings().path_data_dir)  # drawn paths still take the grid from a case NPZ


def _service(cache_entries: int) -> tuple[InferenceService, list[str]]:
    settings = Settings(use_stub_runtime=True, prediction_cache_entries=cache_entries)
    svc = InferenceService.build_default(settings)
    calls: list[str] = []
    orig = svc.path_adapter.infer

    def counting(model, case):
        calls.append(model)
        return orig(model, case)

    svc.path_adapter.infer = counting  # type: ignore[method-assign]
    return svc, calls


# A drawn path (no preset/GT needed); resolve() clamps + resamples it.
PATH = PathSpec(points_xy_m=[[0.0, 0.0], [1.0e-4, 0.0]])


def test_same_model_and_path_infers_once() -> None:
    svc, calls = _service(4)
    svc._predict_path_cached("fno_newphys", PATH)
    svc._predict_path_cached("fno_newphys", PATH)
    assert calls == ["fno_newphys"]  # second call served from cache


def test_multiple_models_cached_independently() -> None:
    svc, calls = _service(4)
    svc._predict_path_cached("fno_newphys", PATH)
    svc._predict_path_cached("rigno_newphys", PATH)
    svc._predict_path_cached("fno_newphys", PATH)  # fno still cached -> no re-infer
    svc._predict_path_cached("rigno_newphys", PATH)  # still cached -> no re-infer
    assert calls == ["fno_newphys", "rigno_newphys"]


def test_frees_gpu_cache_on_miss_only() -> None:
    # A prediction-cache miss runs a forward and must release the CUDA allocator's
    # reserved pool (else it accumulates the peak of every model -> 17 GB); a hit
    # does no GPU work and must not touch it. Regression for the 17 GB footprint.
    # (patch on the class: InferenceService is a frozen dataclass.)
    from unittest.mock import patch

    svc, calls = _service(4)
    with patch.object(type(svc), "_free_gpu_cache", autospec=True) as freed:
        svc._predict_path_cached("fno_newphys", PATH)  # miss -> forward -> free
        svc._predict_path_cached("fno_newphys", PATH)  # hit  -> no forward -> no free
    assert calls == ["fno_newphys"]
    assert freed.call_count == 1


def test_lru_evicts_oldest_beyond_capacity() -> None:
    svc, calls = _service(2)
    svc._predict_path_cached("fno_newphys", PATH)  # [fno]
    svc._predict_path_cached("rigno_newphys", PATH)  # [fno, rigno_1601]
    svc._predict_path_cached("fno_newphys", PATH)  # hit, LRU -> [rigno_1601, fno]
    svc._predict_path_cached("gaot_newphys", PATH)  # evicts rigno_1601 -> [fno, gaot_1601]
    svc._predict_path_cached("fno_newphys", PATH)  # still cached -> no re-infer
    svc._predict_path_cached("rigno_newphys", PATH)  # was evicted -> re-infer
    assert calls == ["fno_newphys", "rigno_newphys", "gaot_newphys", "rigno_newphys"]


def test_cached_forward_miss_hit_and_failure() -> None:
    """The one miss-path wrapper the three routes share: a hit does no work and
    frees nothing; a miss records activity, frees after the forward and stores;
    a failing forward still frees and propagates unchanged."""
    from collections import OrderedDict

    from unittest.mock import patch

    svc, _ = _service(4)
    cache: OrderedDict = OrderedDict()
    runs: list[int] = []

    def run():
        runs.append(1)
        return "value"

    with patch.object(type(svc), "_free_gpu_cache", autospec=True) as freed:
        assert svc._cached_forward(cache, "k", None, run) == "value"
        assert svc._cached_forward(cache, "k", None, run) == "value"
    assert runs == [1] and cache == {"k": "value"} and freed.call_count == 1

    class Boom(RuntimeError):
        pass

    def fail():
        raise Boom("x")

    with patch.object(type(svc), "_free_gpu_cache", autospec=True) as freed:
        try:
            svc._cached_forward(cache, "k2", None, fail)
        except Boom:
            pass
        else:
            raise AssertionError("failure must propagate")
    assert freed.call_count == 1 and "k2" not in cache
