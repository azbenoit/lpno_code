"""custom.val_max_cases: seeded in-training validation subset (lazy path)."""
import numpy as np

from lpno.models.pipeline.preprocessing_datasets import (
    _effective_val_max_cases,
    _subsample_val_cases,
)


def test_zero_or_large_keeps_everything():
    idx = list(range(10, 62))
    assert _subsample_val_cases(idx, 0, seed=42) == idx
    assert _subsample_val_cases(idx, len(idx), seed=42) == idx
    assert _subsample_val_cases(idx, 999, seed=42) == idx


def test_subset_is_seeded_sorted_and_sized():
    idx = list(range(100, 152))
    a = _subsample_val_cases(idx, 5, seed=42)
    b = _subsample_val_cases(idx, 5, seed=42)
    c = _subsample_val_cases(idx, 5, seed=7)
    assert a == b  # deterministic under the same seed
    assert a != c  # seed-dependent
    assert len(a) == 5
    assert a == sorted(a)  # order-preserving
    assert set(a) <= set(idx)


class _Cfg:
    """Minimal RunConfig stand-in for the effective-cap decision."""

    def __init__(self, val_max_cases=5, eval_only=False):
        self.val_max_cases = val_max_cases
        self.eval_only = eval_only


def test_eval_only_scores_the_whole_split():
    """The cap is an in-training speed knob. NS_Energy_laser_PINO_eval reuses
    run_training_pipeline -> build_datasets, so a non-zero default would
    silently score 5 of 52 val cases and report it as the split metric."""
    assert _effective_val_max_cases(_Cfg(val_max_cases=5, eval_only=True)) == 0
    assert _effective_val_max_cases(_Cfg(val_max_cases=0, eval_only=True)) == 0


def test_training_run_keeps_the_cap():
    assert _effective_val_max_cases(_Cfg(val_max_cases=5, eval_only=False)) == 5
    assert _effective_val_max_cases(_Cfg(val_max_cases=0, eval_only=False)) == 0


def test_shipped_configs_bound_in_training_validation():
    """A full-split in-training pass cost 2520 s per validation on 52 cases --
    ~30% of a 32k-step run. The random-path configs must ship a cap, not 0."""
    import re
    from pathlib import Path

    conf_dir = Path(__file__).resolve().parents[1] / "src/lpno/conf/hydra"
    configs = sorted(conf_dir.glob("config_flow3D_RIGNO_randompath*.yaml"))
    assert configs, f"no random-path configs found under {conf_dir}"
    for path in configs:
        found = re.search(r"^\s*val_max_cases:\s*(\d+)", path.read_text(), re.M)
        assert found, f"{path.name} does not set custom.val_max_cases"
        assert int(found.group(1)) > 0, f"{path.name} validates on the whole split"
