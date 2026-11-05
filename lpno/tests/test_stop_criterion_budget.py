"""Early stopping must not silently truncate a configured training budget.

Regression for 2026-08-26: `rigno_calib_straight_tb_expLR_democlone3` (model 69030)
stopped at 16000 of 32000 steps and reported exit 0. `config_flow3D_RIGNO_randompath`
set `stop_criterion.patience: 4000` against `training.max_steps: 32000`, and the
criterion watched `l2_relative_error_T` while the best checkpoint was selected on
`beambox_mae_T`. The two disagreed: l2 last improved at step 12500 and the stop fired
exactly 4000 steps later, while beambox_mae_T was still improving monotonically and
reached its best at 16500 -- the very step training ended.

Two invariants:

1. A config that declares both `stop_criterion.patience` and `training.max_steps`
   must have patience >= max_steps, so the run reaches its budget. Early stopping is
   documented in these configs as "a safety net, not the usual exit".
2. Where a stop metric is declared it must be the metric that selects the checkpoint
   (`beambox_mae_T`), so stopping and selection cannot disagree.
"""

import re
import unittest
from pathlib import Path

CONF = Path(__file__).resolve().parents[1] / "src" / "lpno" / "conf" / "hydra"
SELECTION_METRIC = "beambox_mae_T"


def _block(text: str, header: str) -> str:
    m = re.search(rf"^{header}:\n((?:[ \t]+.*\n|\n)*)", text, re.M)
    return m.group(1) if m else ""


def _int_field(block: str, field: str):
    m = re.search(rf"^\s+{field}:\s*([0-9_]+)\s*$", block, re.M)
    return int(m.group(1).replace("_", "")) if m else None


def _str_field(block: str, field: str):
    m = re.search(rf"""^\s+{field}:\s*["']?([A-Za-z_0-9]+)["']?\s*$""", block, re.M)
    return m.group(1) if m else None


def _flow3d_configs():
    return sorted(CONF.glob("config_flow3D_*.yaml"))


class TestStopCriterionBudget(unittest.TestCase):
    def test_configs_exist(self):
        self.assertTrue(_flow3d_configs(), f"no flow3D configs under {CONF}")

    def test_patience_never_truncates_max_steps(self):
        offenders = []
        for path in _flow3d_configs():
            text = path.read_text()
            patience = _int_field(_block(text, "stop_criterion"), "patience")
            max_steps = _int_field(_block(text, "training"), "max_steps")
            if patience is None or max_steps is None:
                continue
            if patience < max_steps:
                offenders.append(f"{path.name}: patience={patience} < max_steps={max_steps}")
        self.assertEqual(
            [], offenders,
            "early stopping can fire before the configured budget, truncating training "
            "while it reports exit 0:\n  " + "\n  ".join(offenders),
        )

    def test_stop_metric_matches_checkpoint_selection(self):
        offenders = []
        for path in _flow3d_configs():
            metric = _str_field(_block(path.read_text(), "stop_criterion"), "metric")
            if metric is None:
                continue
            if metric != SELECTION_METRIC:
                offenders.append(f"{path.name}: stop metric={metric!r}")
        self.assertEqual(
            [], offenders,
            f"stop criterion must watch {SELECTION_METRIC!r}, the metric that selects the "
            "best checkpoint; otherwise training can stop while the selected metric is "
            "still improving:\n  " + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
