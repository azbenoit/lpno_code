"""`graph_nodes` must be an accepted training_profile key.

`_check_host_ram_against_loader_settings` scales the full-resolution host-RAM floor by
the run's grid: `full_res_host_ram_floor_mb(profile.get('graph_nodes'))`. But
`graph_nodes` was never added to `DEFAULT_TRAINING_PROFILE`, and
`TRAINING_PROFILE_KEYS = frozenset(DEFAULT_TRAINING_PROFILE)`, so a submission
declaring it is refused with "unknown training_profile key(s): `graph_nodes`".

The grid-aware branch was therefore unreachable: every run inherited the flat 40960
anchor, measured on a 600k-node GAOT build peaking at ~57 GB. Measured 2026-08-26, a
cropped 90x40x30 RIGNO arm (108k nodes) has an RSS of 8188 MiB including loader
children and was being made to book 40960 -- which caps a 128 GB box at TWO concurrent
runs, because host RAM is booked machine-wide.

This is the same defect the file already records for `gpu_seat_priority`: "the key was
only ever declared on the EVALUATION profile, so the console's unknown-key check refused
it on a training and the working code was unreachable."
"""

import unittest

from aiida_pino.utils.profiles import (
    DEFAULT_TRAINING_PROFILE,
    TRAINING_PROFILE_KEYS,
    _check_host_ram_against_loader_settings,
    full_res_host_ram_floor_mb,
)

CROPPED_NODES = 90 * 40 * 30          # 108_000, the read-time-cropped straight-path grid
FULL_NODES = 150 * 100 * 40           # 600_000, the anchor GAOT grid


def _profile(host_ram_mb, graph_nodes=None, workers=2):
    p = {
        "host_ram_mb": host_ram_mb,
        "env": {"PINO_LOADER_WORKERS": str(workers)},   # no PINO_LAZY_STRIDE -> "full resolution"
    }
    if graph_nodes is not None:
        p["graph_nodes"] = graph_nodes
    return p


class TestGraphNodesIsAcceptedAndUsed(unittest.TestCase):
    def test_graph_nodes_is_an_accepted_profile_key(self):
        self.assertIn(
            "graph_nodes", DEFAULT_TRAINING_PROFILE,
            "graph_nodes is read by _check_host_ram_against_loader_settings but is not a "
            "declared key, so every submission that sets it is refused and the grid-aware "
            "floor can never be reached",
        )
        self.assertIn("graph_nodes", TRAINING_PROFILE_KEYS)

    def test_declaring_a_small_grid_lowers_the_floor(self):
        """108k nodes must not inherit the 600k-node anchor."""
        self.assertEqual(40960, full_res_host_ram_floor_mb(FULL_NODES))
        self.assertLess(full_res_host_ram_floor_mb(CROPPED_NODES), 40960)

    def test_cropped_arm_passes_at_24576_when_the_grid_is_declared(self):
        """16384 base + 2 workers x 4096 = 24576, above the 8192 floor for this grid."""
        _check_host_ram_against_loader_settings(_profile(24576, graph_nodes=CROPPED_NODES))

    def test_cropped_arm_is_still_refused_below_the_worker_requirement(self):
        with self.assertRaises(ValueError):
            _check_host_ram_against_loader_settings(_profile(16384, graph_nodes=CROPPED_NODES))

    def test_undeclared_grid_keeps_the_historical_flat_floor(self):
        """Unchanged behaviour for every profile that does not declare a grid."""
        self.assertEqual(40960, full_res_host_ram_floor_mb(None))
        with self.assertRaises(ValueError):
            _check_host_ram_against_loader_settings(_profile(24576))

    def test_a_full_resolution_grid_still_demands_the_anchor(self):
        with self.assertRaises(ValueError):
            _check_host_ram_against_loader_settings(_profile(24576, graph_nodes=FULL_NODES))


if __name__ == "__main__":
    unittest.main()
