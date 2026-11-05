"""The full-resolution host-RAM floor must scale with the GRID, not be a flat number.

The 40960 floor was derived from ONE measurement: a 600k-node GAOT graph build peaking
at ~57 GB, against gpu_spool's 1.5x cgroup enforcement. Applied flat, it forces every
full-resolution run to book 40 GB regardless of size -- so two arms on a 180k-node grid
cannot share a 125 GB box even though their real peak is ~10 GB.

Two measured points anchor the scaling (peak host RAM of two training runs):
    97.6k nodes (straight-path moving frame)  ->  2.4 GB peak
    600k nodes (150x100x40 full)              -> 57.2 GB peak
i.e. peak grows ~n^2.4, steeper than linear because it tracks EDGES.

`graph_nodes` is optional: absent, the flat 40960 floor applies exactly as before, so
every existing profile keeps its current behaviour.
"""
import pytest

from aiida_pino.utils.profiles import full_res_host_ram_floor_mb


def test_absent_node_count_keeps_the_historical_floor():
    assert full_res_host_ram_floor_mb(None) == 40960
    assert full_res_host_ram_floor_mb(0) == 40960


def test_the_600k_anchor_reproduces_the_original_floor():
    """The number the floor was derived from must map back to the floor."""
    assert full_res_host_ram_floor_mb(600_000) == 40960


def test_a_larger_grid_demands_more_not_less():
    assert full_res_host_ram_floor_mb(1_200_000) > 40960


def test_the_pvlhs_grid_lands_near_its_measured_peak():
    """90x40x50 = 180k nodes: interpolating the two anchors gives ~10 GB peak, so the
    floor should be in the mid-teens GB -- enough that the 1.5x cgroup ceiling clears
    the peak with margin, and low enough that two arms fit on one box."""
    floor = full_res_host_ram_floor_mb(180_000)
    assert 8192 <= floor <= 24576, floor
    assert 2 * floor < 50_000, 'two arms must fit under a realistic ceiling'


def test_the_straight_path_anchor_is_not_absurdly_over_booked():
    """97.6k nodes measured a FLAT 2.4 GB peak; the floor must not demand 40 GB for it."""
    assert full_res_host_ram_floor_mb(97_600) < 16384


def test_monotone_in_grid_size():
    sizes = [50_000, 100_000, 180_000, 300_000, 600_000, 1_000_000]
    floors = [full_res_host_ram_floor_mb(n) for n in sizes]
    assert floors == sorted(floors)
