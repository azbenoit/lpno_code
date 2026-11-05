"""The multibeam path merge must pair beams by position, not by the `index` field.

Measured 2026-08-12: `case_manifest.derived.beams[i].index` is **1-based** while the
prepin extractor recorded `multibeam_paths[i].index` **0-based**. An index-keyed join
therefore gave manifest beam 0 the paths of extras beam 1, and manifest beam 1 nothing
— and the sidecar still looked populated, so the failure surfaced only downstream as
`KeyError: 'scan_speed_m_per_s'` (the datasource fell back to the single-beam path
because `_multibeam_specs` needs paths on *every* beam, and the mbc40 manifests carry
no top-level `input.laser.scan_speed_m_per_s`).

Position is the only convention both sides genuinely share, and a length mismatch is
now a loud error rather than a silent partial merge.
"""

from __future__ import annotations

import pytest

from aiida_pino.data.pino_simulation import merge_multibeam_paths


def _manifest(n=2, base_index=1):
    return {
        "derived": {
            "beams": [
                {"index": base_index + i, "power_w": 10.0 * (i + 1), "path_length_m": 1e-3}
                for i in range(n)
            ]
        }
    }


def _paths(n=2):
    return [
        {"index": i, "path_xy_m": [[float(i), 0.0], [float(i), 1.0]],
         "path_times_s": [0.0, 1e-3], "turn_off_time_s": 1e-3 * (i + 1)}
        for i in range(n)
    ]


def test_every_beam_receives_its_own_path():
    out = merge_multibeam_paths(_manifest(), _paths(), "mbc40_000")
    beams = out["derived"]["beams"]
    assert all("path_xy_m" in b for b in beams), "a beam was left without a path"
    # positional pairing: manifest beam i gets extras beam i
    assert beams[0]["path_xy_m"][0][0] == pytest.approx(0.0)
    assert beams[1]["path_xy_m"][0][0] == pytest.approx(1.0)
    assert beams[1]["turn_off_time_s"] == pytest.approx(2e-3)


def test_one_based_manifest_index_does_not_shift_the_pairing():
    """The exact bug: manifest 1-based, extras 0-based."""
    out = merge_multibeam_paths(_manifest(base_index=1), _paths(), "c")
    assert out["derived"]["beams"][0]["path_xy_m"][0][0] == pytest.approx(0.0)


def test_length_mismatch_is_loud():
    with pytest.raises(ValueError, match="refusing to guess"):
        merge_multibeam_paths(_manifest(n=3), _paths(n=2), "c")


def test_the_original_manifest_is_not_mutated():
    m = _manifest()
    merge_multibeam_paths(m, _paths(), "c")
    assert "path_xy_m" not in m["derived"]["beams"][0]


def test_no_paths_is_a_passthrough():
    m = _manifest()
    assert merge_multibeam_paths(m, None, "c") == m
    assert merge_multibeam_paths(m, [], "c") == m
