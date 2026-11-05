#!/usr/bin/env python3
"""Joining runs to published anchors, reproducibly.

`melt_metrics.py` holds the metric definitions and `validation/anchors/` holds the
measured targets, but the join between them lived in throwaway scripts, so the
numbers that would go in a manuscript were not regenerable by a reader. These
tests pin the join: schema normalisation across anchor files, condition matching,
the depth-kind refusal, and a summary that reports mean signed error rather than a
bare pass count.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import melt_metrics as mm  # noqa: E402
import score_anchors as sa  # noqa: E402


GALIEGUE_LIKE = {
    "source": {"citation": "X", "provenance_verified": True},
    "depth_kind": "fusion",
    "points": [
        {"power_w": 90.0, "speed_m_s": 0.20, "r0_m": 6.0e-5, "depth_um": 128.25, "width_um": 203.0},
        {"power_w": 100.0, "speed_m_s": 0.20, "r0_m": 6.0e-5, "depth_um": 148.0, "width_um": 203.25},
    ],
}

COCR_LIKE = {   # different key spellings, and depth_kind only at top level
    "source": {"citation": "Y"},
    "depth_kind": "fusion",
    "beam": {"r0_um": 50.0},
    "conditions": [
        {"power_W": 121, "speed_m_per_s": 0.4, "depth_um": 64, "width_um": 156},
    ],
}


class NormalisationTests(unittest.TestCase):
    def test_reads_the_galiegue_schema(self) -> None:
        pts = sa.normalize_points(GALIEGUE_LIKE)
        self.assertEqual(len(pts), 2)
        self.assertAlmostEqual(pts[0]["power_w"], 90.0)
        self.assertAlmostEqual(pts[0]["speed_m_s"], 0.20)
        self.assertAlmostEqual(pts[0]["r0_m"], 6.0e-5)
        self.assertEqual(pts[0]["depth_kind"], "fusion")

    def test_reads_the_cocr_schema_including_beam_level_r0(self) -> None:
        """Anchor files were written by different hands and disagree on key names."""
        pts = sa.normalize_points(COCR_LIKE)
        self.assertEqual(len(pts), 1)
        self.assertAlmostEqual(pts[0]["power_w"], 121.0)
        self.assertAlmostEqual(pts[0]["speed_m_s"], 0.4)
        self.assertAlmostEqual(pts[0]["r0_m"], 5.0e-5)

    def test_points_with_no_measured_depth_are_dropped(self) -> None:
        d = {"depth_kind": "fusion", "points": [
            {"power_w": 70.0, "speed_m_s": 0.2, "r0_m": 6e-5, "depth_um": 0.0},
            {"power_w": 80.0, "speed_m_s": 0.2, "r0_m": 6e-5, "depth_um": 97.2},
        ]}
        self.assertEqual([p["power_w"] for p in sa.normalize_points(d)], [80.0])


class MatchingTests(unittest.TestCase):
    def test_matches_on_power_speed_and_spot(self) -> None:
        pts = sa.normalize_points(GALIEGUE_LIKE)
        m = sa.match_point({"power_w": 90.0, "speed_m_s": 0.20, "r0_m": 6.0e-5}, pts)
        self.assertAlmostEqual(m["depth_um"], 128.25)

    def test_a_different_spot_at_the_same_power_does_not_match(self) -> None:
        """Spot size is the dominant variable; matching on P and V alone is a bug."""
        pts = sa.normalize_points(GALIEGUE_LIKE)
        self.assertIsNone(sa.match_point(
            {"power_w": 90.0, "speed_m_s": 0.20, "r0_m": 1.3e-4}, pts))

    def test_no_match_returns_none_rather_than_the_nearest(self) -> None:
        pts = sa.normalize_points(GALIEGUE_LIKE)
        self.assertIsNone(sa.match_point(
            {"power_w": 300.0, "speed_m_s": 0.20, "r0_m": 6.0e-5}, pts))


class ScoringTests(unittest.TestCase):
    def test_refuses_to_score_across_depth_kinds(self) -> None:
        with self.assertRaises(mm.DepthKindMismatch):
            sa.score_row(sim_depth_um=29.0, sim_width_um=100.0,
                         point={"depth_um": 128.25, "width_um": 203.0, "depth_kind": "fusion"},
                         sim_kind="depression")

    def test_reports_signed_errors_for_both_observables(self) -> None:
        row = sa.score_row(sim_depth_um=60.0, sim_width_um=170.0,
                           point={"depth_um": 128.25, "width_um": 203.0, "depth_kind": "fusion"},
                           sim_kind="fusion")
        self.assertAlmostEqual(row["depth_err_pct"], 100 * (60 - 128.25) / 128.25, places=6)
        self.assertAlmostEqual(row["width_err_pct"], 100 * (170 - 203.0) / 203.0, places=6)
        self.assertFalse(row["depth_in_band"])
        self.assertTrue(row["width_in_band"])

    def test_a_missing_measured_width_scores_depth_only(self) -> None:
        row = sa.score_row(sim_depth_um=60.0, sim_width_um=170.0,
                           point={"depth_um": 128.25, "width_um": None, "depth_kind": "fusion"},
                           sim_kind="fusion")
        self.assertIsNone(row["width_err_pct"])


class SummaryTests(unittest.TestCase):
    def test_summary_reports_mean_signed_error_and_flags_systematic(self) -> None:
        rows = [{"depth_err_pct": -49.0, "depth_in_band": False},
                {"depth_err_pct": -53.0, "depth_in_band": False},
                {"depth_err_pct": -63.0, "depth_in_band": False}]
        s = sa.summarize(rows, "depth")
        self.assertEqual(s.n, 3)
        self.assertEqual(s.n_within_band, 0)
        self.assertAlmostEqual(s.mean_signed_error_pct, -55.0)
        self.assertTrue(s.systematic)

    def test_all_passing_but_one_sided_is_still_flagged(self) -> None:
        rows = [{"depth_err_pct": -13.0, "depth_in_band": True},
                {"depth_err_pct": -11.0, "depth_in_band": True}]
        s = sa.summarize(rows, "depth")
        self.assertEqual(s.n_within_band, 2)
        self.assertTrue(s.systematic, "a one-sign offset is a finding even at 2/2")


class AnchorFileTests(unittest.TestCase):
    def test_every_tracked_anchor_either_normalises_or_is_declared_unsupported(self) -> None:
        """A scorer that silently skips an anchor looks like an anchor with no runs.

        So every tracked anchor file must either normalise to usable points, or
        appear in UNSUPPORTED_SCHEMAS with a reason.
        """
        anchors = sorted((ROOT / "validation" / "anchors").glob("*_targets.json"))
        self.assertTrue(anchors, "no anchor files found")
        for f in anchors:
            with self.subTest(anchor=f.name):
                pts = sa.normalize_points(json.loads(f.read_text()))
                if f.name in sa.UNSUPPORTED_SCHEMAS:
                    self.assertTrue(sa.UNSUPPORTED_SCHEMAS[f.name].strip(),
                                    "an unsupported anchor needs a stated reason")
                    continue
                self.assertTrue(pts, f"{f.name} normalised to zero usable points")
                for p in pts:
                    self.assertIn(p["depth_kind"], mm.VALID_DEPTH_KINDS,
                                  f"{f.name} must declare depth_kind (ACCEPTANCE.md)")
                    self.assertGreater(p["depth_um"], 0.0)

    def test_unsupported_list_has_no_stale_entries(self) -> None:
        """If a schema starts normalising, its excuse must be deleted."""
        for name in sa.UNSUPPORTED_SCHEMAS:
            f = ROOT / "validation" / "anchors" / name
            if not f.is_file():
                continue
            with self.subTest(anchor=name):
                self.assertFalse(sa.normalize_points(json.loads(f.read_text())),
                                 f"{name} now normalises -- remove it from UNSUPPORTED_SCHEMAS")


class RecipeFingerprintTests(unittest.TestCase):
    """Pooling runs from different physics recipes is the error that caused the
    2026-07-30 retractions: a summary over mixed recipes describes nothing.
    """

    def test_fingerprint_is_read_from_the_prepin(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            case = Path(d) / "c"
            case.mkdir()
            (case / "prepin.c").write_text(
                " aplume=0.2, tv1=3123.15, ereflct=0.35, foo=1\n")
            fp = sa.recipe_fingerprint(case)
            self.assertEqual(fp, "aplume=0.2,tv1=3123.15,ereflct=0.35")

    def test_fingerprint_distinguishes_the_absorptivity_route(self) -> None:
        """aplume+tv1 alone cannot separate the standing recipe from the REJECTED
        measured-absorptance arm: t2_206w and t5_206w share aplume=0, tv1=3520 and
        differ only in ereflct (0.2169 tabular vs 0.2017 scalar) -- so a summary
        pooled them, mixing a rejected arm into the standing-recipe ladder.
        """
        with tempfile.TemporaryDirectory() as d:
            std = Path(d) / "std"; std.mkdir()
            (std / "prepin.std").write_text(" aplume=0.0, tv1=3520, ereflct=0.216852515308, iereflct=1\n")
            mea = Path(d) / "mea"; mea.mkdir()
            (mea / "prepin.mea").write_text(" aplume=0.0, tv1=3520, ereflct=0.201621218003\n")
            self.assertNotEqual(sa.recipe_fingerprint(std), sa.recipe_fingerprint(mea))
            self.assertIn("iereflct=1", sa.recipe_fingerprint(std))
            self.assertNotIn("iereflct", sa.recipe_fingerprint(mea))

    def test_missing_prepin_is_unknown_not_a_crash(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(sa.recipe_fingerprint(Path(d)), "unknown")

    def test_mixed_fingerprints_are_detected(self) -> None:
        rows = [{"recipe": "aplume=0.0,tv1=3520"}, {"recipe": "aplume=0.2,tv1=3123.15"}]
        self.assertEqual(len(sa.recipes_present(rows)), 2)
        self.assertEqual(len(sa.recipes_present([rows[0], dict(rows[0])])), 1)


class StationarySpotWeldTests(unittest.TestCase):
    """Spot welds have no scan speed, which is representable as speed = 0.

    Our dwell cases already report speed 0 (a stationary spot sets
    `laser.dwell_time_s`, not `scan_speed_m_per_s`), so the anchor only needs to
    say 0 for the match to work. Dwell is constant at 10 ms across every Hawk
    condition, so (power, 0, spot) is unambiguous.
    """

    HAWK_LIKE = {
        "depth_kind": "fusion",
        "beam": {"spot_diameter_um": 303, "dwell_time_ms": 10},
        "conditions": {
            "_source": "T",                      # non-list keys must be ignored
            "wrought_304L_AR_GR": [
                {"power_W": 118, "energy_J": 1.18, "depth_um": 68.2, "width_um": 310.0},
                {"power_W": 456, "energy_J": 4.56, "depth_um": 491.0, "width_um": 868.0},
            ],
            "am_304L_AB_UT": [
                {"power_W": 118, "energy_J": 1.18, "depth_um": 141.0, "width_um": 283.0},
            ],
        },
    }

    def test_series_of_rows_are_flattened_with_zero_speed(self) -> None:
        pts = sa.normalize_points(self.HAWK_LIKE)
        self.assertEqual(len(pts), 3)
        self.assertTrue(all(p["speed_m_s"] == 0.0 for p in pts))

    def test_spot_diameter_becomes_a_radius(self) -> None:
        pts = sa.normalize_points(self.HAWK_LIKE)
        self.assertAlmostEqual(pts[0]["r0_m"], 303e-6 / 2)

    def test_each_point_carries_its_series_so_variants_are_distinguishable(self) -> None:
        pts = sa.normalize_points(self.HAWK_LIKE)
        got = {(p["label"], p["power_w"]) for p in pts}
        self.assertIn(("wrought_304L_AR_GR", 118.0), got)
        self.assertIn(("am_304L_AB_UT", 118.0), got)

    def test_a_dwell_run_matches_a_zero_speed_point(self) -> None:
        # series= keeps the set unambiguous: this fixture has two variants at 118 W,
        # and match_point deliberately refuses to choose between them (see
        # AmbiguousMatchTests). The point here is only that speed 0 matches at all.
        pts = sa.normalize_points(self.HAWK_LIKE, series="wrought_304L_AR_GR")
        m = sa.match_point({"power_w": 118.0, "speed_m_s": 0.0, "r0_m": 151.5e-6}, pts)
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m["depth_um"], 68.2)

    def test_series_filter_selects_one_variant(self) -> None:
        pts = sa.normalize_points(self.HAWK_LIKE, series="am_304L_AB_UT")
        self.assertEqual({p["label"] for p in pts}, {"am_304L_AB_UT"})


class DuplicateConditionTests(unittest.TestCase):
    """Several runs can match one measured condition (resolution variants, reruns).

    Scoring each is fine; pooling them into a summary is not -- it treats N runs at
    one condition as N independent points, inflating n and skewing the mean. Seen
    2026-07-30: 8 SS304L rows covered only 3 distinct powers, and the pooled
    "6/8 in band" flattered a per-resolution 2/3.
    """

    def test_duplicate_conditions_are_detected(self) -> None:
        rows = [{"power_w": 118.0, "speed_m_s": 0.0, "r0_m": 1.5e-4, "label": "w", "case": "a"},
                {"power_w": 118.0, "speed_m_s": 0.0, "r0_m": 1.5e-4, "label": "w", "case": "b"},
                {"power_w": 355.0, "speed_m_s": 0.0, "r0_m": 1.5e-4, "label": "w", "case": "c"}]
        dups = sa.duplicate_conditions(rows)
        self.assertEqual(len(dups), 1)
        self.assertEqual(sorted(next(iter(dups.values()))), ["a", "b"])

    def test_distinct_conditions_report_no_duplicates(self) -> None:
        rows = [{"power_w": 118.0, "speed_m_s": 0.0, "r0_m": 1.5e-4, "label": "w", "case": "a"},
                {"power_w": 355.0, "speed_m_s": 0.0, "r0_m": 1.5e-4, "label": "w", "case": "c"}]
        self.assertEqual(sa.duplicate_conditions(rows), {})

    def test_same_power_but_different_variant_is_not_a_duplicate(self) -> None:
        rows = [{"power_w": 118.0, "speed_m_s": 0.0, "r0_m": 1.5e-4, "label": "wrought", "case": "a"},
                {"power_w": 118.0, "speed_m_s": 0.0, "r0_m": 1.5e-4, "label": "am_UT", "case": "b"}]
        self.assertEqual(sa.duplicate_conditions(rows), {})


class AmbiguousMatchTests(unittest.TestCase):
    """Several measured variants can share one process condition.

    Hawk reports 4 material variants at identical (power, dwell, spot) whose depths
    differ by up to 2x (68.2 vs 141.0 um at 118 W). Returning "the first match"
    silently picks one and changes the verdict, so ambiguity must be surfaced.
    """

    POINTS = [
        {"power_w": 118.0, "speed_m_s": 0.0, "r0_m": 1.5e-4, "depth_um": 68.2,
         "width_um": 310.0, "depth_kind": "fusion", "label": "wrought"},
        {"power_w": 118.0, "speed_m_s": 0.0, "r0_m": 1.5e-4, "depth_um": 141.0,
         "width_um": 283.0, "depth_kind": "fusion", "label": "am_UT"},
    ]

    def test_all_matches_are_returned(self) -> None:
        got = sa.match_points({"power_w": 118.0, "speed_m_s": 0.0, "r0_m": 1.5e-4}, self.POINTS)
        self.assertEqual(len(got), 2)
        self.assertEqual({p["label"] for p in got}, {"wrought", "am_UT"})

    def test_no_match_returns_empty(self) -> None:
        self.assertEqual(sa.match_points(
            {"power_w": 999.0, "speed_m_s": 0.0, "r0_m": 1.5e-4}, self.POINTS), [])

    def test_single_match_helper_still_works_when_unambiguous(self) -> None:
        one = [self.POINTS[0]]
        m = sa.match_point({"power_w": 118.0, "speed_m_s": 0.0, "r0_m": 1.5e-4}, one)
        self.assertAlmostEqual(m["depth_um"], 68.2)

    def test_single_match_helper_refuses_an_ambiguous_set(self) -> None:
        with self.assertRaises(sa.AmbiguousAnchor):
            sa.match_point({"power_w": 118.0, "speed_m_s": 0.0, "r0_m": 1.5e-4}, self.POINTS)


class BareListAnchorTests(unittest.TestCase):
    """gan_points.json is a bare JSON list, so it cannot declare depth_kind itself.

    The notebook reads it as a list (`json.loads(...) if exists else []`), so wrapping
    it in an object would break that consumer. A tracked `<stem>.meta.json` sidecar
    supplies the declaration instead -- in data, not on a command line, so it cannot
    be asserted differently by different callers.
    """

    BARE = [
        {"group": "Ti64, spot size 95 um", "P": 155.523, "V": 0.6, "r0": 4.75e-05,
         "depth_um": 76.6388, "eta": 0.443},
        {"group": "Ti64, spot size 95 um", "P": 206.395, "V": 0.4, "r0": 4.75e-05,
         "depth_um": 226.766, "eta": 0.605},
    ]

    def test_a_bare_list_normalises(self) -> None:
        pts = sa.normalize_points({"points": self.BARE, "depth_kind": "depression"})
        self.assertEqual(len(pts), 2)
        self.assertAlmostEqual(pts[0]["power_w"], 155.523)
        self.assertAlmostEqual(pts[0]["speed_m_s"], 0.6)
        self.assertAlmostEqual(pts[0]["r0_m"], 4.75e-05)

    def test_load_anchor_merges_a_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "gan_points.json"
            f.write_text(json.dumps(self.BARE))
            (Path(d) / "gan_points.meta.json").write_text(
                json.dumps({"depth_kind": "depression"}))
            anchor = sa.load_anchor(f)
            pts = sa.normalize_points(anchor)
            self.assertTrue(pts)
            self.assertEqual({p["depth_kind"] for p in pts}, {"depression"})

    def test_bare_list_without_a_sidecar_has_no_depth_kind(self) -> None:
        """Undeclared must stay undeclared -- never defaulted to a guess."""
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "x.json"
            f.write_text(json.dumps(self.BARE))
            pts = sa.normalize_points(sa.load_anchor(f))
            self.assertIsNone(pts[0]["depth_kind"])


class PowerToleranceTests(unittest.TestCase):
    """Power tolerance must be relative, not a fixed number of watts.

    A CoCr case built at 120.0 W failed to match the anchor's 121 W point -- a 0.8%
    gap, far inside the measurement's own scatter (+-20 um on a 64 um depth) -- because
    the guard was an absolute 0.5 W. At 456 W the same 0.5 W is 0.1%, needlessly
    strict; at 10 W it would be 5%, too loose.
    """

    PTS = [{"power_w": 121.0, "speed_m_s": 0.4, "r0_m": 5e-5, "depth_um": 64.0,
            "width_um": 156.0, "depth_kind": "fusion", "label": "P3"}]

    def test_a_one_percent_power_gap_matches(self) -> None:
        m = sa.match_point({"power_w": 120.0, "speed_m_s": 0.4, "r0_m": 5e-5}, self.PTS)
        self.assertIsNotNone(m, "120 vs 121 W is 0.8% -- inside measurement scatter")

    def test_a_ten_percent_power_gap_does_not_match(self) -> None:
        self.assertIsNone(sa.match_point(
            {"power_w": 133.0, "speed_m_s": 0.4, "r0_m": 5e-5}, self.PTS))

    def test_tolerance_scales_with_power(self) -> None:
        pts = [dict(self.PTS[0], power_w=456.0)]
        self.assertIsNotNone(sa.match_point(
            {"power_w": 453.0, "speed_m_s": 0.4, "r0_m": 5e-5}, pts),
            "3 W at 456 W is 0.7%, inside tolerance")
        self.assertIsNone(sa.match_point(
            {"power_w": 400.0, "speed_m_s": 0.4, "r0_m": 5e-5}, pts))


class AlloyMatchingTests(unittest.TestCase):
    """A run must not be scored against a different alloy's measurement.

    gan_points.json holds Ti64, SS316, Al6061 and a second Ti64 spot size in one
    file. Matching on (power, speed, spot) alone let an SS316 run match a Gan point
    on 2026-07-30; only a coincidental r0 difference kept it from being a Ti64 row.
    """

    def test_alloy_key_canonicalises_both_sides(self) -> None:
        for text, want in (("Ti64, spot size 95 um", "ti64"),
                           ("Ti6, spot size 140 um", "ti64"),
                           ("SS316, spot size 110 um", "ss316"),
                           ("Al6061, spot size 88 um", "al6061"),
                           ("/path/materials/ti64_tv3520/ti64_tv3520.json", "ti64"),
                           ("/path/materials/ss316l_ext/ss316l_ext.json", "ss316"),
                           ("/path/materials/ss304_raoult/x.json", "ss304"),
                           ("Titanium Alloy Ti Al V IMI 318 Mills data", "ti64"),
                           ("Stainless steel 316L", "ss316"),
                           ("/path/materials/in718_ext/x.json", "in718")):
            with self.subTest(text=text):
                self.assertEqual(sa.alloy_key(text), want)

    def test_unknown_text_yields_none_rather_than_a_guess(self) -> None:
        self.assertIsNone(sa.alloy_key("some unlabelled alloy"))

    def test_a_different_alloy_does_not_match(self) -> None:
        pts = [{"power_w": 208.0, "speed_m_s": 0.6, "r0_m": 5.5e-5, "depth_um": 100.0,
                "width_um": None, "depth_kind": "depression", "label": None,
                "alloy": "ss316"}]
        self.assertEqual(sa.match_points(
            {"power_w": 208.0, "speed_m_s": 0.6, "r0_m": 5.5e-5, "alloy": "ti64"}, pts), [])

    def test_the_same_alloy_still_matches(self) -> None:
        pts = [{"power_w": 208.0, "speed_m_s": 0.6, "r0_m": 5.5e-5, "depth_um": 100.0,
                "width_um": None, "depth_kind": "depression", "label": None,
                "alloy": "ss316"}]
        self.assertEqual(len(sa.match_points(
            {"power_w": 208.0, "speed_m_s": 0.6, "r0_m": 5.5e-5, "alloy": "ss316"}, pts)), 1)

    def test_an_unknown_alloy_on_either_side_does_not_block(self) -> None:
        pts = [{"power_w": 208.0, "speed_m_s": 0.6, "r0_m": 5.5e-5, "depth_um": 100.0,
                "width_um": None, "depth_kind": "depression", "label": None, "alloy": None}]
        self.assertEqual(len(sa.match_points(
            {"power_w": 208.0, "speed_m_s": 0.6, "r0_m": 5.5e-5, "alloy": "ti64"}, pts)), 1)


class ScoringPlanTests(unittest.TestCase):
    """The tracked plan is what makes the campaign's table one command.

    Each anchor needs different flags (which surface to read, which recipe to
    isolate, which material variant); keeping them in a shell history is how the
    wrong ones get used.
    """

    PLAN = ROOT / "validation" / "scoring_plan.json"

    def _skip_if_not_distributed(self, anchor: str) -> None:
        """Skip a subtest whose anchor file the repository may not redistribute.

        `score_anchors.NOT_DISTRIBUTED` lists them with the licence reason; the
        user builds each from the publication (validation/anchors/README.md).
        """
        f = self.PLAN.parent / anchor
        name = Path(anchor).name
        if not f.is_file() and name in sa.NOT_DISTRIBUTED:
            self.skipTest(f"{name} is not distributed with this repository: "
                          f"{sa.NOT_DISTRIBUTED[name]}")

    def test_plan_exists_and_parses(self) -> None:
        self.assertTrue(self.PLAN.is_file())
        plan = json.loads(self.PLAN.read_text())
        self.assertTrue(plan["anchors"])

    def test_every_planned_anchor_file_exists_and_normalises(self) -> None:
        plan = json.loads(self.PLAN.read_text())
        for e in plan["anchors"]:
            with self.subTest(anchor=e["anchor"]):
                self._skip_if_not_distributed(e["anchor"])
                f = self.PLAN.parent / e["anchor"]
                self.assertTrue(f.is_file(), f"{e['anchor']} missing")
                pts = sa.normalize_points(sa.load_anchor(f), series=e.get("series"))
                self.assertTrue(pts, f"{e['anchor']} normalised to nothing")

    def test_every_entry_declares_sim_kind_and_a_reason(self) -> None:
        plan = json.loads(self.PLAN.read_text())
        for e in plan["anchors"]:
            with self.subTest(anchor=e["anchor"]):
                self.assertIn(e.get("sim_kind"), mm.VALID_DEPTH_KINDS)
                self.assertTrue(e.get("note", "").strip(),
                                "an entry must say why it needs its flags")

    def test_planned_sim_kind_matches_the_anchor_depth_kind(self) -> None:
        """Guards the mistake this campaign made four times."""
        plan = json.loads(self.PLAN.read_text())
        for e in plan["anchors"]:
            with self.subTest(anchor=e["anchor"]):
                self._skip_if_not_distributed(e["anchor"])
                pts = sa.normalize_points(
                    sa.load_anchor(self.PLAN.parent / e["anchor"]), series=e.get("series"))
                kinds = {p["depth_kind"] for p in pts}
                self.assertIn(e["sim_kind"], kinds,
                              f"plan reads {e['anchor']} as {e['sim_kind']} but it reports {kinds}")

    def test_unscoreable_anchors_are_declared_with_reasons(self) -> None:
        plan = json.loads(self.PLAN.read_text())
        for name, why in plan.get("_unscoreable", {}).items():
            with self.subTest(anchor=name):
                self.assertTrue(why.strip(), "an unscoreable anchor needs a reason")


class PlanModeKeepsTheGuardsTests(unittest.TestCase):
    """Plan mode must not be a weaker path than scoring one anchor by hand.

    When --plan was added it printed tables without the duplicate-condition and
    mixed-recipe warnings, so SS304L silently pooled its 50um and 25um resolution
    variants -- exactly what the single-anchor path refuses to do quietly.
    """

    def test_run_plan_reports_duplicates_and_mixed_recipes(self) -> None:
        import inspect
        src = inspect.getsource(sa.run_plan)
        self.assertIn("duplicate_conditions", src,
                      "plan mode must warn about over-matched conditions")
        self.assertIn("recipes_present", src,
                      "plan mode must warn when recipes are pooled")


class SpotSizeFilterTests(unittest.TestCase):
    """A plan entry must be able to pin one beam radius.

    The Galiegue anchor holds a fixed-spot power ladder AND a defocus spot-size
    series. Pooling them gave a mean of -48% that mixes r0 60 um rungs (-49..-63%)
    with an r0 130 um point (-23%) -- an anchor-level view, not the manuscript
    figure, which is the ladder at one spot.
    """

    PTS = [{"power_w": 100.0, "speed_m_s": 0.2, "r0_m": 6.0e-5, "depth_um": 148.0,
            "width_um": 203.25, "depth_kind": "fusion", "label": None, "alloy": "ti64"},
           {"power_w": 100.0, "speed_m_s": 0.2, "r0_m": 1.3e-4, "depth_um": 52.0,
            "width_um": 307.9, "depth_kind": "fusion", "label": None, "alloy": "ti64"}]

    def test_filter_keeps_only_the_named_radius(self) -> None:
        got = sa.filter_by_r0(self.PTS, 60.0)
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(got[0]["r0_m"], 6.0e-5)

    def test_none_keeps_everything(self) -> None:
        self.assertEqual(len(sa.filter_by_r0(self.PTS, None)), 2)

    def test_a_radius_with_no_points_yields_empty(self) -> None:
        self.assertEqual(sa.filter_by_r0(self.PTS, 83.0), [])


class ConditionProvenanceTests(unittest.TestCase):
    """Selection-vs-blind must be recorded, not remembered.

    "Which conditions did you fit on?" is the first question a reviewer asks, and
    the honest answer here is strong -- one fitted scalar, everything else blind --
    but only if it is written down where the numbers are.
    """

    PLAN = ROOT / "validation" / "scoring_plan.json"

    def test_every_planned_anchor_has_a_provenance_entry(self) -> None:
        plan = json.loads(self.PLAN.read_text())
        prov = plan["_condition_provenance"]
        for e in plan["anchors"]:
            with self.subTest(anchor=e["anchor"]):
                self.assertIn(e["anchor"], prov)
                self.assertIn(prov[e["anchor"]]["default"], ("blind", "selection"))

    def test_selection_conditions_state_what_they_fitted(self) -> None:
        plan = json.loads(self.PLAN.read_text())
        for name, blk in plan["_condition_provenance"].items():
            for sel in blk.get("selection", []):
                with self.subTest(anchor=name, P=sel.get("power_w")):
                    self.assertTrue(sel.get("why", "").strip())

    def test_the_recipe_has_exactly_one_fitted_condition(self) -> None:
        """One free scalar is the claim; more than one selection condition on the
        Ti64 keyhole anchor would quietly weaken it."""
        plan = json.loads(self.PLAN.read_text())
        gan = plan["_condition_provenance"]["anchors/gan_points.json"]
        self.assertEqual(gan["default"], "blind")
        self.assertEqual(len(gan.get("selection", [])), 1)


class AiidaFallbackScopeTests(unittest.TestCase):
    """The AiiDA fallback must stay inside the runs directory it was handed.

    `iter_cases(runs, aiida=True)` enumerated disk manifests under `runs` and then added
    EVERY case in the database, unscoped. A temp tree holding one run scored four
    unrelated `galiegue_*` cases (self-improve, 2026-09-09), and a recipe comparison
    scoped to one batch would silently average in the rest of the store. The symptom also
    depended on the interpreter -- plain `python3` has no `aiida`, so the fallback yielded
    nothing and the suite passed; under the `pino` env the same test failed and took
    167 s instead of 0.02 s. These tests fake the database so they are hermetic either way.
    """

    @staticmethod
    def _doc():
        """A minimal single-beam manifest: enough for iter_cases, which only needs it
        to parse. Condition matching is covered elsewhere."""
        return {"input": {"laser": {"power_w": 90.0, "scan_speed_m_per_s": 0.2,
                                    "beam_radius_m": 6.0e-5},
                          "material": {"file": "materials/ti64_v3_calibrated.json"}},
                "material": {"name": "ti64_v3_calibrated"}}

    def _fake_manifests(self, entries):
        return lambda: entries

    def _tree(self, tmp, name="in_tree_case"):
        case = Path(tmp) / "runs" / name
        (case / "npz").mkdir(parents=True)
        (case / "case_manifest.json").write_text(json.dumps(self._doc()))
        return Path(tmp) / "runs"

    def test_a_case_outside_the_tree_is_not_yielded(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._tree(tmp)
            outside = str(Path(tmp) / "elsewhere" / "galiegue_p90_v020")
            fake = {"galiegue_p90_v020": ("uuid-1", self._doc(), outside)}
            with mock.patch.object(sa, "_aiida_manifests", self._fake_manifests(fake)):
                names = [c.name for c in sa.iter_cases(runs, aiida=True)]
            self.assertNotIn("galiegue_p90_v020", names)
            self.assertEqual(names, ["in_tree_case"])

    def test_a_case_inside_the_tree_is_still_recovered(self):
        """The fallback's purpose survives: a run whose export was reclaimed still scores."""
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._tree(tmp)
            inside = str(runs / "reclaimed_case")      # never created on disk
            fake = {"reclaimed_case": ("uuid-2", self._doc(), inside)}
            with mock.patch.object(sa, "_aiida_manifests", self._fake_manifests(fake)):
                cases = {c.name: c for c in sa.iter_cases(runs, aiida=True)}
            self.assertIn("reclaimed_case", cases)
            self.assertIsNone(cases["reclaimed_case"].case_dir)
            self.assertEqual(cases["reclaimed_case"].uuid, "uuid-2")

    def test_a_case_with_no_recorded_path_is_excluded_and_announced(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._tree(tmp)
            fake = {"pathless": ("uuid-3", self._doc(), "")}
            buf = io.StringIO()
            with mock.patch.object(sa, "_aiida_manifests", self._fake_manifests(fake)):
                with contextlib.redirect_stdout(buf):
                    names = [c.name for c in sa.iter_cases(runs, aiida=True)]
            self.assertNotIn("pathless", names)
            self.assertIn("record no source path", buf.getvalue())

    def test_aiida_false_still_skips_the_database_entirely(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._tree(tmp)
            inside = str(runs / "reclaimed_case")
            fake = {"reclaimed_case": ("uuid-4", self._doc(), inside)}
            with mock.patch.object(sa, "_aiida_manifests", self._fake_manifests(fake)):
                names = [c.name for c in sa.iter_cases(runs, aiida=False)]
            self.assertEqual(names, ["in_tree_case"])

    def test_plan_mode_forwards_no_aiida_and_band(self):
        """`--plan` accepted `--no-aiida` and `--band` and dropped both: `run_plan`
        called `score_anchor_file` with its defaults, so a reader scoring a solved
        tree with `--no-aiida` (the documented reproducibility command) still
        reached for the database. Found while documenting that command, 2026-09-14."""
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._tree(tmp)
            anchor = Path(tmp) / "anchor.json"
            anchor.write_text(json.dumps(GALIEGUE_LIKE))
            plan = Path(tmp) / "plan.json"
            plan.write_text(json.dumps({"anchors": [{"anchor": "anchor.json", "sim_kind": "fusion"}]}))
            seen = {}

            def fake_score(anchor_path, runs_dir, **kw):
                seen.update(kw)
                return [], 0

            argv = ["score_anchors.py", "--plan", str(plan), "--runs", str(runs),
                    "--no-aiida", "--band", "0.2"]
            with mock.patch.object(sa, "score_anchor_file", fake_score), \
                    mock.patch.object(sys, "argv", argv), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sa.main(), 0)
            self.assertIs(seen.get("aiida"), False)
            self.assertEqual(seen.get("band"), 0.2)

    def test_source_dir_reads_every_recorded_form(self):
        """Both shapes are live in the store: a run directory, and (the
        ti64_random_keyhole150-era imports) the NPZ file itself."""
        self.assertEqual(sa._source_dir("file:///a/b/run_dir", None), "/a/b/run_dir")
        self.assertEqual(sa._source_dir(None, "/a/b/run_dir/npz/c.npz"), "/a/b/run_dir")
        self.assertEqual(sa._source_dir("file:///a/b/flat/c.npz", None), "/a/b/flat")
        self.assertEqual(sa._source_dir(None, None), "")

    def test_cases_outside_the_tree_are_announced_not_dropped_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._tree(tmp)
            outside = str(Path(tmp) / "elsewhere" / "far_case")
            fake = {"far_case": ("uuid-5", self._doc(), outside)}
            buf = io.StringIO()
            with mock.patch.object(sa, "_aiida_manifests", self._fake_manifests(fake)):
                with contextlib.redirect_stdout(buf):
                    list(sa.iter_cases(runs, aiida=True))
            self.assertIn("lie outside", buf.getvalue())
            self.assertIn("1 AiiDA case(s)", buf.getvalue())


class MultiBeamRunConditionsTests(unittest.TestCase):
    """A multi-beam run carries no single (P, V, r0) condition.

    `run_conditions` read `laser["power_w"]` directly. Independent-beam configs
    (`mb40_*`) put power on each beam and leave no top-level scalar, so once those
    runs landed in `runs/` the scorer raised `KeyError: 'power_w'` while scanning
    the tree and **every** anchor file failed to score, not just the multi-beam
    cases.

    Skipping is the correct outcome, not summing. Total power would let a two-beam
    2 x 120 W case match a 240 W single-track anchor, which is a false match: the
    beams sit on separate paths and never deposit that power in one track. The NPZ
    provenance scalars do sum, because there the number labels a stored array
    instead of selecting a comparison.
    """

    def _case(self, tmp: Path, laser: dict) -> Path:
        case = tmp / "case"
        case.mkdir()
        (case / "case_manifest.json").write_text(json.dumps(
            {"input": {"laser": laser, "material": {"file": "materials/ti64_v3_calibrated.json"}},
             "material": {"name": "ti64_v3_calibrated"}}))
        return case

    def test_multi_beam_run_is_skipped_not_summed(self) -> None:
        laser = {"beam_radius_m": 6.0e-5,
                 "beams": [{"power_w": 120.0, "scan_speed_m_per_s": 0.2},
                           {"power_w": 120.0, "scan_speed_m_per_s": 0.4}]}
        with tempfile.TemporaryDirectory() as td:
            case = self._case(Path(td), laser)
            self.assertIsNone(sa.run_conditions(case),
                              "a multi-beam run must not present itself as one condition")

    def test_single_beam_run_still_resolves(self) -> None:
        laser = {"power_w": 90.0, "scan_speed_m_per_s": 0.2, "beam_radius_m": 6.0e-5}
        with tempfile.TemporaryDirectory() as td:
            cond = sa.run_conditions(self._case(Path(td), laser))
        self.assertIsNotNone(cond)
        self.assertAlmostEqual(cond["power_w"], 90.0)
        self.assertAlmostEqual(cond["speed_m_s"], 0.2)

    def test_scoring_a_tree_survives_a_multi_beam_run(self) -> None:
        """The regression proper: a multi-beam run in the tree must not abort the sweep.

        `score_anchor_file` walks runs, so the case needs an `npz/` entry to be
        visited at all. Before the fix this raised `KeyError: 'power_w'` and took
        every anchor down with it.
        """
        import numpy as np

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            case = tmp / "runs" / "mb40_000"
            (case / "npz").mkdir(parents=True)
            np.savez(case / "npz" / "mb40_000.npz", placeholder=np.zeros(1))
            (case / "case_manifest.json").write_text(json.dumps(
                {"input": {"laser": {"beam_radius_m": 6.0e-5,
                                     "beams": [{"power_w": 120.0}, {"power_w": 120.0}]}}}))
            anchor = tmp / "a.json"
            anchor.write_text(json.dumps(GALIEGUE_LIKE))
            rows, ambiguous = sa.score_anchor_file(anchor, tmp / "runs", sim_kind="fusion")
        # The multi-beam run is skipped, so it contributes no row and no ambiguity.
        self.assertEqual(rows, [])
        self.assertEqual(ambiguous, 0)


def _synthetic_case(power_w: float = 150.0, speed: float = 0.4) -> dict:
    """A tiny but structurally real export: 1-D axes, (T, nx, ny, nz) fields.

    Deliberately the same shape a stored `payload.npz` has, because that is the
    point of the AiiDA path -- the bytes are the compact export, so one mapping
    serves both sources and `melt_metrics` needs no conversion.
    """
    nx, ny, nz = 12, 10, 8
    x = np.linspace(-1.0e-4, 1.0e-4, nx).astype(np.float32)
    y = np.linspace(-1.0e-4, 1.0e-4, ny).astype(np.float32)
    z = np.linspace(-1.2e-4, 4.0e-5, nz).astype(np.float32)   # gas above z = 0
    shape = (1, nx, ny, nz)
    temperature = np.full(shape, 300.0, dtype=np.float32)
    fluid = np.zeros(shape, dtype=np.float32)
    fluid[:, :, :, z < 0.0] = 1.0                              # metal below the surface
    # a melt blob straddling the mid-track, wide enough to clear MIN_CELLS
    temperature[:, 4:9, 3:8, -5:-1] = 2600.0
    return {
        "x": x, "y": y, "z": z,
        "time_s": np.asarray([1.0e-4], dtype=np.float32),
        "Temperature": temperature,
        "Fraction Of Fluid": fluid,
        "Melt Region": (temperature > 1898.0).astype(np.float32),
        "power_w": np.asarray(power_w, dtype=np.float32),
        "scan_speed_m_per_s": np.asarray(speed, dtype=np.float32),
    }


class AiidaSourceTests(unittest.TestCase):
    """A solved run whose local NPZ was reclaimed must still be scoreable.

    The scorer enumerated cases by globbing `runs/**/npz/*.npz`, so a case that was
    solved and then had its export reclaimed was indistinguishable from one that
    never ran. Measured 2026-09-04 against the tracked anchors: 54 anchor points
    match at least one run, 15 are scoreable from disk, and 35 more have a genuinely
    solved run whose arrays live only in AiiDA (218 of 220 such names resolve to a
    `PinoSimulationData` node, all `npz_file`, none evicted). Conditions are NOT the
    problem -- every reclaimed directory still carries its `case_manifest.json` and
    its `prepin.*`, so `run_conditions` and `recipe_fingerprint` keep working.

    The AiiDA import must stay lazy. This module imports `score_anchors` at module
    scope and the documented suite runs under the default `python3`, which has no
    `aiida`, so a top-level import would fail all of these tests at collection
    rather than one named test.
    """

    def _case(self, tmp: Path, *, with_npz: bool, name: str = "c0") -> Path:
        case = tmp / "runs" / name
        case.mkdir(parents=True)
        (case / "case_manifest.json").write_text(json.dumps(
            {"input": {"laser": {"power_w": 90.0, "scan_speed_m_per_s": 0.20,
                                 "beam_radius_m": 6.0e-5},
                       "material": {"file": "materials/ti64_v3_calibrated.json"}},
             "material": {"name": "ti64_v3_calibrated"}}))
        if with_npz:
            (case / "npz").mkdir()
            np.savez(case / "npz" / f"{name}.npz", **_synthetic_case(power_w=90.0, speed=0.20))
        return case

    def test_module_works_with_aiida_absent(self) -> None:
        """Offline is a supported mode: this is a validation tool a reader must be
        able to run without a database, and the suite's own interpreter has no aiida."""
        with mock.patch.dict(sys.modules, {"aiida": None, "aiida.orm": None}):
            importlib.reload(sa)
            self.assertTrue(hasattr(sa, "open_case_arrays"))
            sa._aiida_index.cache_clear()
            self.assertEqual(sa._aiida_index(), {},
                             "an unreachable AiiDA must degrade to empty, not raise")
        importlib.reload(sa)

    def test_local_npz_still_scores(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._case(tmp, with_npz=True)
            anchor = tmp / "a.json"
            anchor.write_text(json.dumps(GALIEGUE_LIKE))
            rows, _ = sa.score_anchor_file(anchor, tmp / "runs", sim_kind="fusion")
        self.assertTrue(rows, "a case with a local NPZ must score as it always did")

    def test_reclaimed_case_is_discovered_and_read_from_aiida(self) -> None:
        """The behaviour the feature exists for: a manifest with no `npz/` is still
        enumerated, and its arrays come from the stored payload."""
        payload = _synthetic_case(power_w=90.0, speed=0.20)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._case(tmp, with_npz=False)
            anchor = tmp / "a.json"
            anchor.write_text(json.dumps(GALIEGUE_LIKE))
            @contextlib.contextmanager
            def _fake_payload(uuid):
                yield payload

            with mock.patch.object(sa, "_aiida_index", lambda: {"c0": ["fake-uuid"]}), \
                 mock.patch.object(sa, "_load_aiida_payload", _fake_payload):
                rows, _ = sa.score_anchor_file(anchor, tmp / "runs", sim_kind="fusion")
        self.assertTrue(rows, "a reclaimed but solved case must be scoreable from AiiDA")
        self.assertEqual(rows[0]["case"], "c0")

    def test_local_npz_wins_over_aiida(self) -> None:
        """Disk first: the same bytes, without a database round trip."""
        with tempfile.TemporaryDirectory() as td:
            case = self._case(Path(td), with_npz=True)
            called: list[int] = []
            with mock.patch.object(sa, "_aiida_index", lambda: called.append(1) or {}):
                with sa.open_case_arrays(case) as data:
                    self.assertIsNotNone(data)
            self.assertEqual(called, [],
                             "the AiiDA index must not be built when disk already has it")

    def test_ambiguous_case_name_is_refused_not_guessed(self) -> None:
        """67 reclaimed case names resolve to more than one node, every one with a
        different content hash -- they are re-solves across physics eras. Taking
        whichever the query returned last would silently score the wrong era."""
        with tempfile.TemporaryDirectory() as td:
            case = self._case(Path(td), with_npz=False)
            with mock.patch.object(sa, "_aiida_index", lambda: {"c0": ["uuid-a", "uuid-b"]}):
                with sa.open_case_arrays(case) as data:
                    self.assertIsNone(data,
                                      "two candidate nodes must yield nothing, not a guess")

    def test_case_with_neither_source_yields_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            case = self._case(Path(td), with_npz=False)
            with mock.patch.object(sa, "_aiida_index", lambda: {}):
                with sa.open_case_arrays(case) as data:
                    self.assertIsNone(data)


if __name__ == "__main__":
    unittest.main()
