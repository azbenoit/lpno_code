"""`materialize: true` must survive presubmit and take the branch it claims.

Found 2026-09-07 after a submission with the flag exactly as
specified. Two defects in the same change:

- **Exit 303 before anything staged.** `layouts_compatible` compares the requested layout
  against the derivation dict key-for-key. `materialize` exists only in the derivation -- a
  training profile has no field expressing it, exactly like `derived_fields` and `crop_z_top`
  before it -- so the comparison could never be equal and every submission carrying the flag
  was refused as an incompatible variant. It is a STAGING directive, not a layout property.

- **A spliced report line.** `demote_read_time_transforms` returns `notes` as a list of KEY
  NAMES that callers join into "<keys> moved to the loader, so the dataset stages pack-indexed
  instead of being copied". The materialize branch returned a prose sentence, which produced
  "...reads will not inflate moved to the loader, so the dataset stages pack-indexed instead of
  being copied" -- a message describing the opposite of what happened. Under `materialize`
  NOTHING moves to the loader, so the honest return is no notes at all, which is already the
  documented signal for "the caller should report nothing".

The second was cosmetic but read as evidence the flag was not taking its branch, which is the
worse cost: it made a correct mechanism look broken to the person testing it.
"""

from __future__ import annotations

import unittest

from aiida_pino.utils.dataset_derivation import demote_read_time_transforms
from aiida_pino.workflows.train_model import (
    LAYOUT_COMPARISON_EXEMPT_KEYS,
    layouts_compatible,
)

LAYOUT = {
    "include_time": True, "time_as_batch": True, "time_as_dim": False,
    "moving_ref_frame": False, "moving_frame_steps": 1, "time_avg_window": 1,
    "super_res_factor": 1, "stride": [1, 1, 1], "time_frames": None,
}


class LayoutTest(unittest.TestCase):
    def test_materialize_does_not_make_a_variant_incompatible(self):
        selected = {**LAYOUT, "crop_z_top": 10, "derived_fields": [], "materialize": True}
        self.assertTrue(
            layouts_compatible(dict(LAYOUT), selected),
            "the flag alone refused the submission as an incompatible variant",
        )

    def test_it_is_exempt_for_the_same_reason_as_derived_fields(self):
        self.assertIn("materialize", LAYOUT_COMPARISON_EXEMPT_KEYS)

    def test_a_real_layout_conflict_is_still_refused(self):
        """The exemption must not blunt the guard it lives in."""
        selected = {**LAYOUT, "time_as_dim": True, "materialize": True}
        self.assertFalse(layouts_compatible(dict(LAYOUT), selected))


class NotesTest(unittest.TestCase):
    def test_materialize_reports_no_demotion_notes(self):
        """Notes are KEY NAMES a caller splices into a sentence; nothing moved here."""
        _profile, _env, notes = demote_read_time_transforms(
            {"crop_z_top": 10, "stride": [1, 1, 1], "materialize": True}, {},
            read_time_supported=True,
        )
        self.assertEqual(notes, [], f"a caller would render these as moved to the loader: {notes}")

    def test_the_crop_still_stays_in_the_derivation(self):
        profile, env, _ = demote_read_time_transforms(
            {"crop_z_top": 10, "stride": [1, 1, 1], "materialize": True}, {},
            read_time_supported=True,
        )
        self.assertEqual(profile.get("crop_z_top"), 10)
        self.assertNotIn("PINO_LAZY_CROP_Z_TOP", env)

    def test_a_normal_demotion_still_reports_its_keys(self):
        _profile, _env, notes = demote_read_time_transforms(
            {"crop_z_top": 10, "stride": [1, 1, 1]}, {}, read_time_supported=True
        )
        self.assertTrue(notes, "the ordinary demotion stopped reporting what it moved")
        self.assertTrue(all(len(n) < 40 for n in notes), f"notes must be key names, got {notes}")


if __name__ == "__main__":
    unittest.main()


class BakedStrideTest(unittest.TestCase):
    """A BAKED stride is the fourth instance of the key-for-key comparison defect.

    2026-09-09, five arms with the transform baked so staging would take the
    uncompressed writer. The failure split cleanly by WHAT was baked:

        baked STRIDE  95804, 95809, 95830  -> all exit 303, incompatible variant
        baked CROP    95816, 95822         -> reached staging

    `layouts_compatible` pops `stride` only when an ENV stride is set -- written when the
    normal case was a strided CONFIG against a full-resolution dataset. Baked, the env is
    empty by design, so the pop never fires and the comparison is [1,1,1] against [2,2,1].
    A training profile cannot express a baked stride at all: `build_derivation_profile_from_
    training_profile` reads `env.PINO_LAZY_STRIDE`, so there is no route by which the implied
    layout could carry one.

    Exempted only under `materialize`, not unconditionally: the guard's own history is that
    demanding a match refused 77 historical trainings, and removing it wholesale would give up
    a check that catches an accidental strided dataset. Under `materialize` the bake is the
    stated intent, which is the difference.
    """

    def _layout(self):
        return {
            "include_time": True, "time_as_batch": True, "time_as_dim": False,
            "moving_ref_frame": False, "moving_frame_steps": 1, "time_avg_window": 1,
            "super_res_factor": 1, "stride": [1, 1, 1], "time_frames": None,
        }

    def test_a_baked_stride_under_materialize_is_compatible(self):
        from aiida_pino.workflows.train_model import layouts_compatible

        selected = {**self._layout(), "stride": [2, 2, 1], "materialize": True}
        self.assertTrue(layouts_compatible(self._layout(), selected),
                        "a deliberately baked stride was refused as an incompatible variant")

    def test_a_strided_dataset_WITHOUT_materialize_is_still_refused(self):
        """The guard this exemption lives in must keep catching the accidental case."""
        from aiida_pino.workflows.train_model import layouts_compatible

        selected = {**self._layout(), "stride": [2, 2, 1]}
        self.assertFalse(layouts_compatible(self._layout(), selected))

    def test_an_env_stride_still_exempts_as_before(self):
        from aiida_pino.workflows.train_model import layouts_compatible

        selected = {**self._layout(), "stride": [2, 2, 1]}
        self.assertTrue(layouts_compatible(self._layout(), selected, env_stride="2,2,1"))

    def test_materialize_does_not_excuse_a_real_layout_conflict(self):
        from aiida_pino.workflows.train_model import layouts_compatible

        selected = {**self._layout(), "stride": [2, 2, 1], "time_as_dim": True,
                    "materialize": True}
        self.assertFalse(layouts_compatible(self._layout(), selected))
