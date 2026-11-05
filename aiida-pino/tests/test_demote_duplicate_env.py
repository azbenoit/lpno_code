"""A read-time value the caller ALREADY set is a duplicate, not a composition.

`demote_read_time_transforms` refused to demote whenever the env already carried the twin
var, on the grounds that composing two reductions squares the grid. True when they DIFFER.
When they are the SAME value it is one transform expressed twice, and bailing out leaves
`crop_z_top` baked in the profile -- which (a) makes the derivation payload-changing, so the
collection MATERIALISES, and (b) makes the staged variant record a crop the env var also
asks for, which `_resolve_read_time_crop` then refuses as a double-crop.

Measured 2026-08-28: seven crop-equivalent reruns submitted exactly that way wrote 117 GB
into the runtime cache in 90 minutes, took root to 99 %, and every one of them failed anyway.
"""
import unittest

from aiida_pino.utils.dataset_derivation import demote_read_time_transforms


def _prof(**kw):
    base = {"stride": [1, 1, 1], "crop_z_top": 0, "include_time": True,
            "time_as_batch": True, "time_as_dim": False, "moving_ref_frame": False,
            "moving_frame_steps": 1, "time_avg_window": 1, "super_res_factor": 1}
    base.update(kw)
    return base


class DuplicateEnvValue(unittest.TestCase):
    def test_same_crop_in_env_demotes_instead_of_baking(self):
        p, e, notes = demote_read_time_transforms(
            _prof(crop_z_top=10), {"PINO_LAZY_CROP_Z_TOP": "10"}, read_time_supported=True)
        self.assertEqual(p["crop_z_top"], 0, "must not leave the crop baked -> materialises")
        self.assertEqual(e["PINO_LAZY_CROP_Z_TOP"], "10", "env value must survive untouched")

    def test_same_stride_in_env_demotes(self):
        p, e, _ = demote_read_time_transforms(
            _prof(stride=[2, 2, 1]), {"PINO_LAZY_STRIDE": "2,2,1"}, read_time_supported=True)
        self.assertEqual(list(p["stride"]), [1, 1, 1])
        self.assertEqual(e["PINO_LAZY_STRIDE"], "2,2,1")

    def test_DIFFERENT_crop_still_refuses(self):
        """Genuine ambiguity: which of 10 and 5 did the caller mean? Refuse, unchanged."""
        p, e, notes = demote_read_time_transforms(
            _prof(crop_z_top=10), {"PINO_LAZY_CROP_Z_TOP": "5"}, read_time_supported=True)
        self.assertEqual(p["crop_z_top"], 10, "must be returned untouched")
        self.assertEqual(notes, [])

    def test_DIFFERENT_stride_still_refuses(self):
        p, _, notes = demote_read_time_transforms(
            _prof(stride=[2, 2, 1]), {"PINO_LAZY_STRIDE": "4,4,1"}, read_time_supported=True)
        self.assertEqual(list(p["stride"]), [2, 2, 1])
        self.assertEqual(notes, [])

    def test_both_duplicated_together(self):
        p, e, _ = demote_read_time_transforms(
            _prof(stride=[2, 2, 1], crop_z_top=10),
            {"PINO_LAZY_STRIDE": "2,2,1", "PINO_LAZY_CROP_Z_TOP": "10"},
            read_time_supported=True)
        self.assertEqual(list(p["stride"]), [1, 1, 1])
        self.assertEqual(p["crop_z_top"], 0)

    def test_one_duplicate_one_conflict_refuses_whole(self):
        """A conflict anywhere is still a refusal -- do not half-apply."""
        p, _, notes = demote_read_time_transforms(
            _prof(stride=[2, 2, 1], crop_z_top=10),
            {"PINO_LAZY_STRIDE": "2,2,1", "PINO_LAZY_CROP_Z_TOP": "5"},
            read_time_supported=True)
        self.assertEqual(p["crop_z_top"], 10)
        self.assertEqual(notes, [])

    def test_unsupported_target_still_bakes(self):
        p, _, notes = demote_read_time_transforms(
            _prof(crop_z_top=10), {"PINO_LAZY_CROP_Z_TOP": "10"}, read_time_supported=False)
        self.assertEqual(p["crop_z_top"], 10)
        self.assertEqual(notes, [])


if __name__ == "__main__":
    unittest.main()
