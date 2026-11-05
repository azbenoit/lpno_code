"""`materialize: true` keeps a read-time transform in the derivation so it is BAKED.

`demote_read_time_transforms` moves a crop or stride out of the derivation into loader env
vars, and it is right by default: a stride alone once wrote 46 GB of `.npy` for a transform
the loader applies for free, and off-box it is 13.3 GB of archives against 309 GB of `.npy`.

But it is also what makes uncompressed staging unreachable. Demoted, the profile is a no-op,
members stage as pack-index byte ranges into the object store, and the compact writer never
runs -- so the stored `payload.npz` is what the loader reads, and its members are DEFLATE.
Measured on a full-resolution coherence-OFF FNO arm: 13800 ms/iter with the GPU at 0 % and one
loader worker at 98.6 % CPU, against 133 ms/iter for a comparable run
(2026-09-07). Their attempt to reach the fast path by moving the crop INTO the derivation was
undone by demotion putting it straight back out -- verified on one calcjob, which received
`env.CROP='10'` after the submission cleared it.

So the opt-out belongs here, in series with the room check added the same day: this decides
whether materialising is ASKED FOR, `room_for_uncompressed_stage` decides whether it is
ALLOWED. Approved 2026-09-07 on the trade ~13 GB compressed against ~69 GB
uncompressed for the two runtime fields -- 5.3x disk, not the 23x in the demotion docstring,
which assumed all thirteen fields.

Identity: `materialize` DOES change the dataset, because a baked crop and a read-time crop
produce different stored arrays. Absent or False it is stripped before hashing, so no existing
profile's identity moves -- the same rule as a default `stride` and `time_frames`, and for the
same reason (one test split was minted as a twin of another by a default that was not stripped).
"""

from __future__ import annotations

import unittest

from aiida_pino.utils.dataset_derivation import (
    demote_read_time_transforms,
    derivation_hash,
    validate_derivation_profile,
)

CROPPED = {"crop_z_top": 10, "stride": [1, 1, 1]}


class OptOutTest(unittest.TestCase):
    def test_materialize_keeps_the_crop_in_the_derivation(self):
        profile, env, notes = demote_read_time_transforms(
            {**CROPPED, "materialize": True}, {}, read_time_supported=True
        )
        self.assertEqual(profile.get("crop_z_top"), 10, "the crop was demoted anyway")
        self.assertNotIn("PINO_LAZY_CROP_Z_TOP", env)

    def test_without_it_the_crop_is_still_demoted(self):
        profile, env, notes = demote_read_time_transforms(CROPPED, {}, read_time_supported=True)
        self.assertEqual(env.get("PINO_LAZY_CROP_Z_TOP"), "10")
        self.assertIn(profile.get("crop_z_top"), (0, None))

    def test_it_is_a_validated_boolean(self):
        self.assertIs(validate_derivation_profile({"materialize": True})["materialize"], True)
        with self.assertRaises(ValueError):
            validate_derivation_profile({"materialize": "yes"})


class IdentityTest(unittest.TestCase):
    def test_absent_or_false_does_not_move_any_existing_hash(self):
        base = dict(CROPPED)
        self.assertEqual(derivation_hash(base), derivation_hash({**base, "materialize": False}))

    def test_materialize_true_is_a_different_dataset(self):
        """A baked crop and a read-time crop produce different stored arrays."""
        base = dict(CROPPED)
        self.assertNotEqual(derivation_hash(base), derivation_hash({**base, "materialize": True}))


if __name__ == "__main__":
    unittest.main()
