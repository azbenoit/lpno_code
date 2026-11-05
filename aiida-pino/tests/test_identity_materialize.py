"""`materialize` gates on ENCODING, not on whether the numbers change.

Approved 2026-09-09, for fast training of models with coherence off, reversing
a decline of 2026-09-07. The argument that changed it:

**Materialisation does not change the payload, it changes the encoding.** Rewriting a
DEFLATE-compressed stored `payload.npz` as an uncompressed one produces byte-identical ARRAYS.
So `derivation_changes_payload` -- "do the numbers differ?" -- is the wrong question to gate it
on. The two coincide for a stride or a crop, which is why the feature worked there, and they
come apart exactly at the identity case, where the only thing wrong is the encoding.

The excluded population is the one that needs it most: of a 12-arm coherence-OFF wave, SEVEN
train at true full resolution with no transform to bake, so no payload-changing derivation
exists for them. They pay whole-case DEFLATE inflation on nearly every step -- 13800 ms/iter
with the GPU at 0 % and a loader worker pinned at 98.6 % CPU, against 133 ms/iter for a
comparable arm reading a cheap store. Days against hours. An arm that merely crops got the
cheap reads; an arm that reads everything did not.

Both protections stay exactly as they were: it is opt-in (`materialize: true`), and
`room_for_uncompressed_stage` still gates WHERE, so the compute box (1.6 TB free) is allowed and this
workstation (95 % full, 431 GB of unpackable loose objects) is refused. This widens WHICH
profiles may materialise, never where.
"""

from __future__ import annotations

import inspect
import unittest

from aiida_pino.data.pino_dataset_collection import (
    derivation_changes_payload,
    encoding_changes,
)

IDENTITY = {
    "stride": [1, 1, 1], "crop_z_top": 0, "derived_fields": [], "time_frames": None,
    "include_time": True, "time_as_batch": True, "time_as_dim": False,
    "moving_ref_frame": False, "moving_frame_steps": 1, "time_avg_window": 1,
    "super_res_factor": 1,
}


class PredicateTest(unittest.TestCase):
    def test_materialize_changes_the_encoding(self):
        self.assertTrue(encoding_changes({**IDENTITY, "materialize": True}))

    def test_an_identity_profile_changes_neither_by_default(self):
        self.assertFalse(encoding_changes(IDENTITY))
        self.assertFalse(derivation_changes_payload(IDENTITY))

    def test_it_is_independent_of_the_payload_question(self):
        """The whole point: an identity profile changes no numbers and still needs rewriting."""
        profile = {**IDENTITY, "materialize": True}
        self.assertFalse(derivation_changes_payload(profile),
                         "materialize must not be mistaken for a payload change")
        self.assertTrue(encoding_changes(profile))

    def test_it_ignores_a_non_boolean(self):
        for bad in ("true", 1, None, []):
            self.assertFalse(encoding_changes({**IDENTITY, "materialize": bad}),
                             f"{bad!r} was treated as an opt-in")

    def test_an_empty_profile_is_safe(self):
        self.assertFalse(encoding_changes(None))
        self.assertFalse(encoding_changes({}))


class DispatchTest(unittest.TestCase):
    """Two dispatch points decide this and they have to agree, or the parallel path silently
    reinstates the full copy -- the file's own comment says so."""

    def test_both_dispatch_points_consult_the_encoding_predicate(self):
        from aiida_pino.data import pino_dataset_collection as mod

        src = inspect.getsource(mod)
        self.assertEqual(src.count("or encoding_changes(profile)"), 2,
                         "the two dispatch points disagree about materialize")

    def test_the_room_gate_is_untouched(self):
        """This widens WHICH profiles materialise, never WHERE."""
        from aiida_pino.utils.runtime_cache import (
            MATERIALIZE_SPARE_BYTES,
            room_for_uncompressed_stage,
        )

        GB = 1024 ** 3
        self.assertEqual(MATERIALIZE_SPARE_BYTES, 50 * GB)
        self.assertFalse(room_for_uncompressed_stage(free=91 * GB, needed=69 * GB))
        self.assertTrue(room_for_uncompressed_stage(free=1600 * GB, needed=69 * GB))


if __name__ == "__main__":
    unittest.main()
