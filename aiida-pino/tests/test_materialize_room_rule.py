"""Full-resolution uncompressed staging is allowed only where the destination has room.

A derivation that changes the payload but does NOT stride (a crop, say) currently falls to the
manifest writer, which leaves reads compressed. Decompression then dominates: a full-res
coherence-OFF FNO arm measured 13800 ms/iter with the GPU at 0 % and one loader worker pinned
at 98.6 % CPU -- 123 h for 32000 steps against 133 ms/iter for the fastest comparable run
(2026-09-07). Reading an uncompressed case costs 0.076 s against 1.58 s
compressed, 21x.

The compact-NPZ writer already writes UNCOMPRESSED (`np.savez`) and `_stride_compact_npz_arrays`
is a no-op at (1,1,1), so the only blocker was `if stride == (1, 1, 1): return False` -- a
size-based judgement written when the point of that path was assumed to be SHRINKING data.

The cost is disk: ~722 MB per case, ~69 GB for a 96-member set. User rule (2026-09-07): allow
it when the destination has room with 50 GB still spare afterwards. That admits a compute machine
(1.6 TB free) and refuses the submitting machine (91 GB free, 95 % used), where a full disk stalls
FLOW-3D solves -- without needing a per-box list.
"""

from __future__ import annotations

import unittest

from aiida_pino.utils.runtime_cache import (
    MATERIALIZE_SPARE_BYTES,
    room_for_uncompressed_stage,
)

GB = 1024 ** 3


class RoomRuleTest(unittest.TestCase):
    def test_the_spare_requirement_is_50_gb(self):
        self.assertEqual(MATERIALIZE_SPARE_BYTES, 50 * GB)

    def test_a_roomy_destination_is_allowed(self):
        # the compute box: 1.6 TB free, 69 GB wanted.
        self.assertTrue(room_for_uncompressed_stage(free=1600 * GB, needed=69 * GB))

    def test_this_workstation_is_refused(self):
        # 91 GB free - 69 GB = 22 GB left, under the 50 GB spare.
        self.assertFalse(room_for_uncompressed_stage(free=91 * GB, needed=69 * GB))

    def test_exactly_fifty_spare_is_allowed(self):
        self.assertTrue(room_for_uncompressed_stage(free=119 * GB, needed=69 * GB))

    def test_one_byte_under_is_refused(self):
        self.assertFalse(room_for_uncompressed_stage(free=119 * GB - 1, needed=69 * GB))

    def test_an_unknown_size_is_refused_not_assumed_free(self):
        """Unknown must not read as zero: that is how a gate lets through what it cannot size."""
        self.assertFalse(room_for_uncompressed_stage(free=1600 * GB, needed=None))
        self.assertFalse(room_for_uncompressed_stage(free=None, needed=69 * GB))

    def test_a_negative_or_zero_free_is_refused(self):
        self.assertFalse(room_for_uncompressed_stage(free=0, needed=1 * GB))


if __name__ == "__main__":
    unittest.main()
