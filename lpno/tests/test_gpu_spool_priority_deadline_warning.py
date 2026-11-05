"""A priority seat must ANNOUNCE the deadline it silently buys.

`--priority` does two things, not one: it parks a `.want` marker to jump the
queue, AND it swaps the unbounded seat wait for DEFAULT_PRIORITY_MAX_WAIT_S
(3600 s) "so CI reports a cause". On a saturated box the second half is fatal --
two evals died at 3601 s on 2026-07-29, and a training arm died at 3602 s on
2026-09-10 waiting for a seat ~33 h out. Both times the queue-jump was wanted and
the deadline was not, and both times it was discovered from the corpse.

The deadline stays (CI depends on it), but a run must say up front what it has
bought, so the next reader sees it in the log before it kills them.
"""
import unittest

from lpno import gpu_spool as gs


class PriorityDeadlineIsAnnounced(unittest.TestCase):
    def test_priority_wait_message_states_the_deadline(self):
        msg = gs._seat_wait_notice(priority=True, max_wait_s=gs.DEFAULT_PRIORITY_MAX_WAIT_S)
        self.assertIn("3600", msg)
        self.assertIn("give up", msg.lower())
        # names the escape hatch, or the reader cannot act on the warning
        self.assertIn("--max-wait-s", msg)

    def test_priority_notice_states_the_yield_window(self):
        """The `.want` marker IS the queue-jump, and it is dropped after
        DEFAULT_PRIORITY_YIELD_S (900 s), after which the run continues as an ordinary
        waiter. So "priority" is a 15-minute window, not a durable claim -- for a seat
        hours out the jump is gone for almost the whole wait. A run declined the flag on
        exactly this ground (2026-09-10) after its predecessor's log showed the yield at
        900 s followed by 2700 s of ordinary waiting, which both reviewers read past."""
        msg = gs._seat_wait_notice(priority=True, max_wait_s=0.0,
                                   priority_yield_s=gs.DEFAULT_PRIORITY_YIELD_S)
        self.assertIn("900", msg)
        self.assertTrue("ordinary" in msg.lower() or "yield" in msg.lower())

    def test_non_priority_notice_does_not_mention_the_yield(self):
        msg = gs._seat_wait_notice(priority=False, max_wait_s=0.0,
                                   priority_yield_s=gs.DEFAULT_PRIORITY_YIELD_S)
        self.assertNotIn("900", msg)

    def test_ordinary_seat_says_it_waits_indefinitely(self):
        msg = gs._seat_wait_notice(priority=False, max_wait_s=gs.DEFAULT_SEAT_MAX_WAIT_S)
        self.assertIn("indefinitely", msg.lower())
        self.assertNotIn("give up", msg.lower())

    def test_explicit_unbounded_priority_is_not_warned_about(self):
        # --priority --max-wait-s 0 is the correct combination for a long queue;
        # warning about a deadline that is not set would train people to ignore it.
        msg = gs._seat_wait_notice(priority=True, max_wait_s=0.0)
        self.assertIn("indefinitely", msg.lower())


if __name__ == "__main__":
    unittest.main()
