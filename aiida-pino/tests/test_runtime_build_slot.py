"""Local materialisation is serialised across datasets and waits for room.

Measured 2026-08-27: staging two datasets concurrently wrote 82 GB in
about three minutes (~40 GB + ~42 GB) and took root from 100 GB to 19 GB. No individual
check was wrong -- `refuse_build_that_cannot_fit` is per-build and stateless, so both reads
saw the same 100 GB free before either had written a byte. A space check cannot see a
sibling that has not allocated yet; only a lock can.

The failure shape is worth stating: the operator had checked the TARGET machine's disk
(2.2 TB free) and concluded staging was safe. The local build cost is invisible in
`stage-remote`'s contract.
"""

import multiprocessing
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aiida_pino.calculations.base import runtime_cache_materialise_slot
from aiida_pino.utils import runtime_cache as rc


def _hold(root, log, delay):
    """Take the slot, record enter/exit to a file, hold it briefly.

    A file rather than a multiprocessing Manager: the manager shuts down with the parent
    and dropped the final append, which made a working lock look broken.
    """
    with runtime_cache_materialise_slot(Path(root)):
        with open(log, "a") as fh:
            fh.write("enter\n"); fh.flush()
        time.sleep(delay)
        with open(log, "a") as fh:
            fh.write("exit\n"); fh.flush()


class MaterialiseSlotTest(unittest.TestCase):
    def test_two_builders_do_not_overlap(self):
        with TemporaryDirectory() as d:
            log = str(Path(d) / "events")
            a = multiprocessing.Process(target=_hold, args=(d, log, 1.0))
            a.start()
            # Let the first acquire before the second tries, so the ordering is meaningful.
            for _ in range(150):
                if Path(log).exists() and Path(log).read_text().count("enter"):
                    break
                time.sleep(0.05)
            b = multiprocessing.Process(target=_hold, args=(d, log, 0.1))
            b.start()
            a.join(timeout=60); b.join(timeout=60)
            events = Path(log).read_text().split()
            # enter/exit/enter/exit, never enter/enter: the second waited for the first.
            self.assertEqual(events, ["enter", "exit", "enter", "exit"], events)

    def test_the_lock_is_shared_across_datasets_not_per_cache(self):
        # The whole point: the sibling lock is keyed by cache name, which is why two
        # different datasets could build at once.
        import inspect
        src = inspect.getsource(runtime_cache_materialise_slot)
        self.assertIn(".materialise.lock", src)
        self.assertNotIn("cache_name", src)

    def test_a_dead_holder_does_not_wedge_later_builds(self):
        # flock is released by the kernel on exit, which matters on a box where workers
        # get killed.
        with TemporaryDirectory() as d:
            log = str(Path(d) / "events")
            p = multiprocessing.Process(target=_hold, args=(d, log, 0.2))
            p.start(); p.join(timeout=30)
            start = time.monotonic()
            with runtime_cache_materialise_slot(Path(d)):
                pass
            self.assertLess(time.monotonic() - start, 10)


class WaitForRoomTest(unittest.TestCase):
    def test_a_build_that_can_never_fit_is_refused_without_waiting(self):
        # Waiting cannot make a disk bigger than itself, so this must fail fast rather than
        # burn the timeout.
        members = [object()] * 4
        with TemporaryDirectory() as d:
            huge = 10**18  # larger than any real filesystem here
            original = rc.estimated_derived_bytes
            rc.estimated_derived_bytes = lambda *a, **k: huge
            try:
                start = time.monotonic()
                with self.assertRaises(RuntimeError) as caught:
                    rc.refuse_build_that_cannot_fit(Path(d), members, {}, label="huge")
                self.assertLess(time.monotonic() - start, 20, "it waited on an impossible build")
            finally:
                rc.estimated_derived_bytes = original
        self.assertIn("waiting cannot help", str(caught.exception).lower())

    def test_a_build_that_fits_returns_immediately(self):
        with TemporaryDirectory() as d:
            original = rc.estimated_derived_bytes
            rc.estimated_derived_bytes = lambda *a, **k: 1
            try:
                start = time.monotonic()
                rc.refuse_build_that_cannot_fit(Path(d), [object()], {}, label="tiny")
                self.assertLess(time.monotonic() - start, 5)
            finally:
                rc.estimated_derived_bytes = original

    def test_the_wait_is_configurable_and_bounded(self):
        self.assertGreater(rc.RUNTIME_BUILD_WAIT_S, 0)
        self.assertEqual(rc.RUNTIME_BUILD_WAIT_S, int(os.environ.get("AIIDA_PINO_BUILD_WAIT_S", 1800)))


if __name__ == "__main__":
    unittest.main()
