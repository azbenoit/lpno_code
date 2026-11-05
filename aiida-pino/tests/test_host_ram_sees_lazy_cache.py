"""The host-RAM guard must not ignore `PINO_LAZY_CACHE`, a per-case multiplier.

`_check_host_ram_against_loader_settings` modelled `16384 + workers * per_worker`, with no
term for the loader cache at all -- while `vram.py` already lists `PINO_LAZY_CACHE` in
`MEMORY_RELEVANT_ENV_KEYS` and `eval_adapter` defaults evals to 3. So one module calls the
variable memory-relevant and the admission check could not see it.

It is a case COUNT, and the cases are held resident. A measured staged
kh40 case at ~470 MB post-stride, so `PINO_LAZY_CACHE=96` holds ~45 GB -- several times the
entire modelled requirement. Booking the modelled floor while caching 96 cases is admitted
today and packed beside co-tenants the box cannot hold -- gpu_spool books by the declared
figure. (The 1.5x cgroup kill of BUG-43 is opt-in since lpno f90778e and off on
both boxes, so the cost is mis-packing, not a kill.)

The term here is a deliberate LOWER bound, not a model: 256 MiB per cached case, well under
the 470 MiB measured, so it refuses gross under-declaration without rejecting a booking that
is merely tight. Calibrating it properly needs a measured peak from a run that caches HEAVILY:
391 of 590 stored models set the variable, but every one at 1 or 2, so the 372 measured
peaks calibrate the floor and not this term.
"""

from __future__ import annotations

import unittest

from aiida_pino.utils.profiles import _check_host_ram_against_loader_settings as check


def _profile(*, host_ram_mb, cache=None, workers=4, stride="2,2,1"):
    env = {"PINO_LOADER_WORKERS": str(workers), "PINO_LAZY_STRIDE": stride}
    if cache is not None:
        env["PINO_LAZY_CACHE"] = str(cache)
    return {"host_ram_mb": host_ram_mb, "env": env}


class LazyCacheTest(unittest.TestCase):
    def test_a_large_cache_with_a_bare_booking_is_refused(self):
        # 16384 + 4*1024 = 20480 passes today with any cache size at all.
        with self.assertRaises(ValueError) as caught:
            check(_profile(host_ram_mb=20480, cache=96))
        self.assertIn("PINO_LAZY_CACHE", str(caught.exception),
                      "the refusal must name the variable that caused it")

    def test_a_booking_that_covers_the_cache_is_admitted(self):
        # A hand-sized booking: 64 GB for 96 cached cases.
        check(_profile(host_ram_mb=65536, cache=96))

    def test_a_small_cache_does_not_move_the_floor(self):
        """The eval adapter's default of 3 must not start refusing evals."""
        check(_profile(host_ram_mb=20480, cache=3))

    def test_no_cache_declared_behaves_exactly_as_before(self):
        check(_profile(host_ram_mb=20480))

    def test_an_unparseable_cache_is_ignored_rather_than_fatal(self):
        # A bad value must not turn a booking check into a crash.
        check(_profile(host_ram_mb=20480, cache="lots"))

    def test_serial_loading_is_still_exempt(self):
        # workers <= 0 returns early; the cache lives in the main process there.
        check(_profile(host_ram_mb=4096, cache=96, workers=0))


if __name__ == "__main__":
    unittest.main()
