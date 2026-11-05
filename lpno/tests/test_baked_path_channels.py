"""Baked path channels are used only when the flags that shaped them match.

Path channels are (nt, X, Y, Z) arrays built from the manifest on every cache
miss -- ~1.9 GB of compute per miss for a 16-channel full-res case, to serve ONE
frame. Coherence-ON amortises that over a case's consecutive frames; coherence-OFF
at a small cache pays it per sample, which is the campaign's 4-17 s/step.

Baking them into the case NPZ turns the build into a read (the `precomp_path`
path, which already existed for build_steady_split's co-moving beam_dist). But
`power_weighted_dose` / `power_weighted_conduction` CHANGE the values, so a bake
made under one flag setting must not be served to a run using another -- that
would be silently wrong numbers, the one failure this campaign must not add.
Guard: a bake records its flags; a mismatch falls back to building (correct,
slow) rather than serving the wrong array.
"""
import unittest, tempfile, json
from pathlib import Path

import numpy as np

from lpno.models.pipeline.npz_lazy_store import _baked_channels_usable

META = "path_channel_meta"


class BakedChannelGuard(unittest.TestCase):
    def test_matching_flags_are_usable(self):
        meta = {"power_weighted_dose": False, "power_weighted_conduction": False}
        self.assertTrue(_baked_channels_usable(meta, power_weighted_dose=False,
                                               power_weighted_conduction=False))

    def test_mismatched_dose_flag_is_refused(self):
        meta = {"power_weighted_dose": False, "power_weighted_conduction": False}
        self.assertFalse(_baked_channels_usable(meta, power_weighted_dose=True,
                                                power_weighted_conduction=False))

    def test_mismatched_conduction_flag_is_refused(self):
        meta = {"power_weighted_dose": False, "power_weighted_conduction": False}
        self.assertFalse(_baked_channels_usable(meta, power_weighted_dose=False,
                                                power_weighted_conduction=True))

    def test_absent_meta_is_legacy_and_stays_usable(self):
        # build_steady_split bakes a co-moving beam_dist with no meta; that path
        # predates the guard and must keep working exactly as before.
        self.assertTrue(_baked_channels_usable(None, power_weighted_dose=False,
                                               power_weighted_conduction=False))

    def test_unparseable_meta_refuses_rather_than_guesses(self):
        self.assertFalse(_baked_channels_usable("not-a-dict", power_weighted_dose=False,
                                                power_weighted_conduction=False))


if __name__ == "__main__":
    unittest.main()
