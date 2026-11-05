"""The safe gas crop is measured from the data, and a crop that cuts metal says so loudly.

`crop_z_top` was a constant 10, taken from the deck geometry (surface at z=0, 95 um of
headroom at dz=10 um). Material does not respect that line -- the melt bulges and the
keyhole ejects into the headroom -- so the geometric gas height is not the safe crop.
Measured rim-only over `ti64_preheat60`'s 81-member train split: **81 of 81 cases lose
material at crop 10**. Distribution of metal-free slices: {2:1, 4:3, 5:5, 6:7, 7:9, 8:13,
9:43} -- median 9, but one case (40971) tolerates only 2, and it is a genuine continuous
mound rather than spatter.
"""

from __future__ import annotations

import unittest

import numpy as np

from aiida_pino.utils.gas_crop import (
    MATERIAL_FIELD,
    METAL_FREE_EXTRA,
    crop_advisory,
    metal_free_top_slices,
    survey_members,
)


class _Archive(dict):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def files(self):
        return list(self)


class _Sim:
    """Minimal stand-in: a payload plus a writable extras bag."""

    def __init__(self, material=None, pk=1, cached=None, broken=False):
        self.pk = pk
        self._material = material
        self._broken = broken
        store = {} if cached is None else {METAL_FREE_EXTRA: cached}

        class _Extras:
            all = store

            @staticmethod
            def set(key, value):
                store[key] = value

        self.base = type("B", (), {"extras": _Extras})

    def _open_payload_archive(self):
        if self._broken:
            raise OSError("payload unreadable")
        if self._material is None:
            return _Archive()
        return _Archive({MATERIAL_FIELD: self._material})


def _material(nz, top_metal_index, nt=3, nx=2, ny=2):
    """An array whose highest material-bearing z index is `top_metal_index`."""
    a = np.zeros((nt, nx, ny, nz), dtype=np.float32)
    a[..., : top_metal_index + 1] = 1.0
    return a


class MeasurementTest(unittest.TestCase):
    def test_it_counts_the_metal_free_slices_at_the_top(self):
        # nz=50, topmost material at 41 -> 8 free slices (the measured sim 40757).
        self.assertEqual(metal_free_top_slices(_Sim(_material(50, 41))), 8)

    def test_detached_spatter_does_not_block_the_crop(self):
        """A droplet in flight is not the rim.

        Cropping through spatter loses a few voxels of an object the model cannot track
        anyway; cropping through the rim truncates the surface the melt pool sits in. The
        earlier rule counted ANY material and this case asserted the opposite -- a single
        detached cell dragging the whole dataset's crop down. Measured on
        `ti64_preheat60`, excluding spatter recovers 59896 (2 -> 4 free slices) and 41104
        (3 -> 4).
        """
        a = _material(50, 39)
        a[2, 1, 1, 45] = 1.0          # one detached cell, six slices above the surface
        self.assertEqual(metal_free_top_slices(_Sim(a)), 10)

    def test_material_connected_to_the_substrate_still_blocks_it(self):
        # The other side of the same rule, and the one that matters: sim 40971's mound
        # reaches z=48 as one continuous structure, so only 2 slices are free there.
        a = _material(50, 39)
        a[2, 1, 1, 40:46] = 1.0       # a column joined to the substrate below
        self.assertEqual(metal_free_top_slices(_Sim(a)), 4)

    def test_a_frame_with_only_free_flying_material_yields_no_rim(self):
        # Nothing touches the floor: there is no rim to protect in that frame.
        a = np.zeros((2, 3, 3, 20), dtype=np.float32)
        a[0, 1, 1, 15] = 1.0
        self.assertEqual(metal_free_top_slices(_Sim(a)), 20)

    def test_a_fully_void_column_is_entirely_croppable(self):
        self.assertEqual(metal_free_top_slices(_Sim(np.zeros((2, 2, 2, 12), dtype=np.float32))), 12)

    def test_unmeasurable_is_none_and_never_zero_or_unlimited(self):
        # None must read as "do not know". Returning 0 would disable the crop silently;
        # returning nz would authorise cropping the whole domain.
        self.assertIsNone(metal_free_top_slices(_Sim(broken=True)))
        self.assertIsNone(metal_free_top_slices(_Sim(material=None)))

    def test_the_answer_is_cached_on_the_node(self):
        sim = _Sim(_material(50, 41))
        self.assertEqual(metal_free_top_slices(sim), 8)
        self.assertEqual(sim.base.extras.all[METAL_FREE_EXTRA], 8)
        # A cached value is trusted without re-reading the payload.
        cached = _Sim(material=None, cached=7)
        self.assertEqual(metal_free_top_slices(cached), 7)


class SurveyTest(unittest.TestCase):
    def test_the_dataset_crop_is_the_shallowest_member(self):
        members = [_Sim(_material(50, 41), pk=1), _Sim(_material(50, 47), pk=2)]
        survey = survey_members(members)
        self.assertEqual(survey["safe_crop"], 2, "the shallowest case binds the dataset")
        self.assertEqual(survey["measured"], 2)

    def test_an_unmeasured_member_is_reported_not_folded_in(self):
        # Folding it in as 0 would silently disable the crop for the whole dataset.
        survey = survey_members([_Sim(_material(50, 41), pk=1), _Sim(broken=True, pk=9)])
        self.assertEqual(survey["safe_crop"], 8)
        self.assertEqual(survey["unmeasured"], [9])


class AdvisoryTest(unittest.TestCase):
    def _survey(self):
        return survey_members([_Sim(_material(50, 41), pk=1), _Sim(_material(50, 47), pk=2)])

    def test_a_safe_crop_says_nothing(self):
        self.assertIsNone(crop_advisory(self._survey(), 2))
        self.assertIsNone(crop_advisory(self._survey(), 0))

    def test_an_unsafe_crop_names_the_count_and_the_recommendation(self):
        msg = crop_advisory(self._survey(), 10)
        self.assertIsNotNone(msg)
        self.assertIn("2 of 2", msg)               # how many cases are affected
        self.assertIn("RECOMMENDED crop_z_top=2", msg)  # what to use instead
        self.assertIn("CROP CUTS METAL", msg)      # loud

    def test_it_names_the_worst_offenders(self):
        msg = crop_advisory(self._survey(), 10)
        self.assertIn("2 (keeps 2)", msg)

    def test_unmeasured_members_are_disclosed_not_hidden(self):
        survey = survey_members([_Sim(_material(50, 47), pk=2), _Sim(broken=True, pk=9)])
        msg = crop_advisory(survey, 10)
        self.assertIn("could not be measured", msg)

    def test_an_unmeasurable_dataset_produces_no_false_alarm(self):
        # Nothing measured means nothing known; inventing a warning would be as wrong as
        # inventing a crop.
        self.assertIsNone(crop_advisory(survey_members([_Sim(broken=True)]), 10))


class WiringTest(unittest.TestCase):
    def test_the_submission_seam_surveys_and_advises(self):
        import inspect

        from aiida_pino.utils.presubmit import resolve_training_submission

        src = inspect.getsource(resolve_training_submission)
        self.assertIn("survey_members", src)
        self.assertIn("crop_advisory", src)
        # An explicit request is honoured (warned only); a DEFAULTED one is corrected.
        self.assertIn("not explicit and crop > int(safe)", src)


if __name__ == "__main__":
    unittest.main()


class BakedAdvisoryTest(unittest.TestCase):
    """Baking a metal-cutting crop is not the same warning as applying it at read time.

    Raised 2026-09-07 while testing `materialize`. With a read-time
    crop the advisory is genuinely advisory: the stored arrays keep the material and only the
    served view drops it, so the decision is reversible by clearing an env var. BAKED, the
    metal is physically absent from the staged variant, every model trained on it saw
    truncated geometry, and the only recovery is re-deriving the dataset.

    Measured on `ti64_preheat60`'s lab set: crop 10 removes material from 96 of 96 members with
    a safe value of 5. That is exactly the case where baking changes what the number MEANS.
    """

    def _survey(self):
        return survey_members([_Sim(_material(50, 41), pk=1), _Sim(_material(50, 47), pk=2)])

    def test_a_baked_unsafe_crop_says_it_is_irreversible(self):
        msg = crop_advisory(self._survey(), 10, baked=True)
        self.assertIsNotNone(msg)
        low = msg.lower()
        self.assertTrue(any(w in low for w in ("irreversible", "cannot be recovered", "re-deriv")),
                        f"a baked crop must not read as advisory: {msg}")

    def test_the_read_time_wording_is_unchanged(self):
        msg = crop_advisory(self._survey(), 10)
        self.assertIn("RECOMMENDED crop_z_top=2", msg)
        self.assertNotIn("irreversible", msg.lower())

    def test_a_safe_baked_crop_still_says_nothing(self):
        self.assertIsNone(crop_advisory(self._survey(), 2, baked=True))
