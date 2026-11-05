"""An eval must materialise the conditioning channels its model was trained on.

`build_derivation_profile_from_training_profile` reconstructs "what the model saw" for an
eval's dataset variant -- layout, crop, stride, moving frame -- but emitted no
`derived_fields`. So a model whose input keys include `conduction_field` got an eval variant
WITHOUT the channel baked, and the loader rebuilt it per case on demand, into the shared
`~/.pino/conduction_cache`. Six live evals were refilling that cache at ~60 GB/day on a box
whose full disk kills FLOW-3D solves (2026-09-04). Training stops needing
the cache by putting the field in the derivation profile's `derived_fields`; the eval path
had no way to arrive there.

The signal is the model's own declared inputs. Upstream builds the channel when
`"conduction_field" in requested_set` (npz_datasource.py:1247), and that set comes from the
`custom.*_input_keys` lists: an explicit `++custom.FNO_input_keys=[...]` override in the
training profile (207 of 590 stored models name `conduction_field` there), else the resolved
list in the model's archived `.hydra/config.yaml`. Only fields in `buildable_derived_fields()`
are promoted; trajectory channels like `beam_x` are loader-built and must not be.
"""

from __future__ import annotations

import inspect
import unittest

import pytest

from aiida_pino.utils.dataset_derivation import (
    build_derivation_profile_from_training_profile,
    requested_input_keys,
)


def _profile(overrides):
    return {"env": {}, "hydra_overrides": list(overrides)}


OVERRIDE_WITH = "++custom.FNO_input_keys=[V_scan,P,H_norm,x,y,z,beam_x,path_dose,conduction_field]"
OVERRIDE_WITHOUT = "++custom.FNO_input_keys=[V_scan,P,H_norm,x,y,z,beam_x,path_dose]"


class RequestedKeysTest(unittest.TestCase):
    def test_an_override_list_is_parsed(self):
        keys = requested_input_keys(_profile([OVERRIDE_WITH]))
        self.assertIn("conduction_field", keys)
        self.assertIn("beam_x", keys)

    def test_the_archived_config_is_the_fallback(self):
        cfg = {"custom": {"FNO_input_keys": ["V_scan", "conduction_field"],
                          "Energy_input_keys": ["u", "v", "w"]}}
        keys = requested_input_keys(_profile([]), hydra_config=cfg)
        self.assertIn("conduction_field", keys)
        self.assertIn("u", keys)

    def test_an_override_wins_over_the_archived_config(self):
        # Hydra's own semantics: the override replaced the list, so the archive's copy of the
        # SAME list is what the override overrode -- it must not resurrect a dropped key.
        cfg = {"custom": {"FNO_input_keys": ["V_scan", "conduction_field"]}}
        keys = requested_input_keys(_profile([OVERRIDE_WITHOUT]), hydra_config=cfg)
        self.assertNotIn("conduction_field", keys)

    def test_nothing_declared_is_an_empty_set_not_an_error(self):
        self.assertEqual(requested_input_keys(_profile([])), frozenset())
        self.assertEqual(requested_input_keys({}), frozenset())


class DerivedFieldsFollowTheModelTest(unittest.TestCase):
    def test_a_model_that_trained_on_a_baked_channel_evaluates_on_it(self):
        """What the model SAW: its recorded training derivation, not its input keys.

        Changed 2026-09-11. The input-key promotion materialised an eval variant for
        every conduction-conditioned model -- including the ~640 trained with the loader
        building the channel at read time, which never saw a baked variant at all. Baking
        is now opt-in at training (`materialize: true`), and an eval follows the record.
        """
        layout = build_derivation_profile_from_training_profile(
            _profile(["++custom.FNO_input_keys=[Temperature,conduction_field]"]),
            recorded_derived_fields=["conduction_field"],
        )
        self.assertEqual(layout["derived_fields"], ["conduction_field"])

    def test_a_model_that_conditioned_at_read_time_stays_at_read_time(self):
        layout = build_derivation_profile_from_training_profile(
            _profile(["++custom.FNO_input_keys=[Temperature,conduction_field]"]),
        )
        self.assertEqual(layout["derived_fields"], [])

    def test_a_model_without_it_gets_none(self):
        layout = build_derivation_profile_from_training_profile(_profile(["++custom.FNO_input_keys=[Temperature]"]))
        self.assertEqual(layout["derived_fields"], [])

    @pytest.mark.requires_lpno
    def test_the_record_is_taken_as_is_not_filtered_by_input_keys(self):
        """A recorded bake is a fact about the data the model trained on."""
        layout = build_derivation_profile_from_training_profile(
            _profile(["++custom.FNO_input_keys=[Temperature]"]), recorded_derived_fields=["mat_A"]
        )
        self.assertEqual(layout["derived_fields"], ["mat_A"])

    def test_an_unchanged_model_keeps_an_unchanged_profile(self):
        """No `derived_fields` -> identical profile -> identical derivation hash -> the
        existing variant is still reused. This is what keeps the change narrow."""
        before = build_derivation_profile_from_training_profile(
            _profile([OVERRIDE_WITHOUT]), unrecorded_crop_z_top=0
        )
        self.assertEqual(before["derived_fields"], [])


class WiringTest(unittest.TestCase):
    def test_the_eval_workchain_hands_the_builder_the_archived_config(self):
        from aiida_pino.workflows import evaluate_model as mod

        src = inspect.getsource(mod)
        self.assertIn("model_hydra_config", src, "the archive is not read on the eval path")
        self.assertIn("hydra_config=", src, "the archive is read but not passed to the builder")


if __name__ == "__main__":
    unittest.main()
