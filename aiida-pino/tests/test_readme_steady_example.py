"""The README's steady example must pass the checks a submission runs.

As written until 2026-10-29 it did not: the training profile lacked the required
`cuda_device`, `hydra_overrides` and `wandb_offline`, and once those were added the
submission was refused as `incompatible_variant`, because the derivation profile left
out `time_as_batch`, whose base-layout default `true` switches the moving frame and the
average off, while the training profile's default is the steady `false` (found writing
reproducibility/train_evaluate.ipynb). This reads the two JSON blocks from the README
and runs the profile checks `presubmit` runs on them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from aiida_pino.utils.dataset_derivation import (
    build_derivation_profile_from_training_profile,
    validate_derivation_profile,
)
from aiida_pino.utils.profiles import validate_training_profile
from aiida_pino.workflows.train_model import layouts_compatible

README = Path(__file__).resolve().parents[1] / "README.md"
SECTION = "## Steady (time-averaged, beam-centred) runs"


def _example_profiles():
    body = README.read_text(encoding="utf-8").split(SECTION, 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```json\n(.*?)```", body, flags=re.DOTALL)
    assert len(blocks) == 2, f"expected a training and a derivation JSON block, found {len(blocks)}"
    training = json.loads(blocks[0].replace("<steady-config>", "config_flow3D_RIGNO_pvlhs_steady")
                          .replace("<name>", "readme_steady"))
    return training, json.loads(blocks[1])


def test_the_training_profile_has_every_required_key():
    training, _ = _example_profiles()
    validate_training_profile(training, expected_split_mode="standard")


def test_the_two_profiles_state_the_same_steady_layout():
    training, derivation = _example_profiles()
    effective_training = validate_training_profile(training, expected_split_mode="standard")
    requested = build_derivation_profile_from_training_profile(effective_training)
    effective = validate_derivation_profile(derivation)
    assert layouts_compatible(requested, effective)
    # and the derivation really is steady: not reduced to the base layout
    assert effective["moving_ref_frame"] is True
    assert (effective["time_avg_window"], effective["moving_frame_steps"]) == (30, 30)


def test_leaving_time_as_batch_out_of_the_derivation_is_inferred_as_it_describes():
    """Since 2026-10-30 the moving frame implies time_as_batch=False (with a warning)."""
    training, derivation = _example_profiles()
    derivation.pop("time_as_batch")
    effective_training = validate_training_profile(training, expected_split_mode="standard")
    requested = build_derivation_profile_from_training_profile(effective_training)
    assert layouts_compatible(requested, validate_derivation_profile(derivation))
