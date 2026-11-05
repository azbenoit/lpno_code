"""import-model must record the EFFECTIVE profile, not a defaulted one.

Two defects reported 2026-08-27: the import stored the raw profile (so `compare-models`
called an imported/native pair CONFOUNDED when their overrides were identical), and a
partial profile silently defaulted `env` to `{}`. Since the gas crop became the default,
`env` carries `PINO_LAZY_CROP_Z_TOP` -- the input geometry itself -- so a dropped `env`
is a model record that misstates what it was trained on.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aiida_pino.console.service import AiidaConsoleService

PROFILE = {
    "config_name": "randompath_rigno",
    "host_ram_mb": 24576,
    "gpu_vram_mb": 16000,
    "env": {"PINO_LAZY_CROP_Z_TOP": "10", "PINO_LAZY_MOVING_REF": "1"},
}


class _Computer:
    transport_type = "core.local"


class _Artifacts:
    computer = _Computer()

    def __init__(self, path):
        self._path = str(path)

    def get_remote_path(self):
        return self._path


def _read(path):
    return AiidaConsoleService._recorded_training_profile(object.__new__(AiidaConsoleService), _Artifacts(path))


class RecordedProfileTest(unittest.TestCase):
    def test_the_accessors_used_here_exist_on_the_real_aiida_classes(self):
        # The fake above is only honest if it mirrors the real API. A previous fix in this
        # file's neighbourhood called an invented `get_transport_type()`, and a fake that
        # defined it kept the suite green while the feature was inert in production.
        from aiida import orm

        self.assertTrue(hasattr(orm.RemoteData, "get_remote_path"))
        self.assertTrue(hasattr(orm.RemoteData, "get_authinfo"))
        self.assertIsInstance(orm.Computer.transport_type, property)

    def test_reads_the_effective_profile_a_finished_run_recorded(self):
        with TemporaryDirectory() as d:
            (Path(d) / "train_summary.json").write_text(json.dumps({"profile": PROFILE}))
            recorded, src = _read(d)
        self.assertEqual(recorded["env"]["PINO_LAZY_CROP_Z_TOP"], "10")
        self.assertEqual(recorded["config_name"], "randompath_rigno")
        self.assertTrue(src.endswith("train_summary.json"))

    def test_a_missing_or_broken_summary_is_not_an_error(self):
        # Refusing an import over a missing convenience would be worse than the partial
        # record it replaces, so both of these fall back to the caller's profile.
        with TemporaryDirectory() as d:
            self.assertEqual(_read(d), ({}, ""))
            (Path(d) / "train_summary.json").write_text("{not json")
            self.assertEqual(_read(d), ({}, ""))
            (Path(d) / "train_summary.json").write_text(json.dumps({"steps": 50000}))
            self.assertEqual(_read(d), ({}, ""))

    def test_a_run_trained_outside_aiida_gives_its_profile_from_the_hydra_record(self):
        # 2026-10-08: a cluster retrain imported with no train_summary.json got a defaulted
        # profile (another config, no overrides, 6000 steps, wandb online), and its AiiDA
        # evaluation scored a different beam box and logged to wandb.
        with TemporaryDirectory() as d:
            hydra = Path(d) / ".hydra"
            hydra.mkdir()
            (hydra / "config.yaml").write_text("training:\n  max_steps: 32000\nnetwork_dir: /work/run/model\n")
            (hydra / "hydra.yaml").write_text("hydra:\n  job:\n    config_name: config_flow3D_FNO_pvlhs_steady\n")
            (hydra / "overrides.yaml").write_text(
                "- custom.data_dir=/work/data/train\n- network_dir=/work/run/model\n"
                "- ++custom.seed=42\n- ++custom.FNO_input_keys=[V_scan,P]\n")
            recorded, src = _read(d)
        self.assertEqual(recorded["config_name"], "config_flow3D_FNO_pvlhs_steady")
        self.assertEqual(recorded["max_steps"], 32000)
        self.assertEqual(recorded["hydra_overrides"], ["++custom.seed=42", "++custom.FNO_input_keys=[V_scan,P]"])
        self.assertTrue(recorded["wandb_offline"])
        self.assertTrue(src.endswith(".hydra"))

    def test_the_merged_profile_keeps_env_through_validation(self):
        # The merge is only half the fix: the validator downstream is what actually decides
        # which keys reach the stored node, and it is where a dropped `env` would reappear.
        # A caller who names a key still overrides the record.
        merged = {**PROFILE, "config_name": "override_me"}
        out = AiidaConsoleService._validated_training_profile(
            object.__new__(AiidaConsoleService), merged, split_mode="standard", for_import=True
        )
        self.assertEqual(out["config_name"], "override_me")
        self.assertEqual(out["env"]["PINO_LAZY_CROP_Z_TOP"], "10")
        self.assertEqual(out["env"]["PINO_LAZY_MOVING_REF"], "1")
        self.assertEqual(out["host_ram_mb"], 24576)


if __name__ == "__main__":
    unittest.main()
