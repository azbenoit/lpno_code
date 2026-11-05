"""A moving-frame derived member stages as ARRAYDIR, never as the flat consolidated npz.

Regression with a consumer (a steady RIGNO evaluation). The
flat consolidated npz routes to pino's NpzDirStore, whose lazy path refuses
`moving_ref_frame=true` BY DESIGN -- it cannot know the derivation is already baked. Every
steady eval that ever worked (61779 -> 7.4 K stable; 68119) was served by
`ArrayDirCollectionStore` from the pre-b616eaa arraydir layout; switching derived members
to the flat npz regressed BOTH routes into a steady eval (derive-at-submit and
pre-materialised). Steady members are small -- the time average collapses the
frame axis -- so the disk motive behind the flat npz does not apply to them.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def _task(root, profile):
    import io

    member = Path(root) / "members" / "simulation_0000"
    member.mkdir(parents=True)
    nt, nx, ny, nz = 6, 8, 5, 4
    buf = io.BytesIO()
    np.savez(buf,
             Temperature=np.random.default_rng(0).random((nt, nx, ny, nz), dtype=np.float32),
             x=np.linspace(0, 1, nx, dtype=np.float32),
             y=np.linspace(0, 1, ny, dtype=np.float32),
             z=np.linspace(0, 1, nz, dtype=np.float32),
             time_s=np.linspace(0, 1e-3, nt, dtype=np.float32),
             power_w=np.float32(100.0), scan_speed_m_per_s=np.float32(0.5),
             case_name=np.asarray("case_s"))
    staged = Path(root) / "payload.npz"
    staged.write_bytes(buf.getvalue())
    return {
        "index": 0, "target_dir": str(member), "staged_path": str(staged),
        "profile": profile, "case_name": "case_s", "laser_power": 100.0, "v_scan": 0.5,
        "time_values_s": np.linspace(0, 1e-3, nt).tolist(),
        "requested_fields": ["Temperature"], "case_manifest": {"case_name": "case_s"},
        "moving_frame_slices": [(0, 6), (1, 7), (2, 8)],  # 3 kept frames, window nx-(steps-1)=6
    }, member


class SteadyLayoutTest(unittest.TestCase):
    def test_a_moving_frame_member_writes_npy_not_flat_npz(self):
        from aiida_pino.utils.runtime_export import derived_member_worker

        with TemporaryDirectory() as d:
            task, member = _task(d, {"moving_ref_frame": True, "moving_frame_steps": 3,
                                     "time_avg_window": 3})
            result = derived_member_worker(task)
            self.assertEqual(result["layout"], "arraydir")
            self.assertTrue((member / "Temperature.npy").is_file())
            self.assertTrue((member / "time_values_s.npy").is_file())
            # No flat npz anywhere: its presence is what routes to the refusing store.
            self.assertEqual(list(Path(d).glob("*.npz")), [Path(d) / "payload.npz"])
            self.assertEqual(list(member.glob("*.npz")), [])
            entries = {e["original_name"] for e in result["array_entries"]}
            self.assertIn("Temperature", entries)

    def test_everything_else_still_writes_the_flat_npz(self):
        from aiida_pino.utils.runtime_export import derived_member_worker

        with TemporaryDirectory() as d:
            task, member = _task(d, {"time_frames": 3})
            result = derived_member_worker(task)
            self.assertEqual(result["layout"], "npz_file")
            self.assertTrue((Path(d) / "case_s.npz").is_file())

    def test_the_serial_writer_takes_the_same_branch(self):
        import inspect

        from aiida_pino.data import pino_dataset_collection as pdc

        src = inspect.getsource(pdc.PinoDatasetCollectionData._write_runtime_simulation_manifest)
        self.assertIn('profile.get("moving_ref_frame"', src)
        # The mref branch must return BEFORE the flat writer is reached.
        self.assertLess(src.index('moving_ref_frame'), src.index("write_derived_case_npz("))

    def test_the_fold_writes_the_arraydir_manifest_for_that_layout(self):
        import inspect

        from aiida_pino.data import pino_dataset_collection as pdc

        src = inspect.getsource(pdc.PinoDatasetCollectionData._export_members)
        self.assertIn('result.get("layout") == "arraydir"', src)
        self.assertIn('"array_entries": result["array_entries"]', src)


if __name__ == "__main__":
    unittest.main()
