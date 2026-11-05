"""Frame-sliced reads from MATERIALISED (uncompressed) case NPZs.

The lazy store loads a whole case per miss -- with DEFLATE members that is forced,
you cannot inflate a slice. A materialised store (aiida-pino `materialize: true`)
writes members with compress_type=0, where the inner .npy can be memory-mapped and
sliced per frame: ~5 MB per sample instead of ~1 GB per miss. Measured cost of the
whole-case path on a materialised full-res arm: 7.9 s/step, GPU 0 %, two loader
workers at 96 % CPU and bi >500k blocks/s re-reading whole cases the page cache
kept evicting (2026-09-09, run 95650).

These tests pin the eligibility gate and the value-equality of the mapped path.
"""
import unittest, tempfile, zipfile
from pathlib import Path

import numpy as np

from lpno.models.pipeline.npz_lazy_store import _zip_member_memmaps


def _write_case(path, compressed, nt=5, nx=6, ny=4, nz=3):
    rng = np.random.default_rng(0)
    arrays = {
        "Temperature": rng.random((nt, nx, ny, nz), dtype=np.float32),
        "Fraction Of Fluid": rng.random((nt, nx, ny, nz), dtype=np.float32),
        "x": np.linspace(0, 1, nx, dtype=np.float32),
        "float64_field": rng.random((nt, nx, ny, nz)),  # not float32 -> ineligible
    }
    (np.savez_compressed if compressed else np.savez)(path, **arrays)
    return arrays


class ZipMemberMemmaps(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_uncompressed_members_map_and_match_eager_values(self):
        p = self.dir / "case.npz"
        truth = _write_case(p, compressed=False)
        mm = _zip_member_memmaps(p, ("Temperature", "Fraction Of Fluid"))
        self.assertEqual(set(mm), {"Temperature", "Fraction Of Fluid"})
        for name in mm:
            np.testing.assert_array_equal(np.asarray(mm[name]), truth[name])
            self.assertFalse(mm[name].flags.writeable)  # mode='r': a write raises loudly

    def test_deflate_members_are_not_mapped(self):
        p = self.dir / "case.npz"
        _write_case(p, compressed=True)
        self.assertEqual(_zip_member_memmaps(p, ("Temperature",)), {})

    def test_non_float32_member_is_skipped_not_cast(self):
        # A cast would silently materialise the whole array, defeating the point.
        p = self.dir / "case.npz"
        _write_case(p, compressed=False)
        self.assertEqual(_zip_member_memmaps(p, ("float64_field",)), {})

    def test_slicing_a_mapped_member_equals_the_eager_slice(self):
        # The store's stride/crop path: [..., ::2, ::1, :2:1] on the map must equal
        # the same slice of an eagerly loaded array (basic slicing -> view -> pages
        # touched only for the slice).
        p = self.dir / "case.npz"
        truth = _write_case(p, compressed=False)
        mm = _zip_member_memmaps(p, ("Temperature",))
        got = np.asarray(mm["Temperature"][..., ::2, ::1, :2:1])
        np.testing.assert_array_equal(got, truth["Temperature"][..., ::2, ::1, :2:1])

    def test_missing_file_returns_empty_not_raise(self):
        self.assertEqual(_zip_member_memmaps(self.dir / "absent.npz", ("T",)), {})


if __name__ == "__main__":
    unittest.main()
