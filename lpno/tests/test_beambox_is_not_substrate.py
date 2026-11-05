"""The validation beam box must be a beam box, not the substrate.

Regression for 2026-08-26, the THIRD instance of one bug. `beam_dist` is a peak-1
FOOTPRINT, not a distance, so any predicate of the form `abs(beam_dist) <= n_radii`
with n_radii >= 1 selects every voxel. ANDed with `z <= 0` the "beam box" is exactly
the substrate, and `beambox_mae_*` becomes a substrate MAE under a beam-box name.

Measured on real data (30647834, 2026-08-26): beam_dist range [0.000000, 0.995010],
`abs(beam_dist) <= 2.0` true for 100.000% of voxels.

Signature, five models on the straight-path family -- val beambox lands on eval
SUBSTRATE, not on eval beambox:

    model         val beambox   eval substrate  ratio   eval beambox
    69088 FNO     0.00674       0.00571         1.18    0.02125
    66159 FNO     0.00797       0.00642         1.24    0.02038
    68029 GAOT    0.01897       0.01810         1.05    0.07853
    69770 RIGNO   0.03282       0.03022         1.09    0.09316

Prior instances, both recorded in `validation_beambox.py`'s docstring: 2026-08-10
(8x) and 2026-08-15 (9.6x). Both were fixed by correcting a second copy of the box
predicate in place, which is why a third appeared. These tests pin the behaviour AND
the absence of a second definition.
"""

import re
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src" / "lpno" / "models" / "pipeline"


def _footprint_and_z(nx=32, ny=32, nz=16, sigma_cells=4.0):
    """A realistic peak-1 Gaussian footprint broadcast over z, and a z grid <= 0."""
    yy, xx = np.meshgrid(np.arange(ny) - ny / 2, np.arange(nx) - nx / 2, indexing="ij")
    foot = np.exp(-(xx**2 + yy**2) / (2 * sigma_cells**2))          # peak 1, in [0, 1]
    beam = np.broadcast_to(foot[..., None], (ny, nx, nz)).copy()
    z = np.broadcast_to(np.linspace(-1.0, 0.0, nz), (ny, nx, nz)).copy()
    return beam, z


class TestBeamBoxIsNotSubstrate(unittest.TestCase):
    def test_shared_builder_excludes_most_of_the_substrate(self):
        """The real box must be a strict, small subset of the metal."""
        from lpno.models.pipeline.postprocessing import beam_box_mask

        beam, z = _footprint_and_z()
        box = beam_box_mask(beam, z, n_radii=2.0)
        self.assertIsNotNone(box, "builder returned no box for a valid footprint")
        substrate = (z <= 0).sum()
        self.assertLess(
            box.sum(), substrate,
            "the beam box must exclude quiescent metal; equal to the substrate is the bug",
        )
        self.assertLess(
            box.sum(), 0.5 * substrate,
            f"beam box covers {box.sum()}/{substrate} of the metal -- that is dilution, "
            "not a beam box",
        )

    def test_the_torch_entry_point_returns_a_box_inside_the_metal(self):
        """`model_domain` calls `_beam_box_from_stash`, not `beam_box_mask` directly.

        Folded in from a duplicate file (mine) rather than kept beside this one --
        two test files for one predicate is the shape that produced the bug. Two
        properties this adds: the box is CONTAINED in the metal (a count comparison
        alone does not show that), and the exercised function is the torch entry
        point the validator actually reaches, one layer above the numpy primitive.
        """
        import torch

        from lpno.models.pipeline.validation_beambox import (
            _beam_box_from_stash,
        )

        beam, z = _footprint_and_z()
        box = _beam_box_from_stash(torch.as_tensor(beam), torch.as_tensor(z), n_radii=2.0)
        self.assertIsNotNone(box, "the torch entry point must build a box too")
        substrate = torch.as_tensor(z) <= 0
        self.assertEqual(
            int((box & ~substrate).sum()), 0,
            "the box must lie inside the metal, not merely be smaller than it",
        )

    def test_no_second_hand_rolled_box_definition_in_the_pipeline(self):
        """There must be exactly ONE box definition: the shared builder.

        Instances two and three both came from a second copy being corrected in
        place rather than deleted, so absence of the copy is the invariant.

        Strings and comments are blanked before matching, because
        `validation_beambox.py`'s docstring quotes the buggy predicate on purpose
        as the record of the earlier instances -- matching prose would make this
        test fire on the documentation of the very bug it guards.
        """
        import io
        import tokenize

        def code_only(text: str) -> str:
            """`text` with STRING and COMMENT tokens blanked, line numbers preserved."""
            lines = text.splitlines(keepends=True)
            try:
                toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
            except (tokenize.TokenError, IndentationError):
                return text
            for tok in toks:
                if tok.type == tokenize.STRING:
                    # Blank DOCSTRINGS only. An ordinary string literal is often the
                    # channel name itself (`invar["beam_dist"]`), and blanking those
                    # would erase the signal this test looks for.
                    before = lines[tok.start[0] - 1][: tok.start[1]]
                    if before.strip():
                        continue
                elif tok.type != tokenize.COMMENT:
                    continue
                (r0, c0), (r1, c1) = tok.start, tok.end
                for r in range(r0, r1 + 1):
                    line = lines[r - 1]
                    a = c0 if r == r0 else 0
                    b = c1 if r == r1 else len(line.rstrip("\n"))
                    lines[r - 1] = line[:a] + " " * (b - a) + line[b:]
            return "".join(lines)

        offenders = []
        pattern = re.compile(r"beam_dist.*?\.abs\(\)\s*<=|abs\(\s*beam_dist\s*\)\s*<=")
        for path in sorted(SRC.glob("*.py")):
            for i, line in enumerate(code_only(path.read_text()).splitlines(), start=1):
                if pattern.search(line):
                    offenders.append(f"{path.name}:{i}")
        self.assertEqual(
            [], offenders,
            "hand-rolled beam-box predicate found; route through "
            "postprocessing.beam_box_from_input instead:\n  " + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
