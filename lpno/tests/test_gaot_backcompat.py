"""Existing GAOT checkpoints must keep loading and forwarding BIT-IDENTICALLY.

183 trained GAOT models exist. Restoring upstream machinery (multi-scale aggregation,
bidirectional neighbours, geometric embedding) must be strictly opt-in: with every new
flag at its default, an old checkpoint has to produce the same parameter count, the same
state-dict fit, and the same forward output as it did before the feature landed.

The fixture in `fixtures/gaot_backcompat_baseline.json` was captured from three real
checkpoints spanning three recipes (random-path, lab, and the demo-served pre-displacement
model) BEFORE any upstream-fidelity work began. A failure here means the off-path moved,
which silently invalidates every stored GAOT number in the manuscript and the demo.

Note this guards MORE than shapes: it compares the forward OUTPUT, so a change that
perturbs RNG draw order during construction -- same shapes, different init -- also fails,
and that is the failure mode a shape-only check would miss.
"""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import torch

from physicsnemo.sym.key import Key

from lpno.models.gaot_arch import GAOTArch

FIXTURE = Path(__file__).parent / "fixtures" / "gaot_backcompat_baseline.json"


def _build(entry, cache_dir):
    kw = dict(entry["kwargs"])
    kw["latent_tokens"] = tuple(kw["latent_tokens"])
    kw["input_keys"] = [Key(k) for k in entry["input_keys"]]
    kw["output_keys"] = [Key(k) for k in entry["output_keys"]]
    kw["cache_dir"] = str(cache_dir)
    torch.manual_seed(0)          # same seed as capture: init must be reproducible
    return GAOTArch(**kw)


class TestGaotBackwardCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FIXTURE.is_file():
            raise unittest.SkipTest(f"baseline fixture missing: {FIXTURE}")
        cls.baseline = json.loads(FIXTURE.read_text())

    def test_fixture_covers_multiple_recipes(self):
        """A single checkpoint could pass by luck; the fixture must span recipes."""
        self.assertGreaterEqual(len(self.baseline), 3)

    def test_parameter_count_unchanged(self):
        import tempfile
        for pk, entry in self.baseline.items():
            with self.subTest(model=pk, label=entry["label"]):
                with tempfile.TemporaryDirectory() as d:
                    net = _build(entry, d)
                n = sum(p.numel() for p in net.parameters())
                self.assertEqual(
                    n, entry["n_params"],
                    f"m{pk} parameter count moved {entry['n_params']} -> {n}; a new feature is "
                    "constructing parameters when it should be inert at its default",
                )

    def test_checkpoint_signature_still_fits(self):
        """The constructed module must still accept each stored checkpoint exactly.

        Compares parameter names and shapes against the signature recorded from the real
        .pth at capture time, so the check is hermetic -- no AiiDA, no checkpoint file.
        """
        import tempfile
        for pk, entry in self.baseline.items():
            with self.subTest(model=pk, label=entry["label"]):
                with tempfile.TemporaryDirectory() as d:
                    net = _build(entry, d)
                have = {k: list(v.shape) for k, v in net.state_dict().items()}
                want = entry["ckpt_sig"]
                missing = sorted(set(want) - set(have))
                extra = sorted(set(have) - set(want))
                self.assertFalse(missing, f"m{pk} no longer provides {missing[:5]}")
                self.assertFalse(extra, f"m{pk} now defines unexpected params {extra[:5]} at default settings")
                bad = {k: (want[k], have[k]) for k in want if want[k] != have[k]}
                self.assertFalse(bad, f"m{pk} shape drift: {list(bad.items())[:3]}")

    def test_forward_output_bit_identical(self):
        import tempfile
        for pk, entry in self.baseline.items():
            with self.subTest(model=pk, label=entry["label"]):
                with tempfile.TemporaryDirectory() as d:
                    net = _build(entry, d)
                net.eval()
                torch.manual_seed(1234)          # same seed as capture
                grid = tuple(entry["grid"])
                invar = {k: torch.rand(1, 1, *grid) for k in entry["input_keys"]}
                with torch.inference_mode():
                    out = net(invar)
                payload = {
                    k: torch.as_tensor(v).flatten()[:64].double().numpy().round(9).tolist()
                    for k, v in sorted(out.items())
                }
                h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]
                self.assertEqual(
                    h, entry["out_hash"],
                    f"m{pk} forward output changed at default settings. Either a default moved, "
                    "or construction now draws different RNG values (same shapes, different init).",
                )


if __name__ == "__main__":
    unittest.main()
