"""Import-time contracts for the models package.

Two contracts, both learned the hard way:

1. Importing ``lpno.models`` must NOT import torch. The train/eval
   entrypoints import the package BEFORE bootstrap_cuda_visible_devices_from_argv
   sets CUDA_VISIBLE_DEVICES; a torch/physicsnemo import at package level defeats
   the bootstrap and sends the run to the wrong GPU (rigno_rp5/rp6 OOM'd on GPU0
   with custom.cuda_device=1 when physicsnemo_ext was imported here).

2. Importing ``ns_energy_pipeline`` (loaded AFTER the bootstrap) must register
   the custom ``physicsnemo_ext`` AMP-scaler op. physicsnemo ships the kernel
   .so but never imports it, so without our registration import fp16 training
   crashes at the first scaler.update(). The two contracts pull in
   opposite directions — this pins that the import lives in the right module.
"""

import subprocess
import sys
import unittest


class ModelsPackageImportTorchFreeTest(unittest.TestCase):
    def test_import_does_not_pull_torch(self):
        code = (
            "import sys; import lpno.models; "
            "bad = [m for m in ('torch', 'physicsnemo') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(
            proc.returncode, 0,
            f"lpno.models import pulled in torch/physicsnemo\n{proc.stderr[-500:]}",
        )

    def test_pipeline_import_registers_amp_scaler_op(self):
        # Exit codes: 0 registered, 2 skip (no torch/physicsnemo_ext in env),
        # 1 op missing after import (the regression).
        code = (
            "import sys\n"
            "try:\n"
            "    import torch\n"
            "    import lpno.models.ns_energy_pipeline  # noqa: F401\n"
            "except Exception:\n"
            "    sys.exit(2)\n"
            "try:\n"
            "    import physicsnemo.sym.physicsnemo_ext  # noqa: F401\n"
            "except Exception:\n"
            "    sys.exit(2)\n"
            "sys.exit(0 if hasattr(torch.ops.physicsnemo_ext, '_amp_update_scale_') else 1)"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if proc.returncode == 2:
            self.skipTest("torch/physicsnemo_ext not importable in this environment")
        self.assertEqual(
            proc.returncode, 0,
            "importing ns_energy_pipeline did not register "
            f"physicsnemo_ext._amp_update_scale_ (regression)\n{proc.stderr[-500:]}",
        )


if __name__ == "__main__":
    unittest.main()
