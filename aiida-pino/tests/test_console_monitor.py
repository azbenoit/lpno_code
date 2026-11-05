"""Unit tests for the Monitor tab's pure snapshot parsers (no AiiDA, no GPU)."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aiida_pino.console import monitor

GPU_CSV = """\
0, GPU-aaa, NVIDIA TITAN RTX, 66, 4658, 24576
1, GPU-bbb, NVIDIA TITAN RTX, [N/A], 23678, 24576
"""
APPS_CSV = """\
GPU-aaa, 1298638, 1279
GPU-bbb, 2932545, 23674
GPU-zzz, 999, 1
"""


class ParseGpuCsvTest(unittest.TestCase):
    def test_gpus_and_procs(self):
        gpus = monitor.parse_gpu_csv(GPU_CSV, APPS_CSV)
        self.assertEqual([g["index"] for g in gpus], [0, 1])
        self.assertEqual(gpus[0]["util_pct"], 66)
        self.assertIsNone(gpus[1]["util_pct"])  # [N/A] -> None, not a crash
        self.assertEqual(gpus[0]["mem_total_mib"], 24576)
        self.assertEqual([p["pid"] for p in gpus[1]["procs"]], [2932545])
        self.assertEqual(gpus[1]["procs"][0]["mem_mib"], 23674)
        # unknown gpu uuid in apps output is dropped, not appended blindly
        self.assertEqual(sum(len(g["procs"]) for g in gpus), 2)

    def test_empty_output(self):
        self.assertEqual(monitor.parse_gpu_csv("", ""), [])


class MeminfoTest(unittest.TestCase):
    def test_parse(self):
        text = "MemTotal:       131072000 kB\nMemFree:  1 kB\nMemAvailable: 65536000 kB\n"
        out = monitor.parse_meminfo(text)
        self.assertEqual(out, {"mem_total_gib": 125.0, "mem_available_gib": 62.5})


class StepParseTest(unittest.TestCase):
    def test_last_step(self):
        text = "[11:15:18] - [step:          0] loss:  1.065e+05\n[11:15:30] - [step:        200] loss: 1.1e+04\n"
        self.assertEqual(monitor.last_step(text), 200)
        self.assertIsNone(monitor.last_step("no steps here"))

    def test_waiting_for_gpu(self):
        waiting = "[gpu-seat] waiting for gpu 0 (shared pino spool seat lock)\n[gpu-spool] waiting for the whole card\n"
        self.assertTrue(monitor.waiting_for_gpu(waiting))
        self.assertFalse(monitor.waiting_for_gpu("[step: 100] loss: 1"))
        self.assertFalse(monitor.waiting_for_gpu(""))


class BatchGroupTest(unittest.TestCase):
    def test_batch_key(self):
        self.assertEqual(monitor.batch_key("alsi_ctrl_alsi_c060"), "alsi_ctrl_alsi")
        self.assertEqual(monitor.batch_key("ti64_rk150_case_007"), "ti64_rk150")
        self.assertEqual(monitor.batch_key("no_digits"), "no_digits")
        self.assertEqual(monitor.batch_key("123"), "123")  # never empties the name

    def test_group_batches(self):
        rows = monitor.group_batches(["b_c001", "b_c002", "a_c001"])
        self.assertEqual(rows, [{"batch": "b", "count": 2}, {"batch": "a", "count": 1}])


class SpoolTest(unittest.TestCase):
    def test_task_names_and_gpu_spool(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            q = root / "gpu0" / "queue"
            r = root / "gpu0" / "running"
            q.mkdir(parents=True)
            r.mkdir(parents=True)
            (q / "1.task").write_text(json.dumps({"name": "rigno_a", "cmd": "x"}))
            (q / "2.task").write_text("not json")
            (r / "3.task").write_text(json.dumps({"name": "rigno_running"}))
            snap = monitor.gpu_spool_snapshot(root)
            self.assertEqual(len(snap), 1)
            self.assertEqual(snap[0]["gpu"], "gpu0")
            self.assertEqual(snap[0]["queued_count"], 2)
            self.assertEqual(snap[0]["queued"], ["rigno_a", "2"])
            self.assertEqual(snap[0]["running"], ["rigno_running"])

    def test_missing_root(self):
        self.assertEqual(monitor.gpu_spool_snapshot(pathlib.Path("/nonexistent-gpu-spool")), [])


class AttachAiidaToSpoolTest(unittest.TestCase):
    """A card busy with an AiiDA job has an empty spool queue -- the row must
    not read as idle just because nothing was queued for it."""

    def test_counts_attributed_procs_per_card(self):
        gpus = [
            {"index": 0, "procs": [{"pid": 1, "run_pk": 11}, {"pid": 2, "run_pk": 12}, {"pid": 3}]},
            {"index": 1, "procs": [{"pid": 4, "run_pk": 13}]},
        ]
        rows = [
            {"gpu": "gpu0", "running": [], "queued_count": 3, "queued": []},
            {"gpu": "gpu1", "running": [], "queued_count": 0, "queued": []},
        ]
        out = monitor.attach_aiida_to_spool(rows, gpus)
        self.assertEqual(out[0]["aiida_procs"], 2)  # third proc is not an AiiDA run
        self.assertEqual(out[1]["aiida_procs"], 1)  # busy despite an empty queue

    def test_zero_when_no_aiida_procs_or_no_gpus(self):
        rows = [{"gpu": "gpu0", "running": [], "queued_count": 0, "queued": []}]
        self.assertEqual(monitor.attach_aiida_to_spool(rows, [])[0]["aiida_procs"], 0)
        gpus = [{"index": 0, "procs": [{"pid": 1}]}]
        self.assertEqual(monitor.attach_aiida_to_spool(rows, gpus)[0]["aiida_procs"], 0)


class TailFileTest(unittest.TestCase):
    def test_tail_and_missing(self):
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as fh:
            fh.write("A" * 100 + "\n[step: 42] loss\n")
            path = pathlib.Path(fh.name)
        try:
            self.assertIn("[step: 42]", monitor.tail_file(path, max_bytes=32))
            self.assertNotIn("AAAA", monitor.tail_file(path, max_bytes=20))
        finally:
            path.unlink()
        self.assertEqual(monitor.tail_file(pathlib.Path("/nonexistent.log")), "")


if __name__ == "__main__":
    unittest.main()
