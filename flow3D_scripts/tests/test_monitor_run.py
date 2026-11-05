"""Tests for the FLOW-3D stall/divergence monitor (python/monitor_run.py)."""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import monitor_run as mon  # noqa: E402

# Real-shaped runhyd.log rows (sim_time cycle delt dt_stbl iter res vol %loss ...).
HEALTHY_LOG = """\
         progress                time step            pressure
   sim_time       cycle     delt   dt_stbl/code    iter  res/epsi     volume      %loss
 4.11615E-04       1423   2.75E-07  2.75E-07/st       2  2.82E-14   4.5000E-10  +4.57E-08  0.750
 4.29448E-04       1488   2.75E-07  2.75E-07/st       2  5.88E-15   4.5000E-10  +4.43E-08  0.750
 4.47320E-04       1553   2.73E-07  2.73E-07/st       2  8.06E-15   4.5000E-10  +4.60E-08  0.750
"""
STUCK_ROW = " 4.47320E-04  1553  2.73E-07  2.73E-07/st  2  1E-14  4.5E-10  +4.6E-08  0.75\n"


class ParseTests(unittest.TestCase):
    def test_parses_only_numeric_rows(self):
        rows = mon.parse_progress(HEALTHY_LOG)
        self.assertEqual(len(rows), 3)  # two header lines skipped
        self.assertAlmostEqual(rows[-1].sim_time, 4.47320e-04)
        self.assertAlmostEqual(rows[-1].delt, 2.73e-07)
        self.assertEqual(rows[-1].cycle, 1553)

    def test_empty_and_banner_only(self):
        self.assertEqual(mon.parse_progress(""), [])
        self.assertEqual(mon.parse_progress("---- ----\nsim_time cycle delt"), [])


class NanInfTests(unittest.TestCase):
    def test_flags_nan_inf_overflow(self):
        self.assertTrue(mon.has_naninf("  4.4E-04  1553  NaN  x"))
        self.assertTrue(mon.has_naninf("  vol  Infinity  x"))
        self.assertTrue(mon.has_naninf("  4.4E-04  ********  x"))

    def test_ignores_healthy_log(self):
        self.assertFalse(mon.has_naninf(HEALTHY_LOG))

    def test_ignores_flow3d_banner_asterisks(self):
        # Regression: every runhyd.log opens with a '*'-banner ('****' rows and
        # '**** executing ...'); has_naninf scanned it globally and SIGKILLed every
        # healthy keyhole run on a false overflow. The banner must not count.
        banner = ("*" * 70 + "\n") * 2 + "**** \n**** executing /opt/flow3d/hydr3d\n**** \n"
        self.assertFalse(mon.has_naninf(banner + HEALTHY_LOG))
        # a real '****' field overflow inside a data row is still caught
        self.assertTrue(mon.has_naninf(banner + " 4.4E-04  1553  ****  x\n"))


class DiagnoseTests(unittest.TestCase):
    def setUp(self):
        self.rows = mon.parse_progress(HEALTHY_LOG)

    def test_healthy_no_kill(self):
        kill, _ = mon.diagnose(self.rows, 30, stall_s=900, delt_min=1e-9, min_cycle=100)
        self.assertFalse(kill)

    def test_stall_kills(self):
        kill, reason = mon.diagnose(self.rows, 901, stall_s=900, delt_min=1e-9, min_cycle=100)
        self.assertTrue(kill)
        self.assertIn("stall", reason)

    def test_delt_collapse_kills_past_min_cycle(self):
        collapsed = mon.parse_progress(
            " 4.47320E-04  1553  8.0E-11  8.0E-11/st  2  1E-14  4.5E-10  +4.6E-08  0.75")
        kill, reason = mon.diagnose(collapsed, 10, stall_s=900, delt_min=1e-9, min_cycle=100)
        self.assertTrue(kill)
        self.assertIn("divergence", reason)

    def test_small_delt_ignored_during_startup(self):
        # Same tiny delt but cycle < min_cycle => startup ramp, not divergence.
        startup = mon.parse_progress(
            " 1.0E-06  12  8.0E-11  8.0E-11/st  2  1E-14  4.5E-10  +1.0E-09  0.10")
        kill, _ = mon.diagnose(startup, 10, stall_s=900, delt_min=1e-9, min_cycle=100)
        self.assertFalse(kill)

    def test_no_rows_no_kill(self):
        kill, _ = mon.diagnose([], 99999, stall_s=900, delt_min=1e-9, min_cycle=100)
        self.assertFalse(kill)


class WatchTests(unittest.TestCase):
    def test_stall_triggers_kill(self):
        """A static (stalled) log drives watch() to abort once wall > stall_s."""
        tf = tempfile.NamedTemporaryFile("w", delete=False, suffix=".log")
        tf.write(STUCK_ROW)
        tf.close()
        self.addCleanup(os.unlink, tf.name)
        clock_t = {"now": 0.0}
        killed = {"n": 0}
        rc = mon.watch(
            tf.name, stall_s=900, delt_min=1e-9, poll_s=60, min_progress=1e-6,
            min_cycle=100, until_pid=None,
            clock=lambda: clock_t["now"],
            sleep=lambda dt: clock_t.__setitem__("now", clock_t["now"] + dt),
            killer=lambda *a: killed.__setitem__("n", killed["n"] + 1),
        )
        self.assertEqual(rc, 1)
        self.assertEqual(killed["n"], 1)

    def test_max_solve_s_counts_from_first_output_not_from_start(self):
        """The solve wall bound must exclude the seat wait.

        The log is absent for the first 600 s (queued behind another run), then
        appears; with max_solve_s=300 the abort must land ~900 s in, not at 300 —
        charging the queue to the solve is exactly what kept wall timeouts off.
        """
        log_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(log_dir, ignore_errors=True))
        logpath = os.path.join(log_dir, "runhyd.log")  # not created yet
        clock_t = {"now": 0.0}
        killed_at = {}

        def fake_sleep(dt):
            clock_t["now"] += dt
            if clock_t["now"] >= 600 and not os.path.exists(logpath):
                with open(logpath, "w") as fh:
                    fh.write(HEALTHY_LOG)

        rc = mon.watch(
            logpath, stall_s=10**9, delt_min=1e-9, poll_s=60, min_progress=1e-6,
            min_cycle=100, until_pid=None,
            clock=lambda: clock_t["now"], sleep=fake_sleep,
            killer=lambda *a: killed_at.setdefault("t", clock_t["now"]),
            max_solve_s=300,
        )
        self.assertEqual(rc, 1)
        self.assertGreaterEqual(killed_at["t"], 900,
                                "seat wait was charged to the solve clock")

    def test_exits_when_solve_pid_gone(self):
        """until_pid death makes watch() return 0 without aborting."""
        tf = tempfile.NamedTemporaryFile("w", delete=False, suffix=".log")
        tf.write(HEALTHY_LOG)
        tf.close()
        self.addCleanup(os.unlink, tf.name)
        p = subprocess.Popen(["true"])
        p.wait()  # reaped => pid no longer a live process
        rc = mon.watch(
            tf.name, stall_s=900, delt_min=1e-9, poll_s=60, min_progress=1e-6,
            min_cycle=100, until_pid=p.pid,
            clock=lambda: 0.0, sleep=lambda dt: None, killer=lambda *a: None,
        )
        self.assertEqual(rc, 0)

    def test_alive_detects_self(self):
        self.assertTrue(mon._alive(os.getpid()))

    def test_kill_solver_spares_unrelated_process(self):
        """A stale/reused until_pid that is not a solve (no SOLVE_MARK) kills nothing."""
        q = subprocess.Popen(["sleep", "30"])
        self.addCleanup(lambda: (q.kill(), q.wait()))
        mon.kill_solver(q.pid)  # sleep's cmdline lacks SOLVE_MARK
        time.sleep(0.2)
        self.assertIsNone(q.poll(), "kill_solver killed an unrelated process")

    def test_kill_solver_kills_marked_solve_and_spares_others(self):
        """kill_solver kills the marked solve, and nothing outside its subtree."""
        solve = subprocess.Popen(
            ["bash", "-c", "exec -a %s_probe sleep 30" % mon.SOLVE_MARK])
        other = subprocess.Popen(["sleep", "30"])
        self.addCleanup(
            lambda: [(x.kill(), x.wait()) for x in (solve, other) if x.poll() is None])
        time.sleep(0.1)
        mon.kill_solver(solve.pid)
        time.sleep(0.3)
        self.assertIsNotNone(solve.poll(), "marked solve was not killed")
        self.assertIsNone(other.poll(), "an unrelated process was wrongly killed")

    def test_descendants_includes_children(self):
        parent = subprocess.Popen(["bash", "-c", "sleep 30 & sleep 30 & wait"])
        self.addCleanup(lambda: (parent.kill(), parent.wait()))
        time.sleep(0.2)
        tree = mon._descendants(parent.pid)
        self.assertIn(parent.pid, tree)
        self.assertGreaterEqual(len(tree), 3, "child processes missing from subtree")

    def test_watch_exits_on_pid_reuse_without_firing(self):
        """A reused until_pid (starttime changes while it still 'exists') makes watch
        return 0 without firing -- the stale-monitor guard."""
        calls = {"n": 0}
        starts = iter(["100", "999"])  # baseline, then a different instance
        orig_st, orig_al = mon._proc_starttime, mon._alive
        mon._proc_starttime = lambda pid: next(starts, "999")
        mon._alive = lambda pid: True
        try:
            rc = mon.watch(
                "/nonexistent", stall_s=1.0, delt_min=1e-9, poll_s=0.0,
                min_progress=1e-6, min_cycle=100, until_pid=999999,
                clock=lambda: 0.0, sleep=lambda dt: None,
                killer=lambda *a: calls.__setitem__("n", calls["n"] + 1),
            )
        finally:
            mon._proc_starttime, mon._alive = orig_st, orig_al
        self.assertEqual(rc, 0)
        self.assertEqual(calls["n"], 0, "watch fired after PID reuse")


    def test_watch_reanchors_on_sim_time_regression(self):
        """A requeued case leaves the old attempt's hd3msg in the case dir; the
        monitor's first poll anchors best_sim on the stale max, and the fresh run
        (restarting near sim_time 0) can then never 'progress' within stall_s ->
        false ABORT at exactly wall stall_s (t64k_045, 2026-07-17). A sim_time
        REGRESSION in the watched file must re-anchor the baseline instead."""
        import os, tempfile
        def row(sim):
            return (f" {sim:.5E}       1553   2.73E-07  2.73E-07/st       2"
                    f"  1.00E-14   4.5000E-10  +4.60E-08  0.750\n")
        fd, path = tempfile.mkstemp()
        os.close(fd)
        state = {"clock": 0.0, "poll": 0, "killed": 0}
        def clock():
            state["clock"] += 100.0
            return state["clock"]
        def fake_sleep(dt):
            state["poll"] += 1
            if state["poll"] == 1:      # new run truncates the file, restarts low
                Path(path).write_text(row(1.0e-5))
            else:                        # and advances steadily every poll
                Path(path).write_text(row(1.0e-5 * (state["poll"])))
        alive_left = {"n": 6}
        orig_st, orig_al = mon._proc_starttime, mon._alive
        mon._proc_starttime = lambda pid: "100"
        def alive(pid):
            alive_left["n"] -= 1
            return alive_left["n"] > 0
        mon._alive = alive
        try:
            Path(path).write_text(row(1.059e-3))   # stale hd3msg from the old attempt
            rc = mon.watch(
                path, stall_s=250.0, delt_min=1e-9, poll_s=0.0,
                min_progress=1e-6, min_cycle=100, until_pid=999999,
                clock=clock, sleep=fake_sleep,
                killer=lambda *a: state.__setitem__("killed", state["killed"] + 1),
            )
        finally:
            mon._proc_starttime, mon._alive = orig_st, orig_al
            os.unlink(path)
        self.assertEqual(state["killed"], 0,
                         "monitor false-killed a fresh run behind a stale baseline")
        self.assertEqual(rc, 0)

    def test_watch_stops_once_the_solver_reports_completion(self):
        """After `normal completion` the solver stops writing progress rows, but the
        runner is still alive exporting the NPZ -- so `until_pid` does not die and the
        stall clock keeps running against a sim_time that will never move again.

        That false-killed t64v_019 (2026-07-21) at frame 301/301 of its export, after
        a solve that had finished cleanly: the export simply took longer than stall_s
        under solver-lock contention. t64v_074 and t64v_075 survived the same code path only
        because their exports took 207 s and 159 s against a 900 s stall window.
        The completion marker is already in hd3msg, so the monitor must honour it.
        """
        import os, tempfile
        done = (" 3.00000E-03       9999   2.73E-07  2.73E-07/st       2"
                "  1.00E-14   4.5000E-10  +4.60E-08  0.750\n"
                "               cpu time =    3.52877E+04 seconds\n"
                "           date of completion = 07/21/2026\n"
                "           time               = 16:03:45\n")
        fd, path = tempfile.mkstemp()
        os.close(fd)
        Path(path).write_text(done)
        state = {"clock": 0.0, "killed": 0}

        def clock():  # every poll jumps far past stall_s
            state["clock"] += 5000.0
            return state["clock"]

        orig_st, orig_al = mon._proc_starttime, mon._alive
        mon._proc_starttime = lambda pid: "100"
        mon._alive = lambda pid: True  # runner still alive: it is exporting
        try:
            rc = mon.watch(
                path, stall_s=900.0, delt_min=1e-12, poll_s=0.0,
                min_progress=1e-9, min_cycle=100, until_pid=999999,
                clock=clock, sleep=lambda dt: None,
                killer=lambda *a: state.__setitem__("killed", state["killed"] + 1),
            )
        finally:
            mon._proc_starttime, mon._alive = orig_st, orig_al
            os.unlink(path)

        self.assertEqual(state["killed"], 0, "monitor killed a finished solve mid-export")
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
