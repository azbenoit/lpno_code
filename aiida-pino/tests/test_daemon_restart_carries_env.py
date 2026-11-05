"""A daemon restart must not silently drop the daemon's own env-gated settings.

Measured 2026-08-27: two `verdi daemon restart`s from a shell without
`AIIDA_PINO_REMOTE_READ_TIME` disabled stride demotion for the remote computer, and every
submission carrying a stride then failed `validate_inputs` with exit 303. Three runs lost,
and nothing in the failure pointed at an environment variable. Every daemon-side change
needs a restart to deploy, so this recurs by default and not by mistake.
"""

import os
import subprocess
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "bin" / "daemon_restart.sh"
ROOT = SCRIPT.parents[1]

# Exercise the REAL function by extracting it from the script, rather than restating it
# here: a copy in the test would keep passing after the script stopped doing this.
HARNESS = (
    'eval "$(sed -n \'/^carry_daemon_env() {/,/^}/p\' ' + str(SCRIPT) + ')"\n'
    "daemon_pids() { echo %(pid)s; }\n"
    "DAEMON_ENV_FILE=/nonexistent\n"
    "carry_daemon_env\n"
    "%(echoes)s\n"
)


def _carry(pid, names, caller_env):
    echoes = "\n".join('echo "%s=${%s:-unset}"' % (n, n) for n in names)
    return subprocess.run(
        ["bash", "-c", HARNESS % {"pid": pid, "echoes": echoes}],
        capture_output=True, text=True, timeout=60,
        env={"PATH": os.environ.get("PATH", ""), **caller_env},
    )


def _stand_in_daemon(env):
    return subprocess.Popen(["sleep", "60"], env={"PATH": os.environ.get("PATH", ""), **env})


class CarryEnvTest(unittest.TestCase):
    def test_the_script_parses(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(SCRIPT)]).returncode, 0)

    def test_it_carries_every_pino_variable_from_the_running_daemon(self):
        # A named list would drift: the next variable added would be lost exactly the way
        # AIIDA_PINO_REMOTE_READ_TIME was. So the match is by prefix, and this asserts a
        # variable the script has never heard of is carried too.
        proc = _stand_in_daemon({
            "AIIDA_PINO_REMOTE_READ_TIME": "some-computer",
            "PINO_MATERIAL_OVERLAY": "1",
            "AIIDA_PINO_FUTURE_KNOB": "42",
            "UNRELATED_VAR": "no",
        })
        try:
            out = _carry(proc.pid, ["AIIDA_PINO_REMOTE_READ_TIME", "PINO_MATERIAL_OVERLAY",
                                    "AIIDA_PINO_FUTURE_KNOB", "UNRELATED_VAR"], {})
        finally:
            proc.kill(); proc.wait(timeout=10)
        self.assertIn("AIIDA_PINO_REMOTE_READ_TIME=some-computer", out.stdout, out.stderr)
        self.assertIn("PINO_MATERIAL_OVERLAY=1", out.stdout)
        self.assertIn("AIIDA_PINO_FUTURE_KNOB=42", out.stdout)
        # ...but only that namespace, not the whole environment.
        self.assertIn("UNRELATED_VAR=unset", out.stdout)
        self.assertIn("carried forward from the running daemon", out.stdout)

    def test_an_explicit_value_in_the_callers_shell_wins(self):
        # Otherwise a deliberate change could never be made.
        proc = _stand_in_daemon({"AIIDA_PINO_REMOTE_READ_TIME": "old-computer"})
        try:
            out = _carry(proc.pid, ["AIIDA_PINO_REMOTE_READ_TIME"],
                         {"AIIDA_PINO_REMOTE_READ_TIME": "new-computer"})
        finally:
            proc.kill(); proc.wait(timeout=10)
        self.assertIn("AIIDA_PINO_REMOTE_READ_TIME=new-computer", out.stdout, out.stderr)
        self.assertIn("overridden by this shell", out.stdout)

    def test_it_runs_before_the_stop_not_after(self):
        # Once the workers are gone so is the environment they held, so the ordering is the
        # whole mechanism rather than a detail.
        src = SCRIPT.read_text()
        self.assertLess(src.index("\ncarry_daemon_env\n"), src.index("if ! stop_completely; then"))

    def test_a_cold_start_falls_back_to_an_untracked_env_file(self):
        src = SCRIPT.read_text()
        self.assertIn("AIIDA_PINO_DAEMON_ENV", src)
        self.assertIn("daemon.env", src)
        self.assertTrue((ROOT / "daemon.env.example").is_file())

    def test_the_real_env_file_is_never_committed(self):
        # It holds machine-specific values by design.
        self.assertRegex((ROOT / ".gitignore").read_text(), r"(?m)^daemon\.env$")

    def test_the_example_carries_no_machine_specific_value(self):
        for line in (ROOT / "daemon.env.example").read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                self.fail(f"uncommented setting in the example: {line}")


if __name__ == "__main__":
    unittest.main()
