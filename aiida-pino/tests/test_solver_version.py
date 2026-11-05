"""The FLOW-3D build that produced a result is recorded, and never guessed.

Review finding G: three thousand simulations did not record which version of
the commercial solver made them. The banner is in every solve's `runhyd.log`;
these tests pin what is parsed out of it — and, as importantly, what is not:
the install path in the F3D_HOME banner a dozen lines above is machine-specific
and must never reach the database.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from aiida_pino.parsers.common import stamp_runtime_extras
from aiida_pino.scripts.flow3d_run_adapter import parse_solver_version, read_solver_version

# A real runhyd.log head, with the install path replaced by a placeholder --
# the point of the second assertion below is that no such path is ever read.
BANNER = """
**********************************************************************
**        THE ENVIRONMENT VARIABLE "F3D_HOME" IS BEING              **
**        USED TO SET THE PATH TO THE FLOW-3D EXECUTABLES:          **
**        /opt/vendor/FLOW3D_WELD_2099R9u9
**********************************************************************
 Converting prepin.some_case
**** executing    /opt/vendor/FLOW3D_WELD_2099R9u9/double/hydr3d/hydr3d
 Number of ranks  = 1
 program version :   hydr3d   version     12.3.4.20240101  lnx64      WELD      1.0.0
 version id : double
 restart file ...
"""


def test_parses_the_solver_banner_not_the_install_path():
    solver = parse_solver_version(BANNER)
    assert solver["program"] == "hydr3d"
    assert solver["version"] == "12.3.4.20240101"
    assert solver["build"] == "lnx64 WELD 1.0.0"
    assert solver["precision"] == "double"
    # the F3D_HOME path names a version too (2099R9u9); it must not be what we read
    assert "2099R9u9" not in str(solver)
    assert not any("/opt/vendor" in str(v) for v in solver.values())


def test_a_run_without_a_solver_banner_reports_nothing():
    """Prepare-only runs and solves that die before hydr3d get no version --
    an absent version is a fact, a plausible one is a fabrication."""
    assert parse_solver_version("") == {}
    assert parse_solver_version("Converting prepin.foo\nprepin done\n") == {}


def test_read_solver_version_handles_a_missing_log(tmp_path):
    assert read_solver_version(tmp_path) == {}
    (tmp_path / "runhyd.log").write_text(BANNER, encoding="utf-8")
    assert read_solver_version(tmp_path)["version"] == "12.3.4.20240101"


def test_version_is_promoted_to_a_queryable_extra(aiida_profile_clean):
    """The whole point is answering "which solver made this" with a query, so
    the version has to be an extra, not only a key inside the summary Dict."""
    from aiida import orm

    node = orm.Dict(dict={"probe": 1}).store()  # any node with extras will do
    stamp_runtime_extras(
        node,
        {"solver": {"version": "12.3.4.20240101", "build": "lnx64 WELD 1.0.0"}},
    )
    assert node.base.extras.get("solver_version") == "12.3.4.20240101"
    assert node.base.extras.get("solver_build") == "lnx64 WELD 1.0.0"

    # never overwrite: a re-parse or a backfill pass must be safe to repeat
    stamp_runtime_extras(node, {"solver": {"version": "99.9.9"}})
    assert node.base.extras.get("solver_version") == "12.3.4.20240101"
