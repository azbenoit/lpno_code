#!/usr/bin/env python3
"""Short, no-license tests for the rectangular FLOW-3D workflow.

Full solver-backed validation lives in tests/full_solver_workflow.py and
requires F3D_LICENSE to be configured locally.
"""

from __future__ import annotations

import contextlib
import io
import re
import json
import shutil
import subprocess
import struct
import tempfile
import unittest
import unittest.mock
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "python" / "run_rectangular_case.py"
SMOKE_CONFIG = ROOT / "configs" / "rectangular_moving_laser_smoke.json"
MATERIALS_DIR = ROOT / "materials"
VTK_VALIDATOR = ROOT / "python" / "validate_vtk_series.py"
VTK_COMPARATOR = ROOT / "python" / "compare_legacy_vtk_dirs.py"
REPRO_CONFIG = ROOT / "configs" / "case_fine_1_10um_reproduction.json"

import sys

sys.path.insert(0, str(ROOT / "python"))
import run_rectangular_case as rectangular  # noqa: E402


class RectangularWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        # The floors are tested in their ENFORCING form in either copy: the public
        # assembly rewrites `resource_guards.DEFAULT_MODE` to "warn", under which a
        # floor is one stderr line and no wait (2026-09-15; 4 of these tests failed
        # in the assembled tree). Tests of warn mode set the variable themselves.
        patcher = unittest.mock.patch.dict("os.environ", {"F3D_RESOURCE_GUARDS": "enforce"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_smoke_config_validation_derives_expected_geometry(self) -> None:
        case_name, replacements, manifest = rectangular.load_and_validate(SMOKE_CONFIG, "unit_smoke")

        self.assertEqual(case_name, "unit_smoke")
        self.assertEqual(manifest["units"]["length"], "m")
        self.assertEqual(manifest["derived"]["beam_focus_start_m"], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(manifest["derived"]["scan_path_length_m"], 0.0008)
        self.assertAlmostEqual(manifest["derived"]["scan_time_s"], 0.008)
        self.assertEqual(manifest["material"]["name"], "Titanium Alloy Ti Al V IMI 318 Mills data")
        self.assertEqual(manifest["parallel"]["cores"], 1)
        self.assertEqual(manifest["parallel"]["ranks"], 1)
        self.assertEqual(replacements["__PROJECT_NAME__"], "unit_smoke")
        self.assertIn("__X_MIN_M__", replacements)
        self.assertEqual(replacements["__MATERIAL_LABEL__"], "Titanium Alloy Ti Al V IMI 318 Mills data")

    def test_render_writes_manifest_and_resolves_template_tokens(self) -> None:
        case_name, replacements, manifest = rectangular.load_and_validate(SMOKE_CONFIG, "unit_render", cores_override=4)
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            input_file = rectangular.render(case_dir, case_name, replacements, manifest)
            text = input_file.read_text(encoding="utf-8")
            written_manifest = json.loads((case_dir / "case_manifest.json").read_text(encoding="utf-8"))
            machines = (case_dir / "machines").read_text(encoding="utf-8").splitlines()

        self.assertIn("project='unit_render'", text)
        self.assertIn("fluid1='Titanium Alloy Ti Al V IMI 318 Mills data'", text)
        self.assertNotRegex(text, r"__[A-Z0-9_]+__")
        self.assertEqual(written_manifest["case_name"], "unit_render")
        self.assertEqual(written_manifest["parallel"]["cores"], 4)
        self.assertEqual(written_manifest["parallel"]["ranks"], 4)
        self.assertEqual(len(machines), 4)
        self.assertFalse("license" in json.dumps(written_manifest).lower())

    def test_output_frames_renders_selected_data_interval(self) -> None:
        # Omitted: no tapltd line and null manifest field (default ~100 frames).
        _, replacements, manifest = rectangular.load_and_validate(SMOKE_CONFIG, "unit_noframes")
        self.assertEqual(replacements["__SELECTED_DATA_INTERVAL_LINE__"], "")
        self.assertIsNone(manifest["derived"]["output_frames"])

        # Set (standard profile): tapltd(1) = finish_time / output_frames.
        raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
        raw.setdefault("simulation", {})["output_frames"] = 400
        raw["simulation"]["solver_profile"] = "standard"
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "frames.json"
            cfg.write_text(json.dumps(raw), encoding="utf-8")
            _, replacements, manifest = rectangular.load_and_validate(cfg, "unit_frames")
        finish = manifest["input"]["simulation"]["finish_time_s"]
        self.assertEqual(manifest["derived"]["output_frames"], 400)
        self.assertAlmostEqual(manifest["derived"]["selected_data_interval_s"], finish / 400)
        self.assertIn(f"tapltd(1)={rectangular.fmt(finish / 400)}", replacements["__SELECTED_DATA_INTERVAL_LINE__"])
        # Exactly one tapltd(1) in the rendered prepin (no duplicate vs a profile line).
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            text = rectangular.render(case_dir, "unit_frames", replacements, manifest).read_text(encoding="utf-8")
        self.assertEqual(text.count("tapltd(1)="), 1)

        # Non-positive frame counts are rejected.
        raw["simulation"]["output_frames"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "bad.json"
            cfg.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(rectangular.ConfigError):
                rectangular.load_and_validate(cfg, "unit_bad")

        # output_frames overrides the keyhole/legacy profiles' fixed tapltd(1)=5e-06
        # (needed for batches that choose their own frame interval).
        raw["simulation"]["output_frames"] = 123
        raw["simulation"]["solver_profile"] = "keyhole"
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "keyhole_frames.json"
            cfg.write_text(json.dumps(raw), encoding="utf-8")
            _, replacements, manifest = rectangular.load_and_validate(cfg, "unit_keyhole_frames")
        finish = manifest["input"]["simulation"]["finish_time_s"]
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            text = rectangular.render(case_dir, "unit_keyhole_frames", replacements, manifest).read_text(encoding="utf-8")
        self.assertEqual(text.count("tapltd(1)="), 1)
        self.assertIn(f"tapltd(1)={rectangular.fmt(finish / 123)}", text)
        self.assertNotIn("tapltd(1)=5e-06", text)

        # Without output_frames the keyhole profile keeps its fixed 5e-06 interval.
        del raw["simulation"]["output_frames"]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "keyhole_default.json"
            cfg.write_text(json.dumps(raw), encoding="utf-8")
            _, replacements, manifest = rectangular.load_and_validate(cfg, "unit_keyhole_default")
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            text = rectangular.render(case_dir, "unit_keyhole_default", replacements, manifest).read_text(encoding="utf-8")
        self.assertEqual(text.count("tapltd(1)="), 1)
        self.assertIn("tapltd(1)=5e-06", text)

    def test_case_fine_reproduction_config_renders_legacy_style_case(self) -> None:
        case_name, replacements, manifest = rectangular.load_and_validate(REPRO_CONFIG, "unit_case_fine_repro")
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            input_file = rectangular.render(case_dir, case_name, replacements, manifest)
            text = input_file.read_text(encoding="utf-8")

        self.assertEqual(manifest["derived"]["initialization"], "legacy_halfspace")
        self.assertEqual(manifest["derived"]["solver_profile"], "legacy_case_fine_1")
        self.assertNotIn("bounds_m", manifest["input"]["geometry"])
        self.assertEqual(manifest["derived"]["geometry_placement"], "centered")
        self.assertEqual(
            manifest["derived"]["geometry_dimensions_m"],
            {"length": 0.0009, "width": 0.0004, "thickness": 0.0003, "gas_clearance": 0.0001},
        )
        self.assertEqual(manifest["derived"]["mesh_bounds_m"]["x"], [-0.00045, 0.00045])
        self.assertEqual(manifest["derived"]["mesh_bounds_m"]["y"], [-0.0002, 0.0002])
        self.assertEqual(manifest["derived"]["mesh_bounds_m"]["z"], [-0.0003, 0.0001])
        self.assertAlmostEqual(manifest["derived"]["nominal_mesh_cells"]["x"], 90.0)
        self.assertAlmostEqual(manifest["derived"]["nominal_mesh_cells"]["y"], 40.0)
        self.assertAlmostEqual(manifest["derived"]["nominal_mesh_cells"]["z"], 40.0)
        self.assertAlmostEqual(manifest["parallel"]["cores"], 16)
        self.assertAlmostEqual(manifest["parallel"]["threads_per_rank"], 16)
        self.assertIn("size=1e-05", text)
        self.assertIn("subprocess=1", text)
        self.assertIn("ibct(1)=2", text)
        self.assertIn("ibct(6)=5", text)
        self.assertIn("fbct(1,6)=0", text)
        self.assertNotIn("\n    tbct(1,6)", text)
        self.assertIn("nfls=1", text)
        self.assertIn("fluidRegionName(1)='Region 1'", text)
        self.assertIn("fzh(1)=0", text)
        self.assertIn("hflv1=50", text)
        self.assertIn("t_reference=273.15", text)
        self.assertIn("ilbmhtb(1)=2", text)
        self.assertIn("irefopt=3", text)
        self.assertIn("xwidth0(1)=0.0002", text)
        self.assertIn("powlbm(1,1)=70", text)
        self.assertNotIn("powlbm(2,1)", text)
        self.assertIn("timlbm(1)=0.0", text)
        self.assertNotIn("timlbm(2)", text)
        self.assertIn("anmtyp(1)='f'", text)
        self.assertIn("anmtyp(10)='vel'", text)
        self.assertEqual(manifest["material"]["file"], "materials/ti_imi318_mills/ti_imi318_mills.json")

    def test_centered_geometry_can_use_documented_default_dimensions(self) -> None:
        raw = json.loads(REPRO_CONFIG.read_text(encoding="utf-8"))
        raw["geometry"] = {"placement": "centered"}
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "default_geometry.json"
            config.write_text(json.dumps(raw), encoding="utf-8")
            _, _, manifest = rectangular.load_and_validate(config, "unit_default_geometry")

        # The runner must use the documented defaults from the single schema
        # source, so code and docs cannot drift on the magnitudes.
        import config_schema
        d = config_schema.GEOMETRY_DEFAULTS
        self.assertEqual(
            manifest["derived"]["geometry_dimensions_m"],
            {"length": d["length_m"], "width": d["width_m"],
             "thickness": d["thickness_m"], "gas_clearance": d["gas_clearance_m"]},
        )
        self.assertEqual(manifest["derived"]["mesh_bounds_m"]["x"], [-0.00045, 0.00045])
        self.assertEqual(manifest["derived"]["mesh_bounds_m"]["y"], [-0.0002, 0.0002])
        self.assertEqual(manifest["derived"]["mesh_bounds_m"]["z"], [-0.0003, 0.0001])

    def test_material_can_be_selected_from_material_folder(self) -> None:
        raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
        raw["material"] = {"name": "ss316l"}
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "case.json"
            config.write_text(json.dumps(raw), encoding="utf-8")
            _, replacements, manifest = rectangular.load_and_validate(config, "unit_ss316l")

        self.assertEqual(manifest["material"]["name"], "Stainless steel 316L")
        self.assertEqual(manifest["material"]["file"], "materials/ss316l/ss316l.json")
        self.assertIn("#mu1(1)", replacements["__MATERIAL_TABLES__"])
        self.assertIn("#rhof(1)", replacements["__MATERIAL_TABLES__"])

    def test_material_name_lookup_is_case_insensitive(self) -> None:
        raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
        raw["material"] = "TI_IMI318_MILLS"
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "case.json"
            config.write_text(json.dumps(raw), encoding="utf-8")
            _, _, manifest = rectangular.load_and_validate(config, "unit_lowercase_material")

        self.assertEqual(manifest["material"]["file"], "materials/ti_imi318_mills/ti_imi318_mills.json")

    def test_parallel_threads_must_divide_allocated_cores(self) -> None:
        raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
        raw["parallel"] = {"cores": 3, "threads_per_rank": 2}
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "bad_parallel.json"
            config.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(rectangular.ConfigError, "divisible"):
                rectangular.load_and_validate(config, None)

    def test_rerender_warns_only_when_parameters_change(self) -> None:
        # Re-rendering the same case_name (prepare-only / failed run) is silent
        # when parameters match, but warns before overwriting changed inputs.
        import contextlib
        import io

        case_name, replacements, manifest = rectangular.load_and_validate(SMOKE_CONFIG, "unit_rerender")
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / case_name
            rectangular.render(case_dir, case_name, replacements, manifest)

            quiet = io.StringIO()
            with contextlib.redirect_stderr(quiet):
                rectangular.render(case_dir, case_name, replacements, manifest)
            self.assertEqual(quiet.getvalue(), "")

            changed = json.loads(json.dumps(manifest))
            changed["input"]["laser"]["power_w"] += 50.0
            loud = io.StringIO()
            with contextlib.redirect_stderr(loud):
                rectangular.render(case_dir, case_name, replacements, changed)
            self.assertIn("overwriting", loud.getvalue())

    def test_bounds_with_all_positive_z_gives_clear_error(self) -> None:
        # An all-positive bounds_m z range derives thickness 0; the message must
        # name the real cause (no sub-surface substrate), not "greater than zero".
        raw = json.loads(REPRO_CONFIG.read_text(encoding="utf-8"))
        raw["geometry"] = {
            "bounds_m": {"x": [-0.00045, 0.00045], "y": [-0.0002, 0.0002], "z": [0.0001, 0.0003]},
            "placement": "centered",
        }
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "positive_z.json"
            config.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(rectangular.ConfigError, "below z=0"):
                rectangular.load_and_validate(config, None)

    def test_local_license_env_parser_accepts_ignored_license_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "license.env"
            env_file.write_text(
                "# local only\n"
                "export F3D_LICENSE='LOCAL_TEST_ENDPOINT'\n",
                encoding="utf-8",
            )

            endpoint = rectangular.local_license_endpoint(env_file)

        self.assertEqual(endpoint, "LOCAL_TEST_ENDPOINT")

    def test_invalid_config_failure_modes_are_reported_before_solver_use(self) -> None:
        raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
        raw["mesh"]["cell_size_m"] = raw["laser"]["beam_radius_m"]
        with tempfile.TemporaryDirectory() as tmp:
            bad_config = Path(tmp) / "bad.json"
            bad_config.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(rectangular.ConfigError, "half the beam radius"):
                rectangular.load_and_validate(bad_config, None)

    def test_prepare_only_cli_needs_no_license_and_writes_expected_files(self) -> None:
        case_name = "unit_prepare_only"
        case_dir = ROOT / "runs" / case_name
        shutil.rmtree(case_dir, ignore_errors=True)
        try:
            result = subprocess.run(
                [
                    "python3",
                    str(RUNNER),
                    str(SMOKE_CONFIG),
                    "--case-name",
                    case_name,
                    "--prepare-only",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((case_dir / f"prepin.{case_name}").is_file())
            self.assertTrue((case_dir / "case_manifest.json").is_file())
            self.assertFalse(any(case_dir.glob(f"flsgrf1.{case_name}*")))
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_install_flags_are_documented_and_prepare_only_ignores_them(self) -> None:
        """README "Quick start" documents `--flow3d-env`/`FLOW3D_ENV`,
        `--post-home`/`F3D_POST_HOME` and `--out-root`, and states that
        `--prepare-only` needs neither vendor package. Check both: the flags
        exist, and a prepare-only render succeeds with the two env overrides
        pointing at paths that do not exist."""
        import os
        help_out = subprocess.run(
            ["python3", str(RUNNER), "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout
        for flag in ("--flow3d-env", "--post-home", "--out-root"):
            self.assertIn(flag, help_out)
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ)
            env["FLOW3D_ENV"] = str(Path(tmp) / "no_such_weld" / "flow3dvars.sh")
            env["F3D_POST_HOME"] = str(Path(tmp) / "no_such_post")
            result = subprocess.run(
                ["python3", str(RUNNER), str(SMOKE_CONFIG), "--case-name", "unit_install_flags",
                 "--out-root", tmp, "--prepare-only"],
                cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((Path(tmp) / "unit_install_flags" / "prepin.unit_install_flags").is_file())

    def test_existing_run_output_is_not_silently_overwritten(self) -> None:
        case_name = "unit_overwrite_guard"
        case_dir = ROOT / "runs" / case_name
        shutil.rmtree(case_dir, ignore_errors=True)

        def prepare(extra: list[str]) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["python3", str(RUNNER), str(SMOKE_CONFIG),
                 "--case-name", case_name, "--prepare-only", *extra],
                cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False,
            )

        try:
            # Inputs only (prepare-only) re-render freely.
            self.assertEqual(prepare([]).returncode, 0)
            self.assertEqual(prepare([]).returncode, 0)
            # A solver result present -> refuse, naming the offending artifact.
            (case_dir / f"flsgrf1.{case_name}").write_bytes(b"result")
            blocked = prepare([])
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("Refusing to overwrite", blocked.stderr)
            self.assertIn(f"flsgrf1.{case_name}", blocked.stderr)
            # --overwrite forces through.
            self.assertEqual(prepare(["--overwrite"]).returncode, 0)
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_run_summary_parses_log_and_writes_organized_markdown(self) -> None:
        log = (
            "  end of calculation at   t =    1.60003E-02,      cycle =      21505\n"
            "   normal completion: finish time exceeded\n"
            "           flsgrf1.x-flsgrf1.x file size:   10 gb\n"
            "           elapsed time =    3.09571E+04 seconds, or\n"
            "                             0 days :  8 hours : 35 minutes : 57 seconds\n"
            "               cpu time =    3.09487E+04 seconds\n"
            "           date of completion = 06/30/2026\n"
            "           time               = 04:53:57\n"
        )
        _, _, manifest = rectangular.load_and_validate(SMOKE_CONFIG, "unit_summary")
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            case_dir.mkdir()
            (case_dir / "runhyd.log").write_text(log, encoding="utf-8")

            info = rectangular.parse_solver_log(case_dir / "runhyd.log")
            self.assertEqual(info["cycles"], "21505")
            self.assertIn("normal completion", info["completion"])
            self.assertEqual(info["elapsed_human"], "0 days :  8 hours : 35 minutes : 57 seconds")
            self.assertEqual(info["cpu_seconds"], "3.09487E+04")

            summary = rectangular.write_run_summary(case_dir, "unit_summary", manifest)
            text = summary.read_text(encoding="utf-8")

        self.assertEqual(summary.name, "summary.md")
        for header in ("# Run summary: unit_summary", "## Status", "## Geometry and mesh",
                       "## Laser", "## Solver and parallelism", "## Material", "## Artifacts"):
            self.assertIn(header, text)
        self.assertIn("Ray tracing (irefopt)", text)  # laser absorption mode is visible
        self.assertIn("21505", text)
        self.assertIn("normal completion: finish time exceeded", text)
        self.assertIn("NPZ not exported", text)  # no npz present in this fixture
        self.assertIn("runhyd.log", text)        # artifact listed

    def test_run_summary_merges_runtime_into_manifest(self) -> None:
        # case_manifest.json is written at prepare time, before the solve, and it is
        # the file copied next to the NPZ into the ML datasets -- so if the wall-clock
        # is not merged back in after the solve, the shipped dataset carries no timing.
        log = (
            "  end of calculation at   t =    1.60003E-02,      cycle =      21505\n"
            "   normal completion: finish time exceeded\n"
            "           elapsed time =    3.09571E+04 seconds, or\n"
            "                             0 days :  8 hours : 35 minutes : 57 seconds\n"
            "               cpu time =    3.09487E+04 seconds\n"
            "           date of completion = 06/30/2026\n"
            "           time               = 04:53:57\n"
        )
        _, _, manifest = rectangular.load_and_validate(SMOKE_CONFIG, "unit_runtime")
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            case_dir.mkdir()
            (case_dir / "runhyd.log").write_text(log, encoding="utf-8")
            manifest_file = case_dir / "case_manifest.json"
            manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            rectangular.write_run_summary(case_dir, "unit_runtime", manifest)
            written = json.loads(manifest_file.read_text(encoding="utf-8"))

        run = written["run"]
        self.assertAlmostEqual(run["elapsed_seconds"], 30957.1)
        self.assertAlmostEqual(run["cpu_seconds"], 30948.7)
        self.assertAlmostEqual(run["final_time_s"], 1.60003e-2)
        self.assertEqual(run["cycles"], "21505")
        self.assertEqual(run["elapsed_human"], "0 days :  8 hours : 35 minutes : 57 seconds")
        self.assertEqual(run["completion_datetime"], "06/30/2026 04:53:57")
        self.assertIn("normal completion", run["completion"])
        # The solve appends timing; it must not disturb the rendered inputs.
        self.assertEqual(written["input"], manifest["input"])
        self.assertEqual(written["derived"], manifest["derived"])

    def test_run_summary_without_log_leaves_manifest_alone(self) -> None:
        # A prepare-only or pre-solve failure has no runhyd.log: no empty `run` block.
        _, _, manifest = rectangular.load_and_validate(SMOKE_CONFIG, "unit_nolog")
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            case_dir.mkdir()
            manifest_file = case_dir / "case_manifest.json"
            original = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            manifest_file.write_text(original, encoding="utf-8")
            rectangular.write_run_summary(case_dir, "unit_nolog", manifest)
            self.assertEqual(manifest_file.read_text(encoding="utf-8"), original)

    def test_rerender_ignores_post_solve_runtime_block(self) -> None:
        # render() warns when it overwrites a manifest holding different parameters.
        # The post-solve `run` block is not a parameter, so a continuation re-render
        # of the same case must not trip that warning.
        _, replacements, manifest = rectangular.load_and_validate(SMOKE_CONFIG, "unit_rerender")
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            rectangular.render(case_dir, "unit_rerender", replacements, manifest)
            manifest_file = case_dir / "case_manifest.json"
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            data["run"] = {"elapsed_seconds": 123.0, "completion": "normal completion"}
            manifest_file.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rectangular.render(case_dir, "unit_rerender", replacements, manifest)
            self.assertNotIn("overwriting inputs", err.getvalue())
            # A real parameter change still warns.
            changed = json.loads(json.dumps(manifest))
            changed["input"]["laser"]["power_w"] = 999.0
            err2 = io.StringIO()
            with contextlib.redirect_stderr(err2):
                rectangular.render(case_dir, "unit_rerender", replacements, changed)
            self.assertIn("overwriting inputs", err2.getvalue())

    def _fake_npz(self, path: Path, complete: bool = True) -> None:
        """Write a real (or deliberately truncated) .npz at `path`."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("temperature.npy", b"\x93NUMPY payload")
        if not complete:
            data = path.read_bytes()
            path.write_bytes(data[: len(data) // 2])

    def test_solver_result_discarded_once_npz_is_complete(self) -> None:
        # flsgrf is the bulk of a run (0.7-3.6 GB/case here) and is redundant once the
        # NPZ exists: nothing downstream (summary, AiiDA adapter, training) reads it.
        # Keeping it filled the box to 100% on 2026-07-20 and killed a 300-case sweep.
        case_name = "unit_discard"
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            case_dir.mkdir()
            (case_dir / f"flsgrf1.{case_name}").write_bytes(b"x" * 4096)
            (case_dir / f"flsgrf1000.{case_name}").write_bytes(b"x" * 512)
            self._fake_npz(case_dir / "npz" / f"{case_name}.npz")

            freed = rectangular.discard_solver_result(case_dir, case_name)

            self.assertEqual(freed, 4096 + 512)
            self.assertFalse(any(case_dir.glob(f"flsgrf*.{case_name}*")))
            # The export itself must survive -- it is the deliverable.
            self.assertTrue((case_dir / "npz" / f"{case_name}.npz").is_file())

    def test_solver_result_kept_when_npz_is_missing_or_truncated(self) -> None:
        # Deleting flsgrf without a good NPZ would throw away a completed solve.
        case_name = "unit_keep"
        for label, make_npz in (("missing", None), ("truncated", False)):
            with self.subTest(npz=label), tempfile.TemporaryDirectory() as tmp:
                case_dir = Path(tmp) / "case"
                case_dir.mkdir()
                (case_dir / f"flsgrf1.{case_name}").write_bytes(b"x" * 4096)
                if make_npz is False:
                    self._fake_npz(case_dir / "npz" / f"{case_name}.npz", complete=False)

                self.assertEqual(rectangular.discard_solver_result(case_dir, case_name), 0)
                self.assertTrue((case_dir / f"flsgrf1.{case_name}").is_file())

    def test_free_space_guard_gives_up_only_after_waiting(self) -> None:
        # The guard is what stops a batch from filling the root filesystem; without it
        # the solver stalls mid-write and the monitor kills the case.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            available = rectangular.free_space_gb(target)
            self.assertGreater(available, 0.0)
            rectangular.require_free_space(target, 0.0)  # disabled floor never refuses
            rectangular.require_free_space(target, available / 2)  # room to spare, no wait
            with self.assertRaisesRegex(RuntimeError, "free disk"):
                rectangular.require_free_space(target, available + 1e6, wait_s=0.0)

    def test_free_space_guard_waits_for_a_transient_dip(self) -> None:
        # Refusing instantly is what cost the material_ck300 sweep: the spooler fed
        # 300 cases through a refusing gate in five minutes on a dip that then cleared.
        # A floor that clears while waiting must let the solve through, not fail it.
        readings = iter([1.0, 1.0, 500.0])
        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch.object(rectangular, "free_space_gb", lambda _path: next(readings)):
                rectangular.require_free_space(Path(tmp), 30.0, wait_s=60.0, poll_s=0.0)
        self.assertEqual(list(readings), [])  # all three polls consumed

    def test_machine_headroom_waits_for_memory_then_proceeds(self) -> None:
        # The SSH-lockout risk is active thrashing, i.e. low MemAvailable.
        # Batch wrappers gate on it; the AiiDA path had no gate at all, so this moves
        # it into the runner where every caller inherits it.
        readings = iter([4.0, 4.0, 90.0])
        with unittest.mock.patch.object(rectangular, "available_memory_gb", lambda: next(readings)):
            rectangular.require_machine_headroom(min_mem_gb=25.0, max_load=0.0, wait_s=60.0, poll_s=0.0)
        self.assertEqual(list(readings), [])

    def test_machine_headroom_load_gate_is_opt_in(self) -> None:
        # load1 is a lagging one-minute average and a busy machine legitimately sits
        # above a threshold of 16 while a 10-core sweep runs, so gating on it by
        # default would stall back-to-back solves. Off unless asked for.
        with unittest.mock.patch.object(rectangular, "available_memory_gb", lambda: 90.0):
            with unittest.mock.patch.object(rectangular, "load_average_1min", lambda: 99.0):
                rectangular.require_machine_headroom(min_mem_gb=25.0, max_load=0.0, wait_s=0.0)
                with self.assertRaisesRegex(RuntimeError, "load"):
                    rectangular.require_machine_headroom(min_mem_gb=25.0, max_load=16.0, wait_s=0.0)

    def test_nice_target_raises_priority_but_never_lowers_it(self) -> None:
        # --nice=10 makes the solver yield CPU to sshd, but that only applied when
        # a human wrapped the call in systemd-run: neither the spool worker nor the
        # AiiDA path did, so every queued solve ran at whatever it inherited
        # (observed nice 5 at load 32). It belongs in the
        # runner, where every caller passes through -- same argument as the memory
        # and pause gates above.
        self.assertEqual(rectangular.nice_target(10, 0), 10)
        self.assertEqual(rectangular.nice_target(10, 5), 10)

    def test_nice_target_is_a_no_op_when_already_nice_enough(self) -> None:
        self.assertIsNone(rectangular.nice_target(10, 10))
        # Never lower niceness: that needs privilege and would take CPU from sshd,
        # which is the exact failure this protects against.
        self.assertIsNone(rectangular.nice_target(10, 15))

    def test_nice_can_be_disabled(self) -> None:
        self.assertIsNone(rectangular.nice_target(0, 0))
        self.assertIsNone(rectangular.nice_target(-5, 0))

    def test_pause_sentinel_holds_a_run_until_it_is_cleared(self) -> None:
        # One solver lock, several queues (spool, AiiDA, direct). Stopping the spool worker
        # did not stop AiiDA-submitted solves, so the pause lives at the one place
        # every path goes through: the runner.
        with tempfile.TemporaryDirectory() as tmp:
            spool_dir = Path(tmp)
            paused = spool_dir / "PAUSED"
            paused.write_text("migrating", encoding="utf-8")
            calls = {"n": 0}

            def clear_after_two() -> None:
                calls["n"] += 1
                if calls["n"] >= 2:
                    paused.unlink()

            rectangular.wait_while_paused(spool_dir, poll_s=0.0, _on_poll=clear_after_two)
            self.assertGreaterEqual(calls["n"], 2)
            # Nothing to wait for once the sentinel is gone.
            rectangular.wait_while_paused(spool_dir, poll_s=0.0)

    def test_default_profile_is_keyhole_with_surface_tension(self) -> None:
        # Smoke config omits laser flags + solver_profile, so it gets the keyhole
        # default: gaussian, ray-traced, material surface tension kept (not zeroed).
        _, replacements, manifest = rectangular.load_and_validate(SMOKE_CONFIG, "unit_keyhole")
        self.assertEqual(manifest["derived"]["solver_profile"], "keyhole")
        self.assertEqual(manifest["derived"]["laser_heat_distribution"], "gaussian")
        self.assertEqual(replacements["__LASER_REFLECTION_FLAG__"], "3")
        self.assertEqual(replacements["__LASER_MAX_REFLECTIONS__"], "20")
        self.assertEqual(replacements["__SIGMA__"], "__MATERIAL_SIGMA__")
        self.assertEqual(replacements["__MATERIAL_SIGMA__"], "1.58287")

    def test_ring_beam_profile_renders_multiple_comoving_beamlets(self) -> None:
        raw = json.loads(REPRO_CONFIG.read_text(encoding="utf-8"))
        raw["laser"]["power_w"] = 120.0
        raw["laser"]["beam_profile"] = {
            "type": "ring",
            "angular_samples": 4,
            "radius_fraction": 0.5,
            "beamlet_radius_fraction": 0.2,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ring.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            _, replacements, manifest = rectangular.load_and_validate(path, "unit_ring")
            text = rectangular.render(Path(tmp) / "case", "unit_ring", replacements, manifest).read_text(encoding="utf-8")

        self.assertEqual(manifest["derived"]["laser_beam_profile_type"], "ring")
        self.assertEqual(manifest["derived"]["laser_beam_count"], 4)
        self.assertEqual(text.count("beam_name("), 4)
        self.assertIn("nlbeam=4", text)
        self.assertIn("powlbm(1,1)=30", text)
        self.assertIn("powlbm(1,4)=30", text)
        self.assertIn("xb0lbm(1)=-0.000275", text)
        self.assertIn("yb0lbm(2)=2.5e-05", text)
        self.assertIn("rflbm(4)=1e-05", text)
        self.assertAlmostEqual(sum(b["power_fraction"] for b in manifest["derived"]["laser_beamlets"]), 1.0)

    def test_radial_table_beam_profile_conserves_power(self) -> None:
        raw = json.loads(REPRO_CONFIG.read_text(encoding="utf-8"))
        raw["laser"]["power_w"] = 100.0
        raw["laser"]["beam_profile"] = {
            "type": "radial_table",
            "angular_samples": 4,
            "beamlet_radius_fraction": 0.18,
            "samples": [
                {"radius_fraction": 0.0, "weight": 1.0},
                {"radius_fraction": 0.55, "weight": 3.0},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tabular.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            _, replacements, manifest = rectangular.load_and_validate(path, "unit_tabular")
            text = rectangular.render(Path(tmp) / "case", "unit_tabular", replacements, manifest).read_text(encoding="utf-8")

        self.assertEqual(manifest["derived"]["laser_beam_profile_type"], "radial_table")
        self.assertEqual(manifest["derived"]["laser_beam_count"], 5)
        self.assertIn("nlbeam=5", text)
        self.assertIn("powlbm(1,1)=25", text)
        self.assertIn("powlbm(1,2)=18.75", text)
        self.assertIn("powlbm(1,5)=18.75", text)
        self.assertAlmostEqual(sum(b["power_fraction"] for b in manifest["derived"]["laser_beamlets"]), 1.0)

    def test_multi_beam_profile_rejects_generalized_paths_for_now(self) -> None:
        raw = json.loads(REPRO_CONFIG.read_text(encoding="utf-8"))
        raw["laser"]["beam_profile"] = {"type": "ring", "angular_samples": 4}
        raw["laser"]["path"] = {
            "type": "waypoints",
            "points_m": [[-0.0002, 0.0], [0.0002, 0.0]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_path_ring.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(rectangular.ConfigError, "straight or dwell"):
                rectangular.load_and_validate(path, "unit_bad_path_ring")

    def test_legacy_profile_still_zeroes_surface_tension(self) -> None:
        _, replacements, manifest = rectangular.load_and_validate(REPRO_CONFIG, "unit_legacy_sigma")
        self.assertEqual(manifest["derived"]["solver_profile"], "legacy_case_fine_1")
        self.assertEqual(replacements["__SIGMA__"], "0")

    def test_fresnel_epsilon_inversion_roundtrips(self) -> None:
        for absorptivity in (0.35, 0.10, 0.03):
            epsilon = rectangular.fresnel_epsilon_from_absorptivity(absorptivity)
            a0 = 1.0 - (1.0 + (1.0 - epsilon) ** 2) / (1.0 + (1.0 + epsilon) ** 2)
            self.assertAlmostEqual(a0, absorptivity, places=4)

    def test_fresnel_epsilon_from_resistivity_drude(self) -> None:
        # Liquid Ti-6Al-4V ~1.5e-6 ohm.m at 1.064 um -> epsilon ~0.217 (A0 ~0.35).
        epsilon = rectangular.fresnel_epsilon_from_resistivity(1.5e-6, 1.064e-6)
        self.assertAlmostEqual(epsilon, 0.217, places=2)
        # Higher resistivity -> higher epsilon (more absorption).
        self.assertGreater(rectangular.fresnel_epsilon_from_resistivity(2.0e-6, 1.064e-6), epsilon)

    def test_fresnel_from_material_prefers_resistivity(self) -> None:
        cfg = json.loads(SMOKE_CONFIG.read_text())
        cfg["laser"]["fresnel_from_material"] = True  # ti_imi318_mills: rho(T) CSV, A 0.35
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fresnel.json"
            path.write_text(json.dumps(cfg))
            _, replacements, manifest = rectangular.load_and_validate(path, "unit_fresnel")
        self.assertEqual(manifest["derived"]["laser_fresnel_method"], "resistivity_drude")
        self.assertAlmostEqual(float(replacements["__LASER_REFLECTIVITY__"]), 0.217, places=2)
        self.assertAlmostEqual(manifest["derived"]["laser_normal_absorptivity_from_ereflct"], 0.35, places=2)
        # The material-derived coupling is now the DEFAULT whenever the material
        # carries optical data (ti_imi318_mills ships a resistivity CSV), so no flag is needed.
        _, repl_default, man_default = rectangular.load_and_validate(SMOKE_CONFIG, "unit_fresnel_default")
        self.assertEqual(man_default["derived"]["laser_fresnel_method"], "resistivity_drude")
        self.assertAlmostEqual(float(repl_default["__LASER_REFLECTIVITY__"]), 0.217, places=2)
        # Setting laser.fresnel_from_material=false forces the fixed scalar reflectivity.
        off = json.loads(SMOKE_CONFIG.read_text())
        off["laser"]["fresnel_from_material"] = False
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "off.json"
            path.write_text(json.dumps(off))
            _, repl_off, man_off = rectangular.load_and_validate(path, "unit_fresnel_off")
        self.assertEqual(man_off["derived"]["laser_fresnel_method"], "fixed")
        self.assertEqual(repl_off["__LASER_REFLECTIVITY__"], "0.35")

    def test_fresnel_prefer_resistivity_false_uses_absorptivity(self) -> None:
        # Copper sets prefer_resistivity=false (Drude overestimates at ~1 um), so the
        # measured-absorptivity route is used and both routes are logged.
        cfg = json.loads(SMOKE_CONFIG.read_text())
        cfg["material"] = {"name": "pure_copper"}
        cfg["laser"]["fresnel_from_material"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cu.json"
            path.write_text(json.dumps(cfg))
            _, _, manifest = rectangular.load_and_validate(path, "unit_cu_fresnel")
        d = manifest["derived"]
        self.assertEqual(d["laser_fresnel_method"], "absorptivity_inversion")
        self.assertAlmostEqual(d["laser_normal_absorptivity_from_ereflct"], 0.03, places=2)
        # the rejected (Drude) route is still recorded as an alternative for transparency
        self.assertIn("resistivity_drude", d["laser_fresnel_alternatives"])
        self.assertGreater(d["laser_fresnel_alternatives"]["resistivity_drude"]["normal_absorptivity"], 0.1)

    def test_fresnel_resistivity_renders_temperature_dependent_ereflct(self) -> None:
        # ti_imi318_mills has a resistivity(T) CSV and prefers the resistivity route,
        # so ereflct is rendered as a temperature table (iereflct=1).
        cfg = json.loads(SMOKE_CONFIG.read_text())
        cfg["laser"]["fresnel_from_material"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(cfg))
            case_name, replacements, manifest = rectangular.load_and_validate(path, "unit_eps_t")
            text = rectangular.render(Path(tmp) / "case", case_name, replacements, manifest).read_text()
        self.assertTrue(manifest["derived"]["laser_ereflct_temperature_dependent"])
        self.assertIn("iereflct=1,", text)
        self.assertIn("#ereflct(1)", text)
        # Copper uses the absorptivity route, so no temperature-dependent ereflct table.
        cu = json.loads(SMOKE_CONFIG.read_text())
        cu["material"] = {"name": "pure_copper"}
        cu["laser"]["fresnel_from_material"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cu.json"
            path.write_text(json.dumps(cu))
            _, _, man_cu = rectangular.load_and_validate(path, "unit_cu_eps")
        self.assertFalse(man_cu["derived"]["laser_ereflct_temperature_dependent"])

    @staticmethod
    def _source_files() -> list[str]:
        """Files of this repository, relative to ROOT.

        Inside a git checkout these are the tracked files. Outside one (an
        unpacked archive, where `git ls-files` returns nothing) the tree is
        walked instead, leaving out run outputs and the local licence material
        that is never distributed (`license.env`, `.env*`, `*.lic`, `licenses/`).
        """
        import os

        try:
            out = subprocess.run(
                ["git", "-C", str(ROOT), "ls-files", "-z"],
                capture_output=True, text=True, timeout=60,
            )
            names = [n for n in out.stdout.split("\0") if n] if out.returncode == 0 else []
        except (OSError, subprocess.SubprocessError):
            names = []
        if names:
            return names
        skip_dirs = {".git", "__pycache__", "runs", "licenses"}
        found = []
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fn in filenames:
                if fn == "license.env" or fn.startswith(".env") or fn.endswith(".lic"):
                    continue
                found.append(str(Path(dirpath, fn).relative_to(ROOT)))
        return found

    def test_cli_help_and_docs_use_placeholder_license_endpoint(self) -> None:
        """No repository file may carry a real `PORT@HOST` licence endpoint.

        A hardcoded list of files to check would pass a leak in any file nobody
        thought to list, so every text file of the repository is scanned, which
        also covers files that do not exist yet.
        """
        tracked = self._source_files()
        self.assertTrue(tracked, "found no repository files to scan")

        skip_suffixes = (".png", ".jpg", ".jpeg", ".gif", ".pdf", ".npz", ".npy", ".zip")
        offenders = []
        placeholder_seen = False
        for name in tracked:
            if not name or name.endswith(skip_suffixes):
                continue
            path = ROOT / name
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "PORT@HOST" in text:
                placeholder_seen = True
            for match in re.finditer(r"\b[0-9]{2,5}@[A-Za-z0-9_.-]+", text):
                # `PORT@HOST` is the documented placeholder and is what we want.
                if match.group(0).upper().startswith("PORT@"):
                    continue
                offenders.append(f"{name}:{text[:match.start()].count(chr(10)) + 1}")

        # Name the file and line, never the value -- a failure message that
        # quotes the endpoint would leak it into every CI log.
        self.assertEqual(
            offenders, [],
            f"licence-endpoint-shaped strings in tracked files: {', '.join(offenders)}",
        )
        self.assertTrue(placeholder_seen, "the documented PORT@HOST placeholder went missing")

    def test_full_solver_test_documents_required_local_license_setup(self) -> None:
        text = (ROOT / "tests" / "full_solver_workflow.py").read_text(encoding="utf-8")

        self.assertIn("No FLOW-3D license endpoint was found", text)
        self.assertIn("export F3D_LICENSE='<your-license-endpoint>'", text)
        self.assertIn("license.env", text)
        self.assertIn("PORT@HOST", text)

    def test_material_files_are_valid(self) -> None:
        # Each material is a directory: materials/<name>/<name>.json + CSV tables.
        material_files = sorted(MATERIALS_DIR.glob("*/*.json"))
        self.assertGreaterEqual(len(material_files), 5)
        for material_file in material_files:
            _, manifest = rectangular.load_material(SMOKE_CONFIG, {"file": str(material_file)})
            self.assertTrue(manifest["name"])
            self.assertGreater(manifest["properties"]["liquidus_temperature_k"], 0.0)
            self.assertGreater(manifest["table_rows"]["density_kg_m3"], 0)

    def test_material_molar_mass_renders_clausius_clapeyron_tvexp(self) -> None:
        """A material carrying vaporization_molar_mass_kg_mol must render a
        per-material tvexp = R/(M*clhv1) (the solver's own per-material
        definition; A30 v2).

        The shipped default material now carries M: without it any sweep on the
        default silently renders the legacy saturation curve described in A30,
        which the calibration campaign replaced, defeating the decision to
        regenerate training data with corrected physics. The
        no-M fallback (legacy constant untouched) remains only for materials
        genuinely lacking the field, covered by the stripped-copy branch."""
        base = MATERIALS_DIR / "ti_imi318_mills" / "ti_imi318_mills.json"
        repl, man = rectangular.load_material(SMOKE_CONFIG, {"file": str(base)})
        self.assertIn("__TVEXP__", repl)
        self.assertAlmostEqual(
            float(repl["__TVEXP__"]), 8.314462618 / (0.047867 * 9.83e6), places=9)
        self.assertAlmostEqual(
            man["properties"]["vaporization_molar_mass_kg_mol"], 0.047867)

        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "mat"
            shutil.copytree(base.parent, dst)
            j = json.loads((dst / "ti_imi318_mills.json").read_text())
            del j["properties"]["vaporization_molar_mass_kg_mol"]
            (dst / "ti_imi318_mills.json").write_text(json.dumps(j))
            repl2, _ = rectangular.load_material(
                SMOKE_CONFIG, {"file": str(dst / "ti_imi318_mills.json")})
            self.assertNotIn("__TVEXP__", repl2)

    def test_vtk_validator_detects_temperature_rise_and_melt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = root / "frame_00000.vtk"
            frame.write_bytes(
                b"# vtk DataFile Version 5.1\n"
                b"fixture\n"
                b"BINARY\n"
                b"DATASET POLYDATA\n"
                b"POINTS 2 float\n"
                + struct.pack(">ffffff", 0, 0, 0, 1, 0, 0)
                + b"\nPOINT_DATA 2\n"
                + b"Melt%20Region 1 2 float\n"
                + struct.pack(">ff", 0.0, 0.25)
                + b"\nSolid%20Fraction 1 2 float\n"
                + struct.pack(">ff", 1.0, 0.5)
                + b"\nTemperature 1 2 float\n"
                + struct.pack(">ff", 293.15, 1800.0)
            )
            pvd = root / "series.pvd"
            pvd.write_text(
                '<?xml version="1.0"?>\n'
                '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
                "  <Collection>\n"
                '    <DataSet timestep="0" group="" part="0" file="frame_00000.vtk"/>\n'
                "  </Collection>\n"
                "</VTKFile>\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    "python3",
                    str(VTK_VALIDATOR),
                    str(pvd),
                    "--initial-temperature-k",
                    "293.15",
                    "--min-temperature-rise-k",
                    "100",
                    "--require-melt",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("max_melt_region=0.25", result.stdout)

    def test_all_step_exporter_defaults_to_volume_fields_for_training(self) -> None:
        exporter_text = (ROOT / "python" / "export_all_timesteps_to_vtk.py").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn('default="volume"', exporter_text)
        self.assertIn('fluid.Surface = "Fluid"', exporter_text)
        self.assertIn('fluid.Type = "Isovolume"', exporter_text)
        self.assertIn('"Fraction Of Fluid"', exporter_text)
        self.assertIn('"Solid Fraction"', exporter_text)
        self.assertIn("--representation surface", readme)
        self.assertIn("full 3D `Fluid` isovolume data", readme)

    def test_vtk_export_writes_point_data_and_surface_mode_is_wired(self) -> None:
        # ParaView Contour needs point data; both exporters add a cell->point pass
        # (PassCellData keeps the cell values) and write both. The runner exposes
        # the surface representation through --export vtk-surface.
        all_steps = (ROOT / "python" / "export_all_timesteps_to_vtk.py").read_text(encoding="utf-8")
        final_surface = (ROOT / "python" / "export_fluid_to_vtk.py").read_text(encoding="utf-8")
        self.assertIn("CellDatatoPointData", all_steps)
        self.assertIn("PointDataArrays=POINT_ARRAYS", all_steps)
        self.assertIn("PassCellData = 1", all_steps)
        self.assertIn("CellDatatoPointData", final_surface)
        help_text = subprocess.run(
            ["python3", str(RUNNER), "--help"], capture_output=True, text=True, check=True
        ).stdout
        self.assertIn("vtk-surface", help_text)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("vtk-surface", readme)
        self.assertIn("point data", readme)

    def test_legacy_vtk_directory_comparator_matches_time_suffix_and_arrays(self) -> None:
        def write_vtk(path: Path, temperature: float) -> None:
            path.write_bytes(
                b"# vtk DataFile Version 5.1\n"
                b"fixture\n"
                b"BINARY\n"
                b"DATASET UNSTRUCTURED_GRID\n"
                b"POINTS 1 float\n"
                + struct.pack(">fff", 0.0, 0.0, 0.0)
                + b"\nCELL_DATA 1\n"
                b"Temperature 1 1 float\n"
                + struct.pack(">f", temperature)
                + b"\nMelt%20Region 1 1 float\n"
                + struct.pack(">f", 0.0)
                + b"\nSolid%20Fraction 1 1 float\n"
                + struct.pack(">f", 1.0)
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left"
            right = root / "right"
            left.mkdir()
            right.mkdir()
            write_vtk(left / "old_prefix_9.vtk", 300.0)
            write_vtk(right / "new_prefix_9.vtk", 300.0)
            result = subprocess.run(
                [
                    "python3",
                    str(VTK_COMPARATOR),
                    str(left),
                    str(right),
                    "--array",
                    "Temperature",
                    "--array",
                    "Melt Region",
                    "--array",
                    "Solid Fraction",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("frames=1", result.stdout)

    def test_legacy_vtk_directory_comparator_can_match_nearby_suffix_means(self) -> None:
        def write_vtk(path: Path, temperature_values: tuple[float, float]) -> None:
            path.write_bytes(
                b"# vtk DataFile Version 5.1\n"
                b"fixture\n"
                b"BINARY\n"
                b"DATASET UNSTRUCTURED_GRID\n"
                b"POINTS 2 float\n"
                + struct.pack(">ffffff", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                + b"\nCELL_DATA 2\n"
                b"Temperature 1 2 float\n"
                + struct.pack(">ff", *temperature_values)
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left"
            right = root / "right"
            left.mkdir()
            right.mkdir()
            write_vtk(left / "old_prefix_124.vtk", (300.0, 302.0))
            write_vtk(right / "new_prefix_125.vtk", (299.0, 303.0))
            result = subprocess.run(
                [
                    "python3",
                    str(VTK_COMPARATOR),
                    str(left),
                    str(right),
                    "--array",
                    "Temperature",
                    "--suffix-tolerance-us",
                    "1",
                    "--stat",
                    "mean",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("max_suffix_delta_us=1", result.stdout)


if __name__ == "__main__":
    unittest.main()
