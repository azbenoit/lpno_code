# FLOW-3D WELD: rectangular moving-laser runs

These scripts give programmatic control of FLOW-3D WELD: a JSON config is
rendered into a complete solver input deck, the case is launched through the
vendor launchers, and every stored time step is exported to one NPZ file. Running
a solve needs an installed FLOW-3D WELD (with FLOW-3D POST for the export) and an
active licence of your own; the licence endpoint is read from your environment
(`F3D_LICENSE` or an ignored `license.env`). The prepare-only path
(`--prepare-only`) renders and checks the inputs without the solver or a
licence. The repository contains no part of the solver.

The modelled case is a rectangular metal substrate heated by a vertical laser: a
straight scan by default, or a stationary spot, a curved or scripted path, or up
to four independent beams (`laser.beams`).

The model is generated from JSON. All values exposed
to the user are in SI units: metres, kilograms, seconds, kelvin, watts and
pascals. FLOW-3D's generated `prepin.*` file is complete and can still be
opened in the GUI for inspection.

Mesh convergence, material-property validation, absorption, and boundary
conditions must be checked for any scientific result.

### What the manuscript does not cover

The accompanying manuscript uses the NPZ export, straight, random and
variable-speed scans, stationary spots, independent beams, uniform meshes down to
5 µm and the validation pipeline under `validation/`. The following features are
implemented and tested here but play no part in its results:

- the VTK/PVD exports (`--export vtk`, `vtk-surface`, `vtk-legacy60`, `both`) and
  the VTK helpers (`export_*_vtk.py`, `validate_vtk_series.py`,
  `compare_legacy_vtk_dirs.py`);
- the `waypoints` and `sinusoidal` path types;
- radial beam shaping (`laser.beam_profile`, `ring` and `radial_table`);
- nested mesh refinement (`mesh.refinement`);
- the scaling-law check against Gan et al. 2021 (`python/validate_scaling_law.py`);
- the plume model knobs (`laser.plume_*`), which appear only in early validation
  configs and are retired from every recipe (`assumptions.md` A31).

## What is in this directory

```text
configs/rectangular_moving_laser_smoke.json      Small, fast SI launch test
configs/rectangular_moving_laser_refined_smoke.json
                                                 Smoke test with a nested fine mesh block
configs/rectangular_moving_laser_melt.json       Validated heating/melting example
configs/rectangular_curved_path_sinusoidal.json  Sinusoidal `laser.path` example
configs/case_fine_1_10um_reproduction.json       10 µm legacy line-track reproduction
configs/case_fine_1_keyhole.json                 Same case on the `keyhole` solver profile
configs/case_fine_1_fresnel.json                 Same case with material-derived Fresnel coupling
materials/<name>/<name>.json                     Per-material scalars + optical block (SI units)
materials/<name>/*.csv                           Temperature-dependent tables (temperature_k,value)
materials/sources.md                             Per-parameter provenance and citations
template/prepin.rectangular_moving_laser.template
                                                 FLOW-3D input template
python/run_rectangular_case.py                   JSON validation, rendering, launch, export
python/config_schema.py                          The config schema; generates the README table
python/laser_path.py                             Scan-path construction (waypoints/sinusoidal/random/variable-speed)
python/plot_laser_path.py                        Plot a run's scan path to PNG
bash/run_rectangular_case.sh                     Short shell entry point
python/build_batch.py, python/batch.py           Expand one config into a sweep + a task file
python/rebuild_batch_physics.py                  Rewrite a batch's configs onto the calibrated recipe
python/spool.py                                  Persistent one-at-a-time task queue for runs
bash/spool.sh                                    Short shell entry point for the spooler
python/monitor_run.py                            Optional stall/divergence watchdog for a solve
python/resource_guards.py                        The enforce/warn switch behind every memory/disk guard
runs/<case_name>/                                Isolated generated cases and outputs (untracked)
python/export_all_timesteps_to_npz.py            Every written time step → one structured NPZ (default)
python/npz_block_merge.py                        Stitch multi-rank decomposed blocks into one grid
python/export_fluid_to_vtk.py                    Final workpiece surface → VTK
python/export_all_timesteps_to_vtk.py            Every written time step → VTK/PVD
python/export_final_frame_vtk.py                 Final stored step only → volume/surface VTK
python/export_legacy_even_timesteps_to_vtk.py
                                                 Old 60-frame Fluid isovolume export
python/compare_legacy_vtk_dirs.py                Compare legacy VTK directories
python/validate_vtk_series.py                    Numeric VTK/PVD heating and melt check
python/validate_npz_series.py                    Numeric NPZ heating and melt check
python/verify_npz_checksums.py                   sha256 sidecars that make a corrupt NPZ detectable
python/subsample_npz.py                          Thin a run NPZ to <=N time steps
python/compare_npz_to_reference.py               Compare an NPZ against a reference NPZ
python/reclaim_flsgrf.py                         Apply the flsgrf-cleanup policy to older runs
bash/finalize_run.sh                             Post-solve: subsample NPZ, final-frame VTK, drop FLSGRF
python/melt_metrics.py                           Melt-pool depth/width metric definitions
python/score_anchors.py                          Score runs against published anchors in validation/anchors/
python/validate_scaling_law.py                   Keyhole-depth check against the Gan 2021 scaling law
python/data/gan_scaling_reference.json           Derived (Ke, e/r0) reference points and fitted curve
python/material_conditioning.py                  Material-derived scalars (King enthalpy K1 etc.) for surrogates
validation/                                      Reproducibility deliverable: configs, frozen materials,
                                                 anchors, scoring plan, plan/outcome records, reproduce.py
batches/                                         Tracked record of the two published active-learning loops:
                                                 drivers, design tables, per-round configs and state (no solver output)
validation/make_figures.py                       Calibration figures from the tracked sources
validation/score_spot_mesh_pair.py               NIST-spot 10/5 µm mesh-pair scorer
python/local_inference.py                        Run a trained surrogate locally (needs lpno + torch)
python/predicted_fields.py                       Make a prediction measurable by melt_metrics
python/score_predictions.py                      Score a surrogate's fields like the FLOW-3D campaign
tests/                                           Short no-license regression tests
assumptions.md                                   Non-trivial design assumptions and code choices
potential_bugs.md                                Known issues (P-IDs): symptom, cause, workaround, code
docs/developer_notes.md                          Possible extensions
```

`local_inference.py`, `score_predictions.py` and `validation/make_figures.py`
reach into the `lpno` sibling repository (the neural-operator code
and its data), found through `PINO_SIMULATIONS_ROOT` or a `../lpno`
checkout; they do not run from this repository alone. Everything else needs
only Python, `numpy`, and (for solving and exporting) the FLOW-3D packages
named below.

## Quick start

Python 3.10+ plus two packages the scripts import directly:

```bash
pip install numpy matplotlib
```

`numpy` is needed by every export and validation helper (`python/export_all_timesteps_to_npz.py`,
`python/validate_npz_series.py`) and by most of the test suite; `matplotlib` by
the plotting helpers. FLOW-3D itself comes from the vendor install.

Two vendor packages are needed for a solve and an export, and the runner looks
for both under `~/.local/opt/flow3d/` by default:

| Package | Used for | Default location | Override |
| --- | --- | --- | --- |
| FLOW-3D WELD 2025R1 update 1 | the solver (`runpre_par`, `runhyd_par`), entered by sourcing its `flow3dvars.sh` | `~/.local/opt/flow3d/FLOW3D_WELD_2025R1u1/local/flow3dvars.sh` | `--flow3d-env <path>` or `FLOW3D_ENV` |
| FLOW-3D POST 2025R1 | the result reader and the `pvpython` runtime every export uses | `~/.local/opt/flow3d/FLOW3D_POST_2025R1` | `--post-home <dir>` or `F3D_POST_HOME` |

Those are the versions this repo was validated against; another release needs
the same two entry points (`local/flow3dvars.sh` and `bin/pvpython`) and has not
been tested. A user-local install needs no root. Runs land under `runs/` by
default; `--out-root <dir>` moves a run's `<case_name>` folder elsewhere (batches
use it to keep all their cases together). `--prepare-only` needs neither package.

Configure the FLOW-3D license endpoint in your local shell before solver runs.
Do not commit real license hostnames, ports, MAC addresses, license files, or
license-server details to this repository. The endpoint format is `PORT@HOST`.
From `flow3D_scripts`:

```bash
export F3D_LICENSE='<your-license-endpoint>'
./bash/run_rectangular_case.sh configs/rectangular_moving_laser_smoke.json \
  --case-name my_first_rectangular_case
```

For repeated local runs, you may put the endpoint in ignored `license.env`:

```bash
F3D_LICENSE='<your-license-endpoint>'
```

`license.env` is ignored by git and must stay local to this machine.

The runner checks the JSON before it requests a solver licence. It writes a
new directory at `runs/<case_name>/` and refuses to start when that folder
already holds solver output or exports (`flsgrf*`, `npz/`, `paraview/`),
so an expensive run is never silently overwritten. Use a different
`case_name` for every changed run, or pass `--overwrite` to deliberately
reuse one. Inputs alone (from `--prepare-only` or a pre-solve failure)
re-render without the flag.

To inspect the generated FLOW-3D input without running the solver:

```bash
./bash/run_rectangular_case.sh configs/rectangular_moving_laser_smoke.json \
  --case-name review_case --prepare-only
```

`--prepare-only` does not need `F3D_LICENSE`.

## Coordinate convention and geometry

The coordinates follow an OpenFOAM-style laser reference point:

- `start_point_m = [0, 0, 0]` by default.
- The start and end points are focal points on the top substrate surface:
  `z = 0`.
- The laser is vertical, directed in `-z`; its optical origin is one focal
  distance above the surface.
- The substrate surface is always `z = 0`.
- The solid substrate extends down to `z = -thickness_m`.
- The gas layer extends up to `z = gas_clearance_m`.

By default the generated geometry is a FLOW-3D half-space free surface. In JSON,
you specify dimensions and the runner derives the min/max bounds:

```json
"geometry": {
  "length_m": 0.0009,
  "width_m": 0.0004,
  "thickness_m": 0.0003,
  "gas_clearance_m": 0.0001,
  "placement": "centered"
}
```

If these four dimensions are omitted, the runner uses the same defaults shown
above: `0.9 mm × 0.4 mm × 0.3 mm` substrate with `0.1 mm` gas clearance.
The legacy line-track reproduction uses those values.

`geometry.placement` controls how the x/y bounds are derived:

- `laser_leading_margin` starts the substrate four beam radii before
  `laser.start_point_m` in x and centres it around the start y coordinate.
  This is the default for new moving-laser cases.
- `centered` centres the x/y rectangle around `(0, 0)`. With the defaults
  above, the mesh bounds are derived as:

```text
x: [-0.45 mm, 0.45 mm]
y: [-0.20 mm, 0.20 mm]
z: [-0.30 mm, 0.10 mm]
```

The laser start and end focal points must be at `z = 0` and at least one beam
radius from every x/y edge. The runner writes the derived bounds to
`case_manifest.json` so they can be checked before a solver run.

```text
                         +z (laser origin at z = focal_distance)
                         |
                beam ↓  |    laser focus on z = 0
                         v
               ┌─────────────────────────────────────┐ z = 0
               │---- linear scan along x/y ---------->│
               └─────────────────────────────────────┘ z = -thickness
               mesh top is z = gas_clearance_m
```

No STL is involved. The current path writes FLOW-3D analytic initialization
with `fzh=surface_z_m`, where `surface_z_m` defaults to `0`. The gas layer is
part of the mesh above that free surface. This is the path used for the
legacy line-track reproduction and for new rectangular cases unless a config
explicitly asks for an older compatibility mode.

`geometry.bounds_m` is still accepted for old experiments; the maintained
rectangular workflow uses dimensions plus `placement`.

## JSON configuration

Use `configs/rectangular_moving_laser_smoke.json` as the starting point:

```json
{
  "schema_version": 1,
  "case_name": "my_first_rectangular_case",
  "material": { "name": "ti_imi318_mills" },
  "parallel": {
    "cores": 1,
    "threads_per_rank": 1
  },
  "geometry": {
    "length_m": 0.0024,
    "width_m": 0.0012,
    "thickness_m": 0.0004,
    "gas_clearance_m": 0.0004
  },
  "mesh": { "cell_size_m": 0.000075 },
  "laser": {
    "power_w": 300.0,
    "beam_radius_m": 0.00015,
    "focal_distance_m": 0.01,
    "lens_radius_m": 0.0003,
    "start_point_m": [0.0, 0.0, 0.0],
    "end_point_m": [0.0008, 0.0, 0.0],
    "scan_speed_m_per_s": 0.1
  },
  "simulation": {
    "initial_temperature_k": 293.15,
    "finish_time_s": 0.0085,
    "max_iterations": 5000
  }
}
```

### Parameter meaning and checks

The table below is generated from the runner's schema (`python/config_schema.py`).
Regenerate it after changing the schema with
`python3 python/run_rectangular_case.py --print-config-schema`; a test fails if
this block and the generator disagree.

<!-- BEGIN GENERATED CONFIG SCHEMA -->
| JSON value | Meaning | Requirement enforced by the runner |
| --- | --- | --- |
| `material` | Material name, path, or object | Defaults to `ti_imi318_mills`; see material files below |
| `parallel.cores` | Total local CPU cores allocated to this run | Positive integer; defaults to 1 |
| `parallel.threads_per_rank` | OpenMP threads per MPI rank | Positive integer; must divide `cores`; defaults to 1 |
| `geometry.length_m` | Full x extent | Positive; defaults to `0.0009` |
| `geometry.width_m` | Full y extent | Greater than 2 beam radii; defaults to `0.0004` |
| `geometry.thickness_m` | Solid depth below `z=0` | Positive; defaults to `0.0003` |
| `geometry.gas_clearance_m` | Mesh height above the surface | Positive; defaults to `0.0001` |
| `geometry.placement` | `laser_leading_margin` or `centered` | Defaults to `laser_leading_margin` |
| `geometry.bounds_m` | Compatibility-only explicit `x`, `y`, `z` mesh bounds | Each `[min,max]` pair must be increasing |
| `geometry.initialization` | Compatibility mode: `legacy_halfspace`, `substrate_with_gas`, or `filled_mesh` | Defaults to `legacy_halfspace` |
| `geometry.surface_z_m` | Free-surface height (`fzh(1)`): where the metal/gas interface sits | Defaults to `0.0`; must lie within the mesh z bounds |
| `mesh.cell_size_m` | Nominal Cartesian cell size | At most `beam_radius_m / 2` |
| `mesh.refinement` | Optional nested finer block around the laser track | Omit for a single uniform mesh |
| `mesh.refinement.factor` | Fine cell = `cell_size_m / factor` | Integer `2` or `3` |
| `mesh.refinement.margin_m` | How far the fine block extends past the beam footprint | Positive; single-rank runs only |
| `laser.power_w` | Laser power | Non-negative; required unless `laser.beams` is set (then the default for beams that omit `power_w`) |
| `laser.beam_radius_m` | Beam radius at the surface (`rflbm`); sets the mesh-size and margin checks | Positive; required |
| `laser.focal_distance_m` | Focal distance (`dlflbm`, also the beam origin height `zb0lbm`) | Positive; required |
| `laser.lens_radius_m` | Beam radius at the lens (`r0lbm`) | Positive; required |
| `laser.start_point_m` | Surface focal-point start `[x,y,0]` | Defaults to `[0,0,0]`; z must be zero |
| `laser.end_point_m` | Surface focal-point end `[x,y,0]` | Must be one radius inside all edges |
| `laser.scan_speed_m_per_s` | Magnitude of the in-plane scan velocity | Positive; held constant along the path (nominal/reference speed for `variable_speed`, whose per-leg speeds come from `times_s`) |
| `laser.dwell_time_s` | Stationary (dwell) spot: hold `start_point_m` for this long | Optional; positive. Zero in-plane velocity, `scan_path_length_m` = 0; omit `end_point_m` (or set it equal to the start) and `scan_speed_m_per_s` becomes optional. Used for keyhole-onset experiments (e.g. Simonds et al. 2021) |
| `laser.path` | Optional generalized in-plane scan path; overrides `start_point_m`/`end_point_m` | Object; forces `centered`/`bounds_m` placement; every sampled point must stay at least 4 beam radii inside all x/y edges (hard check) |
| `laser.path.type` | `waypoints`, `sinusoidal`, `random`, or `variable_speed` | Required when `laser.path` is set |
| `laser.path.knots_m` | `variable_speed`: ordered `[x, y]` beam positions (kept verbatim, no arc-length resample) | At least two; a repeated knot with advancing time is a dwell |
| `laser.path.times_s` | `variable_speed`: arrival time at each knot | One per knot; strictly increasing; first must be 0. Per-leg speed = chord / Δt (0 for a dwell) |
| `laser.path.power_w` | `variable_speed`: optional per-leg laser power (e.g. to de-rate dwells) | One value per leg (`len(knots_m) - 1`); non-negative; defaults to `laser.power_w` on every leg |
| `laser.path.points_m` | `waypoints`: ordered `[x, y]` surface points | At least two; z must be zero; joined per `interpolation` |
| `laser.path.interpolation` | `waypoints` join: `linear` polyline or `spline` (Catmull-Rom through the points) | Defaults to `linear` |
| `laser.path.length_m` | `sinusoidal` travel length / `random` path length (reached exactly; the walk self-intersects if the domain is smaller) | Positive; `sinusoidal` requires it; `random` defaults to the domain length (x extent) |
| `laser.path.amplitude_m` | `sinusoidal`: transverse (y) amplitude about the +x travel axis | Non-negative |
| `laser.path.wavelength_m` / `periods` | `sinusoidal`: spatial period along x, or number of periods over `length_m` | Give exactly one |
| `laser.path.seed` | `random`: RNG seed for the reproducible coverage-biased walk | Integer; defaults to 0 |
| `laser.path.step_m` / `max_turn_deg` | `random`: control-point step and max per-step heading change (open-space smoothness; wall recovery may turn sharper) | Default `4*beam_radius` and `45` |
| `laser.path.start_point_m` | `sinusoidal`/`random` start `[x, y, 0]` | Defaults to `[0,0,0]` (sinusoidal) or the cropped region's left edge, mid height (random) |
| `laser.path.sample_spacing_m` | Arc-length spacing of the constant-speed velocity-table samples | Defaults to `beam_radius/2`; capped at 500 samples |
| `laser.turn_off_at_end` | Stop the laser at the end point | Defaults to `true`; set `false` for one-row legacy tables |
| `laser.plume_radius_m` | Override RFPLUME: the plume cylinder radius that engages the plume model | Optional; positive, and requires a non-zero `laser.plume_attenuation`. With the profile default `rfplume=0` the model never engages. When engaged, energy audits show the diverted power leaves the energy balance: the melt receives ~aplume less beam power. Retired from all validated recipes |
| `laser.plume_attenuation` | Override APLUME: fraction of beam power removed by the plume model | Optional; `0 <= value < 1`. Profiles ship `0.0`. Acts as a calibrated beam-power sink (a near-uniform depth cut in matched-pair energy audits). Retired from all validated recipes; kept for future use |
| `laser.plume_condensation` | Override ACONDENS, documented by the vendor as recondensing the plume-diverted power | Optional; `0 <= value <= 1`, requires a non-zero `laser.plume_attenuation`. Measured ablations return nothing at either setting, so APLUME is a loss regardless. Profile default 1.0 |
| `laser.heat_distribution` | `uniform` or `gaussian` | Defaults to `gaussian` (keyhole model) |
| `laser.beam_profile` | Optional multi-beam radial approximation for ring/tabular profiles | Object; `type` is `single`, `ring`, `radial_table`, or `tabular`. Expands to co-moving beamlets that conserve total `power_w`; currently straight/dwell paths only |
| `laser.beam_profile.samples` | `radial_table`/`tabular` radial shells | Non-empty list of `{radius_fraction, weight}` objects; `radius_fraction` in [0,1], weights non-negative and positive in total |
| `laser.beam_profile.angular_samples` | Number of azimuthal beamlets for a ring/shell | Integer 3-64; may be overridden per radial-table sample |
| `laser.beam_profile.beamlet_radius_fraction` | Radius of each beamlet as a fraction of `beam_radius_m` | Positive and <=1; beamlet radius plus radial offset must fit within the nominal beam radius |
| `laser.beams` | Independent beams (`nlbeam` up to 4), each `{path, power_w?, scan_speed_m_per_s?, turn_off_time_s?}` with its own path on one shared time grid | List of 1-4 objects; `path` is required per beam and uses the `laser.path` types; cannot be combined with `laser.path` or `laser.beam_profile`; top-level `power_w`/`scan_speed_m_per_s` are the per-beam defaults |
| `laser.fresnel_from_material` | Derive the Fresnel coefficient ε (`ereflct`) from the material's optical data | Defaults to `true` when the material has optical data (resistivity CSV or absorptivity); prefers `optical.electrical_resistivity_ohm_m` (Drude) then `optical.normal_absorptivity`. Set `false` for the fixed scalar reflectivity |
| `laser.wavelength_m` | Laser wavelength for the Drude Fresnel derivation | Defaults to `1.064e-6` (fiber/Nd:YAG) |
| `laser.radiation_distribution_flag` | `ilbeam`: FLOW-3D beam radiation distribution option | Integer; defaults to `2`; rendered only by the `standard` solver profile |
| `laser.reflection_flag` | `irefopt`: beam reflection model | Integer; defaults to `3` (Fresnel, the model `fresnel_from_material` feeds) |
| `laser.max_reflections` | `nrefmx`: ray-trace reflection count cap | Integer; defaults to `20` |
| `laser.reflection_limit` | `qreflim`: fraction of beam power below which a reflected ray is dropped | Positive; defaults to `0.01` |
| `laser.absorptivity_flag` | `icbmengy`: absorptivity option | Integer; defaults to `0`; rendered only by the `standard` solver profile |
| `laser.absorptivity` | `cbmengy`: fixed beam absorptivity | Non-negative; defaults to `1.0` (full absorption; the Fresnel model then sets the effective value) |
| `laser.reflectivity` | `ereflct`: Fresnel coefficient ε when `fresnel_from_material` is off | Non-negative; defaults to `0.35`; ignored when the material supplies the coefficient |
| `laser.vapor_pressure_a` | `avprslbm`: vapour-pressure curve coefficient A | Positive; defaults to `50650.0` |
| `laser.vapor_pressure_b` | `bvprslbm`: vapour-pressure curve coefficient B | Positive; defaults to `10.95913` |
| `simulation.initial_temperature_k` | Initial metal and gas temperature (`tempi`/`tvoid`) | Positive; required |
| `simulation.finish_time_s` | End time | Must include the scan and the stop event |
| `simulation.output_frames` | Number of exported spatial (selected-data) frames written to the result file | Positive integer; works with every solver profile (when set, it replaces the keyhole/legacy profiles' fixed `tapltd(1)=5e-06`); omit to keep the profile's interval or the FLOW-3D default (~100). Renders `tapltd(1) = finish_time_s / output_frames` |
| `simulation.boundary_temperature_k` | Boundary temperature | Defaults to `initial_temperature_k` |
| `simulation.accommodation_coefficient` | Override RSIZE, the evaporation accommodation coefficient | Optional; 0 < value <= 1. Scales the phase-change rate and the recoil amplitude (`A = RSIZE*PV1`). Profiles hardcode 1.0; literature 0.1-0.5. Ablations show it is self-compensating and saturating, so it cannot serve as a depth-calibration lever; keep the default |
| `simulation.recoil_model` | `if_prsrecoil`: recoil pressure from the vapour-pressure coefficients (`1`) or the evaporation mass flow (`2`) | Optional; `1` or `2`; the rendered default is `1` |
| `simulation.void_vapor_pressure_pa` | `pv0`: vapour partial pressure in void regions | Optional; non-negative. Profile default is `101300` (`standard`) or `50650` (`keyhole`, `legacy_case_fine_1`) |
| `simulation.ambient_pressure_pa` | Pressure on the five open mesh boundaries (`pbct(1,1..5)`, plus `pbct(1,6)` and `p_reference` when set) | Optional; positive; the template default is `101300.0` |
| `simulation.solver_profile` | `keyhole`, `standard`, or `legacy_case_fine_1` | Defaults to `keyhole` (case_fine_1 keyhole tuning, keeps surface tension); `legacy_case_fine_1` zeroes surface tension for the line-track reproduction |
| `simulation.subprocess_id` | FLOW-3D subprocess number | Defaults to `0`; legacy line-track reproduction uses `1` |
| `simulation.max_iterations` | Solver safety cap | Positive integer |
<!-- END GENERATED CONFIG SCHEMA -->

Optional `boundary_conditions` can set each side to `wall` or `pressure`:

```json
"boundary_conditions": {
  "x_min": "wall",
  "x_max": "wall",
  "y_min": "wall",
  "y_max": "wall",
  "z_min": "wall",
  "z_max": "pressure"
}
```

The laser velocity is derived from the end points and the scan speed:

```text
scan_time = norm(end_point_m - start_point_m) / scan_speed_m_per_s
velocity  = (end_point_m - start_point_m) / scan_time
```

The template writes a time table with that constant velocity from `t=0` to
`scan_time`; a tiny positive relative time step later it sets laser power and
velocity to zero. This keeps the path linear and prevents the adaptive final
solver step from moving the beam beyond the requested end point.

Every generated case includes `case_manifest.json`. It stores the original
JSON plus derived domain bounds, beam origin, velocity, travel length, travel
time, nominal cell counts, material data, and parallel allocation. Inspect it
before launching a long run.

After a solver run the manifest also carries a `run` block with the solve's own
timing, merged in from `runhyd.log` when `summary.md` is written:
`elapsed_seconds`, `cpu_seconds`, `final_time_s`, `cycles`, `elapsed_human`,
`completion`, `completion_datetime`, and `result_size`. A prepare-only case has
no `run` block. This is the one timing record that travels with the manifest
when a case is copied into a training dataset; `summary.md` and a batch's
`logs/status.tsv` stay behind in the run directory.

### Curved and scripted scan paths

For anything other than a single straight scan, add a `laser.path` object.  It
replaces `start_point_m`/`end_point_m`.  FLOW-3D integrates the beam origin along
a velocity time table (`utlbm`/`vtlbm`/`timlbm`), so a path is just that table
with more rows.  For the constant-speed types (`waypoints`, `sinusoidal`,
`random`) the beam moves at the constant `scan_speed_m_per_s`: the path is
resampled at equal arc length, which keeps the speed constant, and one table
row is written per sample.  The `variable_speed` type instead sets the timing of
each leg explicitly, so the speed varies along the path (and a leg can dwell).
No template edit is involved.

A hard check rejects the run if any sampled point comes within four beam
radii of a domain edge, so the beam and its keyhole never run off the plate.
Because a path can head in any direction, `laser.path` uses `centered` placement
(or your explicit `geometry.bounds_m`); `laser_leading_margin` is refused.  The
manifest records `laser_path_type`, `laser_path_interpolation`,
`laser_path_num_samples`, and `scan_path_length_m` for inspection.

Four path types are supported:

- **`waypoints`**: an ordered `points_m` time series of `[x, y]` surface
  points, joined either `linear` (a polyline) or `spline` (a Catmull-Rom curve
  through the points):

  ```json
  "path": {
    "type": "waypoints",
    "interpolation": "spline",
    "points_m": [[-0.0012, 0.0], [-0.0004, 0.0005], [0.0005, -0.0005], [0.0012, 0.0]]
  }
  ```

  A `spline` bows slightly past its control points, so leave headroom against the
  4-radius keep-out (these points need a domain of roughly 4 mm x 3 mm or larger).

- **`sinusoidal`**: a sine weave travelling along +x
  (`configs/rectangular_curved_path_sinusoidal.json` is a ready example):

  ```json
  "path": {
    "type": "sinusoidal",
    "start_point_m": [-0.0013, 0.0, 0.0],
    "length_m": 0.0026,
    "amplitude_m": 0.0004,
    "periods": 2
  }
  ```

- **`random`**: a smooth, seeded random walk that fills the cropped region
  inside the 4-radius keep-out.  It starts at the region's left edge (mid height)
  and steers each step toward the least-visited neighbourhood, so it spreads over
  unexplored area; visitation is averaged over in-region cells, so the forbidden
  margin outside does not bias it to hug the boundary.  It walks until it reaches
  `length_m` exactly, peeling along walls where it would dead-end and
  self-intersecting if the domain is smaller than the requested length (allowed
  for now).  `max_turn_deg` caps the per-step turn in open space; recovering off a
  wall may turn sharper.  Same `seed` reproduces the same path exactly:

  ```json
  "path": {
    "type": "random",
    "seed": 1,
    "length_m": 0.01,
    "step_m": 0.0006,
    "max_turn_deg": 45
  }
  ```

- **`variable_speed`**: an explicit position-time trajectory.  Unlike the other
  types (one constant `scan_speed_m_per_s`), the beam speed varies leg to leg:
  `knots_m` are the beam positions verbatim (no arc-length resample) and
  `times_s` is the arrival time at each knot, so each leg runs at
  `chord / Δt`.  A repeated knot with an advancing time is a true dwell (the
  beam holds position, `u = v = 0`).  Optional `power_w` gives one power per leg
  (to de-rate a dwell); it defaults to `laser.power_w` everywhere.  The manifest
  adds `laser_path_variable_speed`, `laser_path_times_s`, and
  `laser_path_leg_speeds_m_per_s`.  A fixed laser deposits energy fast, so keep
  dwell power and duration bounded (for example by absorbed energy and by the
  King–Rubenchik enthalpy).

  ```json
  "path": {
    "type": "variable_speed",
    "knots_m": [[-0.0006, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0006, 0.0002]],
    "times_s": [0.0, 0.0008, 0.0011, 0.0019]
  }
  ```

`sample_spacing_m` (default `beam_radius / 2`) sets how finely the path is
sampled into the velocity table; each sample becomes a constant-velocity leg, so
the beam runs at exactly `scan_speed_m_per_s` throughout. Samples are capped at
500, so a long path may need a coarser spacing. The beam follows the sampled
polyline, so curves finer than the spacing are approximated (see
[assumptions.md](assumptions.md), A5b).

### Independent beams

`laser.beams` runs up to four independent beams (`MAX_INDEPENDENT_BEAMS`) in one
case, each with its own path, power and optional turn-off time. It replaces
`laser.path` and cannot be combined with `laser.beam_profile` (that splits one
beam into co-moving beamlets; this adds separate sources). Each entry is
`{path, power_w?, scan_speed_m_per_s?, turn_off_time_s?}`; `path` takes the same
types as `laser.path`, and the top-level `power_w`/`scan_speed_m_per_s` are the
defaults for beams that omit them. Paths may cross; nothing keeps them apart.

```json
"laser": {
  "power_w": 100.0,
  "scan_speed_m_per_s": 0.2,
  "beam_radius_m": 0.00005, "focal_distance_m": 0.01, "lens_radius_m": 0.0003,
  "beams": [
    {"power_w": 100.0, "path": {"type": "random", "seed": 1, "length_m": 0.003}},
    {"power_w": 140.0, "path": {"type": "random", "seed": 2, "length_m": 0.002}}
  ]
}
```

FLOW-3D indexes `powlbm`/`utlbm`/`vtlbm` by `(row, beam)` but `timlbm` by row
alone, so every beam is rendered on one shared time grid: each beam's transition
times are unioned and the beams are held piecewise-constant across the union,
which reproduces each beam's plateaus exactly. The 500-row table cap is shared,
so per-beam paths are thinned to fit (`samples_used` against `num_samples` in
the manifest says by how much). The manifest's `derived.beams` records each
beam's `path_xy_m`, power, speed and duration; `tests/test_multi_beam_paths.py`
and `tests/test_per_beam_path_recorded.py` cover the rendering and the record.

### Legacy line-track reproduction

`configs/case_fine_1_10um_reproduction.json` reproduces a 70 W,
0.711 m/s line-track case of an earlier solver setup on a 10 µm mesh. It uses:

- derived dimensions of `0.9 mm × 0.4 mm × 0.3 mm` plus `0.1 mm` gas
  clearance;
- `placement: "centered"`, which derives mesh bounds of
  `[-0.45, 0.45] mm × [-0.20, 0.20] mm × [-0.30, 0.10] mm`;
- the default `legacy_halfspace` initialization;
- `legacy_case_fine_1` FLOW-3D WELD defaults;
- one rank with 16 OpenMP threads, matching the old block distribution;
- a `0.00084 s` scan, matching the reference VTK duration.

The most important legacy detail is the z-max pressure opening. The old case
does not put a fixed-temperature top boundary into the gas layer. If that
boundary is written as a normal `tbct(1,6)=300` pressure boundary, FLOW-3D
creates a hot solid-like top layer. The laser ray then records heat flux but
does not couple correctly into the substrate. The legacy profile leaves that
top opening clear.

## Material files

Each material is a directory under `materials/`: `materials/<name>/<name>.json`
holds the scalars + the `optical` block, and one CSV per temperature-dependent
property sits alongside it. A case selects a material by name:

```json
"material": { "name": "ss316l" }
```

or by file (point at the `<name>.json` inside the directory):

```json
"material": { "file": "materials/inconel718/inconel718.json" }
```

The short form also works (`"material": "alsi10mg"`). Name lookup is
case-insensitive, so `"TI_IMI318_MILLS"` resolves to `materials/ti_imi318_mills/`.
If `material` is omitted, the runner uses `materials/ti_imi318_mills/`.

Material directories (each `materials/<name>/`): `ti_imi318_mills` (Ti-6Al-4V
Mills data, default), `ti64_v3_calibrated` (the Ti-6Al-4V of the paper's simulations),
`ti64_v4_measured`, `ti64_v5_tv3420_measured`, `ss304`, `ss304l`, `ss316l`, `inconel718`,
`alsi10mg`, `al6061`, `pure_copper`, `cocr`.

Each material directory uses SI units and contains:

- `<name>.json`: `schema_version: 1`, `material_name`, `flow3d_label`, scalar
  `properties`, and an optional `optical` block (`normal_absorptivity`,
  `prefer_resistivity`);
- one CSV per temperature-dependent table (`temperature_k,<property>` header):
  `density_kg_m3.csv`, `specific_heat_j_kg_k.csv`, `thermal_conductivity_w_m_k.csv`,
  `dynamic_viscosity_pa_s.csv`, `surface_tension_n_m.csv`, and
  `electrical_resistivity_ohm_m.csv` (drives the Fresnel ε(T); see
  `laser.fresnel_from_material`).

Provenance and confidence for every value are tracked in
[materials/sources.md](materials/sources.md). The material data are seed values
for reproducible setup and testing and form no calibrated database. Verify them against
the alloy heat, feedstock state, and temperature range before using results as
process data.

## Parallel allocation

The runner can allocate more than one local core:

```bash
./bash/run_rectangular_case.sh configs/rectangular_moving_laser_melt.json \
  --case-name melt_two_core_check \
  --cores 2
```

`--cores` overrides `parallel.cores` for one run. `--threads-per-rank`
overrides `parallel.threads_per_rank`.

The allocation model is:

```text
ranks = cores / threads_per_rank
```

For the common local case, leave `threads_per_rank` at `1`, so `cores` equals
the number of MPI ranks. The generated `machines` file repeats the local
hostname once per rank.

For one rank, the runner uses the pre-split one-block input. For more than one
rank, it first calls FLOW-3D's decomposition step:

```text
runpre_par -9 <ranks> <threads_per_rank> <case>
runhyd_par <case>
```

This creates `runpre_decompose.log` before `runhyd.log`. Solver runs that share a
licence are serialised by a lock, so they run one at a time: give a run the cores
it needs on the machine running the solver, and do not overlap cases. The [task spooler](#task-spooler) lines runs up and
executes them one after another.

## Provenance-tracked runs (aiida-pino)

The sibling package `../aiida-pino` wraps this runner
(`python/run_rectangular_case.py`) as provenance-tracked AiiDA jobs, files each
batch into a named dataset, and applies resource caps; its own README covers
installation and use. Nothing in this README needs it: the direct runner, the
batch generator and the task spooler are the documented paths, and they are what
the AiiDA jobs call underneath. Two of its conventions show up in
the run outputs:

- an AiiDA-launched solve takes the same `~/.flow3d/spool/.seat.lock` as a
  direct run, so it waits while another solve holds the lock, and the licence
  endpoint still comes only from the job's environment (`F3D_LICENSE`);
- `python/score_anchors.py` can read a run whose NPZ was reclaimed from disk
  back out of the AiiDA store when the `aiida` package and a profile are
  present; `--no-aiida` restricts it to exports on disk, and without `aiida`
  installed it does that on its own.

## Reproducing the experimental validation

`validation/` is the reproducibility pipeline for the experimental validation
campaign reported in the manuscript's supplementary information. It contains
every campaign config (`validation/configs/`; `validation/MANIFEST.md` is the
case-by-anchor inventory), the
frozen material files they reference (relative paths, self-contained), the
measured targets (`validation/anchors/`), the acceptance criteria
(`ACCEPTANCE.md`), the scoring plan (`scoring_plan.json`) and a driver:

```bash
python3 validation/reproduce.py             # no license: render + verify all inputs
python3 validation/reproduce.py --solve     # with a license: re-solve the campaign
python3 python/score_anchors.py --plan validation/scoring_plan.json \
  --runs runs/validation --no-aiida         # score the solved tree against the anchors
```

Scoring needs nothing outside this repository: `score_anchors.py` joins the
exported runs to the anchors with the per-anchor flags recorded in
`scoring_plan.json` (see [Scoring runs against published
measurements](#scoring-runs-against-published-measurements)). The manuscript's
supplementary notebook (`experimental_validation.ipynb` in the
`lpno` sibling repository) presents the same rows and reads the same
tree layout, `<root>/<case_name>/npz/<case_name>.npz`, via `F3D_RUNS_ROOT`. The
solved NPZ exports are solver output and are not distributed with this
repository (`runs/` is ignored); without a FLOW-3D licence the campaign can be
rendered and verified (`reproduce.py`) but not re-scored.

### Nested mesh refinement

Halving `mesh.cell_size_m` everywhere multiplies the cell count by 8 in 3-D, and
the CFL condition can double the number of time steps on top; a matched 10/5 µm
pair of runs took about 12x the CPU time. `mesh.refinement` pays that cost only
where the melt is:

```json
"mesh": {
  "cell_size_m": 1.0e-5,
  "refinement": { "factor": 2, "margin_m": 2.0e-4 }
}
```

That renders a second `&mesh` block at 5 um spanning every laser-track position plus
one beam radius, grown by `margin_m`, snapped outward onto parent cell faces (an
interface cutting through a parent cell is not a legal nesting) and clipped to the
domain. In z it runs from `margin_m + beam_radius_m` below the free surface to
`margin_m` above it. The bounds and cell counts land in
`case_manifest.json` under `derived.mesh_refinement`.

Two constraints, both enforced with a clear error:

- **Single rank only.** Every mesh block is also decomposed per rank, and the exporter
  cannot distinguish a decomposition brick from a nested block. Use
  `--cores N --threads-per-rank N`.
- **The fine block cannot cover the whole domain**, since that is uniform refinement
  with extra steps; lower `mesh.cell_size_m` for that.

**The exported NPZ is the fine block alone.** Blocks at two spacings do not tile one
grid, so `npz_block_merge.select_finest` keeps the finest and drops the coarse far
field (resampling it would invent data at the interface and make the archive
lossy). Everything the metrics read (fusion depth and width, vapour depression) is
measured inside the fine block, so choose `margin_m` from the expected melt
depth: a pool deeper than
`margin_m + beam_radius_m` would leave the fine block and be measured truncated.

Try it without a license:

```bash
python3 python/run_rectangular_case.py configs/rectangular_moving_laser_refined_smoke.json \
  --case-name refine_prepare_check --prepare-only
```

## Batch runs

A batch is a list of run configs derived from one reference JSON, plus a
`tasks.txt` of run commands. `python/build_batch.py` expands a reference config
into one JSON per case and writes that task file. A lock serialises solver runs
that share a licence, so the cases run one at a time: queue `tasks.txt` with the
[task spooler](#task-spooler) and let it drain.

Each simulation's core count comes from its own config or `--cores` (see
[Parallel allocation](#parallel-allocation)); one run uses the whole machine and
cases do not overlap.

A batch is self-contained in the directory you pass to `build_batch.py`:
`configs/` and `tasks.txt` are written there, and each case's run folder lands
in it too (`<batch-dir>/<case_name>/`). So `runs/` does not fill with loose
cases; keep batch directories under `runs/batches/<name>/`. That directory is
untracked, so a batch whose method must outlive the machine running the solver gets a tracked
copy of its drivers, design tables, configs and round state (never its solver
output) under `batches/<name>/`; `batches/README.md` lists the
batches kept that way and what was changed to make them portable.

**Physics eras.** Batches solved before the validation campaign used a
different physics recipe from the validated one (`recoil_model: 2`,
`void_vapor_pressure_pa: 101325`, per-material tvexp and Fresnel coupling,
`materials/ti64_v3_calibrated` for Ti64, no plume; see
`validation/MANIFEST.md`). A run's `prepin.<case>` fingerprint tells the eras
apart (`score_anchors.py --recipe`). Do not mix eras in one training set
without documenting the choice; the aiida-pino datasets carry
`old_physics` / `experimentally_anchored` keywords for this.

### Building a batch

Pick exactly one way to vary parameters:

| Flag | Meaning |
| --- | --- |
| `--grid KEY=SPEC` | Cross product (a sweep). Repeatable for multiple axes. |
| `--zip KEY=SPEC` | Parallel (paired) lists; all `SPEC`s share one length. |
| `--params-tsv FILE` | One case per row; header holds dotted config keys. |

And, with any of the three:

| Flag | Meaning |
| --- | --- |
| `--derive-finish-time-from-track` | Set `simulation.finish_time_s = track_length / scan_speed` per case (fixed-length line tracks with `turn_off_at_end: false`; refused otherwise). |
| `--name-prefix P` | Prefix every generated `case_name` (e.g. `train_`). |
| `--cores N`, `--threads-per-rank N` | Written into every task line for the runner. |
| `--export {npz,vtk,vtk-legacy60,both,none}` | Export mode written into every task line. |

`KEY` is a dotted path into the config (`laser.power_w`,
`simulation.finish_time_s`). `SPEC` is a comma list (`200,300,400`) or, for
grids, the evenly spaced shortcut `lo:hi:n` (`200:400:5` → five values from 200
to 400). Example sweep:

```bash
python3 python/build_batch.py configs/rectangular_moving_laser_smoke.json runs/batches/psweep \
  --grid laser.power_w=200:400:5
# then queue the generated tasks.txt with the spooler (see Running a batch)
```

### Running a batch

`build_batch.py` writes a `tasks.txt` of complete run commands. Queue them with
the spooler and drain the queue one run at a time:

```bash
while read -r c; do bash bash/spool.sh add --cmd "$c"; done < runs/batches/psweep/tasks.txt
bash bash/spool.sh start --drain
```

The spooler moves each finished case to `~/.flow3d/spool/done/` or `.../failed/`
and appends one line per completion to `.../pool.log`. The runner refuses to
overwrite a completed result, so an interrupted batch is safe to requeue and
rerun. Each case holds the solver lock while it solves, so cases run one at a
time; the number of runs is not capped.

### Worked example: a 10 µm line-track sweep

This runs `train`/`val`/`test` sweeps of line-track cases at 10 µm, from a reference
dataset whose directories are named `POWERW_SPEEDmps`. That dataset is not distributed
with this repository; without it, write the per-split TSV of step 1 by hand. Every setting
matches `configs/case_fine_1_10um_reproduction.json` except power and scan
speed; `finish_time_s` is derived so the beam crosses the same fixed track at
each speed (the track is `turn_off_at_end=false`, so without this every other
speed would scan the wrong length; see [assumptions.md](assumptions.md) A21).

1. Turn the dataset directory names into a per-split parameter TSV
   (`POWERW_SPEEDmps` → `power_w`, `scan_speed_m_per_s = SPEED/1000`):

   ```bash
   DATA=<path to the reference dataset>
   python3 - "$DATA" runs/batches/pino <<'PY'
   import sys, re
   from pathlib import Path
   data, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
   pat = re.compile(r"^(\d+)W_(\d+)mps_line_track_vtk$")
   for split in ("train", "val", "test"):
       rows = [(m.group(1), int(m.group(2)) / 1000.0)
               for p in sorted((data / split).iterdir())
               for m in [pat.match(p.name)] if m]
       (out / f"{split}_params.tsv").write_text(
           "laser.power_w\tlaser.scan_speed_m_per_s\n"
           + "".join(f"{p}\t{v}\n" for p, v in rows))
   PY
   ```

2. Build one batch per split (finish time derived from the fixed track, one core
   per case, NPZ output):

   ```bash
   for split in train val test; do
     python3 python/build_batch.py configs/case_fine_1_10um_reproduction.json runs/batches/pino/$split \
       --params-tsv runs/batches/pino/${split}_params.tsv \
       --derive-finish-time-from-track --name-prefix ${split}_ --cores 1 --export npz
   done
   ```

3. Queue every split and drain the queue one run at a time:

   ```bash
   for split in train val test; do
     while read -r c; do bash bash/spool.sh add --cmd "$c"; done < runs/batches/pino/$split/tasks.txt
   done
   bash bash/spool.sh start --drain
   ```

Each case lands in `runs/batches/pino/<split>/<case>/`, with its NPZ at
`runs/batches/pino/<split>/<case>/npz/<case>.npz`; the spooler logs completions
to `~/.flow3d/spool/pool.log`. To cross-check a run against the reference dataset, also
export the 60-frame legacy VTK and compare:

```bash
CASE=runs/batches/pino/test/test_70W_0711mps
F3D_POST_HOME="<FLOW3D_POST install>" "$F3D_POST_HOME/bin/pvpython" \
  python/export_legacy_even_timesteps_to_vtk.py \
  "$CASE/flsgrf1.test_70W_0711mps" \
  "$CASE/paraview/legacy_even_steps" --prefix test_70W_0711mps --frames 60
python3 python/compare_legacy_vtk_dirs.py \
  "$DATA/test/70W_0711mps_line_track_vtk" \
  "$CASE/paraview/legacy_even_steps" \
  --suffix-tolerance-us 1 --stat max --rtol 0.05 \
  --array Temperature --array 'Melt Region' --array 'Solid Fraction'
```

Field means track the reference closely; instantaneous keyhole extrema
(heat-absorption, velocity) can differ more against a reference made with other
settings or another solver version. See
[potential_bugs.md](potential_bugs.md) P12.

## Task spooler

Solver runs that share a licence go one at a time, and the spooler lines them
up: queue runs at any time and one worker executes them in order. It also runs
the `tasks.txt` a batch builds (see [Running a batch](#running-a-batch)).

The queue defaults to a shared per-user directory, `~/.flow3d/spool` (override
with `F3D_SPOOL_DIR` or `--dir`), outside the repo. Any other checkout that uses
the same directory feeds the same queue, and one worker drains it, so solves stay
serialised across checkouts. The worker lock lives in that shared directory, so a
second `start` exits with "already running" and starts no concurrent solve. A
task carries an absolute config path and its own checkout's runner, so the shared
worker runs the right case wherever it was queued. The queue is just files on
disk, so it survives a reboot or a killed worker: the next `start` re-scans
`queue/` and re-runs anything left mid-flight.

```bash
# Enqueue runs whenever (args after the config pass straight to the runner):
bash bash/spool.sh add configs/rectangular_moving_laser_melt.json --cores 2
bash bash/spool.sh add configs/rectangular_moving_laser_smoke.json --case-name smoke2

# Drain the queue, one run at a time:
bash bash/spool.sh start

# See what is queued / running / recently finished:
bash bash/spool.sh status
```

`start` keeps watching for new tasks until you stop it with Ctrl-C; a second
Ctrl-C kills the run still in flight. Pass `--drain` to exit once the queue is
empty instead (one-shot drain). To keep adding jobs while it runs, put the
worker in the background with your shell:

```bash
mkdir -p ~/.flow3d/spool
nohup bash bash/spool.sh start >~/.flow3d/spool/worker.out 2>&1 &
```

Each finished task moves to `~/.flow3d/spool/done/` or `.../failed/`, its console
output goes to `.../logs/<name>.log`, and one line per completion is appended to
`.../pool.log`. Only one worker runs at a time (a lock in the shared spool
directory); a second `start` from any checkout exits with "already running".

| Command | What it does |
| --- | --- |
| `spool.sh add <config.json> [run args...]` | Queue a run; args pass to `run_rectangular_case.py`. |
| `spool.sh add --cmd "<command>"` | Queue an arbitrary shell command. |
| `spool.sh add --name <label> ...` | Label the task (default: config file stem). |
| `spool.sh start [--drain] [--poll S]` | Run the worker (one run at a time). |
| `spool.sh status` | Show queued / running counts and recent completions. |
| `spool.sh --dir <path> <command>` | Use a spool directory other than `~/.flow3d/spool` (or set `F3D_SPOOL_DIR`). |

The spooler only controls *when* runs happen; each FLOW-3D run holds the solver
lock while it solves, so the worker runs them one at a time. There is no cap on
the total number of runs.

Even a direct run that skips the queue (`bash bash/run_rectangular_case.sh …`)
now takes a shared lock (`.seat.lock` in the same spool directory) while the
solver is running, so it waits for any in-flight solve (queued or direct, from
any checkout sharing the spool directory) to finish before it starts, and a second
direct run cannot start mid-solve. The lock is held only around the solver;
`--prepare-only`, rendering, and export don't take it. Routing through the
spool is therefore optional for safety: the queue lines up many runs, and the
lock guarantees that two never solve at once.

## What FLOW-3D model is generated

The generated template uses:

- analytic half-space initialization with the workpiece surface at `z = 0`
  and gas clearance above it;
- a seed one-rank `MESH_MPI` partition. Multi-rank solver runs ask FLOW-3D's
  automatic decomposition step to replace it before solving;
- energy, viscosity, surface tension, density, solidification and phase-change
  models enabled;
- a single top-down, ray-traced circular laser applied to all facing surfaces,
  Gaussian by default (`heat_distribution`). The beam is traced into the keyhole
  and reflected up to `max_reflections` times (see ray tracing below). This
  starts heating from an initially solid substrate;
- material scalar values and temperature tables rendered from the selected
  material JSON file.

By default the runner uses the **`keyhole` solver profile**: the case_fine_1
keyhole tuning (gaussian beam, ray tracing with 20 reflections, full absorption,
recoil pressure on, plume off) with the material's surface tension kept on.
The profile's own defaults render the legacy analytic recoil and
`pv0=50650`; the experimentally validated recipes additionally
set `simulation.recoil_model: 2`, `simulation.void_vapor_pressure_pa: 101325`,
and for Ti64 the calibrated material `materials/ti64_v3_calibrated`; the
campaign behind them is in `validation/` (`MANIFEST.md`, `ACCEPTANCE.md`). Set `simulation.solver_profile` to
`legacy_case_fine_1` only for the line-track reproduction (it zeroes surface
tension to match an older reference), or `standard` for the lighter earlier
tuning.

Ray tracing: `&weld` sets `if_beam=1`, `irefopt=<reflection_flag>` (3),
`nrefmx=<max_reflections>` (20) and `qreflim=<reflection_limit>` (0.01); the beam
is split into rays that reflect inside the keyhole up to `nrefmx` times or until a
ray drops below `qreflim` of its energy, with absorbed energy deposited as a
surface flux (`ibeamsurfflux=1`, no volumetric penetration). `irefopt=3` selects
FLOW-3D's simplified Fresnel model, so the Fresnel coefficient `ereflct` is the
model's ε, which differs from a plain reflectivity. ε is derived per material
by default whenever the material carries optical data
(`laser.fresnel_from_material`, default `true` in that case; set `false` for the
fixed scalar), preferring
`optical.electrical_resistivity_ohm_m` via the Drude relation ε = √(2·ε₀·ω·ρ),
ω = 2πc/`laser.wavelength_m` (default 1.064 µm), and falling back to inverting
`optical.normal_absorptivity`. Provenance and confidence of those optical values
are tracked in [materials/sources.md](materials/sources.md). Each material's
`optical.prefer_resistivity` flag picks the more accurate route (default `true`;
`false` for Cu/Al, where the DC-Drude form overestimates, so they use measured
absorptivity); the runner logs the chosen value and the alternative, and the
manifest records both. These are model choices; validate them against the
target process before interpreting a result.

`laser.beam_profile` approximates radial beam shapes by expanding one laser into
multiple co-moving indexed FLOW-3D beams while conserving total `power_w`.
`type: ring` places equal-power beamlets around one radius; `type:
radial_table`/`tabular` accepts radial shells as `{radius_fraction, weight}` and
samples each non-central shell azimuthally. This is intended for straight or
dwell cases; generalized path tables still use the single-beam representation.

## Running and checking a case

The shell entry point calls the standard-library Python runner. The preferred
direct invocation reads the license endpoint from the local environment:

```bash
export F3D_LICENSE='<your-license-endpoint>'
python3 python/run_rectangular_case.py configs/rectangular_moving_laser_smoke.json \
  --case-name my_first_rectangular_case
```

The runner also accepts `--license PORT@HOST`, but `F3D_LICENSE` or ignored
`license.env` is safer because command-line arguments can leak into shell
history and process listings. The runner sources `flow3dvars.sh`, applies the
local license endpoint after sourcing it, writes the local `machines` file, and
calls the vendor launchers. One-rank runs call `runhyd_par` directly.
Multi-rank runs call `runpre_par -9` first, then `runhyd_par`.

Key output files are:

| File | Purpose |
| --- | --- |
| `prepin.<case_name>` | Exact generated FLOW-3D input, in SI units |
| `case_manifest.json` | Input and derived geometry/laser quantities (includes `laser_path_xy_m`); gains a `run` block with the solve's wall-clock/CPU time after a run |
| `path.png` | Plot of the in-plane scan path (domain, 4-beam-radius keep-out, start/end), written at render time |
| `summary.md` | Human-readable run summary (status, runtime, mesh/laser/material, NPZ result stats), written after a solver run |
| `machines` | Local host slots used by FLOW-3D MPI launch scripts |
| `runpre_decompose.log` | Decomposition log for multi-rank runs |
| `runhyd.log` | Preprocessor and solver log; use this to check completion |
| `export_npz.log` | Automatic all-time-step NPZ export log (default export) |
| `export_vtk.log` | All-time-step VTK/PVD export log (only with `--export vtk`/`both`) |
| `hd3err.<case_name>` | Solver errors, if any |
| `flsgrf1.<case_name>` | Native FLOW-3D result database; deleted automatically after a successful NPZ export (see below) |
| `flsgrf1000.<case_name>` | Initial restart snapshot; not the completed result |

Normal completion is reported near the end of `runhyd.log` as
`normal completion: finish time exceeded`. A nonzero `hd3err.*` file or a
missing `flsgrf1.*` result means the case did not complete. Once the run has
finished and `flsgrf*` has been discarded, use `runhyd.log` and `summary.md` for
that check.

### Disk hygiene: the free-space floor and flsgrf cleanup

`flsgrf1.<case>` is the bulk of a run: 0.7 to 3.6 GB per case for the production
batches, against roughly 30 MB for the NPZ. Nothing downstream reads it: `summary.md`
is built from `runhyd.log`, and the ML datasets ship `npz/<case>.npz` plus
`case_manifest.json`. So the runner handles both ends of the problem itself, on
every path that shells out to it (direct runs, `bash/run_rectangular_case.sh`,
batch wrappers, the task spooler, and the aiida-pino `Flow3DRunCalculation`):

- **Before solving** it waits for at least `--min-free-gb` free (default 30, or
  `F3D_MIN_FREE_GB`; `0` disables), polling for up to `--min-free-wait-s`
  (default 1800, or `F3D_MIN_FREE_WAIT_S`; `0` fails fast) before giving up. This
  matters because a full disk does not fail cleanly: FLOW-3D blocks mid-write,
  `sim_time` stops advancing, and only a watchdog (below) ever ends the case.
  It waits because pressure is normally transient: another case is mid-solve and
  drops its `flsgrf` as soon as it exports. A gate that refuses at once can fail
  every case of a queued sweep during a short dip.
- **Before solving** it also waits for `--min-free-mem-gb` of `MemAvailable`
  (default 25, or `F3D_MIN_FREE_MEM_GB`; `0` disables), since low available RAM is
  what makes a machine stop answering SSH logins. `--max-load` (or `F3D_MAX_LOAD`)
  additionally waits on the 1-minute load average, but is off by default:
  `load1` lags by a minute and a running 10-core sweep keeps it above the usual
  threshold, so enabling it by default would stall back-to-back solves.
- **Before solving** it blocks while `~/.flow3d/spool/PAUSED` exists, so one file
  pauses spool runs, AiiDA runs and direct runs alike (see "Task spooler").
- **Before solving** it raises its own niceness to `--nice` (default 10, or
  `F3D_NICE`; `0` disables) so the solver and its MPI ranks yield CPU to `sshd`
  and to anything else sharing the machine. It only ever raises niceness. This
  used to require wrapping the call in `systemd-run --nice=10` by hand, which
  the spool worker and the AiiDA path did not do, so queued solves ran at
  whatever priority they inherited.
- **After exporting** it deletes `flsgrf*` and prints the space reclaimed, but
  only once `npz/<case>.npz` opens as a complete archive. A missing or truncated
  export leaves the result in place, because it is then the only copy of a
  completed solve.

Pass `--keep-flsgrf` to keep the result database if you plan to re-export later
(VTK, different variables), which requires it. `--export vtk`
or `--export none` writes no NPZ, so the result is kept regardless.

For runs made before this was automatic, `python/reclaim_flsgrf.py <dir>` reports
what the same rule would delete and `--apply` deletes it;
`bash/finalize_run.sh <case> [--max-frames N] [--keep-flsgrf]` does one case at a
time, bundled with NPZ thinning.

**Guards are warnings unless enforced.** The disk floor, the `MemAvailable`/load
gate and the NPZ exporter's scratch-space check all route their refusal through
`python/resource_guards.py`, whose mode is `F3D_RESOURCE_GUARDS`: under `warn` a
low reading prints one `[resource-guard] WARNING: ... -- not enforced` line and
the run proceeds at once (no waiting); under `enforce` the run waits for the
floor and then refuses, as described above. Memory and disk limits are
hardware-dependent, so this repository ships with `warn` as
the default (`DEFAULT_MODE` in that file). On a shared machine, where a full disk
stalls the solver mid-write and low available RAM can lock out SSH, `enforce` is
the safer setting. Set the variable in the environment of the process that
runs the solve to change either. The solver lock, the pause switch and the flsgrf
cleanup are not resource guards and are never turned into warnings.

### Optional watchdog: `monitor_run.py`

The runner itself never aborts a solve. `python/monitor_run.py` is a separate
watchdog that reads the solver's own progress file and SIGKILLs the solve
subtree when `sim_time` stops advancing or the time step collapses:

```bash
python3 python/monitor_run.py runs/<case>/hd3msg.<case> \
  --until-pid <pid of the run_rectangular_case.py process> --stall-s 900
```

Point it at `hd3msg.<case>`; `runhyd.log` is a block-buffered stdout capture
that lags and reports false stalls. Defaults: abort after 900 s
without progress (`--stall-s`), or when `delt` falls below 1e-9 s past cycle 100
(`--delt-min`, `--min-cycle`); `--max-solve-s` (0 = off) caps the wall time
after the first output; `--until-pid` (the runner process, whose subtree is the
solver) makes the monitor exit with the run and is also the safety: it kills
only the subtree of a pid whose command line is still a `run_rectangular_case`
invocation, so a stale monitor cannot take out another case. A batch wrapper can
run it beside every solve; `tests/test_monitor_run.py` covers the decision rules.

## Result export

FLOW-3D result files use the proprietary FLSGRF format. The locally installed
FLOW-3D POST package supplies the compatible reader and `pvpython` runtime.

Every successful JSON-launched case exports one compact NPZ by default to
`runs/<case>/npz/<case>.npz`. The maintained case runs on a single, static,
rectangular mesh block, so its result is already a regular grid: the exporter
reads that grid straight from the FLSGRF reader (no isosurface, no resampling),
strips the reader's one-cell ghost shell, and stores every stored cell, field,
and time step. It keeps the laboratory reference frame and applies no time
averaging, so no information is lost. The whole
`rectangular_moving_laser_melt_v5`-sized run fits in tens of MB, against several
GB for the per-step VTK series.

The `--export` options are:

| Option | Output |
| --- | --- |
| `npz` (default) | One `runs/<case>/npz/<case>.npz` for the whole run |
| `vtk` | Full 3D volume VTK/PVD for every stored step, `runs/<case>/paraview/all_steps/<case>.pvd` |
| `vtk-surface` | `Solidified Liquid` isosurface VTK/PVD for every stored step, `runs/<case>/paraview/all_steps_surface/<case>.pvd` |
| `vtk-legacy60` | Old 60-frame `Fluid` isovolume VTK sequence |
| `both` | NPZ and VTK |
| `none` | No export |

Every VTK mode carries both the exact cell-centered fields and interpolated
point data (FLOW-3D writes cell data; the exporter adds a cell→point pass so
ParaView's `Contour` and other point-based filters work without a manual
`Cell Data to Point Data` step).

### NPZ layout

The single `<case>.npz` keeps the lab-frame regular grid with axis order
`(x, y, z)` and FLOW-3D field names verbatim:

| Key | Shape | Meaning |
| --- | --- | --- |
| `x`, `y`, `z` | `(nx,)`, `(ny,)`, `(nz,)` | cell-centre coordinate axes in metres |
| `time_s` | `(T,)` | stored solver times |
| scalar field | `(T, nx, ny, nz)` | e.g. `Temperature`, `Solid Fraction`, `Melt Region`, `Pressure` |
| `Velocity` | `(T, nx, ny, nz, 3)` | velocity vector components |
| `power_w`, `scan_speed_m_per_s`, `case_name` | scalar | provenance |
| `frame_stride` | scalar | every n-th stored frame was exported (`--frame-stride`, default 1) |

The per-component `X-velocity`/`Y-velocity`/`Z-velocity` arrays are dropped
because they are redundant with `Velocity`. Check the export numerically with:

```bash
python3 python/validate_npz_series.py \
  runs/<case>/npz/<case>.npz \
  --initial-temperature-k 293.15 --min-temperature-rise-k 100 --require-melt
```

`--check-initial` additionally fails the export unless frame 0's median metal
temperature is within `--initial-tolerance-k` (default 25 K) of
`--initial-temperature-k`; the runner passes it after every solve, so a
renderer that drops the configured initial temperature is caught on the first
case of a sweep.

The layout matches what the `lpno` VTK datasource
(`custom_utils/openFoam_datasource.py`, used for both OpenFOAM and FLOW-3D)
produces with `time_axis=True, moving_ref_frame=False, flip_axes=False` (one
simulation per file), so the two pipelines stay interchangeable. To confirm a
run agrees with a reference NPZ from that datasource:

```bash
python3 python/compare_npz_to_reference.py <case>.npz <reference>.npz
```

Means are checked tightly and instantaneous extrema loosely (`--rtol-extrema`,
default 0.25), both relative to each field's dynamic range. Two exports from this
pipeline agree to rounding (the single-rank solver is deterministic, A23), so use
a tolerance near zero for a regression check; see
[potential_bugs.md](potential_bugs.md) P12.

### Melt-pool geometry metrics

`python/melt_metrics.py` is the single definition of every geometric metric used
to compare a run against a published measurement. Import it to avoid
re-deriving the arithmetic:

```bash
python3 python/melt_metrics.py runs/<case>/npz/<case>.npz
```

```python
from melt_metrics import fusion_geometry, depression, domain_validity, score_point
```

| function | what it returns |
|---|---|
| `fusion_geometry` | time-**union** melt depth and width at mid-track (what cross-section metallography sees) |
| `depression`, `depression_series` | steady-window vapour-cavity depth (mean ± sd) and width, both from the **same** window (what in-situ X-ray sees) |
| `domain_validity` | floor temperature and melt-vs-domain span; a clipped case counts for nothing |
| `score_point` | signed % error against one measurement, raising `DepthKindMismatch` if a fusion depth is compared against a depression depth |
| `ladder_summary` | pass count **and** mean signed error, flagging a one-sign offset even when every point passes |

The definitions and the ±30% acceptance band (`melt_metrics.DEFAULT_BAND`) are
stated and justified in `validation/ACCEPTANCE.md`; the band comes from the
source measurements' own scatter. The calibration gates, registered before
the cases were solved, used tighter bands for particular anchors (±15% on two
points of the Gan Ti64 power series, ±20% on the IN718 keyhole blind test), so a
verdict quoted from a gate names its band: the IN718 keyhole depth error of
−25.2% is inside 30% and outside the stricter 20% gate. Fusion and depression
depths are different physical surfaces and should not be compared with each other.

### Scoring runs against published measurements

`python/score_anchors.py` joins exported runs to the measured targets in
`validation/anchors/`, so a reported comparison can be regenerated from the
runs:

```bash
python3 python/score_anchors.py validation/anchors/galiegue2025_ti64_targets.json \
  --runs runs --sim-kind fusion --recipe tv1=3520
```

It prints one row per matched condition (our value, the measured value, signed %
error for depth and width, domain validity, cell size and the run's physics
fingerprint), then a summary giving the pass count and the mean signed error,
flagging a one-sign offset even when every point is inside the band.

Cases are found by their `case_manifest.json`, and their arrays come from
`<case>/npz/` when it is still there and from the simulation's stored payload in
AiiDA when it is not. Reclaiming disk can move exports off the machine, and
scoring only what is on disk then silently drops those runs. A case that lives
only in the database is scored from its node, which also carries its manifest, and
is marked `source=aiida` in the row. Pass `--no-aiida` to restrict scoring to
exports on disk; the tool still runs with no AiiDA installed and no profile
configured, and then sees fewer cases.

| flag | why it exists |
|---|---|
| `--sim-kind fusion\|depression` | states which surface the runs are read as. A mismatch against the anchor's `depth_kind` raises an error, so two different physical surfaces are never compared silently. |
| `--recipe SUBSTRING` | isolates one physics recipe. Without it, runs from different recipes can be pooled, and the summary says so. |
| `--series NAME` | picks one material variant when an anchor reports several at identical conditions. Hawk's four 304L variants differ by up to 2× at 118 W, so choosing silently would change the verdict; the tool raises instead. |
| `--band`, `--r0-tol`, `--p-rtol` | acceptance band (default from `melt_metrics.DEFAULT_BAND`), spot-radius match tolerance, and the **relative** power tolerance (default 1%, because these papers quote power to about that; an absolute watt tolerance is 0.1% at 456 W but 5% at 10 W). |
| `--plan FILE` | score every anchor in a tracked plan in one command (see below). |

Matching is on power, speed and spot radius together. Spot size is the dominant
variable at fixed power, so matching on power and speed alone silently compares
different experiments. Stationary spot welds are represented as speed 0, which
matches a dwell case (`laser.dwell_time_s`); that is only unambiguous while the
anchor's dwell time is constant.

The tool also warns when several runs match one measured condition. Resolution
variants and reruns are legitimate, but counting them as independent points inflates
the sample and skews the mean.

#### Regenerating the whole table

`validation/scoring_plan.json` records, per anchor, the flags it needs and the
reason for each:

```bash
python3 python/score_anchors.py --plan validation/scoring_plan.json --runs runs
```

Each entry sets `sim_kind` (Gan is in-situ X-ray, so `depression`; the
metallography anchors are `fusion`), an optional `recipe` fingerprint substring, an
optional `series` for anchors with several material variants, and an optional
`r0_um` to pin one beam radius. That last one matters: an anchor can hold a
fixed-spot power series and a defocus spot-size series, and a summary over both
mixes two experiments.

Tests guard the plan itself, including that each entry's `sim_kind` matches its
anchor's declared `depth_kind`, and that plan mode still emits the duplicate and
mixed-recipe warnings.

Anchor files that cannot be read as (power, speed, spot) points are listed in
`score_anchors.UNSUPPORTED_SCHEMAS` with a reason, and tests enforce that list in
both directions, because an anchor skipped in silence looks identical to an anchor
with no matching runs. A bare-list anchor gets its `depth_kind` from a tracked
`<stem>.meta.json` sidecar.

### Scaling-law consistency check (Gan 2021)

For materials without experimental anchors, `python/validate_scaling_law.py`
checks a finished case against the universal keyhole scaling of Gan et al.
2021 (Nat Commun 12:2379). It computes the case's keyhole number
`Ke = ηP / ((Tv−T0)·π·ρ·cp·√(αVr0³))` from the manifest and material
properties (ρ, cp, α evaluated at liquidus when the material ships CSV
curves, which is Gan's own property convention), measures the steady-window keyhole
depth from the NPZ (deepest interior gas cell below the surface, `Fraction Of
Fluid < 0.5`, x/y wall columns excluded, mean over the last 30% of the
laser-on first 85% of frames), and compares `e/r0` with a monotone curve
fitted to the paper's 87 Supplementary Data 1 points. The deviation is a
z-score in units of the reference scatter (residual sd 0.93 in `e/r0`):
`|z| ≤ 2` PASS, `≤ 3` WARN, `> 3` FAIL (exit code 1 on FAIL).

**Scope guard.** The curve does not describe aluminium-family alloys: Gan's own
Al6061 measurements sit ~+2σ above it (a direct check scored their measured
332 µm depth at z = +3.02). A FAIL on an Al/Mg material reflects the curve, so
score those cases against an experimental anchor. The gate is meaningful for the
Fe/Ni/Ti/Co families it was fitted on.

```bash
# one case, from its run directory
python3 python/validate_scaling_law.py check runs/<case> [--eta 0.5]

# explicit parameters, no NPZ needed
python3 python/validate_scaling_law.py check --power-w 129.36 --speed-m-per-s 0.4 \
  --radius-m 4.75e-5 --material-json materials/ti_imi318_mills/ti_imi318_mills.json \
  --depth-um 85 --eta 0.5

# every case of a batch, one line each plus a pass/warn/fail summary
python3 python/validate_scaling_law.py sweep runs/batches/<batch>/runs
```

`--eta` is the net absorbed fraction. Gan's reference η is their measured,
Ke-dependent absorptance (~0.3–0.7); without `--eta` the check uses a flat
0.35 and warns, which biases Ke low for deep-keyhole cases. Two known limits:
the reference collapse is tight for the Ti64/Ti6/SS316 groups (per-group
residual sd ≤ 0.36) but Al6061 sits systematically ~+2.2 z above the pooled
curve, so high-diffusivity aluminium-class alloys tend to WARN; and a Ke
outside the reference range [0.77, 11.7] demotes PASS to WARN because the
curve is extrapolated there. `python/data/gan_scaling_reference.json` ships
only derived dimensionless pairs (Ke, e/r0); rebuild it from the (not
redistributed) SD1 spreadsheet with
`python3 python/validate_scaling_law.py build-reference <sd1.xlsx>`.

### VTK output

`--export vtk` writes full 3D `Fluid` isovolume data for every
stored step. Open the `.pvd` file in ParaView and use the time controls to
inspect heating and melting. For example, the validated melt case collection is:

```text
runs/rectangular_moving_laser_melt_v5/paraview/all_steps/rectangular_moving_laser_melt_v5.pvd
```

To export just the final solidified-liquid surface to one VTK for visualization:

```bash
POST_HOME="<FLOW3D_POST_2025R1 install directory>"
F3D_POST_HOME="$POST_HOME" "$POST_HOME/bin/pvpython" \
  python/export_fluid_to_vtk.py \
  runs/rectangular_moving_laser_melt_v5/flsgrf1.rectangular_moving_laser_melt_v5 \
  runs/rectangular_moving_laser_melt_v5/paraview/rectangular_moving_laser_melt_v5_final.vtk
```

The runner can write the surface time series directly with `--export
vtk-surface` (output under `runs/<case>/paraview/all_steps_surface/`), or you
can call the exporter by hand:

```bash
POST_HOME="<FLOW3D_POST_2025R1 install directory>"
F3D_POST_HOME="$POST_HOME" "$POST_HOME/bin/pvpython" \
  python/export_all_timesteps_to_vtk.py \
  runs/<case>/flsgrf1.<case> \
  runs/<case>/paraview/surface_steps \
  --representation surface
```

That option selects the `Solidified Liquid` isosurface at 0.5. It is useful
for checking the workpiece boundary, but it drops the interior 3D field. The
default `--representation volume` keeps the full volume and writes fields such
as `Temperature`, `Melt Region`, `Solid Fraction`, `Fraction Of Fluid`,
`Pressure`, heat input fields, and velocity components.

Because the exported VTK now carries point data alongside the cell data, you
can apply ParaView's `Contour` directly to a scalar such as `Solid Fraction` or
`Fraction Of Fluid` (no `Cell Data to Point Data` step needed). That derived
contour is good for inspection or preprocessing, but it is not guaranteed to be
byte-for-byte identical to FLOW-3D POST's built-in `Solidified Liquid` surface
object.

To keep only the final solidified state (e.g. after deleting the FLSGRF to save space), export just the last frame:

```bash
POST_HOME="<FLOW3D_POST_2025R1 install directory>"
F3D_POST_HOME="$POST_HOME" "$POST_HOME/bin/pvpython" \
  python/export_final_frame_vtk.py runs/<case>/flsgrf1.<case> \
  runs/<case>/paraview/final --prefix <case> --representation volume
```

`bash/finalize_run.sh <case> [--max-frames N] [--keep-flsgrf]` bundles this with `subsample_npz.py` (thin the NPZ to N steps) and deletes the FLSGRF.

To check exported fields numerically:

```bash
python3 python/validate_vtk_series.py \
  runs/<case>/paraview/all_steps/<case>.pvd \
  --initial-temperature-k 293.15 \
  --min-temperature-rise-k 100 \
  --require-melt
```

This reads all VTK frames listed in the PVD file and reports the maximum
temperature, maximum melt-region value, and minimum solid fraction.

For old-style VTK directories, compare frames by their microsecond filename
suffix and selected array summaries:

```bash
python3 python/compare_legacy_vtk_dirs.py <reference_vtk_dir> <candidate_vtk_dir>
```

That command is strict: frame suffixes must match exactly, and min, max, and
mean summaries must match within tight numeric tolerances. For the 10 µm
legacy line-track reproduction, FLOW-3D may save a few frames one microsecond
away from the reference. Use an explicit relaxed mean-field comparison:

```bash
python3 python/compare_legacy_vtk_dirs.py <reference_vtk_dir> \
  runs/case_fine_1_10um_reproduction/paraview/legacy_even_steps \
  --suffix-tolerance-us 1 \
  --stat mean \
  --rtol 0.15 \
  --array Temperature \
  --array 'Melt Region' \
  --array 'Solid Fraction' \
  --array 'Fraction Of Fluid' \
  --array Pressure \
  --array 'Heat Absorption Rate' \
  --array 'Heat Flux Spatial Distribution'
```

This checks all 60 frames, the same cell count, and the mean value of the key
fields. It is not a bitwise identity check.

## Assumptions and known limitations

Two files track design risk and help catch wrong results:

- `assumptions.md` lists the non-trivial assumptions and code choices that have
  a real effect on a run (coordinate convention, single straight scan, mesh
  floor, profile overrides, local-host execution, and so on). Check it first
  when a result looks wrong.
- `potential_bugs.md` lists ways the repo can mislead or fail when used outside
  what it is coded for, each with a stable P-ID, its symptom, cause, workaround
  and place in the code, and a pointer to the relevant assumption.
  `docs/developer_notes.md` lists possible extensions.

Assumptions describe intent; the known limitations describe the consequence and
point back by ID.

## Testing

The tests are split into short no-license tests and full solver-backed tests.
Run the short tests after any code change:

```bash
python3 -m unittest discover -s tests -v
```

The short tests (one module per area under `tests/`, each named for what it
pins) cover: JSON config validation and the README schema table
(`test_config_schema.py`, which also fails if the runner reads a key the table
does not list); derived geometry, laser paths, multi-beam tables, mesh
refinement and template rendering without unresolved placeholders
(`test_rectangular_workflow.py`, `test_laser_path.py`, `test_multi_beam_paths.py`,
`test_mesh_refinement.py`); prepare-only generation without `F3D_LICENSE`,
invalid-config failure reporting, the solver lock, the free-space and headroom
gates, and the flsgrf cleanup rule; the initial-temperature check across every
`&fl` builder; batch generation and the task spooler's enqueue/drain/crash
recovery (`test_batch.py`, `test_spool.py`); NPZ export, checksums, subsampling,
block merging and the reference comparator; the VTK validators and legacy
comparison on small fixtures; melt metrics, anchor scoring, the scaling-law
check and the validation pipeline driver; the stall monitor; and the guard that
no tracked script embeds a home path.

Run the full tests after major changes, new features, or refactors. They read
`F3D_LICENSE` from your shell or ignored `license.env`. If neither exists, the
full test fails with setup instructions. It runs a real solve, so it needs a
FLOW-3D licence:

```bash
export F3D_LICENSE='<your-license-endpoint>'
python3 -m unittest tests.full_solver_workflow -v
```

The full solver test runs the melt config with `--cores 2`, exports all time
steps, and checks that temperature rises and `Melt Region` becomes positive.

The matching manual solver check is:

```bash
export F3D_LICENSE='<your-license-endpoint>'
./bash/run_rectangular_case.sh configs/rectangular_moving_laser_melt.json \
  --case-name melt_two_core_check \
  --cores 2
```

For melt-model checks, run `configs/rectangular_moving_laser_melt.json` or
a short derived case and verify `runhyd.log`, `export_npz.log`, the generated
`npz/<case>.npz` (or PVD time series with `--export vtk`), temperature rise
above ambient, and nonzero melt region for melt-intended cases. The helper
`python/validate_npz_series.py` checks the NPZ numerically,
`python/melt_metrics.py` gives the melt-pool depth/width metrics used to score a
run against a published measurement, and `python/validate_scaling_law.py` checks
keyhole-intended cases against the Gan 2021 depth scaling law (see the export
section).

Tests and docs stay together: every documented user option should have a test,
and every public test path should be described here.

## Practical limits and next steps

- Multi-core support uses FLOW-3D's ADT decomposition for ranks greater than
  one. Check `runpre_decompose.log` when changing mesh size or rank count.
- In-plane paths can be straight (`start_point_m`/`end_point_m`) or an arbitrary
  `laser.path` (waypoints, sinusoidal, or coverage-biased random; see [Curved
  and scripted scan paths](#curved-and-scripted-scan-paths)), kept four beam
  radii inside the domain; `variable_speed` paths carry per-leg speeds, dwells,
  and per-leg power. Multiple simultaneous beams are `laser.beams` (up to four,
  each with its own path; see [Independent beams](#independent-beams)). Tilted
  beams still need an explicit template extension.
- `end_point_m` may have a y component; a straight scan stays a straight line in
  the top surface plane. Use `laser.path` for curves.
- Use at least several cells across the beam radius for useful heat-source
  resolution. The runner's two-cells-per-radius rule is a safety floor; it does
  not establish convergence.
- For a calibrated welding model, change one model feature at a time, check it
  in the GUI and against data, then expose the setting through JSON with a
  matching test.
