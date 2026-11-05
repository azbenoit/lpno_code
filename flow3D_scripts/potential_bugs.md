# Known issues

Ways the code can mislead or fail when it is used outside what it is coded for,
with how to recognise each one and what to do about it. Entries keep stable IDs
because the README and the code cite them; P3, P7, P10, P11, P14, P17, P20, P30
and P32 are retired. "Assumption" names the matching entry in `assumptions.md`,
which records the design choice behind the behaviour.

## Common traps

Settings that are easy to get wrong, with how to detect and avoid each.

### P34. With `turn_off_at_end: false` the laser stays on until `finish_time_s`
- **Symptom:** a dwell fires for `finish_time_s` when that exceeds `dwell_time_s`, and a
  scan keeps moving past `end_point_m` at constant speed and full power.
- **Detect:** the runner prints a WARNING at render with both times, the excess as a
  percentage and the overrun distance. The manifest records `laser_exposure_time_s`,
  `laser_on_past_declared_time` and `laser_overrun_distance_m`, and sets
  `laser_off_time_s` to `None`.
- **Avoid:** set `turn_off_at_end: true` when the beam must stop at `end_point_m` or
  after `dwell_time_s`. The setting renders one `timlbm` row, and the solver holds the
  last row of a time table until the end of the run.
- **Code:** `laser_time_tables` and the manifest fields in
  `python/run_rectangular_case.py`; `tests/test_laser_exposure_time.py`. Assumption: A6, A21.

### P35. A beam that overruns its end point can reach the domain end wall
- **Symptom:** melt touches an x face of the domain, and the pool near that face runs
  deeper than in a larger box, because the adiabatic wall conducts no heat away.
- **Detect:** `domain_validity` reports `end_wall_contact` (melt in contact with an x
  face); compare the manifest's `laser_overrun_distance_m` (P34) with the margin between
  `end_point_m` and the domain edge. The flag is reported and does not change `ok`.
- **Avoid:** lengthen the domain so the overrun stays interior with a thermal margin, or
  set `turn_off_at_end: true`.
- **Code:** `domain_validity`, `Validity.end_wall_contact` in `python/melt_metrics.py`.

### P39. The default Ti64 material is a database seed
- **Symptom:** keyhole cases on the default material drill deeper than measured.
- **Detect:** the manifest names `ti_imi318_mills` with `tv1 = 3123.15 K`.
- **Avoid:** use `materials/ti64_v3_calibrated` (`tv1 = 3520 K`) for keyhole or
  quantitative Ti64 work; the seed combines the per-material exponent with the database
  anchor, which over-drills against measured keyhole depths, and suits conduction-regime
  or qualitative use.
- **Code:** `materials/`. Assumption: A30.

### P37. `laser.reflectivity` is a Fresnel coefficient
- **Symptom:** a fixed `laser.reflectivity` of 0.35 absorbs about half of the beam at
  normal incidence.
- **Detect:** `ereflct` in the rendered `prepin` equals the configured reflectivity.
- **Avoid:** leave `laser.fresnel_from_material` on (the default when the material has
  optical data), or convert a target absorptivity to ε first. With the simplified Fresnel
  model (`irefopt=3`) ε = 0.35 gives a normal-incidence absorptivity of about 0.50, and an
  absorptivity of 0.35 needs ε ≈ 0.217.
- **Code:** `load_and_validate` (`laser.reflectivity`). Assumption: A22, A24.

### P4. Omitting `boundary_conditions` opens all six faces to pressure
- **Symptom:** a case expected to have walls shows outflow through its sides and a
  different heat coupling at the top face.
- **Detect:** every face of the rendered `prepin` is a pressure opening (code 5).
- **Avoid:** set the faces explicitly in `boundary_conditions`.
- **Code:** `boundary_replacements`. Assumption: A17.

### P29. `--overwrite` leaves earlier solver output in place
- **Symptom:** the solver launcher aborts with "OUTPUT FILES ALREADY EXIST" and the case
  fails (exit 521).
- **Detect:** the case directory still holds `flsgrf*`, `hd3*` or exports from the
  earlier run.
- **Avoid:** remove the old case directory and rerun without the flag; the aiida-pino
  option `execution_profile.overwrite` behaves the same. The flag skips the runner's own
  refusal and deletes nothing.
- **Code:** `existing_run_artifacts`, `main`.

## Configuration and rendering

### P1. Power changes along a path only per leg of `variable_speed`
- **Symptom:** a config that expects a power ramp along one path runs at constant power.
- **Cause:** `laser.power_w` and each `laser.beams` entry hold one value. Only
  `variable_speed` paths take an optional `power_w` per leg, and nothing ramps power
  within a leg. A bare `start_point_m`/`end_point_m` scan is one straight,
  constant-speed line.
- **Check:** compare the manifest's `laser_path_type`, `scan_path_length_m` and
  `power_w` with intent.
- **Code:** `python/laser_path.py`; `load_and_validate`. Assumption: A5, A5b.

### P2. Some optional integer flags use a bare `int()`
- **Symptom:** `"max_reflections": 3.5` is truncated to 3 without a message, and a
  string such as `"subprocess_id": "x"` stops with a raw `ValueError` traceback.
- **Cause:** `radiation_distribution_flag`, `reflection_flag`, `max_reflections`,
  `absorptivity_flag` and `subprocess_id` bypass the `integer()` validator.
- **Workaround:** pass whole numbers. **Fix:** route the five flags through `integer()`.
- **Code:** `load_and_validate` in `python/run_rectangular_case.py`.

### P5. A Gaussian beam on a coarse mesh can show little or no heating
- **Symptom:** a case that should melt barely heats.
- **Cause:** near the floor of two cells per beam radius the Gaussian peak is
  under-resolved.
- **Workaround:** refine the mesh before trusting a null result. Assumption: A7, A9.

### P6. The `keyhole` and `legacy_case_fine_1` profiles are physically identical
- **Symptom:** switching between the two profiles changes no result.
- **Cause:** they differ only in the scalar `sigma`, which the tabular
  surface-tension model ignores; surface tension stays on in both.
- **Workaround:** vary surface tension through the σ(T) table, or `ifsft`/`isigma`
  in the template.
- **Code:** `solver_profile_replacements`. Assumption: A12, A24.

### P8. Dimensions that are not multiples of the cell size change the resolution
- **Symptom:** the effective cell size differs from `mesh.cell_size_m`.
- **Cause:** the solver resizes cells to fit the bounds; `derived.nominal_mesh_cells`
  is a float.
- **Check:** read the actual mesh from `runhyd.log`. Assumption: A10.

### P9. Solving and export need a Linux login shell
- **Symptom:** solve or export fails on Windows or from a non-login shell.
- **Cause:** `run_logged` uses `bash -lc`, the vendor environment comes from sourcing
  `flow3dvars.sh`, and the export paths are POSIX.
- **Code:** `run_logged`, `solve`. Assumption: A15.

### P15. The absorptivity multiplier is a fixed 1.0
- **Symptom:** `cbmengy` in the deck does not change with the material.
- **Cause:** `laser.absorptivity` keeps the laser default of 1.0 and is never read from
  the material file; absorption is governed by the Fresnel coefficient (P16, P37).
- **Workaround:** set `laser.absorptivity` explicitly where a case needs another coupling.
- **Code:** `load_and_validate`. Assumption: A8, A24.

### P16. Fresnel inputs are literature values
- **Symptom:** absorbed power, and with it keyhole depth, carries the uncertainty of
  the optical data.
- **Cause:** ε comes from the electrical resistivity through the Drude relation, or
  from inverting `optical.normal_absorptivity`. Provenance and confidence are in
  `materials/sources.md`; the CoCr, AlSi10Mg and Al6061 values are estimates. The
  Drude form overestimates Cu and Al absorption near 1 µm, so those files set
  `optical.prefer_resistivity=false` and use a measured absorptivity.
- **Workaround:** check per-material values against experiment where absolute
  absorption matters.
- **Code:** `fresnel_epsilon_from_resistivity`. Assumption: A22, A25.

### P18. A single-beam path can exceed the beam-table length
- **Symptom:** config validation passes and the preprocessor then refuses the deck at
  row 501 of the beam table.
- **Cause:** `MAX_PATH_SAMPLES = 500` caps samples, each sample becomes two table rows,
  and the tables hold 500 rows (`MAX_BEAM_TABLE_ROWS`). Only `multi_beam_tables`
  checks the row count.
- **Workaround:** keep a single-beam path below about 250 samples with a coarser
  `sample_spacing_m`. **Fix:** guard `velocity_time_tables` on `MAX_BEAM_TABLE_ROWS`.
- **Code:** `python/laser_path.py`; `multi_beam_tables`. Assumption: A5b.

### P21. Ambient-pressure defaults differ between initialization branches
- **Symptom:** a default `legacy_halfspace` deck has 100000 Pa as void and initial
  pressure (`pvoid`, `presi`) and 101300 Pa at the pressure boundaries.
- **Cause:** `fluid_initialization_block` writes 101300 Pa for `substrate_with_gas` and
  100000 Pa for `filled_mesh` and `legacy_halfspace`, while `__AMBIENT_PRESSURE_PA__`
  defaults to 101300 Pa. The `legacy_halfspace` branch also lacks the `igivht=1` that
  the other two set.
- **Workaround:** set `simulation.ambient_pressure_pa` so that both agree. Changing the
  defaults changes the initial state of the reference case and needs re-validation.
- **Code:** `fluid_initialization_block`.

### P36. The analytic recoil model uses fixed coefficients for every material
- **Symptom:** recoil onset and stiffness do not follow the material's own saturation curve.
- **Cause:** with `simulation.recoil_model` unset or 1, the deck uses the analytic
  overpressure with `laser.vapor_pressure_a` = 50650 Pa and `laser.vapor_pressure_b` =
  10.95913 unless the config sets them, and the profiles set `pv0` to 50650 Pa. These
  values are an ideal-gas estimate for one Ti64 setup with air as the gas phase.
- **Workaround:** use `simulation.recoil_model: 2` (recoil from the evaporation mass
  flow) with `simulation.void_vapor_pressure_pa: 101325`, as the validated recipes do,
  or set the two coefficients per material.
- **Code:** `load_and_validate` (`recoil_model`, `vapor_pressure_a`, `vapor_pressure_b`,
  `void_vapor_pressure_pa`).

### P38. A material without a vapour molar mass falls back to a fixed `tvexp`
- **Symptom:** the saturation curve of a new material ignores its latent heat and
  molar mass.
- **Cause:** the renderer computes `tvexp = R/(M·clhv1)` only when
  `vaporization_molar_mass_kg_mol` is present and otherwise writes the template
  constant 5.989e-5. Every shipped material carries M.
- **Workaround:** give every new material file a molar mass.
- **Code:** `load_material` (molar-mass branch). Assumption: A30.

### P40. Property tables are held constant outside their temperature range
- **Symptom:** no Marangoni flow over parts of the liquid range.
- **Cause:** the solver holds a table's end value beyond its range. The σ(T) table of
  the Ti64 seed (`ti_imi318_mills`) ends at 1881 K, below the 1923 K liquidus, so
  dσ/dT is zero across the whole liquid; the calibrated Ti64 table ends at 3123 K and
  is constant above it, while keyhole walls run hotter.
- **Workaround:** extend the tables across the temperatures a case reaches.
- **Code:** `materials/<name>/*.csv`.

## Runs and resources

### P28. Reading a large NPZ materialises every field
- **Symptom:** a reader or scorer is killed for memory on a refined or long export.
- **Cause:** `{k: z[k] for k in z.files}` decompresses the whole archive. Writing
  streams through memmaps, reading does not.
- **Workaround:** load only the fields a metric needs (`Melt Region`, `Temperature`,
  `x`, `y`, `z` and `time_s` cover `fusion_geometry` and `domain_validity`).
- **Code:** every consumer of `npz/<case>.npz`.

### P33. A hard memory cap on the solver's scope can kill a healthy solve
- **Symptom:** a long solve is OOM-killed under a cgroup memory limit although its
  resident memory is about 1 GB.
- **Cause:** `MemoryMax` also counts the write-back page cache of the result file,
  about 13 GB on a long run.
- **Workaround:** use `MemoryHigh` or no memory cap; `CPUQuota` and `--nice` keep a
  shared machine responsive.

## Exports and metrics

### P12. The NPZ comparator's extrema tolerance is loose
- **Symptom:** `compare_npz_to_reference.py` passes two exports whose peaks differ
  noticeably.
- **Cause:** `--rtol-extrema` defaults to 0.25 of each field's range. That band absorbs
  systematic peak differences against a reference made with other settings or another
  solver version, where means can agree while peaks differ.
- **Workaround:** the single-rank solver is deterministic, so a regression check between
  two exports of this pipeline should use a tolerance near zero. Treat large extrema
  gaps against an external reference as a settings or version difference.
- **Code:** `python/compare_npz_to_reference.py`. Assumption: A23.

### P13. NPZ axes differ slightly from VTK cell-centre axes
- **Symptom:** the axis arrays of the NPZ and of the VTK datasource are not
  byte-identical for the same cells.
- **Cause:** the NPZ takes its axes from the result file's point coordinates.
- **Workaround:** compare cell counts and field values.
- **Code:** `python/export_all_timesteps_to_npz.py`. Assumption: A18.

### P19. The floor check in `domain_validity` is absolute
- **Symptom:** preheated cases read `ok = 0` from frame 0, before the laser fires, and
  invalidity follows the preheat temperature.
- **Cause:** `MAX_FLOOR_K = 400` compares the bottom-row maximum with a fixed 400 K.
- **Workaround:** use `Validity.floor_rise_k`, the rise above the substrate's own initial
  temperature. `ok` is left unchanged on purpose; `tests/test_floor_rise.py` pins both.
- **Code:** `domain_validity` in `python/melt_metrics.py`.

### P22. A fusion width read at laser-off is a snapshot
- **Symptom:** two width readings from the same solve disagree by several cells.
- **Cause:** exports end when the laser turns off. When the molten tail is longer than
  the track (IN718 near 1 m/s on a 600 µm track), the bath is still widening and nothing
  has solidified, so the width depends on where it is read: `track_half_cells=5` at
  mid-track and a wide beam-following window give different numbers. Depth is set by
  the keyhole and stays stable.
- **Detection:** a last-frame surface centreline above liquidus at the edge of the
  analysis window means the tail is cut.
- **Workaround:** quote the band and the position with every width.
- **Code:** `fusion_geometry`, `fusion_geometry_at_beam`.

### P23. `depression()` width spans the gap between beams
- **Symptom:** on multi-beam cases the cavity reads far wider than deep, and an
  aspect-based keyhole test classifies no case as keyhole.
- **Cause:** the width is the full y-extent of gas inside metal in a frame, which
  includes the metal between separate cavities.
- **Workaround:** use depth-only verdicts on multi-beam data (depth is taken per frame
  along z and is unaffected). **Fix:** take the width of the connected component that
  holds the deepest gas cell.
- **Code:** `depression` in `python/melt_metrics.py`.

### P24. The geometric metrics assume one track
- **Symptom:** multi-beam widths grow with beam separation, and `domain_validity`
  flags valid cases.
- **Cause:** `fusion_geometry` takes a time-union of the melt region, which spans every
  track of a `laser.beams` case; the depth then belongs to the hottest beam.
- **Workaround:** score multi-beam cases per track; that needs the melt region segmented
  per beam, using the paths in `derived.beams`.
- **Code:** `fusion_geometry`, `domain_validity`.

### P25. Multi-beam NPZ provenance holds totals
- **Symptom:** a case with beams of 141.6 W and 103.2 W reads as 244.9 W in the NPZ.
- **Cause:** `export_provenance_scalars` stores the total delivered power and the mean
  scan speed; per-beam values are only in `case_manifest.json` (`derived.beams`).
- **Workaround:** give a model trained on single- and multi-beam cases the beam count
  as an input.
- **Code:** `export_provenance_scalars`.

### P26. Lengths below about five cells sit at the quantization floor
- **Symptom:** a steady-window length with a standard deviation of exactly zero, such as
  20.0 ± 0.0 µm on a 10 µm grid.
- **Cause:** the value is pinned to one or two cells.
- **Workaround:** quote a length only above about five cells and state the cell count
  below ten; `melt_metrics.cell_size_um` gives the per-case value.
- **Code:** `python/melt_metrics.py`; quantization rule in `validation/ACCEPTANCE.md`.

### P27. Nested refinement flags a truncated pool only on contact
- **Symptom:** a pool deeper than the fine block is measured short.
- **Cause:** a refined run's NPZ is the fine block alone. `depth_truncated` fails
  validity when melt reaches the lowest exported cell, with no warning a cell or two
  earlier, and it does not say whether the box or the substrate floor was reached.
- **Workaround:** choose `margin_m` from the expected melt depth, read
  `derived.mesh_refinement`, and compare a nested case with a uniform one at the same
  resolution before trusting it.
- **Code:** `domain_validity`, `nested_mesh_block`, `npz_block_merge.select_finest`.

### P31. Cropping a fixed number of top cells removes displaced metal
- **Symptom:** a surrogate trained on cropped data learns a flat free surface where the
  real one was raised.
- **Cause:** consumers that drop the gas layer by cell count (`crop_z_top`, default 10 in
  the aiida-pino derivation) also drop metal pushed above the original surface.
- **Workaround:** derive the crop per case from the highest cell that contains metal,
  check that the removed cells hold none, or set `crop_z_top: 0`.
- **Code:** aiida-pino `utils/dataset_derivation.py`; `python/predicted_fields.py`.
