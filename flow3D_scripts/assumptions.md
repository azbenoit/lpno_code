# assumptions.md

Non-trivial assumptions and deliberate code choices in this repo — the ones
whose effect on a result or a run is not obvious from a single line of code.
Trivial input validation and self-evident defaults are out of scope; the runner
source and `README.md` cover those.

Use this file to sanity-check results: if a run looks wrong, a wrong assumption
here is a likely cause. Each entry names where it lives so it can be changed
deliberately. Consequences that arise *from* these assumptions, or from usage
the code does not cover, are listed in `potential_bugs.md` (known
issues), which points back here by ID. This file states
intent; that one states the consequence.

Keep entries current: update an assumption when the code it describes changes,
and delete it when it no longer holds.

## Geometry and coordinates

**A1. The laser focal plane is the substrate top surface at z = 0.**
`start_point_m` and `end_point_m` must have z = 0; solid fills z ∈ [−thickness, 0]
and gas fills z ∈ [0, gas_clearance]. Effect: every geometry and laser quantity
is expressed relative to a fixed surface; a config cannot focus the beam above
or below the surface. Source: `load_and_validate` z=0 check; README "Coordinate
convention and geometry".

**A2. The beam is vertical, directed in −z.**
The optical origin sits one `focal_distance_m` directly above the focus. Effect:
no beam tilt or angle of incidence is modelled. Source: manifest
`beam_optical_origin_at_t0_m`; template.

**A3. Geometry is analytic half-space initialization, not STL.**
Default `legacy_halfspace` sets `fzh = surface_z_m`. Effect: the workpiece is an
axis-aligned box defined by dimensions; arbitrary CAD/STL shapes are not
representable. Source: `fluid_initialization_block`; README.

**A4. `placement` derives the mesh bounds; the default leaves four beam radii of lead-in.**
`laser_leading_margin` sets `x_min = start_x − 4·radius` and requires
`length > 5·radius`; `centered` centres the box on (0, 0). Effect: the substrate
position relative to the scan is implicit in the placement mode, not given
directly. Source: `load_and_validate`.

## Laser and scan path

**A5. Without `laser.path`, the scan is a single straight, constant-velocity segment.**
`velocity = (end − start) / scan_time`, written as one time-table leg from t=0
to `scan_time`. Effect: with only `start_point_m`/`end_point_m`, raster, hatch,
and acceleration are not represented; use `laser.path` (A5b) for curves.
Source: `laser_time_tables`; template.

**A5b. `laser.path` follows the sampled polyline at exactly constant speed.**
A `laser.path` (waypoints/sinusoidal/random) is resampled at equal arc length,
and each leg between samples is emitted as a constant-velocity *plateau* in the
`utlbm`/`vtlbm`/`timlbm` table (two rows per leg, leg duration `chord/speed`, a
tiny `eps` between plateaus) — the same idiom as the straight scan (A6). FLOW-3D
integrates the beam origin along that table. Effects and limits: (a) every leg
runs at exactly `scan_speed_m_per_s`, and the beam traces the sampled polyline
under both possible beam-table semantics (linear interpolation — which the A6
plateau idiom implies — or piecewise-constant hold, for which the duplicate rows
are just redundant); curvature finer than `sample_spacing_m` (default
`beam_radius/2`) is still approximated by the polyline. (b) Samples are capped at
`MAX_PATH_SAMPLES=500` (table ~= 2x that). The FLOW-3D beam-table limit is
MEASURED at 500 rows (`MAX_BEAM_TABLE_ROWS`; row 501 is rejected at
preprocessing), so the sample cap is ~2x too
permissive for a single-beam path -- only `multi_beam_tables` guards the row
count; see `potential_bugs.md` P18. A long/fine path errors and asks for a coarser
`sample_spacing_m`. (c) A hard check rejects any sample within
`4*beam_radius` of an x/y edge. (d) Path mode forces `centered`/`bounds_m`
placement (`laser_leading_margin` is refused). (e) `random` is a seeded,
reproducible coverage-biased walk over the *cropped* region inside the 4-radius
keep-out: it starts at the left edge, mid height (or `start_point_m`), steers
each step toward the least-visited neighbourhood (visitation is averaged over
in-region cells so the forbidden outside does not bias it to hug the boundary),
and walks until it reaches `length_m` (default: the domain length, i.e. the x
extent = cropped length + the two 4-radius margins), trimmed to that length; it
**self-intersects** if the domain is smaller than the requested length — allowed
for now, to be forbidden in a later iteration. `max_turn_deg` caps the per-step turn in
open space, but peeling off a domain wall (rotating in place toward the interior
rather than dead-ending) may turn sharper. Source: `laser_path.py` (`build_path`,
`resample_by_arclength`, `check_within_margin`, `velocity_time_tables`,
`_random_control`); `load_and_validate` path branch.

**A6. The beam is switched off by a second table point a hair after `scan_time`.**
At `scan_time + max(1e-12, scan_time·1e-9)` power and velocity drop to zero.
Effect: stops the adaptive final solver step from carrying the beam past
`end_point`; assumes that tiny offset is resolvable on the solver clock. Source:
`load_and_validate` (`laser_off_time`), `laser_time_tables`.

**A7. Heat-distribution codes are uniform → 0, gaussian → 2.**
Effect: `gaussian` is the default (the keyhole model, A24); `uniform` remains
available. "gaussian" maps to FLOW-3D mode 2 (not 1). Changing this silently
changes the heat-source shape. Source: `load_and_validate`.

**A8. The laser couples to all facing surfaces, with recoil on, limited reflection, and plume off.**
Defaults (keyhole model, A24): `absorptivity` 1.0, `reflectivity` 0.35 (the scalar
fallback — ε is derived from the material by default when it has optical data, see
A22/A25), `max_reflections` 20, `reflection_limit` 0.01, recoil pressure on, plume
forcing off. Effect: these are physics choices that move melt depth and are not
"neutral"; validate them per process. Source: `load_and_validate` defaults,
template, README "What FLOW-3D model is generated".

**A22. The laser is a ray-traced beam with multiple reflections, deposited as a surface flux.**
`&weld` sets `if_beam=1` with `irefopt=<reflection_flag>`, `nrefmx=<max_reflections>`
and `qreflim=<reflection_limit>`: the beam is split into rays traced into the
keyhole and reflected up to `nrefmx` times (or until a ray drops below `qreflim`
of its energy). `irefopt=3` selects FLOW-3D's **simplified Fresnel** model (the
absorption rate depends on incidence angle), with the Fresnel coefficient
`ereflct`=ε (not a plain reflectivity); ε is derived per material by default when
the material carries optical data (A25), via `laser.fresnel_from_material` —
preferring
`optical.electrical_resistivity_ohm_m` through the Drude relation
ε = √(2·ε₀·ω·ρ) (ω from `laser.wavelength_m`, default 1.064 µm), else inverting
`optical.normal_absorptivity` (provenance in `materials/sources.md`). The
case_fine_1 / batch configs use `irefopt=3`,
`nrefmx=20`, `qreflim=0.01`. Absorbed
energy is laid down as a surface flux on facing cells (`ibeamsurfflux=1`), not
volumetrically — `dpenlbm=0`, no beam penetration depth. Effect: keyhole
multiple-reflection absorption is modelled, not a flat top-surface flux, so
`reflection_flag`/`max_reflections`/`reflection_limit` change how much reflected
energy is recaptured in the keyhole. Source: template `&weld`,
`load_and_validate` laser defaults, `solver_profile_replacements`.

## Mesh

**A9. Two cells per beam radius is a hard floor, not a convergence criterion.**
`cell_size_m ≤ beam_radius_m / 2` is enforced; nothing checks convergence.
Effect: a config can pass validation and still be far too coarse to trust.
Source: `load_and_validate`; README "Practical limits and next steps".

**A10. Reported cell counts are non-integer; the solver re-sizes the mesh.**
`nominal_mesh_cells = extent / cell` is a float, and FLOW-3D adjusts the real
cell size to fit the bounds. Effect: the effective cell size can differ from the
requested `cell_size_m` when a dimension is not an exact multiple. Source:
manifest `derived.nominal_mesh_cells`.

## Material

**A11. Material files are seed data in SI units, not a calibrated database.**
Each material is a directory `materials/<name>/`: `<name>.json` (scalars + optical
block) plus one CSV per temperature-dependent property (`temperature_k,<value>`),
including `electrical_resistivity_ohm_m.csv`. Required scalars and CSV tables must
be positive and physically ordered (solidus ≤ liquidus, strictly increasing table
temperatures). Provenance is in `materials/sources.md`; the thermophysical values
are starter engineering values, the optical values are cited per parameter. Effect:
reproducible test inputs, not validated process data. Source: `load_material`,
`read_csv_table`; README "Material files".

**A25. Fresnel ε is rendered temperature-dependent (`ereflct(T)`, `iereflct=1`) from ρ(T).**
When `laser.fresnel_from_material` is on and the chosen route is resistivity/Drude
with a multi-point `electrical_resistivity_ohm_m.csv`, the runner renders a tabular
`#ereflct(1)` (ε per temperature, ε=√(2·ε₀·ω·ρ(T))) and sets `iereflct=1`, so the
scalar `ereflct` is ignored exactly like the scalar `sigma` under `isigma=1` (A12).
Single-point resistivity or the absorptivity route gives a constant ε (`iereflct=0`).
Effect: absorption tracks temperature through ρ(T), matching the laserbeamFOAM-style
physics within FLOW-3D's simplified single-ε Fresnel. This material-derived coupling
is the **default** whenever the material has optical data (a resistivity CSV or
`normal_absorptivity`); `laser.fresnel_from_material` defaults to true in that case
and a config sets it false to force the fixed scalar reflectivity (the legacy
line-track reproduction config does, to stay faithful). Source: `load_and_validate`
(default_fresnel, fresnel_eps_table), template `__LASER_IEREFLCT_LINE__`/`__FRESNEL_EREFLCT_TABLE__`.
**Corollary: `normal_absorptivity = 0.35` is a sentinel meaning "resistivity route, ignore
me", not a measured value.** The seven materials carrying it are exactly the seven with
`optical.prefer_resistivity=true`, so for them the solver reads ρ(T) and never the
absorptivity except as a cross-check. Do not replace those 0.35s with per-alloy figures:
it changes nothing physically and makes the files look like they carry measured optics.
A consumer conditioning a model on an absorptivity-derived channel gets a channel that is
degenerate across all seven. Detail and the per-material table: `materials/sources.md`.

**A12. The scalar `sigma` is inert — surface tension is always on and table-driven.**
The template hard-codes `ifsft=1` (surface-tension model on) and `isigma=1`
(tabular, temperature-dependent), and renders the material's `#sigma(1)` σ(T)
table in every profile. With `isigma=1` FLOW-3D reads that table, so the scalar
`sigma=__SIGMA__` is **not used**: `legacy_case_fine_1` sets it to 0 and `keyhole`/
`standard` set it to the material value, but the result is identical (verified
bit-for-bit). Effect: the legacy profile does **not** disable surface tension, and
the `keyhole` vs `legacy_case_fine_1` distinction is physically a no-op (they
differ only in this inert line). To actually vary surface tension you must change
`ifsft`/`isigma` or the σ(T) table, not the scalar. Source:
`solver_profile_replacements`, template `&props`.

**A24. The default solver profile is `keyhole` (case_fine_1 keyhole tuning).**
A config that omits `simulation.solver_profile` gets `keyhole`, which renders the
ray-traced gaussian-beam keyhole model with the laser defaults of case_fine_1
(gaussian, `reflection_flag` 3, `max_reflections` 20, `absorptivity` 1.0,
`reflectivity`/ε 0.35 — scalar fallback; ε is material-derived by default, A22/A25)
plus the legacy air/gravity/gamma tuning. **Note:** it was
intended to differ from `legacy_case_fine_1` by keeping the material's surface
tension, but that is the inert scalar `sigma` (A12), so the two profiles are
physically identical; surface tension is on (table-driven) in both. The profile
distinction is retained only as a label. Caveat: the tuning hard-codes beam-shape
and top-BC lines (`xwidth0=0.0002`, `hflbm`, the z-max `rwall` opening) tuned for
the case_fine_1 beam (50 µm radius); a very different beam may need these
generalized. Source: `solver_profile_replacements`, `load_and_validate` defaults.

## Solver execution and environment

**A13. All MPI ranks run on the local host.**
`machines` repeats `socket.gethostname()` once per rank; `ranks =
cores / threads_per_rank`. Effect: no multi-node/cluster execution; "cores"
means local cores. Source: `machine_file_text`, `parallel_settings`.

**A14. One rank uses the pre-split input; more than one re-decomposes via `runpre_par -9`.**
Effect: the seed one-block `MESH_MPI` partition is valid only for a single rank;
>1 rank relies on FLOW-3D's ADT decomposition overwriting it before solving.
Source: `solve`.

**A15. The vendor environment comes from a Linux login shell sourcing `flow3dvars.sh`.**
Commands run via `bash -lc` after sourcing the env file; the license is read
from `F3D_LICENSE` / `F3DTKNUX_LICENSE_FILE` / `LM_LICENSE_FILE` or
`license.env`. Effect: the toolchain assumes a POSIX login shell with that
vendor script present. Source: `run_logged`, `solve`, `local_license_endpoint`.

**A16. Overwrite protection keys on completed result artifacts.**
A run is refused when `flsgrf*.<case>*` already exists, or when the `npz/` or
`paraview/` export directories do. Effect: `prepin`,
`case_manifest.json`, and `machines` from a prepare-only or failed run are
regenerated for the same `case_name`; identical parameters re-render silently
(a continuation), but re-rendering with different parameters warns on stderr
before replacing the old inputs. Source: `main`, `render`.

**A17. Omitting `boundary_conditions` opens every face to pressure (code 5).**
Effect: leaving the section out is not "solver defaults" — it writes pressure
openings on all six faces, changing confinement and the z-max heat coupling the
legacy notes warn about. Source: `boundary_replacements`.

**A23. The single-rank solver is deterministic: identical inputs give bit-identical output.**
Re-running a case (same config, same `--cores`) reproduces every field and time
step exactly. Effect: a difference between two of our runs is never noise. A
reference produced with other settings or another solver version can differ in its
peaks while agreeing in its means. Such gaps come from the settings or version, and
keyhole peaks should not be dismissed as stochastic.
(Multi-rank MPI reduction order could in principle perturb this; the maintained
one-rank/OpenMP path did not.)

## Result export

**A18. NPZ export reads the raw FLSGRF structured grid and assumes one static mesh per run; multi-block meshes are stitched.**
The exporter fetches the FLSGRF reader's `vtkMultiBlockDataSet`, collects every
`vtkStructuredGrid` block, keeps the finest where blocks overlap (a
nested-refinement run exports the fine block only) and stitches them
(`npz_block_merge.stitch_blocks(select_finest(blocks))`), deriving the cell
axes and ghost mask once from the first frame and stripping the reader's
one-cell ghost shell via `vtkGhostType != 0`. Effect: the export covers the
full nominal domain (gas layer included) at native — locally finest —
resolution with no resampling; it would mislabel cells if the mesh changed
between steps. Source: `export_all_timesteps_to_npz.py`, `npz_block_merge.py`.

**A19. NPZ export is lab-frame and lossless except for redundant per-component velocities.**
`Noninertialreferenceframe = 0` (no reference-frame change), every stored step
and field is kept with no time averaging, and arrays use axis order `(x, y, z)`.
The mesh-internal `Cell Type`, `Component Number`, `vtkGhostType` and the
`X/Y/Z-velocity` scalars are dropped; `X/Y/Z-velocity` are recoverable from the
kept 3-component `Velocity`. Effect: physics information is preserved while the
file stays compact and matches the aiida-pino datasource layout
(`time_axis=True, moving_ref_frame=False, flip_axes=False`). Source:
`export_all_timesteps_to_npz.py` `SKIP_ARRAYS`, `CELL_ARRAYS`.

## Batch runs

**A20. A batch is independent single-case runs, sequenced by the spooler, not the solver.**
`build_batch.py` deep-copies a reference config per parameter set and writes one
JSON plus a `tasks.txt` of full run commands. The task file is queued with the
spooler (`python/spool.py`), which runs the cases one at a time, because solver
runs that share a licence are serialised by a lock (A27).
Each run is an ordinary `run_rectangular_case.py` invocation isolated by its own
`<out-root>/<case_name>/` directory (default `runs/`; a batch sets `--out-root`
to its own folder so configs, `tasks.txt`, and every case live together), so
runs share nothing but the solver licence. Effect: there is no cross-run
coupling or shared state; correctness of a batch reduces to correctness of one
run plus distinct case names. Source: `batch.py`, `python/spool.py`.

**A26. The spool queue defaults to a shared per-user directory outside the repo.**
`default_spool_dir()` returns `$F3D_SPOOL_DIR` if set, else `~/.flow3d/spool` —
not a folder inside the repo. A single worker drains it, and the `flock` in that
directory blocks a second worker.
`build_command` also absolutizes the config path at enqueue time, so a task runs
the correct case whichever checkout's worker drains it
(the runner path is already absolute, and the runner anchors templates/materials/
`runs/` to its own `__file__`, not cwd). Effect: runs queued from any checkout
interleave through one serial worker; the cost is that the queue lives outside
the repo, and two users pointing `F3D_SPOOL_DIR` at different paths would defeat
the shared lock. Source:
`python/spool.py` (`default_spool_dir`, `build_command`, `run_worker`).

**A27. `solve()` holds a shared lock, so even direct runs are serialised.**
`solve()` wraps the `runpre_par`/`runhyd_par` calls in `seat_lock()`, an exclusive
`flock` on `.seat.lock` inside `default_spool_dir()`. Both a direct
`run_rectangular_case.py` invocation and a spooled run reach the solver through
`solve()`, so a run started while another solve is in flight (direct or queued,
from any checkout sharing `F3D_SPOOL_DIR`) blocks until that solve releases the
lock. The lock is a different file from the spooler's
`.worker.lock` (which only marks a live worker), so an idle worker does not block
a direct run; only an in-flight solve does. Prepare-only, rendering, and export
run outside the lock (they do not start the solver). Effect: "one solve at a time" is enforced
by the lock, not just by convention or by using the queue; the cost is that a
direct run can appear to hang while it waits (it prints a `[seat] waiting ...`
line; Ctrl-C gives up). Source: `python/run_rectangular_case.py`

**A30 (v2). The legacy hardcoded `tvexp` applies only to materials without a
vapor molar mass; every shipped material carries one.** The
renderer emits per-material `tvexp = R/(M*clhv1)` (the solver's own
per-material definition) whenever `vaporization_molar_mass_kg_mol` is present, and falls
back to the template constant `tvexp=5.989e-05` otherwise. That constant is
~3.4x too large for Ti64 (saturation curve flattened; recoil ~17x softer at
4000 K). The Mills seed `materials/ti_imi318_mills` combines the per-material
exponent with its database anchor tv1=3123.15 K, which over-drills against
measured keyhole depth, so it stays a database seed, adequate only for
conduction-regime or non-quantitative use. Quantitative Ti64 work references
`materials/ti64_v3_calibrated` (tv1=3520±~95 K, fitted against measured keyhole
depth; `validation/MANIFEST.md`). Source: renderer
`load_material` molar-mass branch; `materials/sources.md`.

**A31 (v4). The fixed-fraction plume is RETIRED from every recipe; the knobs
remain implemented but unused.** As a fixed fraction it acts as a beam-power
sink: the diverted fraction never reaches the melt. Its role is taken by the
corrected Ti64 saturation anchor, tv1=3520 K (A30). Effect: no config should set
`laser.plume_attenuation`/`plume_radius_m`/`plume_condensation`. The `*_bundle`
configs in `validation/configs/` are the superseded plume arm
(aplume=0.20/rfplume=8×spot). Source: knobs in
`python/run_rectangular_case.py`.
Scope limit for sweeps that extend far above the anchors (for example IN718 to
King H≈48): retiring the plume leaves **no term that grows with intensity**, while
physical vapour shielding does. Far above the anchors
this is a DIRECTIONAL bias: the recipe over-delivers power to the surface, and the
error grows with P/r0, i.e. exactly along the axis a high-enthalpy sweep extends.
No number bounds it, because nothing measures it. Pair this with the vapour-curve
extrapolation: peak temperatures of ~5300 K at the top of such a box sit
~1800 K (Ti64, tv1=3520) to ~2100 K (IN718, tv1=3186) above the only temperature
the saturation curve is pinned at, and p_sat is exponential in T.

**A21. `--derive-finish-time-from-track` sets finish_time = track_length / scan_speed.**
For a fixed straight track with `turn_off_at_end=false`, the beam travels at the
scan speed for the whole run, so the time to cross the track scales as
`1 / speed`. The flag computes `simulation.finish_time_s` from the (overridden)
`start_point_m`/`end_point_m` and per-case speed. Effect: a speed-and-power
sweep keeps the same physical track length at every speed; without it, holding
`finish_time_s` fixed would scan different lengths. It suits `turn_off_at_end=false`
line tracks only. Source: `build_batch.py` `track_length`.

## Validation helpers

**A32. The scaling-law check (Gan 2021) uses liquidus properties, a flat η=0.35
default, and one pooled scatter band.**
`validate_scaling_law.py` computes Ke with ρ, cp, K interpolated at the liquidus
from the material's CSV curves when they exist (Gan's own property convention:
their Ti64 K=33.4, cp=830 are this repo's Mills curves at 1923 K) and with the
scalar properties otherwise, because a constant-property solve really used the
scalar. Effect: for Ti64 the two routes differ by ~2.9× in Ke, so a case checked
without its CSVs lands on a different part of the curve — the output's
`props=` field says which route ran. Without `--eta` a flat 0.35 stands in for
Gan's measured Ke-dependent absorptance (~0.3–0.7), biasing Ke low for
deep-keyhole cases (warned on stderr). The PASS/WARN/FAIL band is the pooled
residual sd (0.93 in e/r0) over all 87 SD1 points; Al6061 sits ~+2.2 z above
the pooled curve, so aluminium-class (high-diffusivity) alloys tend to WARN by
construction. The depth metric mirrors the batch steady-window convention:
steady = last 30% of the laser-on frames, deepest interior gas cell (FoF < 0.5)
below the surface with x/y wall columns excluded. Laser-on is taken from the
case's `derived.laser_off_time_s` when available; the older "first 85% of frames"
rule is only a fallback and is **wrong for cases that run the laser to the end**
(with `finish_time == travel time`, 0.85 discards the deepest 15% of the track
as a cooling tail that does not exist).
See `python/melt_metrics.py`. Source:
`python/validate_scaling_law.py`; reference data
`python/data/gan_scaling_reference.json`; README "Scaling-law consistency
check".

## Melt-pool metrics

**The ±30% acceptance band is derived from the sources' own scatter, not chosen.**
Galiègue quote ~10% along-track variability and measure most conditions at a
single location; their own data inverts by 30% between r₀ 69 and 83 µm; Gan's
156 W series reports the same depth (62 µm) at 0.7 and 0.9 m/s, where depth is
expected to fall with speed; and Vaglio vs Galiègue differ by 12%
at matched power and speed. All points are kept and scored; the band only
sets the tolerance. Effect: agreement claimed tighter than ~10–30% would
be fitting measurement noise. A systematic one-sign offset across a ladder is
still reported even when every point is inside the band. Source:
`validation/ACCEPTANCE.md`; `melt_metrics.DEFAULT_BAND`.

**A refined run exports the fine block only; the coarse far field is dropped, not
resampled.** Nested blocks overlap at two spacings, so they do not tile one grid and
`npz_block_merge.stitch_blocks` refuses them. `select_finest` keeps the finest blocks
and discards the coarse ones. Effect: an NPZ from a refined run covers the refinement
box, **not the whole domain** — smaller extent than an unrefined run of the same
config, and any consumer assuming full-domain coverage sees a cropped field. The
alternative (resampling the coarse block onto the fine grid) was rejected because it
invents data at the interface and would make the archive lossy, which is the one thing
the NPZ format exists to prevent. Source: `python/npz_block_merge.py`
(`select_finest`); `python/export_all_timesteps_to_npz.py` (`merged_grid`).

**Nested-block faces are snapped outward onto parent cell faces.** FLOW-3D does not
reject an interface that cuts through a parent cell; it interpolates across it and the
degradation is silent. The generator therefore floors the low face and ceils the high
face onto the parent lattice before clipping to the domain, so a requested margin is
always met or exceeded, never undershot. Effect: the realised fine box is up to one
parent cell larger per face than `margin_m` asks for. Source:
`python/run_rectangular_case.py` (`nested_mesh_block`).

**The NPZ exporter accumulates through disk-backed memmaps, not RAM.** Each field is
written to a `.npy` memmap in a scratch directory beside the output, filled frame by
frame, then streamed into the compressed NPZ; the scratch directory is removed in a
`finally`. Effect: peak *anonymous* memory is one frame instead of the whole series,
so resolution is not bounded by RAM -- but the export needs ~1.15x the raw series as
**scratch disk**, guarded with an explicit error. `np.savez_compressed` streams from a
memmap in ~16 MB chunks, which is what makes this work rather than relocating the
problem. Source: `python/export_all_timesteps_to_npz.py`.

**A33. Preheat is a thermal-equilibrium initial state: `tempi`, `tvoid` and the
`tbct` boundary series all carry `simulation.initial_temperature_k`.** The
alternative (metal preheated, gas at ambient) is arguably closer to an LPBF
chamber, but equilibrium is the standard furnace-preheat idealization and it
matches what the other two initialization branches already did. Effect: the gas
above the plate starts at plate temperature, so there is no gas-to-metal soak
transient. If a chamber-realistic split
(hot metal, ambient gas) is ever wanted, it is a new config field, not a
reinterpretation of this one. Source: `run_rectangular_case.py`, the three `&fl`
builders; `tests/test_initial_temperature.py` pins `tempi` in all three.
