#!/usr/bin/env python3
"""Single source of truth for the rectangular-case JSON config schema.

``CONFIG_SCHEMA`` documents every config value the runner validates; the README
table is generated from it (see ``run_rectangular_case.py --print-config-schema``)
and a test checks the two match, so the docs cannot drift from this list.

``GEOMETRY_DEFAULTS`` holds the literal default magnitudes the runner falls back
to, so a documented default lives in one place rather than being repeated as a
magic number in ``load_and_validate``.
"""

from __future__ import annotations

# Default magnitudes the runner uses when an optional geometry field is omitted.
# Kept here so the value and its documentation share one home.
GEOMETRY_DEFAULTS = {
    "length_m": 0.0009,
    "width_m": 0.0004,
    "thickness_m": 0.0003,
    "gas_clearance_m": 0.0001,
}

# (JSON value, meaning, requirement enforced by the runner).  Order is the order
# rendered into the README table.
CONFIG_SCHEMA: list[tuple[str, str, str]] = [
    ("`material`", "Material name, path, or object", "Defaults to `ti_imi318_mills`; see material files below"),
    ("`parallel.cores`", "Total local CPU cores allocated to this run", "Positive integer; defaults to 1"),
    ("`parallel.threads_per_rank`", "OpenMP threads per MPI rank", "Positive integer; must divide `cores`; defaults to 1"),
    ("`geometry.length_m`", "Full x extent", f"Positive; defaults to `{GEOMETRY_DEFAULTS['length_m']}`"),
    ("`geometry.width_m`", "Full y extent", f"Greater than 2 beam radii; defaults to `{GEOMETRY_DEFAULTS['width_m']}`"),
    ("`geometry.thickness_m`", "Solid depth below `z=0`", f"Positive; defaults to `{GEOMETRY_DEFAULTS['thickness_m']}`"),
    ("`geometry.gas_clearance_m`", "Mesh height above the surface", f"Positive; defaults to `{GEOMETRY_DEFAULTS['gas_clearance_m']}`"),
    ("`geometry.placement`", "`laser_leading_margin` or `centered`", "Defaults to `laser_leading_margin`"),
    ("`geometry.bounds_m`", "Compatibility-only explicit `x`, `y`, `z` mesh bounds", "Each `[min,max]` pair must be increasing"),
    ("`geometry.initialization`", "Compatibility mode: `legacy_halfspace`, `substrate_with_gas`, or `filled_mesh`", "Defaults to `legacy_halfspace`"),
    ("`geometry.surface_z_m`", "Free-surface height (`fzh(1)`): where the metal/gas interface sits", "Defaults to `0.0`; must lie within the mesh z bounds"),
    ("`mesh.cell_size_m`", "Nominal Cartesian cell size", "At most `beam_radius_m / 2`"),
    ("`mesh.refinement`", "Optional nested finer block around the laser track", "Omit for a single uniform mesh"),
    ("`mesh.refinement.factor`", "Fine cell = `cell_size_m / factor`", "Integer `2` or `3`"),
    ("`mesh.refinement.margin_m`", "How far the fine block extends past the beam footprint", "Positive; single-rank runs only"),
    ("`laser.power_w`", "Laser power", "Non-negative; required unless `laser.beams` is set (then the default for beams that omit `power_w`)"),
    ("`laser.beam_radius_m`", "Beam radius at the surface (`rflbm`); sets the mesh-size and margin checks", "Positive; required"),
    ("`laser.focal_distance_m`", "Focal distance (`dlflbm`, also the beam origin height `zb0lbm`)", "Positive; required"),
    ("`laser.lens_radius_m`", "Beam radius at the lens (`r0lbm`)", "Positive; required"),
    ("`laser.start_point_m`", "Surface focal-point start `[x,y,0]`", "Defaults to `[0,0,0]`; z must be zero"),
    ("`laser.end_point_m`", "Surface focal-point end `[x,y,0]`", "Must be one radius inside all edges"),
    ("`laser.scan_speed_m_per_s`", "Magnitude of the in-plane scan velocity", "Positive; held constant along the path (nominal/reference speed for `variable_speed`, whose per-leg speeds come from `times_s`)"),
    ("`laser.dwell_time_s`", "Stationary (dwell) spot: hold `start_point_m` for this long", "Optional; positive. Zero in-plane velocity, `scan_path_length_m` = 0; omit `end_point_m` (or set it equal to the start) and `scan_speed_m_per_s` becomes optional. Used for keyhole-onset experiments (e.g. Simonds et al. 2021)"),
    ("`laser.path`", "Optional generalized in-plane scan path; overrides `start_point_m`/`end_point_m`", "Object; forces `centered`/`bounds_m` placement; every sampled point must stay at least 4 beam radii inside all x/y edges (hard check)"),
    ("`laser.path.type`", "`waypoints`, `sinusoidal`, `random`, or `variable_speed`", "Required when `laser.path` is set"),
    ("`laser.path.knots_m`", "`variable_speed`: ordered `[x, y]` beam positions (kept verbatim, no arc-length resample)", "At least two; a repeated knot with advancing time is a dwell"),
    ("`laser.path.times_s`", "`variable_speed`: arrival time at each knot", "One per knot; strictly increasing; first must be 0. Per-leg speed = chord / Δt (0 for a dwell)"),
    ("`laser.path.power_w`", "`variable_speed`: optional per-leg laser power (e.g. to de-rate dwells)", "One value per leg (`len(knots_m) - 1`); non-negative; defaults to `laser.power_w` on every leg"),
    ("`laser.path.points_m`", "`waypoints`: ordered `[x, y]` surface points", "At least two; z must be zero; joined per `interpolation`"),
    ("`laser.path.interpolation`", "`waypoints` join: `linear` polyline or `spline` (Catmull-Rom through the points)", "Defaults to `linear`"),
    ("`laser.path.length_m`", "`sinusoidal` travel length / `random` path length (reached exactly; the walk self-intersects if the domain is smaller)", "Positive; `sinusoidal` requires it; `random` defaults to the domain length (x extent)"),
    ("`laser.path.amplitude_m`", "`sinusoidal`: transverse (y) amplitude about the +x travel axis", "Non-negative"),
    ("`laser.path.wavelength_m` / `periods`", "`sinusoidal`: spatial period along x, or number of periods over `length_m`", "Give exactly one"),
    ("`laser.path.seed`", "`random`: RNG seed for the reproducible coverage-biased walk", "Integer; defaults to 0"),
    ("`laser.path.step_m` / `max_turn_deg`", "`random`: control-point step and max per-step heading change (open-space smoothness; wall recovery may turn sharper)", "Default `4*beam_radius` and `45`"),
    ("`laser.path.start_point_m`", "`sinusoidal`/`random` start `[x, y, 0]`", "Defaults to `[0,0,0]` (sinusoidal) or the cropped region's left edge, mid height (random)"),
    ("`laser.path.sample_spacing_m`", "Arc-length spacing of the constant-speed velocity-table samples", "Defaults to `beam_radius/2`; capped at 500 samples"),
    ("`laser.turn_off_at_end`", "Stop the laser at the end point", "Defaults to `true`; set `false` for one-row legacy tables"),
    ("`laser.plume_radius_m`", "Override RFPLUME: the plume cylinder radius that engages the plume model", "Optional; positive, and requires a non-zero `laser.plume_attenuation`. With the profile default `rfplume=0` the model never engages. When engaged, energy audits show the diverted power leaves the energy balance: the melt receives ~aplume less beam power. Retired from all validated recipes"),
    ("`laser.plume_attenuation`", "Override APLUME: fraction of beam power removed by the plume model", "Optional; `0 <= value < 1`. Profiles ship `0.0`. Acts as a calibrated beam-power sink (a near-uniform depth cut in matched-pair energy audits). Retired from all validated recipes; kept for future use"),
    ("`laser.plume_condensation`", "Override ACONDENS, documented by the vendor as recondensing the plume-diverted power", "Optional; `0 <= value <= 1`, requires a non-zero `laser.plume_attenuation`. Measured ablations return nothing at either setting, so APLUME is a loss regardless. Profile default 1.0"),
    ("`laser.heat_distribution`", "`uniform` or `gaussian`", "Defaults to `gaussian` (keyhole model)"),
    ("`laser.beam_profile`", "Optional multi-beam radial approximation for ring/tabular profiles", "Object; `type` is `single`, `ring`, `radial_table`, or `tabular`. Expands to co-moving beamlets that conserve total `power_w`; currently straight/dwell paths only"),
    ("`laser.beam_profile.samples`", "`radial_table`/`tabular` radial shells", "Non-empty list of `{radius_fraction, weight}` objects; `radius_fraction` in [0,1], weights non-negative and positive in total"),
    ("`laser.beam_profile.angular_samples`", "Number of azimuthal beamlets for a ring/shell", "Integer 3-64; may be overridden per radial-table sample"),
    ("`laser.beam_profile.beamlet_radius_fraction`", "Radius of each beamlet as a fraction of `beam_radius_m`", "Positive and <=1; beamlet radius plus radial offset must fit within the nominal beam radius"),
    ("`laser.beams`", "Independent beams (`nlbeam` up to 4), each `{path, power_w?, scan_speed_m_per_s?, turn_off_time_s?}` with its own path on one shared time grid", "List of 1-4 objects; `path` is required per beam and uses the `laser.path` types; cannot be combined with `laser.path` or `laser.beam_profile`; top-level `power_w`/`scan_speed_m_per_s` are the per-beam defaults"),
    ("`laser.fresnel_from_material`", "Derive the Fresnel coefficient ε (`ereflct`) from the material's optical data", "Defaults to `true` when the material has optical data (resistivity CSV or absorptivity); prefers `optical.electrical_resistivity_ohm_m` (Drude) then `optical.normal_absorptivity`. Set `false` for the fixed scalar reflectivity"),
    ("`laser.wavelength_m`", "Laser wavelength for the Drude Fresnel derivation", "Defaults to `1.064e-6` (fiber/Nd:YAG)"),
    ("`laser.radiation_distribution_flag`", "`ilbeam`: FLOW-3D beam radiation distribution option", "Integer; defaults to `2`; rendered only by the `standard` solver profile"),
    ("`laser.reflection_flag`", "`irefopt`: beam reflection model", "Integer; defaults to `3` (Fresnel, the model `fresnel_from_material` feeds)"),
    ("`laser.max_reflections`", "`nrefmx`: ray-trace reflection count cap", "Integer; defaults to `20`"),
    ("`laser.reflection_limit`", "`qreflim`: fraction of beam power below which a reflected ray is dropped", "Positive; defaults to `0.01`"),
    ("`laser.absorptivity_flag`", "`icbmengy`: absorptivity option", "Integer; defaults to `0`; rendered only by the `standard` solver profile"),
    ("`laser.absorptivity`", "`cbmengy`: fixed beam absorptivity", "Non-negative; defaults to `1.0` (full absorption; the Fresnel model then sets the effective value)"),
    ("`laser.reflectivity`", "`ereflct`: Fresnel coefficient ε when `fresnel_from_material` is off", "Non-negative; defaults to `0.35`; ignored when the material supplies the coefficient"),
    ("`laser.vapor_pressure_a`", "`avprslbm`: vapour-pressure curve coefficient A", "Positive; defaults to `50650.0`"),
    ("`laser.vapor_pressure_b`", "`bvprslbm`: vapour-pressure curve coefficient B", "Positive; defaults to `10.95913`"),
    ("`simulation.initial_temperature_k`", "Initial metal and gas temperature (`tempi`/`tvoid`)", "Positive; required"),
    ("`simulation.finish_time_s`", "End time", "Must include the scan and the stop event"),
    ("`simulation.output_frames`", "Number of exported spatial (selected-data) frames written to the result file", "Positive integer; works with every solver profile (when set, it replaces the keyhole/legacy profiles' fixed `tapltd(1)=5e-06`); omit to keep the profile's interval or the FLOW-3D default (~100). Renders `tapltd(1) = finish_time_s / output_frames`"),
    ("`simulation.boundary_temperature_k`", "Boundary temperature", "Defaults to `initial_temperature_k`"),
    ("`simulation.accommodation_coefficient`", "Override RSIZE, the evaporation accommodation coefficient", "Optional; 0 < value <= 1. Scales the phase-change rate and the recoil amplitude (`A = RSIZE*PV1`). Profiles hardcode 1.0; literature 0.1-0.5. Ablations show it is self-compensating and saturating, so it cannot serve as a depth-calibration lever; keep the default"),
    ("`simulation.recoil_model`", "`if_prsrecoil`: recoil pressure from the vapour-pressure coefficients (`1`) or the evaporation mass flow (`2`)", "Optional; `1` or `2`; the rendered default is `1`"),
    ("`simulation.void_vapor_pressure_pa`", "`pv0`: vapour partial pressure in void regions", "Optional; non-negative. Profile default is `101300` (`standard`) or `50650` (`keyhole`, `legacy_case_fine_1`)"),
    ("`simulation.ambient_pressure_pa`", "Pressure on the five open mesh boundaries (`pbct(1,1..5)`, plus `pbct(1,6)` and `p_reference` when set)", "Optional; positive; the template default is `101300.0`"),
    ("`simulation.solver_profile`", "`keyhole`, `standard`, or `legacy_case_fine_1`", "Defaults to `keyhole` (case_fine_1 keyhole tuning, keeps surface tension); `legacy_case_fine_1` zeroes surface tension for the line-track reproduction"),
    ("`simulation.subprocess_id`", "FLOW-3D subprocess number", "Defaults to `0`; legacy line-track reproduction uses `1`"),
    ("`simulation.max_iterations`", "Solver safety cap", "Positive integer"),
]

SCHEMA_BEGIN = "<!-- BEGIN GENERATED CONFIG SCHEMA -->"
SCHEMA_END = "<!-- END GENERATED CONFIG SCHEMA -->"


def to_markdown() -> str:
    """Render CONFIG_SCHEMA as the README parameter table."""
    lines = [
        "| JSON value | Meaning | Requirement enforced by the runner |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {value} | {meaning} | {requirement} |" for value, meaning, requirement in CONFIG_SCHEMA]
    return "\n".join(lines)


if __name__ == "__main__":
    print(to_markdown())
