# Developer notes

## Known limitations

Known issues, with symptom, cause and workaround per P-ID, are in [`potential_bugs.md`](../potential_bugs.md).

## Possible extensions

P-IDs refer to `potential_bugs.md`.

Deterministic scan presets (serpentine, spiral, Hilbert) as `laser.path.type` options on
the existing constant-speed tables; random paths that avoid self-intersection (A5b); a
row-count guard for single-beam paths (P18) and an `--overwrite` that clears old output
(P29); `laser_off_time_s` written into the NPZ, so steady-window scoring needs no
manifest; per-track metrics for multi-beam cases (P23, P24) and a per-case gas crop
(P31); a mesh-convergence study of melt depth and width near the keyhole walls;
viscosity, density, heat-capacity, conductivity and surface-tension tables extended
toward keyhole wall temperatures; a surface-state absorptivity input for Al and Cu, whose
keyhole initiation depends on the flat-surface value; and further sampled process axes:
defocus (which needs the focal plane of A1 to become a config field), temporal power
modulation and chamber pressure.
