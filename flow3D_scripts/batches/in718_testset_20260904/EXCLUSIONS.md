# Frozen IN718 test set: exclusions

Splits are per-simulation and global, so these are applied once at ingest and never
re-applied. `exclude_from_training` is enforced nowhere; only `split=excluded` works,
because the training query selects on `split` alone.

## in718t_010: EXCLUDE (`split=excluded`)

Melt reached **410 µm into a 450 µm plate**: a 30 µm (3-cell) floor margin. The thermal
boundary layer is thicker than three cells, so the solution is plate-bounded, the same
condition flagged as `plate_boundary_affected` in a Ti64 sweep
(melt at 399.998 µm on a 400 µm plate).

It is **in frame**: the deep melt sits under the beam, so it falls inside the 61-cell
beam-centred steady window, and `crop_z_top` is 0 so nothing trims it in z.

It is an LHS fill case (350 W, H 31.6) and no AM-Bench anchor, so excluding it costs
no external validation point. Test set becomes 24 with all 7 anchors intact.

## in718t_021: KEEP, with the reason recorded

Molten metal came within **10 µm (one cell) of the `x_min` wall**, which the checker flags.
Kept because the artifact is **outside the steady window**: `legacy_window_split(61)` gives
behind 41 / ahead 20, and the steady average uses the trailing 30 frames with the beam near
the track end (~x-cell 75), so the window spans roughly 34–90. Cell 1 is excluded for any
beam position ≥ 42.

Consistent with its measurement: depth −3.7 %, width −10.6 % against NIST, both comfortably
in band, because the beam-sectioned cross-section is taken mid-track, far from the wall.

## Why not simply enlarge the domain

A longer x domain is incompatible with the Ti64 base model. The steady window width is
derived from the raw grid (`nx - (Nsteps-1)*super_res` = 90 − 29 = 61), so nx = 120 would
give a 91-cell window and the model was trained on 61. A deeper plate likewise changes the
z extent, and a domain-extent mismatch silently scores gas the model never saw.
Excluding one case is the cheaper and safer correction.
