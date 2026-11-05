# Assumptions

Non-trivial assumptions and deliberate choices with a real effect on results
or runs. Add or update an entry when you make or rely on such a choice. IDs are
stable because code comments cite them; a missing number is a retired entry.

- **A1 — Drawn-path laser physics come from each model's `meta.json`.** The
  `physics` block sets the P and V ranges and defaults (served random-path
  models: P 68-237 W, V 0.21-0.99 m/s, defaults 150 W and 0.6 m/s). Requested
  values outside the range are clamped and the response sets
  `path_meta.physics_clamped`. The Gaussian beam sigma, 5e-5 m, is the same for
  every model (`BEAM_SIGMA_M` in `app/adapter_randompath.py`). A model folder
  without a `physics` block falls back to fixed physics (150 W, 0.8 m/s, a
  1.5 mm arc-length cap; `app/model_physics.py`), and explicit `p_watts` or
  `v_scan_mps` for it is a 422. Effect: a beam size other than 50 µm, or P/V
  outside a model's trained range, cannot be represented without retraining.
- **A2 — H_norm uses the model's fitted denominator.** H_norm =
  (P/sqrt|V|) / `h_norm_denom`, where the denominator is the training split's
  maximum of P/sqrt|V| (`physics.h_norm_denom` in `meta.json` for the path and
  straight models, `ref_scales.H` in `flow3D_params.yaml` for the steady
  models). A model without a denominator gets H_norm = 1.0, which is only
  correct for a training set at a single P/V. Effect: a missing or wrong
  denominator shifts every H_norm input with no error.
- **A3 — Drawn-path timing is per model.** Models with
  `physics.duration_scaled_times: true` (all three served random-path models)
  span the real weld duration, path length / V, on their training step
  `physics.dt_s` (1e-5 s), with render frames picked from that grid (A23).
  A model without the flag was trained on one fixed 2 ms window and keeps it;
  any other window is out of distribution for it. Effect: on a fixed-window
  model, a short drawn path has post-laser-off frames with the beam parked at
  the tip, and a long one is cut off at 2 ms.
- **A4 — Keep-out box = substrate inset by 4 beam radii.**
  (±0.55, ±0.30) mm, the same margin the training-path generator enforced.
  Effect: drawn paths are clamped there; tips never approach domain walls.
- **A5 — Input channels are read from each checkpoint's Hydra config.** The
  adapter builds exactly the channels listed in `custom.FNO_input_keys`, in
  that order, and appends `t_hat = time_s·(V/L)` because every served model was
  trained with `time_as_batch`. The served models also use five process
  channels (`preheat_T0`, `beam_shape_uniform`, `focus_factor`,
  `h_norm_local`, `conduction_field`), built by the model code in `ext/lpno`.
  At load, `ref_scales` and the material `T_s`/`T_l` are checked against the
  adapter's constants, and a checkpoint asking for a channel the adapter
  cannot build is a hard error.
- **A6 — Preset ground truth is recomputed server-side** from the stored
  FLOW-3D NPZ with the evaluation masking (T filled toward 3123 K where the
  fluid fraction is below 0.5, fl gated by the masked alpha; `app/gt.py`).
  Effect: GT and prediction share one postprocess, but the masking must track
  any future change in the evaluation preprocessing.
- **A7 — Single-flight GPU lock; timeouts do not preempt.** A 504 releases
  the lock while the worker thread keeps running; lazy model loading is
  separately serialized by `_load_lock` in the adapter. Effect: a retried
  first request waits instead of double-loading.
- **A8 — Prediction caches are per-route bounded LRUs**
  (`prediction_cache_entries`, default 8), keyed by
  (model, sha256(resampled points + times)). Effect: z-slice changes and 3D
  re-renders of the same path skip inference, and switching between a few
  recent paths does not re-infer.
- **A9 — Ambient differs by dataset; the 2D render clim floor is 300 K.** The
  straight-track simulations start at 273.15 K and the free-form ones at
  293.15 K, and the adapters build `preheat_T0` from those values. The 2D
  slice/path clim [300, 3200] K hides near-ambient structure, including any
  sub-ambient undershoot of the prediction. The 3D block render uses a 0 K
  floor instead (A13).
- **A10 — `LPFNO_PATH_INFER_BATCH` defaults to 1.** Batch 2 roughly doubles
  RIGNO's peak memory for a few percent of speed, which can fail on a GPU
  shared with other jobs. Raise it only on an idle GPU (outputs match batch 1
  to ~0.04 K).
- **A11 — Frame rendering is serial by design.** A thread pool measured
  ~1.7x SLOWER (matplotlib Agg holds a class-level draw lock; noted in
  `app/service.py`). The upgrade path is a process pool.
- **A13 — render3d clim spans GT and prediction.** The run-wide colorbar range
  is [0 K, maximum displayed metal temperature over BOTH the ground-truth and
  prediction series across all frames] (`_pyvista_block_frames` +
  `_block_run_clim` in `app/render.py`). The block is colored by the raw served
  temperature on the cells with alpha >= 0.5, with no fill inversion. Effect:
  the two panels are directly comparable, but the GT keyhole-wall peak usually
  sets the maximum, so the prediction panel renders darker than the same
  request without GT (the models smooth the keyhole peak).
- **A15 — Colorbars are constant across the time series.** Every `/infer_path`
  and `render3d` colorbar is fixed over all frames, never rescaled per frame:
  GT/prediction 2D panels use the fixed field clim (`_path_clim`), the 3D block
  uses one run-wide clim (A13), and the 2D "Prediction - GT" diff panel uses a
  run-wide symmetric range `[-dmax, +dmax]` with `dmax` the max absolute diff
  over ALL frames (computed in `service.py`, passed to `PathFrameRenderer` as
  `diff_dmax`, set once in `__init__`). Effect: frame-to-frame color is
  comparable; a single hot frame does not restyle the whole series.
- **A16 — Straight-line track geometry is fixed; only P/V vary.** Every
  straight case is the laser moving from (-0.300 mm, 0) to (+0.29724 mm, 0)
  (length 0.59724 mm) with total time = length / V. `StraightPathAdapter`
  (`app/adapter_straightpath.py`) subclasses `RandomPathAdapter` and serves
  `fno_cal`, `gaot_cal` and `rigno_cal` on the 90x40x40 grid (RIGNO on its
  90x40x30 training crop, pinned by the folder's `grid.npz`), always 20 frames.
  H_norm uses `physics.h_norm_denom` from `meta.json` (A2); the 203 fallback
  `STRAIGHT_H_REF` applies only to older checkpoints without it. The dose and
  conduction channels are time integrals, so they are integrated on a 5 µs grid
  (the simulations' export step) and the 20 frames are picked from it. Effect:
  the page covers straight tracks at any P/V in the slider range, but physics
  off the straight-track training distribution is unvalidated.
- **A17 — Straight/steady ground truth is served only on a P/V match.** GT is
  returned when the requested (P, V) is within 1 W and 0.005 m/s of a catalog
  case (`app/gt.py` `nearest_pv`, `MATCH_DP_W` / `MATCH_DV_MPS`, tie-break
  `dP^2 + (200 dV)^2`); the frontend's clickable catalog points send the exact
  case P/V, so a click always matches while a free drag usually does not. Both
  pages draw on one 123-case catalog in train, val and test folders: the
  straight cases are per-case NPZ files, the steady cases the per-split
  `var_1_time_steps30_avg30` time-averaged caches, searched test first
  (`steady_gt.SPLITS`). Off-match requests render prediction-only.
- **A18 — The steady comparison is made on the matched case's own window.** The
  steady models predict on a beam-following window whose x position varies
  from case to case. When a GT case matches, the server shifts the prediction
  window to that case's x axis before the forward pass
  (`_steady_window_shift` in `app/service.py`). The 2D comparison draws GT and
  prediction as XZ panels on their own millimetre axes with a shared colorbar;
  the 3D comparison and the reported `gt_mae`/`gt_rmse` resample the GT
  trilinearly onto the prediction grid and score the substrate (z <= 0).
  Effect: if the GT and the runtime window differ in cell count, no shift is
  applied and the comparison includes a window offset (a warning is logged).
- **A19 — Comparison plots name the GT split; the straight 2D view has two
  planes.** Every GT panel title carries the matched case's split, "Ground
  truth (test)" etc., via a `gt_split` passed to the renderers. Steady and
  straight GT can come from train, val or test (A17); random-path presets come
  from each model's own test split (`physics.presets_dir`). The
  `/infer_straight` (and `/infer_path`) 2D view takes `PathView.plane`: "xy" is
  the top view at a z-slice (the drawn path is overlaid, vertical axis y); "xz"
  is a cross-section at a y-slice (vertical axis z, no path, beam marker at the
  surface), matching the steady page's cross-section. The diff/colorbar logic
  is unchanged.
- **A20 — Top view / cross-section is a `plane` on every 2D and 3D route.**
  `plane` ∈ {`xy` top view, `xz` cross-section} is on `InferView` (steady
  `/infer`, default `xz`) and on the two render3d requests (default `xy`). In 2D
  it picks the slice (xy = top surface, xz = y-slice). In 3D, `xz` cuts the
  block/surface at its y-midplane to expose the interior, while `xy` keeps the
  whole block and looks down at the surface. The straight 3D `xz` clips the
  metal block and is not smoothed (smoothing would round the flat cut face).
  Effect: every page offers top view and cross-section in both 2D and 3D.
- **A23 — /infer_path frame times are a cadence, not a count.** Frames span the
  weld duration at `frame_dt_s` (default 1.3e-4 s), so the frame count is a
  consequence of length/speed rather than a fixed 20. The default is set on the
  reference weld (1.5 mm at the variable-path model's default 0.6 m/s =
  2.5 ms -> exactly 20 frames) and is 13x that model's 1e-5 training dt, so
  rendered frames are samples of the dose grid rather than rounded onto it:
  `times_fine` carries the dose integral at the training dt and `frame_idx`
  indexes into it, so a render frame MUST be a member of that grid. The count
  is clamped to 5..100: the floor keeps an animation meaningful, the ceiling
  bounds render cost (rendering, not inference, dominates). Because the count
  is an integer and frames snap to the grid, the effective step differs from
  the requested one and is what `path_meta.frame_dt_s` reports. Fixed-duration
  models are exempt by design: their window is a constant 2 ms, so Δt was never
  ambiguous for them, and their default step is their training cadence
  (2 ms / 19) which reproduces the historical 20 frames.
  `app/adapter_randompath.py` (frame_count_for, default_frame_dt_s).
