# Training and evaluation guide

This guide collects the practical rules for training and scoring the `lpno` models
(FNO, RIGNO, GAOT and the baselines) on the FLOW-3D datasets. The conventions behind
the data pipeline (axes, normalisation, fills, channel order) are documented with
their source locations in `assumptions.md`, cited here by ID (A1, A2, ...).
Known issues, with symptoms, causes and workarounds, are in `potential_bugs.md`,
by the BUG-NN IDs that code comments and tests cite.

## Submitting runs

- Every training and evaluation that produces a reported number is submitted
  through `aiida-pino` (`aiida-pino submit train-eval`, `submit eval`).
  The run then carries its configuration, dataset linkage, seed
  and metrics as provenance. `python -m lpno.gpu_spool add` refuses an unregistered
  training or evaluation unless it is a short smoke test (`training.max_steps <= 100`)
  or `--no-aiida` is passed. The workchain's `training_profile` accepts `env` and
  `hydra_overrides`, so anything the spooler accepts can be submitted.
- Always pass `--config-name`; all three entrypoints default to
  `config_NS_energy_PINO` (A20).
- Run the exact command once as a smoke test before a long run: same interpreter,
  environment and config, with `training.max_steps=10` and a small data directory.
  Reaching step 1 shows that the config composes, the scheduler steps and the loss
  builds.
- Leave `custom.cuda_device` unset (or `auto`) on queued runs; the spooler assigns
  the card. Declare `--vram-mb` and `--ram-mb` from a measured peak. A declaration is
  a booking, and nothing caps what a run then allocates (A44).
- Do not set `gpu_seat_priority` on a training. A priority seat waits for a bounded
  time and then fails the job; it suits short checks only.
- Training and evaluation abort without CUDA unless `PINO_ALLOW_CPU=1` is set, which
  suits evaluations and tutorials. Unit tests run on the CPU:
  `python -m pytest -q` from the package root.

## Data loading

- Consolidated NPZ (one `<case>.npz` plus `<case>.manifest.json` per case) is the
  supported input. Direct VTK directories are refused unless `PINO_ALLOW_VTK=1`
  (A36).
- Variable-length datasets (random paths, variable speed, material sweeps) need the
  lazy loader, `PINO_LAZY_NPZ=1`; the eager loader cannot stack cases with
  different frame counts.
- Full-resolution training is loader-bound unless frames are read case by case.
  The working recipe is `custom.case_coherent_sampling: true` together with
  `PINO_LOADER_WORKERS=2` (or 3 for GAOT) and a host-RAM booking that holds the
  cached cases. Either setting alone leaves the GPU idle.
- A cache miss decompresses all frames of all configured fields of a case, because
  NPZ members are DEFLATE streams without random access. Fields that no input or
  output key consumes are pruned automatically; do not hand-trim `vtk_fields`.
- Spatial striding has two routes: point the run at a pre-strided copy of the
  dataset, or set `PINO_LAZY_STRIDE=2,2,1` on a full-resolution copy. Doing both
  strides the grid twice, and the run silently trains on a quarter of the intended
  grid, so check the grid size of one training file before queueing. The stride is
  applied after decompression, so it saves GPU memory and no loader time.
- In-training validation is subsampled by default (`custom.val_max_cases` 5 cases,
  `custom.val_frames_per_case` 10 frames). Scoring runs ignore both caps. If a run
  looks slow, compare the logged validator time multiplied by the number of
  validation passes with the wall time before changing anything else.

## Model-specific settings

- **RIGNO.** Train with `amp.enabled=false`: fp16 autocast can overflow the
  gradients on this data family (`potential_bugs.md`, BUG-43a). The base configuration
  uses `processor_steps: 18` and `num_graph_variants: 8`, which fit a 24 GB card
  only with `grad_checkpointing: true`; the three settings belong together.
  Graph-variant rotation draws one of the pre-built graphs per training forward and
  pins variant 0 at evaluation (A45). At full resolution on the random-path grid
  (150 x 100 x 40) the region-to-physical edge set no longer fits in 24 GB, so
  RIGNO trains at stride 2,2,1. It rebuilds its graph from whatever grid it is
  given, so training strided and evaluating at full resolution is the normal
  configuration.
- **GAOT.** Runs fp32 (A42). Coordinates are normalised per axis, and the latent
  lattice should follow the index aspect of the grid the model receives (A41).
  The encoder and decoder graphs read and write every physical node (A40).
- **FNO.** Spectral layers require a uniform grid. The refined RIGNO variants with
  an FNO head run fp32, since the head's FFT is unstable in fp16.

## Schedules, checkpoints and monitoring

- One-cycle is the default schedule on the flow3D configs. `scheduler.total_steps`
  is derived in code as `training.max_steps + 200`; do not set it by hand. A hydra
  `.hydra/config.yaml` dump shows the composed value before this derivation, so read
  the schedule that ran from `optim_checkpoint.0.pth` (`scheduler_state_dict`).
- One-cycle produces its model only if the anneal completes. Keep
  `stop_criterion.patience` above `training.max_steps` on one-cycle configs; a
  guard warns about the combination. The flow3D configs that declare a stop
  criterion stop on `beambox_mae_T`, the same validation metric that selects the
  best checkpoint.
- For warm-start fine-tuning, an exponential schedule started at or below
  one-cycle's starting learning rate (about 1e-5) with `decay_steps` near 100 was
  stable; starting higher destabilised some seeds. An exponential checkpoint can be
  resumed with a larger `max_steps`; a completed one-cycle run cannot.
- `stop_criterion.freq` must be a multiple of `training.rec_validation_freq`, and
  `rec_results_freq` and `rec_validation_freq` must not exceed `max_steps`; the run
  refuses to start otherwise. Set `training.rec_constraint_freq: 100` so the log
  prints a step heartbeat between validations, and keep `save_network_freq` at a
  few thousand steps so a stopped run leaves usable weights.
- Hard alpha masking switches on at `custom.mask_warmup_steps` (default 500
  training steps). Under one-cycle from scratch the learning rate is still low at
  that point, and masking a model that has not started to learn alpha stops it from
  learning; pull the peak earlier (`scheduler.pct_start` about 0.03) and move the
  warmup past it (for example 4000).
- A run that completed is not necessarily a trained model. Check that the final
  step equals `max_steps`, that the loss fell, that the count of skipped AMP steps
  is a small fraction of all steps, and that validation values change between
  records. Bit-identical validation values, a zero gradient norm or an L2 relative
  error near 1.0 mean the run is not learning.
- Validation metrics in the TensorBoard file are stored as tensors, so read them
  with `EventAccumulator.Tensors`; their values are in normalised units.

## Metrics

- Temperatures are normalised by `ref_scales.T = 3000 K`; multiply MAE and RMSE by
  3000 to report kelvin. `alpha`, `fl`, IoU and relative errors are unitless, and
  relative errors are omitted for `fl` and `alpha` (A12).
- Whole-domain averages are dominated by quiescent metal and can look healthy while
  the melt pool is wrong. Report three interface families alongside them:
  - **Beam box**: errors within two beam radii of the beam centre, inside the
    metal. The validation curve calls it `beambox_mae_T`; an evaluation stores
    `T.beambox_abs_err_mean`, `beambox_rmse`, `beambox_rel_err_mean`,
    `beambox_n` (the voxel count) and `beambox_iou` for `fl` and `alpha`.
  - **Headroom**: `thermal_headroom_K` and `can_reach_melt` give the margin between
    the predicted peak over substrate metal and the melt threshold
    `(T_s + T_l)/2` (1898 K for Ti-6Al-4V). A negative headroom means every melt
    metric will read zero.
  - **Interface distance**: `surface_msd_um` and `surface_hd95_um` measure the
    distance between predicted and true melt and keyhole surfaces in micrometres
    and separate models more sharply than IoU; `surface_pred_empty` flags a model
    that predicted no interface.
- The tail deficit `tail_deficit_max_K` grows with the number of frames N. A model
  that predicts the phase average of a fluctuating field shows a deficit of about
  sigma * sqrt(2 ln N) even when correct, because the per-frame formulation maps
  identical dose histories to the conditional mean over oscillation phases. Compare
  `tail_deficit_per_sqrt2lnN_K` against the dataset's own frame-to-frame sigma.
- Temperature errors are heavy-tailed and concentrated in the thin interface shell,
  so a gain in MAE often comes with a flat RMSE. State which metric a ranking uses.
- A number is comparable only to one scored on the same domain, the same split and
  the same frames. Lab-frame and moving-window evaluations cover very different
  amounts of quiescent metal, and frame subsampling changes every max statistic.
  Check the tensor shape printed by the evaluation and the stored
  `eval_frame_stride` block before comparing two runs. The automatic evaluation of a
  `TrainEvalWorkChain` scores the validation split.
- Look at the prediction where the laser is: pick the frame with the most true
  melt cells, slice through the melt centroid, and plot truth and prediction side by
  side for `T` and `fl`. A domain midplane slice often misses the melt pool.

## Known limitations

Known issues and limitations are listed in `../potential_bugs.md`.

## Roadmap

These directions are open in the code base. They are not part of the manuscript.

- **Autoregressive rollout.** `autoregressive.mode=rollout` trains one-step pairs
  and evaluates by recursive rollout for `fno`, `rigno`, `gaot` and `deeponet` on
  the eager data path (A31), with a multi-step unrolled loss
  (`UnrolledGridConstraint`), previous-state noise (A32) and feedback clamping as
  stability options. This code exists and is tested. A planned extension trains
  direct maps over all pairs `(t, t + tau)` with `tau` as an input, using the
  existing `DeltaArch` wrapper, so that training memory no longer grows with the
  unroll length.
- **Geometry and materials.** Per-case non-dimensionalisation with resampling to a
  canonical grid, per-case material constants for the fill and the `fl`
  derivation, a normalised-enthalpy conditioning channel, and shape bucketing so
  GAOT can train on mixed grids.
- **Performance.** Restaging lazy datasets as memory-mapped `.npy` files to remove
  the cache-miss decompression; chunked, individually checkpointed edge blocks in
  RIGNO's encoder and decoder to train at full resolution; and striding outside the
  beam box only, for the graph models.
- **Losses and baselines.** A grid PDE residual built from finite-difference or
  spectral derivatives of the output, which avoids second-order autograd, and a
  skip-connected coordinate MLP baseline, which needs grid-aware constraint routing
  first.
