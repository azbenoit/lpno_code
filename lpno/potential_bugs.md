# Known issues

Ways the package can fail or mislead, with the symptom you see, the cause, the
current status or workaround, and where in the code to look. The conventions the
entries refer to are in `assumptions.md` (A1, A2, ...); the practical training and
evaluation rules are in `docs/training_and_evaluation.md`.

IDs are stable because code comments and tests cite them. BUG-43 names two
different issues, listed as BUG-43a and BUG-43b.

## Common traps

Live traps a new user is most likely to meet. Each links to its full entry below,
except BUG-52, which is described here.

### BUG-52. Beam-box MAE depends on the number of cells in the box

- **Symptom.** Two evaluations report beam-box errors over different numbers of
  cells (`beambox_n`), so their means are taken over different regions.
- **Cause.** The box is the substrate within two beam radii of the beam, built from
  the evaluation's own input tensors. Its cell count changes with the served grid,
  the z crop (`crop_z_top` removes metal from it) and the frames scored, and
  differences between models evaluated on the same served grid have also been
  seen without a recorded cause.
- **Workaround.** Compare `beambox_n` beside any beam-box MAE, and treat two numbers
  as comparable only when their counts agree. Substrate MAE is unaffected.
- **Code.** `models/pipeline/postprocessing.py` (`beam_box_from_input`,
  `beam_box_mask`).

### Other common traps

- **fp16 on RIGNO trains nothing** (BUG-43a): set `amp.enabled=false`;
  `config_flow3D_RIGNO_randompath` still enables amp.
- **Older checkpoints fail to load** after an architecture default changed (BUG-54):
  pin the values from the model's `.hydra/config.yaml`.
- **An evaluation far worse than its own validation curve** (BUG-55, BUG-56): an
  unbounded `alpha` or `T` in a few cells dominates the RMSE.
- **Good `T` and `alpha`, empty melt pool** (BUG-59): report `fl` IoU and melt volume.
- **Priority seats fail long jobs** (BUG-58): never set `gpu_seat_priority` on a
  training.
- **Optimistic validation curves** (BUG-67): pass `custom.val_data_dir`.
- **Different `crop_z_top` at train and evaluation** (BUG-62): shifts `h_norm_local`
  and changes `beambox_n` (BUG-52).

## Training

### BUG-43a. fp16 autocast can make a RIGNO training apply no updates

- **Symptom.** The grad scaler backs off on every step, the loss and validation
  curves stay flat, and the run finishes with a checkpoint equal to its
  initial weights.
- **Cause.** On this data family the scaled gradients overflow fp16 (the
  loss sum is of order 1e3 against an initial scale of 65536). Edge masking is
  not required: runs with and without it give identical scaler trajectories. A
  short series of back-offs at the start (about 20 halvings) is the scaler
  finding its scale and is harmless; the failure is a scale that never recovers.
- **Status.** Open; contained. A warning prints when a rigno-family training
  composes with `amp.enabled=true`, and `AmpScalerCollapse` aborts a run after 50
  consecutive skipped steps (`custom.amp_max_consecutive_skips`; opt out with
  `custom.amp_collapse_guard=false`; the scale-floor check `custom.amp_scale_floor`
  is off by default because a low scale with progress is not a collapse). The
  message aggregation runs in fp32 under autocast. Train RIGNO with
  `amp.enabled=false`; `config_flow3D_RIGNO_randompath` still enables amp.
- **Code.** `models/pipeline/amp_guard.py`,
  `models/pipeline/config_loading.py` (`_warn_if_amp_with_rigno_edge_masking`),
  `models/rigno_arch.py` (`InteractionStep`). Tests: `test_amp_collapse_guard.py`,
  `test_amp_masking_warning.py`, `test_rigno_scatter_mean_equiv.py`.

### BUG-37. relobralo can produce NaN gradients for RIGNO on per-frame data

- **Symptom.** `RuntimeError: Function 'MulBackward0' returned nan` inside the loss
  aggregator at a deterministic step, independent of learning rate, clipping and
  edge masking.
- **Cause.** relobralo weights each term by an exponential of loss ratios; when a
  term collapses toward 0 on a sample, the backward computes `0 * inf`.
- **Workaround.** Use `loss: sum`, as the RIGNO random-path configs do.
  `config_flow3D_RIGNO` (time-averaged data) still uses relobralo. Reproduce with
  `custom.detect_anomaly=true`.
- **Code.** `conf/hydra/loss/relobralo_tuned.yaml`, the aggregator in physicsnemo.

### BUG-63. Older models may have stopped at half their step budget

- **Symptom.** A training reports exit 0 while `steps_trained` is well below
  `max_steps`.
- **Cause.** Earlier configs stopped on `l2_relative_error_T` with a patience below
  `max_steps`, while the best checkpoint was selected on `beambox_mae_T`; the stop
  could fire while the selection metric was still improving.
- **Status.** Fixed for new runs: every flow3D config that declares a stop
  criterion stops on `beambox_mae_T` with patience above `max_steps`
  (`tests/test_stop_criterion_budget.py`). Check `steps_trained` on older models.
  The upstream `StopCriterion` accepts a single metric only.

### BUG-41. The RIGNO FNO refinement head adds spectral ringing

- **Symptom.** `custom.model_family=rigno_refined` with the FNO head lowers region
  MAE but adds low-amplitude structured waves to the thermal wake, strongest in
  late frames. MAE does not show it; a field plot does.
- **Cause.** The head keeps only the lowest modes (`[12, 12, 8]` on 75x50x40), so a
  local correction of the melt pool is spread through global modes; the FFT runs
  without domain padding on a non-periodic field.
- **Status.** Open. Nyquist modes with FFT padding
  (`config_flow3D_RIGNO_refined_fno_nyquist_randompath`) make the ringing worse.
  The U-Net head (`config_flow3D_RIGNO_refined_unet_randompath`) removes it and
  over-smooths, which costs accuracy. An H1 (gradient) loss on the head is untested.
- **Code.** `models/rigno_arch.py` (`_FNORefineHead`, `_UNetRefineHead`,
  `RIGNORefinedArch`). Test: `test_unet_refine_head.py`.

### BUG-54. Changing an architecture default orphans older checkpoints

- **Symptom.** Evaluating, warm-starting or ensembling an older model fails with
  `Missing key(s) in state_dict` or `Failed to load evaluation checkpoint`, and the
  workchain reports a generic failure.
- **Cause.** The weights are loaded strictly into a model built from the composed
  configuration, and a default that changes the parameter set (RIGNO
  `processor_steps`, `num_graph_variants`; GAOT `use_geoembed_encoder`) no longer
  matches the checkpoint.
- **Workaround.** Pin the values from the model's own `.hydra/config.yaml` through
  `hydra_overrides`, for example
  `["++arch.rigno.processor_steps=8", "++arch.rigno.num_graph_variants=1"]`. Prefer
  a new config name to editing a shared default when older models still need
  scoring.
- **Code.** `models/pipeline/model_domain.py`, `Arch.load` in physicsnemo.

### BUG-62. `h_norm_local` changes slightly with the z crop

- **Symptom.** A model trained with one `PINO_LAZY_CROP_Z_TOP` and evaluated with
  another sees the `h_norm_local` channel shifted by about 0.8 % on the kept
  planes (more on preheat datasets).
- **Cause.** `_soak_field` treats the z extents as Dirichlet faces, so the top face
  moves with the crop. The other path channels are crop-invariant.
- **Workaround.** Train and evaluate with the same `crop_z_top`.
- **Code.** `custom_utils/npz_datasource.py` (`_soak_field`). Test:
  `test_crop_channel_registration.py`.

### BUG-65. `predict_delta` with GAOT clamps the increment

- **Symptom.** Under `custom.autoregressive.predict_delta=true` with
  `model_family=gaot`, `alpha` can never decrease and `T` can never cool.
- **Cause.** `DeltaArch` adds the base output to the previous state, and GAOT clamps
  its own output (`alpha`, `fl`, `VOF` to [0, 1], plus `output_bounds`) before the
  addition, so the clamp acts on the signed increment.
- **Status.** Open; RIGNO is unaffected. Do not combine the two.
- **Code.** `models/rigno_arch.py` (`DeltaArch`), `models/gaot_arch.py`.

### BUG-66. Mixed spatial grids in one dataset

- **Symptom.** A dataset whose members have different extents fails with a shape
  error at the first case (or validation case) of a new shape.
- **Status.** RIGNO keeps one graph per grid shape and trains on mixed grids with
  `batch_size_grid=1`. GAOT's latent lattice is fixed by config and does not
  support mixed grids, and mixed-shape batches (`batch_size_grid>1`) are not
  supported for any family.
- **Code.** `models/rigno_arch.py` (`_build_graph`), `models/gaot_arch.py`.

### BUG-67. Single-directory standard splits overlap train and validation

- **Symptom.** Validation curves look optimistic on a run given only `custom.data_dir`.
- **Cause.** `get_splits('standard')` validates on a subset of the training indices
  when no separate validation directory is given (A22).
- **Workaround.** Always pass `custom.val_data_dir`.

## Evaluation and metrics

### BUG-04. H_norm denominator of older models

- **Symptom.** A model scored on a different directory sees `H_norm` on a scale it
  was not trained on, and its errors rise.
- **Status.** Fixed. The denominator is fitted once on the training split and
  reused for validation and evaluation; training writes
  `<network_dir>/h_norm_denom.json`. An evaluation uses, in order, an explicit
  `custom.h_norm_denom`, that record, a refit on the training split named by the
  model's archived config, and otherwise the scored split with a banner
  (`[h_norm] scored_split: 1.0` in `model_errors.txt`). aiida-pino backfills the
  record for older models (A9).
- **Code.** `models/pipeline/preprocessing_datasets.py`,
  `models/pipeline/h_norm_record.py`. Tests: `test_preprocessing_velocity_scale.py`,
  `test_h_norm_eval_uses_training_denominator.py`, `test_h_norm_stash_convention.py`,
  `test_h_norm_denom_override.py`.

### BUG-55. The evaluation masks every output with the predicted alpha

- **Symptom.** An evaluation RMSE far above the final validation record of the same
  run, with an `alpha` RMSE above 1.
- **Cause.** Each output, `alpha` included, is filled through a sharp sigmoid of the
  predicted `alpha`, so an `alpha` far outside [0, 1] keeps itself and enters the
  metrics. The final projection is unbounded.
- **Status.** Open for FNO and RIGNO; GAOT clamps `alpha`, `fl` and `VOF` to [0, 1]
  with a straight-through gradient. Cross-check every evaluation against the
  final validation record.
- **Code.** `models/pipeline/postprocessing.py` (`_apply_prediction_postprocessing`),
  `custom_utils/utils.py` (`fill_where_mask_smooth_np`).

### BUG-56. Nothing bounds `T` from above

- **Symptom.** `T` RMSE hundreds of times the MAE, from a few cells per million
  carrying unphysical temperatures.
- **Cause.** Every family predicts `T` with an unbounded linear projection, and alpha
  masking does not cap it: in the metal the raw prediction passes unchanged, and in
  the gas a small fraction of it leaks through the fill.
- **Status.** GAOT has `output_bounds` (a straight-through ceiling), which the flow3D
  GAOT configs set; FNO and RIGNO have none.
- **Code.** `models/gaot_arch.py` (`output_bounds`).

### BUG-59. `fl` is derived, so `T` and `alpha` can look good while melt is empty

- **Symptom.** Low `T` MAE and an `alpha` IoU near 1, with an `fl` IoU near 0 and an
  empty predicted melt pool.
- **Cause.** `fl` comes from the predicted `T` (A13) and never enters the loss, so
  nothing penalises a model whose hot cells fall outside what it predicts as metal.
  The boiling-temperature fill of off-metal cells (A6) makes the hottest target
  cells gas cells, which encourages exactly that misalignment; a per-case fill
  (`mat_Tv`) fixes its magnitude and keeps the association.
- **Workaround.** Report `fl` IoU and melt volume with every comparison, and read
  `can_reach_melt` as a peak-temperature test.
- **Code.** `models/pipeline/postprocessing.py` (`_derive_fl`),
  `models/pipeline/preprocessing_datasets.py` (`_preprocess_var`).

### BUG-51. A model that never predicts gas is not flagged

- **Symptom.** A model predicts `alpha >= 0.5` almost everywhere and still scores a
  low `alpha` MAE, because the gas is a small fraction of the cells. The
  prediction-side boiling fill is gated on the predicted `alpha`, so the gas stays
  cold and the beam-box error rises.
- **Cause.** The validation metrics meant to catch this (`gas_true_n`,
  `gas_recall_alpha`, `gas_pred_fraction_alpha`) do not reach the validation CSV:
  `gas_alpha_validation_metrics` returns `{}` without logging when `alpha` is
  missing from its arguments or the shapes differ.
- **Workaround.** At evaluation, count predicted against true `alpha < 0.5` cells.
- **Code.** `models/pipeline/validation_beambox.py` (`gas_alpha_validation_metrics`,
  `_wrap_l2`).

### BUG-57. Multi-material datasets and single-material constants

- **Symptom.** On a dataset that varies the alloy, `fl`, `melt_threshold_K` and
  `can_reach_melt` use one alloy's solidus and liquidus, and the off-metal fill one
  boiling point.
- **Status.** Partly fixed. The `fl` derivation and the melt threshold use the
  per-case `mat_Ts`/`mat_Tl` channels when the batch carries them, then a melt window
  shared by the whole dataset, then `material_params`; the off-metal fill is per
  case with `mat_Tv`. Without these channels a multi-material run falls back to
  the run-level `material_params`.
- **Code.** `models/pipeline/postprocessing.py` (`metric_melt_threshold_K`,
  `_derive_fl`), `custom_utils/utils.py` (`boiling_temperature`).

### BUG-64. The lazy loader refuses a moving reference frame

- **Symptom.** `ValueError: [NpzDirStore] lazy NPZ loading does not support
  moving_ref_frame=true.`
- **Cause.** The lazy path skips the eager moving-frame transform. Every staged
  dataset uses the lazy path, so models trained in the moving frame cannot be
  scored through it.
- **Workaround.** Use lab-frame configs (A28). Feeding such a model lab-frame data
  through a frame-transfer option gives it a grid it never trained on.
- **Code.** `models/pipeline/npz_lazy_store.py`.

### BUG-61. Lazy arraydir stores serve material channels nobody requested

- **Symptom.** `var` contains `mat_*` channels the model's config does not declare,
  and code that validates, scales or counts channels from `var` miscounts.
- **Cause.** `ArrayDir*Store.read_case` builds all material channels whenever the
  member carries a material payload and the caller requested none. This is the
  path that supplies `mat_*` to material-conditioned configs, so it stays.
- **Workaround.** Filter by the config's declared keys first
  (`_declared_material_channels`).
- **Code.** `models/pipeline/lazy_runtime.py`,
  `models/pipeline/preprocessing_datasets.py`.

## Legacy VTK datasource (only with `PINO_ALLOW_VTK=1`)

### BUG-01. Per-subdirectory layouts lose the case parameters

- **Symptom.** A per-subdirectory batch (one directory per case, timesteps inside)
  aborts the load with a missing-parameter error.
- **Cause.** `read_file` parses `laser_power`/`V_scan` from the timestep filename and
  ignores the parent directory that encodes them; missing tokens are fatal (A5).
- **Status.** Open on the legacy path; consolidated NPZ is the supported input (A36).
- **Code.** `custom_utils/openFoam_datasource.py` (`_group_files_by_subdir`,
  `read_file`). Test: `test_vtk_deprecation_gate.py`.

### BUG-02. Decimal time suffixes collapse a trajectory to single frames

- **Symptom.** A `B`-case, `T`-timestep trajectory loads as `B*T` single frames with
  no error.
- **Cause.** `_base_without_time` strips only integer time tokens while `_parse_time`
  also accepts decimals, so every timestep becomes its own base (A3, A7).
- **Status.** Open on the legacy path.
- **Code.** `custom_utils/openFoam_datasource.py`. Test: `test_vtk_deprecation_gate.py`.

### BUG-12. Grid mapping from the first timestep is reused for all timesteps

- **Symptom.** Fields of a re-meshed or reordered later timestep land on the wrong
  grid cells.
- **Cause.** `uniform_info` is detected from the first timestep and reused for every
  timestep of a base. FLOW-3D exports keep the mesh fixed in time, so they are
  unaffected.
- **Code.** `custom_utils/openFoam_datasource.py` (`parse_directory`).

## GPU spooler

### BUG-42a. A GPU reservation is a booking, not a limit

- **Symptom.** A run admitted beside another dies with a CUDA OOM that looks like its
  own fault.
- **Cause.** `gpu_spool` admits by declared VRAM (`--vram-mb`), and nothing caps what
  a run then allocates.
- **Workaround.** Size `--vram-mb` from a measured peak and round up; leave it off
  for a run whose peak is unknown, which books the whole card (A44).
- **Code.** `gpu_spool.py` (`reserve`).

### BUG-43b. A host-RAM booking does not cap the run by default

- **Symptom.** A run that grows past its `--ram-mb` takes the memory, and the kernel's
  OOM killer may stop a different, larger job.
- **Status.** Every run books host RAM (undeclared runs book
  `PINO_SPOOL_DEFAULT_RAM_MB`), so it is visible to the next admission.
  `PINO_SPOOL_ENFORCE_RAM=1` adds a systemd scope with `MemoryMax` at 1.5x the
  declaration and no swap; it is off by default because the cgroup also charges the
  run's page cache, which can kill a healthy long training.
- **Code.** `gpu_spool.py`. Tests: `test_gpu_spool_ram_enforcement.py`,
  `test_gpu_spool.py`.

### BUG-58. `gpu_seat_priority` fails a job after a bounded wait

- **Symptom.** A job submitted with priority fails after about an hour without
  having started, while jobs without it wait and run.
- **Cause.** A priority seat releases its hold after 900 s and gives up at
  `DEFAULT_PRIORITY_MAX_WAIT_S` (3600 s); an ordinary seat waits indefinitely.
- **Workaround.** Use priority for short jobs only, never for a training.
- **Code.** `gpu_spool.py`.

### BUG-49. Admissions on different cards can jointly overbook host RAM

- **Cause.** `reserve()` takes a per-card admit lock, while the host-RAM check sums
  bookings across all cards; two admissions on different cards in the same window
  can each pass the check, overbooking by up to one declaration. Not observed in
  practice.
- **Fix shape.** A machine-wide lock for the RAM check, or a re-check after the lease
  is written.
- **Code.** `gpu_spool.py` (`_admit_lock`, `reserve`, `booked_ram_mb`).

### BUG-50. A lease being written is charged as 0 MiB

- **Cause.** A lease file is created and then written; a reader that parses it in
  between falls back to `{"vram_mb": 0}` and charges nothing. Same-card admissions
  hold the admit lock across both steps, so this is reachable only across cards, a
  sub-case of BUG-49.
- **Code.** `gpu_spool.py` (`_hold`, `_entries`).
