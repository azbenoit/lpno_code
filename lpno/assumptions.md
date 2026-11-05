# assumptions.md

Non-trivial assumptions and deliberate choices in this repo — the ones whose
effect on a training result or a run is not obvious from a single line of code.
Trivial input validation and self-evident defaults are out of scope; the source
and `README.md` cover those.

Use this file to sanity-check results: if a training run, an eval metric, or an
ingested dataset looks wrong, a wrong assumption here is a likely cause. The
practical consequences, and the known limitations that follow from these choices,
are summarized in `docs/training_and_evaluation.md`.

IDs are stable, and every entry cites the source location it was read from.

## Data ingestion and the VTK datasource

**A1. `flip_axes` remaps OpenFOAM axes `(X,Y,Z) -> (X,Z,-Y)` for coordinates and vectors.**
Coordinates get `coords[:,1] *= -1` then a column swap `[0,2,1]`; vector fields
get the same swap plus a Z-negation. `parse_directory` is called with
`flip_axes = (software == 'openFoam')`, so FLOW-3D/Abaqus data is never flipped.
Effect: stored axis order and vector-component signs depend on the backend;
applying the flip to non-OpenFOAM data transposes geometry silently. Source:
`custom_utils/openFoam_datasource.py:554-556,600-609`;
`models/pipeline/preprocessing_datasets.py:307`.

**A2. The `var_*.npz` cache is keyed by `super_res_factor`, time-mode, and a geometry+flip hash token.**
`parse_directory` names the cache
`var_{super_res_factor}[_{time_suffix}]_{token}.npz`, where the token is the
first 8 hex chars of a sha1 over the x/y/z ranges, resolution, and `flip_axes`;
a hit is then validated by field-name presence.
Effect: a changed grid, range, or flip recomputes instead of reusing a stale
cache. The key still ignores non-geometry knobs such as which resample
device produced it. The cache read/write is **best-effort**: writes
(`np.savez_compressed`) and stale-cache deletes (`_safe_unlink`) are wrapped so a
read-only data dir (e.g. a shared dataset in CI) warns once and recomputes rather
than aborting the parse. Source: `custom_utils/openFoam_datasource.py` (cache-name
block, `_safe_unlink`, and the cache-write `try/except` in `parse_directory`).

**A3. Integer filename time tokens are scaled by `1e-6 s`; decimal tokens are supported only by `_parse_time`.**
`_parse_time` returns `int(token)*1e-6` for integer suffixes (the class docstring
says `1e-7`, a 10x contradiction) and also accepts decimal/scientific tokens, but
`_base_without_time` strips only trailing integers. Effect: decimal-time
filenames break temporal grouping (each frame becomes its own base; legacy VTK
path only, A36); the integer scale sets the
physical `t_hat`. Source: `custom_utils/openFoam_datasource.py:63,768,785,792`.

**A4. Fields are KNN inverse-distance blended by default; categorical fields must be opted out via `categorical_fields`.**
`apply_knn_scalar_torch` weights `k=8` neighbours by `1/(dist+eps)`, so a 0/1
phase/solid-fraction/fluid indicator becomes fractional at grid nodes.
`parse_directory(categorical_fields={...})` resamples the named
fields with pure nearest neighbour (`k=1`) instead; the
default (`None`) keeps blending everything, and the choice is part of the
cache token. Effect: categorical FLOW-3D fields are smoothed as if continuous
unless the caller names them. Source: `custom_utils/openFoam_datasource.py`
(`apply_knn_scalar_torch`, `knn_kwargs` in `parse_directory`).

**A5. Laser power and scan velocity are parsed from the VTK filename stem, and a missing token is fatal.**
`read_file` calls `_parse_params(Path(filename).stem)`; both the
`laser_power` and `V_scan` tokens are required (a zero would give
`H_norm = P/sqrt(|V_scan|) = 0/0 = NaN`), and preprocessing separately refuses a non-positive `P` or a zero
`V_scan` from any datasource at the `H_norm` choke point (a negative `V_scan` is
a reverse-direction scan and is allowed). In per-subdirectory batch mode only the
timestep filename, not the parameter-encoding parent directory, is parsed.
Effect: an unparseable case aborts the load instead of poisoning it. Source:
`custom_utils/openFoam_datasource.py:_require_scan_parameters`;
`models/pipeline/preprocessing_datasets.py:_require_physical_laser_parameters`.

**A6. Off-metal cells (`alpha < 0.5`) are filled with a boiling temperature and zeros.**
Outside metal, `T` is set to `3123 K` (later normalized by `ref_scales['T']`),
other channels to `0`, and `u/v/w` are additionally gated by liquid fraction
`fl`. Effect: these fabricated voxels enter both the targets and the error
metrics. The fill is one `T_BOIL_K` constant shared by preprocessing, the eval
paths and `MaskNode`, divided by the configured `ref_scales['T']` and
overridable per material with `material_params.T_boil`. Source:
`models/pipeline/preprocessing_datasets.py:193-203,241-251`;
`models/pipeline/postprocessing.py:246-267`; `equations/FNO_postprocess.py:55-68`.

**A7. Time-axis mode requires equal timestep counts across all bases and infers the field set from the first file.**
A single unequal group raises; the available field list is taken from
base0/timestep0 and every other file must contain exactly those fields (a
missing locked field raises `KeyError`). Fields present only
in later files are dropped, and `parse_directory` emits one consolidated
`UserWarning` naming them and an example file. Effect: a
source with non-uniform per-frame field availability, or with mis-grouped
timesteps (A3), is rejected or loudly degraded. Source:
`custom_utils/openFoam_datasource.py` (`_track_dropped` in `parse_directory`).

**A26. Consolidated-NPZ geometry is derived from the file; VTK keeps its config box.**
A consolidated case NPZ already stores a uniform grid, so `is_consolidated_npz`
routes it to `npz_datasource.load_consolidated_npz` (no crop/resample) and
`derive_geometry_from_npz` reads `x`/`y`/`z` min/max + lengths back into
`grid_params` before the model, plotters, and FNO mode caps are sized. VTK dirs
are *not* auto-derived: their `custom.geometry` is a deliberate crop (e.g. drops
the gas layer `z>0`), not the raw data extent, so auto-deriving would change
loaded content. Opt a non-NPZ run into derivation with `custom.geometry: auto`.
Effect: arbitrary rectangular NPZ domains load unchanged, but the derived grid
comes from the *training* `data_dir` — a val/test dir with a different grid must
match or be pointed at compatible data. Source:
`custom_utils/npz_datasource.py`, `models/pipeline/config_loading.py` (geometry
derivation before `grid_params` unpack), `preprocessing_datasets.py:load_and_preprocess_data`.

**A27. The moving reference frame shifts one x-cell per timestep regardless of export cadence.**
`crop_shift_moving_frame` shifts each kept timestep by `super_res_factor` cells
along x, which only tracks the laser when the export cadence satisfies
`V_scan * dt == dx`. For the VTK training set this holds by construction; a
consolidated NPZ exported on an independent even-time grid may not, so the
moving-frame pool drifts. This is a data-generation/cadence concern, not a load error — the grid
and fields load correctly; use `moving_ref_frame=false` or match the export
cadence to `dx/V_scan` when it matters. Arbitrary paths use lab-frame path
channels instead (A28). Source: `custom_utils/openFoam_datasource.py:15-38`,
`custom_utils/npz_datasource.py:_apply_moving_frame`.

## Normalization and conditioning

**A8. The global temperature scale is `ref_scales['T']=3000 K` for all temperatures.**
Network `T` and the solidus/liquidus (`fl_Node`) are non-dimensionalized by
`T_ref=3000`. Effect: a temperature scaled by a different reference (e.g. the
melt range `dT`) is silently mis-weighted. Source: `equations/FNO_postprocess.py`
(`fl_Node`); `conf/hydra/config_NS_energy_PINO_flow3D.yaml`.

**A9. `H_norm = P/sqrt(|V_scan|)`, normalized by one denom shared across splits: the train-fitted max (standard/eval) or a fixed `ref_scales['H']` (kfold).**
When `h_norm_mode='max'` (the standard/eval default) `_preprocess_var` fits the
denom `max(var['H_norm'])` once on the authoritative train pass
(`set_runtime_params=True`), stashes it on `run_cfg.laser_params['h_norm_denom']`,
and the later val/eval pass reuses it instead of recomputing a per-split max — so
the same physical `(P,V)` maps to the same model input in train/val/eval. The
lazy path mirrors this (val reuses `train_h_norm`). `h_norm_mode='ref_scale'`
instead divides by the constant `ref_scales['H']` (default 5.6; the kfold path,
and the value the serving adapters standardize on, `H_ref=203`). Source:
`models/pipeline/preprocessing_datasets.py:231` (fit/stash/reuse) and `:638`
(lazy); `models/ns_energy_pipeline.py`. Regression:
`tests/test_preprocessing_velocity_scale.py::test_val_pass_reuses_train_h_norm_denom`.
A standalone evaluation takes the denom that training recorded: training writes
`h_norm_denom.json` beside the weights, and the eval resolves, in order, an explicit
`custom.h_norm_denom`, that record, a refit on the training split named by the archived
config, else the scored split with a banner (`models/pipeline/h_norm_record.py`;
aiida-pino backfills the record for older models). The eval stores what it used as the
`[h_norm]` metrics block. Regression: `tests/test_h_norm_eval_uses_training_denominator.py`.

**A10. The FNO input channel order is the order of `custom.FNO_input_keys`, with fixed append positions.**
The layout is `[FNO_input_keys in listed order, minus 'y' if use_2d]`, then
`t_hat` (if `time_as_batch`), then `<out>_prev` state keys (if autoregressive),
then the mask channel. Effect: reordering the config key list silently reorders
the network's input channels. Source: `models/pipeline/config_loading.py:186-228,631-682`.

**A12. Relative error is normalized by the mean of the ground-truth field (+eps), and is not reported for `fl`/`alpha`.**
`abs_err`/`rmse` are divided by the (aggregate or per-sample) mean of the truth
field. That is correct for a strictly-positive dimensional field like
temperature, but `fl` and `alpha` are already dimensionless fractions in [0, 1]:
their absolute error *is* the relative one, and a low-melt sample's own mean
(~1e-3, or 0) would inflate the ratio without bound. Those keys therefore carry no
`rel_err_mean`/`rel_err_std`/`rel_rmse` (nor a substrate/surface relative), and
their per-sample CSV cell is blank; IoU and the absolute metrics carry them.
Effect: a relative column that is present is meaningful. Source:
`models/pipeline/postprocessing.py:_UNITLESS_KEYS, _has_relative_metrics`.

**A13. `fl` (liquid fraction) is always derived from `T` via a tanh clamp, overwriting any model-provided `fl`.**
Both predicted and truth `fl` come from `smooth_liquid_fraction_numpy(T,...)`
with `beta=2.5`, computed from the pre-mask `T`; `iou_per_batch` returns `1.0`
for empty-vs-empty samples. Effect: a hard 0/1 phase indicator is never used;
`fl` metrics never see the model's own `fl` output. Source:
`models/pipeline/postprocessing.py:235-240,451-455`; `custom_utils/utils.py:147-163`.

## Physics and PDE residuals

**A18. `MaskNode` is a soft tanh clamp (`k=20`) toward fixed fills, not a hard mask.**
Where `alpha<0.5`, `T` is driven toward the caller-supplied `t_fill`
(`T_boil/ref_scales['T']`) and other fields toward `0` with sharpness `k=20`.
Effect: masked cells are blended, not zeroed. Constructing a `MaskNode` that
masks `T` without a `t_fill` raises, so the fill cannot silently disagree with
the training target's scale. Source:
`equations/FNO_postprocess.py:55-68`; `models/pipeline/model_domain.py:135-155`.

## Config, devices, and run modes

**A19. CUDA visibility is decided pre-torch from one config file's `custom:` block, and `cuda_device` is restricted to `{0,1}`.**
`cuda_bootstrap` reads only the named `--config-name` YAML's top-level
`custom.cuda_visible_devices`/`cuda_device` before torch is imported; keys
arriving through Hydra `defaults:` includes are not seen. `configure_device`
rejects any device other than `SUPPORTED_CUDA_DEVICES` with `SystemExit`, and
honours an already-set `CUDA_VISIBLE_DEVICES` (a Slurm allocation, or gpu_spool
pinning a leased seat) instead of overwriting it with `str(cuda_device)`; an
explicit `custom.cuda_visible_devices` still wins. Effect: config-group device
settings are not honoured, but a caller's pin is. Source:
`models/cuda_bootstrap.py:39-118`; `models/pipeline/config_loading.py:configure_device`.

**A20. All three entrypoints default to `config_NS_energy_PINO`, and `include_time`/`alpha_masking` default on only for `software=='flow3D'`.**
Omitting `--config-name` silently uses `config_NS_energy_PINO` (different batch
size, modes, lr) rather than the flow3D profile; a mistyped `software` value
turns off the FLOW-3D temporal/masking defaults. Effect: the effective run
configuration depends on exact string values. Source:
`models/NS_Energy_laser_PINO.py:32`; `models/pipeline/config_loading.py:231-240,576,736`.

**A21. In eval, the trained run's saved Hydra config overrides the eval config file, the data path defaults to the TEST subdir, and `val_dir` is forced equal to `data_dir`.**
`OmegaConf.merge(original_cfg, model_cfg)` makes the saved run's values win
(only explicit CLI overrides are re-applied); `run_mode=eval` resolves the test
subdir and sets `val_directory_path = data_directory_path`. Effect: editing the
eval YAML has no effect on architecture, and eval always scores the test subdir.
Source: `models/pipeline/config_loading.py:517-542,709-723`.

**A22. Standard-mode splitting overlaps train and validation indices when no separate val directory is given.**
`get_splits('standard')` returns `train_idx = arange(n)` and
`val_idx = arange(val_size)`, so validation is measured on a subset of training
data unless a distinct `val_directory` is supplied. Effect: optimistic
validation curves on single-directory runs. Source:
`models/pipeline/preprocessing_datasets.py:671-679`; `models/ns_energy_pipeline.py:86-91`.

**A23. Default dataset paths append a fixed deep subdirectory.**
Left at the `DEFAULT` sentinel, `data_dir`/`val_data_dir`/`param_dir` resolve to
`PINO_SIMULATIONS_DATA_ROOT` (or repo `data/`) + a fixed
`flow3D/original_straight_path_data/{train,val}` suffix; online wandb targets
the entity in `WANDB_ENTITY` (unset = the account default). Effect: defaults assume
one specific data layout; the wandb organization is the caller's. Source:
`models/pipeline/config_loading.py:34-37,707-730,953`.

## Arbitrary-weld-path conditioning

**A28. The laser trajectory is passed as explicit lab-frame input channels derived from the manifest, not via the moving reference frame.**
For arbitrary paths the `*_randompath` configs set `moving_ref_frame=false` +
`time_as_batch=true`, and `npz_datasource` builds `beam_x/beam_y/beam_dist/path_dose`
from each case's sidecar `<case>.manifest.json` (`laser_path_xy_m`, scan speed,
`laser_off_time_s`, `beam_radius_m`). Channels are built only for the names present in
`FNO_input_keys`; a requested path channel with no manifest raises rather than
defaulting to zeros. Effect: the model sees the true beam location/history in the lab
frame; a wrong/missing manifest changes the conditioning, not just a scalar. Source:
`custom_utils/npz_datasource.py` (`_build_path_channels`, `load_consolidated_npz`);
`models/pipeline/preprocessing_datasets.py:load_and_preprocess_data`.

**A29. `path_dose` is normalized per case to a fraction of total dwell, so its scale is beam-position history, not absolute energy.**
`path_dose = cumsum(footprint * dt * [t<=laser_off]) / sum(dt * [t<=laser_off])`, so it
is in `[0,1]` and plateaus after the laser turns off. Where power, speed or beam radius
vary per case, this normalization hides the difference. `beam_x/beam_y` are scaled by
`ref_scales.L` like the coordinates; `beam_dist` peaks at 1. Path channels are excluded
from the alpha melt-masking (they are geometry inputs, not melt-gated fields). Source:
`custom_utils/npz_datasource.py:_build_path_channels`;
`models/pipeline/preprocessing_datasets.py:_preprocess_var`.

**A30. DeepONet routes `path_dose` to the trunk, not the branch.**
The branch must be spatially constant conditioning, so beam-centre scalars go to the
branch while the spatially-varying `path_dose` field is added to the trunk coordinate
set via `custom.deeponet_trunk_extra_keys`. Effect: putting a spatial field in the
branch would collapse the operator factorization; the trunk keeps it location-dependent
as intended. Source: `models/pipeline/config_loading.py` (coordinate_key_order /
`deeponet_trunk_extra_keys`).

**A43. Past the declared path end, a laser with `turn_off_at_end: false` keeps moving along its last leg at that leg's speed.**
FLOW-3D holds the last laser row until `finish_time`, so the beam stays on and, for a
scan, keeps moving. `_beam_path_over_time` therefore extrapolates the centre beyond the
last knot (constant-speed: at `scan_speed`, reproducing `derived.laser_overrun_distance_m`
at finish_time; variable-speed: at the last leg's speed, so a terminal dwell stays put) and
keeps speed/direction live. The condition is `laser_on_past_declared_time` OR
`laser_turn_off_at_end is False`. For older manifests without `laser_on_past_declared_time`
the latter also OVERRIDES `laser_off_time_s`, which their writer set to `scan_time`
regardless of whether the laser turned off. Effect: on overrun frames the path channels and
both moving-frame windows follow the deposited beam instead of freezing 0.14-0.35 mm behind
it; datasets with `finish_time == scan_time` are unaffected. A manifest with neither flag
keeps the old clamp.
The variable-speed extrapolation assumes the solver holds the LAST velocity-table row; it
is consistent with the deck semantics recorded for the single-row case and has no live
dataset yet (no variable-speed base dataset carries `turn_off_at_end: false` with a real
overrun). Source: `custom_utils/npz_datasource.py::_beam_path_over_time`;
`tests/test_beam_path_overrun.py`.

## Autoregressive rollout mode

**A31. `autoregressive.mode=rollout` trains one-step pairs with next-step context and evaluates by recursive rollout from a ground-truth initial state.**
Training pairs map `(context[t+1], state[t]) -> state[t+1]`; eval seeds with GT at
`t=0` (`eval_use_gt_init=true` is the only implemented option) and feeds each
prediction back as `*_prev`. Context (exogenous) channels always come from the
target frame.
Supported families: `fno`, `rigno` (grid tensors), `deeponet` (per-step pointwise
flatten/restore inside the rollout loop; `*_prev` enters as the local per-point
value, so route it to the trunk via `custom.deeponet_trunk_extra_keys` to keep the
branch factorization meaningful). `masked_patch` remains FNO-only and lazy-store-
only. Effect: rollout errors compound over the horizon; `rollout_errors_per_step.csv`
is the place to see it. The rollout scores frames `1..T-1` (the GT-seeded frame 0
is never predicted), while a direct per-frame eval scores all `T` frames including
the easy `t=0` — compare the two on the same `1..T-1` window. Rollout IoU is
computed per frame (the `(B, T-1, ...)` stack is flattened before stats), matching
the direct eval's convention, and the derived `fl` prediction is alpha-masked like
the direct grid eval. Source: `models/pipeline/preprocessing_datasets.py`
(`_build_autoregressive_pairs`, `build_datasets`), `models/pipeline/postprocessing.py`
(`run_autoregressive_rollout_and_metrics`), `models/pipeline/config_loading.py`.

**A32. `autoregressive.prev_noise_std` perturbs only the training-pair previous-state inputs.**
Gaussian noise (std in normalized units) is added to `*_prev` training inputs only —
validation pairs, targets, and context stay clean. Default `0.0`. Effect: a nonzero
value trades one-step accuracy for rollout stability; it changes the training data
distribution, so runs with different values are not directly comparable. By
default the noise is one frozen realization drawn at dataset construction (every
epoch sees the same corrupted pairs). Source:
`models/pipeline/preprocessing_datasets.py` (`_build_autoregressive_pairs`).

**A36. Direct VTK directory loading is deprecated and refused; consolidated NPZ (or an arraydir runtime / aiida dataset) is the supported input path.**
`load_and_preprocess_data` raises a ValueError when the input dir is neither a
consolidated-NPZ dataset nor an arraydir runtime, unless `PINO_ALLOW_VTK=1` is
set. Chosen because the legacy VTK parser has known silent-corruption modes
(per-subdirectory cases parse as P = V_scan = 0, A5; decimal-time
trajectories collapse to T=1, A3) and all active datasets are NPZ. Effect: a stale
config pointing at a raw VTK dir fails loudly with conversion instructions
instead of training on corrupt data; the opt-in keeps deliberate legacy loads
possible. Source: `models/pipeline/preprocessing_datasets.py`
(`load_and_preprocess_data`); regression `tests/test_vtk_deprecation_gate.py`.

**A35. The training-time grid validator streams metrics; its np/vtk dump and plotter figures contain only the first validation batch.**
`StreamingGridValidator` computes l2_relative_error/mae/rmse from
running scalar sums instead of concatenating every batch on host RAM (the
upstream `GridValidator.save_results` behaviour). Deliberate trade: validation metrics are exact over the full
split, but the `save_filetypes` np/vtk artifact and tensorboard slice figures
sample only the first batch (shipped configs set `save_filetypes: ""` and
`validator_plots: false`, so nothing user-visible changes by default). The
final eval in `postprocessing.py` still exports full predictions. Source:
`models/pipeline/model_domain.py` (`StreamingGridValidator`); regression
`tests/test_streaming_grid_validator.py`.

**A34. `arch.rigno.grad_checkpointing` recompute is exact because the checkpointed processor region is deterministic.**
Each `proc_steps` interaction step is wrapped in `torch.utils.checkpoint` (only when
`grad_checkpointing=true`, training, and grad is enabled), so its activations are
recomputed in backward instead of stored. This is numerically identical to the plain
path — asserted on both outputs and gradients in
`tests/test_rigno_grad_checkpoint.py` — because the checkpointed body has no
randomness: regional edge masking (`p_edge_masking`) is drawn once *before* the loop
and the steps carry no dropout, so forward and recompute see the same edges. Effect:
enabling it changes only peak memory, not the trained model, and lets multi-step AR
unrolling (`unroll_steps>=3`) fit in 24 GB (k=3 at the tuned width needs ~27 GB fp32
without it, ~18 GB with). It only checkpoints the base `RIGNOArch` processor, not the
`rigno_refined` FNO head. Source: `models/rigno_arch.py` (`_encode_process_decode`),
`models/pipeline/model_domain.py`.

**A33. A source grid whose resolution differs from `grid_params` is a hard error, not a silent resample.**
`_detect_uniform_grid` distinguishes a genuinely scattered point cloud (KNN
interpolation is the intended path) from cells that already form a *complete*
rectilinear grid — any axis-aligned ROI crop of a grid stays rectilinear, so the
product of the unique-axis counts equals the ROI cell count. When the ROI is a
complete grid but its size is not `grid_params['resolution']`, the load raises
`GridResolutionMismatchError` instead of KNN-resampling real grid data onto the
wrong size (e.g. one case in a sweep exported at a different resolution, which
would otherwise poison training silently). Effect: such a case fails loudly at
load time. Override with `custom.allow_grid_interpolation=true` (threaded through
`RunConfig.allow_grid_interpolation` to `parse_directory`) to resample on purpose;
genuinely scattered data (OpenFOAM) is never affected. Source:
`custom_utils/openFoam_datasource.py` (`_detect_uniform_grid`, `parse_directory`),
`models/pipeline/config_loading.py`, `preprocessing_datasets.py`.

**A44. GPU admission is a ledger of declared memory, never a reading of live GPU memory.**
`gpu_spool` decides whether a run may start by summing the reservations live runs
have declared and comparing that with the card's budget (total minus
`PINO_SPOOL_VRAM_HEADROOM_MB`, default 2048 MiB). It never treats `memory.used`
or `memory.free` as evidence of *room*: PyTorch's caching allocator releases and
re-takes blocks throughout training, so a run that looks small between steps is
not small, and admitting on that reading is how two runs end up fighting for the
same GB. Live free memory is read for exactly one purpose — refusing a *declared*
reservation that would not physically fit right now, which is the only way memory
held outside the ledger (a notebook, another user's process) is noticed;
`PINO_SPOOL_CHECK_FREE=0` disables it. An *exclusive* run (no `--vram-mb`, the
default) is exempt from that floor: it states "nobody else joins me" rather than a
size, so holding it to a byte count would let any stray 2 GB process block every
whole-card run for ever. Effect: packing is opt-in and conservative — a run that
declares nothing behaves exactly as it did when the spool was one-at-a-time.
A run also books **host RAM** (`--ram-mb`), checked
against `MemAvailable` minus `PINO_SPOOL_RAM_HEADROOM_MB` (default 8192 MiB)
plus what other runs have booked, summed machine-wide across both cards. Unlike
VRAM there is no "exclusive" for RAM to fall back on, so an undeclared run books
`PINO_SPOOL_DEFAULT_RAM_MB` (32768 MiB, above the largest training measured)
rather than nothing — booking nothing made a running job invisible to
the next admission, which is the same mistake as trusting live GPU memory, one
step removed. `--ram-mb 0` opts a genuinely small run out. Live
`MemAvailable` *is* trusted here, unlike GPU free memory: the kernel already
discounts reclaimable cache and every process on the machine, so it is a fair
reading of what a new run can take. With `PINO_SPOOL_ENFORCE_RAM=1` (off by
default), a run that *declares* its RAM is additionally capped: it runs in a systemd user scope with `MemoryMax` at 1.5x the declaration
and `MemorySwapMax=0`, so a runaway dies as itself instead of leaving the kernel
to pick the largest RSS (usually the honest big job). The swap cap is load-bearing
-- without it `memory.max` reclaims into swap and the machine thrashes before
anything is killed -- and there
is deliberately no `MemoryHigh`, which throttles a scope into a crawl instead of
killing it and wedges the seat. Undeclared runs are *not* capped: the 32 GB house
default is a guess, and a guessed ceiling kills a long training at hour three.
The cap is off by default because the cgroup also charges a run's page cache,
which can kill a healthy long training; a host with no systemd user manager runs
uncapped rather than not at all. Host RAM is booked because a run can exhaust it
with VRAM to spare. The ledger is shared
across users (`/var/tmp/pino_gpu_leases`) so every user and job on one machine books
against one budget; it holds reservations only, never
commands, so a world-writable directory cannot get a job executed as someone
else. A reservation whose owner died stops counting the moment the kernel drops
its `flock`. Source: `gpu_spool.py` (`reserve`, `free_mb`, `vram_budget_mb`,
`_entries`), `tests/test_gpu_spool.py`.

**A45. RIGNO graph-variant rotation is per-training-forward, not per-epoch.**
The JAX reference rebuilds the regional mesh once per epoch with a fresh PRNG
key; `arch.rigno.num_graph_variants=N` instead pre-builds N graphs (seeds
`graph_seed..+N-1`) and draws one uniformly at every training forward, because
the Arch has no epoch signal and pre-built variants cost nothing per step.
Effect: at least as much graph stochasticity as upstream, and eval is pinned to
variant 0 so metrics stay deterministic and comparable across checkpoints. The
`_forced_variant` hook exists for tests/inference-time variant sweeps. Source:
`models/rigno_arch.py` (`_build_graph`, `_encode_process_decode`);
`tests/test_rigno_model.py`.

**A37. Temporal bundling supervises all B frames from one snapshot; validation and the in-training validator stay one-step.**
`custom.autoregressive.bundle_steps=B` widens the head to predict frames
t+1..t+B from the step-0 context (the upstream stepper's temporal bundling),
so the training loss covers B frames while the val dataset keeps the one-step
layout — val metrics mean the same thing across bundled and unbundled runs,
and the bundled frames are only scored through the rollout eval. The FiLM
adaptation (`arch.rigno.film_keys`) likewise drops the reference
ConditionedNorm's multiply-by-c factor (vanish-at-tau-0 semantics) because the
conditioned scalars (P, V_scan, H_norm) have no zero point with meaning.
Source: `models/pipeline/preprocessing_datasets.py` (`_ar_window_sample`),
`models/pipeline/postprocessing.py` (bundle rollout), `models/rigno_arch.py`
(`_FiLM`); `tests/test_temporal_bundling.py`, `tests/test_rigno_model.py`.

**A38. A prediction/truth shape mismatch is resolved by cropping both to their overlap, never by padding.**
Decoder rounding (U-Net pooling, the pointwise grid restore) can leave a
prediction a voxel or two off the truth grid in either direction.
`_align_preds_to_truth_shapes` center-crops both sides to the elementwise
minimum, so the metrics are computed over a slightly smaller volume than the
truth grid rather than over invented voxels; `_center_crop_numpy` raises if a
caller asks it to grow an array. The z coordinate field is cropped the same way
inside `region_error_stats` so substrate/surface regions still line up. Source:
`models/pipeline/postprocessing.py`; `tests/test_align_no_padding.py`.

**A39. Melt-pool and keyhole geometry metrics reuse the dataset-side mask definitions, and depth is a per-sample scalar, not a fitted shape.**
`melt_geometry_stats` thresholds `fl > 0.5` for the melt pool and `alpha < 0.5`
for the recoil cavity inside the substrate (`z <= 0`), drops the x/y boundary
columns, and measures depth as cells from the surface slice down to the deepest
hit — the same rules `aiida-pino/console/viz.py` applies to the ground-truth
NPZ, so a model number can be read against the dataset's own. Two deliberate
simplifications against that reference: the cavity is not split into connected
components (the aiida stats fit an ellipse to the component around the deepest
point, which matters for the curved random paths), and length/width are axis-
aligned bounding-box spans rather than ellipse radii. Both make a wandering
track read wider than its instantaneous pool; the numbers are a like-for-like
prediction-vs-truth comparison, not a substitute for the per-simulation stats.
Micrometres assume `ref_scales["L"]` is the length scale that normalized `z`;
with no `L` the values are grid units. Source:
`models/pipeline/postprocessing.py`; `tests/test_melt_geometry_metrics.py`.

**A40. GAOT's phys<->latent graphs are built once from the true grid, and both directions bound the *physical* side.**
`models/gaot_arch.py` builds the encoder graph **source-centric** (every physical
node pushes to its `k_encoder` nearest latent tokens) and the decoder graph
**query-centric** (every physical node pulls from its `k_decoder` nearest latent
tokens). Effect: 100% of the input field is read and 100% of the output is
produced, independent of how coarse the latent lattice is. The opposite encoder
convention bounds the latent side instead and silently drops input. The graph depends
only on the grid, so it is built lazily from the *true* (moving-frame-cropped)
shape on the first forward and cached under `arch.gaot.cache_dir`, exactly as
RIGNO caches its graph. Source:
`models/gaot_arch.py` (`_knn_edges_source_centric`, `_knn_edges`, `_build_graph`);
`tests/test_gaot_model.py::test_encoder_reads_every_physical_node`.

**A41. GAOT normalises coordinates per axis, not globally, and its latent lattice is a config choice sized against the cropped grid.**
`_grid_coords` maps each axis independently onto [-1, 1] (matching
`rigno_arch.py:_build_graph`). Upstream GAOT-3D's `utils/scale.py:rescale`
instead takes one global min/max across all axes, which on the strongly
anisotropic flow3D box (900 x 400 x 300 um) would compress y and z into a
fraction of the coordinate range and distort every distance the kernel and the
geometry embedding see. Consequence: `arch.gaot.latent_tokens` should track the
*index* aspect of the grid the model actually receives (61 x 40 x 40 after the
moving-frame crop), not `custom.geometry.resolution` (90 x 40 x 30) and not the
physical box aspect. Source: `models/gaot_arch.py`;
`tests/test_gaot_model.py::test_grid_coords_are_per_dimension_normalised_and_row_major`.

**A42. The GAOT arm runs fp32, and the geometry embedding is off.**
`config_flow3D_GAOT` inherits `config_flow3D_RIGNO`, which declares no `amp` block, so the
run is fp32. This is deliberate rather than inherited-by-accident: the
attentional graph operator normalises with a segment softmax over neighbourhoods
and the statistical geometry embedding calls `torch.linalg.eigvalsh`, both fp16
overflow candidates, and the point of the config is that only the architecture
differs from the RIGNO baseline. Both `use_geoembed_encoder` and
`use_geoembed_decoder` are false: on the uniform Cartesian grid of these datasets the
statistical descriptors (neighbour count, distance mean and variance, neighbourhood
PCA) are near-degenerate, so they cost an eigendecomposition per edge for no gain.
Setting them true reproduces upstream's configuration. Source:
`conf/hydra/config_flow3D_GAOT.yaml`; `models/gaot_arch.py`.
