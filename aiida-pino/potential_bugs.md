# potential_bugs.md: known issues

Ways aiida-pino can mislead or fail, written for anyone debugging a run or
reading an existing database or archive. Each entry gives the symptom, the
cause, the current status or workaround, and where in the code to look.
`assumptions.md` records the intended design choices; this file records the
consequences a user can hit. Several entries concern records written by older
versions of the code, which an imported archive still contains.

IDs are stable. Code and tests also cite `BUG-01` to `BUG-23`: an early audit
batch of ingest and datasource defects (zero conditioning channels, missing
laser metadata, axis flips, stale geometry caches), all fixed, each with a
regression test that names its ID.

When an entry is fixed and has a regression test, delete it; when only part is
fixed, trim it to what still applies.

## Common traps

Each of these can be met when reading an imported archive or nodes written by
older versions of the code. IDs PB-01 to PB-05 and PB-10 to PB-12 are kept here.

**PB-01. The recorded crop and stride can differ from the grid that was served.**
*Symptom:* metadata says one domain, the run served another (30 or 40 z-layers,
strided or full resolution). *Detect:* read `served_grid` on an evaluation node,
or the calcjob's `[grid] serving` log line; older evaluations only have the log.
For non-derived collections, the `effective_*` extras from
`utils/crop_backfill.py` hold the measured extent. *Avoid:* never read crop or
stride from `derivation_profile` or the evaluation env, and check that
`substrate_fraction` matches on both sides of a comparison. The default
`crop_z_top: 10` is recorded on collections that apply no crop; `derived`
collections can under-report one.

**PB-02. An evaluation inherits the model's training `PINO_LAZY_STRIDE`.**
*Symptom:* a model trained strided is scored on the strided grid, with a
plausible number. *Detect:* `training_stride`, `eval_resolution_matches_training`
and the warning on the evaluation; within one dataset, the counted `beambox_n`.
*Avoid:* state the evaluation stride explicitly in the evaluation profile.

**PB-03. Evaluations score the final checkpoint.**
*Symptom:* an evaluation is far worse than the model's own `best_checkpoint.json`,
with exit 0. *Detect:* compare the two values. *Avoid:* score
`u_network.best.pth` with `hydra_overrides: ["++custom.use_best_checkpoint=true"]`
(the `++` is required), and treat the checkpoint choice as a variable of any
comparison.

**PB-04. Steady collections derived before the moving-frame fix can be smeared.**
*Symptom:* the averaging window does not follow the beam (fixed in `lpno`; stale
variants are still reused because `derivation_hash` carries no code version).
*Detect:* compute the beam's travel in cells per frame over the window from the
case manifest and time axis; values well below 1 mark a smeared variant.
*Avoid:* supersede it and derive again.

**PB-05. Near-duplicate collections can both be live.**
*Symptom:* a query bounded to one dataset node misses evaluations filed against
its twin. *Cause:* `derived_fields: None` against `[]`, or schema keys added
after composition, change `content_signature` and `derivation_hash`, so reuse
and the duplicate collapse miss. *Avoid:* resolve a dataset by its member-uuid
set and search every non-superseded collection with that set.

**PB-10. Temperature metrics live in the `kelvin` block.**
*Symptom:* a hand-converted kelvin value disagrees with the recorded one.
*Cause:* `metrics` holds `fl.*` (fluid fraction) beside `alpha.*`, and
multiplying an `fl.*` value by `temperature_scale_k` gives a plausible wrong
number. *Avoid:* read `metric_units.kelvin`; `recorded_kelvin_factor` takes
`(evaluation_metadata, extras)`.

**PB-12. `exclude_from_training` and `usable_for_training` are not enforced.**
*Symptom:* a case marked excluded still trains when it sits in a `train` split.
*Detect:* list members of a training set that carry `exclude_from_training`.
*Avoid:* exclude with `aiida-pino split PK excluded --reason "..."`, which
training refuses.

## Records that can disagree with what ran

### PB-06. `compatibility_signature` does not describe the derivation

- **Symptom.** A cropped and an uncropped collection over the same simulations
  share a `compatibility_signature`, which reads as evidence that they serve the
  same arrays.
- **Cause.** `collection_compatibility_signature` hashes the members' raw
  fingerprint (source type, fields, grid params, shapes), all pre-derivation.
- **Status.** Nothing gates on the field today. Treat it as a member
  compatibility key; read the served shape from PB-01's sources.
- **Where.** `data/pino_dataset_collection.py`.

### PB-07. A profile can record a value that never trained

- **Symptom.** A training profile's structured field (for example
  `weight_decay` or `max_steps`) disagrees with what the run used, and a
  progress bar computed from it is wrong.
- **Cause.** `hydra_overrides` are applied after the structured fields, so an
  override of the same key wins while the stored field keeps its value.
- **Status.** Read side fixed: `override_conflicts()` and `effective_profile()`
  report the value Hydra applied. Resolving overrides into the profile before it
  is stored would change submission behaviour and is left open.
- **Where.** `utils/profiles.py`.

### PB-08. Recorded code versions refer to the development history

- **Symptom.** `runtime.environment.lpno_sha` and similar git SHAs in run
  records do not resolve in the public repository.
- **Cause.** They were stamped from the development checkouts, whose history is
  not part of the public release. Recorded values sit in sealed attributes and
  cannot be rewritten.
- **Workaround.** Treat the SHA as an identifier of the development state. The
  package versions in `runtime.environment` and the published code are the
  reference.
- **Where.** `scripts/live_tail.py` (`collect_environment`).

### PB-09. Older derived views can claim base ownership

- **Symptom.** `base_dataset_of` returns a derived view (for example a stride-2
  view) for some simulations, and `create_base_dataset` over them raises
  `SimulationAlreadyAttachedError`, which `Flow3DBatchWorkChain.finalize`
  reports and swallows.
- **Cause.** Derived builders used to deep-copy the base metadata, including
  `composition_kind="base"`. Both builders now set a non-owning kind; views made
  before that keep the old value.
- **Workaround.** Check `composition_kind` and `dataset_role` on the dataset
  `base_dataset_of` returns; a `derived` role with kind `base` is one of these.
- **Where.** `utils/dataset_derivation.py` (`build_derived_collection_metadata`),
  `data/dataset_hierarchy.py`.

## Conditioning inputs

### PB-13. Constant conditioning channels default silently

- **Symptom.** A channel that should vary per case is constant across a
  dataset, and training converges normally on it.
- **Cause.** `lpno` `npz_datasource.build_constant_path_channels` defaults
  `preheat_T0` to 293.15 K, `beam_shape_uniform` to gaussian, `focus_factor` to
  1.0 and `beam_radius` to 5e-5 m when the case manifest lacks the field.
  `focus_factor` is absent from every manifest, so it is always 1.0.
- **Workaround.** Before training on such a channel, check that the manifests
  carry the field and that its values vary.
- **Where.** `lpno` `npz_datasource.py`.

## Datasets and filing

### PB-14. Batch filing leaves new simulations unassigned

- **Symptom.** A training or evaluation that selects `split == train` (or
  `test`) silently ignores the cases a batch just added.
- **Cause.** `Flow3DBatchWorkChain.finalize` files simulations with no `split`
  extra, even when the batch design carries a split column. Simulations from a
  direct `Flow3DGenerateWorkChain` are also left `unassigned`.
- **Workaround.** Apply the designed splits after filing (`aiida-pino split`, or
  `collection assign-splits`), and check the split distribution of every new
  dataset. `unassigned` is correct for a set that is used whole.
- **Where.** `workflows/flow3d_batch.py`, `workflows/dataset_filing.py`.

### PB-15. Extending a base dataset drops the new batch's metadata

- **Symptom.** After `extend_base_dataset`, the new dataset carries the old
  dataset's description and keywords, which can be wrong for the added members,
  and the extending batch's keywords are missing. A `core` keyword on the first
  generation propagates to every extension, so the dataset name resolves to the
  latest extension and protection tooling refuses operations on it.
- **Cause.** `extend_base_dataset` does not apply the batch's
  `dataset_profile`, and `mark_superseded` carries the old description (when the
  new one is blank) and unions the old keywords.
- **Workaround.** After an extension, check and correct the description and
  keywords by hand. The original reference composition stays reachable by pk.
- **Where.** `data/dataset_hierarchy.py` (`extend_base_dataset`),
  `utils/model_registry.py` (`mark_superseded`).

### PB-16. A killed batch leaves partial run directories

- **Symptom.** Counting `runs/<case>/` directories over-counts solved cases.
- **Cause.** The case being solved at kill time leaves a run directory without
  `npz/<case>.npz`.
- **Workaround.** Count `npz/*.npz`. `aiida-pino file-orphans` files the
  completed cases of a killed batch.
- **Where.** `utils/orphan_recovery.py`.

## Submitting training and evaluation

### PB-17. A training whose inputs the dataset cannot serve is admitted

- **Symptom.** A training that requests a trajectory channel
  (`conduction_field`, `beam_dist`, `path_dose`) on a materialized steady
  derivation takes a GPU seat and fails at the first data read with `KeyError`.
- **Cause.** `materialize_derived_dataset` drops path channels and rebuilds only
  the four constant ones; nothing checks the requested key list against what
  the resolved dataset can serve. `aiida-pino derive-fields` does not help here,
  since the staged derived path never calls the builder whose cache it fills.
- **Workaround.** Run `aiida-pino presubmit` and check the input keys against
  the derivation before submitting.
- **Where.** `workflows/train_model.py` (`validate_inputs`),
  `utils/derived_fields.py`.

### PB-19. The serial derived-export writer may ignore `time_frames`

- **Symptom (not confirmed end to end).** A `time_frames` derivation exported
  through the serial path records the subsampling in its profile and writes
  every native frame.
- **Cause.** `derive_simulation_arrays` has no `time_frames` branch; the worker
  path (`apply_derivation`, `derive_time_values`) applies it.
- **Where.** `data/pino_dataset_collection.py`, `utils/runtime_export.py`.

### PB-20. Zero-copy staging still inflates every array read

- **Symptom.** A training on a full-resolution identity profile is CPU-bound in
  a loader worker with the GPU idle, although staging was zero-copy.
- **Cause.** The pack-index path avoids copying the stored `payload.npz`, but
  that payload is written by `savez_compressed`, so every array read still
  inflates DEFLATE members. Only re-writing with plain `np.savez` avoids it.
- **Where.** `data/pino_simulation.py` (`_export_runtime_npz_case`),
  `assumptions.md` A-PACKSLICE.

## Runtime caches

### PB-22. Runtime cache names depend on the order of `runtime_fields`

- **Symptom.** Two submissions asking for the same fields in a different order
  each build (and, remotely, transfer) their own cache.
- **Cause.** `runtime_cache_name` sorts dict keys and leaves list order as given.
  Normalizing it would rename every existing field-selected cache.
- **Where.** `utils/runtime_cache.py`.

## Processes, daemon and GPU seats

### PB-23. A calcjob can stay "running" after its process has exited

- **Symptom.** A calcjob sits at scheduler state `RUNNING` indefinitely with a
  dead pid and complete outputs on disk; `scheduler_lastchecktime` stops
  advancing. A daemon restart does not clear it.
- **Cause.** An SQLAlchemy transaction-state error in a daemon worker (`Task was
  destroyed but it is pending!`) destroys the worker's pending asyncio tasks,
  including the broker task of the process, and a restart has nothing to
  reattach.
- **Workaround.** Stop the daemon, run `verdi process repair --dry-run`, then
  `verdi process repair` (it lists these as active processes without a process
  task), and start the daemon. To check a training's liveness, combine the
  logged step reaching `max_steps` with a pid check.
- **Where.** aiida-core daemon; `console/monitor.py`.

### PB-24. Killing a job does not always stop it

- **Symptom.** A killed training keeps running and keeps its GPU reservation, or
  a training finishes and is marked exit 500 with no model node.
- **Cause.** With `core.direct` on a remote computer, `verdi process kill` looks
  up the job's descendants with local `psutil`, so the remote training is not
  signalled while AiiDA marks the node killed. Sealing a node does not signal its
  process, and the `gpu_spool` lease is an `flock` held by the process, so the
  card stays booked until the process exits. Killing only the `gpu_spool`
  wrapper of a running job lets the training finish while the step that
  registers its result does not run.
- **Workaround.** Register remote computers with the `pino.direct_pgid`
  scheduler, which kills the process group on the remote side. After any kill,
  check `gpu_spool status` for GPU memory held without a lease and `ps` for the
  process. Recover a finished but unregistered training with
  `aiida-pino submit import-model --artifacts-path <dir>`.
- **Where.** `schedulers/direct_pgid.py`, `lpno` `gpu_spool`.

### PB-25. Finished trainings can exist only in scratch directories

- **Symptom.** A training that reached exit 0 has no `PinoModelData` node, and
  its checkpoints sit only in the calcjob's scratch workdir, which can be
  cleaned.
- **Cause.** A failure between the end of training and model registration (an
  import error, a killed wrapper, PB-23) leaves the work unregistered, and no
  listing shows it.
- **Workaround.** Look for calcjobs with a `train_summary.json` in the workdir
  and no model output, copy `training_output/` somewhere durable, and register
  it with `submit import-model`.
- **Where.** `workflows/train_model.py`, `workflows/import_model.py`.

### PB-26. `seat_priority` orders only runs waiting at the same moment

- **Symptom.** Two FLOW-3D batches with different priorities, each keeping one
  case in flight, alternate case by case.
- **Cause.** A run holds a priority ticket only while it waits, and a run that
  takes a free seat registers none, so a solving high-priority case is invisible
  to the next waiter.
- **Workaround.** Pause the lower-priority batch if strict ordering is needed.
  Holding a batch-level ticket across hand-offs would implement reservation.
- **Where.** `scripts/flow3d_run_adapter.py` (`hold_seat`), `utils/profiles.py`.

### PB-27. The FLOW-3D seat lock serializes one machine only

- **Symptom.** Solves on two machines run concurrently against a licence that
  allows one.
- **Cause.** The lock is an `fcntl.flock` on a local file in the spool directory.
- **Where.** `scripts/flow3d_run_adapter.py`, `flow3D_scripts`
  `python/run_rectangular_case.py`, `assumptions.md` A-SEAT.

## Remote computers

### PB-28. A FLOW-3D case solved on a compute machine cannot be ingested there

- **Symptom.** `Flow3DGenerateWorkChain` on a compute machine solves the case
  and ends with exit 411, the ingest calcjob failing with
  `ModuleNotFoundError: No module named 'aiida_pino'`.
- **Cause.** `run_ingest` reuses the solve's `python_code`, and the ingest
  adapter imports `aiida_pino`, which compute machines do not have installed
  (adapters are uploaded per CalcJob, the package is not).
- **Workaround.** Copy the run directory to the submitting machine and import it
  with `aiida-pino submit raw-flow3d --python-code <local code>`.
- **Where.** `workflows/flow3d_generate.py` (`run_ingest`).

### PB-29. Remote staging checks sizes only, and remote caches cannot be repaired

- **Symptom.** A training on a remote computer dies with `BadZipFile: Bad
  CRC-32` on a file `stage-remote` reported as staged.
- **Cause.** The post-copy verification compares relative paths and sizes, so a
  same-size corruption passes. `runtime_repair` needs `aiida_pino` and the
  object store, neither of which exists on a compute machine, so its
  `|| true` call in the GPU-seat wrapper is a permanent skip there.
- **Workaround.** Compare checksums of a staged copy against the source with the
  page cache dropped, and re-stage on any mismatch. Pack-index staging is local
  only, so size remote disk from materialized sizes. Dataset unions cannot be
  staged remotely; compose their members into one collection first.
- **Where.** `utils/stage_remote.py`, `utils/runtime_repair.py`,
  `assumptions.md` A-REMOTE-CACHE.

## Web console

### PB-30. Remote panels fail intermittently with `core.ssh_async`

- **Symptom.** A remote computer's GPU panel or log tail shows "console could
  not open the connection"; computers on `core.ssh` do not.
- **Cause.** Opening a `core.ssh_async` transport on a thread with a running
  event loop raises `Cannot run awaitable: event loop is running but no
  greenback portal is available`. The console's normal threadpool path does not
  reproduce it, so the trigger is not established.
- **Workaround.** Use `core.ssh` for computers the console must poll. Any fix
  must also respect A-THREADPOOL.
- **Where.** `console/monitor.py` (`gpu_rows_for_computer`), `console/service.py`
  (`_tail_over_transport`).

### PB-31. A transient database failure is reported as 404

- **Symptom.** A console request returns 404 for a node that exists, typically
  while the database restarts.
- **Cause.** A connection error is mapped to "not found". Retry; a 503 for a
  connection error would be the fix.
- **Where.** `web/app/main.py`, `console/service.py`.

### PB-32. Monitor views read local process state

- **Symptom.** On a machine running several FLOW-3D calcjobs, or for jobs on a
  remote computer, the Monitor over-reports which runs hold the seat and cannot
  map GPU processes to runs.
- **Cause.** The seat holder and the GPU-process mapping read `/proc` on the
  machine running the console, falling back to the calcjob process state.
- **Where.** `console/monitor.py` (`seat_holder`), `assumptions.md` A-SEAT.

### PB-33. Monitor batch grouping can merge batches

- **Symptom.** Spool tasks named `sweep1` and `sweep2` (no case suffix) are
  grouped into one row with a wrong count. Cosmetic.
- **Cause.** `monitor.batch_key` strips one trailing `[_-]?c(ase)?\d+` group.
- **Where.** `console/monitor.py`.

### PB-34. Simulations without `Fraction Of Fluid` have no melt-pool stats

- **Symptom.** The stats panel reports per-case "unavailable" for steady or
  reduced-field collections.
- **Cause.** `viz.compute_case_metrics` needs `Fraction Of Fluid`.
- **Where.** `console/viz.py`.

## Tests

### PB-35. Some tests take the production materialize lock

- **Symptom.** The suite hangs on `~/.cache/aiida-pino/runtime/.materialise.lock`
  while a real export runs on the same machine.
- **Cause.** Some tests call `runtime_cache_materialise_slot` under the default
  cache root.
- **Workaround.** Run the suite with `AIIDA_PINO_RUNTIME_CACHE=<tmp dir>`. A
  conftest fixture setting that variable for every test would fix it.
- **Where.** `calculations/base.py` (`runtime_cache_materialise_slot`),
  `tests/conftest.py`.

## Performance notes

### PB-37. Slow paths that are correct

- `PinoDatasetCollectionData.from_simulations` decompresses member 0 to read
  representative shapes that its attributes already hold.
- An `npz_file`-mode `get_array` re-reads the payload on every call, so deriving
  N fields costs N decompressions per simulation.
- The union runtime export (`export_dataset_union`) has no cross-repository
  contract test; the round trip rests on reading the code.
- **Where.** `data/pino_dataset_collection.py`, `data/pino_simulation.py`.
