# assumptions.md

Non-trivial assumptions and deliberate choices in this repo: the ones whose
effect on a result or a run is not obvious from a single line of code. Trivial
input validation and self-evident defaults are out of scope; the source and
`README.md` cover those.

Use this file to sanity-check results: if an ingest or a training run looks
wrong, a wrong assumption here is a likely cause. Known issues that follow
from these choices are listed in `potential_bugs.md`.

Keep entries current: update an assumption when the code it describes changes,
and delete it when it no longer holds. IDs are stable; a gap in the numbering
marks a removed entry.

## Pipeline architecture

**A1. Both ingest paths converge on one node; the ML stage is source-agnostic.**
The FLOW-3D NPZ path (`raw_flow3d_to_dataset`) and the legacy VTK path
(`flow3d_to_vtk` then `vtk_to_dataset`) both emit `PinoDatasetCollectionData`.
Effect: `train`/`eval`/`kfold` consume the dataset node without knowing its
origin, so source differences must be resolved during ingestion (provenance,
metadata, axis convention), not downstream. Source: `pyproject.toml` entry points;
`calculations/eval.py`, `train.py`, `kfold.py`.

**A2. FLOW-3D ingestion is one-step over the NPZ export; the VTK chain is legacy.**
The recommended FLOW-3D path is `raw_flow3d_to_dataset`, which reads a
`flow3D_scripts` run dir (`npz/<case>.npz` + `case_manifest.json`) with numpy
only: no ParaView, no `lpno` import. The two-step
`flow3d_to_vtk` → `vtk_to_dataset` chain (ParaView `pvpython` Code required)
is kept for raw FLSGRF sources without an NPZ export. Effect: a run dir without
an NPZ export needs the legacy chain and its extra setup. Source:
`calculations/raw_flow3d_to_dataset.py`; `workflows/raw_flow3d_import.py`;
`workflows/flow3d_ingest.py`.

**A3. CalcJobs are script-backed and need `lpno` in the Code env.**
`ScriptCalculation` (`calculations/base.py`) stages a bundled adapter from
`aiida_pino.scripts` into the sandbox and runs it as `argv[0]`. The adapters
lazily import `lpno` (the VTK datasource and the FLOW-3D converter)
and raise a clear install hint if it is absent. Effect: the `Code` environment,
not this package, must provide `lpno` and ParaView. Source:
`scripts/*_adapter.py`; `calculations/base.py` (`ScriptCalculation`).

## Data ingestion and provenance

**A4. Per-simulation laser/timing metadata flows in from the case manifest.**
Laser/timing scalars (`V_scan`, `laser_power`, `beam_radius`,
`delta_t_s`, `effective_end_time_s`) are attached to each `PinoSimulationData`
by `read_flow3d_case_metadata` over `runs/<case>/case_manifest.json`
(per-group overrides on the VTK path, direct on the NPZ path). `beam_blocks`
stays `None` for FLOW-3D. Effect: a
FLOW-3D source without case manifests cannot supply laser conditioning, and the
adapters fail loudly instead of storing zeros. Source: `utils/flow3d_import.py`;
`scripts/vtk_to_dataset_adapter.py`; `scripts/raw_flow3d_to_dataset_adapter.py`;
`data/pino_simulation.py`.

**A5. Dataset family identity is hashed from grid params.**
`build_family_identifier` and `collection_compatibility_signature` hash
`grid_params` (x/y/z range, resolution), so geometry, including any axis flip,
is part of dataset-family identity and reuse. Effect: a changed or incorrect
geometry convention fragments families silently. Source:
`data/pino_dataset.py` (`build_family_identifier`);
`data/pino_dataset_collection.py` (`collection_compatibility_signature`).

**A6. FLOW-3D `P`/`V_scan` channels come from case manifests; filename tokens are
an OpenFOAM legacy.** The `lpno` VTK datasource still parses
`results_V_scan_<v>_laser_power_<p>` basenames, but on the FLOW-3D paths the
member `P`/`V_scan` grids are overwritten (or created) from the
`case_manifest.json` values, and an all-zero or metadata-less FLOW-3D member
raises instead of training on zero conditioning. Manifest values also win over
the NPZ provenance scalars (`power_w`/`scan_speed_m_per_s`), which may be NaN;
a disagreement is recorded as a summary warning. Source:
`scripts/vtk_to_dataset_adapter.py` (`_apply_flow3d_laser_channels`);
`scripts/raw_flow3d_to_dataset_adapter.py`.

**A7. `source_type` is the provenance switch and defaults to `flow3d`.**
`vtk_to_dataset` and `raw_flow3d_to_dataset` default `source_type='flow3d'`.
Values are validated against `SUPPORTED_SOURCE_TYPES` (`flow3d`, `openfoam`,
`openfoam_raw`) in the node constructors and the CalcJob ports; no importer in
this repository produces the two OpenFOAM values. Source:
`data/pino_dataset.py`; `utils/data_nodes.py` (`validate_source_type_port`).

## Coordinates and caching

**A8. `flip_axes` is an OpenFOAM orientation convention (Y-negate + Y/Z swap).**
FLOW-3D axes are the SI lab frame and use the identity mapping. The VTK adapter
defaults the flag by `source_type` when a profile omits it
(`resolve_flip_axes`: `flow3d` → `False`, otherwise `True`), and the applied
convention is recorded in metadata as `axis_convention` (`identity` or
`openfoam_y_negate_yz_swap`). An explicit `flip_axes` in the profile always
wins. Source: `utils/vtk_geometry.py` (`resolve_flip_axes`, `axis_convention`).

**A9. The VTK datasource cache is keyed on geometry, and source caches are not
mirrored.** `parse_directory` folds the grid params, `flip_axes` (and any
`categorical_fields`) into the `var_*.npz` cache filename, and the aiida
adapter's symlink mirror excludes pre-existing `var_*.npz` files from the
source tree, so a stale-geometry cache cannot be reused. Source: `lpno`
`openFoam_datasource.py` (`parse_directory`);
`scripts/vtk_to_dataset_adapter.py` (`build_symlink_mirror`).

**A10. Runs are in the lab frame (`moving_ref_frame=False`).**
The dataset adapter hard-codes the lab frame, matching the `flow3D_scripts` NPZ
export (`Noninertialreferenceframe=0`). Effect: no moving-reference-frame
transform is applied at ingest; beam-following views are produced later by the
derivation machinery. Source: `scripts/vtk_to_dataset_adapter.py`.

**A11. The NPZ importer stores the compact NPZ verbatim and serves the
datasource's member layout at read time.** Each simulation keeps the compact
per-case NPZ as one repository object (`storage_mode="npz_file"`): 1-D axes,
provenance scalars and per-field frames, deflate-compressed. At read time
`canonicalize_flow3d_case_arrays` synthesizes every array as `(T, C, X, Y, Z)`
float32 exactly as the VTK datasource would, including broadcast `x`/`y`/`z`
grids, constant `P`/`V_scan` grids and per-frame-constant `time_s` grids (C=1;
`Velocity` C=3). Effect: FLOW-3D NPZ collections are indistinguishable
downstream from VTK-ingested ones, and broadcast channels are never stored.
`PinoSimulationData.store()` refuses an uncompressed `arrays`-mode simulation
unless `AIIDA_PINO_ALLOW_ARRAYS_STORAGE=1` is set (tests and deliberate legacy
imports only). Time subsampling (`time_target_frames`, run-dir layout only)
writes a frame-sliced compact copy with evenly spaced frames, first and last
included. Source: `scripts/raw_flow3d_to_dataset_adapter.py`;
`utils/flow3d_npz.py`; `data/pino_simulation.py`.

**A13. NPZ ingestion trusts `case_manifest.json` over NPZ provenance scalars.**
The NPZ's `power_w`/`scan_speed_m_per_s` scalars may be NaN (the exporter only
records them when passed), so the manifest is the source of truth; manifest
laser power and scan speed must be positive or the case fails loudly, and a
non-NaN NPZ scalar that disagrees with the manifest is recorded as a
`provenance_warnings` entry in the dataset summary. Source:
`scripts/raw_flow3d_to_dataset_adapter.py`; `utils/flow3d_import.py`.

**A14. `Flow3DRunCalculation` leaves large solver artifacts on disk.**
The run adapter invokes `flow3D_scripts`' `run_rectangular_case.py` in place.
The runner root is resolved, in order, from the execution-profile
`flow3d_scripts_root`, the `F3D_SCRIPTS_ROOT` environment variable, and a
sibling `flow3D_scripts` checkout; `--out-root` defaults to `<root>/runs`. Only small artifacts (manifest,
summary, prepin, a `runhyd_tail.log`) are retrieved into the AiiDA repository;
`flsgrf*` and the NPZ stay under the run dir, with absolute paths recorded in
`run_summary`. Licence configuration comes only from the runner's environment,
and the execution profile rejects any licence key, so licence details never
enter the provenance store. A lock file in the spool directory serializes
solver runs that share a licence (see A-SEAT). Source:
`scripts/flow3d_run_adapter.py`; `calculations/flow3d_run.py`;
`utils/profiles.py` (`validate_flow3d_run_execution_profile`).

**A15. Replaced dataset collections are superseded, never deleted.**
When a collection is replaced by a re-ingested reproduction, the old node keeps
its arrays and provenance and only gains extras `superseded=True` +
`replaced_by_uuid`/`replaced_by_pk` (the same convention
`supersede_equivalent_dataset_collections` uses), so existing models and
evaluations that reference it stay resolvable while browsers hide it. Source:
`utils/model_registry.py` (`mark_superseded`,
`supersede_equivalent_dataset_collections`).

**A16. Base-dataset ownership is computed from live membership, not stored on
the simulation.** `base_dataset_of`/`is_orphaned` scan the collections with
`composition_kind="base"` whose `superseded` extra is unset and test membership
by uuid; nothing on the simulation node records its owner. Effect: superseding
a base dataset (the `extend_base_dataset` growth path) releases ownership of
its members implicitly, with no per-simulation cleanup, but any code that
caches an owner pk across a supersession will hold a stale answer. Effect two:
the per-case ingest singletons (`composition_kind="import"`) never claim their
simulation, so a batch member stays orphaned until a real base dataset is
created; only `composition_kind` distinguishes the two, `dataset_role` is
`"base"` for both. Source: `data/dataset_hierarchy.py`.

**A17. Custom datasets keep `dataset_role="composed"`.** A custom dataset
(subset selection by keyword annotations and/or explicit uuids) is stamped
`composition_kind="custom"` but keeps role `composed`, so every downstream
consumer (training profiles, console role filters, derivation) treats it as a
composed dataset; only the hierarchy rules read the kind. Its selection is
recorded in extras `selection_keywords` for inspection, and keyword matching
requires ALL given keywords on a member (`extras["keywords"]`, set-superset,
no partial match). Source: `data/dataset_hierarchy.py::create_custom_dataset`.

**A-EXTRAS-CURRENT. For a simulation, `extras` is the CURRENT value and the
`simulation_metadata` attribute is the value at ingest time.** AiiDA seals
attributes when a node is stored, so anything written or corrected afterwards
can only live in extras. `get_simulation_metadata`
(`data/pino_simulation.py`) therefore reads the stored attribute and
then overlays a fixed key set from extras (`case_name`, `V_scan`,
`laser_power`, `beam_radius`, `delta_t_s`, `laser_stationary`, `description`,
`batch_keyword`, `usable_for_training`), so a reviewer flipping
`usable_for_training` or a retagged `batch_keyword` is reflected.

Effect: `attributes.simulation_metadata` looks authoritative and is not. Some
simulations carry a key only in extras, so an index built on the attribute path
silently omits them. Query extras for any key in the overlay set. For speed,
project the leaf path or the extra; projecting the whole
`attributes.simulation_metadata` document to read one key is an order of
magnitude slower. Source: `get_simulation_metadata`, `utils/annotations.py`.

**A18. The melt regime is a two-gate rule set from visual labelling, not a
physical criterion.** `classify_regime` calls a case keyhole when the
time-averaged cavity aspect (depth / ellipse-fit min radius) exceeds 1.0 **and**
the max cavity depth reaches `KEYHOLE_MIN_DEPTH_RADII` (2.1) beam radii; a
cavity under `LOF_DEPTH_CELLS` cells deep is lack of fusion. Both thresholds
were chosen to reproduce hand-labelled cases, not taken from the literature.
Effect: the boundary is only as good as its anchors, so keyhole/conduction
counts are not comparable across stats versions. Cavity depths are quantized to
the cell size (10 µm on these grids), which is what makes a gate between two
anchors at 90 and 120 µm well defined. Cached melt-pool stats carry
`regime_criterion` (`viz.REGIME_CRITERION`), which names the rule that produced the label.
Source: `console/viz.py` (constants block + `classify_regime`);
`tests/test_viz_stats.py::CALIBRATION_ANCHORS`.

**A19. Width and length are ellipse-fit full axes, not bounding-box extents.**
`melt_*`/`cavity_*` `width`/`length` are twice the second-moment ellipse radii
of the footprint at that quantity's own peak frame, so they follow the pool's
elongation (a curved random track is measured along its track, not across the
grid). The separate `*_xlen_*`/`*_ylen_*` keys are the axis-aligned bounding box
and will disagree with width/length whenever the track is not grid-aligned.
Melt and cavity peak at different frames, so their sizes are not simultaneous
measurements. Source: `console/viz.py` (`_footprint_radii_um`,
`compute_case_metrics`).

## Testing

**A12. The test suite runs with no daemon, compute resource, or license.**
`tests/conftest.py` enables `aiida.tools.pytest_fixtures` (in-memory profile +
`aiida_localhost`), builds Data nodes from numpy arrays via factories, and runs
jobs against fake `Code` executables that synthesize output trees. Effect: tests
need an environment with `aiida-core` installed but no real scheduler or GPU;
running in an env without `aiida` fails at collection. Source: `tests/conftest.py`.

## A-NO-DERIVED-INGEST: there is no ingest path for pre-derived arrays, on purpose

`lpno`'s `build_steady_split.py` writes a time-averaged, beam-centred NPZ
folder, and the obvious next thought is to ingest it as a dataset. There is no
such path (ingest takes raw FLOW-3D NPZ or VTK only), and that is deliberate:
importing derived frames as new simulations would duplicate provenance, giving
the same physics two unrelated lineages and no way to tell later which was the
original.

Derived representations are produced by the derivation machinery from a stored
parent (`dataset_role: derived`, `parent_collection_pk`), so the lineage stays
single.

## A-REMOTE-CACHE: a remote runtime cache is an immutable, pre-staged copy

A job on a non-local computer reads its dataset from a copy that was put on that
machine ahead of time by `aiida-pino stage-remote`, found at
`prepare_for_submission` by cache NAME (`utils/runtime_cache.py`), never by
path. Three things follow, and all three are deliberate.

**It cannot be built at submit time.** The payloads live in the submitting
machine's object store, so building for a remote job would mean exporting tens
of GB locally with no way to put them there. `stage_dataset_input` therefore
refuses, naming the `stage-remote` command, instead of proceeding with a local
path the remote cannot read.

**There is no remote repair.** `runtime_repair.repair_dataset_dir` rebuilds a
vanished cache from the object store at job start; on a compute machine there is
no object store, so a cache deleted out from under a running remote job is
unrecoverable and the job dies. Drain the queue before clearing a cache,
locally and remotely alike.

**A remote cache is never evicted and never pinned by aiida-pino.**
`runtime_cache_evict` walks LOCAL paths and skips jobs on remote computers
entirely: it cannot see the remote copy, and on a machine that shares a home
directory with the submitting machine it would otherwise walk a same-named
local directory belonging to something else. Cleaning a compute machine is a
manual act.

The name a copy is stored under encodes the field selection and `pack_index:
False`, so a field-set change misses (it never silently trains on the wrong
fields), and a locally-indexed cache can never be handed to a remote job.

## A-REPAIR: sealed attributes are rewritten only through recorded backfills

Correcting demonstrably wrong stored data may bypass AiiDA's attribute seal
(`node.backend_entity.set_attribute`, in `utils/integrity.py` and
`PinoSimulationData.compress_to_npz_file`); every such write is marked in the
node's `backfilled_keys` extra (`utils/provenance_mode.mark_backfilled`), and
`python -m aiida_pino.utils.integrity --rehash` brings `_aiida_hash` back into
agreement. Nothing in this repo reuses nodes by AiiDA hash; dataset reuse goes
through `content_signature`.

## A-THREADPOOL: the request threadpool is held under the database pool

**Assumption.** Under burst, a console request should wait for a thread instead
of racing for a database connection it cannot get.

**Where.** `_cap_threadpool_to_db_pool()` in `web/app/main.py`, at startup;
logged as `threadpool` on `startup_profile_load_done` (derived from
`pool_size + max_overflow` minus headroom for aiida's own connections).

**Effect.** Every endpoint is a sync `def`, so anyio runs it on a worker thread,
and aiida-core pins one connection per live thread. With the default 40 threads,
requests beyond the connection pool waited the full 30 s pool timeout and then
returned HTTP 500, on whichever endpoint they happened to hit. With the cap,
concurrent requests queue for a thread and all complete. The trade is
deliberate: a slow answer over a wrong one. If the console is ever run with more
than one uvicorn worker, or the pool is resized, this cap must move with it.

## A-SEAT: the solver lock holder is read from the kernel's lock table

**Assumption.** FLOW-3D runs that share a licence are serialized by an `flock`
on `<spool>/.seat.lock`. The only reliable statement about which run holds it
is `/proc/locks` for that inode; the spool's `running/` entry names the task its
worker *started*, which may itself be blocked on the same lock.

**Where.** `console/monitor.py: seat_holder()` / `parse_proc_locks()`, consumed
by `monitor_status` and `flow3d_queue_status`.

**Effect.** The holder's `/proc/<pid>/cwd` identifies the calcjob, so exactly one
AiiDA run is ever reported as "running" on the lock. This is local-Linux-only by
construction; where `/proc` is unreadable (remote computer, another platform)
the code falls back to the calcjob process state, which over-reports on a
machine running several FLOW-3D calcjobs at once. The lock is a local file, so
it serializes runs on one machine only.

## A-PACKSLICE: staging reads payloads as raw pack byte ranges where possible

**Assumption.** A packed `payload.npz` whose `get_object_meta` reports
`size == pack_length` is stored verbatim (uncompressed) in the pack, so the
byte range `(pack_file, offset, size)` IS the payload, and reading or
`sendfile`-ing that range is equivalent to `repository.open`.

**Where.** `utils/runtime_export.payload_locator` / `read_locator_bytes` /
`sendfile_locator`, used by the parallel dataset export workers, and the
pack-index loader in `lpno`.

**Effect.** Workers read payloads without an AiiDA session, and copies run
kernel-side, roughly 2.7x faster than streaming through the repository. The
compact NPZs are already deflated internally, so the container stores them
uncompressed despite its `compression_algorithm: zlib+1`. When the condition
fails the locator returns None and the parent falls back to `repository.open`,
which is always correct. Verified byte-identical against the serial path in
`tests/test_runtime_export.py`.

A repack (`verdi storage maintain`) moves and reorders objects, so a stored
offset can land on a different object of the same size. The loader therefore
re-resolves every open by content hash through the container's own index and
treats the stored offset as a hint. The offset-plus-ZIP-end-record check
survives only as the fallback for indexes written without content hashes and
for containers it cannot find; that fallback cannot tell two same-size NPZs
apart. Do not run `verdi storage maintain` while staging or training is active.
