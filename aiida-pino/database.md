# database.md: how aiida-pino stores its data

This repo has no separate database of its own. Its "database" is the AiiDA
provenance store of the active profile (a PostgreSQL or SQLite node table plus an
on-disk object repository) holding a set of custom `pino.*` `Data` node classes.
All queries go through `orm.QueryBuilder` over node attributes and extras; there
is no bespoke index or schema table. Every custom node carries a `format_version`
attribute (all currently `1`) and an `aiida_pino_kind` extra that names its role,
which is the coarse type filter the browsers use.

Any AiiDA profile works (`aiida-pino --profile NAME`, or the configured
default). Primary keys are local to one database: an archive import or a
backend migration renumbers them, so an old pk can name a *different* node.
UUIDs survive both and are the safe identifier.

Simulation arrays are stored one way only: `storage_mode="npz_file"`,
the compact per-case NPZ kept verbatim as `payload.npz`. The alternative
(`arrays`, one uncompressed `.npy` per field) is about 20x larger, and `store()`
refuses it unless `AIIDA_PINO_ALLOW_ARRAYS_STORAGE=1` is set. Broadcast channels (`x`, `y`, `z`,
`P`, `V_scan`, `time_s`) are synthesized on read and never stored.

Array data can be deleted while the node, its metadata and its provenance stay
(`aiida-pino evict PK`, dry-run by default, refused on `core` datasets). Such a
node raises `DataEvictedError` if its arrays are read. The disk is returned by
`aiida-pino reclaim`, which repacks only the packs it touched; do *not*
reach for `verdi storage maintain --full`, which rewrites every pack, needs
exclusive access and is the operation `README.md` warns against.

See `assumptions.md` (A1, A4, A5, A7) for the intent behind these choices and
`potential_bugs.md` for the consequences a user can hit.

## Node types

Entry-point names (`pyproject.toml`, `[project.entry-points."aiida.data"]`):

| Entry point | Class | Base | Holds arrays? |
| --- | --- | --- | --- |
| `pino.simulation` | `PinoSimulationData` | `orm.ArrayData` | yes |
| `pino.dataset_collection` | `PinoDatasetCollectionData` | `orm.Data` | no (metadata only) |
| `pino.model` | `PinoModelData` | `orm.FolderData` | files |
| `pino.model_evaluation` | `PinoModelEvaluationData` | `orm.Data` | no (metadata only) |
| `pino.dataset` | `PinoDatasetData` | `orm.ArrayData` | yes (legacy) |
| `pino.dataset_union` | `PinoDatasetUnionData` | `orm.Data` | no (metadata only) |

The last two are the pre-collection data model and have no instances in a
current database. Their classes and the ~30 references in `model_registry.py`
are still carried, but the first four node types describe everything a
database holds.


### `PinoSimulationData` (`orm.ArrayData`)

One physical simulation, one node. Built by `from_npz_file`
(`data/pino_simulation.py`), the production path and the only one the
`store()` guard permits without an explicit env override. `from_arrays` is the
legacy constructor, kept for fixtures and the migration suites.

- Storage: the source compact NPZ verbatim as a single `payload.npz` repository
  object, plus `_time_values_s.npy`. Reads go through
  `canonicalize_flow3d_case_arrays`, which re-derives the canonical
  `(T, C, X, Y, Z)` view and synthesizes the broadcast channels (`x`, `y`, `z`,
  `P`, `V_scan`, `time_s`) from metadata; they are not stored. The
  `array_aliases` attribute maps original field names to stored names.
  `payload_content_hash()` reads the object's sha256 key straight from the
  repository metadata; the object store is content-addressed, so the hash of a
  simulation's data costs nothing to obtain.
- Legacy `arrays` mode wrote one `.npy` per field, uncompressed; `store()`
  refuses it for new nodes and `aiida-pino compress` converts old ones.
- Attributes: `source_type`, `source_origin_kind`, `source_folder_name`,
  `source_group`, `source_uri`, `requested_fields`, `relative_paths`,
  `array_aliases`, `array_shapes`, `num_timesteps`, `spatial_shape`, `delta_t_s`,
  `effective_end_time_s`, the four laser scalars `V_scan`, `laser_power`,
  `beam_radius`, `beam_blocks`, the three annotations `description`,
  `batch_keyword`, `usable_for_training`, `simulation_identifier`,
  `simulation_fingerprint`, and the whole normalized dict again as
  `simulation_metadata`.
- Extras (query surface): `aiida_pino_kind="simulation"`, `source_type`,
  `source_folder_name`, `source_group`, `simulation_identifier`,
  `simulation_fingerprint`, `num_timesteps`, `delta_t_s`, `effective_end_time_s`,
  the four laser scalars, and the three annotations `description`,
  `batch_keyword`, `usable_for_training`.
- Annotations (optional, human-facing; `from_arrays` normalizes and promotes
  them to attributes and extras): `description` is a free-text note on where the
  run came from and how it was made; `batch_keyword` is a regroup tag so
  simulations from one batch share a query key; `usable_for_training` is a bool
  (default `True`) marking whether the run is fit for training. None of the three
  feed `simulation_fingerprint`, so annotating a simulation never changes its
  identity or dedup behaviour. They are mutable through the node extras: flip
  `usable_for_training` or retag `batch_keyword` after storage without
  re-ingesting; `get_simulation_metadata()` reads the current extra value.
  Unset `description`/`batch_keyword` normalize to `""`. On the raw-NPZ import
  path they come from the dataset profile keys `description`, `batch_keyword`,
  `usable_for_training` and apply to every member of the run directory (one
  batch); other import paths get the defaults.
- Identity: `simulation_fingerprint` is a 16-hex SHA1 over source ids, relative
  paths, `grid_params`, and `requested_fields` (`build_simulation_fingerprint`).
  `simulation_identifier` is `folder::group::fingerprint[:8]`.
- Runtime export: `export_runtime_arraydir` writes a `manifest.json` plus the
  per-array `.npy` shards, hardlinking out of the repository when it can.

### `PinoDatasetData` (`orm.ArrayData`)

The single-node, already-gridded dataset. This is the legacy single-tensor
kind; the current ingest path emits collections instead (A1), so this class now
survives mainly as the member type of `PinoDatasetUnionData` and as the output
of the legacy migration.

- Arrays: one `.npy` per field (aliased as above); time is folded into the tensor
  as a dimension or channel, so there is no separate time array.
- Attributes: `source_type`, `dataset_role` (`base`/`derived`),
  `canonical_layout`, `family_identifier`, `dataset_identifier`, `cache_basename`,
  `source_folder_name`, `requested_fields`, `input_dir`, `source_uri`, lineage
  pointers `base_dataset_uuid`/`base_dataset_pk`/`parent_dataset_uuid`/
  `parent_dataset_pk`, `derivation_profile`, `derivation_hash`, `reconstructable`,
  `array_aliases`, `array_shapes`, and the full `dataset_metadata` dict.
- Extras: `aiida_pino_kind="dataset"`, `source_type`, `dataset_identifier`,
  `dataset_role`, `family_identifier`, `canonical_layout`, `reconstructable`,
  `derivation_hash`, and (when set) `base_dataset_uuid`, `parent_dataset_uuid`,
  `source_folder_uuid`, `source_folder_pk`.
- Runtime export: `export_runtime_arraydir` (v1 manifest + shards) and
  `export_npz_dataset` (writes the `var_*.npz` cache basename the PINO datasource
  expects, plus a hardlinked runtime alias).

### `PinoDatasetCollectionData` (`orm.Data`, metadata only)

The current public dataset unit and the node all ingest paths converge on (A1).
It holds no arrays; it is an ordered, metadata-only wrapper over member
simulations. Built by `from_simulations` after `validate_collection_members`
confirms every member shares one `collection_compatibility_signature`.

- Membership: `member_simulation_uuids` and `member_simulation_pks` attributes,
  in order. `iter_member_simulations()` loads them with `orm.load_node`. This is a
  by-value reference, not an AiiDA link (see Provenance below).
- Attributes: `dataset_role` (`base`, `subset`, `composed`, `derived`) and
  `composition_kind` (`import`, `base`, `composed`, `subset`), two overlapping
  vocabularies for one idea. They do not agree row-by-row and collapsing them into
  one field is open work (`docs/developer_notes.md`, "Roadmap");
  `source_type`, `source_uri`, `source_folder_name`, `family_identifier`,
  `cache_basename`, `dataset_identifier`, `compatibility_signature`,
  `simulation_count`, `time_steps_min`/`time_steps_max`, `uniform_timesteps`,
  lineage `base_collection_uuid`/`pk` and `parent_collection_uuid`/`pk`,
  `derivation_profile`, `derivation_hash`, `reconstructable`, and a
  *representative* `array_shapes`/`spatial_shape`/`num_timesteps` computed by
  applying the derivation profile to member 0 (no arrays are stored).

  **Query path:** all of these live under `attributes.dataset_metadata.*`. A
  QueryBuilder filter on the wrong JSONB path returns zero rows without an
  error, so `attributes.dataset_metadata.dataset_role` is
  right and `attributes.dataset_role` silently matches nothing. The exceptions
  promoted to top-level attributes are `member_simulation_uuids`,
  `member_simulation_pks`, `simulation_count`, `compatibility_signature` and
  `content_signature`.
- Extras: `aiida_pino_kind="dataset_collection"`, `source_type`,
  `dataset_identifier`, `dataset_role`, `family_identifier`, `composition_kind`,
  `compatibility_signature`, and `superseded` (bool; see dedup below).
- Runtime export: `export_runtime_arraydir` writes a v2 manifest and, per member,
  either re-exports the stored simulation (base role) or applies the derivation
  profile on the fly via `derive_simulation_arrays` (derived role). A derived
  collection does not duplicate arrays; it stores only the profile and recomputes.

### `PinoDatasetUnionData` (`orm.Data`, metadata only)

A reusable, ordered concatenation of compatible datasets, holding no arrays.
Built by `from_members`. Its members are always `PinoDatasetData` (the
single-node kind).

- Attributes: `union_name`, `union_identifier` (name + 12-hex SHA1 of member
  UUIDs), `member_dataset_uuids`, `member_dataset_pks`, `compatibility_signature`,
  `total_samples`.
- Extras: `aiida_pino_kind="dataset_union"`, `union_identifier`,
  `member_dataset_uuids`, `compatibility_signature`.
- Runtime export (`export_dataset_union`, in `model_registry.py`): stages each
  member's arraydir under `members/` and records sample offsets; it never
  concatenates payloads on disk.

### `PinoModelData` (`orm.FolderData`)

A trained-model bundle. `from_training_artifacts` copies only a compact subset
out of the raw training artifacts folder into the node repository: files with
extensions in `{.ckpt,.pt,.pth,.yaml,.yml,.json,.txt,.csv}` plus the `.hydra`,
`validators` and `monitors` directories (`MODEL_KEEP_EXTENSIONS`/`MODEL_KEEP_DIRECTORIES`).

- Repository: the kept checkpoint(s) and config files. `resolve_primary_checkpoint`
  ranks candidates (`.pth` < `.pt` < `.ckpt`, `u_network*` preferred).
- Attributes: `model_metadata` (holds the frozen `training_profile`,
  `derivation_profile`, `training_source_kind`/`pk`/`uuid`/`label`,
  `training_source_summary`, `primary_checkpoint`, `training_summary`,
  `config_name`) and `copied_files`.
- Extras: `aiida_pino_kind="model"`, `model_name`, and (when present)
  `training_source_uuid`, `training_source_kind`.

### `PinoModelEvaluationData` (`orm.Data`, metadata only)

An immutable evaluation record. `from_payload` stores `evaluation_metadata`
containing `model_pk`/`uuid`/`name`, `dataset_source_kind`/`pk`/`uuid`/`label`
and its full summary, `evaluation_profile`, `derivation_profile`, `metrics`, and
`eval_summary`. Extras: `aiida_pino_kind="model_evaluation"`, `model_uuid`,
`dataset_source_uuid`. `model_evaluations_for_model` queries it by
`extras.model_uuid`.


## Composition

- **simulation to collection**: `PinoDatasetCollectionData.from_simulations`
  validates a shared `compatibility_signature` (source_type, sorted fields,
  `grid_params`, `spatial_shape`, non-time shapes) and stores the ordered member
  UUID/PK lists. A base collection re-exports members verbatim; a derived
  collection stores only a `derivation_profile` and recomputes arrays with
  `derive_simulation_arrays` (spatial super-resolution, moving-reference-frame
  crop/shift, trailing-window time average, time-drop) at export time.
- **collection to union**: unions are built from `PinoDatasetData` only
  (`_flatten_dataset_members`). `validate_union_members` flattens
  nested unions, de-duplicates by UUID, and requires one shared
  `dataset_compatibility_signature`; `total_samples` is the sum of member sample
  counts.
- **model to evaluation**: a `PinoModelData` is produced from a
  training CalcJob's artifacts against a collection; a `PinoModelEvaluationData`
  points at one model plus one dataset source.

## Provenance capture

Ingest and ML runs are ordinary AiiDA processes, so the CalcJob/WorkChain DAG is
the provenance record for *runs*. `RawFlow3DImportWorkChain` submits
`RawFlow3DToDatasetCalculation`, whose parser (`VtkToDatasetParser`, shared with
the legacy VTK path) emits the collection as a create output. Training and eval
nodes are produced inside `@calcfunction`s (`materialize_model_node`,
`materialize_evaluation_node`), so model-to-dataset and
evaluation-to-{model,dataset} are real provenance edges.

Membership is a provenance edge too:

- **Import to simulations.** The import parser (`VtkToDatasetParser`) emits
  every simulation it creates under the CalcJob's dynamic `simulations` output
  namespace, so
  `import job -> PinoSimulationData` is a CREATE link. A simulation that already
  existed is *not* re-emitted: a Data node has exactly one creator, and the
  first import keeps it.
- **Collection to member simulations.** `create_base_dataset` and
  `extend_base_dataset` compose through the `compose_dataset` calcfunction with
  the members as inputs, so `simulation -> compose_dataset -> collection` is a
  real path. `extend_` also passes the dataset it grows as `previous`. The same
  helper (`utils/compose.py`) backs the console's "create collection".
- **Derived collections have no such path.** A derived collection has no creating process, whatever its date, so its
  membership is the `member_simulation_uuids` attribute alone, and a graph export does not follow it.
- Model and evaluation nodes are produced inside `@calcfunction`s
  (`materialize_model_node`, `materialize_evaluation_node`), so
  model-to-dataset and evaluation-to-{model,dataset} have always been real
  edges.

**Nodes created before these membership links existed lack all of this and
cannot be given it.** AiiDA provenance is immutable, and a fabricated edge
would be worse than an absent one. For those older simulations and
collections:

- no simulation has a creating process;
- membership exists only as the `member_simulation_uuids` attribute;
- `verdi archive create -N <old dataset>` exports the collection node alone
  (1 node and no repository files);
- `verdi node delete` on a member does not mention the collections holding it.

`python -m aiida_pino.utils.integrity` reports exactly which nodes are in that
state (expect 2-5 minutes: it walks every simulation and collection and
prints only at the end), along with dangling members and byte-identical duplicates.

Still by-value: union to member datasets, which is moot because there are no
union nodes left.

## How a dataset is marked and queried

### Why a run failed, and where its live output is

- `failure_class` on every failed calcjob, decided at parse time from the
  stdout AiiDA retains. `exit_status 300` is the same for a config crash, an
  OOM and a stall-kill, and the workdir's live-tail file is deleted with the
  workdir, so this extra is what tells them apart afterwards. Its values are
  `no_output`, `waiting_for_resources`, `unclassified`, `oom_gpu`,
  `config_error`, `disk_full` and `python_exception`. `no_output` and
  `unclassified` are different: the first is a retrieval problem, the second a
  gap in the pattern table.
- `python -m aiida_pino.utils.queryable_paths <kind> --live-log PK` walks
  node -> calcjob -> remote workdir -> log from a pk at any level and prints the
  path plus a ready-to-paste `tail -f`. When there is no path it says which of
  the three reasons applies.

**Ops note: `py-spy` needs `ptrace` access.** On a host with
`kernel.yama.ptrace_scope=1` it cannot attach to a live process without root,
so a hung run is diagnosed from `/proc/<pid>` deltas (`status`, `stack`,
`wchan`, `io`) plus the post-mortem traceback. Raising `ptrace_scope` is a
privileged system change and is left to the machine's administrator.

### How far a training run got (extras)

- `max_step_reached` (+ `step_lines_logged`) on training calcjobs, parsed from
  the retrieved stdout and stamped before the summary check, so a run that died
  without writing `train_summary.json` still records its progress. A 0 is a
  finding: it is the OOM-on-first-step signature.
- Why: an exit status says the step failed; it says nothing about whether
  anything was learned. A run that never reached step 1 and one that got past
  step 1000 can fail with the same status, and the `[step: N]` lines in the
  scratch live-tail log are deleted with the workdir.
- Eval jobs log no steps and carry no `max_step_reached` (a `0` there would
  mislead); only `PinoTrainParser` sets `tracks_steps`.

### Grid shape is already queryable: do not iterate to get it

`extras.mesh_cells` holds the spatial grid on most simulations, and
`attributes.array_shapes` is projectable. One projected query over the whole
database takes seconds, while `load_node` costs about 16 ms per simulation, so
iterating over a few thousand simulations takes a minute for the same answer
and makes a whole-database sweep look as if it needs backgrounding and caching.
Project, never iterate.

### Which alloy a simulation is (extras)

- `material_name`, `material_file`, `solidus_k`, `liquidus_k`: read from
  `flow3d_case_manifest.material` at creation (and backfilled on older
  simulations), so the alloy is one filter away and no query has to reach three
  levels down into the manifest.
- `material_source` is on every simulation: `"manifest"` or `"unrecorded"`.
  The gap is recorded as a gap, so `extras.material_source == "unrecorded"`
  finds it in one filter. The unrecorded cases are mostly resampled
  straight-path simulations, whose manifests carry
  only `derived`/`input`/`note` and whose descriptions name no alloy.
  Their alloy is inferable from the straight-path family, but inference is not
  provenance: nothing writes a guessed solidus, because scoring one alloy with
  another's melting point is exactly the silent error
  `melt_metrics.melt_mask` refuses to make.

### Names, sizes and stale pks

- **The name of a collection is its `label`** (a real database column, set on
  every collection and matched by `aiida-pino datasets --search`). There is no
  `collection_name` attribute: projecting `attributes.collection_name` returns
  `None` on every node without an error, because QueryBuilder projects missing
  paths as `None`. The `collection_name` you see in `console/schemas.py` is a
  request field only.
- **`num_members` (extra, all node vintages)**: member count as one integer, so
  "find the collection with N cases" is one filter instead of projecting
  `member_simulation_uuids` over the population and `len()`-ing client-side.
- **`legacy_pk` (extra, on nodes carried over from an earlier database)**: the
  node's pk in that earlier database, which names a different node here.
- **Manifests**: `simulation_metadata.flow3d_case_manifest` is a dict on every
  stored simulation, never a JSON string; the string form only exists when
  reading a `.manifest.json` sidecar from disk. Inside it, `input` is the parsed
  run config (dict) and `source_config` is a path string by design; do not
  `json.loads` it. `case_manifest()` in `data/pino_simulation.py` normalises all
  of these shapes.
- **`held_out` / `train_overlap_n` (extras on evaluations)**: whether the scored
  set shares cases with the model's training set. `held_out == False` renders
  red everywhere and never ranks; absent means unknown provenance, which does
  not mean clean.


- `source_type` (attribute + extra): the provenance switch, `"flow3d"`,
  `"openfoam"`, or `"openfoam_raw"` (A7). Default is `"flow3d"`
  (`DEFAULT_SOURCE_TYPE`); validated against `SUPPORTED_SOURCE_TYPES` in the
  node constructors and CalcJob ports.
- `family_identifier`: `folder__source__hash(grid_params, requested_fields,
  source_uri)` (`build_family_identifier`); shared by a canonical base and all its
  derivatives so a family can be gathered with one filter (A5).
- `dataset_identifier`: `family__variant_token`, where the variant token is the
  cache basename stem, so base and each derived layout are distinct.
- `compatibility_signature`: the union/collection compatibility gate; equal
  signatures may be concatenated.
- `simulation_fingerprint` (extra): the simulation dedup key used by the parser.
- `dataset_role` (`base`/`derived`) + `canonical_layout` + `reconstructable`:
  layout/lineage markers.
- `superseded` (collection extra): the "this one was replaced" marker.
  Superseded collections can be a large share of a working store, so the
  marker matters when reading any listing.
  When a new collection turns out to hold exactly the same data as an older one
  (`content_signature`: the same members with the same derivation, regardless of
  name or role), the older node is marked `superseded=True` with
  `replaced_by_uuid` pointing at the survivor, and every listing hides it. The
  node is not deleted and stays valid as a reference from any model trained on
  it. Two consequences: a dataset you cannot find in the console may be
  superseded rather than gone, and `core` reference sets are never
  superseded: if new content collides with one, the core node wins.

### Attributes corrected after storage, and what `_aiida_hash` means here

AiiDA seals a stored node's attributes. On some collections, stale
`member_simulation_pks` and drifted `content_signature` values were corrected
through the backend setter, around that seal. The values are right; the
correction is not visible from the node itself.

Consequences:

- **`_aiida_hash` is AiiDA's caching hash, not an integrity fingerprint.** A
  rewritten attribute leaves it describing the old content until it is
  recomputed (`python -m aiida_pino.utils.integrity --rehash`), and even then
  it never certifies that a node was unmodified.
- **Some collections advertise members that no longer exist.** They carry a
  `members_lost` extra naming the missing uuids. They are kept, because
  deleting one takes its whole solve record with it. The collection itself is
  empty of *data*, and `iter_member_simulations()` raises on it.

### The stored physical channels and their units

Everything is SI. `x/y/z`, `P`, `V_scan` and `time_s` are *broadcast* channels,
synthesized on read from metadata and never stored as arrays.

| Channel | Meaning | Unit |
| --- | --- | --- |
| `T` | temperature | K |
| `alpha` | metal volume fraction (1 = metal, 0 = void/gas) | – |
| `fl` | liquid fraction of the metal | – |
| `u`, `v`, `w` | velocity components | m/s |
| `p` | pressure | Pa |
| `x`, `y`, `z` | cell-centre coordinates | m |
| `P` | laser power for the case | W |
| `V_scan` | laser scan speed for the case | m/s |
| `time_s` | time of the frame | s |
| `beam_radius` | laser spot radius | m |

Metrics on `T` are stored normalised (divided by `ref_scales["T"]`, usually
3000); each evaluation carries its own factor in `metric_units` (see
`PinoModelEvaluationData` above).

### Labels a user applies (all extras, all mutable)

None of these change a node's identity; they are annotation, and a machine
never writes to the ones a human owns.

| Extra | On | Meaning |
| --- | --- | --- |
| `split` | simulation | one of `train` / `val` / `test` / `eval` / `excluded` / `duplicate_excluded` / `unassigned` (`aiida_pino.split_values.SPLIT_VALUES`). `test` and `eval` are held out; `excluded` and `duplicate_excluded` mean "in no split at all" (use `excluded` for new work); `unassigned` is the default. The console composes subset datasets from `train`/`val`/`test` and records how they were drawn in the subset's `split_recipe` (see `assign_splits`); `aiida-pino split PK VALUE --reason` appends who, when and why to the node's `split_audit` extra. |
| `keywords` | any node | free-list tags. `core` is load-bearing: it marks protected reference sets that deletion, supersession and eviction all refuse. |
| `folder` | dataset, model | display grouping in the console: a slash-separated path in this extra (`console/folders.py`); a node is in at most one folder, and a folder exists as long as something is in it. Not an AiiDA group. |
| `description` / node label | any node | the human title shown in listings. |
| `content_hash` | simulation | sha256 of the stored payload, used to detect the same case imported from two directories. |
| `solver_version` | simulation, FLOW-3D calcjob | the FLOW-3D build that produced it. |
| `superseded`, `replaced_by_uuid` | dataset | retired in favour of another node; filtered out of listings. |

- Training suitability: only `PinoDatasetCollectionData` is accepted for
  training
  (`normalize_dataset_collection_node`); unions and single `PinoDatasetData`
  nodes are effectively not trainable through the current workchains.

## Node graph

The current path, labelled in the pipeline's vocabulary.
Every arrow is a real AiiDA link for nodes created since membership became a
provenance edge; see "Provenance capture" for what older nodes do and do not have.

```
 a FLOW-3D run directory (npz/<case>.npz + case_manifest.json)
      │  raw_source_path (Str)
      ▼
 RawFlow3DToDatasetCalculation ─┬─create─▶ one simulation per case   (PinoSimulationData)
   "the import job"             │            = the fields of one solve, over time
                                └─create─▶ the imported dataset      (PinoDatasetCollectionData)

 simulations ──inputs──▶ compose_dataset ──create──▶ a training dataset
   "which cases"          (calcfunction)              = an ordered, named set of cases

 dataset ──input──▶ PinoTrainCalculation ──create──▶ training artifacts (checkpoints, configs)
                          │
                          └─▶ materialize_model_node ──create──▶ a model  (PinoModelData)

 model + dataset ──inputs──▶ materialize_evaluation_node ──create──▶ an evaluation
                                                        = metrics for that model on that data,
                                                          now carrying their own unit
```

Read backwards, the graph leads from any published number through evaluation →
model → training dataset → the individual simulations → the import job that
produced them.

`PinoDatasetUnionData` and `PinoDatasetData` do not appear because no instance
of either exists.


## Open structural work

Unbuilt ideas are in the roadmap of `docs/developer_notes.md`; this document
describes only what is built.


## FLOW-3D data from flow3D_scripts

FLOW-3D is a first-class source. What a FLOW-3D ingest stores:

- **Ingestion**: `raw_flow3d_to_dataset` (via `RawFlow3DImportWorkChain`) reads
  a `flow3D_scripts` run dir (`npz/<case>.npz` + `case_manifest.json`)
  directly with numpy; no ParaView, no `lpno` import, no `var_*.npz`
  cache.
- **Generation**: `Flow3DRunCalculation` (and `Flow3DGenerateWorkChain` /
  `Flow3DBatchWorkChain`) launches the `flow3D_scripts` runner itself as a
  tracked job (full config JSON pass-through, prepare-only support, NPZ export,
  seat-lock aware) and chains straight into the NPZ import, so a FLOW-3D
  simulation can be produced and ingested with full provenance in one workchain.
- **Conditioning channels**: member `P`/`V_scan` grids are constant fields of
  the manifest's `power_w`/`scan_speed_m_per_s`; an all-zero or metadata-less
  FLOW-3D member raises instead of storing silently-zero conditioning.
- **Per-simulation metadata**: the four laser scalars (`beam_blocks` stays
  `None`, since FLOW-3D has no such concept) as attributes and queryable extras, real
  `time_values_s` (reconciled against the array time axis), plus
  `flow3d_case_manifest` (the full manifest: input config, derived geometry and
  scan-path fields, parallel settings), `flow3d_material` (the manifest's
  material block: properties, optical data, table row counts) and
  `flow3d_material_source` (the parsed `materials/<name>/<name>.json` when a
  `materials_root` is supplied) inside `simulation_metadata`.
- **Coordinates**: `flip_axes` defaults by `source_type` (`flow3d` → identity)
  and the applied `axis_convention` is recorded in metadata.
- **Validation**: `source_type` is enforced against `SUPPORTED_SOURCE_TYPES` in
  the node constructors and CalcJob ports.

Suitability for training is the per-simulation `split` extra (`train`, `val`,
`test`, or `excluded` with an `exclusion_reason`), so "which simulations can a
model train on" is a query on that extra. Setting it automatically at store
time from cheap checks is on the roadmap of `docs/developer_notes.md`.
