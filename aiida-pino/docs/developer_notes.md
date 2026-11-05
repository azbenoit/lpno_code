# Developer notes

Practice and roadmap for aiida-pino. See also `README.md` (user guide),
`database.md` (data model), `assumptions.md` (design choices) and
`potential_bugs.md` (known issues).

## Environment

- Work in one environment with `aiida-core` 2.8, this package installed
  editable and the sibling `lpno` package (`tools/setup_env.py --with-aiida` at
  the repository root builds one). Tests fail at collection without `aiida`.
- Unless `AIIDA_PATH` is set, every environment reads the same
  `~/.aiida/config.json`, and a newer `aiida-core` migrates it in place on first
  use, after which the older version cannot read it. Check installability with
  `pip install --dry-run`, or export `AIIDA_PATH=<scratch dir>` before a trial
  install. AiiDA keeps `config.json.<timestamp>` from before a migration.
- Machine-specific settings (hostnames, credentials, profile files, cluster
  paths) stay in untracked local files such as `daemon.env` (template:
  `daemon.env.example`) and `site.env` (checked by `tools/check_site_leaks.py`).

## Tests and checks

- `tests/conftest.py` provides an in-memory profile, an `aiida_localhost`
  computer, node factories that build stored nodes from numpy arrays
  (`pino_simulation_factory`, `pino_dataset_factory`, `pino_model_factory`, ...)
  and fake executables that synthesize job output trees. Adapter functions are
  tested with `importlib` monkeypatched, so `lpno` and ParaView are never
  imported.
- `python -m pytest -m "not slow" -q` is the fast tier and `python -m pytest -q`
  the full suite. The `slow` marker is applied automatically from each test's
  fixtures; do not set it by hand.
- Use `--timeout=300` (`pytest-timeout`, in the `dev` extra). A hang is usually
  the FLOW-3D lock: the run adapter holds `<spool>/.seat.lock` for its whole
  process, and `F3D_SEAT_HELD` tells a nested call the lock is already held.
- Write the test first and check that it fails without the change. Existing
  tests are part of the oracle, and every bug fix adds a regression test.
- Assert on behaviour: the `CalcInfo` and staged files from
  `prepare_for_submission(SandboxFolder())`; `is_finished_ok` and `exit_status`
  of WorkChains run with `aiida.engine.run_get_node` against a fake Code; and,
  for ingestion, the dataset content (shapes, dtypes, `P`/`V_scan` channels,
  grid params, laser metadata). A green job with an empty dataset is a failure.
- `tests/test_web_app.py` covers HTTP contracts only. Any change to routes,
  schemas, service row shapes, `web/public/app.js` or `styles.css` should also
  pass `web/dev/render_check.py` against a live stack, and its screenshots in
  `web/dev/render_out/` should be looked at.
- Keep `README.md` in step with every new CalcJob, WorkChain, Data type, port,
  profile field and CLI command.

## Deploying a change to the daemon

- Adapters under `src/aiida_pino/scripts/` are uploaded with every CalcJob and
  take effect at the next submission. Code reached from a WorkChain step runs
  inside the daemon and takes effect only after a restart;
  `grep -rn <function> src/aiida_pino/scripts/` tells which side it is on.
- Restart with `bin/daemon_restart.sh`. It checks the result with
  `verdi daemon status` (a bare restart can time out while stopping and leave
  no daemon running), starts workers if none came up, and carries the running
  daemon's `AIIDA_PINO_*` and `PINO_*` variables forward; a cold start reads
  `daemon.env`. `--verify` only checks.
- On a compute machine `lpno` is installed from that machine's own checkout,
  which has to be updated separately (compare `git rev-parse --short HEAD`).
- A stalled daemon is diagnosed with `verdi process repair --dry-run`. A process
  whose broker task was lost is not reattached by a restart; `verdi process
  repair`, run with the daemon stopped, requeues it.

## Storage

- Every `PinoSimulationData` is stored in `npz_file` mode (`assumptions.md`
  A11). Ingestion code ends in `from_npz_file`; `from_arrays` is for tests and
  legacy reads. After an import, `aiida-pino compress PK` (a dry run by default)
  must report zero simulations to convert.
- `aiida-pino evict PK` drops a dataset's array payloads and keeps every node,
  attribute and link. It is a dry run by default, one-way, and refused on `core`
  datasets and anything sharing simulations with one. `aiida-pino reclaim`
  returns the disk by repacking only the affected packs; run it with the daemon
  and console stopped and nothing staging or training.
- `verdi storage backup` packs loose objects first, so free local space must
  exceed the loose-object total, and its `pg_dump` must be at least the server
  version or the backup aborts after packing.
- Staged runtime caches live under `~/.cache/aiida-pino/runtime`
  (`AIIDA_PINO_RUNTIME_CACHE` overrides it).
  `python -m aiida_pino.utils.runtime_evict` lists removable entries and
  `--apply` removes them, skipping entries a live process holds.

## Datasets, splits and annotations

- `Flow3DBatchWorkChain.finalize` files a batch into one base dataset named
  from `dataset_profile.batch_keyword`, with the profile's keywords and
  description. A direct `Flow3DGenerateWorkChain` files nothing; follow it with
  `create_base_dataset`, or `extend_base_dataset` for an existing batch.
- Splits are `split` extras on simulations, served as separate subset nodes.
  `aiida-pino submit train-eval` on the parent dataset trains on `train`,
  validates on `val` and scores the model on `test`; given a split node, the
  post-training eval falls back to `val` and says so. Validation data is held
  out from training but drives checkpoint selection, so a val number is a
  health readout. A test score needs `evaluate_on_test=True` or an
  `EvalOnlyWorkChain` on the test node.
- Datasets and models tagged `core` are reference sets: no deletion (the
  service refuses it), supersession, membership change or re-split of their
  members. New split experiments go in a new derived or custom dataset, and only
  the maintainer grants or removes `core`. Folders (`aiida-pino folder`) are an
  extra and touch no membership.
- Give every node a description and keywords at creation; `aiida-pino describe
  PK "..." --title "..."` and `aiida-pino tag` repair afterwards. Evaluate
  through `pino.evaluate_model`, so metrics attach to the model, and star a
  model worth keeping (`aiida-pino tag PK --add starred`).
- Expensive conditioning fields are computed once per dataset and training grid
  with `aiida-pino derive-fields PK --stride sx,sy,sz`, stored with provenance
  and seeded into the local cache at staging (`--seed-only` re-seeds).

## Training and evaluation

- Every training goes through AiiDA and ends as a `PinoModelData` node. A run
  that died before registration is recovered from its workdir with
  `aiida-pino submit import-model --artifacts-path <dir>`. Warm starts take
  `--init-model PK`.
- `TrainModelWorkChain` runs a 100-step VRAM probe (`run_name` ending in
  `_vramsmoke`) before the real training; the real run is the newer
  `PinoTrainCalculation`. A configuration error fails in the probe with exit
  500 at `harvest_smoke`; read the probe's own `aiida.out` first.
- `training_profile` keys become Hydra overrides and omitted keys take the
  composed config's value. Set `max_steps`, learning rate, decay and batch sizes
  only to differ from the config, taking them from that architecture's config
  file. Match `max_steps` and evaluation frame count across compared runs, and
  use `cuda_device: "auto"`.
- `gpu_vram_mb` sizes itself from recorded peaks when unset. Size `host_ram_mb`
  from the `runtime_max_rss_mb` and `gpu_vram_peak_mb` extras of similar past
  calcjobs; a rejected booking's message names the floor.
- Pass `derivation_profile` whenever the config's layout differs from the
  dataset node; an undeclared layout substitution is refused (exit 305).
- RIGNO does not fit full-resolution random-path data (150x100x40) on a 24 GB
  card; set `PINO_LAZY_STRIDE=2,2,1` in the training profile's `env`, never on
  an already downsampled dataset. Training strided and scoring at full
  resolution is the normal RIGNO configuration; pin `rnode_count` to the trained
  budget when the two differ.
- Report temperature metrics in kelvin: `console/model_metrics.kelvin_factor`
  and `recorded_kelvin_factor` return the factor with its provenance, or `None`.
- Compare evaluations through `utils/comparability.compare_axis_sets` (also
  `service.compare_evaluations` and the Models tab compare button). The axes are
  dataset uuid, split, derivation kind, served grid, eval frame stride and
  training stride; an unknown value never counts as agreement, and maximum
  statistics need equal frame counts (`max_statistics_comparable`).
- `utils/architecture.architecture_keyword_from_names` assigns the architecture
  keyword. Names matching no entry of `ARCHITECTURE_TOKENS` fall back to `fno`,
  so a new model family needs its token added.

## Finding things in the database

- `QueryBuilder` reaches JSON paths at any depth;
  `python -m aiida_pino.utils.queryable_paths simulation --grep <name>` lists
  stored paths with coverage and a query line, and `--live-log PK` finds a run's
  log. `aiida-pino show PK [--json]` prints state, attributes and outputs.
- Performance figures (`graph_nodes`, `ms_per_iter`, `gpu_vram_peak_mb`,
  `host_ram_peak_mb`, `wall_seconds`, calcjob pk and workdir) are recorded on
  the node through `utils/run_metrics.py`.
- Use uuids for durable references; a pk is local to one database and an
  archive import renumbers it. Before reporting an absence or a count, write
  down what the query cannot return.

## Known limitations

Known issues, with symptoms, causes and workarounds, are in
[`../potential_bugs.md`](../potential_bugs.md).

## Withdrawn: k-fold from the console and CLI

`KfoldTrainEvalWorkChain` still runs when submitted directly. Re-enabling the
one-click path needs a test that folds partition the membership with disjoint
test sets, checks of `fold_index = -1`, a fold-membership overlap guard, and a
rule relating folds to `split` extras.

## Roadmap

The end goal is a provenance-tracked loop that generates FLOW-3D simulations,
trains on the ones marked suitable and retrains a shared model as data arrives.
The two-alloy IN718 loop (`flow3D_scripts/batches/in718_loop/`) runs such a
loop for one experiment: it selects cases (half by maximin coverage, half by a
regime bandit fed with per-region model error), submits the FLOW-3D batch
through AiiDA, snapshots the training collection, and retrains and evaluates
through the same workchains. Its state machine, `bin/tick.py`, runs the stages
unattended or with a person approving each solved round
(`flow3D_scripts/batches/README.md`). What remains:

- For the IN718 loop: automatic replacement of dropped or failed cases, a wait
  on licence-server outages, a prepare-only render before each batch, a
  notification when it holds or stops, and its node ids in a configuration file.

- Setting the `split` label automatically at store time from cheap checks
  (laser scalars present, conditioning channels non-zero, fields finite), as the
  IN718 loop's `collect` already does from its boundary check.
- Automatic retraining of a designated generic model when suitable data
  appears, with the retrain policy and the registry marker decided.
- Collection-level parameter summaries, a coverage map, and gap ranking from
  sampling density and recorded evaluation error. Gap-driven solves stay behind
  an explicit approval and budget, since each one uses a licence.
- A coarse-to-fine series of meshes, preceded by a grid-convergence study (Roache GCI)
  on scalar melt-pool quantities and a pilot that checks whether coarse runs
  carry signal in each melting regime.
- Queryable evaluation scores, a canonical evaluation per (model, dataset), a
  leaderboard that lists untested models, and `aiida-pino compare` over a stored
  comparability signature.
- A VRAM footprint key over the resolved architecture that ignores seed
  overrides, an `evaluation_profile` key for the best checkpoint, a CUDA probe
  before remote submission, and one home for `dataset_role` and
  `composition_kind`.
