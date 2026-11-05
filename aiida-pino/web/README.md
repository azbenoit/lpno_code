# PINO Pipeline Console (web)

The default UI for the `aiida-pino` pipeline: browse datasets and models,
compose datasets, monitor processes and read logs. It is a static HTML/JS front
end talking to a small FastAPI backend, reachable over the host's IP. The
architecture mirrors the web demo backend (`self_hosted_web_demo/lpno_web_demo_backend`) (FastAPI app-factory + one
uvicorn worker + a no-build static front end + `dev/run_local.sh`).


## Layout

```
web/
  app/            FastAPI backend
    main.py         create_app() factory, routes, typed-error -> JSON envelope
    aiida_service.py, config.py, schemas.py, security.py, errors.py, logging_utils.py
                    4-7 line shims re-exporting src/aiida_pino/console/ (the service
                    layer, Settings with env prefix AIIDA_PINO_WEB_, request bodies,
                    X-Request-ID + body-size limits, typed exceptions) so tests and
                    the app import one path; the code lives in the package
  public/         static front end (index.html, app.js, styles.css, config*.json)
  dev/            run_local.sh (start/stop/status), serve_frontend.py (no-cache),
                  render_check.py (headless render check)
```


## Walkthroughs and setup

`docs/SETUP.md` is the localhost setup for a fresh clone (prerequisites, every
`AIIDA_PINO_WEB_*` variable, starting and stopping, the health check, the read-only
render check). Each public tab has its own page written against the live page:
`docs/monitor.md`, `docs/environment.md`, `docs/datasets.md`, `docs/compose.md`,
`docs/models.md`, `docs/runs.md`, `docs/logs.md`. A page covers what the tab shows,
every control, what it reads or writes, and the empty state on a fresh profile.

## Execution model: read this first

The console reads whichever AiiDA profile `AIIDA_PINO_WEB_AIIDA_PROFILE`
names, or the configured default when it is unset. Inspection works against any
profile; submission additionally needs a broker (RabbitMQ) and a running daemon,
so the split below is what you hit on a brokerless profile, or when the daemon is
stopped for maintenance:

- **Browse and monitor are broker-free.** Listing datasets/models/processes,
  reading metadata/metrics/logs and dry-run delete previews are plain ORM
  reads and work whenever a profile is loaded.
- **Submissions require a running daemon.** The aiida-pino workchains submit
  their children through the process controller, so a real run needs the
  RabbitMQ broker up and `verdi daemon start`. When the daemon is not
  reachable, every `POST /submit/*` (and `kill`) returns 503
  `daemon_unavailable` with a clear reason, and the console disables the
  submit buttons and shows a banner. The command line can run them without a
  daemon or a broker (`aiida-pino --foreground submit ...`).

Inspection therefore works at any time. Launching a workflow needs the broker
and daemon up; check with `verdi daemon status`, and start your database and
broker services if either is down.

### What the daemon is, and starting it

The daemon is the background worker that *runs* a submitted workflow: it pops
the process off the queue, drives its state machine, checkpoints it, and resumes
it after a reboot. Without it, a `submit()` is stored in the DB but never
advances. It reaches running processes through RabbitMQ, so a broker must be up
first. `Daemon: Down` in the console, like a `Connect call failed
('127.0.0.1', 5672)` error, means no broker, so no daemon.

If no `rabbitmq-server` is installed, the lightest path is a container:

```bash
# 1. Broker (profile points at the default guest:guest@127.0.0.1:5672).
#    3.8.14 is the newest version aiida-core supports (its check is
#    3.6.0 <= v < 3.8.15) and needs NO consumer_timeout tuning — that knob
#    only exists to work around 3.8.15+. First time only:
docker run -d --name aiida-rabbitmq -p 5672:5672 --restart unless-stopped rabbitmq:3.8.14
#    Afterwards: docker start aiida-rabbitmq (if not using the restart policy).

# 2. Daemon. `verdi daemon start` needs verdi on PATH (calling it by absolute
#    path fails with "Unable to find verdi in the path"):
export PATH="/path/to/aiida-env/bin:$PATH"
verdi daemon restart    # or `start`; either clears a stale PID
verdi status            # broker + daemon should both be OK, no version warning

# Undo later: docker stop aiida-rabbitmq && verdi daemon stop
```

`docker` commands may need you in the `docker` group (Podman works the same
way if that is what you have).

**Health.** "Up" in `docker ps` does not check broker liveness; probe with
`docker exec aiida-rabbitmq rabbitmq-diagnostics -q ping`. If the broker is ever
replaced while workchains are active, their queued task messages die with it:
stop the daemon, swap, start, then `verdi process repair` (with the daemon
stopped) to revive orphaned processes. Keep the broker at the supported
RabbitMQ 3.8.14 shown above.

## Run it

```bash
# loopback only
web/dev/run_local.sh start          # http://127.0.0.1:8180/index.html

# reachable from other machines on the LAN (binds 0.0.0.0, advertises the LAN IP)
web/dev/run_local.sh start lan      # http://<host-ip>:8180/index.html
                                    # (127.0.0.1:8180 works too: the loopback
                                    #  aliases of the page port are allowed by CORS)

web/dev/run_local.sh status
web/dev/run_local.sh stop
```

`start lan` writes `public/config.local.json` with the backend origin and pins
the backend's CORS to exactly the page origin, so browse via the host
IP. Backend on `:8100`, front end on `:8180` (override with
`AIIDA_PINO_WEB_PORT` / `AIIDA_PINO_WEB_FRONTEND_PORT`). The Python interpreter
is `AIIDA_PINO_WEB_PYTHON`, else a `pino` conda env under `$HOME` if one exists,
else `python3` on `PATH`; the script exits with a message unless the interpreter
imports `aiida`, `fastapi` and `uvicorn` (the `web` extra). Pick a
named profile with `AIIDA_PINO_WEB_AIIDA_PROFILE`; `AIIDA_PINO_WEB_LAN_IP`
overrides the advertised address in `lan` mode.

Dependencies (`fastapi`, `uvicorn`, `pydantic-settings`, `matplotlib`, `scipy`)
are the `web` optional-dependency group in `pyproject.toml`:
`pip install -e '.[web]'`.

## Endpoints

Generated from `create_app().routes`. Browse / monitor (broker-free):

- `GET /healthz`, `GET /env`, `GET /codes`
- `GET /datasets`, `GET /datasets/options` (the dropdown list), `GET /datasets/{pk}`,
  `GET /datasets/{pk}/cases` (per-case browser)
- `GET /simulations`, `GET /simulations/{pk}/frames`, `GET /simulations/{pk}/plot`
  (PNG slice frame: `field`, `view=surface|long|cross`, `frame`, `contour`),
  `GET /simulations/{pk}/stats`
- `GET /datasets/{pk}/stats`, `POST /datasets/{pk}/stats/compute` (batched),
  `GET /datasets/{pk}/stats/plot?kind=regime|depth_vs_width|melt_hist|pv_map`
- `GET /flow3d/queue` (seat-lock status, the `~/.flow3d/spool` queue, and the
  AiiDA-side FLOW-3D runs/batches under `aiida`; both entry points take the same
  seat, so a spool-only view would report an idle machine while AiiDA saturates it)
- `GET /models`, `GET /models/{pk}`, `GET /models/{pk}/compare/{other}`,
  `GET /models/{pk}/curves/plot`
- `GET /evaluations/{pk}/preview`, `GET /evaluations/{pk}/preview/plot`
  (truth / prediction / error for the one stored case)
- `GET /processes` (`state=all|queued|finished|failed`, applied in the query), `GET /processes/{pk}`, `GET /processes/{pk}/logs`, `GET /processes/{pk}/tail`
- `GET /monitor`
- `GET /folders`, `POST /folders/move`, `POST /folders/rename`, `POST /folders/remove`
- `POST /nodes/{pk}/annotations` (keywords, description, `title`)
- `POST /preview/collection`
- `POST /datasets/{pk}/evaluation_only` (reserve a dataset for evaluation: every
  member becomes `split=test`, and training then refuses it)
- `POST /datasets/{pk}/assign_splits` (random train/val/test assignment with
  fractions summing to 1; `only_unassigned` defaults true, deterministic per
  `seed`. Records the recipe (method, seed, fractions, resulting counts and a
  digest of the member list it drew from) as the dataset's `split_recipe`
  extra, and returns it as `recipe`)
- `POST /collections` (create a native collection), `POST /nodes/delete` (dry-run or confirmed)
- `POST /datasets/{pk}/evict` (metadata-only eviction: delete the array data,
  keep every attribute and provenance link; `dry_run` defaults true and reports
  members + reclaimable bytes, `reason` is recorded on each simulation's
  `data_evicted` extra). Refuses `core` datasets and any dataset whose members
  are in a core set. The call itself frees no disk: run `aiida-pino reclaim`
  afterwards, which repacks only the packs that held the evicted bytes
  (`verdi storage maintain --full` would rewrite every pack).

Submissions and kills (503 when the daemon is down):

- `POST /submit/derive_dataset`
- `POST /processes/{pk}/kill`


Every workflow profile (`training_profile`, `derivation_profile`,
`evaluation_profile`, `dataset_profile`, `simulation_profile`, …) is a free-form
JSON dict validated server-side by the same `aiida_pino.utils.validate_*`
functions the workflows use, so an invalid combination returns 422 before
anything runs. No `license` key is ever accepted or forwarded (FLOW-3D licensing
stays in the runner env).

## Front end

`public/app.js` is dependency-free vanilla JS (no build step). It reads
`config.json` (+ the machine-written `config.local.json`) for the backend
origin, then renders tabs: Monitor, Environment, Datasets, Compose new dataset,
Models, Runs, Logs. The active tab is in the URL hash (`#datasets`), so views
are linkable.


The Runs tab shows the FLOW-3D seat-lock and spool-queue status line, and
filters by process type and by state (queued/running, finished, failed); this is
where to watch queued FLOW-3D solves and training/eval runs.

Dataset composition and splits:

- "Compose new dataset" (next to Datasets) creates collections either by
  combining whole datasets (multi-select; members unioned and deduped) or by
  filtering/picking individual simulations. Console-composed collections carry
  `dataset_role: "composed"` and list at the top level with
  a `composed` badge; the default Role filter (`all`) shows every role. The
  source datasets are recorded (`composed_from` extras) and fold out under the
  composed row like derived sets do. A source used by several compositions
  appears under each of them (display clones; the member data is shared by
  reference, never duplicated) while still listing top-level itself. The
  detail pane shows them as clickable `Composed from` chips.
- The dataset detail pane has "Assign train/val/test splits…": random
  proportioned assignment over the members (defaults 0.8/0.1/0.1, only
  unassigned members unless unchecked, deterministic seed). CLI:
  `aiida-pino collection assign-splits PK --train 0.8 --val 0.1 --test 0.1`
  and `aiida-pino collection create NAME --dataset-pk A --dataset-pk B`.
- Runs filters by process type (data generation / import & conversion / model
  training & eval / derivation) and state (queued-running / finished /
  failed); Logs has a recent-run picker, and "Open in Logs →" carries the pk
  over.
- Runs carry an optional short **title** and free-text notes: "✎ title" in the
  run detail sets the node's native AiiDA `label` (`POST /nodes/{pk}/annotations`
  with `title`), the notes pencil sets `description`. A titled run shows the
  title bold in the list with the process class beneath it, which makes a list
  of otherwise identical `FLOW-3D run + ingest` rows readable.
- The **Monitor** tab (`GET /monitor`, re-polled 5 s after each answer; it
  prints the interval it actually achieves, because the endpoint can take
  10-28 s under load and a fixed interval would stack requests on the one worker)
  shows per-GPU utilisation/memory meters with the compute processes on each
  card (mapped back to their AiiDA run via `/proc/<pid>` cwd ↔ calcjob
  workdir); CPU load and RAM; active AiiDA runs, where training runs get a live
  progress bar parsed from the calcjob live-tail (`[step: N]` against the
  profile's `max_steps`, or a "waiting for GPU seat" badge); the pino gpu_spool
  queue per GPU; and the FLOW-3D spool queue grouped by batch (queued count,
  held `queue_*` dirs, and the seat lock).

The model detail pane covers what the model was trained on, how training went,
what it scores, and how it compares with ground truth.

- Temperature errors are shown in Kelvin, converted with `ref_scales.T` read
  from that model's own archived `.hydra/config.yaml` (3000 K on current
  models). Ratios (`rel_*`, `l2_*`) are never converted. IoU appears as an error
  percentage, `100 * (1 - IoU)`, with its spread in points. If the scale cannot
  be established the pane shows normalised numbers and says why, since a
  guessed Kelvin value would mislead.
- **best T MAE / best T RMSE** chips are computed per evaluation dataset on
  every read, in Kelvin, so a beaten model loses its chip on the next render.
  They are never written to `extras.keywords`: they travel in `auto_keywords`
  and render as a distinct chip with no delete affordance, so machine strings
  and user keywords cannot mix. A model whose temperature scale is unknown is
  excluded from the ranking, since its numbers may not be in K.
- **Curves**: validation MAE/RMSE from the training artifacts' tfevents file
  (read as TensorBoard tensors; `Scalars()` is empty for these tags) and the dense loss
  series from the job's stdout `[step: N] loss:` lines, parsed once and cached
  in a `curves_v1` extra. Two panels, Kelvin left and dimensionless right;
  sharing one axis with `Train/loss_*` at 1e7 would flatten the validation curves.
- **Against ground truth**: truth, prediction and |error| for one field and
  z-slice, from `lazy_preview_batch0.npz` on the evaluation's artifacts. It is
  one case, as the caption says, because the evaluation stores only that one.
- Melt-pool and keyhole geometry (true / pred / abs err / % of true) is a
  table; all 54 raw normalised metrics sit behind a dropdown. A keyhole vs
  conduction split of the metrics is unavailable: evaluations record set-wide
  aggregates only.

Presentation rules:

- Datasets show server-side human names (`display_name`, mapped from the data
  folder names), with the raw folder/pk as a subtitle. Lineage comes from a
  folder map in the service (`_FOLDER_LINEAGE`; the stored `dataset_role`
  attribute is immutable and reads `base` on every imported collection): steady /
  resampled / downsampled collections are `derived` and quick-val slices are
  `subset`, each row carrying `parent_pk`. The Role filter defaults to `all`
  and offers the four real roles (`base`, `composed`, `derived`, `subset`);
  picking one flattens the list to that role. A ▸ toggle on a
  row folds its derived sets and its components in and out (member-set strict
  inclusion remains the fallback for unmapped subsets), and each dataset is
  rendered once even when two listed parents claim it. The quick-val sets are
  small fixed validation slices that a training config can name as
  `val_data_dir` for the cheap in-training validation pass. The detail pane
  shows `Derived from` / `Derived sets` as
  clickable chips that swap the pane to the linked dataset; they are resolved
  over the whole active population, so they work for a dataset of any age. Split membership is
  shown as colored train/val/test/unassigned count chips on the dataset itself;
  splits get no rows of their own.
- **Folders** are the top-level structure of the Datasets and Models tabs. A
  folder is a `folder` extra on the node (`console/folders.py`) holding a
  slash-separated path; a node is in at most one, and the storage is the same as
  for keywords and splits. AiiDA groups would read better
  (`verdi group list` would see them), but creating a group from the console's
  request threadpool raises
  `InvalidRequestError: Object '<DbUser>' is already attached to session N`
  (AiiDA caches the default user against one thread's session). A folder exists exactly as long
  as something is in it, so there is no create or delete step: typing a name
  that does not exist creates it, and filing the last node out makes it
  vanish. Folders sort alphabetically above the unfiled rows, and two
  auto-folders need no filing: single-simulation
  import artifacts land in **Single-run imports** (rendered last, collapsed and
  paged at 100 rows, so hundreds of them stop
  crowding the list) and anything keyworded `self-improve` in
  **Self-improvement (AL)**, which keeps new active-learning rounds tidy without
  maintenance. Explicit filing always wins. Move a dataset from its detail pane
  (Folder row) or with `aiida-pino folder "<path>" <pks...>`. A dataset that is
  also a lineage child shows in both places.
  Models are filed explicitly only: an auto-folder there would be collapsed by
  default and would hide starred models inside it.
- **Lineage.** A derived dataset nests under the dataset it came from, inside
  that dataset's folder (folder → base → derived, each a ▸ toggle). The parent
  comes from the collection's recorded `parent_collection_uuid` when it has one,
  then the legacy `_FOLDER_LINEAGE` name map, then a member-subset guess; the
  row carries `parent_source` so the basis is always visible. A derived set
  follows its base between folders by inheritance (nothing is written on
  the child, so moving or unfiling the base takes it along) unless the child is
  filed explicitly, which wins. Only recorded lineage inherits: the subset guess
  is a display hint, and if it moved datasets, every single-case import would
  land in whichever curated folder happened to contain it. A composed dataset can
  itself have a derived child; both links survive, and a row that is both a
  child and filed elsewhere shows in both places.
- **Filing several at once.** Every row carries a checkbox in the first column,
  and so does every folder header (ticking one selects that whole folder; the
  "keyword: …" virtual header does the same, which turns a keyword filter into
  a bulk move). Ticking anything reveals a bar under the filters: a
  count, one text box with a `<datalist>` of existing folders, **Move**,
  **Unfile** and **Clear**. The checkbox stops the click, so a plain row click
  still opens the detail pane and a header click still collapses its folder.
  The selection survives filter changes (the count then says
  `5 selected (2 hidden by filter)`), and
  after a move the bar stays as a message strip carrying **Undo**, which puts
  every dataset back where it was (a previous auto-folder maps back to
  unfiled, never to a literal folder of that name). Names of computed folders
  (`Single-run imports`, `Self-improvement (AL)`, `keyword: …`) are refused client-side
  before any request; a real folder of the same name would render identically
  and behave differently. Renaming is the same gesture: tick a folder header,
  type the new name, Move. From the CLI, `aiida-pino folder-rename` and
  `folder-remove` do it in one call.
  The list loads a page of 500 by default and says so ("N datasets in view · M
  loaded · T in the store"); "Show all" and any search cover the whole store,
  and a `core` dataset is always pinned onto the first page. Filters always run
  before the cap is applied, so a filtered view never loses datasets that lie
  beyond the first page.
- **Group by keyword** folds every dataset carrying one keyword into a single
  collapsible header row (its sims and split chips are the group totals), so a
  campaign that mints many small collections (the active-learning loop's
  per-round sets, for instance) takes one line in place of twenty. The dropdown
  lists keywords shared by at least two datasets; grouping is a view, nothing is
  moved or duplicated, and tagging datasets with a common keyword is how you
  create a group.
- Models carry a **star**: the leftmost column of the Models tab toggles it, the
  detail pane has one next to the title, starred models sort to the top, and a
  "Starred only" filter narrows the list. A star is nothing but the `starred`
  keyword, so it round-trips through the same annotations endpoint as every
  other keyword and is reachable headless with `aiida-pino tag <pk> --add
  starred` / `aiida-pino models --keyword starred`. Convention: star a model
  worth keeping (a baseline, the model behind a figure or the demo).
- Selecting a dataset opens the detail pane (sticky, own scrollbar): editable
  description + keywords, split composition, then a **Cases** table (chunked at
  100 rows). Selecting a case shows its metadata, its melt-pool statistics and a field
  viewer.
- Long text (descriptions, labels) renders in two-line clamped cells; click to
  expand. Raw keys are relabeled through one `KEY_LABELS` map, values get units
  (`150 W`, `0.8 m/s`, `50 µm`), and process class names map to plain English
  (`RawFlow3DImportWorkChain` → "FLOW-3D NPZ import"); raw strings stay
  reachable via tooltips and the "Full metadata" JSON collapse.
- Workflow profiles are edited in a typed per-key editor (dropdowns for enums
  and booleans, number inputs, add/remove keys); "Edit as raw JSON" stays as a
  collapsible escape hatch on every profile.


## Case viewer, melt-pool stats, model registry

- Selecting a case opens an animated slice viewer (server-rendered PNG frames,
  LP-FNO-style: inferno, stable color range, melt contour; a surface view, the
  default because a fixed cross-section plane is ill-defined for random paths,
  plus longitudinal / transverse views, play/slider) and a melt-pool stats
  block (regime badge, melt volume/depth, cavity geometry incl. ellipse-fit
  min/max radii, peak T, normalized enthalpy H). Cavity/melt masks follow
  `materials_dataset_exploration.ipynb::case_metrics`; since stats v4 the
  regime rule is path-agnostic and time-averaged: an ellipse is fitted to the
  laser-local cavity's surface footprint (the connected component around its
  deepest point) each frame, and a case is **keyhole when the mean
  depth/r_min over frames exceeds 1 *and* the cavity reaches at least 2.5 beam
  radii deep**; below 2 cells of depth it is LoF, otherwise conduction. The
  aspect criterion alone labels many shallow straight-path cases keyhole; the
  depth gate brings the labels in line with the surface plots.
  Every simulation also carries its regime as a keyword (`keyhole` /
  `conduction` / `lof`, mutually exclusive, other keywords untouched), so the
  Compose tab can filter on it. Stats are cached in the simulation extras
  (`meltpool_stats`, versioned: a version bump makes old caches count as
  missing until recomputed via the button or `dataset-stats --compute`).
  `GET /simulations/{pk}/stats` reads that cache only and answers
  `computed: false` when it is empty; computing reads every frame of the
  case (197 s for a 331-frame one under load), so it happens on
  `?compute=1`, which is what the case pane's "Compute now" button sends.
- Dataset detail has a statistics section: regime chips, aggregate table, a
  batched "Compute missing" progress button, and the notebook summary plots
  (regime mix, depth-vs-width keyhole line, melt-volume histogram, P–V map).
- The Models tab is a registry with instant client-side filters: architecture
  (derived family: FNO / PINO, RIGNO, U-Net, …), trained-on dataset,
  evaluated-on set, metric + threshold. Rows show the best metric with its
  eval set; the detail pane lists evaluations with their key metrics. Models
  restored from an earlier database carry their original training profile and
  a remapped training source (the original under
  `model_metadata.original_training_source`).
- **Compare 2… (the ablation referee)**: tick exactly two models and press
  Compare. The detail pane shows a verdict banner (CLEAN / SINGLE-FACTOR /
  CONFOUNDED), every difference classified confound / caution / benign with the
  bookkeeping folded away, member-level training-data overlap, shared
  evaluations with their comparability problems (scored-on-training-data,
  mismatched eval resolution), and a warning when any fact in play was
  backfilled after creation. "A beat B" is only evidence
  about the changed thing when the verdict is SINGLE-FACTOR.
  API: `GET /models/{pk}/compare/{other}`; CLI: `aiida-pino compare-models`.

## Not yet available

- Live streaming log tail: the console offers a one-shot
  `GET /processes/{pk}/tail` and no push stream.
- Per-field interdependent form widgets (client-side constraints such as
  `time_as_dim⇒include_time`): here the server-side
  validators are the guardrail and the JSON block is the escape hatch.

## Render check (required after GUI-affecting changes)

`render_check.py` needs Playwright, which is declared in no extra:

```bash
pip install playwright && playwright install chromium   # once
web/dev/run_local.sh start
python web/dev/render_check.py                          # PASS/FAIL + screenshots
```

Headless Chromium walks every tab, opens dataset/case/model/run details,
exercises clamp expansion, keyword chips, and the profile editors, and fails on
JS console errors or horizontal page overflow (also at 1280 px and 900 px
widths). Screenshots go to `web/dev/render_out/`; review them, since the script
cannot judge aesthetics. Any change to routes, schemas, service row shapes,
`web/public/app.js` or `styles.css` should pass this check before it is
committed (`docs/developer_notes.md`).

## Tests

```bash
python -m pytest tests/test_web_app.py tests/test_web_service.py tests/test_web_assets.py -q
```

`test_web_app.py` covers the app contract with an injected stub service (no
profile): routing, the error envelope, CORS pinning, body-size limits, schema
validation. `test_web_service.py` runs the service against the AiiDA pytest
profile with seeded nodes: browse shapes, node-not-found, and that every
submission is gated on the daemon (it raises and never blocks).
`test_web_assets.py` checks the static front end without a browser: the
scripts exist, parse, and reference no undeclared names.

