# Setting up the console on localhost

For a fresh clone and your own AiiDA profile. Paths are relative to the
`aiida-pino` repo root.

## Prerequisites

1. **Python >= 3.10** (`requires-python` in `pyproject.toml`).
2. **Install the package with the web extra**, into the interpreter you will
   run the console from:

   ```bash
   python3 -m pip install -e '.[web]'
   ```

   The `web` extra is `fastapi`, `uvicorn`, `pydantic-settings`, `matplotlib`
   and `scipy`. The last two are imported at call time, so a console without
   them starts fine and then answers 500 on every plot and melt-pool statistic.
3. **An AiiDA profile.** Browsing and monitoring need only a profile:

   ```bash
   verdi presto                 # a throwaway SQLite profile (aiida-core >= 2.6)
   ```

   The package pins `aiida-core>=2.5,<2.9`; `verdi presto` exists from 2.6.
   It probes for a RabbitMQ on localhost and, if one answers, configures the
   profile with that broker; only when none is reachable does it create the
   profile without one (aiida-core 2.8.0 `cmd_presto.py`, `detect_rabbitmq_config`).

   or use an existing profile by name (see `AIIDA_PINO_WEB_AIIDA_PROFILE`
   below). A message broker and `verdi daemon start` are needed only to
   submit workflows; the public build has no submission tabs, so skip them
   unless you want submissions.
4. On `PATH`: `curl` and `ss` (iproute2), both used by the launcher (`lan`
   mode also calls `hostname -I` and `awk`). `git` is optional: it stamps the
   log boundary line and the served commit in `/healthz`; without it the
   boundary says `sha=unknown` and `served_sha` is `null`.

## Environment variables

The backend settings are a `pydantic_settings.BaseSettings` class in
`src/aiida_pino/console/config.py` with
`SettingsConfigDict(env_file=".env", env_prefix="AIIDA_PINO_WEB_", case_sensitive=False)`.
Three consequences:

- Every field below is read from the process environment as
  `AIIDA_PINO_WEB_<FIELD>`, case-insensitive. Unknown variables in the process
  environment are ignored.
- A file named `.env` is read from the current working directory of the
  backend process. `run_local.sh` `cd`s to the repo
  root before starting uvicorn, so there it is the repo-root `.env`; a direct
  `python -m web.app.main` reads whatever `.env` sits where you run it. Values
  in the process environment win over the file.
- `extra` is left at the `BaseSettings` default, `forbid`. Any key in `.env`
  that is not one of the fields below aborts startup with
  `Extra inputs are not permitted`, whether it carries the prefix
  (`AIIDA_PINO_WEB_BOGUS`) or not (`F3D_SPOOL_DIR`, `AIIDA_PINO_WEB_PYTHON`).
  Keep the launcher-only and Monitor variables out of `.env`; export them in
  the shell instead.

| Variable | Default | Meaning |
|---|---|---|
| `AIIDA_PINO_WEB_APP_NAME` | `aiida-pino-console` | FastAPI title. Cosmetic. |
| `AIIDA_PINO_WEB_HOST` | `0.0.0.0` | Bind host when the app is run directly (`python -m web.app.main`). `run_local.sh` passes `--host` itself, so it does not apply there. |
| `AIIDA_PINO_WEB_PORT` | `8100` | Backend (uvicorn) port. |
| `AIIDA_PINO_WEB_UVICORN_WORKERS` | `1` | Must be 1; any other value is rejected. One process holds one loaded profile. |
| `AIIDA_PINO_WEB_LOG_LEVEL` | `INFO` | Level of the Python root logger (`configure_logging` in `src/aiida_pino/console/logging_utils.py` calls `logging.basicConfig` with it), which the JSON event lines the app writes go through. It reaches uvicorn's own `log_level` only when the app is run directly via `python -m web.app.main`; `run_local.sh` starts uvicorn without `--log-level`, so there uvicorn's access and error logs stay at uvicorn's default. |
| `AIIDA_PINO_WEB_STARTUP_LOAD_SERVICE` | `true` | `false` starts the routes without loading a profile; every data route then answers 503. For contract tests only. |
| `AIIDA_PINO_WEB_AIIDA_PROFILE` | `""` | Profile passed to `load_profile()`. Empty loads the AiiDA default profile. |
| `AIIDA_PINO_WEB_PAGES_ORIGIN` | `""` | Comma-separated page origins trusted by CORS; the `127.0.0.1` and `localhost` aliases of each are added automatically. Empty means no CORS middleware (same-origin deployment). `run_local.sh` sets it to the front-end origin. |
| `AIIDA_PINO_WEB_FLOW3D_SCRIPTS_ROOT` | `""` | `flow3D_scripts` checkout pinned into FLOW-3D submissions as `execution_profile.flow3d_scripts_root`. The service resolves it in this order (`_flow3d_runner_root`, `service.py`): a `flow3d_scripts_root` given in the request's execution profile; then this setting; then the sibling `../flow3D_scripts` checkout beside the `aiida-pino` package, if it holds `python/run_rectangular_case.py` (`default_runner_root` in `utils/flow3d_config.py`). Only if all three are empty does the submission leave the field unset and the adapter fall back to the bundled `external/flow3D_scripts` submodule. `run_local.sh` also exports the sibling checkout into this variable when it exists and the variable is unset. The FLOW-3D submit tab (not in the public build) reads it for the config schema with no fallback; the melt-pool statistics (`console/viz.py`) read `python/melt_metrics.py` from it and fall back to the same sibling checkout. |
| `AIIDA_PINO_WEB_LIST_DEFAULT_LIMIT` | `100` | Page size when the request gives no `limit`, for `GET /models`, `GET /simulations`, `GET /processes` and (submit build only) `GET /kfold_runs` (`Settings.clamp_limit`, `config.py`). `GET /datasets` is the exception: with no `limit` it uses `LIST_MAX_LIMIT` (`service.py:1184`). |
| `AIIDA_PINO_WEB_LIST_MAX_LIMIT` | `500` | Largest page size a request can ask for. `limit=0` means no cap. |
| `AIIDA_PINO_WEB_TAIL_MAX_LINES` | `1000` | Cap on `GET /processes/{pk}/tail`. |
| `AIIDA_PINO_WEB_LOGS_MAX_CHARS` | `200000` | `GET /processes/{pk}/logs`: keeps only the last this many characters of the `console` field (CalcJob scheduler stdout/stderr and output file). The `report` list is not capped (`process_logs`, `service.py`). |
| `AIIDA_PINO_WEB_REQUEST_MAX_BYTES` | `65536` (64 KiB) | POST body limit for ordinary requests. |
| `AIIDA_PINO_WEB_SUBMIT_MAX_BYTES` | `2097152` (2 MiB) | POST body limit for submissions. |

Read by the launcher `web/dev/run_local.sh` from its own environment. They are
not fields of `Settings` in `config.py`, so in `.env` they are an error:

| Variable | Default | Meaning |
|---|---|---|
| `AIIDA_PINO_WEB_PYTHON` | the script's hard-coded `$HOME/miniconda3/envs/pino/bin/python` if that is executable, else `python3` on `PATH` | Interpreter for the backend and the static server. The script exits before starting anything if it cannot import `aiida`, `fastapi` and `uvicorn`. |
| `AIIDA_PINO_WEB_FRONTEND_PORT` | `8180` | Static front-end port. |
| `AIIDA_PINO_WEB_PUBLIC` | `web/public` | Directory served as the front end; `config.local.json` is written there. |
| `AIIDA_PINO_WEB_LAN_IP` | first address of `hostname -I` | Address advertised in `lan` mode. Set it when the first address is a VPN or container bridge. |

Read by the Monitor tab straight from `os.environ` (no `AIIDA_PINO_WEB_`
prefix, not `Settings` fields, so again not for `.env`):

| Variable | Default | Meaning |
|---|---|---|
| `F3D_SPOOL_DIR` | `~/.flow3d/spool` | FLOW-3D spool and seat-lock directory (`service.py`, `flow3d_queue_status`). Absent: the queue shows empty. |
| `PINO_GPU_LEASE_DIR` | `/var/tmp/pino_gpu_leases` | GPU seat ledger written by `gpu_spool` (`monitor.py`, `_lease_root`). Absent: no seat rows. |

`nvidia-smi` on `PATH` feeds the GPU meters; without it the tab shows
"nvidia-smi unavailable" and carries on.

## Starting

```bash
export AIIDA_PINO_WEB_PYTHON="$(command -v python3)"   # the interpreter you installed into
export AIIDA_PINO_WEB_AIIDA_PROFILE=<name>             # omit to use the default profile
web/dev/run_local.sh start                              # loopback only
```

`start lan` binds both ports to `0.0.0.0` and advertises the LAN address, so
anyone on the network can reach the backend and the page. Browse via that
address (the loopback aliases are also allowed by CORS).

Set `AIIDA_PINO_WEB_PYTHON` whenever the interpreter is not the script's
default, for example `AIIDA_PINO_WEB_PYTHON=/path/to/venv/bin/python`.

Two ports:

- **8100** (`AIIDA_PINO_WEB_PORT`): the FastAPI backend, one uvicorn worker
  with the profile loaded.
- **8180** (`AIIDA_PINO_WEB_FRONTEND_PORT`): the static front end,
  `http://127.0.0.1:8180/index.html`. Every `start` rewrites
  `web/public/config.local.json` (gitignored) so the page calls the matching
  backend origin.

The script polls the backend for up to 30 s and prints both URLs only once it
answers; otherwise it prints the last 15 log lines and exits 1.

### Health check

```bash
curl http://127.0.0.1:8100/healthz
```

A good answer has `"status": "ok"`, `"service_loaded": true`, your profile
name, a `daemon` block and the served commit:

```json
{"status":"ok","service_loaded":true,"service":"aiida-console","profile":"<name>",
 "daemon":{"available":false,"reason":"..."},"code":{"served_sha":"...","started_utc":"...","started_monotonic":...}}
```

`daemon.available: false` is normal without a broker; it only disables
submissions. `service_loaded: false` means `STARTUP_LOAD_SERVICE=false`; every
data route then answers 503. A profile that fails to load does not show up
here: the load runs inside the app's lifespan (`web/app/main.py`), so the
exception aborts uvicorn's startup, `/healthz` never answers, and `run_local.sh`
prints the tail of `backend.log` and exits 1.

`web/dev/run_local.sh status` runs the same check plus the front-end HTTP code.

### Stopping and the log

```bash
web/dev/run_local.sh stop
```

Sends SIGTERM to the pids in `web/dev/.run_local_pids`, waits for the port and
the processes, escalates to SIGKILL after 5 s (a long in-flight request can
hold uvicorn open), then removes the pid file.

Logs are `web/dev/backend.log` and `web/dev/frontend.log`, both append-only
and gitignored. Each start writes a
`=== console start <date> sha=<commit> mode=<local|lan> ===` line to
`backend.log`, so `grep -n "=== console start" web/dev/backend.log | tail -1`
finds the current run.

## Tabs

The page lands on Monitor. These seven tabs are what the console serves,
and each has its own page under `web/docs/`.

- **Monitor** ([monitor.md](monitor.md)): GPU, CPU and RAM meters, active
  runs with training progress, the GPU and FLOW-3D queues.
- **Environment** ([environment.md](environment.md)): profile, daemon state,
  `lpno` version, registered codes, packaged training configs.
- **Datasets** ([datasets.md](datasets.md)): folder-first browser, lineage,
  splits, case viewer and melt-pool statistics.
- **Compose new dataset** ([compose.md](compose.md)): build a composed dataset
  from whole datasets or picked simulations.
- **Models** ([models.md](models.md)): model registry, training curves,
  scorecards in Kelvin, two-model comparison.
- **Runs** ([runs.md](runs.md)): process list by type and state, titles and
  notes on a run.
- **Logs** ([logs.md](logs.md)): report logs, CalcJob console and remote tail
  for a run.

## Render check

`web/dev/render_check.py` drives headless Chromium through the tabs the page
actually serves: it reads the tab ids off `#tabbar` (`tabs_on_page`) instead
of a fixed list. It fails on JS errors or horizontal
overflow. When the profile has datasets it also asserts on content, in this
order:

- Datasets: folder header rows exist; the auto-folder of single-simulation
  imports (`Single-run imports`, found by its `auto` badge) exists, is
  collapsed by default and opens to member rows; expand-all yields
  nested rows and `Role=all` reaches depth-2 subset rows; the
  group-by select offers a shared keyword and grouping folds datasets under one
  `keyword:` header that expands back; a base inside the
  `Training splits` folder expands to its derived set.
- Models: searching for one hard-coded model name (the `target` variable in
  the script) finds it in the paged list, and its detail shows the four
  sections, a Kelvin and a percent scorecard value, an automatic best-metric
  chip, two images and a metrics dropdown. A profile without that model fails
  here.
- Runs: the detail pane offers the `✎ title` button.
- Monitor: meters render and the `FLOW-3D queue` section exists.
- Narrow viewports (1280 and 900 px) on Datasets, overflow only.

A small profile can therefore fail on those content checks even when it renders
correctly; an empty profile (no dataset rows) screenshots the empty tabs only
and reports SKIP.
Screenshots go to `web/dev/render_out/`. Playwright is not in any extra:

```bash
pip install playwright && playwright install chromium   # once
web/dev/run_local.sh start
python web/dev/render_check.py                          # read-only
```

It is read-only by default: every `prompt()`/`confirm()` is dismissed, and
no flow that writes runs unless asked. `--exercise-writes` enables exactly
three: multi-select filing of datasets and of models into a new folder
(`zz_render_check`, `zz_render_models`) followed by Undo; the star
toggle on the models list, clicked and clicked back; and adding then
removing a `render-check` keyword chip on the first Compose row, two writes to
a real simulation's extras. Each is meant to leave the store as
found, but only if the undo half succeeds, so run it only against a disposable
profile (a `verdi presto` one), never against a store you care about.
