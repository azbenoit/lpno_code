# LP-NO web demo: backend

The inference server of the self-hosted web demo. It runs the paper's
neural-operator surrogates (FNO, GAOT, RIGNO) for the melt pool of a laser
scanning Ti-6Al-4V, trained on FLOW-3D WELD simulations, and returns the
predictions as rendered PNGs over HTTP. The static frontend is in
`../lpno_web_demo_frontend`. Installation and the quick start are in
[`../README.md`](../README.md); conventions and tests are in
[`../docs/developer_notes.md`](../docs/developer_notes.md), known issues in
[`potential_bugs.md`](potential_bugs.md).

## Pages and models

| page | what it predicts | grid | model folders |
|---|---|---|---|
| Steady (P–V) | time-averaged melt pool in the beam's co-moving frame at power P, speed V | 61×40×40 window | `fno_cal`, `gaot_cal`, `rigno_cal` |
| Straight track (time) | 20 frames along a 0.6 mm straight track at free P, V | 90×40×40 (RIGNO 90×40×30) | `fno_cal`, `gaot_cal`, `rigno_cal` |
| Free-form path (time) | frames along a drawn or held-out path at free P, V | 75×50×40 (training grid) | `fno_newphys`, `gaot_newphys`, `rigno_newphys` |

Each served model is one of the runs reported in the paper; the `model_name` in
its `meta.json` names it in the open record's AiiDA archive. With the optional
ground truth, the steady and straight pages share one catalog of 27 held-out
simulated cases (14 validation, 13 test) and the free-form page offers the 22
held-out test paths. Clicking a catalog case shows the FLOW-3D field next to
the prediction, with the substrate MAE (every z ≤ 0 slice, the paper's
substrate metric) for that case.

The free-form page serves the models' 75×50×40 training grid. The paper scores
them at 150×100×40, which costs about four times more (RIGNO: 16 s and
10.5 GiB per run), so the page does not use it.

## Assets

Model weights and ground truth are not in git; `scripts/assets.py` installs
them under `data/` (layout in [`data/README.md`](data/README.md)):

- `fetch zenodo`: the model weights (required) from the open record,
  doi:10.5281/zenodo.23325086;
- `fetch zenodo-ground-truth --token-file <file>`: the ground truth (optional)
  from the restricted-access record, doi:10.5281/zenodo.23325088, once access
  has been granted on its Zenodo page. These FLOW-3D outputs are distributed
  through that record only;
- `fetch <path-or-URL>`: a local or other copy of either bundle;
- `verify`: lists every served model and ground-truth folder and fails if a
  required file is missing.

Without the ground truth, every page still predicts, the time pages take their
grid from `data/<page>/grid.npz`, and the comparison controls are greyed out.
Building the bundles is for the authors and is described in `data/README.md`.

## Configuration

Settings are pydantic-settings fields in `app/config.py`, overridable by
environment variables prefixed `LPFNO_` or a `.env` file (see `.env.example`).
The ones you may want:

| variable | default | meaning |
|---|---|---|
| `LPFNO_CUDA_DEVICE` | `auto` | `auto` picks the GPU with the most free memory; `0`/`1` pin a card; `cpu` |
| `LPFNO_PAGES_ORIGIN` | `""` | exact frontend origin to allow for CORS when the frontend is not same-origin |
| `LPFNO_MODEL_IDLE_TTL_S` | `600` | drop cached models after this idle time so the GPU is freed |
| `LPFNO_MIN_FREE_VRAM_GIB` | `6` | evict cached models before a request if less than this is free |
| `LPFNO_USE_STUB_RUNTIME` | `false` | serve fabricated fields through the real routes, without any assets |

Run exactly one uvicorn worker: the one-request-at-a-time lock is in-process.

## Repository layout

```
app/
  main.py               FastAPI routes (/healthz, /models, /gt_catalog, /infer*, /cache/clear)
  service.py            request handling, caches, GPU headroom, rendering dispatch
  adapter_pino.py       steady-state models (per-case moving-frame window)
  adapter_randompath.py free-form path models and the shared time-dependent machinery
  adapter_straightpath.py straight-track models (subclasses the random-path adapter)
  gt.py, steady_gt.py   ground-truth loading, masking and P–V matching
  render.py             matplotlib 2D panels and PyVista 3D surfaces
  model_registry.py     data/<page>/models/<id>/meta.json discovery
  submodule.py          locating the lpno model code (ext/lpno)
  config.py             settings
data/<page>/models/<id>/    u_network.0.pth, .hydra/config.yaml, meta.json (+ grid.npz, flow3D_params.yaml)
data/<page>/ground_truth*/  ground-truth NPZ (optional; see data/README.md)
ext/lpno                    link to the repository's lpno package (the model code)
scripts/                    setup_env.sh, assets.py, smoke.py, build_*.py (authors: bundle exports)
dev/                        run_local.sh (local stack), serve_frontend.py (no-cache static server)
docker/                     Dockerfile (built from the repository root) and its .dockerignore
tests/                      pytest suite (stub mode; tests that need assets skip without them)
```

## API

All responses are JSON with PNGs as base64; the browser never receives raw fields.

| route | page | purpose |
|---|---|---|
| `GET /healthz` | – | liveness + runtime summary |
| `GET /models?page=steady\|straight\|path` | all | model dropdown, with each model's physics envelope |
| `GET /gt_catalog?dataset=steady\|straight` | steady, straight | the catalog points by split (empty without ground truth) |
| `POST /infer` | steady | one (P, V) → 2D slice or 3D surface PNG |
| `POST /infer/gt_compare` | steady | prediction vs the matched FLOW-3D case, with MAE/RMSE |
| `POST /infer_straight`, `/infer_straight/render3d` | straight | 20 frames at (P, V); GT triple panel on a catalog match |
| `POST /infer_path`, `/infer_path/render3d`, `GET /infer_path/presets` | path | frames along a drawn or preset path |
| `POST /cache/clear` | – | drop every cached model and report free VRAM |

Request/response schemas are in `app/schemas.py` (pydantic, `extra="forbid"`).
A busy server answers `429`; the frontend retries.

### Request shapes

- `POST /infer`: `client_request_id`, `client_seq`, `mode` (`final`), `model`,
  `inputs {p_watts, v_scan_mps}`, `view {field (T|alpha|fl), plot_mode (2d|3d),
  plane, y_index}`, `render {width_px, height_px}`. `POST /infer/gt_compare`:
  `model`, `inputs`, `field`, `plot_mode`, `plane`, `render`. The steady
  prediction is made on the matched case's own beam-following window (a
  per-case x shift), so the comparison measures the model without a window
  offset; `gt_mae`/`gt_rmse` are the substrate error after resampling the GT
  onto the prediction grid. See `scripts/smoke.py` for a working body of each.
- `POST /infer_straight` (+ `/render3d`): `model`, `inputs {p_watts, v_scan_mps}`,
  `view {field, z_index | plane}`, `render`. Always 20 frames; ground truth when
  (P, V) is within 1 W / 0.005 m/s of a catalog case, as a stacked GT /
  prediction / difference panel. Each frame is predicted independently from the
  beam position and dose history (`time_as_batch`); the dose and conduction
  channels are integrated on the 5 µs export step of the simulations.
- `POST /infer_path` (+ `/render3d`): `model`, `path {points_xy_m | preset}`,
  optional `p_watts`, `v_scan_mps`, `frame_dt_s`, `view`, `render`. Drawn paths
  are clamped into the substrate, resampled at ~25 µm, and animated at the
  requested frame step (5–100 frames follow from the weld duration). Presets
  are the held-out test paths, listed by `GET /infer_path/presets?model=` when
  the ground truth is installed.
- Every model folder's `meta.json` carries its physics envelope (P/V ranges and
  defaults, H_norm denominator, dose weighting), which `GET /models` returns and
  the pages use to set their sliders.

## Tests

```bash
python -m pytest tests -q
```

Runs in stub mode (no GPU). Tests that read the served assets skip when
`data/` is not populated, so a fresh clone passes before `assets.py fetch` and
is fully exercised after it. Against a running server,
`python scripts/smoke.py http://127.0.0.1:8001` calls every route with every
model in 2D and 3D.

## Docker

`docker/Dockerfile` builds the backend image from the repository root (commands
in its header and in [`../README.md`](../README.md)). The served assets are not
in the image: mount the populated `data/` at `/app/data` and a writable volume
at `/app/outputs` for the neighbour-graph cache. The frontend image proxies the
backend routes; a new route needs its own `location` block in
`../lpno_web_demo_frontend/docker/nginx.conf.template`.

## Citation

If you use the demo or the models, cite the paper given in the repository
README.
