# Web demo: developer notes

Notes for running and modifying the demo on your own machine: a FastAPI backend
(`lpno_web_demo_backend`) that runs the neural operators and returns rendered
PNGs, and a static vanilla-JS frontend (`lpno_web_demo_frontend`). The READMEs
cover installation and routes; each half's `assumptions.md` records the
deliberate modelling and rendering choices under stable IDs.

## Assets

Each page reads its models from `data/<page>/models/<id>/`, and any folder there
with a checkpoint or a `meta.json` appears in that page's dropdown. The
architecture (FNO, GAOT, RIGNO) is read from the folder's Hydra config, so the
folder id is free text. `data/README.md` gives the exact layout.

A model folder holds:

- `u_network.0.pth`, the trained weights, and `.hydra/config.yaml`, the training
  config the loader is built from. Both come from the open Zenodo download.
- `meta.json`, tracked in git: `label` (dropdown text), `type` (informational),
  `order`, `default` (exactly one `true` per page), `model_name`, and a `physics`
  block.
- `flow3D_params.yaml` for the steady models (the served geometry and reference
  scales), and optionally `grid.npz` (x, y, z axes in metres) when a model was
  trained on a crop of the dataset grid.

The `physics` block is what the adapters use to build the model inputs:
`p_watts` and `v_scan_mps` (`min`, `max`, `default`), `h_norm_denom` (the
training split's maximum of P/sqrt|V|), `power_weighted_dose`, and for the
free-form models `duration_scaled_times`, `dt_s` (training time step),
`start_anywhere`, `max_path_length_m` (`null` means no trim) and `presets_dir`.
A wrong `h_norm_denom` or a missing `power_weighted_dose` produces wrong inputs
with no error, so copy them from the training run.

Ground truth (the FLOW-3D fields behind the comparison panels, the catalog
points and the free-form presets) is not part of the open download. It is
distributed only through a restricted-access record, under the FLOW-3D data
terms. Without it, the comparison controls are disabled, every page shows
predictions only, and the time pages take their grid from `data/<page>/grid.npz`.

To add or rename a model, also update the `<option>` fallbacks in the frontend's
`index.html`, `line.html` and `path.html`, used when `GET /models` fails.

## API conventions

- Responses carry rendered PNGs as base64 inside JSON, and no raw field data.
- Every request schema forbids unknown fields (`extra="forbid"` in
  `app/schemas.py`), so a misspelt key is a 422.
- Errors share one envelope: `status: "error"`, `error.code`, `error.message`,
  `error.retryable`, and `server_request_id`. Codes: `validation_error` (422),
  `busy` (429), `timeout` (504), `service_unavailable` (503), `internal` (500).
  Messages never contain filesystem paths or state-dict keys.
- The server handles one request at a time: run exactly one uvicorn worker. A
  request that arrives while another runs gets 429, and the pages retry. A 504
  releases the lock, but the GPU work it started runs to completion.
- POST bodies are limited to 16 KiB, or 128 KiB on `/infer_path*`, which
  carries drawn polylines. A larger body is 413; a body without
  `Content-Length` on these routes is 411.
- Out-of-range P and V are clamped and reported (`clamped_inputs`, or
  `path_meta.physics_clamped`). Explicit P/V for a free-form preset is a 422.
- `public/config.json` keeps `backendBaseUrl: ""` (same origin). To point the
  pages at a backend on another port or host, write the untracked
  `public/config.local.json`; the backend's `dev/run_local.sh` does this, and
  CORS then needs `LPFNO_PAGES_ORIGIN` set to the page's origin.

## Tests

- Backend: `python -m pytest tests -q`. The suite runs in stub mode on the CPU,
  and tests that read the served assets skip when they are absent. Tests that
  call the service must use `with TestClient(app) as client:`, because the
  service is attached in the app lifespan.
- With the assets installed and the server running,
  `python scripts/smoke.py http://127.0.0.1:8001` calls every route with every
  model in 2D and 3D and exits non-zero on any failure.
- Frontend: `for f in test/*.mjs; do node "$f" || break; done` checks the syntax
  of every script, the `PVMap` export, the same-origin `config.json`, and the
  one-request-in-flight behaviour of the two time pages.
- After changing a page, render every combination of its toggles (2D/3D, top
  view/cross-section, each model, each field) and look at the images; a control
  that cannot work in some state should be disabled there.

## Rendering rules

- Field panels keep the true physical aspect ratio. Figure height follows from
  width, aspect and fixed margins in inches, and all text stays inside the
  canvas at dpi 100.
- 2D slices show the raw served field, upsampled three times
  (`scipy.ndimage.zoom`, order 1) and drawn with `interpolation="nearest"`;
  bilinear interpolation smears the keyhole edge. An orange contour at 1898 K,
  the midpoint of the Ti-6Al-4V solidus and liquidus, marks the melt-pool edge
  on every 2D plot.
- Colors: inferno for T with limits 300-3200 K, 0-1 for `fl` and `alpha`,
  coolwarm with symmetric limits for "Prediction - GT", a cyan path line and a
  lime open-circle beam marker.
- A colorbar is fixed across all frames of a run (backend A15). The 3D block
  uses one run-wide range spanning both ground truth and prediction (A13).
- `PathFrameRenderer` updates its artists per frame; keep new per-frame state
  inside that pattern.
- The pages request 640 px wide images and show them at intrinsic size, capped
  at the container width, so plots are never stretched or cropped.

## Known issues

Known issues and limitations, with workarounds, are in each half's `potential_bugs.md`.

## Hardware

Times measured with these adapters, 8 CPU threads, `LPFNO_CUDA_DEVICE=cpu`:

| page | FNO | GAOT | RIGNO |
|---|---|---|---|
| steady, one request | 0.57 s | 0.47 s | 1.76 s |
| straight or free-form, 20 frames | 14.5 s | 16.2 s | 81 s |

On a GPU a 20-frame request takes a few seconds, mostly CPU-side channel
building and rendering. The time pages time out at 60 s, so a CPU-only machine
serves the steady page well, FNO and GAOT on the time pages in about 15 s, and
RIGNO there not at all. CPU-only use needs `torch_scatter` and `physicsnemo-sym`
CPU wheels, and PyVista rendering off screen (OSMesa or EGL).

On a GPU the device is chosen at start-up (`LPFNO_CUDA_DEVICE=auto` takes the
card with the most free memory) and models do not move afterwards. If another
job later fills that card, restart the backend. A forward pass that runs out of
memory drops the cached models and retries once; `POST /cache/clear` frees the
card by hand. Keep `LPFNO_PATH_INFER_BATCH=1` on a shared GPU (backend A10).
