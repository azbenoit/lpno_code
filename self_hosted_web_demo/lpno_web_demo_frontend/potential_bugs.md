# Known issues: frontend

Ways the pages can mislead or fail, with the cause, a workaround and where to
look in the code. IDs are stable because code comments cite them; a missing
number is a retired entry. A-numbers refer to `assumptions.md`. Server-side
issues are in `../lpno_web_demo_backend/potential_bugs.md`.

- **P1. The canvas geometry is duplicated in `path.js`.**
  Symptom: the drawable region or the barrier halo differs from what the
  server accepts, and the server re-clamps the path without telling the user.
  Cause: `path.js` hardcodes the substrate (±0.75 × ±0.5 mm), the keep-out
  inset (0.2 mm), the beam radius (50 µm) and the 1.5 mm trim of models without
  a `physics` block, to match the server (A1). Workaround: when the server's
  values change, edit the constants at the top of `public/path.js`. Code:
  `public/path.js`; the server's values are `KEEPOUT_X`, `KEEPOUT_Y` and
  `BEAM_SIGMA_M` in `../lpno_web_demo_backend/app/adapter_randompath.py`.
- **P2. The self-intersection barrier is advisory.**
  Symptom: a direct API call or a modified page can submit a self-crossing
  path, and the backend returns a confident out-of-distribution prediction with
  no indicator. Cause: the barrier runs only in the browser (A3), and the
  server does not check (backend P2). Code: `distToSegmentM` and the barrier
  test in `public/path.js`.
- **P3. A failed ground-truth request looks like missing ground truth.**
  Symptom: the comparison toggle or the preset list is greyed out with the
  tooltip "Ground truth is not installed on this server", although it is
  installed. Cause: when `GET /gt_catalog` or `GET /infer_path/presets` fails
  (backend down, a proxy without the route, CORS), the pages fall back to an
  empty list and grey the control the same way. The straight and free-form
  pages print the error in the status line ("GT points unavailable",
  "Presets unavailable"); the steady page prints nothing. Workaround: open the
  route directly in the browser or with curl, and check the backend log,
  `LPFNO_PAGES_ORIGIN` and the `location` blocks in
  `docker/nginx.conf.template`. Code: `loadSteadyGtCatalog` in
  `public/app.js`, `loadGtCatalog` in `public/line.js`, `loadPresets` in
  `public/path.js`, `setGtAvailable` in `public/common.js`.
- **P4. A non-uniform canvas resize breaks the halo.**
  Symptom: the barrier halo is too wide or too narrow in one direction. Cause:
  CSS resizing of the canvas is handled, but `width`/`height` attributes that
  change its 1.5:1 aspect (plus margins) give x and y different scales, while
  the halo assumes one scale (A4). Workaround: keep the canvas aspect when
  changing its attributes. Code: the mm-to-pixel transform in `public/path.js`.
- **P6. The committed config must stay same-origin.**
  Symptom: every page of the nginx image calls a backend address that exists
  only on the machine that wrote it, and nothing loads. Cause: a non-empty
  `backendBaseUrl` in `public/config.json` is copied into the image (A5).
  Workaround: keep `""` in `config.json` and put a direct backend URL in the
  untracked `public/config.local.json`. `test/check_js.mjs` fails on a
  non-empty value. Code: `loadConfig` in `public/common.js`.
- **P8. The fallback model options must match the model folders.**
  Symptom: every request returns 422 "Unknown model". Cause: when
  `GET /models` fails, each page falls back to the hardcoded `<option>` values
  of its model dropdown, and a renamed or removed model folder leaves a stale
  id. Workaround: update the options in `public/index.html`, `line.html` and
  `path.html` with every change to the backend's `data/<page>/models/`, with
  the `selected` option matching the folder whose `meta.json` has
  `default: true`. Code: `#modelSelect` in the three pages, `loadModels` in
  `public/common.js`.
