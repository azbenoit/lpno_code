# LP-NO web demo: frontend

The browser side of the self-hosted web demo: static HTML, CSS and vanilla
JavaScript, with no build step and no dependencies. The pages send the controls'
values to the backend (`../lpno_web_demo_backend`) and display the PNGs it
returns. Installation and the quick start are in [`../README.md`](../README.md);
conventions are in [`../docs/developer_notes.md`](../docs/developer_notes.md)
and known issues in [`potential_bugs.md`](potential_bugs.md).

## Pages

- `public/index.html` (`app.js`): steady-state P–V demo (`POST /infer`). Power
  and scan speed come from sliders or from the P-V and H-P enthalpy maps; the
  view is a 2D cross-section or a 3D surface. With the ground truth installed,
  a "Compare to ground truth" toggle (default off) shows the catalog cases
  (`GET /gt_catalog?dataset=steady`) as split-coloured points; clicking one
  snaps to that case and shows the FLOW-3D field beside the prediction
  (`POST /infer/gt_compare`).
- `public/line.html` (`line.js`): straight track in time. The backend predicts
  20 frames at the chosen P and V (`POST /infer_straight`,
  `/infer_straight/render3d`) and the page plays them with a time slider. On a
  ground-truth catalog point it shows a prediction / ground truth / difference
  panel.
- `public/path.html` (`path.js`): free-form path in time. Draw a scan path, set
  P and V within the range the selected model declares in its
  `GET /models?page=path` `physics` block, and the backend predicts frames
  spaced by the chosen frame step (`POST /infer_path`). With the ground truth
  installed, the preset list (`GET /infer_path/presets`) offers the held-out
  test paths with a prediction / ground truth / difference panel.

Without the ground truth, the comparison toggles and the preset list are greyed
out and every page shows predictions only.

Shared browser code: `public/common.js` (`window.Common`: config and model
loading, status and error handling, the frame player, the time pages' request
runner) and `public/pvmap.js` (`window.PVMap`: the P-V / H-P maps, the
ground-truth scatter and the H* formula).

## Runtime config

`public/config.json` sets `backendBaseUrl` to `""`: the pages call the backend
routes on their own origin, which is how the nginx image serves them
(`test/check_js.mjs` asserts it). To call a backend on another origin, write
the untracked `public/config.local.json`, for example
`{"backendBaseUrl": "http://127.0.0.1:8001"}`; every page merges it over
`config.json` at load. The backend's `LPFNO_PAGES_ORIGIN` must then be this
page's origin. The backend's `dev/run_local.sh` writes the file and starts both
halves.

## Local preview

`../lpno_web_demo_backend/dev/run_local.sh start` serves `public/` on :8080
next to a backend on :8001. To serve the pages alone, use the no-cache server:

```bash
python ../lpno_web_demo_backend/dev/serve_frontend.py public 127.0.0.1 8080
```

`python3 -m http.server` works too, but browsers then cache the scripts and a
reload can keep stale JavaScript.

## Tests

```bash
for f in test/*.mjs; do node "$f" || break; done
```

checks the syntax of every script, the `PVMap` export, the same-origin
`config.json`, and the one-request-in-flight behaviour of the two time pages
(the real `line.js` and `path.js` in a mock DOM). Then load the pages against a
running backend.

## Nginx container

```bash
docker build -t lpno-demo-frontend -f docker/Dockerfile .
docker run --rm -p 127.0.0.1:8080:80 \
  -e LPFNO_BACKEND_UPSTREAM=http://<backend-host>:8000 lpno-demo-frontend
```

The container serves `public/` with nginx, proxies the backend routes listed
in `docker/nginx.conf.template` to `LPFNO_BACKEND_UPSTREAM` (default
`http://lpno-demo-backend:8000`, the backend container's name in
[`../README.md`](../README.md), "Docker"), serves `config.json` with
`Cache-Control: no-store`, and answers `GET /frontend-healthz`. An `http://`
upstream needs nothing else. An `https://` upstream is verified, so a backend
certificate from a private certificate authority must be mounted at
`/etc/nginx/upstream-certs/backend.crt`; `LPFNO_BACKEND_SERVER_NAME` sets the
name it is checked against. A new backend route needs its own `location` block
in `docker/nginx.conf.template`, or it 404s.
