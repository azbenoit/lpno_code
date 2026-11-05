# LP-NO web demo

The demo runs the trained surrogates of the accompanying paper in a browser, on
your own machine. A FastAPI backend (`lpno_web_demo_backend/`) predicts the
temperature (T), liquid fraction (fl) and metal fraction (alpha) fields and
returns them as rendered images; a static frontend (`lpno_web_demo_frontend/`)
holds the controls. Every page offers the three architectures of the paper
(FNO, GAOT, RIGNO), trained on FLOW-3D WELD simulations of a laser scanning
Ti-6Al-4V.

## Pages

- **Steady state** (`index.html`): the time-averaged melt pool in the frame
  moving with the beam, at a laser power P and scan speed V set with sliders or
  on a P-V or enthalpy-power map, as a 2D cross-section or a 3D surface.
- **Straight track** (`line.html`): 20 frames of a 0.6 mm straight track at a
  chosen P and V, played with a time slider.
- **Free-form path** (`path.html`): a scan path you draw on the substrate, at a
  P and V within the selected model's training range, animated frame by frame.

With the optional ground truth installed, the steady and straight pages also
show the 27 held-out simulated cases as catalog points, and clicking one puts its
FLOW-3D field beside the prediction with the error. The free-form page then
offers the 22 held-out test paths as presets.

## Requirements

- Linux, git and Miniconda (the setup script creates a Python 3.10 environment).
- An NVIDIA GPU with a driver for CUDA 12.8 is recommended; 16 GB of GPU memory
  is enough. On a CPU the steady page answers in 0.5-1.8 s, while the time
  pages take 15-80 s per request and RIGNO exceeds their 60 s timeout
  ([`docs/developer_notes.md`](docs/developer_notes.md), "Hardware").
- About 10 GB of disk: 6 GB for the environment, 2 GB for the downloaded and
  extracted weights, 2 GB for the neighbour-graph cache. The ground truth adds
  a few GB.

## Quick start

```bash
cd self_hosted_web_demo/lpno_web_demo_backend

# 1. the environment (conda env "lpfno_demo", about 10 minutes)
scripts/setup_env.sh          # LPFNO_CUDA=cpu for CPU only, LPFNO_CUDA=cu126 for Volta GPUs
conda activate lpfno_demo

# 2. the model weights, from the open Zenodo record (doi:10.5281/zenodo.23325086)
python scripts/assets.py fetch zenodo

# 3. backend on :8001, frontend on :8080
dev/run_local.sh start        # then open http://127.0.0.1:8080/
dev/run_local.sh status       # or: stop
```

The first request to a GAOT or RIGNO model builds its neighbour graphs, which
takes a minute or two; later requests take a few seconds on a GPU. The backend
log is `dev/backend.log`. `python scripts/smoke.py http://127.0.0.1:8001` calls
every route with every model and exits non-zero if anything fails.

The repository's `tools/setup_env.py` builds the environment for training and
evaluation (`lpno`, `aiida-pino`). It does not install the web server, so the
demo uses `scripts/setup_env.sh`.

### Ground truth (optional)

The ground truth consists of FLOW-3D WELD outputs. They are distributed for
non-commercial research through the restricted-access record only
(doi:10.5281/zenodo.23325088), and access is granted on request from that
record's Zenodo page. Once it is granted, save a Zenodo personal access token
(Account > Applications > Personal access tokens) in a file and run

```bash
python scripts/assets.py fetch zenodo-ground-truth --token-file <file>
```

Without it, every page predicts and the comparison controls are greyed out.
`python scripts/assets.py verify` reports what is installed.

### Starting the two halves by hand

```bash
# backend, from lpno_web_demo_backend/ with the environment active
LPFNO_PAGES_ORIGIN=http://127.0.0.1:8080 \
  uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8001 --workers 1

# frontend, in a second shell from the same folder
echo '{"backendBaseUrl": "http://127.0.0.1:8001"}' > ../lpno_web_demo_frontend/public/config.local.json
python dev/serve_frontend.py ../lpno_web_demo_frontend/public 127.0.0.1 8080
```

Run exactly one worker: the one-request-at-a-time lock is in-process.
Settings keep the `LPFNO_` prefix of the earlier LP-FNO demo. `LPFNO_PAGES_ORIGIN` allows the page's origin for CORS, and the untracked
`config.local.json` points the pages at the backend. With
`LPFNO_USE_STUB_RUNTIME=true` the backend starts without any assets and serves
fabricated fields through the real routes, for interface work.

## Docker

The backend image carries CUDA, PyTorch and the model code and is built from the
repository root. The frontend image is nginx serving the pages and proxying the
backend routes, so the browser talks to a single origin. Neither image was
test-built for this release.

```bash
# from the repository root
docker build -f self_hosted_web_demo/lpno_web_demo_backend/docker/Dockerfile -t lpno-demo-backend .
docker build -f self_hosted_web_demo/lpno_web_demo_frontend/docker/Dockerfile \
  -t lpno-demo-frontend self_hosted_web_demo/lpno_web_demo_frontend

# the weights, on the host (Python 3.8 or newer, standard library only)
python3 data_deposit/scripts/download.py open --only lpno_web_demo_models.tar
tar -xf data_deposit/open_record/lpno_web_demo_models.tar -C self_hosted_web_demo/lpno_web_demo_backend

docker network create lpno-demo
docker run -d --name lpno-demo-backend --network lpno-demo --gpus all \
  -v "$PWD/self_hosted_web_demo/lpno_web_demo_backend/data:/app/data" \
  -v lpno-demo-outputs:/app/outputs lpno-demo-backend
docker run -d --name lpno-demo-frontend --network lpno-demo -p 127.0.0.1:8080:80 \
  -e LPFNO_BACKEND_UPSTREAM=http://lpno-demo-backend:8000 lpno-demo-frontend
```

Then open http://127.0.0.1:8080/. `--gpus all` needs the NVIDIA Container
Toolkit. The ground truth comes from
`python3 data_deposit/scripts/download.py restricted --token-file <file> --only lpno_web_demo_ground_truth.tar`
and is extracted the same way.

## Further reading

- [`docs/developer_notes.md`](docs/developer_notes.md): API conventions, asset
  layout and `meta.json`, tests, rendering rules, hardware.
- [`lpno_web_demo_backend/README.md`](lpno_web_demo_backend/README.md) and
  [`lpno_web_demo_frontend/README.md`](lpno_web_demo_frontend/README.md), and
  the `assumptions.md` and `potential_bugs.md` (known issues) of each half.
- The repository README gives the citation and the licence.
