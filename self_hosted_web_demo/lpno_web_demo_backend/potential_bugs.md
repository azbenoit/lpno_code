# Known issues: backend

Ways the backend can mislead or fail, with the cause, a workaround and where to
look in the code. IDs are stable because code comments cite them; a missing
number is a retired entry. A-numbers refer to `assumptions.md`. Browser-side
issues are in `../lpno_web_demo_frontend/potential_bugs.md`.

- **P2. Drawn paths can be out of distribution.**
  Symptom: a path with sharp corners, dwells, a self-crossing track or
  statistics far from the training random walks gets a confident field and no
  warning. Cause: the server clamps and resamples the path but has no
  out-of-distribution check, and the self-intersection barrier exists only in
  the browser (frontend P2). Workaround: treat such paths as extrapolation; with
  the ground truth installed, the held-out presets show the error on in-range
  paths. Code: `prepare_path` in `app/adapter_randompath.py`. Cites A1, A4.
- **P3. The prediction cache holds eight entries per route.**
  Symptom: switching among more than eight recent requests re-runs inference
  each time and looks like a slowdown. Cause: a bounded LRU keyed by the model
  and a hash of the resampled path and frame times (A8). Workaround: raise
  `LPFNO_PREDICTION_CACHE_ENTRIES`. Code: the prediction caches in
  `app/service.py`.
- **P4. Timed-out work is not cancelled.**
  Symptom: after a 504, an immediate retry waits behind the still-running work
  and can time out again, typically on the first request to a GAOT or RIGNO
  model. Cause: the timeout releases the request lock while the worker thread
  keeps computing, and model loading is serialized separately (A7). Workaround:
  wait for the first request to finish; the slow part is the one-off graph
  build (P18). Code: `asyncio.wait_for` in `app/service.py`; timeouts
  `final_timeout_ms` and `path_timeout_ms` in `app/config.py`.
- **P6. `/healthz` covers only the steady runtime.**
  Symptom: `/healthz` is ok while a straight or free-form model folder, or a
  time page's grid, is broken; the error appears on the first request to that
  page. Cause: `health()` summarizes the steady runtime only. Workaround:
  `python scripts/assets.py verify` checks every page's files, and
  `python scripts/smoke.py <backend URL>` calls every route. Code:
  `InferenceService.health` in `app/service.py`.
- **P7. Preset folders are trusted.**
  Symptom: pointing `LPFNO_PATH_DATA_DIR` or a model's `physics.presets_dir` at
  training cases presents them as held-out presets and overstates accuracy.
  Cause: the server serves whatever cases the folder holds and labels every
  free-form preset "test" (A19). Workaround: keep each model's `presets_dir` on
  its own test split, as `scripts/assets.py` installs it. Code: `_data_dir` and
  `presets` in `app/adapter_randompath.py`.
- **P9. Without headless OpenGL, the 3D views drop the ground-truth panel.**
  Symptom: the 3D views show matplotlib top-surface frames of the prediction
  only, while `path_meta.gt_available` stays `true`. Cause: when PyVista cannot
  render off screen, the renderer falls back to a single-series matplotlib
  view and logs "PyVista block render failed; falling back to matplotlib
  frames". Workaround: install a working EGL or OSMesa stack (the backend
  Docker image installs `libegl1`). Code: the fallback around
  `_pyvista_block_frames` in `app/render.py`.
- **P11. A steady comparison can include a window offset.**
  Symptom: in `/infer/gt_compare`, the predicted melt pool sits a few cells off
  the ground truth along x and the MAE is higher than expected. Cause: the
  prediction is shifted onto the matched case's x window only when both grids
  have the same number of x cells; otherwise no shift is applied and the log
  shows "steady GT has N x cells but the runtime serves M; not shifting" (A18).
  Workaround: use ground truth built on the model's own window, as the
  installed bundle is. Code: `_steady_window_shift` in `app/service.py`.
- **P12. Ground truth exists only at catalog points.**
  Symptom: `/infer_straight` and `/infer/gt_compare` return a prediction alone
  (`gt_available: false`) over most of the P-V plane, and everywhere when no
  ground truth is installed. Cause: ground truth is served within 1 W and
  0.005 m/s of a catalog case (A17), and the catalog is empty without the
  restricted-record bundle. Workaround: pick a point from `GET /gt_catalog`, or
  install the ground truth with
  `python scripts/assets.py fetch zenodo-ground-truth --token-file <file>`.
  Code: `nearest_pv` in `app/gt.py`, `catalog` in `app/steady_gt.py`,
  `gt_catalog` in `app/adapter_straightpath.py`.
- **P14. Steady RIGNO can run out of GPU memory on a shared card.**
  Symptom: a steady RIGNO request returns 500 with a CUDA out-of-memory error
  while the idle backend holds little memory. Cause: the steady RIGNO forward
  builds neighbour graphs on the super-resolved grid and needs several GiB for
  the duration of the call, and the GPU is chosen once at start-up, so a job
  that later fills that card leaves too little room. Status: a forward that
  runs out of memory drops the cached models and retries once. Workaround: free
  the card, call `POST /cache/clear`, restart the backend so it picks the
  freest GPU, or pin a free card with `LPFNO_CUDA_DEVICE`. Code:
  `_infer_retrying_on_oom` in `app/service.py`, `app/gpu.py`.
- **P15. The steady P/V clamp is page-wide.**
  Symptom: a direct API call outside a model's trained range returns a
  confident field. Cause: `/infer` clamps P and V to `LPFNO_P_MIN_WATTS`,
  `LPFNO_P_MAX_WATTS`, `LPFNO_V_MIN_MPS` and `LPFNO_V_MAX_MPS` (40-215 W,
  0.101-0.96 m/s, the served models' range); a model's `meta.json` envelope is
  enforced only by the frontend sliders. The straight route takes free P/V and
  does not clamp. Workaround: when adding a model with a narrower envelope, set
  those limits to it. Code: `Settings.clamp_inputs` in `app/config.py`.
- **P16. One scan speed per run.**
  Symptom: there is no way to request a path whose speed varies along the
  track. Cause: every request carries one `v_scan_mps`. Status: not supported.
  Code: `ModelPhysics` in `app/model_physics.py`.
- **P17. Frames are predicted independently.**
  Each frame is predicted from the beam position and dose history at its own
  time (`time_as_batch`, A5). There is no autoregressive mode that feeds a
  prediction back as the next input; the served checkpoints were not trained
  for one. Code: `_build_invar` and the forward in `app/adapter_randompath.py`.
- **P18. The first request to a graph-based model is slow and shows no progress.**
  Symptom: the first request to a GAOT or RIGNO model runs for a minute or two
  with a spinner and no progress report. Cause: the neighbour graphs are built
  on first use and cached on disk under `outputs/rigno_graphs`
  (`LPFNO_PATH_RIGNO_GRAPH_CACHE_DIR`), and each request returns one response
  at the end. Workaround: keep that directory between runs (the `/app/outputs`
  volume in Docker). Code: `graph_cache_dir` in `app/submodule.py`.
- **P19. Inference runs in fp32 only.**
  Inference is eager PyTorch in fp32. Half precision is untested; some FNO
  models overflowed in fp16 during training at keyhole temperatures (about
  3800 K). Code: the forward calls in `app/adapter_pino.py` and
  `app/adapter_randompath.py`.
- **P20. A time page without ground truth needs its `grid.npz`.**
  Symptom: `/infer_straight` or `/infer_path` fails, and the log shows "no
  ground-truth case and no grid.npz to take the grid from" or "No straight
  case NPZ files". Cause: the time pages take their grid from the first
  ground-truth case and otherwise from `data/<page>/grid.npz`; with neither, no
  grid exists. With ground truth installed, its grid wins, so both must
  describe the same mesh. Workaround: `python scripts/assets.py verify` names
  the missing file. Code: `grid` and `_page_grid` in
  `app/adapter_randompath.py` and `app/adapter_straightpath.py`.
