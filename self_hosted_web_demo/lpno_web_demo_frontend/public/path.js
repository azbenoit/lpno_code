(() => {
  // Substrate top view in metres; keep-out inset is the drawable region.
  const SUBSTRATE = {
    xMin: -0.75e-3,
    xMax: 0.75e-3,
    yMin: -0.5e-3,
    yMax: 0.5e-3,
    insetM: 0.2e-3,
  };
  const MIN_POINT_SPACING_M = 2e-5; // ~20 um client-side decimation
  // Legacy fixed-physics arc-length cap. Per-model now: a model's
  // physics.max_path_length_m overrides it, and null means NO trim (the
  // variable-path model trained path length as a free variable).
  const MAX_ARC_LENGTH_M = 1.5e-3;
  const CANVAS_MARGIN_PX = 20;
  const BEAM_RADIUS_M = 5e-5; // self-intersection barrier radius
  const BARRIER_TAIL_ARC_M = 2 * BEAM_RADIUS_M; // trailing arc exempt from the barrier
  const START_XY_M = [-0.55e-3, 0]; // fixed freehand start (matches GT paths)
  const START_SNAP_RADIUS_M = 1e-4; // pointerdown acceptance radius around the start
  // Shown greyed out until /models delivers the selected model's physics block
  // (the served models' defaults); nothing is sent while the inputs are disabled.
  const FIXED_P_W = 150;
  const FIXED_V_MPS = 0.6;

  const state = {
    config: null,
    baseUrl: '',
    maxPoints: 512,
    drawnPoints: [], // [[x, y], ...] in metres
    preset: null, // { name, points_xy_m } when a preset is selected
    presets: [],
    modelPhysics: {}, // model id -> the /models "physics" block, or undefined (fixed)
    activePointerId: null, // pointerId of the stroke in progress, null when idle
    inFlight: false,
    rerunPending: false, // a path/control change arrived mid-render; rerun once it finishes
    dispatchedKey: '', // run key currently shown or in flight (auto-run dedup)
    view: '3d', // default overall view: 3D surface (matches the main page); '2d' | '3d'
    frameSets: {}, // field -> Promise<[{ time_s, img }]>, 2D sets of the loaded run
    frames3d: null, // Promise<[{ time_s, img }]> 3D surface set of the loaded run, or null
    runKey: '', // model+path signature of the loaded run; a change drops the other view's stale set
    drawHint: '', // transient message shown next to the path length
    lastMeta: null, // result_meta of the last successful 2D run
    lastMeta3d: null, // result_meta of the last successful 3D run
    lastModel: '',
    lastGtAvailable: false,
    gtMetrics: {}, // key `2d:<field>` or `3d` -> {field, mae, rmse} | null
    busyRetries: 0, // 429 auto-retry counter (single-GPU backend; cold model load holds the lock)
  };

  const els = {};
  const C = window.Common;
  const setStatus = (kind, text) => C.setStatus(els, kind, text);
  const showError = (message) => C.showError(els, message);
  const clearError = () => C.clearError(els);
  let player = null; // Common.FramePlayer: displayed frames + slider + Play
  let runner = null; // Common.TimeRunner: coalesced auto-run + render cache + 2D/3D swap

  async function init() {
    Object.assign(els, C.cacheElements(ELEMENT_IDS));
    player = C.FramePlayer(els);
    setStatus('loading', 'Loading config...');
    try {
      state.config = await C.loadConfig();
    } catch (error) {
      showError(`Failed to load config.json\n${String(error)}`);
      setStatus('error', 'Config load failed');
      return;
    }
    state.baseUrl = String(state.config.backendBaseUrl || '');
    state.maxPoints = Number(state.config.pathDemo?.maxPoints ?? 512);
    runner = C.TimeRunner({
      els, state, player,
      infer: `${state.baseUrl}/infer_path`, render3d: `${state.baseUrl}/infer_path/render3d`, label: 'path',
      buildPayload, currentRunKey, runKeyFor, renderSuccess, updateResultInfo, applyViewUi, canRun: hasPath,
    });

    els.zIndexSelect.innerHTML = '<option value="-1">surface (auto)</option>'; // z indices come from the first response's grid
    bindControls();
    bindCanvasPointerEvents();
    applyPhysicsUi(true); // fallback models are fixed-physics until /models says otherwise
    applyViewUi(); // reflect the default 3D view in the toggle + disabled controls
    drawCanvas();
    updatePathLength();
    setStatus('idle', 'Ready');
    clearError();

    void loadModels();
    void loadPresets();
  }

  // On /models failure the fallback <option>s stay and no physics is known,
  // so the P/V inputs stay disabled.
  async function loadModels() {
    const byId = await C.loadModels(state.baseUrl, 'path', els.modelSelect);
    state.modelPhysics = Object.fromEntries(Object.values(byId).filter((m) => m.physics).map((m) => [m.id, m.physics]));
    applyPhysicsUi(true);
    syncFrameDtOutput(); // the frame-count label depends on the model's duration
    drawCanvas(); // start-marker visibility follows the selected model
  }

  // The selected model's physics block when it declares variable P/V; null for
  // fixed-physics models and for the HTML fallback options (fetch failed).
  function currentPhysics() {
    const ph = state.modelPhysics[els.modelSelect.value];
    return ph && ph.p_watts && ph.v_scan_mps ? ph : null;
  }

  function startAnywhere() {
    return Boolean(state.modelPhysics[els.modelSelect.value]?.start_anywhere);
  }

  // Enable the P/V inputs only for variable-physics freehand runs. Fixed models
  // and preset runs (the backend 422s explicit p/v for both) grey them out but
  // still show the applied values. resetValues=true (model switch) loads the
  // model's defaults; otherwise a stale value is snapped into the trained range.
  function syncPvReadout() {
    els.pValue.textContent = `${Math.round(Number.parseFloat(els.pInput.value))} W`;
    els.vValue.textContent = `${Number.parseFloat(els.vInput.value).toFixed(2)} m/s`;
  }

  function applyPhysicsUi(resetValues) {
    const ph = currentPhysics();
    const enabled = Boolean(ph) && !state.preset;
    els.pInput.disabled = !enabled;
    els.vInput.disabled = !enabled;
    if (!ph) {
      els.pInput.value = String(FIXED_P_W);
      els.vInput.value = String(FIXED_V_MPS);
      syncPvReadout();
      return;
    }
    els.pInput.min = String(ph.p_watts.min);
    els.pInput.max = String(ph.p_watts.max);
    els.pInput.step = '1';
    els.vInput.min = String(ph.v_scan_mps.min);
    els.vInput.max = String(ph.v_scan_mps.max);
    els.vInput.step = '0.01';
    if (resetValues) {
      els.pInput.value = String(ph.p_watts.default);
      els.vInput.value = String(ph.v_scan_mps.default);
    } else if (enabled) {
      // Re-enabling (e.g. leaving a preset that showed its own physics).
      const p = Number.parseFloat(els.pInput.value);
      const v = Number.parseFloat(els.vInput.value);
      els.pInput.value = String(clamp(Number.isFinite(p) ? p : ph.p_watts.default, ph.p_watts.min, ph.p_watts.max));
      els.vInput.value = String(clamp(Number.isFinite(v) ? v : ph.v_scan_mps.default, ph.v_scan_mps.min, ph.v_scan_mps.max));
    }
    syncPvReadout();
  }

  // [p_watts, v_scan_mps] to send, or null when they must be omitted (fixed
  // model or preset run — the backend rejects explicit values for those).
  function pvToSend() {
    const ph = currentPhysics();
    if (!ph || state.preset) return null;
    const p = Number.parseFloat(els.pInput.value);
    const v = Number.parseFloat(els.vInput.value);
    return [
      Number.isFinite(p) ? p : ph.p_watts.default,
      Number.isFinite(v) ? v : ph.v_scan_mps.default,
    ];
  }

  const ELEMENT_IDS = [
    'statusBadge', 'latencyText', 'pathCanvas', 'pathLengthValue',
    'presetSelect', 'modelSelect', 'pInput', 'vInput', 'pValue', 'vValue', 'fieldSelect', 'zIndexSelect',
    'frameDtInput', 'frameDtValue',
    'clearBtn', 'runBtn', 'view2dBtn', 'view3dBtn', 'resultImage',
    'resultPlaceholder', 'errorBox', 'resultModeTag', 'resultInfoText',
    'gtMetrics', 'inferTiming', 'playBtn', 'frameSlider', 'frameTimeValue',
    'metaPoints', 'metaLength', 'metaLaserOff', 'metaFrames', 'metaClamped', 'metaTiming',
  ];

  function bindControls() {
    els.clearBtn.addEventListener('click', () => {
      state.drawnPoints = [];
      state.preset = null;
      state.drawHint = '';
      state.dispatchedKey = ''; // so redrawing the same path re-infers
      els.presetSelect.value = '';
      applyPhysicsUi(false); // clearing also leaves preset mode: re-gate P/V
      drawCanvas();
      updatePathLength();
    });

    els.fieldSelect.addEventListener('change', () => runner.onFieldChange());

    els.presetSelect.addEventListener('change', () => {
      const name = els.presetSelect.value;
      state.preset = state.presets.find((p) => p.name === name) || null;
      state.drawHint = '';
      if (state.preset) {
        state.drawnPoints = [];
      }
      applyPhysicsUi(false); // presets run their stored physics: P/V greyed out
      drawCanvas();
      updatePathLength();
      runner.requestRun(); // picking a preset path infers immediately (like a fresh draw)
    });

    els.modelSelect.addEventListener('change', async () => {
      // Presets are per-model, so a preset selected under the previous model is
      // foreign to this one (the backend 422s it): reload the list and keep the
      // selection only when the new model offers the same case; otherwise fall
      // back to freehand and drop the old model's render rather than leaving it
      // under the new model's name.
      const keep = state.preset ? state.preset.name : '';
      state.preset = null;
      applyPhysicsUi(true); // new model: its ranges + defaults (or fixed values)
      drawCanvas(); // start-marker visibility follows physics.start_anywhere
      await loadPresets();
      state.preset = state.presets.find((p) => p.name === keep) || null;
      els.presetSelect.value = state.preset ? keep : '';
      applyPhysicsUi(false); // preset kept -> P/V stay greyed; dropped -> re-gate
      drawCanvas();
      updatePathLength();
      if (hasPath()) runner.requestRun();
      else clearResult('Pick a preset or draw a path');
    });
    // Sliders: echo the value live while dragging (input), but only re-infer on
    // release (change) so a drag doesn't fire a GPU request per pixel.
    els.pInput.addEventListener('input', syncPvReadout);
    els.vInput.addEventListener('input', syncPvReadout);
    els.pInput.addEventListener('change', () => { syncPvReadout(); runner.requestRun(); });
    // Speed changes the weld duration, so it changes the frame count too.
    els.vInput.addEventListener('change', () => { syncPvReadout(); syncFrameDtOutput(); runner.requestRun(); });
    els.frameDtInput.addEventListener('input', syncFrameDtOutput);
    els.frameDtInput.addEventListener('change', () => { syncFrameDtOutput(); runner.requestRun(); });
    els.zIndexSelect.addEventListener('change', () => { runner.requestRun(); });

    // Run inference populates the frame set of whichever view is selected.
    els.runBtn.addEventListener('click', () => { runner.requestRun({ force: true }); });
    els.view2dBtn.addEventListener('click', () => { void runner.setView('2d'); });
    els.view3dBtn.addEventListener('click', () => { void runner.setView('3d'); });
  }

  // --- mm/metre <-> canvas px transform (the single source of truth) ---
  // Physics y points up; canvas y points down, so flip y.
  function canvasGeometry() {
    const canvas = els.pathCanvas;
    const plotW = canvas.width - 2 * CANVAS_MARGIN_PX;
    const plotH = canvas.height - 2 * CANVAS_MARGIN_PX;
    return { x: CANVAS_MARGIN_PX, y: CANVAS_MARGIN_PX, w: plotW, h: plotH };
  }

  function mToCanvas(xM, yM) {
    const g = canvasGeometry();
    return {
      x: g.x + ((xM - SUBSTRATE.xMin) / (SUBSTRATE.xMax - SUBSTRATE.xMin)) * g.w,
      y: g.y + ((SUBSTRATE.yMax - yM) / (SUBSTRATE.yMax - SUBSTRATE.yMin)) * g.h,
    };
  }

  function canvasToM(px, py) {
    const g = canvasGeometry();
    return {
      x: SUBSTRATE.xMin + ((px - g.x) / g.w) * (SUBSTRATE.xMax - SUBSTRATE.xMin),
      y: SUBSTRATE.yMax - ((py - g.y) / g.h) * (SUBSTRATE.yMax - SUBSTRATE.yMin),
    };
  }

  function bindCanvasPointerEvents() {
    const canvas = els.pathCanvas;

    const eventToM = (event) => {
      const rect = canvas.getBoundingClientRect();
      const px = ((event.clientX - rect.left) / rect.width) * canvas.width;
      const py = ((event.clientY - rect.top) / rect.height) * canvas.height;
      if (!Number.isFinite(px) || !Number.isFinite(py)) return null;
      return canvasToM(px, py); // unclamped metres
    };

    const addPoint = (event) => {
      const raw = eventToM(event);
      if (!raw) return;
      // Clamp into the keep-out (drawable) box, matching server behaviour.
      const m = {
        x: clamp(raw.x, SUBSTRATE.xMin + SUBSTRATE.insetM, SUBSTRATE.xMax - SUBSTRATE.insetM),
        y: clamp(raw.y, SUBSTRATE.yMin + SUBSTRATE.insetM, SUBSTRATE.yMax - SUBSTRATE.insetM),
      };
      if (state.drawnPoints.length >= state.maxPoints) return;
      const last = state.drawnPoints[state.drawnPoints.length - 1];
      if (last && Math.hypot(m.x - last[0], m.y - last[1]) < MIN_POINT_SPACING_M) return;
      if (barrierBlocked(m.x, m.y)) {
        state.drawHint = 'blocked: too close to existing track';
        updatePathLength();
        return;
      }
      state.drawHint = '';
      state.drawnPoints.push([m.x, m.y]);
      drawCanvas();
      updatePathLength();
    };

    canvas.addEventListener('pointerdown', (event) => {
      if (state.preset) return; // freehand disabled while a preset is shown
      if (state.activePointerId !== null) return; // ignore extra touches mid-stroke
      const raw = eventToM(event);
      if (!raw) return;
      let startPt;
      if (startAnywhere()) {
        // physics.start_anywhere models train with random starts: begin at the
        // press point, clamped into the keep-out box like every other point.
        startPt = [
          clamp(raw.x, SUBSTRATE.xMin + SUBSTRATE.insetM, SUBSTRATE.xMax - SUBSTRATE.insetM),
          clamp(raw.y, SUBSTRATE.yMin + SUBSTRATE.insetM, SUBSTRATE.yMax - SUBSTRATE.insetM),
        ];
      } else {
        // Fixed-start models: strokes must begin at the green marker.
        if (Math.hypot(raw.x - START_XY_M[0], raw.y - START_XY_M[1]) > START_SNAP_RADIUS_M) {
          state.drawHint = 'start at the green dot';
          updatePathLength();
          return;
        }
        startPt = [START_XY_M[0], START_XY_M[1]]; // fresh path, snapped to the start
      }
      state.activePointerId = event.pointerId;
      state.drawnPoints = [startPt];
      state.drawHint = '';
      try {
        canvas.setPointerCapture(event.pointerId);
      } catch (_) {
        // pointer capture may fail on some browsers/input types
      }
      drawCanvas();
      updatePathLength();
    });

    canvas.addEventListener('pointermove', (event) => {
      if (event.pointerId !== state.activePointerId) return;
      addPoint(event);
    });

    const endStroke = (event) => {
      if (event.pointerId !== state.activePointerId) return;
      state.activePointerId = null;
      try {
        if (typeof event.pointerId === 'number' && canvas.hasPointerCapture?.(event.pointerId)) {
          canvas.releasePointerCapture(event.pointerId);
        }
      } catch (_) {
        // ignore pointer capture release errors
      }
    };

    canvas.addEventListener('pointerup', (event) => {
      const wasActive = event.pointerId === state.activePointerId;
      endStroke(event);
      if (wasActive) runner.requestRun(); // a completed freehand path infers immediately
    });
    canvas.addEventListener('pointercancel', endStroke);
    canvas.addEventListener('lostpointercapture', () => {
      state.activePointerId = null;
    });
  }

  function pathLengthM(points) {
    let total = 0;
    for (let i = 1; i < points.length; i++) {
      total += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
    }
    return total;
  }

  function distToSegmentM(px, py, ax, ay, bx, by) {
    const dx = bx - ax;
    const dy = by - ay;
    const len2 = dx * dx + dy * dy;
    const t = len2 ? clamp(((px - ax) * dx + (py - ay) * dy) / len2, 0, 1) : 0;
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }

  // True when the candidate point is within one beam radius of any earlier
  // path segment, excluding the trailing 2*r of arc length at the current end
  // (otherwise normal continuation would self-block).
  // O(n) scan per candidate, fine at the 512-point cap.
  function barrierBlocked(px, py) {
    const pts = state.drawnPoints;
    if (pts.length < 2) return false;
    const cum = [0];
    for (let i = 1; i < pts.length; i++) {
      cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    }
    const cutoff = cum[cum.length - 1] - BARRIER_TAIL_ARC_M;
    if (cutoff <= 0) return false;
    for (let i = 0; i < pts.length - 1; i++) {
      if (cum[i] >= cutoff) break;
      let [bx, by] = pts[i + 1];
      if (cum[i + 1] > cutoff) {
        // Segment straddles the cutoff: only its leading part is a barrier.
        const t = (cutoff - cum[i]) / (cum[i + 1] - cum[i]);
        bx = pts[i][0] + t * (pts[i + 1][0] - pts[i][0]);
        by = pts[i][1] + t * (pts[i + 1][1] - pts[i][1]);
      }
      if (distToSegmentM(px, py, pts[i][0], pts[i][1], bx, by) < BEAM_RADIUS_M) return true;
    }
    return false;
  }

  function activePoints() {
    return state.preset ? state.preset.points_xy_m : state.drawnPoints;
  }

  // The selected model's arc-length cap in metres, or null when it does not trim.
  function maxArcLengthM() {
    const ph = currentPhysics();
    if (!ph) return MAX_ARC_LENGTH_M; // fixed-physics models keep the legacy cap
    return ph.max_path_length_m === null || ph.max_path_length_m === undefined
      ? null // explicit null = variable-length model: never trimmed
      : ph.max_path_length_m;
  }

  function updatePathLength() {
    const lenMm = pathLengthM(activePoints()) * 1000;
    const capM = maxArcLengthM();
    const trimmed = capM !== null && lenMm > capM * 1000
      ? ` (will be trimmed to ${(capM * 1000).toFixed(3)})`
      : '';
    const hint = state.drawHint ? ` | ${state.drawHint}` : '';
    els.pathLengthValue.textContent = `${lenMm.toFixed(3)} mm${trimmed}${hint}`;
    // The predicted frame count depends on exactly this: path length and speed.
    syncFrameDtOutput();
  }

  function drawCanvas() {
    const canvas = els.pathCanvas;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const grad = ctx.createLinearGradient(0, 0, w, h);
    grad.addColorStop(0, '#e7f4f6');
    grad.addColorStop(1, '#f9ead9');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    // Substrate outline
    const tl = mToCanvas(SUBSTRATE.xMin, SUBSTRATE.yMax);
    const br = mToCanvas(SUBSTRATE.xMax, SUBSTRATE.yMin);
    ctx.fillStyle = '#fbfcfd';
    ctx.fillRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
    ctx.strokeStyle = 'rgba(22,34,53,0.45)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);

    // Keep-out (drawable) box, dashed
    const itl = mToCanvas(SUBSTRATE.xMin + SUBSTRATE.insetM, SUBSTRATE.yMax - SUBSTRATE.insetM);
    const ibr = mToCanvas(SUBSTRATE.xMax - SUBSTRATE.insetM, SUBSTRATE.yMin + SUBSTRATE.insetM);
    ctx.save();
    ctx.setLineDash([6, 5]);
    ctx.strokeStyle = 'rgba(12,109,124,0.6)';
    ctx.lineWidth = 1;
    ctx.strokeRect(itl.x, itl.y, ibr.x - itl.x, ibr.y - itl.y);
    ctx.restore();

    ctx.fillStyle = 'rgba(22,34,53,0.65)';
    ctx.font = '12px IBM Plex Sans, Segoe UI, sans-serif';
    ctx.fillText('x: -0.75 to 0.75 mm, y: -0.5 to 0.5 mm', tl.x + 4, br.y + 14);

    const points = activePoints();
    const tracePath = () => {
      ctx.beginPath();
      points.forEach(([x, y], i) => {
        const c = mToCanvas(x, y);
        if (i === 0) ctx.moveTo(c.x, c.y);
        else ctx.lineTo(c.x, c.y);
      });
    };

    if (points.length) {
      // Beam-radius exclusion halo under the freehand polyline.
      if (!state.preset && points.length >= 2) {
        const pxPerM = canvasGeometry().w / (SUBSTRATE.xMax - SUBSTRATE.xMin);
        ctx.strokeStyle = 'rgba(255,140,0,0.25)';
        ctx.lineWidth = 2 * BEAM_RADIUS_M * pxPerM;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        tracePath();
        ctx.stroke();
      }

      ctx.strokeStyle = state.preset ? '#d6721f' : '#0c6d7c';
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      tracePath();
      ctx.stroke();

      if (state.preset) {
        // Preset start marker
        const start = mToCanvas(points[0][0], points[0][1]);
        ctx.fillStyle = '#162235';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(start.x, start.y, 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }

    if (!state.preset && !startAnywhere()) {
      // Fixed freehand start marker (hidden for start-anywhere models).
      const s = mToCanvas(START_XY_M[0], START_XY_M[1]);
      ctx.fillStyle = '#1f8f4a';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(s.x, s.y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    canvas.style.cursor = state.preset ? 'default' : 'crosshair';
  }

  // Presets are per-model: each model is only offered the ground-truth cases it
  // was trained for (the variable-path model's test split is not rp3's, and vice
  // versa — the backend 422s a foreign preset). Reloaded on every model change.
  async function loadPresets() {
    const model = els.modelSelect.value;
    const url = `${state.baseUrl}/infer_path/presets?model=${encodeURIComponent(model)}`;
    try {
      const resp = await fetch(url, { cache: 'no-store', credentials: 'omit' });
      const body = await resp.json();
      if (!resp.ok || body.status !== 'ok' || !Array.isArray(body.presets)) {
        throw new Error(body?.error?.message || `HTTP ${resp.status}`);
      }
      state.presets = body.presets;
      const options = ['<option value="">Freehand drawing</option>'];
      for (const p of state.presets) {
        // Variable-physics GT sets carry their own P/V — show them so the case's
        // conditions are visible before running it.
        const pv = Number.isFinite(p.p_watts) && Number.isFinite(p.v_scan_mps)
          ? ` (${Math.round(p.p_watts)} W, ${p.v_scan_mps.toFixed(2)} m/s)`
          : '';
        options.push(
          `<option value="${escapeHtml(p.name)}">${escapeHtml(p.name)}${escapeHtml(pv)}</option>`
        );
      }
      els.presetSelect.innerHTML = options.join('');
    } catch (error) {
      // Presets are optional; freehand still works.
      state.presets = [];
      els.presetSelect.innerHTML = '<option value="">Freehand drawing</option>';
      setStatus('idle', `Presets unavailable: ${String(error)}`);
    }
    // The presets are the simulated test paths, so they exist only where the ground truth is installed.
    C.setGtAvailable(els.presetSelect, state.presets.length > 0);
  }

  // --- request plumbing (Common.TimeRunner hooks) ---
  function hasPath() {
    return Boolean(state.preset) || state.drawnPoints.length >= 2;
  }

  // null (with the message) when nothing is drawn yet.
  function buildPayload(is3d) {
    if (!hasPath()) {
      showError('Draw a path with at least 2 points (or pick a preset) before running inference.');
      return null;
    }
    const pathDemo = state.config.pathDemo || {};
    const payload = {
      client_request_id: C.makeRequestId(),
      model: els.modelSelect.value,
      path: state.preset
        ? { preset: state.preset.name }
        : { points_xy_m: state.drawnPoints },
      render: {
        width_px: Number(pathDemo.render?.width_px ?? 900),
        height_px: Number(pathDemo.render?.height_px ?? 520),
      },
    };
    const pv = pvToSend(); // omitted for fixed models and presets (backend 422)
    if (pv) {
      payload.p_watts = pv[0];
      payload.v_scan_mps = pv[1];
    }
    payload.frame_dt_s = frameDtSeconds();
    // Lazy per-field 2D rendering: request only the selected field; other
    // fields are fetched on demand (render-only thanks to the server's
    // prediction cache). The 3D surface run takes no view.
    if (!is3d) payload.view = { field: els.fieldSelect.value, z_index: Number.parseInt(els.zIndexSelect.value, 10) };
    return payload;
  }

  // --- frame cadence ---------------------------------------------------- //
  // Frames are spaced by a time step, so the count follows from the weld
  // duration rather than being pinned at 20. These mirror the server's
  // frame_count_for() so the control can show the consequence of a step
  // BEFORE the request, not just after it.
  const MIN_FRAMES = 5;
  const MAX_FRAMES = 100;
  const FIXED_WINDOW_S = 2.0e-3; // fixed-duration models' trained window

  function frameDtSeconds() {
    const us = Number(els.frameDtInput && els.frameDtInput.value);
    return Number.isFinite(us) && us > 0 ? us * 1e-6 : null;
  }

  // Weld duration of what is currently selected: arc length / speed, from the
  // preset's own stored physics or the drawn path plus the speed control.
  // null when nothing is drawn yet, or for fixed-window models (their span is
  // a constant, so the step just subdivides it).
  function currentDurationS() {
    if (!state.modelPhysics[els.modelSelect.value]?.duration_scaled_times) {
      return FIXED_WINDOW_S;
    }
    const length = pathLengthM(activePoints());
    if (!(length > 0)) return null;
    const v = state.preset && Number.isFinite(state.preset.v_scan_mps)
      ? state.preset.v_scan_mps : Number(els.vInput.value);
    return v > 0 ? length / v : null;
  }

  function predictedFrames(dtS) {
    const total = currentDurationS();
    if (!(total > 0) || !(dtS > 0)) return null;
    return Math.max(MIN_FRAMES, Math.min(MAX_FRAMES, Math.round(total / dtS) + 1));
  }

  function syncFrameDtOutput() {
    if (!els.frameDtInput || !els.frameDtValue) return;
    const us = Number(els.frameDtInput.value);
    const n = predictedFrames(us * 1e-6);
    // The correlation the control exists to make visible.
    els.frameDtValue.textContent = n == null
      ? `${us} µs`
      : `${us} µs -> ${n} frame${n === 1 ? '' : 's'}`;
  }

  // What a response IS: model, applied physics, the polyline the server used
  // and the frame cadence. A change drops the other view's cached set.
  function runKeyFor(body, payload) {
    const pts = body.path_meta?.points_xy_m;
    return JSON.stringify([
      body.model,
      // Applied laser physics (null for fixed models): a P/V change is a new run.
      body.path_meta?.p_watts ?? null, body.path_meta?.v_scan_mps ?? null,
      Array.isArray(pts) && pts.length ? pts : payload.path,
      // A different step is a different set of frames, so a different run.
      body.path_meta?.n_frames ?? payload.frame_dt_s ?? null,
    ]);
  }

  function applyViewUi() {
    const is3d = state.view === '3d';
    C.setSegment([els.view2dBtn, els.view3dBtn], is3d ? els.view3dBtn : els.view2dBtn);
    els.fieldSelect.disabled = is3d;
    els.fieldSelect.title = is3d ? 'fields apply to the 2D view' : '';
    // Z slice only affects the 2D top view; the 3D render ignores it (the run key
    // nulls it), so disable it in 3D instead of leaving an inert control.
    if (els.zIndexSelect) {
      els.zIndexSelect.disabled = is3d;
      els.zIndexSelect.title = is3d ? 'z slice applies to the 2D view' : '';
    }
  }

  // What identifies the currently requested render: model, view, laser P/V,
  // 2D field/slice, the active path and the frame step. Used to skip redundant runs and to detect
  // that the user drew a new path while a render was in flight.
  function currentRunKey() {
    return JSON.stringify([
      state.view, els.modelSelect.value,
      pvToSend(), // P/V participate for variable-physics freehand runs
      state.view === '2d' ? els.fieldSelect.value : null,
      state.view === '2d' ? els.zIndexSelect.value : null,
      state.preset ? ['preset', state.preset.name] : ['path', state.drawnPoints],
      frameDtSeconds(), // a different step is a different frame set (runKeyFor keys on n_frames)
    ]);
  }

  function updateResultInfo() {
    const gtNote = state.lastGtAvailable ? ' | GT available' : '';
    if (state.view === '3d') {
      const meta = state.lastMeta3d || {};
      els.resultModeTag.textContent = `${state.lastModel} / 3D`;
      els.resultInfoText.textContent =
        `3D surface | grid=${(meta.grid_shape_xyz || []).join('x') || '-'}`;
    } else {
      const meta = state.lastMeta || {};
      els.resultModeTag.textContent = `${state.lastModel} / ${meta.panel || 'single'}`;
      els.resultInfoText.textContent =
        `${els.fieldSelect.value} | z=${meta.z_index_used ?? '-'} | grid=${(meta.grid_shape_xyz || []).join('x') || '-'}${gtNote}`;
    }
    const key = state.view === '3d' ? '3d' : `2d:${els.fieldSelect.value}`;
    C.setGtMetrics(els.gtMetrics, state.gtMetrics[key]);
    els.resultModeTag.className = 'badge ok';
  }

  function renderSuccess(body) {
    const meta = body.result_meta || {};
    const pathMeta = body.path_meta || {};
    if (meta.panel === '3d') state.lastMeta3d = meta;
    else state.lastMeta = meta;
    state.lastModel = body.model;
    state.lastGtAvailable = Boolean(pathMeta.gt_available);
    C.setZIndexOptions(els.zIndexSelect, (meta.grid_shape_xyz || [])[2]);

    const is3d = meta.panel === '3d';
    const mField = is3d ? 'T' : (meta.field || 'T');
    const key = is3d ? '3d' : `2d:${mField}`;
    state.gtMetrics[key] = pathMeta.gt_available && pathMeta.gt_mae != null
      ? { field: mField, mae: pathMeta.gt_mae, rmse: pathMeta.gt_rmse } : null;
    updateResultInfo();

    els.metaPoints.textContent = String(pathMeta.n_points_used ?? '-');
    els.metaLength.textContent = Number.isFinite(pathMeta.path_length_m)
      ? `${(pathMeta.path_length_m * 1000).toFixed(3)} mm` : '-';
    els.metaLaserOff.textContent = Number.isFinite(pathMeta.laser_off_time_s)
      ? `${(pathMeta.laser_off_time_s * 1000).toFixed(3)} ms` : '-';
    // Effective cadence, not what was asked for: frames snap to the model's
    // dose grid and the count is capped, so echo what was rendered.
    els.metaFrames.textContent = Number.isFinite(pathMeta.n_frames)
      ? `${pathMeta.n_frames} @ ${Number.isFinite(pathMeta.frame_dt_s)
          ? `${(pathMeta.frame_dt_s * 1e6).toFixed(1)} µs` : '-'}`
      : '-';
    const clampBits = [
      pathMeta.clamped ? 'path' : '',
      pathMeta.physics_clamped ? 'P/V' : '',
    ].filter(Boolean);
    els.metaClamped.textContent = clampBits.length ? `yes (${clampBits.join(', ')})` : 'no';
    // A preset replays a stored case: show its actual physics in the greyed-out
    // inputs so the user sees what was applied.
    if (state.preset) {
      // written to the <output>s directly: a range input clamps its value to
      // its own min/max, and a preset may sit outside the model's envelope
      if (Number.isFinite(pathMeta.p_watts)) { els.pInput.value = String(pathMeta.p_watts); els.pValue.textContent = `${Math.round(pathMeta.p_watts)} W`; }
      if (Number.isFinite(pathMeta.v_scan_mps)) { els.vInput.value = String(pathMeta.v_scan_mps); els.vValue.textContent = `${Number(pathMeta.v_scan_mps).toFixed(2)} m/s`; }
    }
    const t = body.timing_ms || {};
    els.metaTiming.textContent =
      `infer ${Math.round(t.infer || 0)} ms, render ${Math.round(t.render || 0)} ms`;

    // Show the polyline the server actually used — but only if the user hasn't
    // started or drawn a new path since this run was dispatched. Otherwise the
    // completed run's polyline would clobber the latest draw (which a queued
    // rerun is about to render).
    const unchanged = state.activePointerId === null && currentRunKey() === state.dispatchedKey;
    if (unchanged && Array.isArray(pathMeta.points_xy_m) && pathMeta.points_xy_m.length) {
      if (state.preset) {
        state.preset = { ...state.preset, points_xy_m: pathMeta.points_xy_m };
      } else {
        state.drawnPoints = pathMeta.points_xy_m;
      }
      drawCanvas();
      updatePathLength();
      state.dispatchedKey = currentRunKey(); // the server polyline is now the shown identity
    }
  }

  // Drop the shown render (e.g. a preset the new model does not offer): the
  // panel goes back to its placeholder instead of showing a stale model's frames.
  function clearResult(why) {
    player.stop();
    player.frames = [];
    els.resultImage.classList.add('hidden');
    els.resultPlaceholder.classList.remove('hidden');
    els.frameSlider.disabled = true;
    els.playBtn.disabled = true;
    els.resultModeTag.textContent = 'none';
    els.resultModeTag.className = 'badge idle';
    els.resultInfoText.textContent = why;
    C.setGtMetrics(els.gtMetrics, null);
    els.inferTiming.classList.add('hidden');
    state.dispatchedKey = '';
    state.runKey = '';
    state.frameSets = {};
    state.frames3d = null;
    setStatus('idle', why);
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (ch) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[ch]));
  }

  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

  window.addEventListener('DOMContentLoaded', init);
})();
