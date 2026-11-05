(() => {
  const MAP_W = 640, MAP_H = 435; // both selection-map canvases

  const state = {
    config: null,
    latestIssuedSeq: 0,
    latestRenderedSeq: 0,
    inFlight: null, // { controller, seq, startedAt } while a request is out
    pendingFinal: null, // a render requested while one was in flight (latest wins)
    pvMapPointerActive: false,
    hpMapPointerActive: false,
    controls: {
      p_watts: 50,
      v_scan_mps: 0.26,
      h_norm: null,
      field: 'T',
      plot_mode: '3d',
      plane: 'xz', // 'xz' cross-section (default) | 'xy' top view
      y_index: -1,
      show_t_contour: true,
      t_contour_k: 1898.0,
    },
    lastResponse: null,
    pvMapHolder: {}, // PVMap background cache (P-V map)
    hpMapHolder: { variant: 'hp' }, // PVMap background cache (H-P map; hMin set from config)
    hNormBounds: null,
    showGt: false, // "Compare to ground truth" toggle (off preserves the default 3D melt-pool view)
    gtCatalog: [], // /gt_catalog?dataset=steady P-V points
    selectedGtName: null, // name of the clicked GT dot, for the map highlight
    busyRetries: 0, // 429 auto-retry counter (Common.handleErrorResponse)
  };

  const els = {};
  const C = window.Common;
  const M = window.PVMap;
  const setStatus = (kind, text) => C.setStatus(els, kind, text);
  const showError = (message) => C.showError(els, message);
  const clearError = () => C.clearError(els);
  const computeHStar = (p, v) => M.computeHStar(p, v);

  const ELEMENT_IDS = [
    'statusBadge','latencyText',
    'pSlider','vSlider','hSlider','pValue','vValue','hValue','modelSelect','fieldSelect','plotMode2dBtn','plotMode3dBtn',
    'planeXzBtn','planeTopBtn',
    'yIndexInput','tContourInput','yIndexLabel','tContourLabel','showContourLabel',
    'showContourCheckbox','finalNowBtn','resultImage','resultPlaceholder',
    'errorBox','resultModeTag','resultInfoText','gtMetrics','inferTiming','metaP','metaV','metaHStar','metaHModel','metaHNorm',
    'pvMapCanvas','hpMapCanvas','gtCompareCheckbox',
  ];

  async function init() {
    Object.assign(els, C.cacheElements(ELEMENT_IDS));
    els.finalNowBtn.addEventListener('click', () => queueLatestFinal('manual-final'));
    setStatus('loading', 'Loading config...');
    try {
      state.config = await C.loadConfig();
      state.config.finalTimeoutMs = Number(state.config.finalTimeoutMs ?? 30000);
      state.config.requestMaxBodyBytes = Number(state.config.requestMaxBodyBytes ?? 16384);
    } catch (error) {
      showError(`Failed to load config.json\n${String(error)}`);
      setStatus('error', 'Config load failed');
      return;
    }

    configureControlsFromConfig();
    syncControlOutputs();
    drawMaps();
    setStatus('idle', 'Ready');
    clearError();

    // Render immediately with the default P/V, like the straight page.
    void issueInferRequest({ reason: 'initial-load' });

    void loadModels();
    void loadSteadyGtCatalog();
  }

  async function loadModels() {
    state.modelsById = await C.loadModels(state.config.backendBaseUrl, 'steady', els.modelSelect);
    applyModelCaps();
  }

  // P/V slider envelope for the selected model (meta.json "physics" via
  // /models). The H slider keeps the page bounds; an out-of-envelope H-derived
  // P/V is clamped server-side and the response's clamped_inputs reflects it.
  function applyModelCaps() {
    const info = (state.modelsById || {})[els.modelSelect.value] || {};
    const bounds = state.config && state.config.bounds;
    if (!bounds) return;
    const ph = info.physics || null;
    const clampSlider = (input, cfg, rng, key) => {
      if (!input || !cfg) return;
      // Snap the model envelope onto the config step grid: a range input's
      // value grid is min + k*step, so a raw min like 0.101 with step 0.01
      // would turn every slider value into x.xx1.
      const step = Number(cfg.step) || 1;
      const onGrid = (x, round) => Number((round(x / step) * step).toFixed(6));
      const min = rng && rng.min != null ? onGrid(Number(rng.min) - 1e-9, Math.ceil) : cfg.min;
      const max = rng && rng.max != null ? onGrid(Number(rng.max) + 1e-9, Math.floor) : cfg.max;
      input.min = String(min);
      input.max = String(max);
      const cur = Math.min(max, Math.max(min, Number(state.controls[key])));
      state.controls[key] = cur;
      input.value = String(cur);
    };
    clampSlider(els.pSlider, bounds.p_watts, ph && ph.p_watts, 'p_watts');
    clampSlider(els.vSlider, bounds.v_scan_mps, ph && ph.v_scan_mps, 'v_scan_mps');
    syncControlOutputs();
    drawMaps();
  }

  async function loadSteadyGtCatalog() {
    try {
      const resp = await fetch(`${state.config.backendBaseUrl || ''}/gt_catalog?dataset=steady`, { cache: 'no-store', credentials: 'omit' });
      const body = await resp.json();
      if (resp.ok && body.status === 'ok' && Array.isArray(body.entries)) {
        state.gtCatalog = body.entries;
        drawMaps();
      }
    } catch (_) {
      // GT points are optional; the page works without them.
    }
    C.setGtAvailable(els.gtCompareCheckbox, state.gtCatalog.length > 0);
  }

  function configureControlsFromConfig() {
    const { bounds, defaultView } = state.config;

    setupRange(els.pSlider, bounds.p_watts);
    setupRange(els.vSlider, bounds.v_scan_mps);
    state.controls.p_watts = Number(bounds.p_watts.initial ?? bounds.p_watts.min);
    state.controls.v_scan_mps = Number(bounds.v_scan_mps.initial ?? bounds.v_scan_mps.min);
    els.pSlider.value = String(state.controls.p_watts);
    els.vSlider.value = String(state.controls.v_scan_mps);
    state.hNormBounds = buildHNormSliderBounds(bounds);
    setupRange(els.hSlider, state.hNormBounds);
    els.hSlider.value = String(state.hNormBounds.initial);

    state.controls.field = defaultView.field ?? 'T';
    state.controls.plot_mode = defaultView.plot_mode === '3d' ? '3d' : '2d';
    state.controls.y_index = Number(defaultView.y_index ?? -1);
    state.controls.show_t_contour = Boolean(defaultView.show_t_contour);
    state.controls.t_contour_k = Number(defaultView.t_contour_k ?? 1898.0);
    els.fieldSelect.value = state.controls.field;
    els.yIndexInput.value = String(state.controls.y_index);
    els.showContourCheckbox.checked = state.controls.show_t_contour;
    els.tContourInput.value = String(state.controls.t_contour_k);

    els.pvMapCanvas.width = MAP_W; els.pvMapCanvas.height = MAP_H;
    els.hpMapCanvas.width = MAP_W; els.hpMapCanvas.height = MAP_H;
    state.pvMapHolder = {};
    // The H-P map's y axis starts at the lowest reachable H* (or the notebook's
    // 0.75 A_corr floor, whichever is lower).
    state.hpMapHolder = { variant: 'hp', hMin: Math.min(0.75 * M.vizConstants().A_corr, getGlobalFeasibleHRange(bounds).min) };

    bindControlEvents();
    updatePlotModeButtons();
    updatePlaneButtons();
    updateViewModeUi();
  }

  function setupRange(input, cfg) {
    input.min = String(cfg.min);
    input.max = String(cfg.max);
    input.step = String(cfg.step);
  }

  function bindControlEvents() {
    // Live auto-render while dragging: every move queues a render and
    // issueInferRequest coalesces to at-most-one-in-flight (latest wins), so
    // the single-GPU backend never piles up. 'change' (release) is the final.
    els.pSlider.addEventListener('input', () => onSliderInput('p_watts', Number(els.pSlider.value)));
    els.vSlider.addEventListener('input', () => onSliderInput('v_scan_mps', Number(els.vSlider.value)));
    els.pSlider.addEventListener('change', () => onSliderInput('p_watts', Number(els.pSlider.value)));
    els.vSlider.addEventListener('change', () => onSliderInput('v_scan_mps', Number(els.vSlider.value)));
    els.hSlider.addEventListener('input', () => onHSliderInput(Number(els.hSlider.value)));
    els.hSlider.addEventListener('change', () => onHSliderInput(Number(els.hSlider.value)));

    els.fieldSelect.addEventListener('change', () => {
      state.controls.field = els.fieldSelect.value;
      syncControlOutputs();
      queueLatestFinal('field-change');
    });
    els.modelSelect.addEventListener('change', () => {
      applyModelCaps();
      queueLatestFinal('model-change');
    });
    els.plotMode2dBtn.addEventListener('click', () => setPlotMode('2d'));
    els.plotMode3dBtn.addEventListener('click', () => setPlotMode('3d'));
    els.planeXzBtn.addEventListener('click', () => setPlane('xz'));
    els.planeTopBtn.addEventListener('click', () => setPlane('xy'));
    els.yIndexInput.addEventListener('change', () => {
      state.controls.y_index = sanitizeInt(els.yIndexInput.value, -1);
      els.yIndexInput.value = String(state.controls.y_index);
      syncControlOutputs();
      queueLatestFinal('y-index-change');
    });
    els.showContourCheckbox.addEventListener('change', () => {
      state.controls.show_t_contour = els.showContourCheckbox.checked;
      syncControlOutputs();
      queueLatestFinal('contour-toggle');
    });
    els.tContourInput.addEventListener('change', () => {
      state.controls.t_contour_k = Number(els.tContourInput.value || '1898');
      els.tContourInput.value = String(state.controls.t_contour_k);
      syncControlOutputs();
      queueLatestFinal('contour-level-change');
    });

    els.gtCompareCheckbox.addEventListener('change', () => {
      state.showGt = els.gtCompareCheckbox.checked;
      if (!state.showGt) state.selectedGtName = null;
      drawMaps(); // redraw both the P-V and H-P maps
      queueLatestFinal('gt-toggle'); // switches the result to/from the GT comparison
    });

    bindMapPointerEvents(els.pvMapCanvas, state.pvMapHolder, 'pvMapPointerActive', (raw) => {
      const b = state.config.bounds;
      applyPV(
        M.snapToStep(raw.p, b.p_watts.min, b.p_watts.step, b.p_watts.min, b.p_watts.max),
        M.snapToStep(raw.y, b.v_scan_mps.min, b.v_scan_mps.step, b.v_scan_mps.min, b.v_scan_mps.max),
      );
    });
    bindMapPointerEvents(els.hpMapCanvas, state.hpMapHolder, 'hpMapPointerActive', (raw) => {
      applyPHSelection(raw.p, raw.y);
    });
  }

  function setPlotMode(mode) {
    const nextMode = mode === '3d' ? '3d' : '2d';
    if (state.controls.plot_mode === nextMode) return;
    state.controls.plot_mode = nextMode;
    updatePlotModeButtons();
    updatePlaneButtons();
    updateViewModeUi();
    queueLatestFinal('plot-mode-change');
  }

  function updatePlotModeButtons() {
    const is3d = state.controls.plot_mode === '3d';
    C.setSegment([els.plotMode2dBtn, els.plotMode3dBtn], is3d ? els.plotMode3dBtn : els.plotMode2dBtn);
  }

  // Orientation: 'xz' cross-section (2D slice / 3D cut-away) or 'xy' top view.
  function setPlane(plane) {
    const next = plane === 'xy' ? 'xy' : 'xz';
    if (state.controls.plane === next) return;
    state.controls.plane = next;
    updatePlaneButtons();
    updateViewModeUi();
    queueLatestFinal('plane-change');
  }

  function updatePlaneButtons() {
    // Top view is available in both 2D and 3D on the steady page.
    const xz = state.controls.plane === 'xz';
    C.setSegment([els.planeXzBtn, els.planeTopBtn], xz ? els.planeXzBtn : els.planeTopBtn);
  }

  function updateViewModeUi() {
    const is3d = state.controls.plot_mode === '3d';
    const isTop = state.controls.plane === 'xy';
    // Y-index picks the XZ cross-section's y-slice (n/a in 3D or top view). The
    // melt-pool T contour is drawn on both 2D slices, so only 3D hides it.
    const controls = [
      [els.yIndexInput, els.yIndexLabel, 'Y Index', is3d || isTop, 'cross-section only'],
      [els.tContourInput, els.tContourLabel, 'T Contour (K)', is3d, '2D only'],
      [els.showContourCheckbox, els.showContourLabel, 'Show T contour', is3d, '2D only'],
    ];
    for (const [input, wrapper, baseLabel, hide, reason] of controls) {
      input.disabled = hide;
      wrapper.classList.toggle('control-disabled', hide);
      const labelSpan = wrapper.querySelector('span');
      if (labelSpan) labelSpan.textContent = hide ? `${baseLabel} (${reason})` : baseLabel;
    }
  }

  function onSliderInput(key, value) {
    state.controls[key] = value;
    state.selectedGtName = null;
    syncControlOutputs();
    drawMaps();
    queueLatestFinal('slider');
  }

  function onHSliderInput(value) {
    state.selectedGtName = null;
    applyPHSelection(state.controls.p_watts, value);
    queueLatestFinal('h-slider');
  }

  function applyPV(p, v) {
    state.controls.p_watts = p;
    state.controls.v_scan_mps = v;
    els.pSlider.value = String(p);
    els.vSlider.value = String(v);
    syncControlOutputs();
    drawMaps();
  }

  // (P, H*) -> snapped P and the V that gives that H* at P (within the feasible
  // H range for P, so V stays inside the slider bounds).
  function applyPHSelection(pRaw, hRaw) {
    const b = state.config.bounds;
    const p = M.snapToStep(Number(pRaw), b.p_watts.min, b.p_watts.step, b.p_watts.min, b.p_watts.max);
    const hRange = getFeasibleHRangeForP(p, b);
    const hReq = Number(hRaw);
    const h = clamp(Number.isFinite(hReq) ? hReq : hRange.min, hRange.min, hRange.max);
    const v = M.snapToStep(M.computeVFromHStar(p, h), b.v_scan_mps.min, b.v_scan_mps.step, b.v_scan_mps.min, b.v_scan_mps.max);
    applyPV(p, v);
  }

  // Both selection maps: scatter off = free pick/drag anywhere on the plot;
  // scatter on = GT dots are the only clickable targets (buttons), a click off
  // any dot and drags do nothing, and a hit snaps to that case's exact P/V so
  // the release renders its GT comparison. `applyRaw({p, y})` maps a pointer's
  // raw axis values onto the controls (y = V on the P-V map, H* on the H-P map).
  function bindMapPointerEvents(canvas, holder, activeKey, applyRaw) {
    const applyPointer = (event, opts) => {
      const raw = M.eventToAxes(event, canvas, holder, opts);
      if (!raw) return false;
      state.selectedGtName = null; // a free pick, not a GT point
      applyRaw(raw);
      return true;
    };
    const snapToGtHit = (event) => {
      if (!state.showGt || !state.gtCatalog.length) return false;
      const hit = M.hitTestGt(event, canvas, holder.bg, state.gtCatalog, { radiusPx: 12 });
      if (!hit) return false;
      applyPV(hit.p_watts, hit.v_scan_mps);
      state.selectedGtName = hit.name;
      drawMaps();
      return true;
    };
    const finish = (event, triggerFinal) => {
      state[activeKey] = false;
      try {
        if (typeof event.pointerId === 'number' && canvas.hasPointerCapture?.(event.pointerId)) {
          canvas.releasePointerCapture(event.pointerId);
        }
      } catch (_) { /* ignore pointer capture release errors */ }
      // GT toggle on: keep the dot snapped in pointerdown, don't overwrite it
      // with the release position (that reintroduced free picks).
      if (!state.showGt) applyPointer(event, { clampToPlot: true });
      if (triggerFinal) queueLatestFinal('map-release');
    };

    canvas.addEventListener('pointerdown', (event) => {
      const ok = snapToGtHit(event) || (!state.showGt && applyPointer(event, { clampToPlot: false }));
      if (!ok) return;
      state[activeKey] = true;
      try { canvas.setPointerCapture(event.pointerId); } catch (_) { /* may fail on some input types */ }
      queueLatestFinal('map-drag');
    });
    canvas.addEventListener('pointermove', (event) => {
      if (!state[activeKey] || state.showGt) return; // buttons-only: no free drag while scatter is on
      if (applyPointer(event, { clampToPlot: true })) queueLatestFinal('map-drag');
    });
    canvas.addEventListener('pointerup', (event) => { if (state[activeKey]) finish(event, true); });
    canvas.addEventListener('pointercancel', (event) => { if (state[activeKey]) finish(event, false); });
    canvas.addEventListener('lostpointercapture', () => { state[activeKey] = false; });
  }

  // Coalescing render queue: at most one request in flight; a request made
  // meanwhile is remembered (latest wins) and issued when the current one ends.
  function queueLatestFinal(reason) {
    state.pendingFinal = { reason: String(reason || 'final') };
    drainQueuedRequests();
  }

  function drainQueuedRequests() {
    if (state.inFlight || !state.pendingFinal) return;
    const pending = state.pendingFinal;
    state.pendingFinal = null;
    void issueInferRequest({ reason: pending.reason });
  }

  async function issueInferRequest(meta = {}) {
    if (!state.config) return;
    clearError();
    if (state.inFlight) { queueLatestFinal(meta.reason); return; }
    // With the GT toggle on, the render is a prediction-vs-ground-truth
    // comparison (its own endpoint).
    if (state.showGt) return issueGtCompare(meta);
    state.pendingFinal = null;

    const seq = ++state.latestIssuedSeq;
    const dims = state.config.render;
    const payload = {
      client_request_id: C.makeRequestId(),
      client_seq: seq,
      mode: 'final',
      model: els.modelSelect.value || 'fno_cal',
      inputs: {
        p_watts: Number(state.controls.p_watts),
        v_scan_mps: Number(state.controls.v_scan_mps),
      },
      view: {
        field: state.controls.field,
        plot_mode: state.controls.plot_mode,
        plane: state.controls.plane,
        y_index: sanitizeInt(state.controls.y_index, -1),
        show_t_contour: state.controls.plot_mode === '2d' ? Boolean(state.controls.show_t_contour) : false,
        t_contour_k: Number(state.controls.t_contour_k),
      },
      render: { width_px: Number(dims.width_px), height_px: Number(dims.height_px) },
    };

    const encoded = new TextEncoder().encode(JSON.stringify(payload));
    if (encoded.byteLength > state.config.requestMaxBodyBytes) {
      showError(`Request body too large on client (${encoded.byteLength} bytes)`);
      setStatus('error', 'Request too large');
      return;
    }

    const controller = new AbortController();
    state.inFlight = { controller, seq, startedAt: performance.now() };
    const timeoutMs = state.config.finalTimeoutMs;
    const timeoutId = window.setTimeout(() => controller.abort('timeout'), timeoutMs);
    setStatus('loading', `request #${seq} (${meta.reason || 'manual'})`);

    try {
      const response = await fetch(`${state.config.backendBaseUrl || ''}/infer`, {
        method: 'POST', cache: 'no-store', credentials: 'omit',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: controller.signal,
      });
      const text = await response.text();
      let body = null;
      try { body = text ? JSON.parse(text) : null; } catch (_) { body = null; }

      if (seq < state.latestIssuedSeq) return;
      if (!response.ok) {
        C.handleErrorResponse(els, state, response.status, body, () => queueLatestFinal('busy-retry'));
        return;
      }
      if (!body || body.status !== 'ok') {
        showError('Backend returned non-JSON or invalid JSON success payload.');
        setStatus('error', 'invalid response');
        return;
      }
      if (Number(body.client_seq) !== seq || seq !== state.latestIssuedSeq) return;

      state.latestRenderedSeq = seq;
      state.lastResponse = body;
      state.busyRetries = 0;
      renderSuccess(body);
      const elapsed = Math.round(performance.now() - state.inFlight.startedAt);
      els.latencyText.textContent = `render ${elapsed} ms (server total ${Math.round(body.timing_ms?.total || 0)} ms)`;
      C.showInferTiming(els.inferTiming, body, elapsed);
      setStatus('ok', 'rendered');
    } catch (error) {
      if (controller.signal.aborted) {
        if (String(controller.signal.reason || 'aborted') === 'timeout') {
          setStatus('error', 'timeout');
          showError(`Request timed out after ${timeoutMs} ms`);
        }
        return;
      }
      setStatus('error', 'network error');
      showError(`Request failed: ${String(error)}`);
    } finally {
      clearTimeout(timeoutId);
      if (state.inFlight?.seq === seq) state.inFlight = null;
      drainQueuedRequests();
    }
  }

  async function issueGtCompare(meta = {}) {
    const seq = ++state.latestIssuedSeq;
    const dims = state.config.render;
    const payload = {
      client_request_id: C.makeRequestId(),
      model: els.modelSelect.value || 'fno_cal',
      inputs: {
        p_watts: Number(state.controls.p_watts),
        v_scan_mps: Number(state.controls.v_scan_mps),
      },
      field: state.controls.field,
      y_index: sanitizeInt(state.controls.y_index, -1),
      // In 3D the comparison is stacked iso-surface blocks (GT above prediction);
      // plane selects cross-section vs top view — matches the plain 3D view.
      plot_mode: state.controls.plot_mode,
      plane: state.controls.plane,
      render: { width_px: Number(dims.width_px), height_px: Number(dims.height_px) },
    };

    const controller = new AbortController();
    state.inFlight = { controller, seq, startedAt: performance.now() };
    const timeoutId = window.setTimeout(() => controller.abort('timeout'), state.config.finalTimeoutMs);
    setStatus('loading', `GT compare #${seq} (${meta.reason || 'manual'})`);

    try {
      const response = await fetch(`${state.config.backendBaseUrl || ''}/infer/gt_compare`, {
        method: 'POST', cache: 'no-store', credentials: 'omit',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: controller.signal,
      });
      const text = await response.text();
      let body = null;
      try { body = text ? JSON.parse(text) : null; } catch (_) { body = null; }
      if (seq < state.latestIssuedSeq) return;
      if (!response.ok) { C.handleErrorResponse(els, state, response.status, body, () => queueLatestFinal('busy-retry')); return; }
      if (!body || body.status !== 'ok') { showError('Backend returned an invalid GT-compare payload.'); setStatus('error', 'invalid response'); return; }
      state.latestRenderedSeq = seq;
      state.busyRetries = 0;
      renderGtCompareSuccess(body);
      const elapsed = Math.round(performance.now() - state.inFlight.startedAt);
      els.latencyText.textContent = `gt_compare ${elapsed} ms (server total ${Math.round(body.timing_ms?.total || 0)} ms)`;
      C.showInferTiming(els.inferTiming, body, elapsed);
      setStatus('ok', body.result_meta?.gt_available ? 'GT compared' : 'prediction (no GT at this P/V)');
    } catch (error) {
      if (controller.signal.aborted) {
        if (String(controller.signal.reason || '') === 'timeout') { setStatus('error', 'gt-compare timeout'); showError('GT compare timed out'); }
        return;
      }
      setStatus('error', 'network error');
      showError(`GT compare failed: ${String(error)}`);
    } finally {
      clearTimeout(timeoutId);
      if (state.inFlight?.seq === seq) state.inFlight = null;
      drainQueuedRequests();
    }
  }

  function renderGtCompareSuccess(body) {
    const plot = body.plot || {};
    if (plot.png_base64) {
      els.resultImage.src = `data:image/png;base64,${plot.png_base64}`;
      els.resultImage.classList.remove('hidden');
      els.resultPlaceholder.classList.add('hidden');
    }
    const rm = body.result_meta || {};
    els.resultModeTag.textContent = rm.gt_available ? `GT vs pred${rm.gt_split ? ` (${rm.gt_split})` : ''}` : 'prediction';
    els.resultModeTag.className = 'badge ok';
    els.resultInfoText.textContent = rm.gt_available
      ? `${rm.field || '-'} | GT ${rm.gt_split || ''} P=${fmt(rm.gt_p_watts, 4)} W, V=${fmt(rm.gt_v_scan_mps, 4)} m/s | grid=${(rm.grid_shape_xyz || []).join('x') || '-'}`
      : `${rm.field || '-'} | no ground truth near this P/V | grid=${(rm.grid_shape_xyz || []).join('x') || '-'}`;
    C.setGtMetrics(els.gtMetrics, rm.gt_available && rm.gt_mae != null
      ? { field: rm.field, mae: rm.gt_mae, rmse: rm.gt_rmse } : null);
    const c = body.clamped_inputs || {};
    els.metaP.textContent = fmt(c.p_watts, 4);
    els.metaV.textContent = fmt(c.v_scan_mps, 5);
    els.metaHModel.textContent = fmt(computeHModel(c.p_watts, c.v_scan_mps), 6);
    els.metaHNorm.textContent = fmt(modelHNorm(c.p_watts, c.v_scan_mps), 6);

    state.selectedGtName = null;
    if (rm.gt_available) {
      const match = state.gtCatalog.find((g) => Math.abs(g.p_watts - rm.gt_p_watts) < 1 && Math.abs(g.v_scan_mps - rm.gt_v_scan_mps) < 1e-3);
      state.selectedGtName = match ? match.name : null;
    }
    drawMaps();
  }

  function renderSuccess(body) {
    const plot = body.plot || {};
    if (plot.mime_type === 'image/png' && plot.png_base64) {
      els.resultImage.src = `data:image/png;base64,${plot.png_base64}`;
      els.resultImage.classList.remove('hidden');
      els.resultPlaceholder.classList.add('hidden');
    }
    const meta = body.result_meta || {};
    const plotMode = String(meta.plot_mode_used || '2d').toUpperCase();
    els.resultModeTag.textContent = `${body.model || els.modelSelect.value} / ${plotMode}`;
    els.resultModeTag.className = 'badge ok';
    const yText = meta.plot_mode_used === '3d' ? 'surface' : `y=${meta.y_index_used ?? '-'}`;
    els.resultInfoText.textContent = `${plotMode} | ${meta.field || '-'} | ${yText} | grid=${(meta.grid_shape_xyz || []).join('x') || '-'}`;
    C.setGtMetrics(els.gtMetrics, null);

    const c = body.clamped_inputs || {};
    const d = body.derived || {};
    els.metaP.textContent = fmt(c.p_watts, 4);
    els.metaV.textContent = fmt(c.v_scan_mps, 5);
    els.metaHModel.textContent = fmt(d.h_model, 6);
    els.metaHNorm.textContent = fmt(modelHNorm(c.p_watts, c.v_scan_mps), 6);
  }

  function syncControlOutputs() {
    const p = clampToBounds(state.controls.p_watts, state.config?.bounds?.p_watts);
    const v = clampToBounds(state.controls.v_scan_mps, state.config?.bounds?.v_scan_mps);
    const h = computeHStar(p, v);
    state.controls.p_watts = p;
    state.controls.v_scan_mps = v;
    state.controls.h_norm = h;
    els.pValue.textContent = `${fmt(p, 3)} W`;
    els.vValue.textContent = `${fmt(v, 4)} m/s`;
    els.hValue.textContent = fmt(h, 5);
    if (state.hNormBounds) {
      els.hSlider.value = String(clamp(Number(h), Number(state.hNormBounds.min), Number(state.hNormBounds.max)));
    }
    els.metaP.textContent = fmt(p, 4);
    els.metaV.textContent = fmt(v, 5);
    els.metaHModel.textContent = fmt(computeHModel(p, v), 6);
    els.metaHStar.textContent = fmt(h, 6);
    els.metaHNorm.textContent = fmt(modelHNorm(p, v), 6);
    updatePlotModeButtons();
    updateViewModeUi();
  }

  // The model's own normalised enthalpy input: H_model over the denominator the
  // model was trained with (meta.json physics.h_norm_denom via /models). Computed
  // here, from the controls, so the card is never stale: it used to be read off
  // /infer responses only, and a GT-compare response carries no derived block.
  function modelHNorm(p, v) {
    const ph = ((state.modelsById || {})[els.modelSelect.value] || {}).physics || null;
    const denom = ph && Number(ph.h_norm_denom);
    return denom > 0 ? computeHModel(p, v) / denom : NaN;
  }

  function drawMaps() {
    if (!state.config) return;
    const gt = { gtPoints: state.gtCatalog, showGt: state.showGt, selectedName: state.selectedGtName };
    M.draw(els.pvMapCanvas, state.pvMapHolder, { p: state.controls.p_watts, v: state.controls.v_scan_mps, ...gt });
    M.draw(els.hpMapCanvas, state.hpMapHolder, { p: state.controls.p_watts, h: computeHStar(state.controls.p_watts, state.controls.v_scan_mps), ...gt });
  }

  function buildHNormSliderBounds(bounds) {
    const base = M.vizConstants();
    const feasible = getGlobalFeasibleHRange(bounds);
    const min = Math.min(0.75 * base.A_corr, feasible.min);
    const max = Math.max(2.85 * base.A_corr, feasible.max);
    const p0 = Number(bounds.p_watts.initial ?? bounds.p_watts.min);
    const v0 = Number(bounds.v_scan_mps.initial ?? bounds.v_scan_mps.min);
    const initial = clamp(computeHStar(p0, v0), min, max);
    return { min: Number(min.toFixed(4)), max: Number(max.toFixed(4)), step: 0.001, initial: Number(initial.toFixed(4)) };
  }

  function getGlobalFeasibleHRange(bounds) {
    const minH = computeHStar(bounds.p_watts.min, bounds.v_scan_mps.max);
    const maxH = computeHStar(bounds.p_watts.max, bounds.v_scan_mps.min);
    return { min: Math.min(minH, maxH), max: Math.max(minH, maxH) };
  }

  function getFeasibleHRangeForP(p, bounds) {
    const pClamped = clamp(Number(p), Number(bounds.p_watts.min), Number(bounds.p_watts.max));
    const hAtVMax = computeHStar(pClamped, Number(bounds.v_scan_mps.max));
    const hAtVMin = computeHStar(pClamped, Number(bounds.v_scan_mps.min));
    return { min: Math.min(hAtVMax, hAtVMin), max: Math.max(hAtVMax, hAtVMin) };
  }

  function computeHModel(p, v) {
    return Number(p) / Math.sqrt(Math.max(Math.abs(Number(v)), 1e-12));
  }

  function clampToBounds(value, cfg) {
    if (!cfg) return Number(value);
    return clamp(Number(value), Number(cfg.min), Number(cfg.max));
  }

  function sanitizeInt(value, fallback) {
    const n = Number.parseInt(String(value), 10);
    return Number.isFinite(n) ? n : fallback;
  }

  function fmt(value, digits = 4) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 'n/a';
    const d = Math.trunc(Number(digits));
    if (!Number.isFinite(d) || d <= 0) return String(Math.round(n));
    return n.toLocaleString(undefined, { maximumSignificantDigits: clamp(d, 1, 21), useGrouping: false });
  }

  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

  window.addEventListener('DOMContentLoaded', init);
})();
