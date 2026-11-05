(() => {
  const state = {
    config: null,
    baseUrl: '',
    bounds: null,
    p_watts: 110,
    v_scan_mps: 0.34,
    view: '3d', // default overall view is the 3D surface (matches the main page)
    frameSets: {}, // field -> Promise<[{time_s, img}]>
    frames3d: null,
    runKey: '',
    inFlight: false,
    busyRetries: 0, // 429 auto-retry counter (backend is single-GPU; a cold model load holds the lock)
    gtPoints: [], // /gt_catalog?dataset=straight P-V points
    showGt: false, // default: arbitrary P-V; toggle on to pick GT cases
    selectedGt: null, // matched GT case name, or null
    plane: 'xy', // '2D slice' plane: 'xy' top view | 'xz' cross-section
    mapHolder: {}, // PVMap background cache holder
    pvPointerActive: false,
    rerunPending: false, // a P/V/control change arrived mid-render; rerun once it finishes
    dispatchedKey: '', // run key currently shown or in flight (auto-run dedup)
    lastModel: '',
    lastMeta: null,
    lastMeta3d: null,
    lastGtAvailable: false,
    lastSplit: null,
    gtMetrics: {}, // key `2d:<field>` or `3d` -> {field, mae, rmse} | null
  };

  const els = {};
  const C = window.Common;
  const setStatus = (kind, text) => C.setStatus(els, kind, text);
  const showError = (message) => C.showError(els, message);
  const clearError = () => C.clearError(els);
  let player = null; // Common.FramePlayer: displayed frames + slider + Play
  let runner = null; // Common.TimeRunner: coalesced auto-run + render cache + 2D/3D swap

  const ELEMENT_IDS = [
    'statusBadge', 'latencyText', 'pvMapCanvas', 'showGtCheckbox',
    'pSlider', 'vSlider', 'pValue', 'vValue',
    'modelSelect', 'fieldSelect', 'zIndexSelect',
    'runBtn', 'view2dBtn', 'view3dBtn', 'planeTopBtn', 'planeXzBtn',
    'resultImage', 'resultPlaceholder', 'errorBox', 'resultModeTag', 'resultInfoText',
    'gtMetrics', 'inferTiming', 'playBtn', 'frameSlider', 'frameTimeValue',
    'metaP', 'metaV', 'metaGt', 'metaLaserOff', 'metaTiming',
  ];

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
    state.bounds = state.config.bounds;
    state.p_watts = Number(state.bounds.p_watts.initial ?? 110);
    state.v_scan_mps = Number(state.bounds.v_scan_mps.initial ?? 0.34);
    runner = C.TimeRunner({
      els, state, player,
      infer: `${state.baseUrl}/infer_straight`, render3d: `${state.baseUrl}/infer_straight/render3d`, label: 'straight',
      buildPayload, currentRunKey, runKeyFor, renderSuccess, updateResultInfo, applyViewUi,
    });

    setupRange(els.pSlider, state.bounds.p_watts);
    setupRange(els.vSlider, state.bounds.v_scan_mps);
    els.pSlider.value = String(state.p_watts);
    els.vSlider.value = String(state.v_scan_mps);
    els.zIndexSelect.innerHTML = '<option value="-1">surface (auto)</option>'; // z indices come from the first response's grid
    bindControls();
    bindMapPointerEvents();
    syncOutputs();
    drawMap();
    setStatus('idle', 'Ready');
    clearError();

    await loadModels();
    await loadGtCatalog();
    applyViewUi(); // sync buttons/disabled state to the default view before the first render
    void runner.run(true);
  }

  async function loadModels() {
    state.modelsById = await C.loadModels(state.baseUrl, 'straight', els.modelSelect);
    applyModelCaps();
  }

  // P/V envelope for the selected model: a "physics" block in its /models
  // entry narrows the P/V sliders to the model's trained envelope.
  function applyModelCaps() {
    const info = (state.modelsById || {})[els.modelSelect.value] || {};
    const bounds = state.config ? state.config.bounds : null;
    const ph = info.physics || null;
    const clampSlider = (input, cfg, rng, setter) => {
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
      const cur = Math.min(max, Math.max(min, Number(input.value)));
      input.value = String(cur);
      setter(cur);
    };
    if (bounds) {
      clampSlider(els.pSlider, bounds.p_watts, ph && ph.p_watts, (v) => { state.p_watts = v; });
      clampSlider(els.vSlider, bounds.v_scan_mps, ph && ph.v_scan_mps, (v) => { state.v_scan_mps = v; });
      syncOutputs();
      drawMap();
    }
  }

  function setupRange(input, cfg) {
    input.min = String(cfg.min);
    input.max = String(cfg.max);
    input.step = String(cfg.step);
  }

  function bindControls() {
    // Live auto-run while dragging (like the steady page): each move requests a
    // run; the runner coalesces to at-most-one-in-flight so the single-GPU
    // backend never piles up. 'change' (release) is the coalesced final.
    els.pSlider.addEventListener('input', () => { applyPV(Number(els.pSlider.value), state.v_scan_mps, null); runner.requestRun(); });
    els.vSlider.addEventListener('input', () => { applyPV(state.p_watts, Number(els.vSlider.value), null); runner.requestRun(); });
    els.pSlider.addEventListener('change', () => { runner.requestRun(); });
    els.vSlider.addEventListener('change', () => { runner.requestRun(); });

    els.showGtCheckbox.addEventListener('change', () => {
      state.showGt = els.showGtCheckbox.checked;
      drawMap();
    });

    els.fieldSelect.addEventListener('change', () => runner.onFieldChange());
    els.modelSelect.addEventListener('change', () => { applyModelCaps(); runner.requestRun(); });
    els.zIndexSelect.addEventListener('change', () => { runner.requestRun(); });

    els.runBtn.addEventListener('click', () => { runner.requestRun({ force: true }); });
    els.view2dBtn.addEventListener('click', () => { void runner.setView('2d'); });
    els.view3dBtn.addEventListener('click', () => { void runner.setView('3d'); });
    els.planeTopBtn.addEventListener('click', () => setPlane('xy'));
    els.planeXzBtn.addEventListener('click', () => setPlane('xz'));
  }

  function applyPV(p, v, gtName) {
    state.p_watts = clamp(Number(p), state.bounds.p_watts.min, state.bounds.p_watts.max);
    state.v_scan_mps = clamp(Number(v), state.bounds.v_scan_mps.min, state.bounds.v_scan_mps.max);
    state.selectedGt = gtName;
    els.pSlider.value = String(state.p_watts);
    els.vSlider.value = String(state.v_scan_mps);
    syncOutputs();
    drawMap();
  }

  function syncOutputs() {
    els.pValue.textContent = `${state.p_watts.toFixed(0)} W`;
    els.vValue.textContent = `${state.v_scan_mps.toFixed(3)} m/s`;
    els.metaP.textContent = `${state.p_watts.toFixed(0)} W`;
    els.metaV.textContent = `${state.v_scan_mps.toFixed(3)} m/s`;
  }

  function drawMap() {
    if (!window.PVMap) return;
    window.PVMap.draw(els.pvMapCanvas, state.mapHolder, {
      p: state.p_watts, v: state.v_scan_mps, bounds: state.bounds,
      gtPoints: state.gtPoints, showGt: state.showGt, selectedName: state.selectedGt,
    });
  }

  function bindMapPointerEvents() {
    const canvas = els.pvMapCanvas;
    const setFromPointer = (event, drag) => {
      if (state.showGt) {
        // Scatter on: GT dots are the only clickable targets (buttons). Empty
        // space and drags do nothing — you cannot pick an arbitrary P/V.
        if (drag) return false;
        const hit = window.PVMap.hitTestGt(event, canvas, state.mapHolder.bg, state.gtPoints, { radiusPx: 12 });
        if (!hit) return false;
        applyPV(hit.p_watts, hit.v_scan_mps, hit.name);
        return true;
      }
      const pv = window.PVMap.eventToPV(event, canvas, state.mapHolder, state.bounds, { clampToPlot: drag });
      if (!pv) return false;
      applyPV(pv.p, pv.v, null);
      return true;
    };

    canvas.addEventListener('pointerdown', (event) => {
      if (!setFromPointer(event, false)) return;
      state.pvPointerActive = true;
      try { canvas.setPointerCapture(event.pointerId); } catch (_) { /* ignore */ }
    });
    canvas.addEventListener('pointermove', (event) => {
      if (!state.pvPointerActive) return;
      if (setFromPointer(event, true)) runner.requestRun();
    });
    const end = (event) => {
      if (!state.pvPointerActive) return;
      state.pvPointerActive = false;
      try { if (canvas.hasPointerCapture?.(event.pointerId)) canvas.releasePointerCapture(event.pointerId); } catch (_) { /* ignore */ }
      runner.requestRun();
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('lostpointercapture', () => { state.pvPointerActive = false; });
  }

  async function loadGtCatalog() {
    try {
      const resp = await fetch(`${state.baseUrl}/gt_catalog?dataset=straight`, { cache: 'no-store', credentials: 'omit' });
      const body = await resp.json();
      if (!resp.ok || body.status !== 'ok' || !Array.isArray(body.entries)) {
        throw new Error(body?.error?.message || `HTTP ${resp.status}`);
      }
      state.gtPoints = body.entries;
      drawMap();
    } catch (error) {
      setStatus('idle', `GT points unavailable: ${String(error)}`);
    }
    C.setGtAvailable(els.showGtCheckbox, Array.isArray(state.gtPoints) && state.gtPoints.length > 0);
  }

  // --- request plumbing (Common.TimeRunner hooks) ---
  function buildPayload(is3d) {
    const pathDemo = state.config.pathDemo || {};
    const payload = {
      client_request_id: C.makeRequestId(),
      model: els.modelSelect.value,
      inputs: { p_watts: state.p_watts, v_scan_mps: state.v_scan_mps },
      render: {
        width_px: Number(pathDemo.render?.width_px ?? 640),
        height_px: Number(pathDemo.render?.height_px ?? 520),
      },
    };
    if (is3d) payload.plane = state.plane; // top view vs cut-away cross-section (3D)
    else payload.view = { field: els.fieldSelect.value, z_index: Number.parseInt(els.zIndexSelect.value, 10), plane: state.plane };
    return payload;
  }

  // What identifies the currently requested render — used to skip redundant runs
  // and to detect that a mid-flight move actually changed the selection.
  function currentRunKey() {
    return JSON.stringify([
      state.view, els.modelSelect.value, state.plane,
      state.view === '2d' ? els.fieldSelect.value : null,
      state.view === '2d' ? els.zIndexSelect.value : null,
      state.p_watts.toFixed(3), state.v_scan_mps.toFixed(4),
    ]);
  }

  function runKeyFor(body) {
    const sm = body.straight_meta || {};
    // gt_name distinguishes a dataset case from hand-typed values that happen
    // to match it — same numbers, but only one of them carries a GT panel.
    return JSON.stringify([
      body.model, Number(sm.p_watts).toFixed(3), Number(sm.v_scan_mps).toFixed(4), sm.gt_name || '',
    ]);
  }

  function applyViewUi() {
    const is3d = state.view === '3d';
    C.setSegment([els.view2dBtn, els.view3dBtn], is3d ? els.view3dBtn : els.view2dBtn);
    els.fieldSelect.disabled = is3d;
    // Top view / Cross section applies to both 2D and 3D; the z-slice is only
    // meaningful for the 2D top view.
    els.zIndexSelect.disabled = is3d || state.plane === 'xz';
  }

  function setPlane(plane) {
    if (state.plane === plane) return;
    state.plane = plane;
    applyPlaneUi();
    // Route through requestRun like the other controls: coalesces an in-flight
    // render (rerunPending) instead of dropping the click, and consults the
    // render cache so flipping plane back and forth doesn't re-infer.
    runner.requestRun();
  }

  function applyPlaneUi() {
    const xz = state.plane === 'xz';
    C.setSegment([els.planeTopBtn, els.planeXzBtn], xz ? els.planeXzBtn : els.planeTopBtn);
    // z-slice picks a 2D top-view plane; cross-section and 3D don't use it.
    els.zIndexSelect.disabled = xz || state.view === '3d';
  }

  function updateResultInfo() {
    if (state.view === '3d') {
      const meta = state.lastMeta3d || {};
      els.resultModeTag.textContent = `${state.lastModel} / 3D`;
      els.resultInfoText.textContent = `3D surface | grid=${(meta.grid_shape_xyz || []).join('x') || '-'}`;
    } else {
      const meta = state.lastMeta || {};
      const gt = state.lastGtAvailable ? ` | GT (${state.lastSplit})` : ' | no GT';
      els.resultModeTag.textContent = `${state.lastModel} / ${meta.panel || 'single'}`;
      els.resultInfoText.textContent = `${els.fieldSelect.value} | z=${meta.z_index_used ?? '-'} | grid=${(meta.grid_shape_xyz || []).join('x') || '-'}${gt}`;
    }
    const key = state.view === '3d' ? '3d' : `2d:${els.fieldSelect.value}`;
    C.setGtMetrics(els.gtMetrics, state.gtMetrics[key]);
    els.resultModeTag.className = 'badge ok';
  }

  function renderSuccess(body) {
    const meta = body.result_meta || {};
    const sm = body.straight_meta || {};
    if (meta.panel === '3d') state.lastMeta3d = meta; else state.lastMeta = meta;
    state.lastModel = body.model;
    state.lastGtAvailable = Boolean(sm.gt_available);
    state.lastSplit = sm.split || null;
    C.setZIndexOptions(els.zIndexSelect, (meta.grid_shape_xyz || [])[2]);

    const is3d = meta.panel === '3d';
    const mField = is3d ? 'T' : (meta.field || 'T');
    const key = is3d ? '3d' : `2d:${mField}`;
    state.gtMetrics[key] = sm.gt_available && sm.gt_mae != null
      ? { field: mField, mae: sm.gt_mae, rmse: sm.gt_rmse } : null;

    // Reflect a server-side snap to a matched GT case's exact P/V. For free P/V
    // there is no snap, and rewriting the sliders here would fight a live drag
    // (and drop the user's latest selection when a render finishes) — leave it.
    if (sm.gt_available && Number.isFinite(sm.p_watts) && Number.isFinite(sm.v_scan_mps)) {
      applyPV(sm.p_watts, sm.v_scan_mps, sm.gt_name);
    }
    updateResultInfo();

    // 34 of the 123 case names carry a legacy train_/val_/test_ prefix that
    // contradicts the served split; show the name without it, the split beside.
    els.metaGt.textContent = sm.gt_available ? `${String(sm.gt_name).replace(/^(train|val|test)_/, '')} (${sm.split})` : 'none (free P/V)';
    els.metaLaserOff.textContent = Number.isFinite(sm.laser_off_time_s) ? `${(sm.laser_off_time_s * 1000).toFixed(3)} ms` : '-';
    const t = body.timing_ms || {};
    els.metaTiming.textContent = `infer ${Math.round(t.infer || 0)} ms, render ${Math.round(t.render || 0)} ms`;
  }

  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

  window.addEventListener('DOMContentLoaded', init);
})();
