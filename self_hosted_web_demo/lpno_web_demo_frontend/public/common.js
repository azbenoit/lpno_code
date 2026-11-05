// Shared page plumbing for the three demo pages (config, model list, status
// line, error handling, GT/timing captions, frame player). Loaded before the
// page script; exposes window.Common. No page logic lives here.
(() => {
  // config.json + optional untracked config.local.json (backendBaseUrl for
  // direct-backend dev). The override is a SHALLOW merge: a top-level key in
  // config.local.json replaces the whole key from config.json.
  async function loadConfig() {
    const resp = await fetch('config.json', { cache: 'no-store' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status} while fetching config.json`);
    const cfg = await resp.json();
    try {
      const local = await fetch('config.local.json', { cache: 'no-store' });
      if (local.ok) Object.assign(cfg, await local.json());
    } catch (_) { /* no local override */ }
    return cfg;
  }

  // Fill the model <select> from GET /models?page=<page>; on any failure the
  // hardcoded fallback <option>s in the HTML stay. Returns {id: entry}.
  async function loadModels(baseUrl, page, selectEl) {
    try {
      const resp = await fetch(`${baseUrl || ''}/models?page=${page}`, { cache: 'no-store', credentials: 'omit' });
      const body = await resp.json();
      if (!resp.ok || body.status !== 'ok' || !Array.isArray(body.models) || !body.models.length) return {};
      selectEl.replaceChildren(...body.models.map((m) => {
        const o = document.createElement('option');
        o.value = m.id; o.textContent = m.label; o.selected = Boolean(m.default);
        return o;
      }));
      if (!selectEl.value) selectEl.selectedIndex = 0;
      return Object.fromEntries(body.models.map((m) => [m.id, m]));
    } catch (_) {
      return {};
    }
  }

  function cacheElements(ids) {
    const els = {};
    for (const id of ids) els[id] = document.getElementById(id);
    return els;
  }

  function setStatus(els, kind, text) {
    els.statusBadge.textContent = kind;
    els.statusBadge.className = `badge ${kind}`;
    if (text) els.latencyText.textContent = text;
  }
  function showError(els, message) { els.errorBox.textContent = message; els.errorBox.classList.remove('hidden'); }
  function clearError(els) { els.errorBox.textContent = ''; els.errorBox.classList.add('hidden'); }

  function makeRequestId() {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    return `req-${Date.now()}-${Math.floor(Math.random() * 1e9)}`;
  }

  // Segmented toggle: one active button among `buttons`.
  function setSegment(buttons, active) {
    for (const b of buttons) {
      if (!b) continue;
      b.classList.toggle('is-active', b === active);
      b.setAttribute('aria-pressed', String(b === active));
    }
  }

  // Caption under a GT comparison: prediction-vs-GT error metrics.
  function setGtMetrics(el, m) {
    if (!el) return;
    if (!m || m.mae == null || !Number.isFinite(m.mae)) {
      el.classList.add('hidden');
      el.textContent = '';
      return;
    }
    const unit = m.field === 'T' ? ' K' : '';
    const dp = m.field === 'T' ? 1 : 4;
    el.innerHTML =
      `Error vs FLOW-3D ground truth &mdash; ` +
      `MAE <strong>${m.mae.toFixed(dp)}${unit}</strong> &middot; ` +
      `RMSE <strong>${m.rmse.toFixed(dp)}${unit}</strong>`;
    el.classList.remove('hidden');
  }

  // Raw timing line: model inference vs server render vs network/browser, so a
  // slow response can be attributed to the right stage. elapsedMs=null marks a
  // browser-cache replay (no request was made).
  function showInferTiming(el, body, elapsedMs) {
    if (!el) return;
    const t = body?.timing_ms;
    if (!t) { el.classList.add('hidden'); el.textContent = ''; return; }
    const fmtDur = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);
    const steps = Array.isArray(body.times_s) ? body.times_s.length : 0;
    const perStep = steps > 1 ? ` (${steps} steps &middot; ${fmtDur((t.infer || 0) / steps)}/step)` : '';
    const tail = elapsedMs == null
      ? 'replayed from browser cache'
      : `network + browser <strong>${fmtDur(Math.max(0, elapsedMs - (t.total || 0)))}</strong>`;
    el.innerHTML =
      `Model inference <strong>${fmtDur(t.infer || 0)}</strong>${perStep} &middot; ` +
      `server render <strong>${fmtDur(t.render || 0)}</strong> &middot; ${tail}`;
    el.classList.remove('hidden');
  }

  // Non-2xx handling shared by every page. 429: the backend is single-GPU, so a
  // busy answer is usually a cold model load or another tab's run — retry up
  // to 60 times (>= 400 ms apart) via `retry()`; `state.busyRetries` is the
  // counter (the page resets it on a user-initiated run or a success).
  function handleErrorResponse(els, state, status, body, retry) {
    const code = body?.error?.code || 'unknown';
    const message = body?.error?.message || `HTTP ${status}`;
    if (status === 429) {
      setStatus(els, 'busy', code === 'busy' ? 'busy (429)' : 'rate-limited');
      const retryAfter = Math.max(400, Number(body?.error?.retry_after_ms ?? 400));
      if (state.busyRetries < 60) {
        state.busyRetries += 1;
        showError(els, `Server busy — retrying (${state.busyRetries})...\n${message}`);
        window.setTimeout(retry, retryAfter);
      } else {
        showError(els, `Server busy (${status})\n${message}\nTry again in a moment.`);
      }
      return;
    }
    if (status === 504) { setStatus(els, 'error', 'timeout'); showError(els, `Backend timeout (504)\n${message}`); return; }
    if (status === 503) { setStatus(els, 'error', 'not ready'); showError(els, `Backend not ready (503)\n${message}`); return; }
    setStatus(els, 'error', `failed (${status})`);
    showError(els, `Request failed (${status})\n${message}\n${body ? JSON.stringify(body, null, 2) : ''}`);
  }

  // Frame player for the time pages: owns the displayed frame list, the slider
  // and the Play button. `frames` is [{ time_s, img }].
  function FramePlayer(els, { intervalMs = 250 } = {}) {
    const p = { frames: [], index: 0, timer: null };
    p.set = (index) => {
      if (!p.frames.length) return;
      const i = Math.min(p.frames.length - 1, Math.max(0, Math.round(index)));
      p.index = i;
      const frame = p.frames[i];
      els.resultImage.src = frame.img.src;
      els.resultImage.classList.remove('hidden'); // a page may hide it when it clears the result
      els.resultPlaceholder.classList.add('hidden');
      els.frameSlider.value = String(i);
      els.frameTimeValue.textContent = `${(frame.time_s * 1000).toFixed(3)} ms`;
    };
    p.show = (frames, index) => {
      p.frames = frames;
      els.frameSlider.max = String(frames.length - 1);
      els.frameSlider.disabled = false;
      els.playBtn.disabled = false;
      // Open mid-sweep: frame 0 is the initial condition (a uniform block) where
      // the models paint a small hot spot that is not there; the melt pool a
      // visitor came to see is a few frames in. Play continues from the shown frame.
      p.set(index ?? Math.floor(frames.length / 2));
    };
    p.stop = () => {
      if (p.timer) { window.clearInterval(p.timer); p.timer = null; }
      els.playBtn.textContent = 'Play';
    };
    els.frameSlider.addEventListener('input', () => { p.stop(); p.set(Number(els.frameSlider.value)); });
    els.playBtn.addEventListener('click', () => {
      if (p.timer) { p.stop(); return; }
      if (!p.frames.length) return;
      els.playBtn.textContent = 'Pause';
      p.timer = window.setInterval(() => p.set((p.index + 1) % p.frames.length), intervalMs);
    });
    return p;
  }

  // Decode one field's frames; resolves to [{ time_s, img }].
  function preloadFrames(timesS, pngList) {
    return Promise.all(pngList.map((png, i) => new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve({ time_s: Number(timesS[i]), img });
      img.onerror = () => reject(new Error('Failed to decode a frame PNG'));
      img.src = `data:image/png;base64,${png}`;
    })));
  }

  // Z-slice <select>: "surface (auto)" + 0..nz-1. nz comes from a response's
  // grid_shape_xyz (never a client constant: the straight FNO/GAOT grid is
  // 90x40x40, its RIGNO 90x40x30, the path grid 75x50x40); rebuilt only when
  // the count changes, keeping the selection when it is still in range.
  function setZIndexOptions(selectEl, nz) {
    const n = Number(nz);
    if (!Number.isFinite(n) || n <= 0 || selectEl.dataset?.nz === String(n)) return;
    const prev = selectEl.value;
    selectEl.innerHTML = ['<option value="-1">surface (auto)</option>']
      .concat(Array.from({ length: n }, (_, i) => `<option value="${i}">${i}</option>`)).join('');
    selectEl.value = prev !== '' && Number(prev) < n ? prev : '-1';
    if (selectEl.dataset) selectEl.dataset.nz = String(n);
  }

  // Request state machine shared by the two time pages (straight track,
  // free-form path): coalesced auto-run (at most one request in flight, a
  // change mid-render flags ONE rerun with the latest selection), an LRU
  // render cache keyed by the page's run identity, lazy per-field 2D sets and
  // one 3D set per loaded run, and the 2D/3D swap. `page` supplies:
  //   els, state ({config, view, inFlight, rerunPending, dispatchedKey, runKey,
  //     frameSets, frames3d, busyRetries}), player (FramePlayer),
  //   infer / render3d (endpoint URLs), label (status prefix of a 2D run),
  //   buildPayload(is3d) -> payload, or null when the page cannot run yet,
  //   currentRunKey() (what the controls ask for), runKeyFor(body, payload)
  //   (what a response is), renderSuccess(body), updateResultInfo(),
  //   applyViewUi(), canRun() (default: always).
  function TimeRunner(page) {
    const { els, state, player } = page;
    const canRun = page.canRun || (() => true);
    const cache = new Map();
    const CACHE_MAX = 48;

    function setRunBusy(busy) {
      state.inFlight = busy;
      els.runBtn.disabled = busy;
      els.view2dBtn.disabled = busy;
      els.view3dBtn.disabled = busy;
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: 'POST', cache: 'no-store', credentials: 'omit',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
      const text = await response.text();
      let body = null;
      try { body = text ? JSON.parse(text) : null; } catch (_) { body = null; }
      return { response, body };
    }

    // Cache a decoded frame set + its response under the full render identity
    // so re-selecting a model already rendered for this selection re-displays
    // it with no fetch and no re-inference.
    function cachePut(key, set, body) {
      cache.delete(key);
      cache.set(key, { set, body });
      while (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value);
    }

    // Serve `key` from the render cache if present; returns true when it did.
    function serveCached(key) {
      const hit = cache.get(key);
      if (!hit) return false;
      cache.delete(key);
      cache.set(key, hit); // LRU bump
      // The per-run sets belong to whatever ran last (possibly another model):
      // drop them and re-point runKey at the served response, so a later
      // field/view switch re-serves or re-fetches for THIS model.
      state.frameSets = {};
      state.frames3d = null;
      state.runKey = page.runKeyFor(hit.body, {});
      hit.set.then((frames) => {
        if (page.currentRunKey() !== key) return; // controls moved on before decode
        player.show(frames, player.index); // keep the timestep to compare like-for-like
        page.renderSuccess(hit.body);
        setStatus(els, 'ok', 'cached (no recompute)');
        els.latencyText.textContent = 'served from cache';
        showInferTiming(els.inferTiming, hit.body, null);
      }).catch((error) => showError(els, `Failed to load cached frames: ${String(error)}`));
      return true;
    }

    function swapToSet(setPromise) {
      const view = state.view, sets = state.frameSets, f3d = state.frames3d;
      return setPromise.then((frames) => {
        // Ignore stale swaps: the view toggled again or a new run replaced the sets.
        if (state.view !== view || state.frameSets !== sets || state.frames3d !== f3d) return;
        player.show(frames, player.index);
        page.updateResultInfo();
      }).catch((error) => showError(els, `Failed to load frames: ${String(error)}`));
    }

    // fresh=true: a user-initiated run (slider resets); fresh=false: a lazy
    // fetch for the current view/field (slider position kept).
    async function run(fresh) {
      if (state.inFlight || !state.config) return;
      clearError(els);
      const is3d = state.view === '3d';
      const payload = page.buildPayload(is3d);
      if (!payload) return;
      const cacheKey = page.currentRunKey();
      setRunBusy(true);
      player.stop();
      setStatus(els, 'loading', is3d ? '3D render request in flight...' : 'Inference request in flight...');
      const startedAt = performance.now();
      try {
        const { response, body } = await postJson(is3d ? page.render3d : page.infer, payload);
        if (!response.ok) {
          handleErrorResponse(els, state, response.status, body, () => { if (!state.inFlight && canRun()) void run(fresh); });
          return;
        }
        if (!body || body.status !== 'ok' || !Array.isArray(body.times_s) || !Array.isArray(body.frames) || !body.frames.length) {
          showError(els, 'Backend returned an invalid success payload.');
          setStatus(els, 'error', 'invalid response');
          return;
        }
        const set = preloadFrames(body.times_s, body.frames);
        const key = page.runKeyFor(body, payload);
        if (key !== state.runKey) { state.frameSets = {}; state.frames3d = null; } // new run: every cached set is stale
        state.runKey = key;
        if (is3d) state.frames3d = set;
        else state.frameSets[body.result_meta?.field || payload.view.field] = set;
        player.show(await set, fresh ? undefined : player.index); // fresh run: the player's mid-sweep default
        state.busyRetries = 0;
        page.renderSuccess(body);
        cachePut(cacheKey, set, body);
        const elapsed = Math.round(performance.now() - startedAt);
        els.latencyText.textContent = `${is3d ? 'render3d' : page.label} ${elapsed} ms (server total ${Math.round(body.timing_ms?.total || 0)} ms)`;
        showInferTiming(els.inferTiming, body, elapsed);
        setStatus(els, 'ok', `${player.frames.length} frames rendered`);
      } catch (error) {
        setStatus(els, 'error', 'network error');
        showError(els, `Request failed: ${String(error)}`);
      } finally {
        setRunBusy(false);
        drainRerun();
      }
    }

    // Auto-run entry point (drag, draw, control change). Coalesces to
    // at-most-one-in-flight so the single-GPU backend never piles up.
    function requestRun(opts) {
      const force = Boolean(opts && opts.force);
      if (!canRun()) return;
      state.busyRetries = 0; // a user-initiated run resets the 429 backoff
      const key = page.currentRunKey();
      if (state.inFlight) {
        if (force || key !== state.dispatchedKey) state.rerunPending = true;
        return;
      }
      if (!force && serveCached(key)) { state.dispatchedKey = key; return; }
      if (!force && key === state.dispatchedKey && player.frames.length) return;
      state.dispatchedKey = key;
      void run(true);
    }

    function drainRerun() {
      if (!state.rerunPending) return;
      state.rerunPending = false;
      requestRun();
    }

    // 2D slice <-> 3D surface. Both views share times_s, so swaps keep the
    // slider position; a missing set is fetched lazily (the server caches the
    // prediction, so that typically costs only the render).
    async function setView(view) {
      if (state.view === view || state.inFlight) return;
      state.view = view;
      page.applyViewUi();
      if (serveCached(page.currentRunKey())) return;
      const set = view === '3d' ? state.frames3d : state.frameSets[els.fieldSelect.value];
      if (set) await swapToSet(set);
      else if (canRun()) await run(false);
    }

    // Field switch (2D only): instant when that field's set is cached for this
    // run, otherwise a lazy render-only fetch. Nothing before the first run.
    function onFieldChange() {
      if (state.view !== '2d') return;
      if (serveCached(page.currentRunKey())) return;
      const field = els.fieldSelect.value;
      const sets = state.frameSets;
      const set = sets[field];
      if (!set) {
        if (!state.inFlight && state.runKey && canRun()) void run(false);
        return;
      }
      set.then((frames) => {
        if (state.frameSets !== sets || els.fieldSelect.value !== field) return; // stale switch
        player.show(frames, player.index);
        page.updateResultInfo();
      }).catch((error) => showError(els, `Failed to load ${field} frames: ${String(error)}`));
    }

    return { run, requestRun, setView, onFieldChange };
  }

  // Ground truth is optional: the FLOW-3D fields are distributed only in the restricted-access record, so a
  // backend set up from the open record has no GT and its catalog is empty. A GT control (checkbox or select) is
  // enabled only once the backend reports at least one GT case; otherwise it is greyed out with an explanation.
  const GT_MISSING_TITLE =
    'Ground truth is not installed on this server. It comes from the restricted-access data record (see the README).';
  function setGtAvailable(control, available) {
    if (!control) return;
    control.disabled = !available;
    if (!available && 'checked' in control) control.checked = false;
    const row = typeof control.closest === 'function' ? control.closest('label') : null;
    if (row) {
      row.classList.toggle('is-disabled', !available);
      row.title = available ? '' : GT_MISSING_TITLE;
    }
  }

  window.Common = {
    loadConfig, loadModels, cacheElements, setStatus, showError, clearError, makeRequestId,
    setSegment, setGtMetrics, showInferTiming, handleErrorResponse, FramePlayer, preloadFrames,
    setZIndexOptions, TimeRunner, setGtAvailable,
  };
})();
