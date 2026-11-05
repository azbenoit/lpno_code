// Runtime check for the straight-line tab's live auto-run coalescer (line.js).
// Loads the REAL line.js in a mock DOM with gated /infer_straight fetches and
// asserts the invariant that keeps the single-GPU backend from piling up:
// several P/V moves during a render collapse to ONE rerun, fired with the
// latest selection. Run: `node test/check_line_autorun.mjs` (also runs in CI).
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const tick = () => new Promise((r) => setTimeout(r, 0));

// --- minimal mock DOM -------------------------------------------------------
function makeEl() {
  const handlers = {};
  return {
    handlers, value: '', min: '', max: '', step: '', textContent: '',
    innerHTML: '', disabled: false, checked: false,
    classList: { add() {}, remove() {}, toggle() {} }, setAttribute() {},
    addEventListener(type, fn) { (handlers[type] ||= []).push(fn); },
    dispatch(type, ev = {}) { (handlers[type] || []).forEach((fn) => fn(ev)); },
    replaceChildren(...opts) { this.value = String((opts.find((o) => o.selected) || opts[0] || {}).value ?? ''); },
    setPointerCapture() {}, releasePointerCapture() {}, hasPointerCapture() { return false; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; },
  };
}

const elements = {};
const getEl = (id) => (elements[id] ||= makeEl());

// Gated /infer_straight fetches: each call records its payload and parks a
// resolver so the test can hold a render "in flight" while it fires more moves.
const straightCalls = [];
const gate = [];
function fetchMock(url, opts) {
  const body = opts && opts.body ? JSON.parse(opts.body) : null;
  const jsonResp = (obj) => ({ ok: true, status: 200, json: async () => obj, text: async () => JSON.stringify(obj) });
  if (url.endsWith('config.json')) {
    return Promise.resolve(jsonResp({
      backendBaseUrl: '',
      bounds: { p_watts: { min: 50, max: 400, step: 1, initial: 110 },
                v_scan_mps: { min: 0.05, max: 1.5, step: 0.001, initial: 0.34 } },
      pathDemo: { render: { width_px: 64, height_px: 52 } },
    }));
  }
  if (url.endsWith('config.local.json')) return Promise.resolve({ ok: false, status: 404, json: async () => ({}), text: async () => '' });
  // A model envelope whose min is off the config step grid (0.101 vs step 0.001 -> on grid;
  // 0.1005 is not) so the slider-min snap is exercised.
  if (url.includes('/models')) {
    return Promise.resolve(jsonResp({ status: 'ok', models: [{ id: 'm', label: 'm', default: true,
      physics: { p_watts: { min: 40, max: 215, default: 110 }, v_scan_mps: { min: 0.1005, max: 0.9604, default: 0.34 } } }] }));
  }
  if (url.includes('/gt_catalog')) return Promise.resolve(jsonResp({ status: 'ok', entries: [] }));
  if (url.includes('/infer_straight')) {
    const p = body?.inputs?.p_watts;
    const v = body?.inputs?.v_scan_mps;
    return new Promise((resolve) => {
      gate.push(() => resolve(jsonResp({
        status: 'ok', model: body.model, times_s: [0], frames: ['AA=='],
        straight_meta: { p_watts: p, v_scan_mps: v, path_length_m: 5.9724e-4, laser_off_time_s: 1e-3, gt_available: false, gt_name: null, split: null, points_xy_m: [] },
        result_meta: { grid_shape_xyz: [90, 40, 30], z_index_used: 29, field: 'T', panel: 'single' },
        timing_ms: { infer: 1, render: 1, total: 2 },
      })));
      straightCalls.push({ p, v });
    });
  }
  return Promise.resolve({ ok: false, status: 404, json: async () => ({}), text: async () => '' });
}
async function releaseNextGate() { const g = gate.shift(); if (g) g(); await tick(); await tick(); }

class ImageMock {
  set src(_v) { queueMicrotask(() => this.onload && this.onload()); }
}

const win = {
  addEventListener(type, fn) { (win._h ||= {})[type] = fn; },
  setInterval: () => 0, clearInterval: () => {}, setTimeout: (f, _d) => setTimeout(f, 0), clearTimeout: () => {},
};
const sandbox = {
  window: win, document: { getElementById: getEl, createElement: () => ({}) }, fetch: fetchMock,
  performance: { now: () => 0 }, Image: ImageMock,
  globalThis: { crypto: { randomUUID: () => 'req-test' } },
  console, setTimeout, queueMicrotask, JSON, Math, Number, Promise, Array, String, Boolean, Date,
};
sandbox.globalThis = sandbox; // line.js reads globalThis.crypto
sandbox.crypto = { randomUUID: () => 'req-test' };

// --- run --------------------------------------------------------------------
function fail(msg) { console.error(`FAIL: ${msg}`); process.exit(1); }

(async () => {
  vm.runInNewContext(readFileSync('public/common.js', 'utf8'), sandbox);
  vm.runInNewContext(readFileSync('public/line.js', 'utf8'), sandbox);
  await win._h.DOMContentLoaded();            // init(): config + gt + initial runInference
  await tick();
  if (straightCalls.length !== 1) fail(`initial load should issue 1 render, got ${straightCalls.length}`);
  await releaseNextGate();                    // finish the initial render -> idle
  // The model envelope is snapped onto the config step grid (a range input's
  // value grid is min + k*step, so an off-grid min would shift every value).
  const vSlider = getEl('vSlider');
  if (vSlider.min !== '0.101' || vSlider.max !== '0.96') fail(`V slider envelope should sit on the 0.001 grid, got [${vSlider.min}, ${vSlider.max}]`);

  const pSlider = getEl('pSlider');
  // Move 1: idle -> dispatches render #2 at P=200 (parks on the gate).
  pSlider.value = '200'; pSlider.dispatch('input');
  await tick();
  if (straightCalls.length !== 2) fail(`first move should dispatch (expected 2 calls, got ${straightCalls.length})`);
  if (straightCalls[1].p !== 200) fail(`render #2 should use P=200, got ${straightCalls[1].p}`);

  // Moves 2 & 3 arrive mid-flight: must coalesce (no new dispatch yet).
  pSlider.value = '300'; pSlider.dispatch('input');
  pSlider.value = '350'; pSlider.dispatch('input');
  await tick();
  if (straightCalls.length !== 2) fail(`mid-flight moves must not pile up (expected 2 calls, got ${straightCalls.length})`);

  // Completing render #2 drains exactly one rerun, at the latest P=350.
  await releaseNextGate();
  if (straightCalls.length !== 3) fail(`completion should drain one rerun (expected 3 calls, got ${straightCalls.length})`);
  if (straightCalls[2].p !== 350) fail(`rerun should use latest P=350, got ${straightCalls[2].p}`);

  // No selection change since -> no further reruns after it completes.
  await releaseNextGate();
  if (straightCalls.length !== 3) fail(`idle after rerun should not dispatch again (got ${straightCalls.length})`);

  console.log('OK: line.js auto-run coalesces moves to one latest-wins rerun (3 renders for 4 moves)');
})().catch((e) => fail(e.stack || e.message));
