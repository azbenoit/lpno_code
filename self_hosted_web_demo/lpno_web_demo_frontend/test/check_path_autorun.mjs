// Runtime check for the random-path tab's draw-to-infer coalescer (path.js).
// Loads the REAL path.js in a mock DOM, draws a freehand path, then draws a
// second path while the first render is still in flight, and asserts the
// single-GPU invariant: the mid-render redraw does not pile up — it coalesces
// to ONE rerun with the latest path, and the completed first run does not
// clobber the newly drawn path. Run: `node test/check_path_autorun.mjs` (CI).
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const tick = () => new Promise((r) => setTimeout(r, 0));

// Canvas transform mirror (must match path.js SUBSTRATE / CANVAS_MARGIN_PX).
const SUB = { xMin: -0.75e-3, xMax: 0.75e-3, yMin: -0.5e-3, yMax: 0.5e-3 };
const MARGIN = 20, CW = 940, CH = 560;
const mToC = (xM, yM) => {
  const w = CW - 2 * MARGIN, h = CH - 2 * MARGIN;
  return {
    x: MARGIN + ((xM - SUB.xMin) / (SUB.xMax - SUB.xMin)) * w,
    y: MARGIN + ((SUB.yMax - yM) / (SUB.yMax - SUB.yMin)) * h,
  };
};
const START = mToC(-0.55e-3, 0);       // fixed freehand start (green dot)
const A_END = mToC(0, 0);              // path A second point
const B_END = mToC(0.3e-3, 0.2e-3);   // path B second point

// --- minimal mock DOM -------------------------------------------------------
const noopCtx = new Proxy({}, { get: (_t, p) => (p === 'createLinearGradient' ? () => ({ addColorStop() {} }) : () => {}) });
function makeEl(extra = {}) {
  const handlers = {};
  const classes = new Set();
  return Object.assign({
    handlers, value: '', innerHTML: '', textContent: '', disabled: false, title: '', style: {},
    classList: { add: (c) => classes.add(c), remove: (c) => classes.delete(c), toggle() {}, contains: (c) => classes.has(c) }, setAttribute() {},
    addEventListener(type, fn) { (handlers[type] ||= []).push(fn); },
    dispatch(type, ev = {}) { (handlers[type] || []).forEach((fn) => fn(ev)); },
  }, extra);
}
const elements = {};
const getEl = (id) => (elements[id] ||= id === 'pathCanvas' ? makeEl({
  width: CW, height: CH,
  getContext: () => noopCtx,
  getBoundingClientRect: () => ({ left: 0, top: 0, width: CW, height: CH }),
  setPointerCapture() {}, releasePointerCapture() {}, hasPointerCapture: () => false,
}) : makeEl());

// Gated /infer_path fetches: record each call's path and park a resolver.
const pathCalls = [];
const gate = [];
function fetchMock(url, opts) {
  const body = opts && opts.body ? JSON.parse(opts.body) : null;
  const jsonResp = (o) => ({ ok: true, status: 200, json: async () => o, text: async () => JSON.stringify(o) });
  if (url.endsWith('config.json')) return Promise.resolve(jsonResp({ backendBaseUrl: '', pathDemo: { maxPoints: 512, render: { width_px: 64, height_px: 52 } } }));
  if (url.endsWith('config.local.json')) return Promise.resolve({ ok: false, status: 404, json: async () => ({}), text: async () => '' });
  // Presets are per-model: only model_a offers the stored case r_010 (158 W, 0.6 m/s).
  if (url.includes('/infer_path/presets')) {
    const presets = url.includes('model=model_a') ? [{ name: 'r_010', points_xy_m: [[-5.5e-4, 0], [0, 0]], p_watts: 158, v_scan_mps: 0.6 }] : [];
    return Promise.resolve(jsonResp({ status: 'ok', presets }));
  }
  if (url.includes('/infer_path')) {
    const preset = body?.path?.preset || null;
    const pts = preset ? [[-5.5e-4, 0], [0, 0]] : (body?.path?.points_xy_m || []);
    return new Promise((resolve) => {
      gate.push(() => resolve(jsonResp({
        status: 'ok', model: body.model, times_s: [0, 1e-5], frames: ['AA==', 'AA=='],
        path_meta: { n_points_used: pts.length, path_length_m: 5e-4, laser_off_time_s: 1e-3, clamped: false, preset, gt_available: false, points_xy_m: pts,
          ...(preset ? { p_watts: 158, v_scan_mps: 0.6 } : {}) },
        result_meta: { grid_shape_xyz: [75, 50, 40], z_index_used: 39, field: 'T', panel: 'single' },
        timing_ms: { infer: 1, render: 1, total: 2 },
      })));
      pathCalls.push({ pts, dt: body.frame_dt_s ?? null, preset, model: body.model });
    });
  }
  return Promise.resolve({ ok: false, status: 404, json: async () => ({}), text: async () => '' });
}
async function releaseNextGate() { const g = gate.shift(); if (g) g(); await tick(); await tick(); }

class ImageMock { set src(_v) { queueMicrotask(() => this.onload && this.onload()); } }
const win = { addEventListener(type, fn) { (win._h ||= {})[type] = fn; }, setInterval: () => 0, clearInterval() {}, setTimeout: (f) => setTimeout(f, 0), clearTimeout() {} };
const sandbox = {
  window: win, document: { getElementById: getEl }, fetch: fetchMock,
  performance: { now: () => 0 }, Image: ImageMock, console, setTimeout, queueMicrotask,
  JSON, Math, Number, Promise, Array, String, Boolean, Date,
};
sandbox.globalThis = sandbox;
sandbox.crypto = { randomUUID: () => 'req-test' };

// Simulate a full freehand stroke: pointerdown at the start, one move, pointerup.
function drawPath(canvas, end, pointerId) {
  canvas.dispatch('pointerdown', { pointerId, clientX: START.x, clientY: START.y });
  canvas.dispatch('pointermove', { pointerId, clientX: end.x, clientY: end.y });
  canvas.dispatch('pointerup', { pointerId });
}

function fail(m) { console.error(`FAIL: ${m}`); process.exit(1); }

(async () => {
  vm.runInNewContext(readFileSync('public/common.js', 'utf8'), sandbox);
  vm.runInNewContext(readFileSync('public/path.js', 'utf8'), sandbox);
  await win._h.DOMContentLoaded();   // init(): config + presets, no auto-run (nothing drawn)
  await tick();
  if (pathCalls.length !== 0) fail(`no inference before a path is drawn, got ${pathCalls.length}`);

  const canvas = getEl('pathCanvas');
  // Draw path A -> auto-run render #1 (parks on the gate).
  drawPath(canvas, A_END, 1);
  await tick();
  if (pathCalls.length !== 1) fail(`drawing a path should auto-infer (expected 1 call, got ${pathCalls.length})`);
  const aLen = pathCalls[0].pts.length;
  if (aLen !== 2) fail(`path A should have 2 points, got ${aLen}`);

  // Draw path B while render #1 is still in flight -> must coalesce, not pile up.
  drawPath(canvas, B_END, 2);
  await tick();
  if (pathCalls.length !== 1) fail(`mid-render redraw must not pile up (expected 1 call, got ${pathCalls.length})`);

  // Completing render #1 drains exactly one rerun, for the latest path B.
  await releaseNextGate();
  if (pathCalls.length !== 2) fail(`completion should drain one rerun (expected 2 calls, got ${pathCalls.length})`);
  const bx = pathCalls[1].pts[1][0];
  if (Math.abs(bx - 0.3e-3) > 1e-6) fail(`rerun should use latest path B (x=0.3e-3), got ${bx}`);

  // No change since -> completing the rerun dispatches nothing further.
  await releaseNextGate();
  if (pathCalls.length !== 2) fail(`idle after rerun should not dispatch again (got ${pathCalls.length})`);

  // Changing the frame step alone is a new run (the dispatch key carries it).
  getEl('frameDtInput').value = '50';
  getEl('frameDtInput').dispatch('change');
  await tick();
  if (pathCalls.length !== 3) fail(`a frame-step change should re-infer (expected 3 calls, got ${pathCalls.length})`);
  if (Math.abs(pathCalls[2].dt - 50e-6) > 1e-12) fail(`rerun should carry frame_dt_s=50e-6, got ${pathCalls[2].dt}`);
  await releaseNextGate();

  // The completed run must surface the raw model time (timing_ms) as the
  // inference-timing line with a per-step breakdown, so users can tell
  // inference from render/network waiting.
  const it = getEl('inferTiming');
  if (!/Model inference/.test(it.innerHTML)) fail(`inferTiming line not rendered: "${it.innerHTML}"`);
  if (!/2 steps/.test(it.innerHTML)) fail(`inferTiming missing per-step breakdown: "${it.innerHTML}"`);

  // Preset run: the greyed P/V readout shows the case's stored physics, and a
  // model switch keeps the preset when the new model offers it, else drops the
  // old model's render instead of showing it under the new model's name.
  getEl('modelSelect').value = 'model_a';
  await getEl('modelSelect').handlers.change[0]();
  await tick();
  await releaseNextGate(); // the drawn path re-ran under model_a
  getEl('presetSelect').value = 'r_010';
  getEl('presetSelect').dispatch('change');
  await tick();
  const presetCall = pathCalls[pathCalls.length - 1];
  if (presetCall.preset !== 'r_010') fail(`picking a preset should infer it, last call: ${JSON.stringify(presetCall)}`);
  await releaseNextGate();
  if (getEl('pValue').textContent !== '158 W' || getEl('vValue').textContent !== '0.60 m/s') {
    fail(`preset run should echo its physics in the readout, got "${getEl('pValue').textContent}" / "${getEl('vValue').textContent}"`);
  }
  const before = pathCalls.length;
  getEl('modelSelect').value = 'model_b';
  await getEl('modelSelect').handlers.change[0]();
  await tick();
  if (pathCalls.length !== before) fail(`a model without the preset must not infer (got ${pathCalls.length - before} calls)`);
  if (getEl('presetSelect').value !== '') fail('preset selection should be dropped when the new model lacks it');
  if (getEl('resultPlaceholder').classList.contains('hidden')) fail('result placeholder should be shown after dropping the preset');
  if (getEl('resultModeTag').textContent !== 'none') fail(`result tag should be reset, got "${getEl('resultModeTag').textContent}"`);

  console.log('OK: path.js infers on draw and coalesces a mid-render redraw to one latest-wins rerun');
})().catch((e) => fail(e.stack || e.message));
