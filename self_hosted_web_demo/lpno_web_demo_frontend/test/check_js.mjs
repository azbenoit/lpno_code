// Runtime smoke test for the browser JS. `node --check` only catches syntax,
// not a ReferenceError at IIFE-execution time — which once left `window.PVMap`
// undefined (an export listed a renamed variable), silently blanking every map.
// This executes pvmap.js in a mock DOM and asserts the module actually exports
// its API. Run: `node test/check_js.mjs` (also runs in CI).
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import vm from 'node:vm';

const files = ['public/common.js', 'public/pvmap.js', 'public/app.js', 'public/line.js', 'public/path.js'];
let failed = false;

// 1) Syntax check every browser script.
for (const f of files) {
  try {
    execFileSync(process.execPath, ['--check', f]);
  } catch {
    console.error(`SYNTAX FAIL: ${f}`);
    failed = true;
  }
}

// 2) pvmap.js must define window.PVMap with its public API when executed.
const win = {};
const sandbox = {
  window: win,
  document: { createElement: () => ({ getContext: () => ({}) }) },
};
try {
  vm.runInNewContext(readFileSync('public/pvmap.js', 'utf8'), sandbox);
  const need = ['draw', 'eventToPV', 'drawGtScatter', 'drawSplitLegend', 'hitTestGt', 'splitColor', 'pvToXY'];
  if (!win.PVMap) {
    console.error('FAIL: window.PVMap is undefined after loading pvmap.js (IIFE threw before the export)');
    failed = true;
  } else {
    const missing = need.filter((k) => typeof win.PVMap[k] !== 'function');
    if (missing.length) {
      console.error('FAIL: window.PVMap missing functions:', missing.join(', '));
      failed = true;
    } else {
      console.log('OK: window.PVMap exports', need.join(', '));
    }
  }
} catch (e) {
  console.error('FAIL: pvmap.js threw at load:', e.message);
  failed = true;
}

// 3) The committed config must stay same-origin: a LAN/localhost backendBaseUrl
// would be baked into the container image and point every visitor's browser at
// a backend address that only exists on the machine that wrote it. A direct
// backend URL belongs in the untracked public/config.local.json.
const cfg = JSON.parse(readFileSync('public/config.json', 'utf8'));
if (cfg.backendBaseUrl !== '') {
  console.error(`FAIL: public/config.json backendBaseUrl must be "" (got ${JSON.stringify(cfg.backendBaseUrl)})`);
  failed = true;
} else {
  console.log('OK: config.json backendBaseUrl is same-origin ("")');
}

process.exit(failed ? 1 : 0);
