// Shared P-V (power vs scan-speed) normalized-enthalpy map, plus a ground-truth
// scatter overlay coloured by dataset split. Used by the steady-state page
// (app.js, which also draws the H-P variant: `holder.variant = 'hp'`) and the
// straight-track page (line.js). One H* formula lives here (vizConstants +
// computeHStar / computeVFromHStar); the pages do not carry their own.
(() => {
  const MAP_TICK_FONT = '13px IBM Plex Sans, Segoe UI, sans-serif';
  const MAP_AXIS_FONT = '15px IBM Plex Sans, Segoe UI, sans-serif';
  const MAP_LEGEND_FONT = '13px IBM Plex Sans, Segoe UI, sans-serif';
  const MAP_MARKER_FONT = '13px IBM Plex Sans, Segoe UI, sans-serif';

  // Dataset-split marker styles for the GT scatter, matching the enthalpy
  // notebook (OF_normalized_enthalpy_params): high-contrast fills with a
  // contrasting edge so the dots read clearly over the plasma/viridis map.
  const SPLIT_STYLES = {
    train: { fill: '#ffffff', edge: '#111111' },
    val: { fill: '#ffd400', edge: '#111111' },
    test: { fill: '#000000', edge: '#ffffff' },
  };
  function splitStyle(split) { return SPLIT_STYLES[split] || { fill: '#888888', edge: '#111111' }; }
  function splitColor(split) { return splitStyle(split).fill; }

  function lerp(a, b, t) { return a + (b - a) * t; }
  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  const PLASMA_STOPS = [
    [0.0, [13, 8, 135]], [0.25, [126, 3, 168]], [0.5, [204, 71, 120]],
    [0.75, [248, 149, 64]], [1.0, [240, 249, 33]],
  ];
  const VIRIDIS_STOPS = [
    [0.0, [68, 1, 84]], [0.25, [59, 82, 139]], [0.5, [33, 145, 140]],
    [0.75, [94, 201, 97]], [1.0, [253, 231, 37]],
  ];
  function rampColorRgb(stops, tIn) {
    const t = clamp(Number(tIn), 0, 1);
    for (let i = 1; i < stops.length; i++) {
      const [t1, c1] = stops[i];
      const [t0, c0] = stops[i - 1];
      if (t <= t1) {
        const u = (t - t0) / Math.max(t1 - t0, 1e-12);
        return [Math.round(lerp(c0[0], c1[0], u)), Math.round(lerp(c0[1], c1[1], u)), Math.round(lerp(c0[2], c1[2], u))];
      }
    }
    return stops[stops.length - 1][1];
  }
  const plasmaColorRgb = (t) => rampColorRgb(PLASMA_STOPS, t);
  const viridisColorRgb = (t) => rampColorRgb(VIRIDIS_STOPS, t);

  // P-V map constants: the notebook's H* model (A, rho, C, Tm-T0, D, a) and the
  // axis/colorbar layout. `yAxis: 'v'` = scan speed on the y axis.
  function vizConstants() {
    const rho = 4420.0, C = 750.0, Tm = 1900.0 - 300.0, D = 8.1e-6, a = 5e-5;
    const A_corr = (0.35 / 0.15) / (4420.0 / 4506.0) / (Math.sqrt(8.1e-6 / 9.863e-6)) / (750.0 / 900.0);
    const constFactor = Math.PI * (rho ** 2) * (C ** 2) * (Tm ** 2) * D * (a ** 3);
    return {
      A: 0.35, A_corr, constFactor,
      xMin: 30, xMax: 220, yMin: 0.05, yMax: 1.15,
      yAxis: 'v', yTicks: [0.2, 0.4, 0.6, 0.8, 1.0], yDecimals: 1, yLabel: 'Scan Speed (m/s)',
      colormap: plasmaColorRgb,
      cbarTicks: [0, 2, 4, 6, 8], cbarDecimals: 0, cbarLabel: 'Normalized Enthalpy (-)',
      // Regime boundaries as H* levels, re-anchored for the calibrated
      // (experimentally anchored) physics on 2026-09-02. Each line is set from
      // its OWN measurement on the calibrated GT, not by scaling the legacy
      // triple [2.75, 4.8, 8.0] -- a single multiplicative shift was tried first
      // and is wrong, because the three boundaries do not move together.
      //
      //  lack of fusion  3.5  the shallowest MELTING calibrated case sits at
      //                       H* 3.69 (20 um), so the no-fusion boundary is just
      //                       below it. No calibrated case fails to melt, so this
      //                       is an upper bound, not a fitted crossing.
      //  conduction      6.4  melt depth first reaches one beam radius (50 um) at
      //                       H* 6.53.
      //  unstable
      //  keyhole        10.0  set by the user (2026-09-02). Note this is NOT the
      //                       keyhole ONSET, which is measured at H* ~7.98: the
      //                       dataset's keyhole-named cases (kh40_*, khc_*) occupy
      //                       H* 8.29..10.67 while every other case is 3.69..7.67
      //                       -- disjoint -- and melt depth doubles (50 -> 120 um)
      //                       across that same gap. This line marks the stronger
      //                       INSTABILITY condition further into the keyhole
      //                       regime, so only the deepest cases (H* 10.65, 10.67;
      //                       melt depth 190-200 um) sit beyond it.
      contourLevels: [3.5, 6.4, 10.0],
      contourColors: ['#1540ff', '#1e8f31', '#df3c1f'],
      cbarMax: 3 * A_corr,
    };
  }

  // H-P variant (steady page): H* on the y axis, colour = the scan speed that
  // gives (P, H*), regime levels as horizontal lines. `hMin` comes from the
  // page's feasible H range (default 0.75 A_corr).
  function hpVizConstants(hMin) {
    const base = vizConstants();
    return {
      ...base,
      // yMax 11 covers the catalog (14 cases sit at H* 9..10.5; a 9.0 top pinned
      // them, and their hit targets, on the edge) and keeps the 10.0 unstable-
      // keyhole level on screen.
      yAxis: 'h', yMin: hMin ?? 0.75 * base.A_corr, yMax: 11.0,
      yTicks: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], yDecimals: 0, yLabel: 'Normalized Enthalpy (-)',
      colormap: viridisColorRgb,
      cbarMax: 2.0, cbarTicks: [0, 0.5, 1.0, 1.5, 2.0], cbarDecimals: 1, cbarLabel: 'Scan Speed (m/s)',
    };
  }

  // Normalized enthalpy H*(P, V) of the map, and its inverse V(P, H*).
  function computeHStar(p, v, viz = vizConstants()) {
    const vv = Math.max(Math.abs(Number(v)), 1e-12);
    return (viz.A * Number(p)) / Math.sqrt(viz.constFactor * vv);
  }
  function computeVFromHStar(p, h, viz = vizConstants()) {
    const hSafe = Math.max(Math.abs(Number(h)), 1e-12);
    return ((viz.A * Number(p)) ** 2) / (viz.constFactor * (hSafe ** 2));
  }

  // Build (and memoize on `holder`) the static heatmap + axes + colorbar.
  // `holder.variant === 'hp'` draws the H-P map (with `holder.hMin`).
  function ensureBackground(canvas, holder) {
    if (!canvas) return null;
    if (holder.bg && holder.bg.width === canvas.width && holder.bg.height === canvas.height) return holder.bg;

    const w = canvas.width, h = canvas.height;
    const bg = document.createElement('canvas');
    bg.width = w; bg.height = h;
    const ctx = bg.getContext('2d');
    const viz = holder.variant === 'hp' ? hpVizConstants(holder.hMin) : vizConstants();
    const isH = viz.yAxis === 'h';
    // Cell colour: H*(P, V) on the P-V map; V(P, H*) on the H-P map.
    const cellValue = isH ? (p, y) => computeVFromHStar(p, y, viz) : (p, y) => computeHStar(p, y, viz);
    // Regime level curves: V(P) at constant H* on the P-V map; horizontal on H-P.
    const contourY = isH ? (_p, level) => level : (p, level) => computeVFromHStar(p, level, viz);

    ctx.clearRect(0, 0, w, h);
    const outerGrad = ctx.createLinearGradient(0, 0, w, h);
    outerGrad.addColorStop(0, '#f8fafc');
    outerGrad.addColorStop(1, '#eff3f6');
    ctx.fillStyle = outerGrad;
    roundRect(ctx, 0, 0, w, h, 10); ctx.fill();

    const layout = { left: 68, top: 22, right: 22, bottom: 64, cbarGap: 12, cbarW: 18 };
    layout.plotX = layout.left;
    layout.plotY = layout.top;
    layout.plotW = Math.max(120, w - layout.left - layout.right - layout.cbarGap - layout.cbarW);
    layout.plotH = Math.max(120, h - layout.top - layout.bottom);
    layout.cbarX = layout.plotX + layout.plotW + layout.cbarGap;
    layout.cbarY = layout.plotY;
    layout.cbarH = layout.plotH;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(layout.plotX, layout.plotY, layout.plotW, layout.plotH);

    ctx.save();
    ctx.beginPath();
    ctx.rect(layout.plotX, layout.plotY, layout.plotW, layout.plotH);
    ctx.clip();
    const nx = 180, ny = 110;
    for (let iy = 0; iy < ny; iy++) {
      const v0 = lerp(viz.yMax, viz.yMin, iy / ny);
      const v1 = lerp(viz.yMax, viz.yMin, (iy + 1) / ny);
      const y0 = layout.plotY + (iy * layout.plotH) / ny;
      const y1 = layout.plotY + ((iy + 1) * layout.plotH) / ny;
      for (let ix = 0; ix < nx; ix++) {
        const p0 = lerp(viz.xMin, viz.xMax, ix / nx);
        const p1 = lerp(viz.xMin, viz.xMax, (ix + 1) / nx);
        const t = clamp(cellValue(0.5 * (p0 + p1), 0.5 * (v0 + v1)) / viz.cbarMax, 0, 1);
        const [r, g, b] = viz.colormap(t);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        const x = layout.plotX + (ix * layout.plotW) / nx;
        ctx.fillRect(x, y0, layout.plotW / nx + 1, (y1 - y0) + 1);
      }
    }
    for (let idx = 0; idx < viz.contourLevels.length; idx++) {
      const level = viz.contourLevels[idx];
      ctx.strokeStyle = viz.contourColors[idx];
      ctx.lineWidth = 2;
      ctx.beginPath();
      let started = false;
      const nSamples = 400;
      for (let i = 0; i <= nSamples; i++) {
        const pVal = lerp(viz.xMin, viz.xMax, i / nSamples);
        const vVal = contourY(pVal, level);
        if (!Number.isFinite(vVal) || vVal < viz.yMin || vVal > viz.yMax) { started = false; continue; }
        const x = layout.plotX + ((pVal - viz.xMin) / (viz.xMax - viz.xMin)) * layout.plotW;
        const y = layout.plotY + ((viz.yMax - vVal) / (viz.yMax - viz.yMin)) * layout.plotH;
        if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
      }
      ctx.stroke();
    }
    ctx.restore();

    ctx.strokeStyle = 'rgba(22,34,53,0.3)';
    ctx.lineWidth = 1;
    ctx.strokeRect(layout.plotX, layout.plotY, layout.plotW, layout.plotH);

    ctx.font = MAP_TICK_FONT;
    ctx.fillStyle = 'rgba(22,34,53,0.85)';
    ctx.strokeStyle = 'rgba(22,34,53,0.35)';
    for (const tick of [50, 75, 100, 125, 150, 175, 200]) {
      if (tick < viz.xMin || tick > viz.xMax) continue;
      const x = layout.plotX + ((tick - viz.xMin) / (viz.xMax - viz.xMin)) * layout.plotW;
      ctx.beginPath(); ctx.moveTo(x, layout.plotY + layout.plotH); ctx.lineTo(x, layout.plotY + layout.plotH + 5); ctx.stroke();
      const label = String(tick);
      ctx.fillText(label, x - ctx.measureText(label).width / 2, layout.plotY + layout.plotH + 20);
    }
    for (const tick of viz.yTicks) {
      if (tick < viz.yMin || tick > viz.yMax) continue;
      const y = layout.plotY + ((viz.yMax - tick) / (viz.yMax - viz.yMin)) * layout.plotH;
      ctx.beginPath(); ctx.moveTo(layout.plotX - 5, y); ctx.lineTo(layout.plotX, y); ctx.stroke();
      const label = tick.toFixed(viz.yDecimals);
      ctx.fillText(label, layout.plotX - 8 - ctx.measureText(label).width, y + 4);
    }

    ctx.font = MAP_AXIS_FONT;
    const xLabel = 'Power (W)';
    ctx.fillText(xLabel, layout.plotX + layout.plotW / 2 - ctx.measureText(xLabel).width / 2, h - 12);
    ctx.save();
    ctx.translate(18, layout.plotY + layout.plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(viz.yLabel, 0, 0);
    ctx.restore();

    for (let i = 0; i < layout.cbarH; i++) {
      const t = 1 - i / Math.max(layout.cbarH - 1, 1);
      const [r, g, b] = viz.colormap(t);
      ctx.strokeStyle = `rgb(${r},${g},${b})`;
      ctx.beginPath();
      ctx.moveTo(layout.cbarX, layout.cbarY + i + 0.5);
      ctx.lineTo(layout.cbarX + layout.cbarW, layout.cbarY + i + 0.5);
      ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(22,34,53,0.3)';
    ctx.strokeRect(layout.cbarX, layout.cbarY, layout.cbarW, layout.cbarH);
    ctx.font = MAP_TICK_FONT;
    for (const tick of viz.cbarTicks.filter((v) => v <= viz.cbarMax + 0.25)) {
      const y = layout.cbarY + (1 - tick / viz.cbarMax) * layout.cbarH;
      ctx.beginPath(); ctx.moveTo(layout.cbarX + layout.cbarW, y); ctx.lineTo(layout.cbarX + layout.cbarW + 4, y); ctx.stroke();
      ctx.fillText(tick.toFixed(viz.cbarDecimals), layout.cbarX + layout.cbarW + 6, y + 3);
    }
    ctx.save();
    ctx.translate(layout.cbarX + layout.cbarW + 36, layout.cbarY + layout.cbarH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.font = MAP_AXIS_FONT;
    ctx.fillText(viz.cbarLabel, 0, 0);
    ctx.restore();

    // Regime legend (bottom-right; no GT case of the 123-case catalog sits
    // under it on either map, checked 2026-09-14 -- same for the split legend
    // top-left, given the H-P yMax above).
    const legendItems = [
      { color: viz.contourColors[2], label: 'unstable keyhole' },
      { color: viz.contourColors[1], label: 'conduction' },
      { color: viz.contourColors[0], label: 'lack of fusion' },
    ];
    const legendW = 154, legendH = 64;
    const legendX = layout.plotX + layout.plotW - legendW - 8;
    const legendY = layout.plotY + layout.plotH - legendH - 8;
    ctx.fillStyle = 'rgba(255,255,255,0.78)';
    roundRect(ctx, legendX, legendY, legendW, legendH, 8); ctx.fill();
    ctx.strokeStyle = 'rgba(22,34,53,0.18)'; ctx.stroke();
    ctx.font = MAP_LEGEND_FONT;
    ctx.fillStyle = 'rgba(22,34,53,0.9)';
    legendItems.forEach((it, idx) => {
      const y = legendY + 16 + idx * 17;
      ctx.strokeStyle = it.color;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(legendX + 10, y); ctx.lineTo(legendX + 34, y); ctx.stroke();
      ctx.fillText(it.label, legendX + 40, y + 3);
    });

    holder.bg = { canvas: bg, width: w, height: h, layout, viz };
    return holder.bg;
  }

  // (P, y) -> pixel, y being V on the P-V map or H* on the H-P map.
  function pvToXY(p, v, layout, viz) {
    return {
      x: layout.plotX + ((clamp(p, viz.xMin, viz.xMax) - viz.xMin) / (viz.xMax - viz.xMin)) * layout.plotW,
      y: layout.plotY + ((viz.yMax - clamp(v, viz.yMin, viz.yMax)) / (viz.yMax - viz.yMin)) * layout.plotH,
    };
  }

  // Catalog point -> pixel for this map's y axis.
  function gtToXY(layout, viz) {
    return viz.yAxis === 'h'
      ? (pt) => pvToXY(pt.p_watts, computeHStar(pt.p_watts, pt.v_scan_mps, viz), layout, viz)
      : (pt) => pvToXY(pt.p_watts, pt.v_scan_mps, layout, viz);
  }

  // Scatter of GT cases coloured by split, with an optional highlighted point.
  // opts.toXY(pt) -> {x, y} overrides the map's own catalog-point mapping.
  function drawGtScatter(ctx, layout, viz, points, opts = {}) {
    if (!points || !points.length) return;
    const selected = opts.selectedName || null;
    const toXY = opts.toXY || gtToXY(layout, viz);
    ctx.save();
    ctx.beginPath();
    ctx.rect(layout.plotX, layout.plotY, layout.plotW, layout.plotH);
    ctx.clip();
    for (const pt of points) {
      const { x, y } = toXY(pt);
      const isSel = selected && pt.name === selected;
      const style = splitStyle(pt.split);
      ctx.beginPath();
      ctx.arc(x, y, isSel ? 7 : 4.5, 0, Math.PI * 2);
      ctx.fillStyle = style.fill;
      ctx.fill();
      ctx.lineWidth = isSel ? 3 : 1.2;
      ctx.strokeStyle = isSel ? '#12d0ff' : style.edge;
      ctx.stroke();
    }
    ctx.restore();
  }

  // Legend chip row for the split colours; returns nothing, draws top-left.
  function drawSplitLegend(ctx, layout) {
    const items = [['train', 'Train'], ['val', 'Val'], ['test', 'Test']];
    const boxW = 128, boxH = 62;
    // Top-left, not bottom-left: with the full 123-case catalog the low-P/low-V
    // corner is dense and the legend sat on top of the points. Offset 26px below
    // plotY to clear the "P=.. W, V=.. m/s" readout chip, which is a 16px strip
    // at the very top of the canvas.
    const x = layout.plotX + 8, y = layout.plotY + 26;
    ctx.fillStyle = 'rgba(255,255,255,0.82)';
    roundRect(ctx, x, y, boxW, boxH, 8); ctx.fill();
    ctx.strokeStyle = 'rgba(22,34,53,0.18)'; ctx.stroke();
    ctx.font = MAP_LEGEND_FONT;
    ctx.fillStyle = 'rgba(22,34,53,0.9)';
    items.forEach(([key, label], i) => {
      const cy = y + 16 + i * 16;
      const style = splitStyle(key);
      ctx.beginPath(); ctx.arc(x + 14, cy - 3, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = style.fill; ctx.fill();
      ctx.strokeStyle = style.edge; ctx.lineWidth = 1.2; ctx.stroke();
      ctx.fillStyle = 'rgba(22,34,53,0.9)';
      ctx.fillText(`${label} GT`, x + 26, cy);
    });
  }

  // Nearest GT point within `radiusPx` of the pointer, or null. `cache` is the
  // map's background cache ({ layout, viz }); the y axis follows its variant.
  function hitTestGt(event, canvas, cache, points, opts = {}) {
    if (!points || !points.length || !cache) return null;
    const radiusPx = opts.radiusPx ?? 9;
    const { layout, viz } = cache;
    const toXY = opts.toXY || gtToXY(layout, viz);
    const rect = canvas.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * canvas.width;
    const py = ((event.clientY - rect.top) / rect.height) * canvas.height;
    let best = null, bestD = radiusPx * radiusPx;
    for (const pt of points) {
      const { x, y } = toXY(pt);
      const d = (px - x) ** 2 + (py - y) ** 2;
      if (d <= bestD) { best = pt; bestD = d; }
    }
    return best;
  }

  // Full draw: background + optional GT scatter/legend + current marker at
  // (opts.p, opts.v) on the P-V map or (opts.p, opts.h) on the H-P map.
  function draw(canvas, holder, opts = {}) {
    const cache = ensureBackground(canvas, holder);
    if (!cache) return null;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(cache.canvas, 0, 0);
    const { layout, viz } = cache;
    const isH = viz.yAxis === 'h';

    if (opts.showGt && opts.gtPoints) {
      drawSplitLegend(ctx, layout);
      drawGtScatter(ctx, layout, viz, opts.gtPoints, { selectedName: opts.selectedName });
    }

    const yVal = isH ? opts.h : opts.v;
    if (opts.p != null && yVal != null) {
      const { x, y } = pvToXY(opts.p, yVal, layout, viz);
      ctx.save();
      ctx.beginPath(); ctx.rect(layout.plotX, layout.plotY, layout.plotW, layout.plotH); ctx.clip();
      ctx.strokeStyle = 'rgba(255,255,255,0.65)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, layout.plotY); ctx.lineTo(x, layout.plotY + layout.plotH); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(layout.plotX, y); ctx.lineTo(layout.plotX + layout.plotW, y); ctx.stroke();
      ctx.fillStyle = '#101820'; ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 5.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.restore();

      ctx.font = MAP_MARKER_FONT;
      const label = isH
        ? `P=${Number(opts.p).toFixed(0)} W, H=${Number(opts.h).toFixed(3)}`
        : `P=${Number(opts.p).toFixed(0)} W, V=${Number(opts.v).toFixed(3)} m/s`;
      const tw = ctx.measureText(label).width;
      const bx = Math.max(8, Math.min(canvas.width - tw - 16, layout.plotX + 8));
      ctx.fillStyle = 'rgba(255,255,255,0.72)';
      roundRect(ctx, bx, 6, tw + 8, 16, 6); ctx.fill();
      ctx.strokeStyle = 'rgba(22,34,53,0.12)'; ctx.stroke();
      ctx.fillStyle = 'rgba(22,34,53,0.92)';
      ctx.fillText(label, bx + 4, 18);
    }
    return cache;
  }

  // Pointer -> raw axis values {p, y} (y = V or H* per variant), or null when
  // outside the plot and !clampToPlot.
  function eventToAxes(event, canvas, holder, { clampToPlot = false } = {}) {
    const cache = ensureBackground(canvas, holder);
    if (!cache) return null;
    const rect = canvas.getBoundingClientRect();
    const xRaw = ((event.clientX - rect.left) / rect.width) * canvas.width;
    const yRaw = ((event.clientY - rect.top) / rect.height) * canvas.height;
    if (!Number.isFinite(xRaw) || !Number.isFinite(yRaw)) return null;
    const { layout, viz } = cache;
    const within = xRaw >= layout.plotX && xRaw <= layout.plotX + layout.plotW &&
      yRaw >= layout.plotY && yRaw <= layout.plotY + layout.plotH;
    if (!within && !clampToPlot) return null;
    const x = clamp(xRaw, layout.plotX, layout.plotX + layout.plotW);
    const y = clamp(yRaw, layout.plotY, layout.plotY + layout.plotH);
    return {
      p: viz.xMin + ((x - layout.plotX) / layout.plotW) * (viz.xMax - viz.xMin),
      y: viz.yMax - ((y - layout.plotY) / layout.plotH) * (viz.yMax - viz.yMin),
    };
  }

  // Pointer -> snapped {p, v}. `bounds` gives the slider min/max/step to snap to.
  function eventToPV(event, canvas, holder, bounds, opts = {}) {
    const raw = eventToAxes(event, canvas, holder, opts);
    if (!raw) return null;
    return {
      p: snapToStep(raw.p, bounds.p_watts.min, bounds.p_watts.step, bounds.p_watts.min, bounds.p_watts.max),
      v: snapToStep(raw.y, bounds.v_scan_mps.min, bounds.v_scan_mps.step, bounds.v_scan_mps.min, bounds.v_scan_mps.max),
    };
  }

  function snapToStep(value, origin, step, min, max) {
    const snapped = origin + Math.round((value - origin) / step) * step;
    const decimals = String(step).includes('.') ? String(step).split('.')[1].length : 0;
    return clamp(Number(snapped.toFixed(decimals)), min, max);
  }

  window.PVMap = {
    vizConstants, hpVizConstants, computeHStar, computeVFromHStar,
    ensureBackground, draw, eventToAxes, eventToPV, drawGtScatter, drawSplitLegend,
    hitTestGt, pvToXY, splitColor, splitStyle, SPLIT_STYLES, plasmaColorRgb, viridisColorRgb,
    roundRect, lerp, clamp, snapToStep,
  };
})();
