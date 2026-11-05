# Assumptions

Non-trivial assumptions and deliberate choices with a real effect on what
the user sees. IDs are stable; a missing number is a retired entry.

- **A1 — Geometry constants are duplicated client-side.** `path.js` hardcodes
  the substrate box (±0.75, ±0.5 mm), keep-out inset (0.2 mm), beam radius
  (50 um) and the legacy max arc length (1.5 mm, used only for a model without
  a `physics` block) to match the server. The z-slice count is NOT a client
  constant: both time pages build it from each response's `grid_shape_xyz`. Effect: server-side changes must be mirrored here or the canvas
  shows a different drawable region than the server enforces (the server
  re-clamps, so the failure is visual, not physical).
- **A2 — Freehand paths start at (-0.55, 0) mm for fixed-start models.** That
  is where their random training paths start; the green dot snaps pointerdown
  within 100 um. Models whose `/models` physics block has
  `start_anywhere: true` skip the snap (pointerdown starts the path at the
  press point, clamped into the keep-out box) and hide the green dot. Presets
  draw their own polyline and are unaffected.
- **A3 — The self-intersection barrier is client-side only.** Points within
  one beam radius of an earlier segment are rejected while drawing
  (trailing 2r exempt); the server does not enforce this. Consequence of
  bypass is an out-of-distribution prediction, not a crash.
- **A4 — The canvas mm→px scale is uniform in x and y** (640x440 px with
  20 px margins over a 1.5x1.0 mm domain). The orange barrier halo width
  (2r in px) relies on this single scale; changing canvas dimensions must
  preserve the aspect or the halo lies.
- **A5 — `backendBaseUrl` is "" (same-origin) in the committed
  `config.json`**; local direct-backend testing overrides it via the
  gitignored + dockerignored `public/config.local.json`, which the backend's
  launcher `../lpno_web_demo_backend/dev/run_local.sh` writes when it starts
  both halves.
- **A6 — Frame sets accumulate lazily per run.** Each response decodes one
  field's frames (or the 3D set) into `Image` objects; sets are cached per run
  and only fetched when viewed. A completed run whose model or server-used path
  differs from the loaded run drops all cached sets (`runKey` in `path.js`).
- **A7 — Frame count is taken from the response** (`frames.length`, slider
  sized dynamically). The straight page always gets 20; the path page gets
  5 to 100, following the weld duration and the requested frame step.
- **A9 — The straight-line tab is P/V-driven, not a preset list.** `line.js`
  sets power/scan-speed from sliders or the shared `pvmap.js` enthalpy map and
  posts `/infer_straight`. The "Compare to ground truth" toggle defaults **off**
  (free P/V anywhere on the map). When on, the split-coloured GT points from
  `/gt_catalog?dataset=straight` become the ONLY clickable targets — buttons,
  not a background you can click through: a click off any dot does nothing and
  dragging is disabled. A `Top view / Cross section` toggle sits next to 2D/3D
  and sends `view.plane` (`xy` top view at the z-slice, or `xz` cross-section at
  y-centre); the z-slice select is disabled in `xz`.
- **A10 — GT toggle defaults off; both maps are buttons when on.** The
  steady-state "Compare to ground truth" checkbox (`app.js`) starts unchecked so
  the default landing view is the unchanged 3D melt pool. When on, split-coloured
  GT points show on BOTH the P-V and H-P maps, each becomes buttons-only (clicks
  off a dot and drags do nothing), and clicking a dot on either map snaps to that
  case's exact P/V and routes the final render to `/infer/gt_compare`.
  Off-catalog P/V renders the prediction alone.
- **A13 — One map builder, one H* formula.** Both selection maps of the steady
  page and the straight page's map are drawn by `pvmap.js` (`holder.variant =
  'hp'` selects the H-P variant); `PVMap.computeHStar` / `computeVFromHStar`
  on the map's `vizConstants` (Tm − T0 = 1600 K, what the regime lines were
  anchored with) are the only H* formula in the frontend, so the H slider, the
  H-P axis, the "H*" meta row and the regime lines agree with each other.
  Effect: the server's `derived.h_star` (a 1573 K variant, 1.7 % higher) is
  not shown; only its normalised `h_norm_model` is. Every steady render is the
  full-size request (no preview mode); dragging queues a render per move and
  the page coalesces to one request in flight.
