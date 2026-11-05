/* PINO Pipeline Console — static single-page client for the aiida-pino web
 * backend. No build step: plain ES modules-free JS. Reads public/config.json
 * (+ config.local.json override written by dev/run_local.sh) to find the
 * backend, then drives the console's operations. */
(() => {
  "use strict";

  const state = { cfg: {}, baseUrl: "", env: null, activeTab: "monitor" };
  // Auto-folder for single-simulation import artifacts (server: console/folders.py).
  const ORPHANED_FOLDER = "Single-run imports";   // matches folders.ORPHANED
  const FOLDER_PAGE = 100;                       // rows shown inside one open folder
  const AUTO_AL_FOLDER = "Self-improvement (AL)";

  // ---------------------------------------------------------------- config
  async function loadConfig() {
    const cfg = await fetch("config.json", { cache: "no-store" }).then((r) => r.json()).catch(() => ({}));
    try {
      const local = await fetch("config.local.json", { cache: "no-store" });
      if (local.ok) Object.assign(cfg, await local.json());
    } catch (_) {}
    return cfg;
  }

  function url(path) { return `${state.baseUrl}${path}`; }
  function reqId() {
    try { return crypto.randomUUID(); } catch (_) { return `r-${Date.now()}-${Math.floor(Math.random() * 1e6)}`; }
  }

  // One in-flight GET per key. A browse listing takes tens of seconds on this
  // database, and without this a filter change left the old request running and
  // let whichever finished last win -- so the list could end up showing the
  // filter you had already moved away from.
  const inflight = new Map();

  async function getJson(path, opts) {
    const key = (opts && opts.key) || null;
    let signal;
    if (key) {
      const previous = inflight.get(key);
      if (previous) previous.abort();
      const controller = new AbortController();
      inflight.set(key, controller);
      signal = controller.signal;
    }
    try {
      const resp = await fetch(url(path), {
        cache: "no-store", credentials: "omit", signal,
        headers: { "X-Request-ID": reqId() },
      });
      return unwrap(resp);
    } finally {
      if (key && inflight.get(key) && inflight.get(key).signal === signal) inflight.delete(key);
    }
  }

  function aborted(error) {
    return error && (error.name === "AbortError" || error.code === 20);
  }
  async function postJson(path, body) {
    const resp = await fetch(url(path), {
      method: "POST", cache: "no-store", credentials: "omit",
      headers: { "Content-Type": "application/json", "X-Request-ID": reqId() },
      body: JSON.stringify(body || {}),
    });
    return unwrap(resp);
  }
  async function unwrap(resp) {
    let body = null;
    try { body = await resp.json(); } catch (_) { body = null; }
    if (!resp.ok) {
      // The envelope carries a correlation id that makes a failure findable in
      // the backend log; it was parsed and discarded on every error.
      const rid = body && body.server_request_id ? ` [${String(body.server_request_id).slice(0, 8)}]` : "";
      const detail = body && body.error
        ? `${body.error.code}: ${body.error.message}${rid}`
        : `HTTP ${resp.status}${rid}`;
      const err = new Error(detail);
      err.envelope = body;
      err.status = resp.status;
      throw err;
    }
    return body;
  }

  // ------------------------------------------------------------------ DOM
  function el(tag, props, ...kids) {
    const node = document.createElement(tag);
    if (props) {
      for (const [k, v] of Object.entries(props)) {
        if (v == null) continue;
        if (k === "class") node.className = v;
        else if (k === "html") node.innerHTML = v;
        else if (k === "text") node.textContent = v;
        else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
        else if (k === "value") node.value = v;
        // Booleans are properties, not attributes. `setAttribute("checked",
        // false)` sets the attribute to the string "false", which HTML reads as
        // present-and-therefore-true: every "Show ..." checkbox rendered TICKED
        // while its state was off, and the first click then appeared to do
        // nothing because it toggled the state to match what was already drawn.
        else if (typeof v === "boolean") node[k] = v;
        else node.setAttribute(k, v);
      }
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }
  const $ = (id) => document.getElementById(id);
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }
  function fmt(v) {
    if (v == null) return "—";
    if (Array.isArray(v)) return v.length ? v.join(", ") : "—";
    if (typeof v === "object") return JSON.stringify(v);
    return String(v);
  }
  // Auto-loading the next page must require a real SCROLL. Observing the sentinel alone
  // chain-loaded the entire store: each appended page re-renders the list, the fresh
  // sentinel is observed again while still in view, and it fetches again -- 21 sequential
  // requests for 2180 datasets, which was SLOWER than the single big fetch it replaced.
  //
  // So: arm on the first scroll after each page, and disarm while a fetch is in flight.
  // A list that fits the window never auto-loads, which is the case that ran away.
  function observeForMore(sentinel, isBusy, onMore) {
    if (typeof IntersectionObserver !== "function") return null;
    let scrolled = false;
    const io = new IntersectionObserver((entries) => {
      if (isBusy()) return;
      if (!entries.some((en) => en.isIntersecting)) return;
      // Not scrolled yet: the sentinel was already in view when observed, which is the
      // runaway case -- firing here would chain-load the whole store unasked.
      if (!scrolled) return;
      io.disconnect();
      window.removeEventListener("scroll", onScroll, { capture: true });
      onMore();
    }, { rootMargin: "200px" });
    // CAPTURE, because a `scroll` event does not bubble: the list scrolls inside its own
    // panel, so a plain window listener never heard it.
    //
    // RE-OBSERVE on the first scroll, which is the part that took two attempts. An
    // IntersectionObserver reports CHANGES: it fired once at attach time (the sentinel sits
    // just below the fold and is inside the 200px margin), that call was dropped because
    // nothing had scrolled yet, and no change ever followed -- so scrolling did nothing.
    // Re-observing replays the current state to the callback, now that a scroll has
    // happened.
    const onScroll = () => {
      if (scrolled) return;
      scrolled = true;
      io.unobserve(sentinel);
      io.observe(sentinel);
    };
    window.addEventListener("scroll", onScroll, { passive: true, capture: true });
    io.observe(sentinel);
    return io;
  }

  function notice(kind, msg) { return el("div", { class: `notice ${kind}` }, msg); }
  // A finished run with a non-zero exit is a failure; the green FINISHED badge
  // beside "exit 521" read as success in the list.
  function stateBadge(s, exit) {
    const err = (s || "").toLowerCase() === "finished" && exit ? " state-finished-err" : "";
    return el("span", { class: `badge state-${(s || "").toLowerCase()}${err}` }, s || "?");
  }

  // -------------------------------------------- human labels + formatting
  const KEY_LABELS = {
    pk: "PK", uuid: "UUID", label: "Label", display_name: "Name", describe: "Summary",
    description: "Description", keywords: "Keywords", split: "Split", split_counts: "Split composition",
    dataset_role: "Role", source_type: "Source", canonical_layout: "Layout",
    family_identifier: "Family ID", dataset_identifier: "Dataset ID", source_folder_name: "Source folder",
    source_display: "Source", num_timesteps: "Time steps", simulation_count: "Simulations",
    requested_fields: "Fields", stored_size: "Stored size", base_dataset_pk: "Base dataset PK",
    num_samples: "Samples", num_arrays: "Arrays", compatibility_signature: "Compatibility signature",
    V_scan: "Scan speed", laser_power: "Laser power", beam_radius: "Beam radius",
    delta_t_s: "Δt", effective_end_time_s: "End time", grid: "Grid", case_name: "Case",
    usable_for_training: "Usable for training", batch_keyword: "Batch", simulation_fingerprint: "Fingerprint",
    model_name: "Model name", config_name: "Training config", training_source_label: "Trained from",
    training_source_kind: "Training source kind", evaluation_count: "Evaluations", primary_checkpoint: "Checkpoint",
    process_type: "Process type", state: "State", exit_status: "Exit code", process_status: "Status",
    finished_ok: "Finished OK", ctime: "Created", mtime: "Updated", source_folder_pk: "Source folder PK",
    architecture: "Architecture", training_source_display: "Trained on",
    // "Training steps" was fed max_steps, which is the ceiling REQUESTED. The
    // stopping criterion stopped 31 of 158 runs short of it (worst 4000 of
    // 48000), so the two numbers are named apart.
    now_named: "Now named", max_steps: "Step budget (requested)", steps_trained: "Steps trained",
    derived_from: "Derived from", derived_sets: "Derived sets", composed_from: "Composed from",
    regime: "Melt regime", melt_vol_um3: "Melt volume", vol_um3: "Depression volume",
    depth_um: "Depression depth", xlen_um: "Depression length (x)", ylen_um: "Depression width (y)",
    melt_depth_um: "Melt depth", aspect_depth_over_ylen: "Aspect (depth/width)",
    cavity_rmin_um: "Cavity min radius", cavity_rmax_um: "Cavity max radius",
    cavity_rmin_um_mean: "Cavity min radius (mean)",
    melt_depth_um_mean: "Melt depth (mean)", melt_depth_um_max: "Melt depth (max)",
    tmax_k: "Peak temperature", H: "Normalized enthalpy H", peak_frame: "Peak frame",
    keyhole_fraction: "Keyhole fraction", computed: "Cases computed", total: "Cases",
    regime_counts: "Regimes", melt_vol_um3_mean: "Melt volume (mean)", melt_vol_um3_max: "Melt volume (max)",
    depth_um_mean: "Depression depth (mean)", depth_um_max: "Depression depth (max)",
    tmax_k_mean: "Peak T (mean)", tmax_k_max: "Peak T (max)",
  };
  function keyLabel(k) { return KEY_LABELS[k] || String(k).replace(/_/g, " "); }
  // 263.97427102428 -> 264, 1.0807692307692 -> 1.081; non-numbers pass through.
  function sig4(v) { const n = Number(v); return Number.isFinite(n) ? parseFloat(n.toPrecision(4)) : v; }
  function fmtValue(key, v) {
    if (v == null || v === "") return "—";
    if (key === "laser_power") return `${sig4(v)} W`;
    if (key === "V_scan") return `${sig4(v)} m/s`;
    if (key === "beam_radius") return `${Math.round(Number(v) * 1e9) / 1e3} µm`;
    if (key === "delta_t_s" || key === "effective_end_time_s") return `${v} s`;
    if (key === "tmax_k" || key === "tmax_k_mean" || key === "tmax_k_max") return `${Math.round(v)} K`;
    if (/_um$/.test(key) || /_um_(mean|max)$/.test(key)) return `${Number(v).toFixed(1)} µm`;
    if (/_um3/.test(key)) return `${(Number(v) / 1e6).toFixed(2)}×10⁶ µm³`;
    if (key === "H" || key === "aspect_depth_over_ylen" || key === "keyhole_fraction") return Number(v).toFixed(2);
    if ((key === "ctime" || key === "mtime") && typeof v === "string") {
      const d = new Date(v);
      if (!Number.isNaN(d.getTime())) return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
    }
    return fmt(v);
  }
  // Friendly names for aiida-pino process classes (raw class kept as tooltip).
  const PROCESS_LABELS = {
    RawFlow3DImportWorkChain: "FLOW-3D NPZ import", RawFlow3DToDatasetCalculation: "FLOW-3D NPZ conversion",
    RawOpenFoamImportWorkChain: "OpenFOAM import", RawOpenFoamToDatasetCalculation: "OpenFOAM conversion",
    Flow3DIngestWorkChain: "FLOW-3D VTK ingest", Flow3DToVtkCalculation: "FLOW-3D VTK conversion",
    Flow3DGenerateWorkChain: "FLOW-3D run + ingest", Flow3DRunCalculation: "FLOW-3D solver run",
    Flow3DBatchWorkChain: "FLOW-3D batch", VtkToDatasetCalculation: "VTK conversion",
    TrainModelWorkChain: "Train model", TrainEvalWorkChain: "Train + evaluate",
    EvalOnlyWorkChain: "Evaluate model", KfoldTrainEvalWorkChain: "K-fold train/eval",
    PinoTrainCalculation: "Training job", PinoEvalCalculation: "Evaluation job",
    ImportModelWorkChain: "Import model", DeriveDatasetWorkChain: "Derive dataset",
    WarmRuntimeCacheCalculation: "Warm runtime cache", compose_dataset: "Compose dataset",
    materialize_derived_dataset: "Derive dataset (record)",  // matched on the label's last segment
    LaserbeamFoamSteadyGenerateWorkChain: "LaserbeamFoam steady run",
    LaserbeamFoamSteadyBatchWorkChain: "LaserbeamFoam steady batch",
    LaserbeamFoamSteadyRunCalculation: "LaserbeamFoam solver run",
    "_compose_dataset_calcfunction.<locals>.compose_dataset": "Compose dataset",
    materialize_model_node: "Materialize model", materialize_evaluation_node: "Materialize evaluation",
    EvaluateModelWorkChain: "Evaluate model",
  };
  // A calcfunction defined inside a function carries its scope in the label
  // (`compose_dataset_calcfunction.<locals>.compose_dataset`): match the last segment too.
  function processName(raw) {
    if (!raw) return "?";
    return PROCESS_LABELS[raw] || PROCESS_LABELS[String(raw).split(".").pop()] || raw;
  }
  function fmtStepSize(s) {
    return s >= 1e-3 ? `${Number((s * 1e3).toPrecision(3))} ms` : `${Number((s * 1e6).toPrecision(3))} µs`;
  }
  // Two-line clamped text cell; click toggles the full text.
  function clampEl(text) {
    const d = el("div", { class: "clamp", title: "click to expand" }, text || "—");
    d.addEventListener("click", (e) => { e.stopPropagation(); d.classList.toggle("open"); });
    return d;
  }
  const REGIME_COLORS = { LoF: "#c44e52", conduction: "#4c72b0", keyhole: "#55a868" };
  function regimeBadge(reg) {
    if (!reg) return el("span", { class: "muted small" }, "—");
    return el("span", { class: "badge", style: `background:${REGIME_COLORS[reg] || "#888"}22; color:${REGIME_COLORS[reg] || "#555"}` }, reg);
  }
  function splitChips(counts) {
    const box = el("span", { class: "chips" });
    for (const k of ["train", "val", "test", "unassigned"]) {
      if (counts && counts[k]) box.append(el("span", { class: `chip split-${k}`, title: `${counts[k]} ${k}` }, `${counts[k]} ${k}`));
    }
    if (!box.childNodes.length) box.append(el("span", { class: "muted small" }, "—"));
    return box;
  }

  // -------------------------------------------------------------- env bar
  function daemonOk() { return !!(state.env && state.env.daemon && state.env.daemon.available); }

  function renderEnvStrip() {
    const strip = clear($("envStrip"));
    if (!state.env) return;
    const e = state.env;
    const d = e.daemon || {};
    strip.append(
      el("span", { class: "env-item" }, "profile ", el("strong", {}, fmt(e.profile))),
      el("span", { class: "env-item" }, "pino ", el("strong", {}, fmt(e.lpno_version))),
      el("span", { class: "env-item" },
        "daemon ",
        el("span", { class: `badge ${d.available ? "ok" : "warn"}`, title: d.reason || "" }, d.available ? "up" : "down")
      )
    );
  }

  // ----------------------------------------------------------------- tabs
  const TABS = [
    { id: "monitor", label: "Monitor" },
    { id: "environment", label: "Environment" },
    { id: "datasets", label: "Datasets" },
    { id: "compose", label: "Compose new dataset" },
    { id: "models", label: "Models" },
    { id: "runs", label: "Runs" },
    { id: "logs", label: "Logs" },
  ];

  function renderTabbar() {
    const bar = clear($("tabbar"));
    for (const t of TABS) {
      bar.append(el("button", {
        class: state.activeTab === t.id ? "is-active" : "",
        "data-tab": t.id,
        onclick: () => selectTab(t.id),
      }, t.label));
    }
  }
  function selectTab(id, pushHash = true) {
    state.activeTab = id;
    if (pushHash && location.hash !== `#${id}`) history.replaceState(null, "", `#${id}`);
    renderTabbar();
    const view = clear($("view"));
    (RENDERERS[id] || (() => view.append(notice("warn", "not implemented"))))(view);
  }
  window.addEventListener("hashchange", () => {
    const id = location.hash.slice(1);
    if (RENDERERS[id] && id !== state.activeTab) selectTab(id, false);
  });

  // -------------------------------------------------------------- helpers
  function table(rows, columns, onRow, selectedPk) {
    if (!rows || !rows.length) return el("div", { class: "detail-empty" }, "No rows.");
    const hasWidths = columns.some((c) => c.w);
    const thead = el("tr", {}, ...columns.map((c) => el("th", {}, c.label)));
    const body = el("tbody");
    for (const r of rows) {
      const tr = el("tr", {
        class: [r._rowClass || "", onRow ? "clickable" : "", selectedPk != null && r.pk === selectedPk ? "is-selected" : ""].join(" ").trim() || null,
        onclick: onRow ? () => onRow(r, tr) : null,
      }, ...columns.map((c) => {
        const v = c.get ? c.get(r) : r[c.key];
        const rendered = v && v.nodeType ? v : (c.clamp ? clampEl(fmt(v)) : fmtValue(c.key, v));
        return el("td", { class: [c.num ? "num" : "", c.nowrap ? "nowrap" : ""].join(" ").trim() || null }, rendered);
      }));
      body.append(tr);
    }
    const tbl = el("table", { class: hasWidths ? "fixed" : null },
      hasWidths ? el("colgroup", {}, ...columns.map((c) => el("col", { style: c.w ? `width:${c.w}` : null }))) : null,
      el("thead", {}, thead), body);
    return el("div", { class: "table-wrap" }, tbl);
  }
  function kv(obj, keys) {
    const dl = el("dl", { class: "kv" });
    const entries = keys ? keys.map((k) => [k, obj[k]]) : Object.entries(obj || {});
    for (const [k, v] of entries) {
      if (v === undefined) continue;
      const dd = (v && v.nodeType) ? v
        : (v && typeof v === "object") ? el("span", { class: "mono" }, fmt(v))
        : fmtValue(k, v);
      dl.append(el("dt", {}, keyLabel(k)), el("dd", {}, dd));
    }
    return dl;
  }
  function jsonBlock(obj) { return el("pre", { class: "json" }, JSON.stringify(obj, null, 2)); }
  function codeSelect(role) {
    const codes = (state.env && state.env.codes && state.env.codes.all) || [];
    const opts = codes
      .filter((c) => !role || c.role === role)
      .map((c) => el("option", { value: c.label }, `${c.label} (${c.role})`));
    if (!opts.length) opts.push(el("option", { value: "" }, "— no code —"));
    return el("select", {}, ...opts);
  }
  async function datasetOptions(text) {
    // pickers need pk + name, not member stats
    const b = await getJson("/datasets/options").catch(() => ({ datasets: [] }));
    const byName = (d) => `#${d.pk} · ${d.display_name || d.label}${d.simulation_count != null ? ` (${d.simulation_count} sims)` : ""}`;
    return (b.datasets || []).map((d) => el("option", { value: d.pk, title: d.label || "" }, (text || byName)(d)));
  }
  async function modelOptions() {
    const b = await getJson(`/models?limit=${state.cfg.listLimit || 100}`).catch(() => ({ models: [] }));
    return (b.models || []).map((m) => el("option", { value: m.pk, title: m.label || "" }, `#${m.pk} · ${m.model_name || m.label}`));
  }
  function submitGate(container) {
    if (daemonOk()) return true;
    // "Unknown" and "down" are different states and used to print the same
    // sentence. One slow /env (a saturated pool answers in 33 s and 500s) set
    // state.env = null and every submit button in the app rendered disabled
    // under "AiiDA daemon unavailable" — a cause nothing had checked — and
    // stayed that way until F5, because /env is only re-read on page load.
    if (!state.env) {
      const retry = el("button", { class: "button small", onclick: async (e) => {
        e.target.disabled = true;
        await refreshEnv();
        selectTab(state.activeTab, false);
      } }, "Retry");
      container.append(notice("warn", el("span", {},
        "Environment not loaded — the backend did not answer /env. Submissions are held until it does. ", retry)));
      return false;
    }
    const reason = (state.env.daemon && state.env.daemon.reason) || "AiiDA daemon unavailable";
    container.append(notice("warn",
      `Submissions are disabled: ${reason}. Browsing and monitoring still work.`));
    return false;
  }
  // The developer notes ask for a title and a description on every AiiDA
  // process, and the console offered annotation only as a post-hoc prompt() on
  // the run detail pane — so every run it launched was created with
  // title:null, description:null and the Runs list was a column of identical
  // `Flow3DGenerateWorkChain` rows, which is the exact problem that rule exists
  // to prevent. These are passed to runSubmit and applied to the new pk.
  function titleFields(placeholder) {
    const title = el("input", { type: "text", placeholder: placeholder || "short name for the Runs list" });
    const description = el("input", { type: "text", placeholder: "what and why (one line)" });
    return {
      title, description,
      node: el("div", { class: "form-grid" }, field("Run title", title), field("Notes", description)),
      get: () => ({ title: title.value.trim(), description: description.value.trim() }),
    };
  }
  async function runSubmit(btn, out, fn, annotate) {
    clear(out);
    btn.disabled = true;
    const prev = btn.textContent;
    btn.textContent = "Submitting…";
    try {
      const res = await fn();
      out.append(notice("ok", `Submitted pk ${res.pk} — ${res.label || ""}. Track it in the Runs tab.`));
      const marks = annotate ? annotate() : null;
      if (res.pk && marks && (marks.title || marks.description)) {
        try { await postAnnotations(res.pk, marks); }
        catch (e) { out.append(notice("warn", `submitted, but the title did not save: ${e.message}`)); }
      }
      (res.notes || []).forEach((n) => out.append(notice(/honoured/.test(n) ? "ok" : "warn", n)));
    } catch (e) {
      const envelope = (e.envelope || {}).error || {};
      out.append(notice("err", envelope.retryable
        ? `${e.message} (retryable${envelope.retry_after_ms ? ` in ~${Math.round(envelope.retry_after_ms / 1000)} s` : ""})`
        : e.message));
      // On a 5xx the request reached the server and its outcome is unknown;
      // there is no idempotency key on this path, so a second click is how you
      // end up with two runs. Keep the button down until the user has looked.
      if (e.status >= 500) {
        out.append(notice("warn", "The run may or may not have been created — check the Runs tab before resubmitting."));
        btn.textContent = "Check Runs tab";
        return;
      }
    } finally {
      if (btn.textContent !== "Check Runs tab") {
        btn.disabled = false;
        btn.textContent = prev;
      }
    }
  }

  // ------------------------------------------------------- tab: environment
  async function renderEnvironment(view) {
    view.append(el("div", { class: "panel stack" },
      el("div", { class: "row" }, el("h2", {}, "Environment"), el("span", { class: "spacer" }),
        el("button", { class: "button small", onclick: () => refreshEnv().then(() => selectTab("environment")) }, "Refresh")),
      state.env ? envBody() : notice("warn", "environment not loaded")));

    function envBody() {
      const e = state.env;
      const daemonLine = `${e.daemon.available ? "available" : "unavailable"} — ${e.daemon.reason}`;
      if (e.degraded) {
        return el("div", { class: "stack" },
          notice("warn", "/env failed — showing /healthz daemon status only; press Refresh"),
          kv({ daemon: daemonLine }));
      }
      const isRetired = (r) => /^RETIRED[-_]/i.test(r.label || "");
      const codeRows = ((e.codes && e.codes.all) || []).map((r) => (isRetired(r) ? { ...r, _rowClass: "muted" } : r));
      return el("div", { class: "stack" },
        kv({
          profile: e.profile,
          "lpno version": e.lpno_version,
          daemon: daemonLine,
        }),
        el("div", {}, el("h3", {}, "Registered codes"),
          table(codeRows, [
            { label: "pk", key: "pk", num: true },
            { label: "label", get: (r) => (isRetired(r)
              ? el("span", {}, r.label, " ", el("span", { class: "chip split-unassigned" }, "retired"))
              : r.label) },
            { label: "role", key: "role" },
            { label: "executable", key: "executable" },
          ])),
        el("div", {}, el("h3", {}, "Packaged training configs"),
          el("div", { class: "chips" }, ...(e.config_names || []).map((n) => el("span", { class: "chip mono" }, n)))));
    }
  }

  // ---------------------------------------------------------- tab: datasets
  function browseTab(view, opts) {
    // Generic list + detail browser used by datasets & models.
    const sel = { pk: null };
    const controls = el("div", { class: "controls-grid" });
    const listBox = el("div", {}, el("div", { class: "detail-empty" }, "Loading…"));
    const detail = el("div", { class: "panel detail-pane" }, el("div", { class: "detail-empty" }, "Select a row."));
    opts.controls(controls, reload);
    view.append(el("div", { class: "split" },
      el("div", { class: "panel stack" },
        el("div", { class: "row" }, el("h2", {}, opts.title), el("span", { class: "spacer" }),
          el("button", { class: "button small", onclick: reload }, "Refresh")),
        controls, listBox),
      detail));

    async function reload() {
      clear(listBox).append(el("div", { class: "detail-empty" }, "Loading…"));
      try {
        const rows = await opts.fetchRows();
        clear(listBox);
        // A capped list must say so: silently showing a page of a larger store
        // is how whole datasets went missing without anyone noticing.
        const status = opts.status && opts.status(rows);
        if (status) listBox.append(el("div", { class: "muted small", style: "padding:.25rem .1rem" }, status));
        listBox.append(table(rows, opts.columns, async (r, tr) => {
          if (r._onRow) { r._onRow(); return; }  // group headers toggle, they have no detail
          sel.pk = r.pk;
          [...listBox.querySelectorAll("tr.is-selected")].forEach((n) => n.classList.remove("is-selected"));
          tr.classList.add("is-selected");
          clear(detail).append(el("div", { class: "detail-empty" }, "Loading…"));
          try { clear(detail).append(await opts.renderDetail(r.pk)); }
          catch (e) { clear(detail).append(notice("err", e.message)); }
        }, sel.pk));
        // `loadMore` is optional: a list that provides one gets a sentinel after the table
        // and fetches the next page when it scrolls into view. Lists without one behave
        // exactly as before.
        if (opts.loadMore && opts.hasMore && opts.hasMore()) {
          const sentinel = el("div", { class: "muted small pager-sentinel" },
            opts.moreLabel ? opts.moreLabel() : "scrolling loads more");
          listBox.append(sentinel);
          let busy = false;
          observeForMore(sentinel, () => busy, async () => {
            busy = true;
            sentinel.textContent = "Loading more\u2026";
            await opts.loadMore();
            reload();
          });
        }
      } catch (e) {
        clear(listBox).append(notice("err", e.message));
      }
    }
    reload();
  }

  // Multi-select filing, shared by the Datasets and Models tabs. `picked` holds
  // node pks only (never a "folder:…" header pseudo-pk) and survives filter
  // changes on purpose: filter, tick, clear the filter, tick more. `matched` is
  // what the current filters show, so the bar can admit how much of the
  // selection is out of sight instead of quietly moving it.
  function folderTools({ reload, rows, autoFolder }) {
    const picked = new Set();
    let matched = new Set();
    let lastMove = null;   // { prev: Map(pk -> folder before) } for Undo
    let bar = null, count = null, input = null, msg = null, dl = null;

    // Every clone of a pk ticks together, without a redraw that would lose the
    // scroll position; folder headers follow their members.
    function syncBoxes(pks, on) {
      for (const pk of pks) {
        for (const box of document.querySelectorAll(`input.pick[data-pk="${pk}"]`)) {
          box.checked = on;
          box.indeterminate = false;
        }
      }
      for (const box of document.querySelectorAll("input.pick[data-member-pks]")) {
        const members = box.getAttribute("data-member-pks").split(",").filter(Boolean).map(Number);
        const all = members.length > 0 && members.every((pk) => picked.has(pk));
        box.checked = all;
        box.indeterminate = !all && members.some((pk) => picked.has(pk));
      }
    }
    function updateBar() {
      if (!bar) return;
      const hidden = [...picked].filter((pk) => !matched.has(pk)).length;
      // With nothing selected the bar survives as a message strip, because it
      // carries Undo — hiding it outright would take the undo with it.
      bar.hidden = picked.size === 0 && !lastMove;
      bar.classList.toggle("msg-only", picked.size === 0);
      count.textContent = `${picked.size} selected` + (hidden ? ` (${hidden} hidden by filter)` : "");
    }
    async function refreshFolderList() {
      if (!dl) return;
      try {
        const body = await getJson("/folders");
        clear(dl).append(...(body.folders || []).map((name) => el("option", { value: name })));
      } catch (e) { /* suggestions are a convenience; typing still works */ }
    }
    // Mirror a confirmed server move onto the cached rows: the POST either
    // moved every pk or raised (and the error path refetches), so this is not
    // an optimistic write, and it skips a multi-second refetch per move.
    function applyFolder(pks, path) {
      const wanted = new Set(pks);
      for (const row of rows() || []) {
        if (wanted.has(row.pk)) row.folder = path || autoFolder(row);
      }
    }
    async function move(path) {
      const pks = [...picked];
      if (!pks.length) return;
      const clean = (path || "").trim().replace(/^\/+|\/+$/g, "");
      // The computed folders are not stored; a real folder of the same name
      // would render identically and behave differently.
      if (clean === ORPHANED_FOLDER || clean === AUTO_AL_FOLDER || /^keyword:/i.test(clean)) {
        clear(msg).append(el("span", {}, `"${clean}" is a computed folder — pick another name.`));
        return;
      }
      const prev = new Map(pks.map((pk) => [pk, (rows() || []).find((r) => r.pk === pk)?.folder || ""]));
      clear(msg).append(el("span", {}, "moving…"));
      try {
        await postJson("/folders/move", { pks, path: clean });
      } catch (err) {
        alert(err.message);          // keep the selection: a partial move shows up in the reload
        clear(msg);
        reload(false);
        return;
      }
      lastMove = { prev };
      picked.clear();
      input.value = "";
      applyFolder(pks, clean);
      await refreshFolderList();
      reload(true, clean);
      clear(msg).append(el("span", {}, `Moved ${pks.length} → ${clean || "(unfiled)"}`),
        el("button", { class: "button small", onclick: undo }, "Undo"));
      updateBar();
    }
    async function undo() {
      if (!lastMove) return;
      const byFolder = new Map();
      for (const [pk, folder] of lastMove.prev) {
        // A computed folder was never stored: undoing into it means unfiling.
        const target = (folder === ORPHANED_FOLDER || folder === AUTO_AL_FOLDER) ? "" : folder;
        if (!byFolder.has(target)) byFolder.set(target, []);
        byFolder.get(target).push(pk);
      }
      try {
        for (const [target, pks] of byFolder) {
          await postJson("/folders/move", { pks, path: target });
          applyFolder(pks, target);
        }
      } catch (err) { alert(err.message); }
      lastMove = null;
      await refreshFolderList();
      reload(true);
      clear(msg);
      updateBar();
    }
    function buildBar() {
      dl = document.getElementById("folder-names")
        || document.body.appendChild(el("datalist", { id: "folder-names" }));
      count = el("span", { class: "bulk-count" }, "0 selected");
      input = el("input", { class: "folder-input", list: "folder-names", "aria-label": "Target folder",
        placeholder: "folder or folder/sub… (a new name creates it)",
        onkeydown: (e) => { if (e.key === "Enter") move(input.value); } });
      msg = el("span", { class: "ds-sub bulk-msg" });
      bar = el("div", { class: "bulk-bar", hidden: "hidden" },
        count, input,
        el("button", { class: "button small primary", onclick: () => move(input.value) }, "Move"),
        el("button", { class: "button small", title: "remove these from their folder",
          onclick: () => move("") }, "Unfile"),
        // The bar could only file. Tagging a selection one row at a time is the
        // thing a campaign actually needs, and the endpoint was already there.
        el("button", { class: "button small", title: "add one keyword to every selected row",
          onclick: async () => {
            const keyword = prompt("Keyword to add to the selected rows");
            if (!keyword) return;
            const pks = [...picked];
            let done = 0;
            for (const pk of pks) {
              try { await postAnnotations(pk, { keywords_add: [keyword.trim()] }); done += 1; }
              catch (err) { msg.textContent = `${done} tagged, then: ${err.message}`; break; }
            }
            if (done === pks.length) msg.textContent = `tagged ${done} row(s) "${keyword.trim()}"`;
            reload(true);
          } }, "Add keyword…"),
        el("button", { class: "button small", onclick: () => {
          picked.clear();
          lastMove = null;
          clear(msg);
          for (const box of document.querySelectorAll("input.pick")) { box.checked = false; box.indeterminate = false; }
          updateBar();
        } }, "Clear"),
        msg);
      refreshFolderList();
      return bar;
    }
    // The selection column. `memberPks` turns a folder header into a
    // select-all-in-folder tick; a plain row selects itself.
    function pickColumn(memberPks) {
      return { label: "", w: "3%", get: (r) => {
        const pks = memberPks(r);
        if (!pks.length) return el("span");
        const on = pks.every((pk) => picked.has(pk));
        const box = el("input", {
          type: "checkbox", class: "pick",
          "data-pk": pks.length === 1 && pks[0] === r.pk ? String(r.pk) : null,
          "data-member-pks": pks.length === 1 && pks[0] === r.pk ? null : pks.join(","),
          "aria-label": `Select ${r.display_name || r.model_name || r.label || r.pk}`,
          checked: on ? "checked" : null,
          onclick: (e) => {
            e.stopPropagation();     // the row click still opens the detail pane
            const add = e.target.checked;
            for (const pk of pks) add ? picked.add(pk) : picked.delete(pk);
            syncBoxes(pks, add);
            updateBar();
          },
        });
        if (!on && pks.some((pk) => picked.has(pk))) box.indeterminate = true;
        return box;
      } };
    }
    return {
      picked, buildBar, pickColumn, updateBar,
      setMatched: (pks) => { matched = pks; updateBar(); },
    };
  }

  function renderDatasets(view) {
    // Every role by default. Derived views (steady / resampled / downsampled)
    // fold under their base dataset behind a \u25b8 toggle; quick-val subsets
    // nest one level deeper. Role picks one role and flattens to that list.
    const f = { search: "", role: "all", source: "all", group: "", showAll: false,
      superseded: false, showSplits: false };
    const expanded = new Set();
    let cache = null, keepCache = false, reloadFn = null, groupSel = null;
    // Page size for browsing. `dsOffset` is how many are already loaded; `loadMore`
    // advances it and the shared list component re-fetches.
    const DATASET_PAGE = 60;
    let dsOffset = 0;
    // Set ONLY by `loadMore`. Every other refetch -- a filter change, a keyword click, a
    // Refresh -- is a new query and must start at the top; inferring that from `dsOffset`
    // alone would have made a filter change continue from wherever the scroll had reached.
    let dsAppending = false;
    // Server-computed aggregates for this payload. Declared: an assignment to an
    // undeclared name is a ReferenceError under "use strict", which took the
    // whole Datasets tab down for the four commits after 890865d.
    let folderTotals = {}, keywordTotals = {};
    let datasetTotal = 0, datasetTruncated = false;
    // Keywords shared by at least two datasets are the ones worth grouping by.
    const refreshGroupOptions = (rows) => {
      if (!groupSel) return;
      // The server counts over the whole matching population. Counting the
      // loaded page offered 11 keywords instead of 21, with `self-improve` as
      // 15 against 65, and left `conduction` (10) and `keyhole` (9) out.
      const counts = new Map(Object.entries(keywordTotals || {}));
      if (!counts.size) for (const r of rows) for (const k of r.keywords || []) counts.set(k, (counts.get(k) || 0) + 1);
      const keys = [...counts.entries()].filter(([, n]) => n > 1)
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
      clear(groupSel).append(el("option", { value: "" }, "— none —"),
        ...keys.map(([k, n]) => el("option", { value: k }, `${k} (${n})`)));
      if (keys.some(([k]) => k === f.group)) groupSel.value = f.group; else f.group = "";
    };
    const picker = folderTools({
      rows: () => cache,
      autoFolder: (row) => ((row.simulation_count || 0) <= 1 ? ORPHANED_FOLDER
        : (row.keywords || []).includes("self-improve") ? AUTO_AL_FOLDER : null),
      reload: (fromCache, openFolder) => {
        if (openFolder) expanded.add(`folder:${openFolder}`);
        keepCache = !!fromCache;
        reloadFn();
      },
    });
    browseTab(view, {
      title: "Datasets",
      controls(c, reload) {
        reloadFn = reload;
        const instant = (apply) => (v) => { apply(v); keepCache = true; reload(); };
        groupSel = el("select", { title: "Fold every dataset carrying this keyword into one collapsible group",
          onchange: (e) => instant((v) => { f.group = v; })(e.target.value) });
        c.append(
          field("Search", el("input", {
            type: "search",
            placeholder: "search all datasets",
            // Server-side: `?search=` filters the whole store (68 matches for
            // "alsi"), while filtering the fetched page found 27 of 79 because
            // the default fetch is 500 of 810 rows. Dropping the cache forces
            // the refetch that makes the search authoritative.
            oninput: (e) => {
              f.search = e.target.value.toLowerCase();
              cache = null;
              reload();
            },
          })),
          field("Role", selectVals2([["all", "all roles"], ["base", "base only"], ["composed", "composed only"],
            ["derived", "derived only"], ["subset", "subsets / splits"]],
            instant((v) => { f.role = v; }), f.role)),
          field("Source", selectVals(["all", "flow3d"], instant((v) => { f.source = v; }))),
          field("Group by keyword", groupSel),
          // The list is capped by default so a first paint stays fast; without a
          // way to turn that off, older sets (the core reference ones among
          // them) are simply unreachable by scrolling. limit=0 = no cap.
          field("Show all", el("input", {
            type: "checkbox", checked: f.showAll,
            title: "List every dataset instead of the newest page",
            onchange: (e) => { f.showAll = e.target.checked; cache = null; reload(); },
          })),
          // Supersession retires about half the collections and hides them
          // everywhere, so a replaced dataset was indistinguishable from one
          // that never existed.
          // Auto splits are views of datasets already in the list, so they are
          // off by default; they were 17 rows of clutter above the real sets.
          field("Show splits", el("input", {
            type: "checkbox", checked: f.showSplits,
            title: "Also list the automatic train/val/test split subsets",
            onchange: (e) => { f.showSplits = e.target.checked; reload(); },
          })),
          field("Show replaced", el("input", {
            type: "checkbox", checked: f.superseded,
            title: "Also list datasets that were replaced by a newer node",
            onchange: (e) => { f.superseded = e.target.checked; cache = null; reload(); },
          })),
          picker.buildBar());
      },
      // `rows` is what the table renders: data rows AND collapsed folder
      // headers. Counting them as datasets read "19 of 810" for a view covering
      // 489, and expanding one folder moved the number to 427 with no change to
      // the underlying set. Count the datasets a folder stands for, not the
      // header, and keep "in view" separate from "loaded".
      // A collapsed folder stands for its loaded members, an open one for none (its
      // members are rows of their own, so counting both read 2815 -> 2894 on one
      // click), and a "+N more" row for the N it hides.
      status: (rows) => {
        const inView = rows.reduce(
          (n, r) => n + (r._group ? (r._open ? 0 : r._loaded) : (r._hidden ?? 1)), 0);
        const loaded = cache ? cache.length : inView;
        const parts = [`${inView} dataset${inView === 1 ? "" : "s"} in view`];
        if (loaded !== inView) parts.push(`${loaded} loaded`);
        if (datasetTotal && datasetTotal !== loaded) parts.push(`${datasetTotal} in the store`);
        return parts.join(" · ") + (datasetTruncated ? ' — tick "Show all" to load every one' : "");
      },
      // NO `loadMore` here on purpose. Paging this tab by ctime broke its TREE: the
      // Datasets view nests folders -> base datasets -> derived children, and a base whose
      // derived sets are older than the page boundary rendered as an empty expander. The
      // headless render check caught it as four separate structural failures ("expanding
      // the datasets table showed no nested rows", "no depth-2 subset rows", ...). The
      // backend still accepts `offset` and is tested, so a tree-aware pager -- one that
      // pulls a base's children with it -- can use it without further server work.
      fetchRows: async () => {
        if (!cache || !keepCache) {
          // Paged unless the caller explicitly asked for everything. "Show all" and a
          // search both mean the whole store on purpose, and the server answers those
          // uncapped; browsing takes 60 at a time, which it answers in ~1 s against ~19 s
          // for all 2180.
          const wantsAll = true;  // see the note on loadMore above
          const paged = !wantsAll;
          if (!dsAppending) dsOffset = 0;
          // limit=0 means "no cap"; omit the parameter entirely otherwise —
          // an empty `limit=` is not an int and 422s.
          const body = await getJson(`/datasets?role=all${f.showAll ? "&limit=0" : ""}`
            + `${f.superseded ? "&include_superseded=1" : ""}`
            // A search must cover the whole store, so it is never capped.
            + `${f.search ? `&search=${encodeURIComponent(f.search)}&limit=0` : ""}`
            + `${paged ? `&limit=${DATASET_PAGE}&offset=${dsOffset}` : ""}`,
            { key: "datasets" });
          const incoming = body.datasets || [];
          if (paged && dsAppending && dsOffset > 0) {
            // Append; de-duplicate by pk so a dataset created mid-scroll cannot repeat.
            const seen = new Set((cache || []).map((r) => r.pk));
            cache = (cache || []).concat(incoming.filter((r) => !seen.has(r.pk)));
          } else {
            cache = incoming;
          }
          folderTotals = body.folders || {};
          keywordTotals = body.keywords || {};
          datasetTotal = body.total ?? cache.length;
          datasetTruncated = Boolean(paged ? (body.has_more ?? body.truncated) : body.truncated);
        }
        keepCache = false;
        dsAppending = false;
        refreshGroupOptions(cache);
        // An auto training split is a view of a dataset that is already in the
        // list; 17 of them pushed the real datasets off the screen. Hidden by
        // default, still reachable under their base or via "Show splits".
        // The prefix must be the *auto* folder: matching "Training splits"
        // alone also hid 12 hand-made sets that live in a folder of that name
        // ("Original straight path data train split", 65 sims, and friends).
        // The flattened folder counts too. `startsWith("Training splits (auto)")` stopped
        // matching when the 2026-09-01 curation collapsed those subfolders into one flat
        // `Training splits`, so the folder rendered 27 of its 156 sets -- and the 27 that
        // showed were the hand-made leftovers, not the splits.
        const isAutoSplit = (r) => !!r.subset_split
          || String(r.folder || "").startsWith("Training splits (auto)")
          || String(r.folder || "") === "Training splits";
        // No client-side search clause: the query was already answered by the
        // server over the whole store, and the two searched different fields.
        // The server matches label/family/source folder/folder, the client
        // matched display_name/label/source folder/description, so every row
        // matched only on a server-only field was fetched and then thrown away:
        // "Core reference" returned 6 rows and rendered 0, "Self-improvement"
        // 66 and 0. Re-filtering a filtered result can only subtract.
        const match = (r) => (f.showSplits || f.role === "subset" || !isAutoSplit(r))
          && (f.role === "all" || r.dataset_role === f.role)
          && (f.source === "all" || r.source_type === f.source);
        const byPk = new Map(cache.map((r) => [r.pk, r]));
        // A dataset is top-level unless it hangs off a listed parent. Filing it
        // somewhere its base is not pulls it back out to that folder, so an
        // explicit move is never silently swallowed by inheritance.
        const isTop = (r) => {
          const parent = byPk.get(r.parent_pk);
          if (!parent) return true;
          // Only a *recorded* parent may demote a dataset out of its folder.
          // `inferred` and `name map` are display guesses from member overlap,
          // and they were hiding real datasets: core 306 and 3847 sit in "Core
          // reference" but were nested under a guessed parent, so the folder
          // read "2 datasets" when it holds four.
          if (r.parent_source !== "recorded") return true;
          return !!r.folder_explicit && r.folder !== parent.folder;
        };
        // Children = anything that records this row as its parent, whatever its
        // role (a composed set can have a derived child), plus the source
        // datasets a composed set was combined from. A row can appear under
        // several parents — rows are cloned for display, the data is shared.
        const kids = (r) => {
          const lineage = cache.filter((c) => c.parent_pk === r.pk && c.pk !== r.pk);
          const comps = (r.component_pks || []).map((pk) => byPk.get(pk)).filter(Boolean);
          // Deduped: a source dataset of a composed set is usually ALSO inferred
          // to be its child, so the naive concatenation rendered it twice.
          // Dataset 1601 showed "Ti64 random keyhole paths" twice and its
          // conduction half not at all, which reads as a corrupt composition.
          const seen = new Set();
          return [...comps, ...lineage].filter((c) => !seen.has(c.pk) && seen.add(c.pk));
        };
        const subtreeMatch = (r, d = 0) => match(r) || (d < 2 && kids(r).some((c) => subtreeMatch(c, d + 1)));
        const ordered = [];
        const matchedPks = new Set();
        // Folders (AiiDA groups) are the structure: each becomes a collapsible
        // header, unfiled datasets stay at the top level, and the auto
        // "Orphaned" folder of single-simulation import artifacts sits last and
        // closed. A chosen keyword acts as a virtual folder over the same
        // machinery. `off` shifts the indent of a grouped subtree without
        // touching its lineage depth.
        const folderOf = (r) => (f.group && (r.keywords || []).includes(f.group))
          ? `keyword: ${f.group}` : (r.folder || "");
        const layout = (tops, emit) => {
          // Folder paths are `/`-separated, and every segment is a real folder:
          // "Training splits (auto)/X (#1601)" lives *inside* "Training splits
          // (auto)". Grouping by the whole string made the two siblings, so a
          // folder and its children looked like unrelated duplicates and moving
          // a folder into another read as a rename.
          const groups = new Map();
          const loose = [];
          for (const r of tops) {
            const key = folderOf(r);
            if (!key) { loose.push(r); continue; }
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key).push(r);
          }
          const nodes = new Map();
          const ensure = (path) => {
            if (!nodes.has(path)) nodes.set(path, { children: new Set(), members: [] });
            return nodes.get(path);
          };
          for (const [path, members] of groups) {
            const segments = path.split("/").map((s) => s.trim()).filter(Boolean);
            let acc = "";
            for (const segment of segments) {
              const parent = acc;
              acc = acc ? `${acc}/${segment}` : segment;
              ensure(acc);
              if (parent) ensure(parent).children.add(acc);
            }
            if (acc) ensure(acc).members.push(...members);
          }
          const subtreeMembers = (path) => {
            const node = nodes.get(path);
            if (!node) return [];
            return [...node.members, ...[...node.children].flatMap(subtreeMembers)];
          };
          const emitFolder = (path, depth) => {
            const node = nodes.get(path);
            const all = subtreeMembers(path);
            const key = `folder:${path}`;
            const name = path.split("/").pop();
            const open = expanded.has(key) || (!!f.search && !path.startsWith(ORPHANED_FOLDER));
            for (const m of all) matchedPks.add(m.pk);
            // The server counts distinct simulations per folder; summing the
            // rows here double-counts everything two datasets share (783
            // memberships over 550 distinct simulations in "Core reference").
            // Role / group-by-keyword filter on the client, so under either the
            // server-wide totals disagree with the folder's own Splits chips.
            const served = (f.role === "all" && !f.group) ? folderTotals[path] : null;
            ordered.push({
              pk: key, _group: true, _children: served ? served.datasets : all.length,
              _loaded: all.length, _open: open, _depth: depth,
              _memberPks: all.map((m) => m.pk),
              _rowClass: "group-row", display_name: name, folder: path,
              simulation_count: served
                ? served.simulations
                : all.reduce((n, m) => n + (m.simulation_count || 0), 0),
              split_counts: all.reduce((acc, m) => {
                for (const [k, v] of Object.entries(m.split_counts || {})) acc[k] = (acc[k] || 0) + v;
                return acc;
              }, {}),
              _onRow: () => {
                if (expanded.has(key)) expanded.delete(key); else expanded.add(key);
                keepCache = true;
                reloadFn();
              },
            });
            if (!open) return;
            for (const child of [...node.children].sort((a, b) => a.localeCompare(b))) {
              emitFolder(child, depth + 1);
            }
            // 640 single-run imports in one flat table is not a list, it is a
            // wall. Show a page and say what is behind it; search covers all.
            const cap = f.search ? node.members.length : FOLDER_PAGE;
            for (const m of node.members.slice(0, cap)) emit(m, depth + 1);
            if (node.members.length > cap) {
              ordered.push({
                pk: `more:${path}`, _rowClass: "subset-row", _depth: depth + 1,
                _hidden: node.members.length - cap,
                display_name: `+${node.members.length - cap} more in ${name} — search to find one`,
                _onRow: () => {},
              });
            }
          };
          const roots = [...nodes.keys()].filter((path) => !path.includes("/"));
          const named = roots.filter((n) => n !== ORPHANED_FOLDER).sort((a, b) => a.localeCompare(b));
          for (const name of named) emitFolder(name, 0);
          for (const r of loose) { matchedPks.add(r.pk); emit(r, 0); }
          if (nodes.has(ORPHANED_FOLDER)) emitFolder(ORPHANED_FOLDER, 0);
        };
        if (f.role !== "all") {
          layout(cache.filter(match), (r, off) => {
            matchedPks.add(r.pk);
            ordered.push(Object.assign({}, r, { _depth: off, _children: 0, _open: false, _rowClass: off ? "subset-row" : null }));
          });
          picker.setMatched(matchedPks);
          return ordered;
        }
        const tops = cache.filter(isTop);
        // A component of two listed parents used to render under both, and a
        // search expands every folder, so 8 of the 34 datasets matching "ti64"
        // came out two or three times — which reads as a corrupt composition.
        // Each dataset renders once: in its own folder if it has one, else
        // under the first parent that lists it.
        const emitted = new Set(tops.map((r) => r.pk));
        const push = (r, depth, off = 0) => {
          const children = depth >= 2 ? [] : kids(r).filter((c) => subtreeMatch(c) && !emitted.has(c.pk));
          children.forEach((c) => emitted.add(c.pk));
          const shown = depth + off;
          const row = Object.assign({}, r, {
            _depth: shown,
            _children: children.length,
            _open: children.length > 0 && (expanded.has(r.pk) || !!f.search),
            _rowClass: shown ? `subset-row${shown > 1 ? " depth-2" : ""}` : null,
          });
          matchedPks.add(r.pk);
          ordered.push(row);
          if (row._open) for (const ch of children) push(ch, depth + 1, off);
        };
        layout(tops.filter((r) => subtreeMatch(r)), (r, off) => push(r, 0, off));
        picker.setMatched(matchedPks);
        return ordered;
      },
      columns: [
        picker.pickColumn((r) => (r._group ? (r._memberPks || []) : (typeof r.pk === "number" ? [r.pk] : []))),
        { label: "Dataset", w: "23%", get: (r) => {
            const toggle = r._children ? el("button", {
              class: "ds-toggle",
              title: r._group ? (r._open ? "Collapse folder" : "Expand folder")
                : r._open ? "Hide linked sets" : `Show ${r._children} linked set${r._children > 1 ? "s" : ""}`,
              onclick: (e) => {
                e.stopPropagation();
                if (expanded.has(r.pk)) expanded.delete(r.pk); else expanded.add(r.pk);
                keepCache = true;
                reloadFn();
              },
            }, r._open ? "\u25be" : "\u25b8") : null;
            const badge = r._group ? el("span", { class: "chip" }, r.display_name === ORPHANED_FOLDER ? "auto" : "folder")
              : (r.dataset_role && r.dataset_role !== "base")
                ? el("span", { class: `chip role-${r.dataset_role}` }, r.dataset_role) : null;
            // Data evicted: the record is intact, only the arrays are gone.
            const replaced = !r._group && r.superseded
              ? el("span", { class: "chip evicted", title: r.replaced_by_pk ? `replaced by #${r.replaced_by_pk}` : "retired" },
                  r.replaced_by_pk ? `replaced by #${r.replaced_by_pk}` : "replaced")
              : null;
            const evicted = !r._group && r.evicted_count
              ? el("span", { class: "chip evicted", title: `${r.evicted_count} of ${r.simulation_count} simulations are metadata-only` },
                  r.evicted_count === r.simulation_count ? "metadata only" : "partly evicted")
              : null;
            const sub = r._group
              ? [`${r._children} dataset${r._children > 1 ? "s" : ""}`]
              : [`#${r.pk}`, r.source_folder_name || r.source_type || ""];
            if (!r._group && r._children && !r._open) sub.push(`${r._children} linked`);
            return el("div", { class: "ds-name" },
              el("strong", {}, toggle, (r._depth ? "\u21b3 " : "") + (r.display_name || r.label || `Dataset ${r.pk}`), badge, replaced, evicted),
              el("span", { class: "ds-sub" }, sub.filter(Boolean).join(" \u00b7 ")));
          } },
        { label: "Sims", key: "simulation_count", num: true, w: "6%" },
        { label: "Splits", w: "20%", get: (r) => splitChips(r.split_counts) },
        // Uniform member steps → the count; varying steps → the (near-uniform)
        // step size Δt, which is the better indicator there; else a range.
        { label: "Steps", num: true, w: "9%", get: (r) => {
            if (r.steps_min != null && r.steps_min === r.steps_max) return String(r.steps_min);
            if (r.step_size_s != null) return el("span",
              { title: `steps vary (${r.steps_min}–${r.steps_max}); near-uniform step size` },
              `Δt ${fmtStepSize(r.step_size_s)}`);
            if (r.steps_min != null) return `${r.steps_min}–${r.steps_max}`;
            return r.num_timesteps != null ? String(r.num_timesteps) : "—";
          } },
        { label: "Description", w: "23%", get: (r) => (r._group ? el("span") : descriptionCell(r)) },
        { label: "Keywords", w: "16%", get: (r) => (r._group ? el("span") : keywordCell(r)) },
      ],
      renderDetail: renderDatasetDetail,
    });
  }

  async function renderDatasetDetail(pk) {
    const d = await getJson(`/datasets/${pk}`);
    const name = d.display_name || d.label || `Dataset ${pk}`;
    const casesBox = el("div", {}, el("div", { class: "detail-empty" }, "Loading cases\u2026"));
    const caseDetail = el("div", {});
    loadCases();

    async function loadCases() {
      try {
        const body = await getJson(`/datasets/${pk}/cases`);
        renderCaseTable(body.cases || []);
      } catch (e) {
        clear(casesBox).append(e.status === 404
          ? el("div", { class: "muted small" }, "No per-case browser for this dataset type.")
          : notice("err", e.message));
      }
    }
    function renderCaseTable(cases, shown = 100) {
      const slice = cases.slice(0, shown);
      const parts = [
        table(slice, [
          { label: "Case", key: "case_name", nowrap: true },
          { label: "Split", get: (c) => el("span", { class: `chip split-${c.split || "unassigned"}` }, c.split || "unassigned") },
          { label: "P", key: "laser_power", num: true },
          { label: "V", key: "V_scan", num: true },
          { label: "Steps", key: "num_timesteps", num: true },
          { label: "Grid", key: "grid", nowrap: true },
        ], (c, tr) => { markSel(casesBox, tr); showCase(c); }),
      ];
      if (cases.length > shown) {
        parts.push(el("div", { class: "row", style: "margin-top:8px" },
          el("button", { class: "button small", onclick: () => renderCaseTable(cases, shown + 200) },
            `Show more (${cases.length - shown} hidden)`)));
      }
      clear(casesBox).append(...parts);
    }
    function showCase(c) {
      clear(caseDetail).append(el("div", { class: "case-detail stack" },
        el("div", { class: "row" }, el("h3", {}, c.case_name || `Case ${c.pk}`),
          el("span", { class: "muted small" }, `simulation #${c.pk}`), el("span", { class: "spacer" }),
          splitCell(c)),
        kv({ laser_power: c.laser_power, V_scan: c.V_scan, beam_radius: c.beam_radius,
          num_timesteps: c.num_timesteps, grid: c.grid, usable_for_training: c.usable_for_training }),
        el("div", { class: "row" }, el("span", { class: "muted small" }, "Keywords:"), keywordCell(c)),
        caseStats(c),
        caseViewer(c)));
      caseDetail.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    return el("div", { class: "stack" },
      el("div", { class: "row" }, el("h2", {}, name),
        el("span", { class: "muted small" }, `#${pk}`), el("span", { class: "spacer" }),
        evictButton(pk), deleteButton([pk], `dataset ${pk}`)),
      el("div", { class: "row" }, el("span", { class: "muted small" }, "Description:"),
        descriptionCell({ pk, description: d.description })),
      el("div", { class: "row" }, el("span", { class: "muted small" }, "Keywords:"),
        keywordCell({ pk, keywords: d.keywords || [] })),
      el("div", { class: "row" }, el("span", { class: "muted small" }, "Folder:"),
        folderCell({ pk, folder: d.folder }),
        d.folder_inherited_from
          ? el("span", { class: "muted small", title: "it follows its base; filing it here explicitly overrides that" },
              `inherited from #${d.folder_inherited_from.pk}`)
          : el("span")),
      kv({
        split_counts: splitChips(d.split_counts),
        dataset_role: d.dataset_role, source_type: d.source_type,
        source_folder_name: d.source_folder_name, canonical_layout: d.canonical_layout,
        num_timesteps: d.num_timesteps, simulation_count: d.simulation_count,
        derived_from: d.derived_from ? datasetLink(d.derived_from) : undefined,
        derived_sets: (d.derived_sets || []).length
          ? el("span", {}, ...d.derived_sets.map((s) => datasetLink(s))) : undefined,
        composed_from: (d.composed_from || []).length
          ? el("span", {}, ...d.composed_from.map((s) => datasetLink(s))) : undefined,
        family_identifier: el("span", { class: "mono" }, fmt(d.family_identifier)),
      }),
      assignSplitsForm(pk),
      d.requested_fields ? el("details", {},
        el("summary", { class: "muted small" }, `Fields (${(d.requested_fields || []).length})`),
        el("div", { class: "mono", style: "padding-top:6px" }, fmt(d.requested_fields))) : null,
      el("div", {}, el("h3", {}, "Cases"), casesBox),
      caseDetail,
      datasetStats(pk),
      el("details", {}, el("summary", { class: "muted small" }, "Full metadata"), jsonBlock(d.metadata || d)));
  }

  // Random train/val/test assignment over a collection's members.
  function assignSplitsForm(pk) {
    const num = (v) => el("input", { type: "number", value: v, step: "0.05", min: "0", max: "1", style: "width:72px" });
    const tr = num("0.8"), va = num("0.1"), te = num("0.1");
    const onlyUn = el("input", { type: "checkbox", checked: "checked" });
    const out = el("div", {});
    const btn = el("button", { class: "button small", onclick: run }, "Assign");
    async function run() {
      btn.disabled = true;
      clear(out).append(el("span", { class: "muted small" }, "Assigning…"));
      try {
        const r = await postJson(`/datasets/${pk}/assign_splits`, {
          train: parseFloat(tr.value) || 0, val: parseFloat(va.value) || 0, test: parseFloat(te.value) || 0,
          only_unassigned: onlyUn.checked,
        });
        clear(out).append(notice("ok",
          `assigned ${r.assigned.train} train / ${r.assigned.val} val / ${r.assigned.test} test`
          + (r.skipped ? `; ${r.skipped} already assigned, kept` : "")
          + (r.recipe ? ` — recipe recorded (seed ${r.recipe.seed})` : "")
          + " — reselect the dataset to refresh the chips"));
      } catch (e) { clear(out).append(notice("err", e.message)); }
      btn.disabled = false;
    }
    return el("details", {},
      el("summary", { class: "muted small" }, "Assign train/val/test splits…"),
      el("div", { class: "row", style: "margin-top:6px;flex-wrap:wrap;gap:8px" },
        field("train", tr), field("val", va), field("test", te),
        el("label", { class: "checkline" }, onlyUn, "only unassigned"), btn),
      out);
  }

  // Chip linking to another dataset's detail, swapped into the same pane.
  function datasetLink(ref) {
    return el("button", {
      class: "chip ds-link",
      onclick: async (e) => {
        const pane = e.target.closest(".detail-pane");
        if (!pane) return;
        clear(pane).append(el("div", { class: "detail-empty" }, "Loading…"));
        try { const node = await renderDatasetDetail(ref.pk); clear(pane).append(node); }
        catch (err) { clear(pane).append(notice("err", err.message)); }
      },
    }, `${ref.display_name || `Dataset ${ref.pk}`} (#${ref.pk})`);
  }

  // Per-case melt-pool statistics (computed server-side, cached in extras).
  // Reading the cache is instant; computing reads every frame of the case and
  // took 197 s on a 331-frame one, so clicking a row asks the cache only and
  // the work is one deliberate click away.
  function caseStats(c) {
    const box = el("div", {}, el("span", { class: "muted small" }, "Loading melt-pool stats\u2026"));
    const render = (b) => {
      if (b.computed === false) {
        const btn = el("button", { class: "button small", onclick: () => {
          btn.disabled = true;
          clear(box).append(el("span", { class: "muted small" },
            "Computing melt-pool stats\u2026 this reads every frame of the case and can take minutes."));
          load(true);
        } }, "Compute now");
        clear(box).append(el("div", { class: "row" },
          el("span", { class: "muted small" }, "Melt-pool stats not computed for this case."), btn));
        return;
      }
      if (!b.stats) {
        clear(box).append(el("span", { class: "muted small" },
          `No melt-pool stats for this case (${b.unavailable || "fields unavailable"}).`));
        return;
      }
      const st = b.stats;
      clear(box).append(el("div", { class: "stack" },
        el("div", { class: "row" }, el("h3", {}, "Melt-pool stats"), regimeBadge(st.regime)),
        kv({ melt_vol_um3: st.melt_vol_um3, melt_depth_um: st.melt_depth_um,
          depth_um: st.depth_um, cavity_rmin_um: st.cavity_rmin_um, cavity_rmax_um: st.cavity_rmax_um,
          xlen_um: st.xlen_um, ylen_um: st.ylen_um, tmax_k: st.tmax_k, H: st.H })));
    };
    const load = (compute) => getJson(`/simulations/${c.pk}/stats${compute ? "?compute=1" : ""}`)
      .then(render)
      .catch((e) => clear(box).append(notice("err", e.message)));
    load(false);
    return box;
  }

  // Dataset-level melt-pool statistics: summary, batched compute, notebook plots.
  function datasetStats(pk) {
    const box = el("div", { class: "stack" }, el("div", { class: "muted small" }, "Loading stats\u2026"));
    const kinds = [["regime", "Regime mix"], ["depth_vs_width", "Keyhole line"], ["melt_hist", "Melt volume"], ["pv_map", "P\u2013V map"]];
    let plotKind = "regime";

    async function refresh() {
      try {
        const b = await getJson(`/datasets/${pk}/stats`);
        const sum = b.summary || {};
        // Subtract the tombstoned cases: their fields cannot yield melt metrics
        // at all, so offering them as "missing" never reaches zero and each
        // click re-runs a compute that is guaranteed to fail.
        const unavailable = sum.unavailable || 0;
        const missing = (sum.total || 0) - (sum.computed || 0) - unavailable;
        const plotUrl = () => url(`/datasets/${pk}/stats/plot?kind=${plotKind}&_=${Date.now()}`);
        // Switching the plot only swaps the image; a refresh() re-fetched the summary too.
        const kindSel = selectVals2(kinds, (v) => { plotKind = v; img.src = plotUrl(); });
        kindSel.value = plotKind;
        const img = sum.computed
          ? el("img", { class: "viz-frame", src: plotUrl(), alt: "dataset stats plot" })
          : el("div", { class: "muted small" }, "No cases computed yet.");
        const computeBtn = el("button", { class: "button small" }, missing ? `Compute missing (${missing})` : "All computed");
        const progress = el("span", { class: "muted small" });
        computeBtn.disabled = !missing;
        computeBtn.addEventListener("click", async () => {
          computeBtn.disabled = true;
          let remaining = missing;
          const failed = [];
          try {
            while (remaining > 0) {
              const r = await postJson(`/datasets/${pk}/stats/compute?limit=25`, {});
              remaining = r.remaining;
              failed.push(...(r.failed || []));
              progress.textContent = `computing\u2026 ${r.total - remaining}/${r.total}`;
              if (!r.computed_now && !(r.failed || []).length) break;  // no progress: stop looping
            }
          } catch (e) { progress.textContent = e.message; }
          if (failed.length) progress.textContent = `${failed.length} case(s) have no stats: ${failed.slice(0, 3).join("; ")}`;
          refresh();
        });
        const regimes = sum.regime_counts
          ? el("span", { class: "chips" }, ...Object.entries(sum.regime_counts).map(([k2, n]) =>
              el("span", { class: "chip", style: `background:${REGIME_COLORS[k2] || "#888"}22; color:${REGIME_COLORS[k2] || "#555"}` }, `${n} ${k2}`)))
          : null;
        // .filter(Boolean): DOM append() renders a null child as the text "null".
        clear(box).append(...[
          el("div", { class: "row" }, el("h3", {}, "Melt-pool statistics"),
            el("span", { class: "muted small" }, `${sum.computed || 0}/${sum.total || 0} cases`
              + (unavailable ? ` · ${unavailable} without the fields melt metrics need` : "")),
            el("span", { class: "spacer" }), computeBtn, progress),
          regimes,
          sum.computed ? kv({ keyhole_fraction: sum.keyhole_fraction,
            melt_vol_um3_mean: sum.melt_vol_um3_mean, melt_vol_um3_max: sum.melt_vol_um3_max,
            depth_um_mean: sum.depth_um_mean, depth_um_max: sum.depth_um_max,
            tmax_k_mean: sum.tmax_k_mean, tmax_k_max: sum.tmax_k_max }) : null,
          sum.computed ? el("div", { class: "row" }, field("Plot", kindSel)) : null,
          img].filter(Boolean));
      } catch (e) { clear(box).append(notice("err", e.message)); }
    }
    refresh();
    return box;
  }

  // Per-case animated slice viewer: surface / longitudinal / transverse PNG
  // frames served by /simulations/{pk}/plot, stepped by a slider or play loop.
  function caseViewer(c) {
    const box = el("div", { class: "stack case-viewer" }, el("div", { class: "detail-empty" }, "Loading viewer\u2026"));
    const simPk = c.pk;
    getJson(`/simulations/${simPk}/frames`).then((info) => {
      const fields = info.fields || [];
      if (!fields.length || !info.num_frames) {
        clear(box).append(el("div", { class: "muted small" }, "No plottable arrays for this case."));
        return;
      }
      const st = {
        field: fields.includes("Temperature") ? "Temperature" : fields[0],
        view: "surface", frame: Math.max(0, info.num_frames - 1), playing: null, contour: true,
      };
      const img = el("img", { class: "viz-frame", alt: "case plot" });
      const slider = el("input", { type: "range", min: "0", max: String(info.num_frames - 1), value: String(st.frame) });
      const timeLab = el("span", { class: "muted small nowrap" });
      const playBtn = el("button", { class: "button small" }, "\u25b6");
      const fieldSel = el("select", {}, ...fields.map((f) => el("option", { value: f, selected: f === st.field ? "selected" : null }, f)));
      // Surface first: for random paths a fixed cross-section plane is
      // ill-defined, the top view is always meaningful.
      const viewSel = selectVals2([["surface", "Surface"], ["long", "Longitudinal"], ["cross", "Transverse"]], () => {});
      const contourCb = el("input", { type: "checkbox", checked: "checked" });

      function urlFor(frame) {
        return url(`/simulations/${simPk}/plot?field=${encodeURIComponent(st.field)}&view=${st.view}&frame=${frame}&contour=${st.contour}`);
      }
      function show(frame) {
        st.frame = frame;
        slider.value = String(frame);
        img.src = urlFor(frame);
        const t = (info.times_s || [])[frame];
        timeLab.textContent = t != null ? `t = ${(t * 1e3).toFixed(3)} ms (${frame + 1}/${info.num_frames})` : `frame ${frame + 1}/${info.num_frames}`;
        if (frame + 1 < info.num_frames) new Image().src = urlFor(frame + 1); // prefetch
      }
      function stop() { if (st.playing) { clearInterval(st.playing); st.playing = null; playBtn.textContent = "\u25b6"; } }
      playBtn.addEventListener("click", () => {
        if (st.playing) { stop(); return; }
        playBtn.textContent = "\u23f8";
        // Frames render server-side slower than the tick; advancing regardless ran the
        // label six frames ahead of the image.
        st.playing = setInterval(() => { if (img.complete) show((st.frame + 1) % info.num_frames); }, 450);
      });
      slider.addEventListener("input", () => { stop(); show(parseInt(slider.value, 10)); });
      fieldSel.addEventListener("change", () => { st.field = fieldSel.value; show(st.frame); });
      viewSel.addEventListener("change", () => { st.view = viewSel.value; show(st.frame); });
      contourCb.addEventListener("change", () => { st.contour = contourCb.checked; show(st.frame); });

      clear(box).append(
        el("div", { class: "row" },
          field("Field", fieldSel), field("View", viewSel),
          el("label", { class: "checkline", style: "align-self:end" }, contourCb, "melt contour")),
        el("div", { class: "row viz-controls" }, playBtn, slider, timeLab),
        img);
      show(st.frame);
    }).catch((e) => clear(box).append(notice("err", e.message)));
    return box;
  }

  // Models: full registry with instant client-side filtering (architecture,
  // trained-on dataset, metric-on-eval-set threshold). Rows are fetched once;
  // filters re-render locally, so filtering is immediate.
  function severityColor(sev) {
    return { confound: "#c0392b", unknown: "#b8860b", caution: "#b8860b", benign: "#6b7280" }[sev] || "#6b7280";
  }

  function renderCompareReport(r) {
    const box = el("div", { class: "stack" });
    box.append(el("div", { class: "row" },
      el("strong", {}, `Compare: ${r.a.name} (#${r.a.pk}) vs ${r.b.name} (#${r.b.pk})`)));
    const clean = r.verdict.startsWith("CLEAN") || r.verdict.startsWith("SINGLE-FACTOR:");
    // Only the head ("CONFOUNDED:") is bold: a 79-factor verdict in bold read as a wall.
    const colon = r.verdict.indexOf(": ");
    box.append(el("div", {
      style: `border:2px solid ${clean ? "#1e8e3e" : "#c0392b"};`
        + `background:${clean ? "#1e8e3e22" : "#c0392b22"};`
        + "padding:6px 10px;border-radius:4px",
    }, colon > 0
      ? [el("strong", {}, r.verdict.slice(0, colon + 1)), r.verdict.slice(colon + 1)]
      : el("strong", {}, r.verdict)));
    box.append(el("div", { class: "muted small" }, `Training data: ${r.training_data}`));
    if ((r.backfilled_facts || []).length) {
      box.append(el("div", { class: "muted small" },
        `⚠ reconstructed (backfilled) facts in play: ${r.backfilled_facts.join(", ")}`));
    }
    const substantive = (r.differences || []).filter((d) => d.severity !== "benign");
    if (substantive.length) {
      const table = el("table", {}, el("thead", {}, el("tr", {},
        el("th", {}, "severity"), el("th", {}, "factor"), el("th", {}, "field"),
        el("th", {}, "A"), el("th", {}, "B"))));
      const body = el("tbody", {});
      for (const d of substantive) {
        body.append(el("tr", {},
          el("td", { style: `color:${severityColor(d.severity)};font-weight:600` }, d.severity),
          el("td", { class: "nowrap" }, d.factor || ""), el("td", { class: "nowrap" }, d.field),
          el("td", { class: "small" }, String(d.a)), el("td", { class: "small" }, String(d.b))));
      }
      table.append(body);
      box.append(el("div", { style: "overflow-x:auto" }, table));
    } else {
      box.append(el("div", { class: "muted small" }, "No substantive differences."));
    }
    const benign = (r.differences || []).filter((d) => d.severity === "benign");
    if (benign.length) {
      box.append(el("details", {}, el("summary", { class: "muted small" },
        `${benign.length} bookkeeping difference(s) (names, timestamps, GPU booking)`),
        ...benign.map((d) => el("div", { class: "muted small" },
          `${d.field}: ${String(d.a)} vs ${String(d.b)}`))));
    }
    const evals = r.shared_evaluations || [];
    box.append(el("h3", {}, `Shared evaluations (${evals.length})`));
    if (!evals.length) {
      box.append(el("div", { class: "muted small" },
        "No dataset was scored by both models — there is nothing legitimate to compare yet."));
    }
    for (const ev of evals) {
      const line = el("div", { class: ev.comparable ? "" : "muted" },
        el("strong", {}, ev.comparable ? "✓ " : "✗ "),
        `${ev.dataset_label || ev.dataset_uuid}: T abs err ${ev.a.t_abs_err_mean} vs ${ev.b.t_abs_err_mean}`,
        ev.problems.length ? el("span", { style: "color:#c0392b" }, `  [${ev.problems.join("; ")}]`) : "");
      box.append(line);
    }
    return box;
  }

  function renderModels(view) {
    const sel = { pk: null };
    let all = [], modelTotal = 0, modelTruncated = false;
    const f = { search: "", arch: "all", trained: "all", evalSet: "all", metric: "", maxVal: "", starred: false };
    const expanded = new Set();      // open folder headers
    const picker = folderTools({
      rows: () => all,
      autoFolder: () => null,     // models are filed explicitly, never automatically
      reload: () => redraw(),
    });
    // A page takes 25-156 s on a loaded backend; a bare "Loading\u2026" is indistinguishable
    // from a hung one, so the note counts the seconds while it is on screen.
    function loadingNote() {
      const n = el("div", { class: "detail-empty" }, "Loading\u2026");
      const t0 = Date.now();
      const timer = setInterval(() => {
        if (!n.isConnected) { clearInterval(timer); return; }
        n.textContent = `Loading\u2026 ${Math.round((Date.now() - t0) / 1000)} s`;
      }, 1000);
      return n;
    }
    // "best T MAE (set A)" and "best T MAE (set B)" both read "best T MAE" once the
    // dataset is moved into the tooltip; four identical chips on one row looked like a
    // rendering glitch. One chip per text, a count when it repeats, every set in the title.
    function autoChips(badges) {
      const byText = new Map();
      for (const b of badges || []) {
        const text = b.split(" (")[0];
        if (!byText.has(text)) byText.set(text, []);
        byText.get(text).push(b);
      }
      return [...byText].map(([text, full]) => el("span", { class: "chip auto", title: full.join("\n") },
        full.length > 1 ? `${text} \u00d7${full.length}` : text));
    }
    const listBox = el("div", {}, loadingNote());
    const detail = el("div", { class: "panel detail-pane" }, el("div", { class: "detail-empty" }, "Select a model."));
    const controls = el("div", { class: "controls-grid wide" });
    view.append(el("div", { class: "split" },
      el("div", { class: "panel stack" },
        el("div", { class: "row" }, el("h2", {}, "Models"), el("span", { class: "spacer" }),
          el("button", { class: "button small", onclick: load }, "Refresh")),
        controls, listBox),
      detail));

    // The column is "best", so it must be the minimum, and it must be the same
    // key on every row or the column cannot be read down its own length. It
    // used to return the FIRST evaluation that had any key, which was not the
    // minimum on 15 of 30 multi-eval models, and let the key vary per row.
    const METRIC_KEY_ORDER = ["T.rmse", "T.rel_rmse", "T.abs_err_mean"];
    function bestMetricKey(rows) {
      for (const key of METRIC_KEY_ORDER) {
        if (rows.some((r) => (r.evals || []).some((ev) => (ev.metrics || {})[key] != null))) return key;
      }
      return null;
    }
    function primaryMetric(row, key) {
      if (!key) return null;
      let best = null;
      for (const ev of row.evals || []) {
        const value = (ev.metrics || {})[key];
        if (value == null) continue;
        // Kelvin when the evaluation recorded its own scale, so the list agrees
        // with the detail pane instead of showing 0.0270 against its 80.9 K.
        const kelvin = (ev.kelvin || {})[key];
        if (best === null || value < best.value) {
          best = { key, value, kelvin, set: ev.dataset_display || ev.dataset_label, n: 0 };
        }
      }
      if (best) best.n = (row.evals || []).filter((ev) => (ev.metrics || {})[key] != null).length;
      return best;
    }
    function metricCell(row, key) {
      const pm = primaryMetric(row, key);
      if (!pm) return el("span", { class: "muted small" }, "no metrics");
      const shown = pm.kelvin != null
        ? `${Number(pm.kelvin).toPrecision(4)} K`
        : `${Number(pm.value).toPrecision(3)} (normalised)`;
      return el("span", { class: "small", title: `${pm.key}; best of ${pm.n} evaluation(s)` },
        shown,
        el("span", { class: "muted small" }, ` (${pm.set || "?"})`));
    }
    let metricKey = null;
    function matches(row) {
      if (f.starred && !isStarred(row)) return false;
      // No client-side search clause: a search always refetches the population with
      // `?search=` (populationKey), and the server also matches the training dataset's
      // label, which a client re-filter on model_name/label/config_name would hide.
      if (f.arch !== "all" && row.architecture !== f.arch) return false;
      if (f.trained !== "all" && (row.training_source_display || "?") !== f.trained) return false;
      if (f.evalSet !== "all" || f.metric || f.maxVal !== "") {
        const evs = (row.evals || []).filter((ev) => f.evalSet === "all" || (ev.dataset_display || ev.dataset_label) === f.evalSet);
        if (!evs.length) return false;
        if (f.metric) {
          const vals = evs.map((ev) => (ev.metrics || {})[f.metric]).filter((v) => v != null);
          if (!vals.length) return false;
          if (f.maxVal !== "" && !vals.some((v) => v <= Number(f.maxVal))) return false;
        }
      }
      return true;
    }
    function rebuildControls() {
      // Rebuilding replaces the search box; if it held the caret, the new one takes it.
      const refocus = document.activeElement && document.activeElement.type === "search" && controls.contains(document.activeElement);
      clear(controls);
      const archs = [...new Set(all.map((r) => r.architecture).filter(Boolean))].sort();
      const trained = [...new Set(all.map((r) => r.training_source_display || "?"))].sort();
      const evalSets = [...new Set(all.flatMap((r) => (r.evals || []).map((e2) => e2.dataset_display || e2.dataset_label)).filter(Boolean))].sort();
      const metrics = [...new Set(all.flatMap((r) => (r.evals || []).flatMap((e2) => Object.keys(e2.metrics || {}))))].sort();
      // A filter change has to re-FETCH whenever the loaded population is not the one the
      // filter needs: a page cannot answer for the profile, a search fetched for "rigno"
      // cannot answer "fno", and clearing every filter has to go back to paging. The
      // population is keyed by what it was fetched with; a matching key redraws locally.
      const onFilter = () => { if (populationKey() !== loadedKey) loadPage(true); else redraw(); };
      let searchTimer = null;
      const searchIn = el("input", { type: "search", value: f.search, oninput: (e) => {
        f.search = e.target.value;
        // Every keystroke is a different server search; wait for the typing to pause.
        clearTimeout(searchTimer);
        searchTimer = setTimeout(onFilter, 300);
      } });
      const archSel = el("select", { onchange: (e) => { f.arch = e.target.value; onFilter(); } },
        el("option", { value: "all" }, "all"), ...archs.map((a) => el("option", { value: a, selected: f.arch === a ? "selected" : null }, a)));
      const trainedSel = el("select", { onchange: (e) => { f.trained = e.target.value; onFilter(); } },
        el("option", { value: "all" }, "all"), ...trained.map((t) => el("option", { value: t, selected: f.trained === t ? "selected" : null }, t)));
      const evalSel = el("select", { onchange: (e) => { f.evalSet = e.target.value; onFilter(); } },
        el("option", { value: "all" }, "any / none"), ...evalSets.map((t) => el("option", { value: t, selected: f.evalSet === t ? "selected" : null }, t)));
      const metricSel = el("select", { onchange: (e) => { f.metric = e.target.value; onFilter(); } },
        el("option", { value: "" }, "\u2014"), ...metrics.map((m) => el("option", { value: m, selected: f.metric === m ? "selected" : null }, m)));
      const maxIn = el("input", { type: "number", step: "any", placeholder: "\u2264 max", value: f.maxVal, oninput: (e) => { f.maxVal = e.target.value; onFilter(); } });
      const starChk = el("input", { type: "checkbox", onchange: (e) => { f.starred = e.target.checked; onFilter(); } });
      if (f.starred) starChk.checked = true;
      const compareBtn = el("button", { class: "button small", title:
        "select exactly two models (checkboxes), then compare their provenance, "
        + "hyperparameters, data and evaluations — the ablation referee",
        onclick: async () => {
          const pks = [...picker.picked];
          // The report replaces whatever detail was open, so its row must stop reading as selected.
          sel.pk = null;
          [...listBox.querySelectorAll("tr.is-selected")].forEach((n) => n.classList.remove("is-selected"));
          if (pks.length !== 2) {
            // an inline message, where the report will appear -- not an alert()
            clear(detail);
            detail.append(el("div", { class: "detail-empty" },
              `Compare needs exactly two models ticked (${pks.length} selected). `
              + "Tick two checkboxes in the list, or use \u201cCompare with\u2026\u201d on a model's detail page."));
            return;
          }
          clear(detail);
          detail.append(el("div", { class: "detail-empty" }, "Comparing…"));
          try {
            const report = await getJson(`/models/${pks[0]}/compare/${pks[1]}`);
            clear(detail);
            detail.append(renderCompareReport(report));
          } catch (err) {
            clear(detail);
            detail.append(el("div", { class: "detail-empty" }, `compare failed: ${err.message}`));
          }
        } }, "Compare 2…");
      controls.append(
        field("Search", searchIn), field("Architecture", archSel), field("Trained on", trainedSel),
        field("Evaluated on", evalSel), field("Metric", metricSel), field("Threshold", maxIn),
        field("Starred only", starChk), compareBtn, picker.buildBar());
      if (refocus) { searchIn.focus(); searchIn.setSelectionRange(f.search.length, f.search.length); }
    }
    function redraw() {
      // Starred first: they are the models a later reader should look at.
      const matched = all.filter(matches).sort((a, b) => (isStarred(b) ? 1 : 0) - (isStarred(a) ? 1 : 0));
      picker.setMatched(new Set(matched.map((r) => r.pk)));
      // Same folder layout as the Datasets tab: a collapsible header per
      // folder, unfiled models at the top level, folders sorted by name.
      const groups = new Map();
      const loose = [];
      for (const r of matched) {
        if (!r.folder) { loose.push(r); continue; }
        if (!groups.has(r.folder)) groups.set(r.folder, []);
        groups.get(r.folder).push(r);
      }
      const rows = [];
      for (const name of [...groups.keys()].sort((a, b) => a.localeCompare(b))) {
        const members = groups.get(name);
        const key = `folder:${name}`;
        const open = expanded.has(key) || !!f.search;
        rows.push({
          pk: key, _group: true, _children: members.length, _open: open, _rowClass: "group-row",
          _memberPks: members.map((m) => m.pk), model_name: name, folder: name,
          _onRow: () => {
            if (expanded.has(key)) expanded.delete(key); else expanded.add(key);
            redraw();
          },
        });
        if (open) for (const m of members) rows.push(Object.assign({}, m, { _rowClass: "subset-row" }));
      }
      rows.push(...loose);
      metricKey = bestMetricKey(all);   // one key for the whole column
      clear(listBox).append(
        // A capped list must say it is capped: the server filters the whole
        // population and reports the total, so the page can name what it left.
        el("div", { class: "muted small", style: "margin-bottom:6px" },
          matched.length === all.length
            ? `${all.length}${modelTruncated ? ` of ${modelTotal}` : ""} models loaded`
            : `${matched.length} shown of ${all.length}${modelTruncated ? ` loaded · ${modelTotal} on the profile` : ""}`),
        table(rows, [
          picker.pickColumn((r) => (r._group ? (r._memberPks || []) : (typeof r.pk === "number" ? [r.pk] : []))),
          { label: "\u2605", w: "3%", get: (r) => (r._group ? el("span") : starCell(r, redraw)) },
          // Widths sum to 100 and every header fits its column at 1280 with the detail
          // pane open (~720 px of table): the old 6-7 % columns clipped STEPS, RESOLUTION
          // and DESC., and the nowrap Arch. cell ran into the next column.
          { label: "Model", w: "24%", get: (r) => (r._group
              ? el("div", { class: "ds-name" },
                  el("strong", {}, el("button", { class: "ds-toggle" }, r._open ? "\u25be" : "\u25b8"),
                    r.model_name, el("span", { class: "chip" }, r.folder === AUTO_AL_FOLDER ? "auto" : "folder")),
                  el("span", { class: "ds-sub" }, `${r._children} model${r._children > 1 ? "s" : ""}`))
              : el("div", { class: "ds-name" },
                  el("strong", {}, r.model_name || r.label || `Model ${r.pk}`, ...autoChips(r.auto_keywords)),
                  el("span", { class: "ds-sub" }, `#${r.pk} \u00b7 ${r.config_name || ""}`))) },
          { label: "Arch.", w: "9%", get: (r) => (r._group ? el("span") : (r.architecture || "—")) },
          { label: "Trained on", w: "12%", clamp: true, get: (r) => {
              if (r._group) return el("span");
              const canonical = r.training_source_canonical;
              const name = r.training_source_display || "—";
              // 44 of 156 models trained on a dataset that has since been
              // merged. Two models on the same data under two node ids read as
              // two different datasets unless the survivor is named.
              return canonical
                ? el("span", { title: `merged into #${canonical.pk}` }, name,
                    el("span", { class: "chip", title: `now ${canonical.display || `#${canonical.pk}`}` }, `now ${canonical.display || `#${canonical.pk}`}`))
                : name;
            } },
          // Steps TRAINED, not the budget requested. 31 of 158 runs stopped
          // short of their max_steps (worst 4000 of 48000) and this column
          // printed the budget, so two models that trained 11k and 18k steps
          // both read "18000".
          { label: "Steps", w: "8%", get: (r) => {
              if (r._group) return el("span");
              const trained = r.steps_trained, budget = r.max_steps;
              if (trained == null) return budget != null
                ? el("span", { class: "muted", title: `step budget requested; steps actually trained not recorded` }, `≤${budget}`)
                : el("span", {}, "—");
              if (budget != null && trained < budget) {
                return el("span", { title: `stopped early: ${trained} of a requested ${budget} steps` },
                  String(trained), el("span", { class: "muted small", style: "display:block" }, `/${budget}`));
              }
              return el("span", {}, String(trained));
            } },
          // Resolution belongs beside the score. `env.PINO_LAZY_STRIDE`
          // downsamples the grid inside the loader and reached no record, so 25
          // of the 35 models with a known stride trained on a coarser grid than
          // their page implied — including the best-T-MAE holder
          // (13.138 K), whose derivation profile says [1,1,1] while it trained
          // 2x2x1. An error on a coarser grid is not comparable with one on the
          // full grid. ("Resolution", not "Grid": the case browser already has a
          // Grid column meaning the spatial shape.)
          { label: el("span", { title: "training resolution: the full grid, or the loader stride it was downsampled by" }, "Stride"),
            w: "9%", nowrap: true, get: (r) => {
              if (r._group || !r.training_stride) return el("span", { class: "muted small" }, "—");
              const stride = r.training_stride.join("×");
              return r.full_resolution
                ? el("span", { class: "muted small", title: "trained on the full grid" }, "full")
                : el("span", { class: "chip evicted",
                      title: `trained on a strided grid (${stride}) — scores are not comparable with full-resolution models` },
                    stride);
            } },
          { label: metricKey ? `Best ${metricKey}` : "Best metric", w: "14%",
            get: (r) => (r._group ? el("span") : metricCell(r, metricKey)) },
          // Narrow, not 16%: it renders an em dash on almost every model, and the width it held was
          // what squeezed the name column into breaking identifiers across three lines.
          { label: "Desc", w: "7%", get: (r) => (r._group ? el("span") : descriptionCell(r)) },
          { label: "Keywords", w: "11%", get: (r) => (r._group ? el("span") : keywordCell(r)) },
        ], async (r, tr) => {
          if (r._onRow) { r._onRow(); return; }   // folder headers toggle, they have no detail
          sel.pk = r.pk;
          markSel(listBox, tr);
          clear(detail).append(el("div", { class: "detail-empty" }, "Loading\u2026"));
          try { clear(detail).append(await modelDetail(r.pk)); }
          catch (e) { clear(detail).append(notice("err", e.message)); }
        }, sel.pk));
      // After the table, so scrolling to the bottom of the list is what asks for more.
      attachPager(listBox);
    }
    // Load a PAGE at a time and fetch the next as the list is scrolled. Asking for all
    // 447 models took 22.7 s before the row build was optimised and 13.8 s after, and the
    // tab read "Loading..." for every second of it. A 60-row page answers in ~1.9 s.
    const PAGE = 60;
    let pageLoading = false;
    // FILTERING MUST SEE EVERY MODEL. `matches()` runs client-side, so a search against
    // a paged list would answer "no results" for a model that exists but has not been
    // scrolled to -- silently wrong, and the render check caught it by looking for a
    // model older than the first page. So any active filter loads the whole population
    // (`limit=0`: the 500 cap left the 161 oldest of 661 models invisible to every
    // filter, and said nothing) in one request and turns paging off; only unfiltered
    // BROWSING pages.
    //
    // A search goes to the SERVER, which filters before building rows -- that is both
    // correct (it sees every model, not the loaded page) and much cheaper than pulling
    // all 448 rows back to filter here, which took ~14 s. Other filters have no server
    // equivalent, so they still pull the population and filter client-side.
    function populationKey() {
      if (f.search) return `/models?limit=0&search=${encodeURIComponent(f.search)}`;
      const filtering = f.starred || f.arch !== "all" || f.trained !== "all"
        || f.evalSet !== "all" || f.metric || f.maxVal !== "";
      return filtering ? "/models?limit=0" : "page";
    }
    let loadedKey = null;     // the population `all` was fetched with
    let generation = 0;       // bumped by every reset, so a superseded fetch cannot append
    async function loadPage(reset) {
      // A reset must not wait for a page in flight: the keyed getJson aborts that page,
      // and the generation guard below drops its rows if it somehow lands. Typing a
      // search while a 60 s page was pending used to be silently ignored.
      if (pageLoading && !reset) return;
      pageLoading = true;
      const key = populationKey();
      const mine = reset ? ++generation : generation;
      if (reset) {
        all = [];
        clear(listBox).append(loadingNote());
      }
      try {
        const paged = key === "page";
        const body = await getJson(paged ? `/models?limit=${PAGE}&offset=${all.length}` : key, { key: "models" });
        if (mine !== generation) return;
        const page = body.models || [];
        // Append, never replace: a re-render must not drop what is already scrolled past.
        // De-duplicated by pk because a model created between two page fetches shifts the
        // window and would otherwise repeat a row.
        if (paged) {
          const seen = new Set(all.map((r) => r.pk));
          all = all.concat(page.filter((r) => !seen.has(r.pk)));
        } else {
          all = page;
        }
        loadedKey = key;
        modelTotal = body.total ?? all.length;
        // A filtered view holds everything, so it must not offer to load more.
        modelTruncated = paged ? Boolean(body.has_more ?? body.truncated) : false;
        rebuildControls();
        redraw();
      } catch (e) {
        if (!aborted(e) && mine === generation && !all.length) clear(listBox).append(notice("err", e.message));
      } finally { if (mine === generation) pageLoading = false; }
    }
    async function load() { await loadPage(true); }
    // The sentinel sits after the table; when it scrolls into view the next page is
    // fetched. An observer rather than a scroll handler so it costs nothing while idle.
    function attachPager(container) {
      if (!modelTruncated) return;
      const sentinel = el("div", { class: "muted small pager-sentinel" },
        `${all.length} of ${modelTotal} loaded — scrolling loads more`);
      container.append(sentinel);
      observeForMore(sentinel, () => pageLoading, () => {
        if (modelTruncated) loadPage(false);
      });
    }
    // Numbers a person reads and believes: every one carries its unit and the
    // dataset it was measured on, and a Kelvin value is shown only when the
    // model's own archived scale established the conversion.
    function fmtK(v) { return v == null ? "—" : `${v.toFixed(1)} K`; }
    function fmtPct(v) { return v == null ? "—" : `${v.toFixed(2)} %`; }
    function rankNote(rank, unit) {
      if (!rank) return null;
      const parts = [`rank ${rank.rank}/${rank.of}`];
      if (rank.rank === 1) parts.push("best");
      else parts.push(`best ${rank.best.toFixed(1)}${unit} · +${rank.delta.toFixed(1)}${unit}`);
      return el("span", { class: "muted small" }, parts.join(" · "));
    }
    function scorecard(ev, m) {
      const k = ev.kelvin || {}, iou = ev.iou_pct || {}, ranks = ev.ranks || {};
      const norm = !m.t_ref_k;
      const tv = (key) => (norm ? (ev.metrics[key] != null ? ev.metrics[key].toPrecision(3) : "—") : fmtK(k[key]));
      const rows = [
        ["T MAE (substrate)", tv("T.substrate_abs_err_mean"), null],
        ["T MAE (whole domain)", tv("T.abs_err_mean"), rankNote(ranks.t_mae_k, " K")],
        ["T RMSE (substrate)", tv("T.substrate_rmse"), null],
        ["T RMSE (whole domain)", tv("T.rmse"), rankNote(ranks.t_rmse_k, " K")],
        ["T MAE (surface)", tv("T.surface_abs_err_mean"), null],
        ["T RMSE (surface)", tv("T.surface_rmse"), null],
        ["α IoU error", fmtPct(iou["alpha.iou_err_pct"]),
          iou["alpha.iou_std_pp"] != null ? el("span", { class: "muted small" }, `± ${iou["alpha.iou_std_pp"].toFixed(2)} pp`) : null],
        ["fl IoU error", fmtPct(iou["fl.iou_err_pct"]),
          iou["fl.iou_std_pp"] != null ? el("span", { class: "muted small" }, `± ${iou["fl.iou_std_pp"].toFixed(2)} pp`) : null],
      ];
      const grid = el("div", { class: "scorecard" });
      for (const [label, value, note] of rows) {
        grid.append(el("span", { class: "muted small" }, label), el("span", { class: "val" }, value),
          note || el("span"));
      }
      return grid;
    }
    function geometryTable(rows, scale) {
      if (!rows.length) return null;
      // The stored geometry uses one cell size from the z spacing on all three
      // axes, so on an anisotropic grid x/y lengths are under-scaled and volumes
      // more so. The server applies the recorded correction; say so here, or the
      // numbers silently disagree with anything computed from the raw metrics.
      const corrected = rows.some((r) => r.anisotropy_corrected);
      const note = corrected
        ? el("div", { class: "muted small" },
            `x/y lengths ×${scale?.length_correction_xy ?? 2} and volumes ×${scale?.volume_correction ?? 4}: `
            + `this dataset is stride ${(scale?.stride || []).join("×") || "anisotropic"}, so the stored `
            + `geometry was computed with the z cell size on every axis. Depth (z) is unchanged. `
            + `Not comparable with an isotropic evaluation unless both are corrected.`)
        : undefined;
      return el("div", {},
        el("h3", {}, "Melt pool & keyhole"),
        note,
        table(rows, [
          { label: "", w: "22%", get: (r) => el("span", {},
              `${r.region} ${r.quantity}`,
              r.anisotropy_corrected
                ? el("span", { class: "muted small", title: `corrected ×${r.anisotropy_corrected}` }, " ✓scaled")
                : undefined) },
          { label: "true", num: true, get: (r) => (r.true == null ? "—" : `${r.true.toPrecision(4)} ${r.unit}`) },
          { label: "pred", num: true, get: (r) => (r.pred == null ? "—" : `${r.pred.toPrecision(4)} ${r.unit}`) },
          { label: "abs err", num: true, get: (r) => (r.abs_err == null ? "—" : `${r.abs_err.toPrecision(3)} ${r.unit}`) },
          { label: "% of true", num: true, get: (r) => (r.rel_pct == null ? "—" : `${r.rel_pct.toFixed(1)} %`) },
        ]));
    }
    // Curves and the ground-truth comparison are images the server renders;
    // both degrade to one muted line rather than an empty frame.
    function curvePanel(pk, curves) {
      if (!curves || (!curves.validation && !curves.loss)) {
        return el("div", {}, el("h3", {}, "How training went"),
          el("div", { class: "muted small" }, curves?.reason || "No training curves recorded for this model."));
      }
      const kinds = [];
      if (curves.validation) kinds.push(["validation", "Validation MAE / RMSE"]);
      if (curves.loss) kinds.push(["loss", "Training loss"]);
      const img = el("img", { class: "viz-frame", alt: "training curves" });
      const show = (kind) => { img.src = url(`/models/${pk}/curves/plot?kind=${kind}&_=${Date.now()}`); };
      const sel = selectVals2(kinds, show);
      show(kinds[0][0]);
      return el("div", { class: "stack case-viewer" }, el("h3", {}, "How training went"),
        el("div", { class: "row" }, field("Plot", sel)), img);
    }
    function previewPanel(evalPk) {
      const box = el("div", { class: "stack case-viewer" }, el("h3", {}, "Against ground truth"),
        el("div", { class: "muted small" }, "Loading preview\u2026"));
      (async () => {
        let info;
        try { info = await getJson(`/evaluations/${evalPk}/preview`); }
        catch (e) { clear(box); return; }   // nothing stored: the section disappears
        if (!info.fields || !info.fields.length) { clear(box); return; }
        let fieldName = info.fields.includes("T") ? "T" : info.fields[0];
        let z = info.z_default;
        const img = el("img", { class: "viz-frame", alt: "ground truth vs prediction" });
        const zLabel = el("span", { class: "muted small" });
        const draw = () => {
          img.src = url(`/evaluations/${evalPk}/preview/plot?field=${fieldName}&z=${z}`);
          zLabel.textContent = `z ${z + 1}/${info.nz}`;
        };
        const slider = el("input", { type: "range", min: "0", max: String(Math.max(0, info.nz - 1)), value: String(z),
          oninput: (e) => { z = Number(e.target.value); draw(); } });
        clear(box).append(
          el("h3", {}, "Against ground truth"),
          el("div", { class: "muted small" }, "Preview sample \u2014 batch 0 of the eval set. One case, not a subset."),
          el("div", { class: "row" }, field("Field", selectVals(info.fields, (v) => { fieldName = v; draw(); })),
            el("span", { class: "spacer" }), zLabel),
          slider, img);
        draw();
      })();
      return box;
    }

    async function modelDetail(pk) {
      const m = await getJson(`/models/${pk}`);
      const compareWith = el("button", { class: "button small", title:
        "compare this model's provenance, hyperparameters, data and evaluations "
        + "against another model pk" ,
        onclick: async () => {
          const other = prompt(`Compare model #${pk} with which model pk?`);
          if (!other) return;
          clear(scoreBox);
          scoreBox.append(el("div", { class: "detail-empty" }, "Comparing\u2026"));
          try {
            const report = await getJson(`/models/${pk}/compare/${parseInt(other, 10)}`);
            clear(scoreBox);
            scoreBox.append(renderCompareReport(report));
          } catch (err) {
            clear(scoreBox);
            scoreBox.append(el("div", { class: "detail-empty" }, `compare failed: ${err.message}`));
          }
        } }, "Compare with\u2026");
      const evals = m.eval_rows || [];
      let current = evals[0] || null;
      const scoreBox = el("div", { class: "stack" });
      const previewBox = el("div", {});
      const renderEval = () => {
        clear(scoreBox);
        clear(previewBox);
        if (!current) { scoreBox.append(el("div", { class: "muted small" }, "Not evaluated yet.")); return; }
        scoreBox.append(
          el("div", { class: "row" }, el("strong", {}, current.dataset_display || current.dataset_label || "eval"),
            el("span", { class: "muted small" }, `eval #${current.pk}${current.ctime ? " · " + current.ctime.slice(0, 10) : ""}`)),
          // held_out === false is a hard fact stamped on the node; render it
          // louder than the numbers it disqualifies. undefined/null stays
          // silent -- unknown provenance is not an accusation.
          ...(current.held_out === false ? [el("div", {
            style: "border:2px solid #c0392b;background:#c0392b22;color:#c0392b;"
              + "font-weight:700;padding:6px 10px;border-radius:4px",
          }, `⚠ ${current.warning || "SCORED ON TRAINING DATA — training-fit, not a held-out result"}`)] : []),
          scorecard(current, m),
          el("div", { class: "muted small" }, m.t_ref_k
            ? `Kelvin values use ${m.t_ref_source}.`
            : `Normalised units: ${m.t_ref_source} — no Kelvin conversion is shown rather than a guessed one.`),
          ...[geometryTable(current.geometry || [], current.geometry_scale)].filter(Boolean),
          // The evaluation stores set-wide aggregates only, so a keyhole vs
          // conduction split cannot be recovered after the fact.
          el("div", { class: "muted small" },
            "Keyhole/conduction split not available: the evaluation records set-wide aggregates, not per-case metrics."),
          el("details", {}, el("summary", { class: "muted small" }, // Nine of these keys carry their unit in the name (T.melt_threshold_K =
          // 1898.0, geometry.melt_hd95_um = 14.12): reading them as normalised
          // and multiplying by a 3000 K reference gives 5.7 million K.
          `All ${Object.keys(current.metrics || {}).length} raw metrics (as recorded; some keys name their own unit)`),
            table(Object.entries(current.metrics || {}).map(([k, v]) => ({ k, v })), [
              { label: "metric", key: "k" },
              { label: "value", num: true, get: (r) => Number(r.v).toPrecision(5) },
            ])));
        previewBox.append(previewPanel(current.pk));
      };
      renderEval();
      // No fixed widths: in the split layout a 10 % eval column broke "#102904" over three
      // lines and the metric headers clipped. Auto layout sizes each column to its content.
      const evalTable = evals.length ? table(evals, [
        { label: "eval", nowrap: true, get: (r) => `#${r.pk}` },
        { label: "dataset", clamp: true, get: (r) => r.dataset_display || r.dataset_label || "—" },
        { label: "T MAE", num: true, get: (r) => (m.t_ref_k ? fmtK((r.kelvin || {})["T.abs_err_mean"]) : "—") },
        { label: "T RMSE", num: true, get: (r) => (m.t_ref_k ? fmtK((r.kelvin || {})["T.rmse"]) : "—") },
        { label: "α IoU err", num: true, get: (r) => fmtPct((r.iou_pct || {})["alpha.iou_err_pct"]) },
        { label: "fl IoU err", num: true, get: (r) => fmtPct((r.iou_pct || {})["fl.iou_err_pct"]) },
        { label: "", get: (r) => el("span", { class: "chips" }, ...autoChips(r.badges)) },
      ], (r) => { current = r; renderEval(); }, current ? current.pk : null) : el("div", { class: "muted small" }, "Not evaluated yet.");

      return el("div", { class: "stack" },
        el("div", { class: "row" }, starCell({ pk, keywords: m.keywords || [] }, load),
          el("h2", {}, m.model_name || `Model ${pk}`),
          el("span", { class: "muted small" }, `#${pk}`), el("span", { class: "spacer" }),
          compareWith, deleteButton([pk], `model ${pk}`)),
        (m.auto_keywords || []).length
          ? el("div", { class: "chips" }, ...(m.auto_keywords || []).map((t) => el("span", { class: "chip auto", title: "assigned automatically from the evaluations, not a user keyword" }, t)))
          : el("span"),
        el("div", { class: "row" }, el("span", { class: "muted small" }, "Description:"),
          descriptionCell({ pk, description: m.description })),
        el("div", { class: "row" }, el("span", { class: "muted small" }, "Keywords:"),
          keywordCell({ pk, keywords: m.keywords || [] })),
        el("div", { class: "row" }, el("span", { class: "muted small" }, "Folder:"),
          folderCell({ pk, folder: m.folder })),
        el("div", {}, el("h3", {}, "Trained on"),
          kv({ architecture: m.architecture, config_name: m.config_name,
            training_source_display: m.training_source_display,
            // The dataset it read was later merged into another: name the
            // survivor, without pretending the model consumed that node.
            now_named: m.training_source_canonical
              ? `${m.training_source_canonical.display || "dataset"} (#${m.training_source_canonical.pk})`
              : undefined,
            // `training_source_kind` is "dataset_collection" on every model
            // that exists; a row that never varies is furniture. Shown only if
            // it is ever something else.
            training_source_kind: m.training_source_kind === "dataset_collection" ? undefined : m.training_source_kind,
            max_steps: (m.training_profile || {}).max_steps,
            evaluation_count: m.evaluation_count, primary_checkpoint: m.primary_checkpoint })),
        curvePanel(pk, m.curves),
        el("div", {}, el("h3", {}, `Scores (${evals.length} evaluation${evals.length === 1 ? "" : "s"})`), evalTable),
        scoreBox,
        previewBox,
        el("details", {}, el("summary", { class: "muted small" }, "Training profile"), jsonBlock(m.training_profile || {})),
        el("details", {}, el("summary", { class: "muted small" }, "Full metadata"), jsonBlock(m)));
    }
    load();
  }

  // -------------------------------------------------------------- tab: runs
  // The type filter's categories are matched in the query: see
  // `_PROCESS_CATEGORIES` in aiida_pino/console/service.py.
  function renderRuns(view) {
    const sel = { pk: null };
    const listBox = el("div", {}, el("div", { class: "detail-empty" }, "Loading…"));
    const detail = el("div", { class: "panel detail-pane" }, el("div", { class: "detail-empty" }, "Select a run."));
    const onlyPino = el("input", { type: "checkbox", checked: true, onchange: () => reload() });
    const showSmokes = el("input", { type: "checkbox", onchange: () => reload() });
    const f = { cat: "all", state: "all" };
    const catSel = selectVals2([["all", "all types"], ["generation", "data generation (FLOW-3D)"],
      ["conversion", "data import & conversion"], ["models", "model training / eval"],
      ["derivation", "dataset derivation"], ["other", "other"]], (v) => { f.cat = v; reload(); });
    const stateSel = selectVals2([["all", "any state"], ["queued", "queued / running"],
      ["finished", "finished OK"], ["failed", "failed / killed"]], (v) => { f.state = v; reload(); });
    // FLOW-3D seat + spool queue status (broker-free; the spool serializes
    // solves across entry points).
    const queueBox = el("div", { class: "muted small" });
    getJson("/flow3d/queue").then((q) => {
      if (!q || !q.exists) {
        queueBox.textContent = `FLOW-3D spool directory not found${q && q.spool_dir ? ` (${q.spool_dir})` : ""} — queue state unknown.`;
        return;
      }
      const a = q.aiida || {};
      const bits = [
        // Name the holder: "in use" over a 23-deep queue does not say whether
        // the machine is working or wedged, and a run has sat on this seat for
        // days before now.
        // The holder is either an AiiDA calcjob (named by pk) or a spool task
        // (named by the spool's own running entry); both take the same lock.
        `FLOW-3D seat: ${q.seat_held == null ? "unknown" : q.seat_held
          ? `in use${(q.seat_holder || {}).pk ? ` by #${q.seat_holder.pk}`
              : (q.running || []).length ? ` by spool task ${q.running[0].name}` : ""}${
              (q.seat_holder || {}).waiting_pids ? `, ${q.seat_holder.waiting_pids.length} blocked behind it` : ""}`
          : "free"}`,
        // Both entry points take the same seat, so both have to be shown: the
        // spool-only box read "0 queued" while four AiiDA batches were waiting.
        `${q.queued_count} spool task(s) queued`,
        `${a.run_count || 0} AiiDA run(s) in flight`,
      ];
      if (a.batch_count) bits.push(`${a.batch_count} batch workchain(s) active`);
      // 305 tasks sat in queue_held_* and were named on neither tab.
      const held = (q.held || []).reduce((n, h) => n + (h.count || 0), 0);
      if (held) bits.push(`${held} held aside (not draining)`);
      if (q.running && q.running.length) bits.push(`running: ${q.running.map((r) => r.name).join(", ")}`);
      if (q.queued && q.queued.length) bits.push(`next: ${q.queued.slice(0, 3).map((r) => r.name).join(", ")}${q.queued_count > 3 ? "…" : ""}`);
      queueBox.textContent = bits.join(" · ");
    }).catch((e) => {
      // Blank read as "nothing is queued" — and this call fails under exactly
      // the load that already makes the console unreliable, while the thing it
      // hides is whether the seat is held and how deep the backlog is.
      if (!aborted(e)) queueBox.textContent = `FLOW-3D queue status unavailable (${e.message})`;
    });
    view.append(el("div", { class: "split" },
      el("div", { class: "panel stack" },
        el("div", { class: "row" }, el("h2", {}, "Runs"), el("span", { class: "spacer" }),
          el("label", { class: "checkline" }, onlyPino, "aiida-pino only"),
          el("label", { class: "checkline" }, showSmokes, "show smokes"),
          el("button", { class: "button small", onclick: reload }, "Refresh")),
        queueBox,
        el("div", { class: "controls-grid" }, field("Type", catSel), field("State", stateSel)),
        listBox),
      detail));

    async function reload() {
      clear(listBox).append(el("div", { class: "detail-empty" }, "Loading…"));
      try {
        // State and type filters are applied by the query, not to the fetched
        // page: filtering 100 fetched rows for failures returned 2 on a database
        // with 395 killed processes, and for FLOW-3D runs returned 0 whenever
        // the 100 most recent were all evals.
        const q = `/processes?only_pino=${onlyPino.checked}&include_smokes=${showSmokes.checked ? 1 : 0}`
          + `&limit=${state.cfg.listLimit || 100}&state=${encodeURIComponent(f.state)}`
          + `&category=${encodeURIComponent(f.cat)}`;
        const rows = (await getJson(q, { key: "runs" })).processes;
        clear(listBox).append(table(rows, [
          { label: "PK", key: "pk", num: true },
          { label: "Process", get: (r) => runNameCell(r) },
          { label: "State", get: (r) => stateBadge(r.state, r.exit_status) },
          { label: "Exit", key: "exit_status", num: true },
          { label: "Created", key: "ctime", nowrap: true },
        ], (r, tr) => { sel.pk = r.pk; markSel(listBox, tr); showRun(r.pk); }, sel.pk));
        if (rows.length) listBox.append(el("div", { class: "muted small" }, `showing the ${rows.length} most recent matching runs`));
      } catch (e) { if (!aborted(e)) clear(listBox).append(notice("err", e.message)); }
    }
    async function showRun(pk) {
      clear(detail).append(el("div", { class: "detail-empty" }, "Loading…"));
      try {
        const p = await getJson(`/processes/${pk}`);
        const canKill = !p.terminated;
        const killBtn = el("button", { class: "button danger small", disabled: (!daemonOk() || !canKill) ? "disabled" : null }, "Kill");
        const out = el("div", {});
        killBtn.addEventListener("click", async () => {
          killBtn.disabled = true;
          try { await postJson(`/processes/${pk}/kill`, {}); out.append(notice("ok", "kill requested")); }
          catch (e) { out.append(notice("err", e.message)); }
        });
        const titleBtn = el("button", { class: "button small edit-btn", title: "edit run title", onclick: async () => {
          const next = prompt("Run title (short, shown in the list; empty clears)", p.title || "");
          if (next == null) return;
          try { await postAnnotations(pk, { title: next }); showRun(pk); reload(); }
          catch (e) { alert(e.message); }
        } }, "✎ title");
        clear(detail).append(el("div", { class: "stack" },
          el("div", { class: "row" }, el("h2", { title: p.label || "" }, p.title || processName(p.label)),
            el("span", { class: "muted small" }, `#${pk}`), stateBadge(p.state, p.exit_status), el("span", { class: "spacer" }),
            // One unit, so the three buttons wrap together instead of splitting across lines at 1280.
            el("span", { class: "row nowrap" }, titleBtn, killBtn, deleteButton([pk], `run ${pk}`))),
          p.title ? el("div", { class: "muted small" }, processName(p.label)) : null,
          el("div", { class: "row" }, el("span", { class: "muted small" }, "Notes:"), descriptionCell(p)),
          out,
          runWaitLine(p),
          // "Prepare-only run: skipping dataset ingest." is in the workchain
          // report, which this pane does not fetch — so a finished prepare-only
          // run showed outputs, no dataset, and no reason.
          ((p.metrics || {}).run_summary || {}).prepare_only
            ? notice("ok", "Prepare-only run: the case was rendered and validated; no solve, no licence seat, and no dataset ingest.")
            : null,
          kv(p, ["label", "process_type", "state", "exit_status", "process_status", "finished_ok", "ctime", "mtime"]),
          p.terminated ? null : liveLogBox(pk),
          p.exception ? el("div", {}, el("h3", {}, "Exception"), el("pre", { class: "json" }, p.exception)) : null,
          el("div", {}, el("h3", {}, "Outputs"),
            p.outputs && p.outputs.length ? table(p.outputs, [{ label: "label", key: "label" }, { label: "pk", key: "pk", num: true }, { label: "type", key: "type" }]) : el("div", { class: "muted small" }, "none")),
          Object.keys(p.metrics || {}).length ? el("div", {}, el("h3", {}, "Parsed metrics"), jsonBlock(p.metrics)) : null,
          el("div", { class: "row" }, el("button", { class: "button small", onclick: () => { state.logsPk = pk; selectTab("logs"); } }, "Open in Logs →"))));
      } catch (e) { clear(detail).append(notice("err", e.message)); }
    }
    reload();
  }

  function fmtElapsed(iso) {
    if (!iso) return null;
    const seconds = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    if (seconds < 90) return `${Math.round(seconds)} s`;
    if (seconds < 5400) return `${Math.round(seconds / 60)} min`;
    if (seconds < 172800) return `${(seconds / 3600).toFixed(1)} h`;
    return `${(seconds / 86400).toFixed(1)} days`;
  }
  // A queued FLOW-3D run showed a state badge and nothing else, for jobs that
  // wait days behind the seat lock — indistinguishable from the adapter
  // deadlock this project has already hit once. Say how long, and behind what.
  function runWaitLine(p) {
    if (p.terminated) return null;
    const box = el("div", { class: "muted small" }, `${p.state} for ${fmtElapsed(p.ctime) || "?"}`);
    getJson("/flow3d/queue").then((q) => {
      if (!q || !q.exists) return;
      const holder = q.seat_holder || {};
      const ahead = (q.queue || []).findIndex((row) => row.pk === p.pk);
      const bits = [`${p.state} for ${fmtElapsed(p.ctime) || "?"}`];
      if (holder.pk === p.pk) bits.push("holding the FLOW-3D seat");
      else if (ahead > 0) bits.push(`${ahead} run(s) ahead of it in the queue`);
      else if (ahead === 0 && q.seat_held) bits.push("next for the seat");
      if (holder.pk && holder.pk !== p.pk) bits.push(`seat held by #${holder.pk}`);
      box.textContent = bits.join(" · ");
    }).catch(() => {});
    return box;
  }
  // The live console of a running job, which /processes/{pk}/logs already
  // returns and only the Logs tab was showing.
  function liveLogBox(pk) {
    const pre = el("pre", { class: "log" }, "loading…");
    const box = el("details", {}, el("summary", { class: "muted small" }, "Live output (last lines)"), pre);
    box.addEventListener("toggle", async () => {
      if (!box.open) return;
      try {
        const l = await getJson(`/processes/${pk}/logs`);
        // `console` is the job's own stdout; `report` is the workchain's log.
        // A workchain has no console of its own, so fall back to its report.
        const text = (l.console || "").trimEnd()
          || (l.report || []).map((r) => `${r.time} ${r.level} ${r.message}`).join("\n");
        pre.textContent = text ? text.split("\n").slice(-25).join("\n") : "(nothing written yet)";
      } catch (e) { pre.textContent = `unavailable: ${e.message}`; }
    });
    return box;
  }

  // ----------------------------------------------------------- tab: monitor
  // Short user title over the process class; shared by Runs and Monitor.
  function runNameCell(r) {
    const hover = `${r.label || ""}${r.description ? " — " + r.description : ""}`;
    if (!r.title) return el("span", { title: hover }, processName(r.label));
    if (r.title === processName(r.label)) return el("span", { title: hover }, el("strong", {}, r.title));
    return el("span", { title: hover }, el("strong", {}, r.title),
      el("div", { class: "muted small" }, processName(r.label)));
  }
  function meter(frac, text) {
    const pct = Math.max(0, Math.min(100, Math.round((frac || 0) * 100)));
    return el("span", { class: "meter-wrap", title: text || `${pct}%` },
      el("span", { class: "meter" }, el("span", { class: "meter-fill", style: `width:${pct}%` })),
      el("span", { class: "muted small nowrap" }, text || `${pct}%`));
  }
  // Label + meter as one non-wrapping unit, so "mem" cannot end a line with
  // its bar alone on the next one.
  function labelled(label, m) { return el("span", { class: "row nowrap" }, el("span", { class: "muted small" }, label), m); }
  function fmtMib(v) {
    if (v == null) return "—";
    return v >= 1024 ? `${(v / 1024).toFixed(1)} GiB` : `${v} MiB`;
  }
  // Coarse age for a status line: "4 min", "2.1 h", "3.0 days".
  function fmtAge(seconds) {
    const s = Math.max(0, Number(seconds) || 0);
    if (s < 90) return `${Math.round(s)} s`;
    if (s < 5400) return `${Math.round(s / 60)} min`;
    if (s < 172800) return `${(s / 3600).toFixed(1)} h`;
    return `${(s / 86400).toFixed(1)} days`;
  }

  function progressCell(r) {
    const p = r.progress;
    if (!p) return r.phase === "queued" ? el("span", { class: "badge info" }, "queued") : el("span", { class: "muted small" }, "—");
    if (p.waiting_for_gpu) return el("span", { class: "badge warn" }, "waiting for GPU seat");
    if (p.step == null) return el("span", { class: "muted small" }, "starting…");
    const frac = p.max_steps ? p.step / p.max_steps : 0;
    // A tail says what a run REACHED, not that it is still running: a stopped job's log
    // keeps its last step forever. a compute box sat at 0% util with no compute processes while
    // two of its runs showed a moving step count. So a log nobody has written to for a
    // while is rendered as a LAST-SEEN reading, not as progress.
    // Holding a GPU is the fact; the log's age is only a fallback for when no snapshot of
    // that machine came back. A 300 s threshold on its own called an AR run stale while it
    // sat at 100% util -- it just logs every ~66 s -- and called a run that finished ten
    // minutes ago live. `on_gpu === false` with a snapshot in hand is not a heuristic.
    const STALE_AFTER_S = 900;
    const age = p.tail_age_s;
    const stale = p.gpu_snapshot_seen && p.on_gpu === false
      ? true
      : (typeof age === "number" && age > STALE_AFTER_S);
    const bar = stale
      ? el("span", { class: "muted small", title:
            p.on_gpu === false
              ? "this run holds no GPU process, so it is not training now; this is the last "
                + `step its log recorded${typeof age === "number" ? `, ${fmtAge(age)} ago` : ""}`
              : `the live log has not been written for ${fmtAge(age)}; this is the last step `
                + "it recorded, not current progress — the job may have stopped" },
          `last seen step ${p.step}${p.max_steps ? ` / ${p.max_steps}` : ""}`
          + (typeof age === "number" ? ` · ${fmtAge(age)} ago` : ""))
      : meter(frac, `step ${p.step}${p.max_steps ? ` / ${p.max_steps}` : ""}`);
    // The beam-box MAE beside the step: a counter says a run is moving, this says whether
    // it is moving in the right direction. Kelvin ONLY when a scale actually resolved --
    // otherwise the raw value is shown and marked `norm.`, because the logged number is
    // normalised and a Kelvin converted by guess is worse than an honest ratio.
    const bb = p.beambox_mae;
    // A stale row shows no BB MAE: the number would be as old as the step beside it, and a
    // metric with no time attached is exactly what gets quoted later as current.
    if (!bb || stale) return bar;
    const kelvin = p.beambox_kelvin;
    const label = kelvin != null ? `${kelvin} K` : `${Number(bb.value).toPrecision(3)} norm.`;
    const title = kelvin != null
      ? `beam-box MAE at step ${bb.step}; Kelvin from ${p.beambox_kelvin_source || "the shipped config"}`
      : `beam-box MAE at step ${bb.step}, normalised — no reference scale resolved, so no Kelvin is shown`;
    return el("div", { class: "stack-tight" }, bar,
      el("span", { class: "muted small", title }, `BB MAE ${label}`));
  }
  async function renderMonitor(view) {
    const box = el("div", { class: "panel stack" }, el("div", { class: "detail-empty" }, "Loading…"));
    view.append(box);
    // /monitor answers in 3-12 s under load on a single uvicorn worker, so a
    // fixed 5 s interval queued requests behind each other and starved every
    // other tab. Reschedule 5 s after the previous answer instead, and report
    // the interval that is actually happening.
    let timer = null, gap = 5;
    const stop = () => { if (timer) clearTimeout(timer); timer = null; };
    load();

    async function load() {
      stop();
      if (state.activeTab !== "monitor" || !document.body.contains(box)) return;
      const started = Date.now();
      let m;
      try { m = await getJson("/monitor", { key: "monitor" }); }
      catch (e) {
        // Keep the last good snapshot on screen; a bare error box for the 5 s
        // retry window reads as "the fleet is gone".
        if (!aborted(e)) {
          const stale = box.querySelector(".notice.err"); if (stale) stale.remove();
          box.prepend(notice("err", `${e.message} — retrying in 5 s`));
        }
        timer = setTimeout(load, 5000); return;
      }
      gap = Math.round((Date.now() - started) / 1000) + 5;
      timer = setTimeout(load, 5000);
      const gpuPanels = (m.gpus || []).map((g) => el("div", { class: "subpanel stack" },
        // Name the machine. Two boxes both have a "GPU 0", so without this the
        // panel shows repeated indices and the reader cannot tell which card is
        // where -- the same ambiguity as a metric that does not say which
        // regime produced it.
        el("h3", {}, `${g.hostname ? `${g.hostname} · ` : ""}GPU ${g.index} · ${g.name}`),
        el("div", { class: "row" },
          labelled("util", meter((g.util_pct || 0) / 100, `${g.util_pct == null ? "?" : g.util_pct}%`)),
          labelled("mem", meter(g.mem_total_mib ? g.mem_used_mib / g.mem_total_mib : 0, `${fmtMib(g.mem_used_mib)} / ${fmtMib(g.mem_total_mib)}`))),
        g.procs && g.procs.length ? table(g.procs, [
          { label: "PID", key: "pid", num: true },
          { label: "GPU mem", get: (p) => fmtMib(p.mem_mib), num: true },
          { label: "What", get: (p) => p.run_pk
            ? el("span", {}, el("strong", {}, p.run_title || processName(p.run_label)), el("span", { class: "muted small" }, ` · run #${p.run_pk}`))
            // Never render an unnamed process as "?". Beside cards that DO name
            // their run, a "?" reads as "nothing running" — that misled two
            // readers into calling a box idle while it trained at step 7100.
            // An unidentified process must still announce that it exists.
            : p.cmd
              ? el("span", { class: "mono", title: p.cmd }, p.cmd.length > 90 ? `${p.cmd.slice(0, 87)}…` : p.cmd)
              : el("span", { class: "muted" }, `unidentified process (pid ${p.pid})`) },
        ]) : el("div", { class: "muted small" },
          // "no compute processes" is a claim about the BOX. Say it only when
          // the query succeeded and returned nothing; a query that FAILED must
          // not render as an idle card (a compute box showed 100% util with an empty
          // list). Same rule as the unidentified-pid case above.
          g.processes_unknown
            ? "process list unavailable on this computer — the utilisation above is still accurate"
            : "no compute processes")));
      const cpu = m.cpu || {};
      const cpuPanel = el("div", { class: "subpanel stack" },
        el("h3", {}, "CPU / memory"),
        el("div", { class: "row" },
          labelled("load", meter(cpu.cores ? cpu.load_1m / cpu.cores : 0, `${cpu.load_1m == null ? "?" : cpu.load_1m} / ${cpu.cores || "?"} cores`)),
          labelled("RAM", meter(cpu.mem_total_gib ? 1 - cpu.mem_available_gib / cpu.mem_total_gib : 0,
            cpu.mem_total_gib ? `${(cpu.mem_total_gib - cpu.mem_available_gib).toFixed(1)} / ${cpu.mem_total_gib} GiB` : "?"))));
      const active = m.active_processes || [];
      const spool = m.gpu_spool || [];
      const f3 = m.flow3d || {};
      const gpuAiida = m.aiida_gpu_running || 0;
      const gpuQueue = m.gpu_queue || [];
      const f3aiida = f3.aiida_running || 0;
      const f3Queue = f3.queue || [];
      // append() renders null children as the text "null" — filter them.
      clear(box).append(...[
        el("div", { class: "row" }, el("h2", {}, "Monitor"), el("span", { class: "spacer" }),
          el("span", { class: "muted small" }, `refreshes 5 s after each answer (~${gap} s) · updated ${new Date().toLocaleTimeString()}`)),
        m.gpu_note ? notice("warn", m.gpu_note) : null,
        Object.keys(m.aiida_calcjobs || {}).length
          ? el("div", { class: "muted small" }, "active calcjobs: "
              + Object.entries(m.aiida_calcjobs).map(([k, v]) => `${v}\u00d7 ${processName(k)}`).join(" · "))
          : null,
        el("div", { class: "monitor-grid" }, ...gpuPanels, cpuPanel),
        el("h3", {}, "Active AiiDA runs"),
        active.length ? table(active, [
          { label: "PK", key: "pk", num: true },
          { label: "Process", get: (r) => runNameCell(r) },
          { label: "State", get: (r) => stateBadge(r.state) },
          { label: "Progress", get: (r) => progressCell(r) },
          // Three batch workchains have been "waiting" since 2026-07-31 with no
          // elapsed column: a badge alone cannot separate slow from wedged.
          { label: "Waiting", nowrap: true, get: (r) => el("span", { class: "muted small" }, fmtElapsed(r.ctime) || "—") },
          { label: "Created", key: "ctime", nowrap: true },
        ], (r) => { state.logsPk = r.pk; selectTab("logs"); }) : el("div", { class: "muted small" }, "no active runs"),
        el("div", { class: "row" }, el("h3", {}, "GPU queue"),
          gpuAiida ? el("span", { class: "badge warn" }, `${gpuAiida} running`) : null,
          m.aiida_gpu_queued || gpuQueue.length
            ? el("span", { class: "badge info" }, `${Math.max(m.aiida_gpu_queued || 0, gpuQueue.length)} queued`) : null),
        // What each card holds right now (running procs + booked seats). No
        // per-card "queued" count: the queue is not per-card and lives below.
        spool.length ? el("div", { class: "stack" }, ...spool.map((s) => el("div", { class: "row" },
          el("strong", {}, s.gpu),
          el("span", { class: "muted small" }, s.running.length ? `running: ${s.running.join(", ")}`
            : (s.busy || s.aiida_procs) ? "busy (AiiDA seat)" : "idle"),
          s.seat_count ? el("span", { class: "badge warn" },
            `${s.seat_count} seat${s.seat_count === 1 ? "" : "s"}`
            + `${s.seat_vram_mb ? ` · ${fmtMib(s.seat_vram_mb)} booked` : ""}`) : null,
          s.seats && s.seats.length ? el("span", { class: "muted small" },
            s.seats.map((h) => h.name).slice(0, 4).join(", ")) : null)))
          : el("div", { class: "muted small" }, "no gpu spool on this machine"),
        // One queue for AiiDA-submitted and spool-submitted work, in order,
        // with the VRAM each job booked (blank = exclusive / whole card).
        gpuQueue.length ? el("div", { class: "stack" }, ...gpuQueue.map((q) => el("div", { class: "row" },
          el("span", { class: "muted small" }, `${q.pos}.`),
          q.pk ? el("a", { class: "small", onclick: () => { state.logsPk = q.pk; selectTab("logs"); } }, `#${q.pk}`)
            : el("span", { class: "badge" }, "spool"),
          el("span", { class: "small" }, q.title || processName(q.name)),
          el("span", { class: "muted small" }, q.vram_mb ? `${fmtMib(q.vram_mb)} booked`
            : q.vram_auto ? "auto-sized from measured peak"
            : (q.source === "aiida" ? "exclusive card" : "")),
          el("span", { class: "spacer" }),
          el("span", { class: `badge ${q.source === "aiida" ? "info" : ""}` }, q.source))))
          : el("div", { class: "muted small" }, "nothing queued"),
        el("div", { class: "row" }, el("h3", {}, "FLOW-3D queue"),
          el("span", { class: `badge ${f3.seat_held || f3aiida ? "warn" : "ok"}` },
            f3.seat_held == null && !f3aiida ? "seat unknown"
              : f3.seat_held ? `seat in use${(f3.seat_holder || {}).pk ? ` · #${f3.seat_holder.pk}` : ""}`
                : f3aiida ? "seat between cases" : "seat free"),
          f3aiida ? el("span", { class: "badge warn" }, `${f3aiida} running`) : null,
          // Rows, not cases: one grouped spool row stood for 20 tasks, so the
          // badge read "1 queued" directly above a line reading "20 cases".
          // Batch cases not yet materialised count too: with
          // max_concurrent_runs=1 only one child process exists, so 299 of
          // one batch's cases (~3 days of seat time) were invisible here.
          f3Queue.length || f3.pending_batch_cases ? el("span", { class: "badge info" },
            `${f3Queue.reduce((n, q) => n + (q.count || 1), 0) + (f3.pending_batch_cases || 0)} queued`) : null),
        f3.running && f3.running.length ? el("div", { class: "muted small" }, `running: ${f3.running.map((r) => r.name).join(", ")}`) : null,
        // One queue: AiiDA-submitted runs (one at a time) and any
        // spool batches, in order.
        f3Queue.length ? el("div", { class: "stack" }, ...f3Queue.map((q) => el("div", { class: "row" },
          el("span", { class: "muted small" }, `${q.pos}.`),
          q.pk ? el("a", { class: "small", onclick: () => { state.logsPk = q.pk; selectTab("logs"); } }, `#${q.pk}`)
            : el("span", { class: "badge" }, "spool"),
          el("span", { class: "small" }, q.title || processName(q.name) || q.name),
          q.count ? el("span", { class: "muted small" }, `${q.count} case${q.count === 1 ? "" : "s"}`) : null,
          el("span", { class: "spacer" }),
          el("span", { class: `badge ${q.source === "aiida" ? "info" : ""}` }, q.source))))
          : el("div", { class: "muted small" }, "nothing queued"),
        // Live batches with unmaterialised cases: one row per batch, so the
        // committed work is visible even when the process queue above is empty.
        ...(((f3.aiida || {}).batches) || []).filter((b) => b.pending_runs).map((b) => el("div", { class: "row" },
          el("a", { class: "small", onclick: () => { state.logsPk = b.pk; selectTab("logs"); } }, `#${b.pk}`),
          el("span", { class: "small" }, b.title || "batch"),
          el("span", { class: "muted small" },
            `${b.pending_runs} of ${b.planned_runs} case(s) not yet started`),
          el("span", { class: "spacer" }),
          el("span", { class: "badge info" }, "batch"))),
        f3.held && f3.held.length ? el("div", { class: "muted small" },
          `held (not draining): ${f3.held.map((h) => `${h.name} (${h.count})`).join(", ")}`) : null,
        f3.total_waiting != null ? el("div", { class: "muted small" },
          `${f3.total_waiting} case(s) waiting in total (${f3.queued_count || 0} spool + ${f3.aiida_queued || 0} AiiDA`
          + `${f3.pending_batch_cases ? ` + ${f3.pending_batch_cases} batch case(s) not yet started` : ""})`
          + `${(f3.held || []).length ? ` · ${(f3.held || []).reduce((n, h) => n + (h.count || 0), 0)} held aside` : ""}`) : null,
      ].filter(Boolean));
    }
  }

  // -------------------------------------------------------------- tab: logs
  function renderLogs(view) {
    const pkInput = el("input", { type: "number", placeholder: "process pk" });
    const linesInput = el("input", { type: "number", value: "200" });
    const placeholder = el("option", { value: "" }, "loading recent runs…");
    const recent = el("select", {}, placeholder);
    recent.addEventListener("change", () => { if (recent.value) { pkInput.value = recent.value; loadLogs(); } });
    getJson("/processes?only_pino=true&limit=50").then((b) => {
      placeholder.textContent = "— pick a recent run —";
      // Calcfunctions (process_type is a dotted python path, not an
      // `aiida.calculations:`/`aiida.workflows:` entry point) record neither a
      // report log nor a console, so they only ever rendered "(none)/(none)".
      recent.append(...(b.processes || []).filter((p) => /^aiida\./.test(p.process_type || "")).map((p) =>
        el("option", { value: p.pk }, `#${p.pk} · ${processName(p.label)} · ${p.state || "?"}`)));
      if (pkInput.value) currentPk(); // a pk handed over from another tab loaded before the options did
    }).catch(() => { placeholder.textContent = "(recent runs unavailable)"; });
    const out = el("div", { class: "stack" }, el("div", { class: "detail-empty" }, "Pick a recent run or enter a process pk."));
    const tailOut = el("div", { class: "stack" });
    view.append(el("div", { class: "panel stack" },
      el("div", { class: "row" }, el("h2", {}, "Logs")),
      // Bare buttons in the grid: wrapping them in a blank-labelled <label>
      // gave them an empty accessible name and a clickable invisible strip.
      el("div", { class: "controls-grid wide" },
        el("label", { style: "grid-column: span 2" }, el("span", {}, "Recent runs"), recent), // option text runs to ~40 chars
        field("Process pk", pkInput),
        field("Tail lines", linesInput),
        el("div", { class: "row", style: "align-self: end" },
          el("button", { class: "button", onclick: loadLogs }, "Load logs"),
          el("button", { class: "button", onclick: loadTail }, "Remote tail"))),
      out, tailOut));
    if (state.logsPk) { pkInput.value = state.logsPk; state.logsPk = null; loadLogs(); }

    function currentPk() {
      const pk = parseInt(pkInput.value, 10);
      // Keep the dropdown honest: it showed the earlier pick while the panes
      // held a pk typed into the input.
      recent.value = String(pk || "");
      if (recent.selectedIndex < 0) recent.selectedIndex = 0;
      return pk;
    }
    async function loadLogs() {
      const pk = currentPk();
      if (!pk) { clear(out).append(notice("warn", "enter a pk")); return; }
      clear(tailOut);
      clear(out).append(el("div", { class: "detail-empty" }, "Loading…"));
      try {
        const l = await getJson(`/processes/${pk}/logs`);
        const rep = (l.report || []).map((r) => `[${r.time || ""}] ${r.level}: ${r.message}`).join("\n");
        clear(out).append(
          el("div", {}, el("h3", {}, "Report logs"), el("pre", { class: "log" }, rep || "(none)")),
          el("div", {}, el("h3", {}, "CalcJob console"), el("pre", { class: "log" }, l.console || "(none)")));
      } catch (e) { clear(out).append(notice("err", e.message)); }
    }
    async function loadTail() {
      const pk = currentPk();
      if (!pk) { clear(tailOut).append(notice("warn", "enter a pk")); return; }
      // The tail is its own section under the loaded panes, not a replacement for them.
      clear(tailOut).append(el("div", { class: "detail-empty" }, "Tailing…"));
      try {
        const t = await getJson(`/processes/${pk}/tail?lines=${parseInt(linesInput.value, 10) || 200}`);
        clear(tailOut).append(
          t.note ? notice("warn", t.note) : null,
          // `tail: null` means the file is not there; only real output goes in
          // the log pane, where it is indistinguishable from the job's own.
          el("div", {}, el("h3", {}, `Remote tail ${t.path ? "· " + t.path : ""}`),
            t.tail == null ? el("div", { class: "muted small" }, "No live tail for this job.")
              : el("pre", { class: "log" }, t.tail || "(empty)")));
      } catch (e) { clear(tailOut).append(notice("err", e.message)); }
    }
  }

  // ------------------------------------------------------- tab: train / eval
  // Coarse architecture family of a training config name (mirrors the model
  // registry's families, so the dropdowns speak the same language).
  async function renderCompose(view) {
    const panel = el("div", { class: "panel stack" }, el("h2", {}, "Compose new dataset"));
    view.append(panel);
    const filterKeywords = [];
    const source = selectVals(["all", "flow3d"], () => {});
    const folder = el("input", { type: "text", placeholder: "folder contains…" });
    const splitFilter = selectVals(["all", ...SPLIT_VALUES], () => {});
    const kwFilter = chipsInput(filterKeywords, () => {});
    const listBox = el("div", {}, el("div", { class: "detail-empty" }, "Refresh to load simulations."));
    const name = el("input", { type: "text", placeholder: "collection name" });
    const desc = el("input", { type: "text", placeholder: "optional description" });
    const collKeywords = [];
    const collKw = chipsInput(collKeywords, () => {});
    const skipDesc = el("input", { type: "checkbox" });
    const filteredBtn = el("button", { class: "button" }, "Create collection from filtered sims");
    const out = el("div", {});
    const picked = new Set();

    function simQuery(limit) {
      const p = new URLSearchParams({ source: source.value, folder: folder.value, limit: String(limit) });
      if (splitFilter.value !== "all") p.set("split", splitFilter.value);
      for (const k of filterKeywords) p.append("keyword", k);
      return `/simulations?${p.toString()}`;
    }
    function collectionBody(pks) {
      return {
        simulation_pks: pks,
        collection_name: name.value.trim(),
        skip_description: skipDesc.checked,
        description: desc.value.trim() || null,
        keywords: collKeywords.length ? [...collKeywords] : null,
      };
    }

    async function refresh() {
      picked.clear(); // the rebuilt table has fresh unchecked boxes; a hidden pick would still be sent
      clear(out);
      clear(listBox).append(el("div", { class: "detail-empty" }, "Loading…"));
      try {
        const pageLimit = state.cfg.listLimit || 100;
        const rows = (await getJson(simQuery(pageLimit))).simulations;
        filteredBtn.textContent = `Create collection from ${rows.length}${rows.length >= pageLimit ? "+" : ""} filtered sims`;
        clear(listBox).append(table(rows, [
          { label: "Pick", w: "4%", get: (r) => { const cb = el("input", { type: "checkbox", onclick: (e) => { e.stopPropagation(); e.target.checked ? picked.add(r.pk) : picked.delete(r.pk); } }); return cb; } },
          { label: "Case", w: "30%", get: (r) => el("div", { class: "ds-name" },
              el("strong", {}, r.case_name || r.label || `#${r.pk}`),
              el("span", { class: "ds-sub" }, `#${r.pk} · ${r.source_display || r.source_folder_name || ""}`)) },
          { label: "P", key: "laser_power", num: true, w: "9%" },
          { label: "V", key: "V_scan", num: true, w: "9%" },
          { label: "Steps", key: "num_timesteps", num: true, w: "7%" },
          { label: "Split", w: "11%", get: splitCell },
          { label: "Keywords", get: keywordCell },
        ], null));
      } catch (e) { clear(listBox).append(notice("err", e.message)); }
    }
    async function previewSel() {
      if (!picked.size) { clear(out).append(notice("warn", "pick some simulations")); return; }
      clear(out).append(el("div", { class: "detail-empty" }, "Validating…"));
      try {
        const r = await postJson("/preview/collection", { simulation_pks: [...picked] });
        clear(out).append(notice("ok", `${r.member_count} compatible member(s) · signature ${r.signature || "—"}`), jsonBlock(r));
      }
      catch (e) { clear(out).append(notice("err", e.message)); }
    }
    async function save() {
      if (!picked.size || !name.value.trim()) { clear(out).append(notice("warn", "pick simulations and enter a name")); return; }
      clear(out).append(el("div", { class: "detail-empty" }, "Saving…"));
      try { const r = await postJson("/collections", collectionBody([...picked])); clear(out).append(notice("ok", `collection pk ${r.pk} (${r.created ? "created" : "reused"})`)); }
      catch (e) { clear(out).append(notice("err", e.message)); }
    }
    // The same two checks the server makes, before the query that costs 24-72 s
    // under load: a missing description came back as a 422 only after that wait.
    function missingField() {
      if (!name.value.trim()) return "enter a collection name";
      if (!desc.value.trim() && !skipDesc.checked) {
        return "enter a description (what this dataset is for and how it was selected), "
          + "or tick \"no description\" if it genuinely needs none";
      }
      return null;
    }
    async function createFromFilter() {
      const problem = missingField();
      if (problem) { clear(out).append(notice("warn", problem)); return; }
      clear(out).append(el("div", { class: "detail-empty" }, "Collecting matching simulations…"));
      try {
        const rows = (await getJson(simQuery(SERVER_LIST_CAP))).simulations || [];
        if (!rows.length) { clear(out).append(notice("warn", "no simulations match the current filters")); return; }
        if (rows.length >= SERVER_LIST_CAP) {
          clear(out).append(notice("warn",
            `Filter matches hit the server list cap (${SERVER_LIST_CAP}); the full match set may be larger. ` +
            "Narrow the filters instead of creating a silently truncated collection."));
          return;
        }
        const r = await postJson("/collections", collectionBody(rows.map((x) => x.pk)));
        clear(out).append(notice("ok", `collection pk ${r.pk} (${r.created ? "created" : "reused"}) from ${rows.length} filtered sim(s)`));
      } catch (e) { clear(out).append(notice("err", e.message)); }
    }
    filteredBtn.addEventListener("click", createFromFilter);

    // Combine whole datasets: union of all members (deduped), role "composed".
    const dsMulti = el("select", { multiple: "multiple", size: "8", style: "min-width:320px" });
    // display_name alone made derived variants of one dataset read identically; the label carries the derivation.
    datasetOptions((d) => `#${d.pk} · ${d.label || d.display_name}${d.simulation_count != null ? ` (${d.simulation_count} sims)` : ""}`)
      .then((opts) => dsMulti.append(...opts));
    const dsFilter = el("input", { type: "text", placeholder: "filter datasets…", style: "max-width:320px", oninput: () => {
      const norm = (t) => t.toLowerCase().replace(/_/g, " "); // labels use underscores, people type spaces
      const q = norm(dsFilter.value.trim());
      for (const o of dsMulti.options) o.hidden = q !== "" && !norm(o.textContent).includes(q);
    } });
    const combineBtn = el("button", { class: "button primary", onclick: combine }, "Combine selected datasets");
    async function combine() {
      const pks = [...dsMulti.selectedOptions].map((o) => parseInt(o.value, 10));
      if (pks.length < 1) { clear(out).append(notice("warn", "select at least one dataset")); return; }
      const problem = missingField();
      if (problem) { clear(out).append(notice("warn", problem)); return; }
      clear(out).append(el("div", { class: "detail-empty" }, "Combining…"));
      try {
        const r = await postJson("/collections", { ...collectionBody([]), dataset_pks: pks });
        clear(out).append(notice("ok", `composed dataset pk ${r.pk} (${r.created ? "created" : "reused"}) from ${pks.length} dataset(s)`));
      } catch (e) { clear(out).append(notice("err", e.message)); }
    }

    panel.append(
      el("div", { class: "form-grid" }, field("New dataset name", name), field("Description", desc),
        field("Keywords", collKw)),
      el("label", { class: "checkline" }, skipDesc, "no description (say so deliberately)"),
      el("div", {}, el("h3", {}, "Combine datasets"),
        el("div", { class: "muted small", style: "margin:4px 0 8px" },
          "Union of all member simulations of the selected datasets (Ctrl/Cmd-click for several). The result is a composed dataset."),
        dsFilter,
        el("div", { class: "row", style: "margin-top:6px" }, dsMulti, combineBtn)),
      el("hr"),
      el("div", {}, el("h3", {}, "Pick individual simulations"),
        el("div", { class: "controls-grid" },
          field("Source", source), field("Folder", folder), field("Split", splitFilter),
          field("Keywords (all must match)", kwFilter),
          el("button", { class: "button", style: "align-self:end", onclick: refresh }, "Refresh"))),
      listBox,
      el("div", { class: "row" },
        el("button", { class: "button", onclick: previewSel }, "Preview compatibility"),
        el("button", { class: "button primary", onclick: save }, "Create from picked sims"),
        filteredBtn),
      out);
    refresh();
  }

  // ------------------------------------------------- annotations (broker-free)
  const SPLIT_VALUES = ["train", "val", "test", "unassigned"];
  const SUGGESTED_KEYWORDS = ["keyhole", "conduction", "lof", "random path", "variable material",
    "variable path length", "straight path", "even timesteps", "steady", "downsampled", "smoke"];
  // ponytail: mirrors Settings.list_max_limit default; only used to detect truncation.
  const SERVER_LIST_CAP = 500;

  function ensureKwDatalist() {
    if ($("kwSuggest")) return;
    document.body.append(el("datalist", { id: "kwSuggest" },
      ...SUGGESTED_KEYWORDS.map((k) => el("option", { value: k }))));
  }

  // Local chips input (filters / collection form): edits `values` in place.
  function chipsInput(values, onChange) {
    ensureKwDatalist();
    const box = el("span", { class: "chips" });
    const input = el("input", { type: "text", class: "chip-add", list: "kwSuggest", placeholder: "add keyword…" });
    function redraw() {
      clear(box);
      for (const k of values) {
        box.append(el("span", { class: "chip" }, k, el("button", { type: "button", title: "remove", onclick: () => {
          values.splice(values.indexOf(k), 1); redraw(); if (onChange) onChange(values);
        } }, "×")));
      }
      box.append(input);
    }
    function commit() {
      const v = input.value.trim();
      input.value = "";
      if (v && !values.includes(v)) { values.push(v); redraw(); if (onChange) onChange(values); }
    }
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); commit(); } });
    input.addEventListener("change", commit); // fires on datalist pick
    redraw();
    return box;
  }

  async function postAnnotations(pk, body) { return postJson(`/nodes/${pk}/annotations`, body); }

  // A star is the keyword STAR_KEYWORD with a one-click toggle, so it filters,
  // lists and survives exactly like every other keyword (CLI: aiida-pino tag
  // <pk> --add starred). onChange lets a caller redraw when the row flips.
  const STAR_KEYWORD = "starred";
  const isStarred = (row) => (row.keywords || []).includes(STAR_KEYWORD);
  function starCell(row, onChange) {
    const btn = el("button", { class: "star-toggle", type: "button" });
    function paint() {
      btn.textContent = isStarred(row) ? "★" : "☆";
      btn.classList.toggle("is-on", isStarred(row));
      btn.title = isStarred(row) ? "Starred — click to unstar" : "Star this model";
    }
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const body = isStarred(row) ? { keywords_remove: [STAR_KEYWORD] } : { keywords_add: [STAR_KEYWORD] };
      btn.disabled = true;
      try { writeThrough(row, { keywords: (await postAnnotations(row.pk, body)).keywords || [] }); }
      catch (err) { alert(err.message); }
      btn.disabled = false;
      paint();
      if (onChange) onChange();
    });
    paint();
    return btn;
  }

  // Row cell: keyword chips synced to the annotations endpoint on add/remove.
  // Rows are cloned for display (indent, folder nesting, group headers). An
  // edit written to the clone was lost on the next re-render, which every
  // filter and expand triggers with the cache kept — so a keyword or a
  // description saved on the server reverted on screen a moment later.
  function writeThrough(row, patch) {
    Object.assign(row, patch);
    if (row._src) Object.assign(row._src, patch);
  }
  function keywordCell(row) {
    ensureKwDatalist();
    const box = el("span", { class: "chips", onclick: (e) => e.stopPropagation() });
    const input = el("input", { type: "text", class: "chip-add", list: "kwSuggest", placeholder: "+" });
    async function post(body) {
      try { writeThrough(row, { keywords: (await postAnnotations(row.pk, body)).keywords || [] }); }
      catch (e) { alert(e.message); }
      redraw();
    }
    function redraw() {
      clear(box);
      for (const k of row.keywords || []) {
        box.append(el("span", { class: "chip", title: k }, k,
          el("button", { type: "button", title: "remove", onclick: () => post({ keywords_remove: [k] }) }, "×")));
      }
      box.append(input);
    }
    function commit() { const v = input.value.trim(); input.value = ""; if (v) post({ keywords_add: [v] }); }
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); commit(); } });
    input.addEventListener("change", commit);
    redraw();
    return box;
  }

  // Row cell: split select for simulations, saved on change.
  function splitCell(row) {
    const sel = el("select", { class: "split-select", onclick: (e) => e.stopPropagation() },
      ...SPLIT_VALUES.map((v) => el("option", { value: v, selected: (row.split || "unassigned") === v ? "selected" : null }, v)));
    sel.addEventListener("change", async () => {
      try { row.split = (await postAnnotations(row.pk, { split: sel.value })).split; }
      catch (e) { alert(e.message); sel.value = row.split || "unassigned"; }
    });
    return sel;
  }

  // Row cell: clamped description text with an inline prompt()-based edit button.
  function folderCell(row) {
    // Same control as the bulk bar: typing a name that does not exist creates
    // the folder (set_folder does get_or_create), so there is no separate
    // "new folder" step to find.
    const text = el("span", {}, row.folder || "—");
    const wrap = el("div", { class: "desc-cell" }, text);
    const commit = async (input) => {
      try {
        row.folder = (await postJson("/folders/move", { pks: [row.pk], path: input.value })).folder;
        text.textContent = row.folder || "—";
      } catch (err) { alert(err.message); }
      clear(wrap).append(text, btn);
    };
    const btn = el("button", { class: "button small edit-btn", title: "move to folder", onclick: (e) => {
      e.stopPropagation();
      const input = el("input", { class: "folder-input", list: "folder-names", value: row.folder || "",
        placeholder: "folder or folder/sub… (empty to unfile)",
        onkeydown: (ev) => {
          ev.stopPropagation();
          if (ev.key === "Enter") commit(input);
          if (ev.key === "Escape") clear(wrap).append(text, btn);
        },
      });
      clear(wrap).append(input,
        el("button", { class: "button small", onclick: (ev) => { ev.stopPropagation(); commit(input); } }, "Save"));
      input.focus();
      input.select();
    } }, "✎");
    wrap.append(btn);
    return wrap;
  }

  function descriptionCell(row) {
    const text = clampEl(row.description || "—");
    const btn = el("button", { class: "button small edit-btn", title: "edit description", onclick: async (e) => {
      e.stopPropagation();
      const next = prompt("Description", row.description || "");
      if (next == null) return;
      try {
        writeThrough(row, { description: (await postAnnotations(row.pk, { description: next })).description });
        text.textContent = row.description || "—";
      } catch (err) { alert(err.message); }
    } }, "✎");
    return el("div", { class: "desc-cell" }, text, btn);
  }

  // ------------------------------------------------- typed profile editor
  // Per-key controls (dropdowns for enums/booleans, number inputs) instead of
  // a raw JSON blob; raw JSON stays available as a collapsible escape hatch.
  const PROFILE_ENUMS = {
    split_mode: ["standard", "kfold"],
    export: ["npz", "vtk", "vtk-surface", "vtk-legacy60", "both", "none"],
  };
  function profileEditor(initial) {
    const obj = JSON.parse(JSON.stringify(initial || {}));
    // Keys whose text does not parse. A failed parse used to only recolour the
    // border: `obj[key]` kept its previous value and `get()` returned it, so a
    // typo in the `laser` block produced a success notice, a pk, and a job
    // running the *old* parameters — a wrong-physics run that spends a seat and
    // whose provenance Dict records a config the user never saw.
    const invalid = new Set();
    const grid = el("div", { class: "profile-grid" });
    const rawTa = el("textarea", {});
    const rawErr = el("div", { class: "muted small" });
    const syncRaw = () => { rawTa.value = JSON.stringify(obj, null, 2); };

    function schemaHint(path) {
      const entry = (state.flow3dSchema || {})[path];
      if (!entry) return path;
      return `${path}\n${entry.meaning}\n${entry.constraint}`;
    }
    function nestedBlock(section, value) {
      const box = el("div", { class: "profile-grid nested" });
      const draw = () => {
        clear(box);
        for (const leaf of Object.keys(value)) {
          const path = `${section}.${leaf}`;
          const current = value[leaf];
          let control;
          if (typeof current === "number") {
            control = el("input", { type: "number", step: "any", value: String(current), title: schemaHint(path),
              oninput: (e) => {
                const n = Number(e.target.value);
                if (e.target.value !== "" && !Number.isNaN(n)) { value[leaf] = n; syncRaw(); }
              } });
          } else if (typeof current === "boolean") {
            control = el("select", { title: schemaHint(path), onchange: (e) => { value[leaf] = e.target.value === "true"; syncRaw(); } },
              el("option", { value: "true", selected: current ? "selected" : null }, "true"),
              el("option", { value: "false", selected: current ? null : "selected" }, "false"));
          } else if (current !== null && typeof current === "object") {
            control = el("input", { type: "text", class: "mono", value: JSON.stringify(current), title: schemaHint(path),
              oninput: (e) => {
                try { value[leaf] = JSON.parse(e.target.value); invalid.delete(path); e.target.style.borderColor = ""; syncRaw(); }
                catch (_) { invalid.add(path); e.target.style.borderColor = "var(--err)"; }
              } });
          } else {
            control = el("input", { type: "text", value: current == null ? "" : String(current), title: schemaHint(path),
              oninput: (e) => { value[leaf] = e.target.value; syncRaw(); } });
          }
          const documented = (state.flow3dSchema || {})[path];
          box.append(
            el("span", { class: "pkey", title: schemaHint(path) },
              keyLabel(leaf), documented ? el("span", { class: "muted small" }, " ⓘ") : null),
            control,
            el("button", { type: "button", class: "button small", title: `remove ${path}`,
              onclick: () => { delete value[leaf]; draw(); syncRaw(); } }, "×"));
        }
        box.append(el("span", { class: "muted small", style: "grid-column: 1 / -1" },
          documentedCount(section, value)));
      };
      draw();
      return box;
    }
    function documentedCount(section, value) {
      const schema = state.flow3dSchema || {};
      const known = Object.keys(value).filter((leaf) => schema[`${section}.${leaf}`]).length;
      if (!Object.keys(schema).length) return `${section}: hover a field name for its meaning (schema unavailable)`;
      return `${section}: ${known} of ${Object.keys(value).length} fields documented — hover for meaning and constraint`;
    }
    function controlFor(key) {
      const v = obj[key];
      if (PROFILE_ENUMS[key]) {
        return el("select", { onchange: (e) => { obj[key] = e.target.value; syncRaw(); } },
          ...PROFILE_ENUMS[key].map((o) => el("option", { value: o, selected: String(v) === o ? "selected" : null }, o)));
      }
      if (typeof v === "boolean") {
        return el("select", { onchange: (e) => { obj[key] = e.target.value === "true"; syncRaw(); } },
          el("option", { value: "true", selected: v ? "selected" : null }, "true"),
          el("option", { value: "false", selected: v ? null : "selected" }, "false"));
      }
      if (typeof v === "number") {
        return el("input", { type: "number", step: "any", value: String(v), oninput: (e) => {
          const n = Number(e.target.value);
          if (e.target.value !== "" && !Number.isNaN(n)) { obj[key] = n; syncRaw(); }
        } });
      }
      if (v !== null && typeof v === "object" && !Array.isArray(v)) {
        // One level of recursion. The FLOW-3D config's physics lives in nested
        // blocks, and rendering each as ONE json line meant the `laser` block
        // was a 191-character text field with no units, no ranges, and a typo
        // anywhere in it invalidating the whole block. Each leaf gets its own
        // labelled control, with the runner's own meaning/constraint as the
        // tooltip where the schema documents it.
        return nestedBlock(key, v);
      }
      if (Array.isArray(v)) {
        // Arrays stay JSON: they are short ([0,0,0]) and positional.
        return el("input", { type: "text", class: "mono", value: JSON.stringify(v), oninput: (e) => {
          try { obj[key] = JSON.parse(e.target.value); invalid.delete(key); e.target.style.borderColor = ""; syncRaw(); }
          catch (_) { invalid.add(key); e.target.style.borderColor = "var(--err)"; }
        } });
      }
      return el("input", { type: "text", value: v == null ? "" : String(v), oninput: (e) => { obj[key] = e.target.value; syncRaw(); } });
    }
    function redraw() {
      clear(grid);
      if (!Object.keys(obj).length) grid.append(el("span", { class: "muted small", style: "grid-column: 1 / -1" }, "No keys set — add one below or paste raw JSON."));
      for (const key of Object.keys(obj)) {
        grid.append(
          el("span", { class: "pkey", title: key }, keyLabel(key)),
          controlFor(key),
          el("button", { type: "button", class: "button small", title: `remove ${key}`, onclick: () => { delete obj[key]; redraw(); syncRaw(); } }, "×"));
      }
    }
    const addKey = el("input", { type: "text", placeholder: "key" });
    const addVal = el("input", { type: "text", placeholder: "value (JSON or text)" });
    const addBtn = el("button", { type: "button", class: "button small", onclick: () => {
      const k = addKey.value.trim();
      if (!k) return;
      let v = addVal.value;
      try { v = JSON.parse(addVal.value); } catch (_) { /* keep as string */ }
      obj[k] = v;
      addKey.value = ""; addVal.value = "";
      redraw(); syncRaw();
    } }, "Add key");
    rawTa.addEventListener("change", () => {
      try {
        const parsed = JSON.parse(rawTa.value || "{}");
        for (const k of Object.keys(obj)) delete obj[k];
        Object.assign(obj, parsed);
        rawErr.textContent = "";
        invalid.delete("(raw JSON)");
        redraw();
      } catch (e) { invalid.add("(raw JSON)"); rawErr.textContent = `invalid JSON: ${e.message}`; }
    });
    redraw(); syncRaw();
    const node = el("div", { class: "stack profile-editor" },
      grid,
      el("div", { class: "row" }, addKey, addVal, addBtn),
      el("details", {}, el("summary", { class: "muted small" }, "Edit as raw JSON"), rawTa, rawErr));
    return {
      node,
      get: () => {
        if (invalid.size) {
          throw new Error(`fix the invalid value(s) before submitting: ${[...invalid].join(", ")}`);
        }
        return JSON.parse(JSON.stringify(obj));
      },
    };
  }

  // ------------------------------------------------------------- shared UI
  function field(label, control) { return el("label", {}, el("span", {}, label), control); }
  function advanced(label, control) { return el("details", {}, el("summary", { class: "muted small" }, label), control); }
  function selectVals(vals, onchange) {
    const s = el("select", { onchange: (e) => onchange(e.target.value) }, ...vals.map((v) => el("option", { value: v }, v)));
    return s;
  }
  function selectVals2(pairs, onchange, current) {
    // Without `selected` the control paints its first option while the state
    // says something else, so the label contradicts what is on screen.
    return el("select", { onchange: (e) => onchange(e.target.value) },
      ...pairs.map(([v, l]) => el("option", { value: v, selected: current != null && v === current }, l)));
  }
  function markSel(box, tr) { [...box.querySelectorAll("tr.is-selected")].forEach((n) => n.classList.remove("is-selected")); tr.classList.add("is-selected"); }
  function emptyToNull(obj) { return obj && Object.keys(obj).length ? obj : null; }
  function evictButton(pk) {
    // Delete the arrays, keep the record. A dry run first tells the user how
    // much disk it frees and how many sims are affected before they commit.
    const btn = el("button", { class: "button small", title: "delete the array data, keep metadata and provenance" }, "Evict data…");
    btn.addEventListener("click", async () => {
      const dry = await postJson(`/datasets/${pk}/evict`, { dry_run: true }).catch((e) => ({ error: e.message }));
      if (dry.error) { alert(dry.error); return; }
      if (!dry.would_evict) { alert(dry.already_evicted ? "Already metadata-only." : "Nothing to evict."); return; }
      const mib = (dry.reclaimable_bytes / 1048576).toFixed(0);
      if (!confirm(`Evict array data for ${dry.would_evict} of ${dry.members} simulations in dataset ${pk}?\n\n`
        + `Frees ~${mib} MiB (reclaimed by \`aiida-pino reclaim\`). Metadata and provenance are kept; `
        + `the data itself is gone and this cannot be undone without re-importing from source.`)) return;
      const reason = prompt("Note to record on the evicted data (optional):", "") || null;
      try {
        const r = await postJson(`/datasets/${pk}/evict`, { dry_run: false, reason });
        alert(`Evicted ${r.evicted} simulation(s), ${(r.bytes_freed / 1048576).toFixed(0)} MiB.\n${r.reclaim_hint}`);
        selectTab(state.activeTab);
      } catch (e) { alert(e.message); }
    });
    return btn;
  }

  function deleteButton(pks, label) {
    const btn = el("button", { class: "button danger small" }, "Delete…");
    btn.addEventListener("click", async () => {
      const dry = await postJson("/nodes/delete", { pks, dry_run: true }).catch((e) => ({ error: e.message }));
      if (dry.error) { alert(dry.error); return; }
      if (!confirm(`Delete ${label}? This affects ${dry.affected ? dry.affected.length : "?"} node(s):\n${(dry.affected || []).join(", ")}`)) return;
      try { await postJson("/nodes/delete", { pks, dry_run: false }); alert("deleted"); selectTab(state.activeTab); }
      catch (e) { alert(e.message); }
    });
    return btn;
  }

  const RENDERERS = {
    environment: renderEnvironment,
    datasets: renderDatasets,
    models: renderModels,
    runs: renderRuns,
    monitor: renderMonitor,
    logs: renderLogs,
    compose: renderCompose,
  };

  // Optional tab modules register themselves here, before init(). The console is complete
  // without any.
  function registerTab({ id, label, render }) {
    TABS.push({ id, label });
    RENDERERS[id] = render;
  }
  window.PinoConsole = {
    registerTab,
    helpers: {
      state, $, el, clear, notice, table, getJson, postJson, codeSelect, datasetOptions,
      modelOptions, profileEditor, submitGate, titleFields, runSubmit, splitChips, daemonOk,
      processName, field, advanced, emptyToNull, selectVals2,
    },
  };

  // ---------------------------------------------------------------- boot
  async function refreshEnv() {
    try { state.env = await getJson("/env"); }
    catch (e) {
      // A blocked cross-origin fetch reaches JS as a bare "Failed to fetch"
      // with no mention of CORS or of which origin was refused, and the page
      // pins its backend in config.local.json — so say where it was pointed.
      if (/failed to fetch|networkerror/i.test(e.message || "")) {
        $("backendLine").textContent =
          `backend unreachable at ${state.baseUrl || "same origin"} — page origin ${location.origin}. `
          + "If they differ, browse the URL the backend was started with (web/dev/run_local.sh prints it), "
          + "or restart it with that origin.";
      } else {
        // Only when the message is NOT the origin case: this line used to run
        // unconditionally and overwrite the diagnostic above one statement after
        // writing it, so every cross-origin failure showed the bare
        // "backend error: Failed to fetch" the diagnostic exists to replace.
        $("backendLine").textContent = `backend error: ${e.message}`;
      }
      // /env walks the whole environment (codes, configs, versions) and is the
      // first casualty of a busy backend. /healthz answers the one question the
      // submit gate actually asks, in ~1.7 s where /env took 33 s, so fall back
      // to it rather than declaring the environment unknown.
      try {
        const health = await getJson("/healthz");
        state.env = health && health.daemon ? { daemon: health.daemon, degraded: true } : null;
        if (state.env) $("backendLine").textContent = `backend degraded: /env failed, using /healthz — ${e.message}`;
      } catch (_) { state.env = null; }
    }
    renderEnvStrip();
  }
  async function init() {
    state.cfg = await loadConfig();
    state.baseUrl = state.cfg.backendBaseUrl || "";
    $("backendLine").textContent = state.baseUrl ? `backend ${state.baseUrl}` : "backend: same origin";
    renderTabbar();
    await refreshEnv();
    // Honor a deep-link hash on load; otherwise land on Monitor (the default).
    const hashed = location.hash.slice(1);
    selectTab(RENDERERS[hashed] ? hashed : state.activeTab);
  }
  window.addEventListener("DOMContentLoaded", init);
})();
