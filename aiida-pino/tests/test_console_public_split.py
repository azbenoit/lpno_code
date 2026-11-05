"""The console's submit tabs live in two optional files, so the public export is the private
console minus those files -- no patching, no divergent fork.

Since 2026-09-14 the exported open-source monitor drops the Train/Eval, K-Fold,
FLOW-3D and Import tabs. Each of those is a submission surface for the original deployment's
solver seat, GPU spool and compute machines; a public reader has none of them, and shipping the
UI for them is shipping dead ends. The manuscript's provenance-store reader needs Monitor,
Environment, Datasets, Compose, Models, Runs and Logs.

Mechanism: `web/public/app.js` is the core (tabbar, helpers, the seven kept tabs) and exposes
`window.PinoConsole.registerTab`; `web/public/app.submit.js` registers the four submit tabs
through it. `web/app/main.py` registers the read/compose routes and calls
`submit_routes.register(app, _service)` only if that module imports. Deleting the two files
is the whole thinning, and both halves are pinned here so nothing quietly moves back.
"""

from __future__ import annotations

import builtins
import importlib
import pathlib
import re


WEB = pathlib.Path(__file__).resolve().parents[1] / "web"
CORE = WEB / "public" / "app.js"
REMOVED_TABS = ("train", "kfold", "import", "flow3d")
KEPT_TABS = ("monitor", "environment", "datasets", "compose", "models", "runs", "logs")


def test_the_core_script_has_no_submit_tab_left_in_it():
    core = CORE.read_text()
    for name in ("renderTrain", "renderKfold", "renderImport", "renderFlow3d", "evalSection"):
        assert name not in core, f"{name} must live in app.submit.js"
    tabs = re.search(r"const TABS = \[(.*?)\];", core, re.S).group(1)
    for tab in REMOVED_TABS:
        assert f'id: "{tab}"' not in tabs
    for tab in KEPT_TABS:
        assert f'id: "{tab}"' in tabs
    assert "window.PinoConsole" in core and "registerTab" in core


def test_render_check_takes_its_tab_list_from_the_page():
    src = (WEB / "dev" / "render_check.py").read_text()
    assert not re.search(r'^TABS\s*=\s*\[', src, re.M), "a hardcoded tab list drifts from the export"


def test_the_app_builds_without_the_submit_routes_module(monkeypatch):
    """Delete `submit_routes.py` and the console still serves; only the submit surface is gone."""
    from web.app import main as main_mod

    real_import = builtins.__import__

    def no_submit(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("submit_routes") or (level and fromlist and "submit_routes" in str(fromlist)):
            raise ImportError("submit_routes removed for the public export")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", no_submit)
    importlib.reload(main_mod)
    app = main_mod.create_app(settings=main_mod.Settings(startup_load_service=False), service=object())
    paths = {r.path for r in app.routes}
    assert "/submit/derive_dataset" in paths, "Compose stays"
    assert "/datasets" in paths and "/monitor" in paths and "/flow3d/queue" in paths
    for gone in ("/submit/train_eval", "/submit/eval_only", "/submit/kfold", "/submit/flow3d_generate",
                 "/submit/raw_flow3d_import", "/submit/import_model", "/kfold_runs", "/training_presets",
                 "/profile_keys/{which}", "/processes/{pk}/training_inputs"):
        assert gone not in paths, gone
    monkeypatch.setattr(builtins, "__import__", real_import)
    importlib.reload(main_mod)


def test_render_check_writes_nothing_unless_told_to():
    """2026-09-14: pointed at the live console, the checker filed all 73 datasets of a real folder
    into `zz_render_check` and found no Undo button. Every DB-writing flow is behind one flag."""
    src = (WEB / "dev" / "render_check.py").read_text()
    assert "--exercise-writes" in src
    for marker in ("Multi-select filing", "Star toggle on the models list"):
        i = src.index(marker)
        assert "if args.exercise_writes" in src[max(0, i - 900):i], marker
    # The compose keyword add+remove is two writes to a real simulation's extras; it was
    # left ungated by the first pass (found by the docs re-check, 2026-09-14). Every
    # `.fill(` on the page must sit after an `exercise_writes` gate in its block.
    i = src.index('chip_input.fill("render-check")')
    assert "args.exercise_writes and chip_input.count()" in src[max(0, i - 400):i]
    # The private submission tabs are visited only when the page serves them.
    i = src.index('index.html#train')
    assert '"train" in TABS' in src[max(0, i - 300):i]
