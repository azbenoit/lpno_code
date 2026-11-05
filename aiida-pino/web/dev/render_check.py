"""Headless render check for the web console.

Loads every tab, drives the main interactions (row selection, case browser,
clamp expansion, chips), and fails on JS errors or page-level horizontal
overflow. Screenshots land in --out for visual review.

Usage (needs `pip install playwright` + `playwright install chromium`):
    python web/dev/render_check.py [--url http://127.0.0.1:8180] [--out web/dev/render_out]

Run it against a live stack (web/dev/run_local.sh start) after any change that
can affect the GUI. Read-only by default; `--exercise-writes` adds the folder-filing
flows and must only ever point at a disposable profile. Read the screenshots afterwards — the script catches
errors and overflow, not ugliness.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from playwright.sync_api import sync_playwright


def tabs_on_page(page) -> list[str]:
    """The tabs THIS build serves, read off the tabbar rather than listed here.

    Optional tab modules can add tabs. A hardcoded list drifted from whichever build was under test and
    either skipped tabs or navigated to ones that do not exist. `environment` is checked
    first because the rest of the script relies on it having loaded the env strip.
    """
    ids = page.eval_on_selector_all("#tabbar button[data-tab]", "els => els.map(e => e.dataset.tab)")
    return ["environment"] + [t for t in ids if t != "environment"]



def main() -> int:
    ap = argparse.ArgumentParser()
    default_url = "http://127.0.0.1:8180"
    cfg = pathlib.Path(__file__).resolve().parents[1] / "public" / "config.local.json"
    if cfg.exists():
        try:
            backend = json.loads(cfg.read_text()).get("backendBaseUrl", "")
            if backend:
                default_url = backend.replace(":8100", ":8180")
        except Exception:  # noqa: BLE001
            pass
    ap.add_argument("--url", default=default_url)
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "render_out"))
    ap.add_argument("--exercise-writes", action="store_true",
                    help="also drive the folder-filing and undo flows, which WRITE to the database "
                         "behind the console; only against a disposable profile")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    overflow: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 950})
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on(
            "console",
            lambda m: errors.append(f"console.{m.type}: {m.text}")
            if m.type == "error" and "favicon" not in m.text
            else None,
        )
        # Never hang on prompt()/confirm()/alert(); dismiss = cancel, so no mutation.
        page.on("dialog", lambda d: d.dismiss())

        def settle(ms: int = 400) -> None:
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(ms)

        def wait_list(timeout: int = 45000) -> bool:
            """Wait for the *list* pane, not just any table row.

            `main table tbody tr` also matches the detail pane, which keeps its
            rows from the previous selection — so it returns immediately while
            the list is still fetching, and every assertion after it reads an
            empty table."""
            try:
                page.wait_for_selector("main .panel.stack table tbody tr", timeout=timeout)
                return True
            except Exception:  # noqa: BLE001
                return False

        def wait_rows(selector: str = "main table tbody tr", timeout: int = 15000) -> bool:
            try:
                page.wait_for_selector(selector, timeout=timeout)
                return True
            except Exception:  # noqa: BLE001
                return False

        def check_overflow(name: str) -> None:
            over = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            if over > 2:
                overflow.append(f"{name}: page overflows horizontally by {over}px")

        def shot(name: str) -> None:
            check_overflow(name)
            page.screenshot(path=str(out / f"{name}.png"), full_page=False)

        page.goto(f"{args.url}/index.html#environment")
        page.wait_for_selector("#tabbar button[data-tab]")
        TABS = tabs_on_page(page)
        settle(800)
        shot("01-environment")

        # Warm the heavy dataset listing once (cold profile + 200+ rows can take
        # ~20s on the first hit); otherwise the first assertion races the load.
        page.goto(f"{args.url}/index.html#datasets")
        has_data = wait_list()
        settle(300)

        # Every assertion below this point is about how REAL rows render, so on a
        # profile with no datasets the check used to report 30-odd ERRORs that all
        # meant "you have no data yet" -- unusable for a contributor on their own
        # box. Screenshots of the empty tabs are still worth having, so take those
        # and stop before the row assertions.
        if not has_data:
            for i, tab in enumerate(TABS[1:], start=2):
                page.goto(f"{args.url}/index.html#{tab}")
                settle(400)
                shot(f"{i:02d}-{tab}")
            browser.close()
            print(f"screenshots: {out}")
            for e in errors:
                print("ERROR:", e)
            if errors:
                print("RENDER CHECK: FAIL")
                return 1
            print("RENDER CHECK: SKIP - this profile has no datasets, so only the "
                  "empty-state rendering was checked")
            return 0

        for i, tab in enumerate(TABS[1:], start=2):
            page.goto(f"{args.url}/index.html#{tab}")
            settle(600)
            shot(f"{i:02d}-{tab}")

        # Datasets: row -> detail -> case row -> case detail; expand a clamp.
        page.goto(f"{args.url}/index.html#datasets")
        settle(400)
        wait_rows()
        rows = page.locator("main table tbody tr:not(.group-row)")
        if rows.count():
            rows.first.click()
            settle(600)
            wait_rows(".detail-pane table tbody tr")
            shot("20-dataset-detail")
            clamp = page.locator(".clamp").first
            if clamp.count():
                clamp.click()
                page.wait_for_timeout(200)
                shot("21-clamp-expanded")
            case_rows = page.locator(".detail-pane table tbody tr")
            if case_rows.count():
                case_rows.first.click()
                settle(500)
                # viewer PNG frame + stats panel load async
                try:
                    page.wait_for_selector(".case-viewer img.viz-frame", timeout=30000)
                    page.wait_for_timeout(1200)
                except Exception:  # noqa: BLE001
                    errors.append("case viewer image did not load")
                shot("22-case-detail")
                # step one frame via the slider, switch view, screenshot again
                slider = page.locator(".case-viewer input[type=range]")
                if slider.count():
                    slider.first.fill("0")
                    page.wait_for_timeout(1200)
                    shot("23-case-viewer-frame0")
            # dataset stats section (plot img or compute button)
            stats_head = page.get_by_text("Melt-pool statistics")
            if stats_head.count():
                stats_head.first.scroll_into_view_if_needed()
                page.wait_for_timeout(1000)
                shot("24-dataset-stats")

        # Folders: every dataset sits in a folder header or at the top level,
        # and the auto "Orphaned" folder of single-simulation import artifacts
        # is present and closed until clicked.
        page.goto(f"{args.url}/index.html#datasets")
        settle(500)
        wait_rows()
        wait_list()
        # The list arrives before its folder headers are drawn (the grouped render
        # follows the fetch); waiting on the first row alone read an ungrouped
        # table and reported "no folder header rows" against a page that had seven.
        try:
            page.wait_for_selector("tr.group-row", timeout=60000)
        except Exception:  # noqa: BLE001 - the assertion below reports it
            pass
        headers = page.locator("tr.group-row")
        if not headers.count():
            errors.append("no folder header rows on the datasets table")
        # The auto folder of single-simulation import artifacts is the group row
        # carrying the `auto` badge, whatever it is named (it was "Orphaned", it is
        # "Single-run imports" now; `ORPHANED_FOLDER` in app.js is the one place).
        orphaned = page.locator("tr.group-row", has=page.locator(".chip", has_text="auto"))
        if not orphaned.count():
            errors.append("no auto folder for single-simulation collections")
        else:
            # Runs before the expand-all block below, which opens every folder.
            if orphaned.first.locator(".ds-toggle", has_text="▾").count():
                errors.append("the Single-run imports folder is expanded by default")
            shot("29-datasets-folders")
            orphaned.first.click()
            page.wait_for_timeout(500)
            if not page.locator("tr.subset-row").count():
                errors.append("expanding the Orphaned folder showed no member rows")
            shot("30-datasets-orphaned-open")

        # Lineage tree: expand a base row's derived sets, then Role=all must
        # also reveal quick-val subsets nested one level deeper.
        page.goto(f"{args.url}/index.html#datasets")
        settle(500)
        wait_rows()
        wait_list()
        # Folder headers and lineage parents share the ▸ toggle, and which comes
        # first depends on how the datasets happen to be filed — so expand
        # everything and assert on the nesting that results.
        def expand_all(limit=30):
            for _ in range(limit):  # handles go stale on each re-render
                collapsed = page.locator(".ds-toggle", has_text="▸")
                if not collapsed.count():
                    return True
                collapsed.first.click()
                page.wait_for_timeout(250)
            return False

        if not page.locator(".ds-toggle").count():
            errors.append("no expander on the datasets table")
        else:
            expand_all()
            if not page.locator("tr.subset-row").count():
                errors.append("expanding the datasets table showed no nested rows")
            shot("25-datasets-expanded")
        page.locator("main .controls-grid select").first.select_option("all")
        page.wait_for_timeout(500)
        expand_all()
        if not page.locator("tr.subset-row.depth-2").count():
            errors.append("Role=all + expand-all showed no depth-2 subset rows")
        shot("26-datasets-role-all")

        # Keyword grouping: picking a keyword folds its datasets into one
        # collapsible header row; clicking the header expands them back.
        page.goto(f"{args.url}/index.html#datasets")
        settle(500)
        wait_rows()
        group_sel = page.locator("main .controls-grid select").last
        if len(group_sel.locator("option").all()) < 2:
            errors.append("no shared keyword offered in the group-by select")
        else:
            group_sel.select_option(index=1)
            page.wait_for_timeout(500)
            header = page.locator("tr.group-row", has_text="keyword:")
            if not header.count():
                errors.append("group-by keyword produced no group header row")
            else:
                shot("27-datasets-grouped")
                header.first.click()
                page.wait_for_timeout(400)
                if not page.locator("tr.subset-row").count():
                    errors.append("expanding a keyword group showed no member rows")
                shot("28-datasets-group-expanded")

        # Lineage: a derived dataset nests under its base *inside* the base's
        # folder, so opening a folder then a base row reaches depth 2.
        page.goto(f"{args.url}/index.html#datasets")
        settle(500)
        wait_list()
        # Any folder whose members carry a `▸` toggle holds a derived dataset; the
        # folder that does is a property of the profile, not a fixed name ("Training
        # splits" was a top-level folder once and is nested now).
        try:
            page.wait_for_selector("tr.group-row", timeout=60000)
        except Exception:  # noqa: BLE001
            pass
        splits = page.locator("tr.group-row").filter(has_not=page.locator(".chip", has_text="auto"))
        member_toggle = page.locator("tr.subset-row .ds-toggle")
        opened = 0
        for i in range(splits.count()):
            splits.nth(i).click()
            page.wait_for_timeout(700)
            opened += 1
            if member_toggle.count():
                break
        if not opened:
            errors.append("no folder holding derived datasets to check lineage in")
        else:
            if not member_toggle.count():
                errors.append("no derived dataset nested inside its folder")
            else:
                member_toggle.first.click()
                page.wait_for_timeout(600)
                if not page.locator("tr.subset-row.depth-2").count():
                    errors.append("a base inside a folder did not expand to its derived set")
                shot("34-lineage-in-folder")

        # WRITE-EXERCISING checks: filing datasets and models into folders, then undoing.
        # Opt-in since 2026-09-14. Run against the LIVE console, the multi-select test filed all
        # 73 datasets of a real folder into `zz_render_check` and the undo path found no Undo
        # button (the control had moved), so the moves persisted until restored by hand from a
        # before/after record that happened to exist. A render check must not be
        # able to rearrange the database it is pointed at unless the operator says so.
        if args.exercise_writes:
            # Multi-select filing: tick rows, type a folder name that does not exist
            # (typing it is how one is created), move, then undo — so the database
            # is left exactly as found, like the star toggle below.
            page.goto(f"{args.url}/index.html#datasets")
            settle(500)
            wait_rows()
            wait_list()
            picks = page.locator("tr:not(.group-row) input.pick")
            if picks.count() < 1:
                errors.append("no selection checkboxes on the datasets table")
            else:
                picks.first.click()
                page.wait_for_timeout(200)
                if page.locator(".bulk-bar").is_hidden():
                    errors.append("selecting a dataset did not show the bulk bar")
                if not page.locator(".bulk-count").inner_text().startswith("1 selected"):
                    errors.append(f"bulk bar miscounted: {page.locator('.bulk-count').inner_text()!r}")
                header_box = page.locator("tr.group-row input.pick").first
                if header_box.count():
                    header_box.click()
                    page.wait_for_timeout(300)
                    if page.locator(".bulk-count").inner_text().startswith("1 selected"):
                        errors.append("ticking a folder header did not select its members")
                # a computed folder name must be refused before any request
                page.locator(".folder-input").first.fill("Orphaned")
                page.locator(".bulk-bar button", has_text="Move").click()
                page.wait_for_timeout(300)
                if "computed folder" not in page.locator(".bulk-msg").inner_text():
                    errors.append("moving into the computed Orphaned folder was not refused")
                page.locator(".folder-input").first.fill("zz_render_check")
                page.locator(".bulk-bar button", has_text="Move").click()
                page.wait_for_timeout(2500)
                if not page.locator("tr.group-row", has_text="zz_render_check").count():
                    errors.append("filing into a new folder did not create its header row")
                shot("31-datasets-filed")
                undo = page.locator(".bulk-msg button", has_text="Undo")
                if not undo.count():
                    errors.append("no Undo offered after a move")
                else:
                    undo.click()
                    page.wait_for_timeout(2500)
                    if page.locator("tr.group-row", has_text="zz_render_check").count():
                        errors.append("Undo did not put the datasets back")

            # The models tab carries the same folder machinery.
            page.goto(f"{args.url}/index.html#models")
            settle(700)
            wait_rows()
            wait_list()
            if not page.locator("main table tbody tr input.pick").count():
                errors.append("no selection checkboxes on the models table")
            else:
                page.locator("tr:not(.group-row) input.pick").first.click()
                page.wait_for_timeout(200)
                page.locator(".folder-input").first.fill("zz_render_models")
                page.locator(".bulk-bar button", has_text="Move").click()
                page.wait_for_timeout(2000)
                if not page.locator("tr.group-row", has_text="zz_render_models").count():
                    errors.append("filing a model into a folder did not create its header row")
                shot("32-models-filed")
                undo = page.locator(".bulk-msg button", has_text="Undo")
                if undo.count():
                    undo.click()
                    page.wait_for_timeout(2000)
                    if page.locator("tr.group-row", has_text="zz_render_models").count():
                        errors.append("Undo did not put the model back")

        # Model detail: physical units, badges, curves, comparison images.
        page.goto(f"{args.url}/index.html#models")
        settle(700)
        wait_list()
        # The list is PAGED (60 at a time), so an older model is not on screen until it is
        # searched for. Searching is also the behaviour worth pinning: filtering runs
        # client-side, so a paged list that filtered only its loaded rows would answer "no
        # results" for a model that exists. Typing here must reach the whole profile.
        # Two things at once, so pick the target carefully. The search must reach the WHOLE
        # profile (the list is paged 60 at a time, and filtering runs client-side, so a
        # paged list filtering only its loaded rows would answer "no results" for a model
        # that exists). And the model opened must CARRY a best-metric badge, or the chip
        # assertion below cannot pass: `alsi_rigno_r04_DIAG_correctHP` has an evaluation but
        # no badges, so pointing the search at it turned a reachable check into a failing
        # one. A badge is awarded dynamically, so this target is a current badge-holder
        # rather than a permanent one -- if it ever loses the badge, re-point it at another
        # from `auto_keywords` rather than weakening the assertion.
        target = "fno_ck300_lab_32k_cond12"
        search = page.locator("main .panel.stack input[type=search]").first
        if search.count():
            search.fill(target)
            page.wait_for_timeout(12000)
        row = page.locator("main .panel.stack tr:not(.group-row)", has_text=target)
        if not row.count():
            errors.append("searching for a model did not find it (paged list)")
        else:
            row.first.click()
            page.wait_for_timeout(6000)
            heads = page.locator(".detail-pane h3").all_inner_texts()
            for wanted in ("Trained on", "How training went", "Melt pool & keyhole", "Against ground truth"):
                if not any(wanted in h for h in heads):
                    errors.append(f"model detail is missing the {wanted!r} section")
            vals = page.locator(".scorecard .val").all_inner_texts()
            if not any(v.endswith(" K") for v in vals):
                errors.append("model scorecard shows no Kelvin temperature error")
            if not any(v.endswith(" %") for v in vals):
                errors.append("model scorecard shows no IoU percentage error")
            if not page.locator(".chip.auto").count():
                errors.append("no automatic best-metric chip on an evaluated model")
            if page.locator(".detail-pane img.viz-frame").count() < 2:
                errors.append("model detail did not render both the curve and the comparison image")
            if not page.locator(".detail-pane details summary", has_text="metrics").count():
                errors.append("the full metric list is not behind a dropdown")
            shot("33-model-detail")

        if args.exercise_writes:  # the star is an annotation write too
            # Star toggle on the models list: star, confirm, unstar (net zero).
            page.goto(f"{args.url}/index.html#models")
            settle(700)
            wait_rows()
            star = page.locator("main table tbody tr .star-toggle").first
            if not star.count():
                errors.append("no star toggle on the models table")
            else:
                was_on = "is-on" in (star.get_attribute("class") or "")
                star.click()
                page.wait_for_timeout(900)
                star = page.locator("main table tbody tr .star-toggle").first
                now_on = "is-on" in (star.get_attribute("class") or "")
                if now_on == was_on:
                    errors.append("clicking the star did not toggle it")
                shot("35-models-starred")
                star.click()  # restore
                page.wait_for_timeout(900)

        # Models + Runs detail panes.
        for name, tab in (("30-model-detail", "models"), ("31-run-detail", "runs")):
            page.goto(f"{args.url}/index.html#{tab}")
            settle(700)
            wait_rows()
            r = page.locator("main table tbody tr")
            if r.count():
                r.first.click()
                settle(700)
                shot(name)
        # Run detail must offer the title editor (short names for the runs list).
        if not page.locator("button:has-text('✎ title')").count():
            errors.append("run detail has no '✎ title' edit button")

        # Monitor: gauges must render (GPU/CPU meters) and refresh without errors.
        page.goto(f"{args.url}/index.html#monitor")
        settle(800)
        try:
            page.wait_for_selector(".meter", timeout=15000)
        except Exception:  # noqa: BLE001
            errors.append("monitor tab rendered no meters")
        if not page.get_by_text("FLOW-3D queue").count():
            errors.append("monitor tab missing the FLOW-3D queue section")
        shot("36-monitor")

        # Compose: refresh happens on open. Adding + removing a keyword chip on the
        # first sim row is two WRITES to a real simulation's extras ("net zero" only
        # if the removal succeeds), so it runs only with --exercise-writes.
        page.goto(f"{args.url}/index.html#compose")
        settle(400)
        wait_rows()
        shot("40-compose")
        chip_input = page.locator("main table tbody tr").first.locator("input.chip-add")
        if args.exercise_writes and chip_input.count():
            chip_input.fill("render-check")
            chip_input.press("Enter")
            settle(500)
            shot("41-compose-chip-added")
            removed = False
            for chip in page.locator("main table tbody tr").first.locator(".chip").all():
                if "render-check" in (chip.inner_text() or ""):
                    chip.locator("button").click()
                    removed = True
                    break
            settle(500)
            if not removed:
                errors.append("compose: added keyword chip not found for removal")
            shot("42-compose-chip-removed")

        # Train form with the typed profile editors opened -- only where the page
        # serves that tab (the public export ships no submission tabs).
        if "train" in TABS:
            page.goto(f"{args.url}/index.html#train")
            settle(800)
            for summ in page.locator("main details > summary").all()[:2]:
                summ.click()
            page.wait_for_timeout(300)
            # The form fills in from six async fetches; wait for it before shooting.
            try:
                page.wait_for_selector("main button:has-text('Submit training')", timeout=20000)
            except Exception:  # noqa: BLE001
                errors.append("train form never rendered its submit button")
            shot("50-train-profile-editors")

            # Training data comes from the dataset's own split labels: no val-dataset
            # override may exist, and an unlabelled dataset must block submission.
            labels = [t.strip().lower() for t in page.locator("main label > span").all_inner_texts()]
            if any("val dataset" in t for t in labels):
                errors.append("train form still offers a val-dataset override")
            submit = page.locator("main button:has-text('Submit training')")
            if submit.count() and page.locator("main .notice.warn").count():
                if submit.is_enabled():
                    errors.append("submit enabled although the dataset has no train/val labels")

        # Narrow viewports on the busiest tab.
        for width, name in ((1280, "60-datasets-1280"), (900, "61-datasets-900")):
            page.set_viewport_size({"width": width, "height": 950})
            page.goto(f"{args.url}/index.html#datasets")
            settle(700)
            wait_rows()
            rows2 = page.locator("main table tbody tr")
            if rows2.count():
                rows2.first.click()
                settle(700)
            shot(name)

        browser.close()

    print(f"screenshots: {out}")
    for e in errors:
        print("ERROR:", e)
    for o in overflow:
        print("OVERFLOW:", o)
    if errors or overflow:
        print("RENDER CHECK: FAIL")
        return 1
    print("RENDER CHECK: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
