"""The dataset listing's row cap must never drop a CURATED set.

The cap exists because an uncapped listing of 2529 rows measured 14.3 s, so removing it is
not the alternative. But it protected only `core` rows, and after the 2026-09-01 curation
358 of 2529 rows sit in a filed folder outside the computed `Single-run imports` bucket --
so the cap could silently drop a curated set and the browser would show a folder missing
members it demonstrably has. A listing that omits rows without saying so is the same
failure class as a query whose scope is invisible in its answer.
"""

from __future__ import annotations

import inspect
import unittest

from aiida_pino.console.folders import ORPHANED


def _truncate(rows, limit):
    """The service's own cap. A pure helper precisely so this needs no source surgery."""
    from aiida_pino.console.service import cap_dataset_rows

    return cap_dataset_rows(list(rows), limit)


def _row(pk, *, folder=None, keywords=(), ctime="2026-01-01"):
    return {"pk": pk, "folder": folder, "keywords": list(keywords), "ctime": ctime}


class TruncationTest(unittest.TestCase):
    def test_a_curated_row_survives_a_cap_that_would_drop_it(self):
        # One filed row buried under many orphans, with a cap far below the total.
        rows = [_row(i, folder=ORPHANED) for i in range(50)]
        rows.append(_row(999, folder="Experimental validation"))
        kept = {r["pk"] for r in _truncate(rows, limit=5)}
        self.assertIn(999, kept, "a filed dataset was dropped by the cap")

    def test_core_is_still_protected(self):
        rows = [_row(i, folder=ORPHANED) for i in range(50)]
        rows.append(_row(777, keywords=["core"]))
        self.assertIn(777, {r["pk"] for r in _truncate(rows, limit=5)})

    def test_the_orphan_bucket_is_what_the_cap_trims(self):
        # Otherwise "protect everything" would just remove the cap, and the 14.3 s
        # uncapped listing is why the cap exists.
        rows = [_row(i, folder=ORPHANED) for i in range(50)]
        kept = _truncate(rows, limit=5)
        self.assertLessEqual(len(kept), 5)

    def test_an_unfiled_row_is_not_treated_as_curated(self):
        # `folder` absent means nobody filed it; only a real folder counts.
        rows = [_row(i, folder=ORPHANED) for i in range(50)]
        rows.append(_row(888, folder=None))
        rows.append(_row(889, folder=""))
        kept = {r["pk"] for r in _truncate(rows, limit=3)}
        self.assertNotIn(888, kept)
        self.assertNotIn(889, kept)


class AutoSplitFolderTest(unittest.TestCase):
    def test_the_only_automatic_writer_files_one_flat_folder(self):
        from aiida_pino.console import service as svc_mod

        src = inspect.getsource(svc_mod.AiidaConsoleService._split_subset_collection)
        self.assertIn('self.set_folder([node.pk], "Training splits")', src)
        # A per-base-dataset path is what minted 18 subfolders holding 108 sets.
        self.assertNotIn("Training splits (auto)/", src)

    def test_the_browser_hides_the_flattened_folder_too(self):
        from pathlib import Path

        import aiida_pino

        app_js = (Path(aiida_pino.__file__).parents[2] / "web" / "public" / "app.js").read_text()
        self.assertIn('=== "Training splits"', app_js,
                      "isAutoSplit does not recognise the flattened folder")


if __name__ == "__main__":
    unittest.main()
