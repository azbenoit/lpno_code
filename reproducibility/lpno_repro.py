"""Helpers shared by the reproducibility notebooks.

The open record (doi:10.5281/zenodo.23325086) holds three AiiDA archives: the main one, the per-round retrains of
the acquisition study and the evaluations added on 2026-10-29. `Store` opens each one read-only (an archive is a
valid AiiDA storage backend, so nothing is imported into a profile) and looks nodes up by UUID in each in turn.

Metric conventions, the same for every table of the manuscript (each evaluation stores its metrics in the
`evaluation_metadata` attribute):

    T MAE          metric_units.kelvin["T.substrate_abs_err_mean"]       substrate cells, K
    T RMSE         metric_units.kelvin["T.substrate_rmse"]               substrate cells, K
    BB MAE         metrics.T.beambox_abs_err_mean x temperature_scale_k  beam box, K
    Rel. err.      100 x metrics.T.substrate_rel_err_mean                substrate cells, %
    Melt-pool IoU  metrics.fl.iou_mean                                   liquid fraction > 0.5
    Peak ratio     metrics.T.pred_T_max_K / metrics.T.true_T_max_K

Rows over three training seeds print the arithmetic mean and the sample standard deviation (n - 1), each rounded once.
"""
import math
import statistics
from pathlib import Path

ARCHIVES = ("lpno_provenance.aiida", "lpno_provenance_rounds.aiida", "lpno_provenance_evals.aiida")


class Store:
    """The deposited archives, opened read-only.

    `fetch(uuids)` returns {uuid: {"archive", "label", "type", "attributes", "extras"}} for every node found, searching
    the archives in order (a node present in several, as a copy inside an addition, is read from the first).
    """

    def __init__(self, deposit_dir, names=ARCHIVES):
        from aiida.storage.sqlite_zip.backend import SqliteZipBackend
        self.dir = Path(deposit_dir)
        self.profiles = []
        for name in names:
            path = self.dir / name
            if path.exists():
                self.profiles.append((name, SqliteZipBackend.create_profile(str(path))))
            else:
                print(f"[store] {name} not found in {self.dir}: skipped")
        self._cache = {}

    def _load(self, profile):
        from aiida import load_profile
        load_profile(profile, allow_switch=True)

    def fetch(self, uuids):
        from aiida import orm
        want = [u for u in dict.fromkeys(uuids) if u not in self._cache]
        for name, profile in self.profiles:
            if not want:
                break
            self._load(profile)
            qb = orm.QueryBuilder().append(orm.Node, filters={"uuid": {"in": want}},
                                           project=["uuid", "label", "node_type", "attributes", "extras"])
            for uuid, label, ntype, attrs, extras in qb.iterall():
                self._cache[uuid] = {"archive": name, "label": label, "type": ntype, "attributes": attrs,
                                     "extras": extras}
            want = [u for u in want if u not in self._cache]
        return {u: self._cache[u] for u in uuids if u in self._cache}

    def query(self, archive, fn):
        """Run fn(orm) with the named archive loaded; for walks that are easier as queries than as UUID lists."""
        from aiida import orm
        self._load(dict(self.profiles)[archive])
        return fn(orm)


def metrics(attributes):
    """The tabled metrics of one evaluation node, in the manuscript's units."""
    md = attributes["evaluation_metadata"]
    m, units = md["metrics"], md["metric_units"]
    T, kelvin, scale = m["T"], units["kelvin"], float(units["temperature_scale_k"])
    return {"T MAE": kelvin["T.substrate_abs_err_mean"],
            "T RMSE": kelvin["T.substrate_rmse"],
            "BB MAE": T["beambox_abs_err_mean"] * scale,
            "Rel. err.": 100.0 * T["substrate_rel_err_mean"],
            "Melt-pool IoU": (m.get("fl") or {}).get("iou_mean", math.nan),
            "Peak ratio": T["pred_T_max_K"] / T["true_T_max_K"]}


def decimals(printed):
    """Decimal places of a printed value ("7.73 ± 0.05" -> 2, "456" -> 0)."""
    head = printed.split("±")[0].strip()
    return len(head.split(".")[1]) if "." in head else 0


def aggregate(values, how, places):
    """Format a cell the way the manuscript prints it."""
    if how == "single":
        assert len(values) == 1, values
        return f"{values[0]:.{places}f}"
    mean = statistics.fmean(values)
    if how == "mean":
        return f"{mean:.{places}f}"
    return f"{mean:.{places}f} ± {statistics.stdev(values):.{places}f}"


def gnu_time_seconds(text):
    """Wall clock from GNU time's verbose output ("Elapsed (wall clock) time (h:mm:ss or m:ss): 31:49.74")."""
    line = next(l for l in text.splitlines() if "Elapsed (wall clock)" in l)
    parts = [float(p) for p in line.rsplit(" ", 1)[1].split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
