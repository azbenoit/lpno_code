"""Parser for PINO k-fold jobs."""

from __future__ import annotations

from aiida import orm

from .common import FolderResultParser


class PinoKfoldParser(FolderResultParser):
    """Parse PINO k-fold artifacts and aggregated metrics."""

    folder_label = "kfold_artifacts"
    folder_subpath = "kfold_output"
    summary_label = "kfold_summary"
    summary_filename = "kfold_summary.json"

    def parse(self, **kwargs):
        exit_code = super().parse(**kwargs)
        if exit_code is not None:
            return exit_code
        metrics_payload = self.read_metrics_with_stdout_fallback()
        if not metrics_payload:
            return self.exit_codes.ERROR_MISSING_METRICS
        metrics = orm.Dict(dict=metrics_payload)
        metrics.label = "metrics"
        self.out("metrics", metrics)
