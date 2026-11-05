"""Parser for PINO evaluation jobs."""

from __future__ import annotations

from aiida import orm

from .common import FolderResultParser


class PinoEvalParser(FolderResultParser):
    """Parse PINO evaluation outputs and metrics."""

    folder_label = "eval_artifacts"
    folder_subpath = "eval_output"
    summary_label = "eval_summary"
    summary_filename = "eval_summary.json"

    def parse(self, **kwargs):
        exit_code = super().parse(**kwargs)
        if exit_code is not None:
            return exit_code
        metrics_payload = self.read_metrics_with_stdout_fallback()
        if not metrics_payload:
            # Say WHY, on the node. 42 evals sit at exit 500 with nothing but
            # remote_folder/retrieved, and the cause (a Hydra crash, an OOM, a
            # bad checkpoint path) lived only in aiida.out -- the same
            # discard-the-reason shape as round 6's F7. The stdout tail is the
            # best available diagnosis, so store it where a listing can show it.
            self._record_failure_tail()
            return self.exit_codes.ERROR_MISSING_METRICS
        metrics = orm.Dict(dict=metrics_payload)
        metrics.label = "metrics"
        self.out("metrics", metrics)

    def _record_failure_tail(self) -> None:
        """The last meaningful stdout lines, as a `failure_tail` extra."""
        import re

        try:
            names = [n for n in self.retrieved.base.repository.list_object_names() if n.endswith(".out")]
            if not names:
                return
            raw = self.retrieved.base.repository.get_object_content(names[0])
            # Strip ANSI first: wandb writes `\x1b[1;34mwandb\x1b[0m:`, so a
            # plain "wandb:" substring test misses every one of its lines.
            lines = re.sub(r"\x1b\[[0-9;]*m", "", raw).splitlines()
            meaningful = [l for l in lines if l.strip() and "wandb:" not in l][-12:]
            if meaningful:
                self.node.base.extras.set("failure_tail", "\n".join(meaningful)[:2000])
                self.logger.error("eval produced no metrics; stdout tail: %s", meaningful[-1][:200])
        except Exception:  # noqa: BLE001 - diagnosis must never worsen the failure
            pass
