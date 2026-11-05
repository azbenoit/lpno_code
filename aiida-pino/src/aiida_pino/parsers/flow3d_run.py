"""Parser for FLOW-3D run jobs launched through flow3D_scripts."""

from __future__ import annotations

from aiida import orm
from aiida.common import exceptions

from .common import FolderResultParser, stamp_runtime_extras


class Flow3DRunParser(FolderResultParser):
    """Parse the JSON run summary and the curated run-artifact folder."""

    summary_filename = "flow3d_run_summary.json"

    def _npz_evidence(self) -> dict | None:
        """What the run's own NPZ says, when the summary JSON is missing.

        The summary is adapter bookkeeping; the NPZ is the artifact that
        matters. On 2026-07-22 every failed run on the development
        machine (six of them) had a complete, openable NPZ on disk -- 6 of 6
        were bookkeeping failures with no data loss -- and because the batch
        does not retry, each case silently dropped out of every dataset while
        looking like a legitimate failure. At the observed ~1-in-30 rate that is
        ~10 lost cases per 300-case sweep.

        Returns None when there is no usable NPZ, so a genuine failure still
        fails.
        """
        try:
            retrieved = self.retrieved
            names = [n for n in retrieved.base.repository.list_object_names()
                     if n.endswith(".npz")]
        except Exception:  # noqa: BLE001 - no retrieved folder at all
            return None
        if not names:
            return None
        # Existence is not enough: a truncated archive would fail later, in the
        # importer, where the cause is much harder to see.
        import io
        import zipfile

        for name in names:
            try:
                raw = retrieved.base.repository.get_object_content(name, mode="rb")
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    members = archive.namelist()
                    if archive.testzip() is not None or not members:
                        continue
                return {"npz": name, "arrays": len(members), "bytes": len(raw)}
            except Exception:  # noqa: BLE001 - unreadable/truncated: not evidence
                continue
        return None

    def parse(self, **kwargs):
        summary = self.read_json(self.summary_filename, required=False)
        if summary is None:
            evidence = self._npz_evidence()
            if evidence is None:
                return self.exit_codes.ERROR_MISSING_SUMMARY
            # The solve produced data. Say so loudly on the node and let the run
            # stand, rather than discarding a completed case over its receipt.
            self.node.base.extras.set("summary_missing_npz_ok", evidence)
            self.logger.warning(
                f"{self.summary_filename} is missing, but {evidence['npz']} holds "
                f"{evidence['arrays']} arrays and opens cleanly; treating the run as "
                "completed. The summary-derived runtime extras are absent for this run."
            )
            summary = {"status": "ok", "recovered_from": "npz", **evidence}
        stamp_runtime_extras(self.node, summary)

        run_summary = orm.Dict(dict=summary)
        run_summary.label = "run_summary"
        self.out("run_summary", run_summary)

        if summary.get("status") != "ok":
            return self.exit_codes.ERROR_RUN_FAILED

        try:
            run_artifacts = self.build_folder_output("run_artifacts")
        except exceptions.ParsingError:
            return self.exit_codes.ERROR_MISSING_ARTIFACTS
        run_artifacts.label = "run_artifacts"
        run_artifacts.base.extras.set("aiida_pino_kind", "run_artifacts")
        # Material identity out of the rendered prepin this folder retains.
        # `run_config.material.file` is a path and the manifest records only a
        # row count, so "which sigma arm is this" was unanswerable by query
        # while the values sat right here.
        from aiida_pino.utils.material_identity import stamp_from_artifacts

        stamp_from_artifacts(
            self.node, run_artifacts, (summary.get("manifest") or {}).get("material")
        )
        self.out("run_artifacts", run_artifacts)
