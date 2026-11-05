"""Parser for PINO training jobs."""

from __future__ import annotations

from .common import FolderResultParser


class PinoTrainParser(FolderResultParser):
    """Parse PINO training artifact outputs."""

    #: Training logs `[step: N]`; eval jobs do not, so only this parser records
    #: how far the run got.
    tracks_steps = True

    folder_label = "training_artifacts"
    folder_subpath = "training_output"
    summary_label = "train_summary"
    summary_filename = "train_summary.json"
