"""Console service layer shared by the web console (`web/app`) and the
`aiida-pino` CLI.

Submodules that need FastAPI / pydantic-settings (`config`, `schemas`,
`security`, `service`) require the ``web`` optional extra; this package
init stays import-free so ``import aiida_pino`` works without it.
"""
