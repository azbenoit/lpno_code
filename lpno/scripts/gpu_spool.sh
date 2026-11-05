#!/usr/bin/env bash
# Thin, shell-friendly entry point for the per-GPU training spooler.
set -euo pipefail

exec python -m lpno.gpu_spool "$@"
