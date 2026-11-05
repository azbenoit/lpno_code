#!/usr/bin/env bash
# Thin, shell-friendly entry point for the validated JSON workflow.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT_DIR/python/run_rectangular_case.py" "$@"
