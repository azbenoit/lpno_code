#!/usr/bin/env bash
# Thin, shell-friendly entry point for the FLOW-3D task spooler.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT_DIR/python/spool.py" "$@"
