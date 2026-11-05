#!/usr/bin/env bash
# Local console stack: one uvicorn backend (loads the AiiDA profile) + a static
# frontend. Mirrors the web demo backend's dev/run_local.sh.
#
# Usage: web/dev/run_local.sh [start [local|lan]|stop|status]
#   start        - loopback only:  http://127.0.0.1:<FE>/index.html
#   start lan    - bind 0.0.0.0, advertise this machine's LAN IP so other machines
#                  can browse http://<ip>:<FE>/index.html (exposes the ports to
#                  the LAN). Browse via the IP, not 127.0.0.1, or CORS rejects.
#
# Every mode (re)writes public/config.local.json so the page calls the matching
# backend origin; CORS on the backend is pinned to exactly that page origin.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"          # aiida-pino repo root
PUBLIC="${AIIDA_PINO_WEB_PUBLIC:-$ROOT/web/public}"
# Interpreter: AIIDA_PINO_WEB_PYTHON, else a `pino` conda env under $HOME if one
# exists, else python3 on PATH (the active venv). Checked before anything is
# started: an earlier version nohup'd a missing binary and still printed both
# URLs, so the quick start looked like it worked and silently did nothing.
PY="${AIIDA_PINO_WEB_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$HOME/miniconda3/envs/pino/bin/python" ]; then PY="$HOME/miniconda3/envs/pino/bin/python"
  else PY="$(command -v python3 || true)"; fi
fi
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "error: python interpreter not found: ${PY:-python3}" >&2
  echo "       set AIIDA_PINO_WEB_PYTHON to an interpreter with aiida and the web extra installed" >&2
  exit 1
fi
if ! "$PY" -c "import aiida, fastapi, uvicorn" 2>/dev/null; then
  echo "error: $PY lacks aiida/fastapi/uvicorn; pip install -e '.[web]' there, or set AIIDA_PINO_WEB_PYTHON" >&2
  exit 1
fi
BACKEND_PORT="${AIIDA_PINO_WEB_PORT:-8100}"
FRONTEND_PORT="${AIIDA_PINO_WEB_FRONTEND_PORT:-8180}"
PROFILE="${AIIDA_PINO_WEB_AIIDA_PROFILE:-}"
# The flow3D_scripts checkout the console's FLOW-3D submits pin. Unset, the
# adapter falls back to the bundled `external/flow3D_scripts` submodule, which
# predates the 2026-07-29 calibration campaign and silently ignores
# `recoil_model` / `void_vapor_pressure_pa`. Default to the sibling checkout
# next to this repo, which is where every other launch path points.
if [ -z "${AIIDA_PINO_WEB_FLOW3D_SCRIPTS_ROOT:-}" ] \
   && [ -f "$(dirname "$ROOT")/flow3D_scripts/python/run_rectangular_case.py" ]; then
  AIIDA_PINO_WEB_FLOW3D_SCRIPTS_ROOT="$(cd "$(dirname "$ROOT")/flow3D_scripts" && pwd)"
fi
FLOW3D_ROOT="${AIIDA_PINO_WEB_FLOW3D_SCRIPTS_ROOT:-}"
PIDFILE="$ROOT/web/dev/.run_local_pids"
LOGDIR="$ROOT/web/dev"

case "${1:-start}" in
start)
  if [ -f "$PIDFILE" ] && kill -0 "$(head -1 "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    echo "already running (see $PIDFILE); '$0 stop' first" >&2; exit 1
  fi
  MODE="${2:-local}"
  case "$MODE" in
    local) IP=127.0.0.1; BIND=127.0.0.1 ;;
    lan)   IP="${AIIDA_PINO_WEB_LAN_IP:-$(hostname -I | awk '{print $1}')}"; BIND=0.0.0.0 ;;
    *) echo "usage: $0 [start [local|lan]|stop|status]" >&2; exit 2 ;;
  esac
  for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
    if ss -tln | grep -q ":$port "; then
      echo "port $port already in use; free it first: ss -tlnp | grep $port" >&2; exit 1
    fi
  done
  printf '{\n  "backendBaseUrl": "http://%s:%s"\n}\n' "$IP" "$BACKEND_PORT" > "$PUBLIC/config.local.json"
  cd "$ROOT"
  # APPEND, never truncate. The logs are the only record of an intermittent
  # fault, and the act of deploying a fix for one is a restart -- so a
  # truncating `>` destroys the evidence for exactly the bug being chased. On
  # 2026-08-21 a greenback/event-loop traceback was read out of this file, five
  # restarts happened while fixing it, and by the time anyone looked again the
  # traceback was gone. A boundary line makes runs separable without losing any.
  {
    echo "=== console start $(date -Is) sha=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown) mode=$MODE ==="
  } >> "$LOGDIR/backend.log"
  AIIDA_PINO_WEB_PAGES_ORIGIN="http://$IP:$FRONTEND_PORT" \
  AIIDA_PINO_WEB_AIIDA_PROFILE="$PROFILE" \
  AIIDA_PINO_WEB_FLOW3D_SCRIPTS_ROOT="$FLOW3D_ROOT" \
    setsid nohup "$PY" -m uvicorn web.app.main:create_app --factory \
    --host "$BIND" --port "$BACKEND_PORT" --workers 1 >> "$LOGDIR/backend.log" 2>&1 &
  echo "$!" > "$PIDFILE"
  setsid nohup "$PY" "$ROOT/web/dev/serve_frontend.py" "$PUBLIC" "$BIND" "$FRONTEND_PORT" >> "$LOGDIR/frontend.log" 2>&1 &
  echo "$!" >> "$PIDFILE"
  echo "mode     $MODE (bind $BIND)"
  # `setsid`, not just `nohup`: nohup ignores SIGHUP but NOT SIGTERM, and the
  # console keeps the launching shell's process GROUP, so a group-kill at
  # session teardown takes it down. On 2026-08-21 the console died that way with
  # a clean "Application shutdown complete" in the log -- no crash, just SIGTERM
  # -- and the user found it gone. setsid puts it in its own session and group,
  # so it outlives whatever started it.
  # nohup succeeds even when uvicorn dies on the next line (bad profile, missing
  # extra, port taken by something ss did not show), so printing two URLs here
  # was a start message that could not tell success from failure. Poll /healthz.
  for _ in $(seq 30); do
    curl -s -m 2 -o /dev/null "http://127.0.0.1:$BACKEND_PORT/healthz" && break
    sleep 1
  done
  if curl -s -m 2 -o /dev/null "http://127.0.0.1:$BACKEND_PORT/healthz"; then
    echo "backend  http://$IP:$BACKEND_PORT (log: web/dev/backend.log)"
    echo "console  http://$IP:$FRONTEND_PORT/index.html"
  else
    echo "backend  FAILED to answer /healthz within 30s" >&2
    tail -n 15 "$LOGDIR/backend.log" >&2
    exit 1
  fi
  ;;
stop)
  PIDS=""
  [ -f "$PIDFILE" ] && PIDS=$(tr '\n' ' ' < "$PIDFILE")
  [ -n "$PIDS" ] && kill $PIDS 2>/dev/null || true
  # `kill` returns as soon as the signal is DELIVERED, not once the process has
  # exited and released its socket, so `stop && start` raced its own teardown
  # and failed with "port already in use" on a port nothing was listening on.
  for _ in $(seq 40); do
    if ! (exec 3<>/dev/tcp/127.0.0.1/"$BACKEND_PORT") 2>/dev/null; then break; fi
    sleep 0.25
  done
  # A closed port is NOT proof the process died. uvicorn drops its listener on
  # SIGTERM and then waits for in-flight requests; a sync endpoint running in
  # the threadpool (a long stats compute) can hold it there indefinitely. On
  # 2026-08-15 that left an orphaned backend alive for 23.7 h holding 89 GB --
  # invisible to a later `stop` (pidfile already deleted) and to `status` (no
  # port), and it drove the box to 8 GB MemAvailable. Confirm the exit and
  # escalate; only then drop the pidfile.
  for pid in $PIDS; do
    for _ in $(seq 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "pid $pid ignored SIGTERM (in-flight request?); sending SIGKILL" >&2
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  rm -f "$PIDFILE"
  echo "stopped"
  ;;
status)
  URL=$(grep -oE 'http://[^"]+' "$PUBLIC/config.local.json" 2>/dev/null | head -1)
  curl -s -m 3 "${URL:-http://127.0.0.1:$BACKEND_PORT}/healthz" || echo "backend not responding"
  echo
  curl -s -o /dev/null -w "frontend :$FRONTEND_PORT -> %{http_code}\n" -m 3 "http://127.0.0.1:$FRONTEND_PORT/index.html" || true
  ;;
*)
  echo "usage: $0 [start [local|lan]|stop|status]" >&2; exit 2 ;;
esac
