#!/usr/bin/env bash
# Local demo stack: one uvicorn backend + the static frontend on :8080.
# Usage: dev/run_local.sh [start [local|lan]|stop|status]
#   start        - loopback only, backend :8001: browse http://127.0.0.1:8080/
#   start lan    - bind 0.0.0.0, backend :8001, this machine's first IP address
#                  (or LPFNO_LAN_IP): other machines on the network can browse
#                  http://<ip>:8080/. This exposes :8001 and :8080, with no
#                  authentication, to everyone on that network.
# Both modes (re)write public/config.local.json so the pages call the matching
# backend origin. CORS allows that exact page origin only, so in lan mode browse
# via the IP address, not via 127.0.0.1.
set -euo pipefail

SERVER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND_PUBLIC="${LPFNO_FRONTEND_PUBLIC:-$SERVER_DIR/../lpno_web_demo_frontend/public}"
# The python that has the server installed: the active environment by default
# (`conda activate lpfno_demo` first, see scripts/setup_env.sh), or point
# LPFNO_PYTHON at an interpreter explicitly.
PY="${LPFNO_PYTHON:-$(command -v python || command -v python3)}"
PIDFILE="$SERVER_DIR/dev/.run_local_pids"
LOGDIR="$SERVER_DIR/dev"

case "${1:-start}" in
start)
  if ! (cd "$SERVER_DIR" && "$PY" -c "import app.main") >/dev/null 2>&1; then
    echo "python '$PY' cannot import the server (app.main). Activate the environment" >&2
    echo "created by scripts/setup_env.sh (conda activate lpfno_demo) or set LPFNO_PYTHON." >&2
    exit 1
  fi
  if [ -f "$PIDFILE" ] && kill -0 "$(head -1 "$PIDFILE")" 2>/dev/null; then
    echo "already running (see $PIDFILE); use '$0 stop' first" >&2
    exit 1
  fi
  MODE="${2:-local}"
  case "$MODE" in
    local)  IP=127.0.0.1; BIND=127.0.0.1; PORT=8001 ;;
    lan)    IP="${LPFNO_LAN_IP:-$(hostname -I | awk '{print $1}')}"; BIND=0.0.0.0; PORT=8001 ;;
    *) echo "usage: $0 [start [local|lan]|stop|status]" >&2; exit 2 ;;
  esac
  for port in "$PORT" 8080; do
    if ss -tln | grep -q ":$port "; then
      echo "port $port is already in use (stale server?); free it first: ss -tlnp | grep $port" >&2
      exit 1
    fi
  done
  printf '{\n  "backendBaseUrl": "http://%s:%s"\n}\n' "$IP" "$PORT" > "$FRONTEND_PUBLIC/config.local.json"
  cd "$SERVER_DIR"
  # Don't pin CUDA_VISIBLE_DEVICES: expose all GPUs so the backend
  # (LPFNO_CUDA_DEVICE=auto) loads each model on the freest one, skipping a full
  # GPU0. To force a GPU, set CUDA_VISIBLE_DEVICES or LPFNO_CUDA_DEVICE before
  # calling this script — both are inherited by the uvicorn child.
  # expandable_segments reduces CUDA fragmentation so inference is less likely to
  # OOM when the chosen GPU is tight (shared with training jobs).
  LPFNO_PAGES_ORIGIN="http://$IP:8080" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    nohup "$PY" -m uvicorn app.main:create_app --factory \
    --host "$BIND" --port "$PORT" --workers 1 > "$LOGDIR/backend.log" 2>&1 &
  echo "$!" > "$PIDFILE"
  nohup "$PY" "$SERVER_DIR/dev/serve_frontend.py" "$FRONTEND_PUBLIC" "$BIND" 8080 \
    > "$LOGDIR/frontend.log" 2>&1 &
  echo "$!" >> "$PIDFILE"
  echo "mode     $MODE (bind $BIND, backend :$PORT)"
  echo "backend  http://$IP:$PORT (log: dev/backend.log) - model load takes ~30 s"
  echo "frontend http://$IP:8080/"
  ;;
stop)
  [ -f "$PIDFILE" ] && xargs -r kill < "$PIDFILE" 2>/dev/null || true
  rm -f "$PIDFILE"
  echo "stopped"
  ;;
status)
  # Follow the backend the last start wrote into config.local.json; fall back to
  # loopback :8001.
  URL=$(grep -oE 'http://[^"]+' "$FRONTEND_PUBLIC/config.local.json" 2>/dev/null | head -1)
  curl -s -m 3 "${URL:-http://127.0.0.1:8001}/healthz" || echo "backend not responding"
  echo
  curl -s -o /dev/null -w "frontend :8080 -> %{http_code}\n" -m 3 http://127.0.0.1:8080/path.html || true
  ;;
*)
  echo "usage: $0 [start [local|lan]|stop|status]" >&2; exit 2 ;;
esac
