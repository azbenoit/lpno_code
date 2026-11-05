#!/usr/bin/env bash
# Restart the AiiDA daemon and VERIFY it came back up.
#
# Why this exists: `verdi daemon restart` has left the daemon DOWN four times in two
# days (2026-08-22/23). Its stop phase times out --
#
#     Stopping the daemon... FAILED
#     Critical: Connection to the daemon timed out.
#
# -- the start phase never runs, and you are left with no daemon and running CalcJobs
# that nobody is monitoring. Twice that went unnoticed for a while. The command's own
# exit status is not a reliable signal, so a restart has to be verified with `status`
# and not by the return code. This makes that mechanical.
#
# Usage:
#   bin/daemon_restart.sh              restart, then verify
#   bin/daemon_restart.sh --verify     verify only, change nothing
#
# Env: AIIDA_PROFILE (default: the configured default profile), AIIDA_DAEMON_WORKERS (default 3).
set -uo pipefail

PROFILE="${AIIDA_PROFILE:-$(python -c 'from aiida.manage.configuration import get_config; print(get_config().default_profile_name)' 2>/dev/null)}"
[ -n "$PROFILE" ] || { echo "daemon_restart: no AiiDA profile (set AIIDA_PROFILE)" >&2; exit 2; }
WORKERS="${AIIDA_DAEMON_WORKERS:-3}"

# `verdi daemon` shells out to `verdi` BY BARE NAME, so calling it through an absolute
# path from a non-activated env stops the daemon and then fails to start it
# (`ConfigurationError: Unable to find 'verdi' in the path`). Requiring it on PATH here
# is the same rule, enforced instead of documented.
if ! command -v verdi >/dev/null 2>&1; then
    echo "daemon_restart: \`verdi\` is not on PATH." >&2
    echo "  Activate the environment first (or prepend its bin/ to PATH) and retry;" >&2
    echo "  calling verdi by absolute path breaks its own subprocess calls." >&2
    exit 2
fi

# ONE status call, captured. An earlier version called `verdi daemon status` twice --
# once to test, once to print -- and on 2026-08-24 the daemon died BETWEEN the two: the
# test saw it running, the printout said "Error: The daemon is not running", and the
# script reported "OK, daemon is UP" over a daemon that was down. A verifier that samples
# twice does not verify one state, it verifies two, and this one exists precisely because
# the daemon dies at unpredictable moments during a restart.
verify() {
    local status
    status="$(verdi -p "$PROFILE" daemon status 2>&1)"
    printf '%s\n' "$status" | sed -n '1,6p'
    printf '%s' "$status" | grep -q "Daemon is running"
}

# Workers of THIS profile, by executable and argv rather than a loose pattern: a bare
# `pgrep -f "daemon worker"` matches the very shell running it, which has already cost this
# project a self-kill.
daemon_pids() {
    ps -u "$(id -u)" -o pid=,args= \
        | grep -F "/bin/python" \
        | grep -E "start-circus|daemon worker" \
        | grep -F -- "-p $PROFILE" \
        | grep -vF "grep" \
        | awk '{print $1}'
}

# The failure this function exists for, measured 2026-08-27. `verdi daemon stop` times out on
# its default timeout, reports FAILED, and LEAVES THE WORKERS RUNNING -- they reparent to
# init. `daemon start` then adds a second generation beside the first, and both consume the
# same RabbitMQ queue. Four orphaned workers plus three new ones is how 71 active processes
# ended up with no task anyone would poll: the whole fleet looked alive and nothing advanced
# it, and the only visible symptom was a `scheduler_lastchecktime` frozen at the restart.
#
# So: a long timeout first (300 s succeeds where the default does not), then verify by PROCESS
# that nothing survived, then signal what did. Starting on top of survivors is the bug.
stop_completely() {
    verdi -p "$PROFILE" daemon stop --timeout 300 2>&1 | sed 's/^/  /'
    local left
    left="$(daemon_pids)"
    [ -z "$left" ] && return 0
    echo "daemon_restart: stop left workers running; signalling $(printf '%s' "$left" | wc -w)" >&2
    # shellcheck disable=SC2086
    kill $left 2>/dev/null || true
    sleep 5
    left="$(daemon_pids)"
    [ -z "$left" ] && return 0
    echo "daemon_restart: workers survived TERM; sending KILL" >&2
    # shellcheck disable=SC2086
    kill -9 $left 2>/dev/null || true
    sleep 3
    left="$(daemon_pids)"
    if [ -n "$left" ]; then
        echo "daemon_restart: FAILED -- workers still running: $left" >&2
        echo "  Starting now would run two generations against one queue; refusing." >&2
        return 1
    fi
    return 0
}

# Every env-gated daemon behaviour is reset by ANY restart from ANY shell, and every
# daemon-side change needs a restart to deploy -- so losing one is not a mistake
# somebody makes once, it is the default outcome. Measured 2026-08-27: two
# `verdi daemon restart`s deploying an unrelated fix ran from a shell without
# `AIIDA_PINO_REMOTE_READ_TIME`, which silently disabled stride demotion for the remote
# computer. Every submission whose derivation carried a stride then died at
# `validate_inputs` with exit 303 ([2,2,1] against [1,1,1]) -- three runs lost before the
# cause was found, and nothing in the failure pointed at an environment variable.
#
# So the restart CARRIES the running daemon's own settings forward. The shell that restarts
# no longer has to know what the daemon was configured with; only a deliberate change does,
# and an explicit value in the caller's environment still wins.
#
# Every `AIIDA_PINO_*` and `PINO_*` var is carried rather than a named list: a list drifts
# out of date silently, and the next var added would be lost exactly the way this one was.
DAEMON_ENV_FILE="${AIIDA_PINO_DAEMON_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/daemon.env}"

carry_daemon_env() {
    local pid carried=() skipped=() name value
    pid="$(daemon_pids | head -n1)"
    if [ -n "$pid" ] && [ -r "/proc/$pid/environ" ]; then
        while IFS= read -r -d '' entry; do
            name="${entry%%=*}"
            value="${entry#*=}"
            case "$name" in
                AIIDA_PINO_*|PINO_*) ;;
                *) continue ;;
            esac
            # The caller's own value wins: that is how a deliberate change is made.
            if [ -n "${!name:-}" ]; then
                [ "${!name}" != "$value" ] && skipped+=("$name")
                continue
            fi
            export "$name=$value"
            carried+=("$name")
        done < "/proc/$pid/environ"
    fi
    # Cold start: no daemon to read, so fall back to an optional untracked env file. Keep
    # machine-specific values there, never in a tracked file.
    if [ -z "$pid" ] && [ -r "$DAEMON_ENV_FILE" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$DAEMON_ENV_FILE"
        set +a
        echo "daemon_restart: no daemon running; loaded $DAEMON_ENV_FILE"
    fi
    if [ ${#carried[@]} -gt 0 ]; then
        echo "daemon_restart: carried forward from the running daemon: ${carried[*]}"
    fi
    if [ ${#skipped[@]} -gt 0 ]; then
        echo "daemon_restart: overridden by this shell: ${skipped[*]}"
    fi
    if [ -n "$pid" ] && [ ${#carried[@]} -eq 0 ] && [ ${#skipped[@]} -eq 0 ]; then
        echo "daemon_restart: the running daemon had no AIIDA_PINO_*/PINO_* settings to carry"
    fi
}

if [ "${1:-}" = "--verify" ]; then
    if verify; then echo "daemon_restart: daemon is UP"; exit 0; fi
    echo "daemon_restart: daemon is DOWN on profile '$PROFILE'" >&2
    exit 1
fi

echo "daemon_restart: restarting profile '$PROFILE'..."
# BEFORE the stop: once the workers are gone, so is the environment they were holding.
carry_daemon_env
# STOP AND CONFIRM, then start. `verdi daemon restart` does both in one step and cannot be
# told that its stop failed, which is exactly how two generations end up sharing a queue.
if ! stop_completely; then
    exit 1
fi
verdi -p "$PROFILE" daemon start "$WORKERS" 2>&1 | sed 's/^/  /'
if verify; then
    echo "daemon_restart: OK, daemon is UP ($(daemon_pids | wc -w) processes, one generation)"
    exit 0
fi

echo "daemon_restart: FAILED -- daemon is still DOWN on profile '$PROFILE'." >&2
echo "  Running CalcJobs are unmonitored until this is resolved." >&2
exit 1
