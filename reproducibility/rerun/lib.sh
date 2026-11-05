# Sourced by the cluster scripts on the laptop side. Loads the private site file and
# refuses to go on while a value is missing or still a placeholder.
#
# Site file, first found: $LPNO_SITE_ENV, ~/.config/lpno/site.env, reproducibility/rerun/site.env
# (template: reproducibility/rerun/site.env.example). Nothing site-specific is written anywhere
# else; tools/check_site_leaks.py refuses its values in any committed file.
set -euo pipefail
CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$CLUSTER_DIR" rev-parse --show-toplevel)"

_site_file() {
  local f
  for f in "${LPNO_SITE_ENV:-}" "$HOME/.config/lpno/site.env" "$CLUSTER_DIR/site.env"; do
    if [ -n "$f" ] && [ -f "$f" ]; then echo "$f"; return 0; fi
  done
  echo "cluster: no site file; copy reproducibility/rerun/site.env.example to ~/.config/lpno/site.env and fill it in" >&2
  return 1
}

# KEY=VALUE lines; values may contain spaces; ' #' starts a comment.
_load_site() {
  local line k v
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue ;; esac
    k="${line%%=*}"; v="${line#*=}"
    v="${v%% \#*}"; v="${v%"${v##*[![:space:]]}"}"; v="${v#\"}"; v="${v%\"}"
    export "$k=$v"
  done < "$1"
}

SITE_FILE="$(_site_file)"
_load_site "$SITE_FILE"

need() {
  local k
  for k in "$@"; do
    [ -n "${!k:-}" ] || { echo "cluster: $SITE_FILE has no $k" >&2; exit 1; }
    case "${!k}" in *"<"*">"*) echo "cluster: $k in $SITE_FILE is still a placeholder" >&2; exit 1 ;; esac
  done
}
need SITE_SSH REMOTE_CODE REMOTE_MAMBA REMOTE_ENV REMOTE_DATA REMOTE_RUNS \
     SLURM_ACCOUNT SLURM_PARTITION SLURM_QOS SLURM_BUILD_QOS SLURM_GRES

# shellcheck disable=SC2086  # SITE_SSH_OPTS is a list of options on purpose
remote() { ssh ${SITE_SSH_OPTS:-} "$SITE_SSH" "$@"; }

# submit <script under reproducibility/rerun/> <qos> <gres or ""> [-- KEY=VALUE ...]: sbatch on the
# cluster with every site value passed in, never written into the script.
submit() {
  local script="$1" qos="$2" gres="$3"; shift 3
  [ "${1:-}" = "--" ] && shift
  local exports="ALL,REMOTE_CODE=$REMOTE_CODE,REMOTE_MAMBA=$REMOTE_MAMBA,REMOTE_ENV=$REMOTE_ENV"
  exports="$exports,REMOTE_DATA=$REMOTE_DATA,REMOTE_RUNS=$REMOTE_RUNS"
  local kv
  for kv in "$@"; do exports="$exports,$kv"; done
  local g=""; [ -n "$gres" ] && g="--gres=$gres"
  local t=""; [ -n "${SUBMIT_TIME:-}" ] && t="--time=$SUBMIT_TIME"   # overrides the script's #SBATCH --time
  local d=""; [ -n "${SUBMIT_AFTER:-}" ] && d="--dependency=afterok:$SUBMIT_AFTER"
  remote "mkdir -p '$REMOTE_RUNS/logs' && cd '$REMOTE_CODE' && sbatch --parsable \
    --account='$SLURM_ACCOUNT' --partition='$SLURM_PARTITION' --qos='$qos' $g $t $d \
    --output='$REMOTE_RUNS/logs/%x-%j.out' --export='$exports' 'reproducibility/rerun/$script'"
}
