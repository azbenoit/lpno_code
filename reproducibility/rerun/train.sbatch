#!/bin/bash
# Train one model on a GPU node with the settings of a reference model: its archived
# .hydra (config name and override list, replayed by reproducibility/rerun/replay_overrides.py) on
# collections exported by reproducibility/rerun/export_deposit.py. Submitted by `reproducibility/rerun/submit.sh train`,
# which passes the site values; nothing site-specific here.
#
#   RUN_NAME   name of the run; output in $REMOTE_RUNS/$RUN_NAME (refused if it holds weights)
#   LIKE       reference model under $REMOTE_DATA/models, e.g. fno_maximin_r05_alloyflag_s42
#   TRAIN, VAL collection directories under $REMOTE_DATA
#   SEED       optional, replaces the reference seed; MAX_STEPS optional (smoke tests)
#SBATCH --job-name=lpno-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
set -euo pipefail
: "${REMOTE_CODE:?}" "${REMOTE_ENV:?}" "${REMOTE_DATA:?}" "${REMOTE_RUNS:?}"
: "${RUN_NAME:?}" "${LIKE:?}" "${TRAIN:?}" "${VAL:?}"
PY="$REMOTE_ENV/bin/python"
REF="$REMOTE_DATA/models/$LIKE/.hydra"
RUN="$REMOTE_RUNS/$RUN_NAME"
for d in "$REF" "$REMOTE_DATA/$TRAIN" "$REMOTE_DATA/$VAL"; do
  [ -e "$d" ] || { echo "missing $d (upload it with reproducibility/rerun/submit.sh upload)" >&2; exit 1; }
done
if [ -e "$RUN/model/u_network.0.pth" ]; then
  echo "$RUN already holds a trained model; choose another RUN_NAME" >&2; exit 1
fi
mkdir -p "$RUN/model"
cd "$RUN"   # hydra writes outputs/<name>_<stamp>/ under the working directory

CONFIG="$("$PY" "$REMOTE_CODE/reproducibility/rerun/replay_overrides.py" "$REF" --config-name)"
mapfile -t OVR < <("$PY" "$REMOTE_CODE/reproducibility/rerun/replay_overrides.py" "$REF" \
  ${SEED:+--seed "$SEED"} ${MAX_STEPS:+--max-steps "$MAX_STEPS"})
export PINO_LOADER_WORKERS="${PINO_LOADER_WORKERS:-0}" WANDB_MODE=offline
# Single process: see the same line in eval.sbatch.
unset SLURM_PROCID
# The arraydir loader memory-maps every array of every member, one file descriptor each:
# ~3500 for a 295-case set and ~5000 for the 419-case one, past the usual soft limit of 1024.
HARD="$(ulimit -Hn)"; [ "$HARD" = unlimited ] && HARD=65536
ulimit -n "$(( HARD < 65536 ? HARD : 65536 ))"
CODE_AT="$(cat "$REMOTE_CODE/.deployed_commit" 2>/dev/null || git -C "$REMOTE_CODE" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "train $RUN_NAME: $CONFIG like $LIKE, train $TRAIN, val $VAL, code $CODE_AT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi here"

"$PY" -u -m lpno.models.NS_Energy_laser_PINO --config-name="$CONFIG" \
  custom.name="$RUN_NAME" custom.data_dir="$REMOTE_DATA/$TRAIN" \
  custom.val_data_dir="$REMOTE_DATA/$VAL" network_dir="$RUN/model" "${OVR[@]}"

# The run's config travels with its weights, as aiida-pino's train adapter does; eval reads it.
HYDRA_DIR="$(ls -td "$RUN"/outputs/*/.hydra | head -n 1)"
mkdir -p "$RUN/model/.hydra"
cp "$HYDRA_DIR"/*.yaml "$RUN/model/.hydra/"
echo "model ready: $RUN/model"
