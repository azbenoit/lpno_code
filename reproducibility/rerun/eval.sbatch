#!/bin/bash
# Evaluate one model on one exported collection the way aiida-pino's eval adapter does: the
# model's own archived .hydra/config.yaml, its overrides replayed, every case scored.
# Submitted by `reproducibility/rerun/submit.sh eval`; nothing site-specific here.
#
#   MODEL  a deposited model under $REMOTE_DATA/models, or a run name under $REMOTE_RUNS
#   TEST   collection directory under $REMOTE_DATA
#   EVAL_OVERRIDES  optional extra Hydra overrides, space-separated, appended after the replayed
#          ones (e.g. '++custom.eval_per_case_melt_window=true')
#   OUT_TAG  optional suffix for the output directory, to keep a re-score beside the first eval
#SBATCH --job-name=lpno-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
: "${REMOTE_CODE:?}" "${REMOTE_ENV:?}" "${REMOTE_DATA:?}" "${REMOTE_RUNS:?}" "${MODEL:?}" "${TEST:?}"
PY="$REMOTE_ENV/bin/python"
if [ -e "$REMOTE_RUNS/$MODEL/model/.hydra/config.yaml" ]; then
  M="$REMOTE_RUNS/$MODEL/model"
elif [ -e "$REMOTE_DATA/models/$MODEL/.hydra/config.yaml" ]; then
  M="$REMOTE_DATA/models/$MODEL"
else
  echo "no model $MODEL under $REMOTE_RUNS or $REMOTE_DATA/models" >&2; exit 1
fi
[ -e "$REMOTE_DATA/$TEST/manifest.json" ] || { echo "missing collection $REMOTE_DATA/$TEST" >&2; exit 1; }
OUT="$REMOTE_RUNS/evals/${MODEL}_on_${TEST}_${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}${OUT_TAG:+_$OUT_TAG}"
mkdir -p "$OUT"
cd "$OUT"

mapfile -t OVR < <("$PY" "$REMOTE_CODE/reproducibility/rerun/replay_overrides.py" "$M/.hydra")
export PINO_LOADER_WORKERS="${PINO_LOADER_WORKERS:-0}" WANDB_MODE=offline
# physicsnemo's DistributedManager takes SLURM_PROCID to mean an srun launch and then reads
# srun's SLURM_LAUNCH_NODE_IPADDR, which a batch script does not have (TypeError at start-up).
# These runs are single-process, as on the machine that trained the published models.
unset SLURM_PROCID
# One file descriptor per memory-mapped array; see train.sbatch.
HARD="$(ulimit -Hn)"; [ "$HARD" = unlimited ] && HARD=65536
ulimit -n "$(( HARD < 65536 ? HARD : 65536 ))"
CODE_AT="$(cat "$REMOTE_CODE/.deployed_commit" 2>/dev/null || git -C "$REMOTE_CODE" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "eval $MODEL on $TEST, code $CODE_AT"

"$PY" -u -m lpno.models.NS_Energy_laser_PINO_eval --config-dir="$M/.hydra" --config-name=config \
  hydra.run.dir="$OUT" custom.name="${MODEL}_eval" network_dir="$M" \
  custom.data_dir="$REMOTE_DATA/$TEST" custom.val_data_dir="$REMOTE_DATA/$TEST" "${OVR[@]}" ${EVAL_OVERRIDES:-}

ERRORS="$(find "$OUT" -path '*fno_eval/model_errors.txt' | head -n 1)"
[ -n "$ERRORS" ] || { echo "the evaluation wrote no fno_eval/model_errors.txt" >&2; exit 1; }
echo "metrics: $ERRORS"
cat "$ERRORS"
