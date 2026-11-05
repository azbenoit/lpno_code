#!/usr/bin/env bash
# Run a training or an evaluation on THIS machine, without Slurm and without AiiDA: the same
# job scripts the cluster uses (reproducibility/rerun/train.sbatch, reproducibility/rerun/eval.sbatch), with this
# checkout as the code and the active Python environment (tools/setup_env.py) as the env.
#
#   bash reproducibility/rerun/run_local.sh eval  DATA=<data dir> RUNS=<runs dir> MODEL=<model> TEST=<collection>
#   bash reproducibility/rerun/run_local.sh train DATA=<data dir> RUNS=<runs dir> RUN_NAME=<name> LIKE=<model> \
#                                   TRAIN=<collection> VAL=<collection> [SEED=<n>] [MAX_STEPS=<n>]
#
# DATA holds the collections written by reproducibility/rerun/export_deposit.py and a models/ folder of
# deposited models (export_deposit.py model); RUNS receives runs/<RUN_NAME>/model and evals/.
set -euo pipefail
kind="${1:-}"; shift || true
case "$kind" in
  train|eval) ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
for kv in "$@"; do
  case "$kv" in
    [A-Z]*=*) export "${kv?}" ;;
    *) echo "run_local: expected KEY=VALUE, got '$kv'" >&2; exit 1 ;;
  esac
done
: "${DATA:?DATA=<data dir> is required}" "${RUNS:?RUNS=<runs dir> is required}"
REMOTE_CODE="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE_ENV="$(cd "$(dirname "$(command -v python)")/.." && pwd)"   # the active environment
REMOTE_DATA="$(cd "$DATA" && pwd)"
mkdir -p "$RUNS"
REMOTE_RUNS="$(cd "$RUNS" && pwd)"
export REMOTE_CODE REMOTE_ENV REMOTE_DATA REMOTE_RUNS
echo "run_local $kind: code $REMOTE_CODE, python $REMOTE_ENV/bin/python"
exec bash "$REMOTE_CODE/reproducibility/rerun/$kind.sbatch"
