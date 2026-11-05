#!/bin/bash
# Many evaluations in one GPU allocation: eval.sbatch per MODEL:TEST pair, PARALLEL at a time.
# A steady eval takes a minute or less, so one job per eval spends more time queued than
# scoring. Submitted by `reproducibility/rerun/submit.sh evals`; nothing site-specific here.
#
#   EVALS           pairs separated by ';', each MODEL:TEST (meanings as in eval.sbatch)
#   PARALLEL        evaluations at once on the one GPU (default 3)
#   EVAL_OVERRIDES  and OUT_TAG, passed to every evaluation (see eval.sbatch)
#SBATCH --job-name=lpno-evals
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
: "${REMOTE_CODE:?}" "${REMOTE_RUNS:?}" "${EVALS:?}"
HERE="$REMOTE_CODE/reproducibility/rerun"
IFS=';' read -ra pairs <<<"$EVALS"
par="${PARALLEL:-3}"
echo "evals: ${#pairs[@]} evaluations, $par at a time, overrides: ${EVAL_OVERRIDES:-none}"
status=0
running=0
for pair in "${pairs[@]}"; do
  IFS=':' read -r model test <<<"$pair"
  [ -n "$model" ] && [ -n "$test" ] || { echo "evals: bad entry '$pair'" >&2; exit 1; }
  log="$REMOTE_RUNS/logs/lpno-evals-${SLURM_JOB_ID:-local}-${model}-on-${test%%__*}.out"
  (MODEL="$model" TEST="$test" bash "$HERE/eval.sbatch" >"$log" 2>&1 \
     && echo "evals: ok   $model on $test" || { echo "evals: FAIL $model on $test (log $log)"; exit 1; }) &
  running=$((running + 1))
  if [ "$running" -ge "$par" ]; then
    wait -n || status=1
    running=$((running - 1))
  fi
done
while [ "$running" -gt 0 ]; do
  wait -n || status=1
  running=$((running - 1))
done
exit "$status"
