#!/bin/bash
# Several trainings on ONE GPU, each followed by its evaluations, in one allocation. A steady FNO
# run on these grids peaks at ~2 GB of GPU memory and keeps about one CPU core busy (single-threaded
# loading, PINO_LOADER_WORKERS=0), so a lone run leaves most of the card idle. Each run is
# train.sbatch then eval.sbatch per test set, run as plain scripts inside this allocation, so a
# packed run is the same run as a lone one. Submitted by `reproducibility/rerun/submit.sh pack`.
#
#   PACK   runs separated by ';', each RUN_NAME:LIKE:TRAIN[:SEED] (meanings as in train.sbatch;
#          a run's own SEED wins over the pack-wide one)
#   VAL    validation collection; SEED and MAX_STEPS optional, as in train.sbatch
#   TESTS  optional test collections separated by ':', each scored after its run trains
#SBATCH --job-name=lpno-pack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
: "${REMOTE_CODE:?}" "${REMOTE_RUNS:?}" "${PACK:?}" "${VAL:?}"
HERE="$REMOTE_CODE/reproducibility/rerun"
IFS=';' read -ra runs <<<"$PACK"
IFS=':' read -ra tests <<<"${TESTS:-}"
# Share the cores out: oversubscribed intra-op threads slow every run down.
threads=$(( ${SLURM_CPUS_PER_TASK:-16} / ${#runs[@]} ))
export OMP_NUM_THREADS=$(( threads > 0 ? threads : 1 ))
echo "pack: ${#runs[@]} runs on one GPU, ${#tests[@]} test set(s) each, OMP_NUM_THREADS=$OMP_NUM_THREADS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

pids=(); names=()
for spec in "${runs[@]}"; do
  IFS=':' read -r name like train seed <<<"$spec"
  [ -n "$name" ] && [ -n "$like" ] && [ -n "$train" ] || { echo "pack: bad entry '$spec'" >&2; exit 1; }
  log="$REMOTE_RUNS/logs/lpno-pack-${SLURM_JOB_ID:-local}-$name.out"
  (
    SEED="${seed:-${SEED:-}}" RUN_NAME="$name" LIKE="$like" TRAIN="$train" bash "$HERE/train.sbatch"
    for t in "${tests[@]}"; do
      MODEL="$name" TEST="$t" bash "$HERE/eval.sbatch"
    done
  ) >"$log" 2>&1 &
  pids+=("$!"); names+=("$name")
  echo "pack: started $name (log $log)"
  sleep 30   # stagger start-up: the first reads of every member hit the shared file system at once
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "pack: ok   ${names[$i]}"; else echo "pack: FAIL ${names[$i]}"; status=1; fi
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true
exit "$status"
