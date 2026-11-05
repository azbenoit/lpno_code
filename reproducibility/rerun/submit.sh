#!/usr/bin/env bash
# Laptop-side entry point for cluster runs. Every site value comes from the private
# site file (reproducibility/rerun/lib.sh); see reproducibility/rerun/README.md for the whole sequence.
#
#   bash reproducibility/rerun/submit.sh deploy                 # committed code -> cluster
#   bash reproducibility/rerun/submit.sh build-env              # training environment (build QOS, no GPU)
#   bash reproducibility/rerun/submit.sh check-gpu              # one short GPU job: torch sees the card
#   bash reproducibility/rerun/submit.sh upload <dir> [entry..] # exported collections/models -> cluster data dir
#   bash reproducibility/rerun/submit.sh train KEY=VALUE ...    # RUN_NAME LIKE TRAIN VAL [SEED MAX_STEPS DEBUG=1]
#   bash reproducibility/rerun/submit.sh eval KEY=VALUE ...     # MODEL TEST [DEBUG=1: debug QOS, smoke tests only]
#   bash reproducibility/rerun/submit.sh pack KEY=VALUE ...     # PACK='run:like:train[:seed];...' VAL [TESTS='a:b' SEED MAX_STEPS]
#                                                 # several trainings (+ their evals) on one GPU
#   bash reproducibility/rerun/submit.sh evals KEY=VALUE ...    # EVALS='model:test;...' [PARALLEL EVAL_OVERRIDES OUT_TAG]
#                                                 # AFTER=<jobid> on either: wait for that job to succeed
#   bash reproducibility/rerun/submit.sh status                 # your queued and running jobs
#   bash reproducibility/rerun/submit.sh log <jobid> [lines]    # tail a job's log
source "$(dirname "$0")/lib.sh"

# KEY=VALUE pairs only: they become job environment through sbatch --export, which
# splits on commas.
pairs() {
  local kv
  for kv in "$@"; do
    case "$kv" in
      [A-Z]*=*,*) echo "submit: '$kv' contains a comma, which sbatch --export cannot carry" >&2; exit 1 ;;
      [A-Z]*=*) ;;
      *) echo "submit: expected KEY=VALUE, got '$kv'" >&2; exit 1 ;;
    esac
  done
}

# DEBUG=1 among the pairs sends a train/eval job to the debug QOS. That QOS is for short
# tests of the setup (a validation eval, a MAX_STEPS training), not results, and misusing it
# can cost cluster access, so the opt-in is explicit and fenced: a 50-minute limit, one debug
# job at a time, and no training without MAX_STEPS.
QOS="$SLURM_QOS"
debug_qos() {
  local kv keep=() debug=0 steps=0
  for kv in "$@"; do
    case "$kv" in
      DEBUG=1) debug=1 ;;
      DEBUG=*) ;;
      AFTER=*)   # start only once job(s) AFTER finished successfully (Slurm afterok; ids joined by ':')
        [[ "${kv#AFTER=}" =~ ^[0-9]+(:[0-9]+)*$ ]] || { echo "submit: AFTER takes job ids, got '$kv'" >&2; exit 1; }
        export SUBMIT_AFTER="${kv#AFTER=}" ;;
      MAX_STEPS=*) steps=1; keep+=("$kv") ;;
      *) keep+=("$kv") ;;
    esac
  done
  PAIRS=("${keep[@]}")
  [ "$debug" = 1 ] || return 0
  need SLURM_DEBUG_QOS
  if { [ "$cmd" = train ] || [ "$cmd" = pack ]; } && [ "$steps" = 0 ]; then
    echo "submit: DEBUG=1 trainings must set MAX_STEPS; a full training belongs on $SLURM_QOS" >&2; exit 1
  fi
  if [ -n "$(remote "squeue --me -h -q '$SLURM_DEBUG_QOS' -o %i")" ]; then
    echo "submit: a debug job is already queued or running; one at a time" >&2; exit 1
  fi
  QOS="$SLURM_DEBUG_QOS"; export SUBMIT_TIME=00:50:00
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  deploy)    bash "$CLUSTER_DIR/deploy.sh" ;;
  build-env) submit build_env.sbatch "$SLURM_BUILD_QOS" "" -- "$@" ;;
  check-gpu) submit check_gpu.sbatch "${SLURM_DEBUG_QOS:-$SLURM_QOS}" "$SLURM_GRES" -- "$@" ;;
  upload)
    [ $# -ge 1 ] || { echo "usage: submit.sh upload <local data dir> [entry ...]" >&2; exit 1; }
    src="${1%/}"; shift
    if [ $# -gt 0 ]; then entries=("${@/#/$src/}"); else shopt -s nullglob; entries=("$src"/*); fi
    [ ${#entries[@]} -gt 0 ] || { echo "upload: nothing to send in $src" >&2; exit 1; }
    remote "mkdir -p '$REMOTE_DATA'"
    # -H keeps the hard links export_deposit.py makes between collections sharing members;
    # --partial lets an interrupted transfer resume.
    # shellcheck disable=SC2086
    rsync -aH --partial --info=progress2 -z -e "ssh ${SITE_SSH_OPTS:-}" -- "${entries[@]}" "$SITE_SSH:$REMOTE_DATA/"
    ;;
  train) pairs "$@"; debug_qos "$@"; submit train.sbatch "$QOS" "$SLURM_GRES" -- "${PAIRS[@]}" ;;
  eval)  pairs "$@"; debug_qos "$@"; submit eval.sbatch "$QOS" "$SLURM_GRES" -- "${PAIRS[@]}" ;;
  pack)  pairs "$@"; debug_qos "$@"; submit pack.sbatch "$QOS" "$SLURM_GRES" -- "${PAIRS[@]}" ;;
  evals) pairs "$@"; debug_qos "$@"; submit evals.sbatch "$QOS" "$SLURM_GRES" -- "${PAIRS[@]}" ;;
  status)    remote "squeue --me -o '%.10i %.24j %.8T %.10M %.12l %R'" ;;
  log)       remote "tail -n ${2:-60} '$REMOTE_RUNS/logs/'*-'$1'.out" ;;
  help|*)    sed -n '2,16p' "$0" ;;
esac
