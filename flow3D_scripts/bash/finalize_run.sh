#!/usr/bin/env bash
# Post-process a completed run to keep only compact artifacts:
#   - subsample the NPZ to <= N frames (default 100)
#   - export the final-frame volume + surface VTK
#   - delete the multi-GB FLSGRF result (unless --keep-flsgrf)
#
#   bash bash/finalize_run.sh <case_name> [--max-frames N] [--keep-flsgrf]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE="${1:?usage: finalize_run.sh <case_name> [--max-frames N] [--keep-flsgrf]}"; shift
MAX_FRAMES=100; KEEP_FLSGRF=0
while [ $# -gt 0 ]; do
  case "$1" in
    --max-frames) MAX_FRAMES="$2"; shift 2;;
    --keep-flsgrf) KEEP_FLSGRF=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
CDIR="$REPO/runs/$CASE"
FLSGRF="$CDIR/flsgrf1.$CASE"
POST_HOME="${F3D_POST_HOME:-$HOME/.local/opt/flow3d/FLOW3D_POST_2025R1}"
cd "$REPO"
python3 python/subsample_npz.py "$CDIR/npz/$CASE.npz" --max-frames "$MAX_FRAMES"
F3D_POST_HOME="$POST_HOME" "$POST_HOME/bin/pvpython" python/export_final_frame_vtk.py \
  "$FLSGRF" "$CDIR/paraview/final" --prefix "$CASE" --representation volume
F3D_POST_HOME="$POST_HOME" "$POST_HOME/bin/pvpython" python/export_final_frame_vtk.py \
  "$FLSGRF" "$CDIR/paraview/final" --prefix "$CASE" --representation surface
if [ "$KEEP_FLSGRF" -eq 0 ]; then
  rm -f "$CDIR/flsgrf1.$CASE" "$CDIR/flsgrf1000.$CASE"
  echo "deleted FLSGRF for $CASE"
fi
echo "finalized $CASE (NPZ<=$MAX_FRAMES frames, final volume+surface VTK)"
