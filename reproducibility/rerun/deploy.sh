#!/usr/bin/env bash
# Copy the COMMITTED code (HEAD of this checkout) to the cluster. Only what training
# and evaluation need travels: lpno without its notebooks, the cluster
# scripts and the tools. Uncommitted changes do not travel; commit first.
#
#   bash reproducibility/rerun/deploy.sh
source "$(dirname "$0")/lib.sh"
cd "$REPO_ROOT"
if [ -n "$(git status --porcelain -- lpno reproducibility/rerun tools)" ]; then
  echo "deploy: uncommitted changes under lpno/, reproducibility/rerun/ or tools/ will NOT be deployed" >&2
fi
commit="$(git rev-parse HEAD)"
size=$(git archive --format=tar HEAD -- lpno reproducibility/rerun tools \
       ':(exclude)lpno/src/notebooks' | wc -c)
echo "deploy: $commit, $((size / 1024 / 1024)) MB"
git archive --format=tar HEAD -- lpno reproducibility/rerun tools ':(exclude)lpno/src/notebooks' \
  | remote "mkdir -p '$REMOTE_CODE' && tar -x -C '$REMOTE_CODE' && echo $commit > '$REMOTE_CODE/.deployed_commit'"
echo "deploy: done; the cluster copy is at commit $commit"
