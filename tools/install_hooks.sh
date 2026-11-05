#!/usr/bin/env bash
# Install the pre-commit hook that runs tools/check_site_leaks.py before every commit.
# A hook already in place is kept and still runs after the check. Hooks live in the
# shared git directory, so this covers every worktree of the clone; a checkout that
# predates the checker skips it with a note instead of failing.
set -euo pipefail
# The repository this script belongs to, whatever directory it is called from: resolving
# "the current repository" instead once installed the hook into a different clone.
cd "$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
[ -f tools/check_site_leaks.py ] || { echo "install_hooks: not the repository that holds tools/check_site_leaks.py" >&2; exit 1; }
hooks="$(git config core.hooksPath || true)"
[ -n "$hooks" ] || hooks="$(git rev-parse --git-common-dir)/hooks"
mkdir -p "$hooks"
if [ -e "$hooks/pre-commit" ] && ! grep -q check_site_leaks "$hooks/pre-commit"; then
  mv "$hooks/pre-commit" "$hooks/pre-commit.before-site-leaks"
  echo "kept the existing hook as pre-commit.before-site-leaks"
fi
cat > "$hooks/pre-commit" <<'EOF'
#!/usr/bin/env bash
# installed by tools/install_hooks.sh: refuse site-specific or confidential text
set -e
root="$(git rev-parse --show-toplevel)"
if [ -f "$root/tools/check_site_leaks.py" ]; then
  python3 "$root/tools/check_site_leaks.py"
else
  echo "pre-commit: this checkout has no tools/check_site_leaks.py yet; leak check skipped" >&2
fi
prev="$(git rev-parse --git-common-dir)/hooks/pre-commit.before-site-leaks"
if [ -x "$prev" ]; then exec "$prev" "$@"; fi
exit 0
EOF
chmod +x "$hooks/pre-commit"
python3 tools/check_site_leaks.py --self-test
echo "pre-commit hook installed in $hooks"
