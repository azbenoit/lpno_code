#!/usr/bin/env python3
"""Refuse a commit that would put site-specific or confidential details into git.

Cluster work needs a host, a user name, an account, scratch paths and the like.
Those live in a local, git-ignored file (`reproducibility/rerun/site.env`, or the file named by
LPNO_SITE_ENV), never in tracked files. This check enforces that, in two ways:

1. Every value in the local site file is a forbidden literal. Whatever you put
   there (host alias, account, user, cluster name, remote paths) cannot be
   committed anywhere in the repository. Add extra words to SITE_DENY in that
   file, comma-separated.
2. Generic patterns that are site-specific by construction: a real path under
   /scratch/ or /home/<name>/, an IPv4 address, a licence endpoint PORT@HOST,
   a private key, an SSH public key, a host-key fingerprint. And the site file
   itself, staged under any name ending in `site.env`.

It scans only what a commit adds (staged lines and new file names), so text
already in the repository is judged when it is next touched. A line that must
carry a pattern on purpose (a documented placeholder) can end with the marker
`site-leak: allow`.

    python3 tools/check_site_leaks.py            # staged changes (the pre-commit hook)
    python3 tools/check_site_leaks.py --all      # every tracked text file
    python3 tools/check_site_leaks.py --self-test
    bash tools/install_hooks.sh                  # run it before every commit

The committed patterns name no site on purpose: a guard that listed the hosts it
protects would itself leak them.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ALLOW_MARKER = "site-leak: allow"
# Placeholders already used in tests and docs; real paths look nothing like these.
PLACEHOLDER = r"(?:user|other|someone_else|job|caches|aiida-pino|<[^>]*>|\$\{?\w+\}?|\.\.\.)"
PATTERNS = [
    ("scratch path", re.compile(r"/scratch/(?!" + PLACEHOLDER + r"(?:/|\b|$))[A-Za-z0-9_.-]+")),
    ("home path", re.compile(r"/home/(?!" + PLACEHOLDER + r"(?:/|\b|$))[a-z_][A-Za-z0-9_.-]*")),
    ("macOS home path", re.compile(r"/Users/(?!" + PLACEHOLDER + r"(?:/|\b|$))[A-Za-z][A-Za-z0-9_.-]*")),
    # loopback, any-address, the RFC 5737 documentation ranges and the console tests'
    # placeholder LAN address are not a site
    ("IPv4 address", re.compile(r"(?<![\w.=~^<>!-])(?!127\.0\.0\.1\b|0\.0\.0\.0\b|10\.0\.0\.5\b"
                                r"|192\.0\.2\.|198\.51\.100\.|203\.0\.113\.)"
                                r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\w.])")),
    ("licence endpoint PORT@HOST", re.compile(
        r"(?<![\w.])\d{2,5}@(?!(?:host|secret-host|hostname|HOST|example(?:\.\w+)*)\b)[A-Za-z][A-Za-z0-9.-]*")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("SSH public key", re.compile(r"\bssh-(?:rsa|ed25519|dss) AAAA")),
    ("host-key fingerprint", re.compile(r"SHA256:[A-Za-z0-9+/]{43}")),
]
SITE_FILE_NAME = re.compile(r"(^|/)([^/]*[._-])?site\.env$")
# Embedded binary data (notebook images), including the first line of one, which follows its JSON key.
BASE64ISH = re.compile(r"^[\s\"']*(?:\"[\w./+-]+\":\s*\")?[A-Za-z0-9+/=]{120,}")
TRIVIAL = {"0", "1", "true", "false", "yes", "no", "none", "gpu", "cpu", "debug", "normal", "master",
           "default", "academic", "gpu:1", "-o BatchMode=yes",
           "build", "long", "test", "main"}


def repo_root() -> Path:
    """The repository this checker belongs to, wherever it is called from."""
    here = Path(__file__).resolve().parent
    out = subprocess.run(["git", "-C", str(here), "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True)
    return Path(out.stdout.strip() or here.parent)


def site_literals(root: Path) -> list[str]:
    """Every value in the local site file(s), plus SITE_DENY words."""
    files = [Path(os.environ["LPNO_SITE_ENV"])] if os.environ.get("LPNO_SITE_ENV") else []
    files += [root / "cluster" / "site.env", Path.home() / ".config" / "lpno" / "site.env"]
    words: set[str] = set()
    for f in files:
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            val = val.split(" #", 1)[0].strip().strip("'\"")
            parts = val.split(",") if key.strip() == "SITE_DENY" else [val]
            for p in parts:
                p = p.strip()
                if len(p) >= 3 and p.lower() not in TRIVIAL and "$" not in p:
                    words.add(p)
    return sorted(words, key=len, reverse=True)


def scan_line(text: str, literals: list[str]) -> list[str]:
    if ALLOW_MARKER in text or BASE64ISH.match(text):
        return []
    hits = []
    low = text.lower()
    for w in literals:
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(w.lower()) + r"(?![A-Za-z0-9])", low):
            hits.append(f"site value '{w}'")
    for name, pat in PATTERNS:
        m = pat.search(text)
        if m:
            hits.append(f"{name} '{m.group(0)}'")
    return hits


def staged() -> tuple[list[tuple[str, int, str]], list[str]]:
    names = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                           capture_output=True, text=True).stdout.split()
    diff = subprocess.run(["git", "diff", "--cached", "-U0", "--no-color", "--diff-filter=ACMR"],
                          capture_output=True, text=True).stdout
    lines, cur, n = [], None, 0
    for ln in diff.splitlines():
        if ln.startswith("+++ "):
            cur = ln[6:] if ln.startswith("+++ b/") else None
        elif ln.startswith("@@"):
            m = re.search(r"\+(\d+)", ln)
            n = int(m.group(1)) if m else 0
        elif ln.startswith("+") and cur:
            lines.append((cur, n, ln[1:]))
            n += 1
    return lines, names


def tracked() -> tuple[list[tuple[str, int, str]], list[str]]:
    names = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split("\n")
    names = [x for x in names if x]
    lines = []
    for f in names:
        p = Path(f)
        try:
            data = p.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:4096]:
            continue
        for i, ln in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            lines.append((f, i, ln))
    return lines, names


def check(lines, names, literals) -> list[str]:
    me = "tools/check_site_leaks.py"
    bad = [f"{f}: the local site file must never be committed (rename it, do not force-add it)"
           for f in names if SITE_FILE_NAME.search(f)]
    for f, n, text in lines:
        if f == me:
            continue
        for h in scan_line(text, literals):
            bad.append(f"{f}:{n}: {h}")
    return bad


def self_test() -> int:
    lit = ["mycluster", "acct-42"]
    must = ["ssh mycluster.example.org", "sbatch --account=acct-42 job.sh", "cd /scratch/myhpc/alice",
            "/home/alice/data", "export F3D_LICENSE=27000@licbox", "ip 10.12.0.5",
            "-----BEGIN OPENSSH PRIVATE KEY-----", "ssh-ed25519 AAAAC3Nza", "SHA256:" + "A" * 43]
    clean = ["nvidia-cublas-cu12==12.8.4.1", "/scratch/job", "/home/user/aiida_run", "/scratch/$USER/x",
             "/home/<user>/x", "http://127.0.0.1:8180", "MyCluster2 is fine", "pip install torch==2.8.0",
             "cd /scratch/myhpc/alice  # site-leak: allow"]
    fails = [s for s in must if not scan_line(s, lit)] + [s for s in clean if scan_line(s, lit)]
    names_bad = [n for n in ["reproducibility/rerun/site.env", "x/prod.site.env"] if not SITE_FILE_NAME.search(n)]
    names_ok = [n for n in ["reproducibility/rerun/site.env.example", "docs/website.envelope"] if SITE_FILE_NAME.search(n)]
    for f in fails + names_bad + names_ok:
        print("self-test FAILED on:", f)
    print("self-test", "passed" if not (fails or names_bad or names_ok) else "FAILED")
    return 1 if (fails or names_bad or names_ok) else 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    root = repo_root()
    os.chdir(root)
    literals = site_literals(root)
    lines, names = tracked() if "--all" in sys.argv else staged()
    bad = check(lines, names, literals)
    if bad:
        print("check_site_leaks: refusing, site-specific or confidential text would be committed:")
        for b in bad[:60]:
            print("  " + b)
        if len(bad) > 60:
            print(f"  ... and {len(bad) - 60} more")
        print("Move the value to the local site file (reproducibility/rerun/site.env, git-ignored) and read it at\n"
              "run time; for a deliberate generic placeholder end the line with '# site-leak: allow'.")
        return 1
    scope = "tracked files" if "--all" in sys.argv else "staged changes"
    print(f"check_site_leaks: clean ({len(lines)} lines of {scope}, {len(literals)} local site values checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
