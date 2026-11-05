#!/usr/bin/env python3
"""Detect corrupted NPZ exports, and write the sidecars that make detection cheap.

Why this exists (2026-08-05): an eval on `ti64_random150` died with
`zipfile.BadZipFile: Bad CRC-32 for file 'Temperature.npy'` -- on-disk corruption
in a case NPZ. Nothing had detected it, because `<case>.manifest.json` records the
FLOW-3D run config (case_name, material, parallel, run) and carries no checksum.
The only way to notice was to decompress the whole archive and wait for a reader
to trip over it, which is how it surfaced: days later, as a crashed eval.

No kernel I/O errors were logged, so the leading cause is a write that hit ENOSPC
-- the root filesystem had filled to 100% during object-store packing, and a zip
member truncated mid-write fails CRC exactly this way.

Two modes, because existing data has no baseline to compare against:

  --write    compute sha256 and write `<file>.sha256` beside each NPZ.
  (default)  verify. With a sidecar, compare sha256 -- catches any change,
             including a corrupt member nothing has read yet. Without one, fall
             back to a zip CRC test, which needs no baseline and is what catches
             the failure above on data exported before sidecars existed.

Corruption does NOT silently poison results: numpy verifies CRC on read, so a bad
member raises rather than returning wrong numbers. The risk this closes is the
member nothing happens to read, plus the days of wall clock spent discovering it.

Usage:
    python3 python/verify_npz_checksums.py DATA_DIR [DATA_DIR ...]
    python3 python/verify_npz_checksums.py DATA_DIR --write
    python3 python/verify_npz_checksums.py FILE.npz --quiet

Exit status is 1 if anything failed, so it can gate a batch.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

SIDECAR_SUFFIX = ".sha256"
_READ_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """Streamed, so a 78 MB case NPZ does not land in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_path(npz_path: Path) -> Path:
    return npz_path.with_suffix(npz_path.suffix + SIDECAR_SUFFIX)


def write_sidecar(npz_path: Path) -> str:
    """Record the digest of a file that is good *now*.

    Deliberately does not CRC-check first: writing a sidecar for an already
    corrupt file would freeze the corruption in as the expected state. Verify
    before writing sidecars over data of unknown provenance.
    """
    digest = sha256_file(npz_path)
    sidecar_path(npz_path).write_text(f"{digest}  {npz_path.name}\n")
    return digest


def read_sidecar(npz_path: Path) -> str | None:
    side = sidecar_path(npz_path)
    if not side.exists():
        return None
    text = side.read_text().strip()
    return text.split()[0] if text else None


def check_file(npz_path: Path) -> tuple[bool, str]:
    """(ok, reason). Uses the sidecar when present, else a CRC test."""
    expected = read_sidecar(npz_path)
    if expected is not None:
        actual = sha256_file(npz_path)
        if actual == expected:
            return True, "sha256 ok"
        return False, f"sha256 mismatch: expected {expected[:12]}..., got {actual[:12]}..."
    try:
        with zipfile.ZipFile(npz_path) as archive:
            bad_member = archive.testzip()
    except zipfile.BadZipFile as exc:
        return False, f"BadZipFile: {exc}"
    except Exception as exc:  # noqa: BLE001 - report, do not mask, an unreadable file
        return False, f"{type(exc).__name__}: {exc}"
    if bad_member is not None:
        return False, f"bad CRC in member {bad_member!r}"
    return True, "crc ok (no sidecar)"


def iter_npz(targets: list[Path]) -> list[Path]:
    found: list[Path] = []
    for target in targets:
        if target.is_dir():
            found.extend(sorted(target.rglob("*.npz")))
        elif target.suffix == ".npz":
            found.append(target)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="NPZ files or directories to scan")
    parser.add_argument("--write", action="store_true",
                        help="write .sha256 sidecars instead of verifying")
    parser.add_argument("--quiet", action="store_true", help="only report failures")
    args = parser.parse_args(argv)

    files = iter_npz(args.paths)
    if not files:
        print("no .npz files found", file=sys.stderr)
        return 1

    failures: list[tuple[Path, str]] = []
    for index, npz_path in enumerate(files, 1):
        if args.write:
            digest = write_sidecar(npz_path)
            if not args.quiet:
                print(f"[{index}/{len(files)}] wrote {sidecar_path(npz_path).name} {digest[:12]}...")
            continue
        ok, reason = check_file(npz_path)
        if not ok:
            failures.append((npz_path, reason))
            print(f"FAIL {npz_path}: {reason}")
        elif not args.quiet:
            print(f"[{index}/{len(files)}] ok {npz_path.name} ({reason})")

    if args.write:
        print(f"wrote {len(files)} sidecars")
        return 0
    print(f"checked {len(files)} files, {len(failures)} failed")
    for path, reason in failures:
        print(f"  {path}: {reason}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
