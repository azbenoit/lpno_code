#!/usr/bin/env python3
"""Download the two Zenodo records of the LP-NO data deposit into this repository.

The open record (doi:10.5281/zenodo.23325086, CC BY-NC-SA 4.0) holds the AiiDA archives with every trained model,
evaluation and simulation configuration, the provenance logs, the published measurements and the web demo's model
weights. The restricted-access record (doi:10.5281/zenodo.23325088, same licence) holds the FLOW-3D WELD output (the
simulated fields) and, optionally, the web demo's ground truth. Access to it is granted on request from its Zenodo
page; once granted, download with a personal access token of the Zenodo account that was granted access
(created under Account > Applications > Personal access tokens on zenodo.org). Python 3.8 or newer.

    python data_deposit/scripts/download.py open                      # -> data_deposit/open_record/
    python data_deposit/scripts/download.py open --only lpno_provenance.aiida provenance_log.csv
    python data_deposit/scripts/download.py restricted --token-file ~/.config/zenodo/token --join
    python data_deposit/scripts/download.py restricted --only payloads_ti64_rand150_newphys --join
    python data_deposit/scripts/download.py open --list               # list the files and sizes only

`--only` keeps files whose name starts with one of the given prefixes. Files already present with the right size
and MD5 are skipped, so an interrupted download can be restarted. `--join` rejoins every `<name>.tarNN`-style part
set (`<name>.part00`, `<name>.part01`, ...) into `<name>` and checks it against the record's SHA256SUMS;
`--remove-parts` then deletes the parts. The token can also come from the ZENODO_TOKEN environment variable, and
is never printed. Only the Python standard library is used.

The notebooks in reproducibility/ and the web demo find the records in the default folders without further setup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://zenodo.org/api/records"
RECORDS = {"open": "23325086", "restricted": "23325088"}
DEPOSIT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = {"open": DEPOSIT / "open_record", "restricted": DEPOSIT / "restricted_record"}
PART = re.compile(r"^(?P<base>.+)\.part(?P<n>\d{2,})$")


def _token(args: argparse.Namespace) -> str | None:
    if args.token_file:
        return Path(args.token_file).expanduser().read_text().strip()
    return os.environ.get("ZENODO_TOKEN") or None


def _request(url: str, token: str | None) -> urllib.request.Request:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def list_files(record: str, token: str | None, draft: bool) -> list[dict]:
    url = f"{API}/{record}/draft/files" if draft else f"{API}/{record}/files"
    try:
        with urllib.request.urlopen(_request(url, token), timeout=120) as r:  # noqa: S310
            body = json.load(r)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404):
            raise SystemExit(
                f"record {record}: HTTP {exc.code}. The restricted record needs access granted on its Zenodo page and "
                f"a personal access token of that account (--token-file or ZENODO_TOKEN); a record that is not yet "
                f"published is visible only to its owners (--draft with an owner's token).") from exc
        raise
    return [e for e in body.get("entries", []) if e.get("status", "completed") == "completed"]


def md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - Zenodo's own checksum, used for integrity only
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(entry: dict, dest: Path, token: str | None) -> None:
    out = dest / entry["key"]
    want_md5 = str(entry.get("checksum", ""))
    want_md5 = want_md5[4:] if want_md5.startswith("md5:") else ""
    if out.is_file() and out.stat().st_size == entry.get("size") and (not want_md5 or md5(out) == want_md5):
        print(f"  have     {entry['key']}")
        return
    tmp = out.with_name(out.name + ".partial")
    size = entry.get("size") or 0
    print(f"  download {entry['key']} ({size / 1e9:.2f} GB)", flush=True)
    with urllib.request.urlopen(_request(entry["links"]["content"], token), timeout=600) as r, \
            open(tmp, "wb") as fh:  # noqa: S310
        done, step = 0, max(size // 20, 1 << 26)
        mark = step
        for chunk in iter(lambda: r.read(1 << 22), b""):
            fh.write(chunk)
            done += len(chunk)
            if size and done >= mark:
                print(f"           {100 * done / size:5.1f} %", flush=True)
                mark += step
    if want_md5 and md5(tmp) != want_md5:
        tmp.unlink()
        raise SystemExit(f"MD5 mismatch for {entry['key']}: the download was removed, run the command again")
    tmp.replace(out)


def join_parts(dest: Path, remove: bool) -> None:
    groups: dict[str, list[Path]] = {}
    for p in dest.iterdir():
        m = PART.match(p.name)
        if m:
            groups.setdefault(m["base"], []).append(p)
    sums = {}
    if (dest / "SHA256SUMS").is_file():
        for line in (dest / "SHA256SUMS").read_text().splitlines():
            if "  " in line:
                digest, name = line.split("  ", 1)
                sums[name.strip().lstrip("*")] = digest
    for base, parts in sorted(groups.items()):
        parts.sort(key=lambda p: int(PART.match(p.name)["n"]))
        target = dest / base
        print(f"  join     {base} from {len(parts)} parts", flush=True)
        with open(target, "wb") as out:
            for p in parts:
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 22), b""):
                        out.write(chunk)
        if base in sums:
            if sha256(target) != sums[base]:
                raise SystemExit(f"SHA256 mismatch for the joined {base}; the parts are kept")
            print(f"           sha256 ok")
        else:
            print(f"           (no SHA256SUMS line for {base}; not checked)")
        if remove:
            for p in parts:
                p.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", choices=sorted(RECORDS), help="which record to download")
    ap.add_argument("--dest", help="target folder (default data_deposit/<record>_record/)")
    ap.add_argument("--only", nargs="+", metavar="PREFIX", help="download only files whose name starts with these")
    ap.add_argument("--list", action="store_true", help="list the record's files and stop")
    ap.add_argument("--token-file", help="file holding a Zenodo personal access token")
    ap.add_argument("--draft", action="store_true", help="read an unpublished draft (owners only)")
    ap.add_argument("--join", action="store_true", help="rejoin .partNN sets and check them against SHA256SUMS")
    ap.add_argument("--remove-parts", action="store_true", help="with --join: delete the parts after a good join")
    args = ap.parse_args()

    token = _token(args)
    record = RECORDS[args.record]
    entries = list_files(record, token, args.draft)
    if args.only:
        entries = [e for e in entries if any(e["key"].startswith(p) for p in args.only)
                   or e["key"] == "SHA256SUMS" and args.join]
    total = sum(e.get("size") or 0 for e in entries)
    print(f"record {record} ({args.record}): {len(entries)} files, {total / 1e9:.2f} GB")
    if args.list:
        for e in entries:
            print(f"  {e['key']}  {(e.get('size') or 0) / 1e9:.3f} GB")
        return
    dest = Path(args.dest).expanduser() if args.dest else DEFAULT_DEST[args.record]
    dest.mkdir(parents=True, exist_ok=True)
    for e in entries:
        fetch(e, dest, token)
    if args.join:
        join_parts(dest, args.remove_parts)
    print(f"done: {dest}")


if __name__ == "__main__":
    sys.exit(main())
