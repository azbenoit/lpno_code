#!/usr/bin/env python
"""Fetch, verify and (for the authors) bundle the demo's served assets.

The served set lives under data/ and is gitignored except for the small tracked files (meta.json,
flow3D_params.yaml, grid.npz, READMEs). It comes in two parts:

* the model weights (required): `lpno_web_demo_models.tar` in the open Zenodo record (doi:10.5281/zenodo.23325086),
  every served checkpoint with its Hydra config;
* the ground truth (optional): `lpno_web_demo_ground_truth.tar` in the restricted-access record
  (doi:10.5281/zenodo.23325088), the FLOW-3D fields of the held-out cases the pages compare against. Access is granted
  on request from the record's Zenodo page. Without it the demo predicts at any power, speed or drawn path, and the
  ground-truth controls are greyed out.

    python scripts/assets.py fetch zenodo                                   # weights from the open record
    python scripts/assets.py fetch zenodo-ground-truth --token-file <file>  # optional, restricted record
    python scripts/assets.py fetch /path/or/URL/to/<bundle>.tar             # a local or other copy
    python scripts/assets.py verify                                         # what is installed

    python scripts/assets.py bundle --part models|ground_truth              # authors: build a bundle

What is bundled and verified is read from the same settings the server reads (app.config.Settings), so it cannot
drift from the code. The neighbour-graph cache (outputs/rigno_graphs) is rebuilt on the first request and is not
part of either bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tarfile
import urllib.request
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DOWNLOAD_PY = REPO.parents[1] / "data_deposit" / "scripts" / "download.py"
ZENODO = {"zenodo": ("open", "lpno_web_demo_models.tar"),
          "zenodo-ground-truth": ("restricted", "lpno_web_demo_ground_truth.tar")}


def _settings():
    from app.config import Settings

    return Settings()


def _path_gt_dirs(s) -> list[Path]:
    """The random-path ground truth is per model (physics.presets_dir under path_data_dir)."""
    from app.model_registry import discover_models

    out = []
    for m in discover_models(s.path_models_dir):
        sub = ((m.get("physics") or {}).get("presets_dir") or "").strip()
        if sub and Path(s.path_data_dir) / sub not in out:
            out.append(Path(s.path_data_dir) / sub)
    return out


def served_roots(part: str) -> list[Path]:
    s = _settings()
    if part == "models":
        return [Path(s.steady_models_dir), Path(s.straight_models_dir), Path(s.path_models_dir)]
    return [Path(s.steady_gt_data_dir), Path(s.straight_data_dir)] + _path_gt_dirs(s)


def served_files(part: str) -> list[Path]:
    """Every file of one part, relative to the repo root. The model part carries only what git does not."""
    out: list[Path] = []
    for r in served_roots(part):
        full = REPO / r
        if not full.exists():
            print(f"  missing (not bundled): {r}", file=sys.stderr)
            continue
        for f in sorted(full.rglob("*")):
            if not f.is_file() or "__pycache__" in f.parts:
                continue
            if part == "models" and f.name in ("meta.json", "flow3D_params.yaml", "grid.npz", ".gitkeep"):
                continue  # tracked in git
            out.append(f.relative_to(REPO))
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_bundle(args: argparse.Namespace) -> None:
    name = "models" if args.part == "models" else "ground_truth"
    out = Path(args.out or f"lpno_web_demo_{name}.tar")
    files = served_files(args.part)
    total = sum((REPO / f).stat().st_size for f in files)
    print(f"bundling {len(files)} files, {total / 2**30:.2f} GiB -> {out}")
    manifest = io.StringIO()
    with tarfile.open(out, "w") as tar:
        for f in files:
            tar.add(REPO / f, arcname=str(f))
            manifest.write(f"{sha256(REPO / f)}  {f}\n")
    Path(str(out) + ".sha256").write_text(manifest.getvalue())
    print(f"wrote {out} ({out.stat().st_size / 2**30:.2f} GiB) and {out}.sha256 ({date.today():%Y-%m-%d})")


def _zenodo_download(which: str, token_file: str | None) -> Path:
    """Download one bundle and its .sha256 from a Zenodo record with data_deposit/scripts/download.py."""
    spec = importlib.util.spec_from_file_location("lpno_download", DOWNLOAD_PY)
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    record_kind, tar_name = ZENODO[which]
    token = Path(token_file).expanduser().read_text().strip() if token_file else None
    entries = [e for e in dl.list_files(dl.RECORDS[record_kind], token, draft=False)
               if e["key"] in (tar_name, tar_name + ".sha256")]
    if not any(e["key"] == tar_name for e in entries):
        raise SystemExit(f"{tar_name} is not in the {record_kind} record")
    dest = REPO / "data" / "_downloads"
    dest.mkdir(parents=True, exist_ok=True)
    for e in entries:
        dl.fetch(e, dest, token)
    return dest / tar_name


def cmd_fetch(args: argparse.Namespace) -> None:
    src = args.source
    if src in ZENODO:
        local = _zenodo_download(src, args.token_file)
    elif src.startswith(("http://", "https://")):
        local = REPO / "data" / "_downloads" / Path(src).name
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {src} -> {local}")
        with urllib.request.urlopen(src) as r, open(local, "wb") as fh:  # noqa: S310
            shutil.copyfileobj(r, fh, length=1 << 20)
        try:
            with urllib.request.urlopen(src + ".sha256") as r:  # noqa: S310
                Path(str(local) + ".sha256").write_bytes(r.read())
        except Exception:  # noqa: BLE001
            print("  (no .sha256 next to the tar; skipping checksum verification)")
    else:
        local = Path(src).expanduser()
    if not local.is_file():
        raise SystemExit(f"not a file: {local}")
    print(f"extracting {local} into {REPO}")
    with tarfile.open(local) as tar:
        for m in tar.getmembers():
            if m.name.startswith(("/", "..")) or ".." in Path(m.name).parts or not m.name.startswith("data/"):
                raise SystemExit(f"refusing unexpected member path in tar: {m.name}")
            if not (m.isfile() or m.isdir()):
                raise SystemExit(f"refusing non-regular member in tar: {m.name}")
        tar.extractall(REPO)
    sums = Path(str(local) + ".sha256")
    if sums.is_file():
        bad = 0
        for line in sums.read_text().splitlines():
            digest, rel = line.split("  ", 1)
            if sha256(REPO / rel) != digest:
                print(f"  CHECKSUM MISMATCH: {rel}")
                bad += 1
        print("checksums ok" if not bad else f"{bad} checksum mismatches")
        if bad:
            raise SystemExit(1)
    cmd_verify(args)


def cmd_verify(_args: argparse.Namespace) -> None:
    from app.model_registry import discover_models

    s = _settings()
    ok = True
    print("served models (required):")
    for page, mdir in (("steady", s.steady_models_dir), ("straight", s.straight_models_dir),
                       ("path", s.path_models_dir)):
        for m in discover_models(mdir):
            folder = REPO / mdir / m["id"]
            has_ckpt = (folder / "u_network.0.pth").is_file()
            has_cfg = (folder / ".hydra" / "config.yaml").is_file()
            flag = "ok " if (has_ckpt and has_cfg) else "MISSING"
            ok &= has_ckpt and has_cfg
            meta = json.loads((folder / "meta.json").read_text())
            print(f"  {flag} {page:8s} {m['id']:14s} {m.get('label', ''):6s} {meta.get('model_name', '')}")
    print("ground truth (optional; comparison controls are greyed out without it):")
    gts = [("steady", Path(s.steady_gt_data_dir)), ("straight", Path(s.straight_data_dir))]
    gts += [(f"path/{d.name}", d) for d in _path_gt_dirs(s)]
    have_gt = {}
    for page, d in gts:
        n = sum(1 for _ in (REPO / d).rglob("*.npz")) if (REPO / d).is_dir() else 0
        have_gt[page.split("/")[0]] = have_gt.get(page.split("/")[0], False) or n > 0
        print(f"  {'ok ' if n else 'not installed'} {page:14s} {d} ({n} npz)")
    # the time pages take their grid from the ground truth when it is installed, else from data/<page>/grid.npz
    for page, mdir in (("straight", s.straight_models_dir), ("path", s.path_models_dir)):
        grid = REPO / Path(mdir).parent / "grid.npz"
        if not have_gt.get(page) and not grid.is_file():
            print(f"  MISSING {page}: no ground truth and no {grid.relative_to(REPO)}")
            ok = False
    if not ok:
        raise SystemExit("assets incomplete -- run `python scripts/assets.py fetch zenodo`")
    print("all required assets present")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle", help="tar one part of the served assets (authors)")
    b.add_argument("--part", choices=("models", "ground_truth"), required=True)
    b.add_argument("--out", help="output tar path (default lpno_web_demo_<part>.tar)")
    f = sub.add_parser("fetch", help="download or copy a bundle and extract it into this checkout")
    f.add_argument("source", help="zenodo, zenodo-ground-truth, or a local path / http(s) URL of a .tar")
    f.add_argument("--token-file", help="Zenodo personal access token (restricted record only)")
    sub.add_parser("verify", help="check the served models (and report the optional ground truth)")
    args = ap.parse_args()
    {"bundle": cmd_bundle, "fetch": cmd_fetch, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
