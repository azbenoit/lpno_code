#!/usr/bin/env python3
"""Copy the deposit's documentation from data_deposit/ into a record directory, and refresh the
SHA256SUMS lines of exactly the files it copies (authors' tool, used when the records are updated).

The deposit has two records: the open one (the AiiDA archives with the models, evaluations and
configurations) and the restricted-access one (the FLOW-3D WELD output). Their README files, the
provenance logs, the restricted-file manifests and the data licence (`LICENSE_data.txt`, copied
as `LICENSE.txt`) are versioned in data_deposit/; the record directories are not under version control. A record is
recognised by what it holds: `lpno_provenance.aiida` (open) or `payloads_*.tar` (restricted). Every other line
of SHA256SUMS is kept as it is and in its order, so the big files are never re-hashed; a new file's
line is appended, and the line of a file no longer in the directory is dropped.

    python data_deposit/scripts/sync.py <record dir>          # shows what would change
    python data_deposit/scripts/sync.py <record dir> --yes    # copies and rewrites SHA256SUMS
"""
import hashlib
import shutil
import sys
from pathlib import Path

DEPOSIT = Path(__file__).resolve().parents[1]
DOCS = DEPOSIT / "record_docs"
LOGS = DEPOSIT / "provenance_logs"
LISTS = DEPOSIT / "restricted_file_lists"
SHARED = {
    "files_description.md": DOCS / "files_description.md",
    "restricted_files_main.csv": LISTS / "restricted_files_main.csv",
    "restricted_files_rounds.csv": LISTS / "restricted_files_rounds.csv",
    "restricted_files_evals.csv": LISTS / "restricted_files_evals.csv",
    "LICENSE.txt": DOCS / "LICENSE_data.txt",
}
RECORDS = {
    "open": {
        "README.md": DOCS / "README_open_record.md",
        "README_AiiDA.md": DOCS / "README_AiiDA.md",
        "provenance_log.csv": LOGS / "provenance_log.csv",
        "provenance_log_rounds.csv": LOGS / "provenance_log_rounds.csv",
        "provenance_log_evals.csv": LOGS / "provenance_log_evals.csv",
        **SHARED,
    },
    "restricted": {
        "README.md": DOCS / "README_restricted_record.md",
        **SHARED,
    },
}


# The open record also carries the published measurements of reproducibility/experimental_data as one zip, built
# here with fixed timestamps and order so the same folder always gives the same bytes.
EXPERIMENTAL = DEPOSIT.parent / "reproducibility" / "experimental_data"


def experimental_zip(dest: Path) -> None:
    import zipfile
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(p for p in EXPERIMENTAL.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(str(Path("experimental_data") / f.relative_to(EXPERIMENTAL)), (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, f.read_bytes())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_of(target: Path):
    if (target / "lpno_provenance.aiida").is_file():
        return "open"
    if any(target.glob("payloads_*.tar")):
        return "restricted"
    return None


def main() -> int:
    target = Path(sys.argv[1]).expanduser()
    apply = "--yes" in sys.argv[2:]
    record = record_of(target)
    if record is None:
        sys.exit(f"{target} holds neither lpno_provenance.aiida nor payloads_*.tar: not a deposit record")
    print(f"{target}: {record} record")
    sums_path = target / "SHA256SUMS"
    sums = {}
    if sums_path.is_file():
        for line in sums_path.read_text().splitlines():
            digest, name = line.split(None, 1)
            sums[name.strip()] = digest
    for name in [n for n in sums if not (target / n).exists()]:
        print(f"{name}: no longer in the directory, line dropped")
        del sums[name]
    for name, src in RECORDS[record].items():
        new = sha256(src)
        old = sha256(target / name) if (target / name).exists() else None
        print(f"{name}: {'unchanged' if new == old else ('added' if old is None else 'changed')}")
        if apply and new != old:
            shutil.copy2(src, target / name)
        sums[name] = new
    if record == "open":
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            built = Path(tmp) / "experimental_data.zip"
            experimental_zip(built)
            new = sha256(built)
            old = sha256(target / built.name) if (target / built.name).exists() else None
            print(f"{built.name}: {'unchanged' if new == old else ('added' if old is None else 'changed')}")
            if apply and new != old:
                shutil.copy2(built, target / built.name)
            sums[built.name] = new
    unlisted = sorted(p.name for p in target.iterdir() if p.is_file() and p.name != "SHA256SUMS"
                      and p.name not in sums)
    if unlisted:
        print(f"not in SHA256SUMS (add their lines by hand, they are not hashed here): {unlisted}")
    if apply:
        sums_path.write_text("".join(f"{digest}  {name}\n" for name, digest in sums.items()))
        print(f"SHA256SUMS: {len(sums)} entries")
    else:
        print("dry run; pass --yes to copy and rewrite SHA256SUMS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
