"""Split an AiiDA archive into an open copy and a restricted bundle, and scrub site-specific strings.

How the two deposit records were made from the exported archives. FLOW-3D WELD output moves to the
restricted bundle: field arrays (`.npz`, `.vti` and similar: evaluation previews and dumps, training
snapshots, all carrying solver fields) and the solver's console logs (`runhyd_tail.log`, and `aiida.out`
of `pino.flow3d_run` jobs). Everything else stays in the open archive. Workstation paths, host names, the
scheduler job owner and the computer's address are replaced by neutral values, in the database and in
every text object; TensorBoard event files are rewritten record by record with fresh checksums. Each node
that lost files gets an `lpno_restricted_files` extra listing them (path, object sha256, reason), and
`_aiida_hash` is dropped from every node whose content changed.

usage: python data_deposit/scripts/split_restricted.py <in.aiida> <out-dir> <tag> <local scrub map .json>
writes <out-dir>/<stem>.aiida, <out-dir>/restricted/restricted_files_<tag>.tar and .csv, and
<out-dir>/split_<tag>_report.json. The input archive is only read.
"""
import csv
import hashlib
import io
import json
import os
import re
import sqlite3
import struct
import sys
import tarfile
import time
import zipfile

SRC, OUT, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
WORK = os.path.join(OUT, f"work_{TAG}")
os.makedirs(WORK, exist_ok=True)
os.makedirs(os.path.join(OUT, "restricted"), exist_ok=True)

# ---- rules -------------------------------------------------------------------------------------
# The site-specific strings to replace live in a local JSON file that is never committed:
#   {"text": [["<old>", "<new>"], ...], "user": "<OS user name>"}
# Text objects and database columns get every pair, then the bare user name becomes "user". Binary
# event files get only the pairs of equal length (protobuf strings carry their length).
_MAP = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else {"text": [], "user": None}
TEXT_REPL = [tuple(p) for p in _MAP["text"]]
BARE_USER = re.compile(rf"\b{re.escape(_MAP['user'])}\b") if _MAP.get("user") else None
BIN_REPL = [(a.encode(), b.encode()) for a, b in TEXT_REPL if len(a) == len(b)]
FIELD_EXT = (".npz", ".npy", ".vti", ".vtk", ".vtu", ".h5", ".hdf5")
SOLVER_LOGS = ("runhyd_tail.log",)
BINARY_EXT = (".pth", ".png", ".npz", ".npy", ".vti", ".h5", ".hdf5", ".pt", ".ckpt")
_ALT = [re.escape(a) for a, _ in TEXT_REPL] + ([BARE_USER.pattern] if BARE_USER else [])
LEAK = re.compile("|".join(_ALT)) if _ALT else re.compile(r"(?!)")


def scrub_text(s):
    for a, b in TEXT_REPL:
        s = s.replace(a, b)
    return BARE_USER.sub("user", s) if BARE_USER else s


def move_reason(path, proc):
    leaf = path.split("/")[-1]
    if leaf.endswith(FIELD_EXT):
        return "FLOW-3D field data"
    if leaf in SOLVER_LOGS:
        return "FLOW-3D solver log"
    if leaf == "aiida.out" and proc == "pino.flow3d_run":
        return "FLOW-3D solver log"
    return None


# ---- TFRecord (TensorBoard event file) rewrite --------------------------------------------------
_T = []
for i in range(256):
    c = i
    for _ in range(8):
        c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
    _T.append(c)


def crc32c(b):
    c = 0xFFFFFFFF
    for x in b:
        c = _T[(c ^ x) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def masked(c):
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def rewrite_tfrecords(raw):
    out, pos, changed = bytearray(), 0, 0
    while pos < len(raw):
        (n,) = struct.unpack_from("<Q", raw, pos)
        data = raw[pos + 12: pos + 12 + n]
        new = data
        for a, b in BIN_REPL:
            new = new.replace(a, b)
        if new != data:
            assert len(new) == len(data)
            changed += 1
            out += raw[pos: pos + 12] + new + struct.pack("<I", masked(crc32c(new)))
        else:
            out += raw[pos: pos + 12 + n + 4]
        pos += 12 + n + 4
    assert pos == len(raw), "trailing bytes in event file"
    return bytes(out), changed


# ---- read the archive ---------------------------------------------------------------------------
t0 = time.time()
zin = zipfile.ZipFile(SRC)
names = set(zin.namelist())
dbfile = os.path.join(WORK, "db.sqlite3")
with zin.open("db.sqlite3") as f, open(dbfile, "wb") as g:
    while chunk := f.read(1 << 22):
        g.write(chunk)
meta_json = scrub_text(zin.read("metadata.json").decode())
db = sqlite3.connect(dbfile)

creator = {}
for out_id, ptype in db.execute("SELECT l.output_id, n.process_type FROM db_dblink l JOIN db_dbnode n "
                                "ON n.id = l.input_id WHERE l.type IN ('create', 'return')"):
    creator.setdefault(out_id, (ptype or "").split(":")[-1])

new_objects = {}      # new key -> bytes
rewritten = {}        # old key -> new key (memo, content-addressed)
moved_rows = []       # manifest rows
restricted_objs = {}  # stored key -> bytes source (old key, needs_scrub)
report = {"moved_files": 0, "moved_bytes": 0, "rewritten_text": 0, "rewritten_tfevents_records": 0,
          "nodes_changed": 0, "unknown_binary": {}}


def put(content):
    k = hashlib.sha256(content).hexdigest()
    new_objects[k] = content
    return k


def process_tree(tree, path, proc, moved):
    out = {}
    for name, sub in (tree.get("o") or {}).items():
        p = f"{path}/{name}" if path else name
        if "k" not in sub:
            out[name] = process_tree(sub, p, proc, moved)
            continue
        k = sub["k"]
        why = move_reason(p, proc)
        if why:
            moved.append((p, k, why))
            continue
        if k not in rewritten:
            leaf = name
            nk = k
            if leaf.startswith("events.out.tfevents"):
                new, nrec = rewrite_tfrecords(zin.read(f"repo/{k}"))
                if nrec:
                    nk = put(new)
                    report["rewritten_tfevents_records"] += nrec
            elif not leaf.endswith(BINARY_EXT):
                raw = zin.read(f"repo/{k}")
                try:
                    s = raw.decode("utf8")
                except UnicodeDecodeError:
                    s = None
                    ext = os.path.splitext(leaf)[1] or leaf
                    report["unknown_binary"][ext] = report["unknown_binary"].get(ext, 0) + 1
                if s is not None:
                    t = scrub_text(s)
                    if t != s:
                        nk = put(t.encode("utf8"))
                        report["rewritten_text"] += 1
            rewritten[k] = nk
        out[name] = {"k": rewritten[k]}
    return {"o": out} if out or "o" in tree else tree


rows = db.execute("SELECT id, uuid, node_type, process_type, repository_metadata, extras FROM db_dbnode").fetchall()
for nid, uuid, ntype, ptype, meta, extras in rows:
    if not meta:
        continue
    tree = json.loads(meta)
    proc = (ptype or "").split(":")[-1] if ptype else creator.get(nid, "-")
    moved = []
    new_tree = process_tree(tree, "", proc, moved)
    ex = json.loads(extras or "{}")
    changed = new_tree != tree
    if moved:
        listing = []
        for p, k, why in moved:
            size = zin.getinfo(f"repo/{k}").file_size
            if why == "FLOW-3D solver log":
                raw = zin.read(f"repo/{k}")
                stored = hashlib.sha256(scrub_text(raw.decode("utf8", "replace")).encode("utf8")).hexdigest()
                restricted_objs.setdefault(stored, (k, True))
            else:
                stored = k
                restricted_objs.setdefault(stored, (k, False))
            listing.append({"path": p, "object": stored, "reason": why})
            moved_rows.append([TAG, uuid, (ntype or "").rsplit(".", 2)[-2], proc, p, stored, size, why])
            report["moved_files"] += 1
            report["moved_bytes"] += size
        ex["lpno_restricted_files"] = listing
    if changed:
        ex.pop("_aiida_hash", None)
        report["nodes_changed"] += 1
        db.execute("UPDATE db_dbnode SET repository_metadata = ?, extras = ? WHERE id = ?",
                   (json.dumps(new_tree), json.dumps(ex), nid))
print(f"[{time.time() - t0:.0f}s] trees processed: {report}", flush=True)

# ---- scrub every text column, drop stale hashes where attributes changed ------------------------
tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
for t in tables:
    cols = [r[1] for r in db.execute(f"PRAGMA table_info({t})")]
    pk = "rowid"
    for c in cols:
        upd = []
        for rid, v in db.execute(f"SELECT {pk}, {c} FROM {t}"):
            if isinstance(v, str) and LEAK.search(v):
                upd.append((scrub_text(v), rid))
        if upd:
            db.executemany(f"UPDATE {t} SET {c} = ? WHERE {pk} = ?", upd)
            print(f"  scrubbed {t}.{c}: {len(upd)} rows", flush=True)
            if t == "db_dbnode" and c == "attributes":
                for _, rid in upd:
                    (e,) = db.execute("SELECT extras FROM db_dbnode WHERE rowid = ?", (rid,)).fetchone()
                    ej = json.loads(e or "{}")
                    if ej.pop("_aiida_hash", None) is not None:
                        db.execute("UPDATE db_dbnode SET extras = ? WHERE rowid = ?", (json.dumps(ej), rid))
db.commit()
db.execute("VACUUM")
db.close()

# ---- what stays, what goes ----------------------------------------------------------------------
db = sqlite3.connect(dbfile)
referenced = set()


def keys(tree):
    for sub in (tree.get("o") or {}).values():
        if "k" in sub:
            yield sub["k"]
        else:
            yield from keys(sub)


for (meta,) in db.execute("SELECT repository_metadata FROM db_dbnode"):
    referenced.update(keys(json.loads(meta or "{}")))
db.close()
in_zip = {n[5:] for n in names if n.startswith("repo/") and len(n) > 5}
still_moved = {old for old, _ in restricted_objs.values()} & referenced
assert not still_moved, f"moved content still referenced by kept paths: {len(still_moved)}"
to_delete = sorted(in_zip - referenced)
to_add = sorted(k for k in referenced if k not in in_zip)
assert set(to_add) <= set(new_objects), "referenced key with no content"
report.update(repo_objects_before=len(in_zip), deleted=len(to_delete), added=len(to_add))
print(f"[{time.time() - t0:.0f}s] objects: {len(in_zip)} before, {len(to_delete)} dropped, {len(to_add)} added",
      flush=True)

# ---- restricted bundle --------------------------------------------------------------------------
rdir = os.path.join(OUT, "restricted")
with open(os.path.join(rdir, f"restricted_files_{TAG}.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["archive", "node_uuid", "node_type", "process", "path_in_node", "object_sha256", "bytes", "reason"])
    w.writerows(moved_rows)
with tarfile.open(os.path.join(rdir, f"restricted_files_{TAG}.tar"), "w") as tar:
    for stored, (old, scrub) in sorted(restricted_objs.items()):
        raw = zin.read(f"repo/{old}")
        if scrub:
            raw = scrub_text(raw.decode("utf8", "replace")).encode("utf8")
        assert hashlib.sha256(raw).hexdigest() == stored
        ti = tarfile.TarInfo(f"objects/{stored}")
        ti.size, ti.mtime, ti.mode = len(raw), 0, 0o644
        tar.addfile(ti, io.BytesIO(raw))
print(f"[{time.time() - t0:.0f}s] restricted bundle written: {len(restricted_objs)} objects", flush=True)

# ---- open archive: metadata and database first (AiiDA's reader looks for them among the first four
# entries), then every kept object, recompressed --------------------------------------------------
stem = os.path.basename(SRC)
dst = os.path.join(OUT, stem)
keep = sorted(k for k in referenced if k in in_zip)
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.writestr("metadata.json", meta_json)
    z.write(dbfile, "db.sqlite3")
    z.writestr(zipfile.ZipInfo("repo/"), b"")
    for k in keep:
        with zin.open(f"repo/{k}") as f, z.open(f"repo/{k}", "w", force_zip64=True) as g:
            while chunk := f.read(1 << 24):
                g.write(chunk)
    for k in to_add:
        z.writestr(f"repo/{k}", new_objects[k])
report["seconds"] = round(time.time() - t0)
with open(os.path.join(OUT, f"split_{TAG}_report.json"), "w") as f:
    json.dump(report, f, indent=1)
print(f"[{time.time() - t0:.0f}s] done: {dst}", flush=True)
