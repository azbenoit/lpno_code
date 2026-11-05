#!/usr/bin/env python3
"""Fetch the published experimental data used by experimental_validation.ipynb.

Downloads into ../../data/experimental/ (gitignored -- the notebook reads from
there, nothing large is committed). All sources are open:

  NIST A-AMB2022-01  DOI 10.18434/mds2-2525  Ti64 bare-plate scan:
      calibrated absorptance trace + 50 kfps X-ray radiographs
  NIST AM Bench AMB2022-03  DOI 10.18434/mds2-2718  IN718 single tracks:
      melt-pool cross-section measurements
  Gan et al. 2021, Nat Commun 12:2379 (open access) Supplementary Data 1:
      keyhole depth / length / front-wall angle for Ti64, SS316, Al6061

Re-derived aggregates (amb718_targets.json, gan_points.json) are written next to
the raw files so the notebook never re-parses spreadsheets.
"""

from __future__ import annotations

import io
import json
import statistics as st
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

DEST = Path(__file__).resolve().parents[2] / "data" / "experimental"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
AMB718 = ("https://data.nist.gov/od/ds/ark:/88434/mds2-2718/"
          "AMB2022-718-SH1-MeltPool_Cross-Section_Measurement_Results.xlsx")
TI64_ABS = ("https://data.nist.gov/od/ds/ark:/88434/mds2-2525/"
            "Scan%20on%20Bare%20Metal_Calibrated%20Absorption%20Data.csv")
TI64_XRAY = ("https://data.nist.gov/od/ds/ark:/88434/mds2-2525/"
             "Scan%20on%20Bare%20Metal_XrayImages_processed_captioned.zip")
GAN_SD1 = ("https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-021-22704-0/"
           "MediaObjects/41467_2021_22704_MOESM4_ESM.xlsx")


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as fh:
        return fh.read()


def sheet_rows(blob: bytes) -> list[dict]:
    """Read sheet1 of an xlsx into {column_letter: value} rows (stdlib only)."""
    z = zipfile.ZipFile(io.BytesIO(blob))
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        shared = ["".join(t.text or "" for t in si.iter(NS + "t"))
                  for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(NS + "si")]
    name = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))[0]
    rows = []
    for row in ET.fromstring(z.read(name)).iter(NS + "row"):
        d = {}
        for c in row.iter(NS + "c"):
            v = c.find(NS + "v")
            col = "".join(ch for ch in c.get("r") if ch.isalpha())
            d[col] = "" if v is None else (shared[int(v.text)] if c.get("t") == "s" else v.text)
        rows.append(d)
    return rows


def num(d: dict, key: str):
    try:
        return float(d.get(key, ""))
    except ValueError:
        return None


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)

    # --- IN718 cross-sections -> per-condition aggregates -------------------
    rows = sheet_rows(get(AMB718))
    agg = defaultdict(lambda: {"d": [], "w": []})
    for r in rows[1:]:
        V, P, D = num(r, "E"), num(r, "F"), num(r, "G")
        d, w = num(r, "H"), num(r, "I")
        if None in (V, P, D, d, w):
            continue
        # Column D is "Case and Line No.". Only rows labelled "Line ..." are the
        # isolated single tracks; the bare-numbered cases (1..47) are pads and
        # multi-line scans that share P/V/spot but have neighbour remelting, so
        # lumping them in corrupts the single-track width (n=192 instead of 6).
        if not str(r.get("D", "")).strip().lower().startswith("line"):
            continue
        agg[(P, V, D)]["d"].append(d)
        agg[(P, V, D)]["w"].append(w)
    targets = {
        str(k): {"n": len(v["d"]),
                 "depth_um": round(st.mean(v["d"]), 1), "depth_sd": round(st.stdev(v["d"]), 1),
                 "width_um": round(st.mean(v["w"]), 1), "width_sd": round(st.stdev(v["w"]), 1)}
        for k, v in sorted(agg.items()) if len(v["d"]) > 1
    }
    (DEST / "amb718_targets.json").write_text(json.dumps(targets, indent=1))
    print(f"AMB2022-03 IN718: {len(targets)} conditions")

    # --- Gan keyhole points ------------------------------------------------
    rows = sheet_rows(get(GAN_SD1))
    label, pts = None, []
    for r in rows[1:]:
        if r.get("A"):
            label = r["A"]
        P = num(r, "B")
        if P is None:
            continue
        pts.append(dict(group=label, P=P, eta=num(r, "C"), V=num(r, "D"), r0=num(r, "F"),
                        depth_um=(num(r, "O") or 0) * 1e6,
                        length_um=num(r, "P"), tan_theta=num(r, "Q")))
    (DEST / "gan_points.json").write_text(json.dumps(pts, indent=1))
    print(f"Gan 2021: {len(pts)} keyhole points")

    # --- Ti64 absorptance trace + radiographs -------------------------------
    (DEST / "nist_ti64_scan_absorption.csv").write_bytes(get(TI64_ABS))
    print("NIST Ti64 absorptance trace saved")
    xray = DEST / "xray"
    xray.mkdir(exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(get(TI64_XRAY)))
    names = sorted(n for n in z.namelist() if n.endswith(".tif"))
    for n in names[::5]:                      # every 5th frame is plenty for figures
        (xray / Path(n).name).write_bytes(z.read(n))
    print(f"NIST radiographs: {len(names[::5])} of {len(names)} frames")


if __name__ == "__main__":
    main()
