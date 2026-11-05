# Where each anchor's numbers came from, and how to regenerate them

The published PDFs and dataset archives are not committed: their
licences differ (Scientific Reports is CC-BY, Mendeley Data sets carry their own
terms), one archive is 105 MB, and none of it is needed. Measured values in a
table are facts; the extracted JSON in this directory is the redistributable part.

Put downloaded sources anywhere outside the repo and point `F3D_SOURCES_DIR` at
it. Nothing in the repo depends on them at run time; they are only needed to
re-derive an anchor file or to check a quote.

| anchor file | source | how to obtain |
|---|---|---|
| `galiegue2025_ti64_targets.json` | Galiègue et al., *Sci Rep* **15**:43243 (2025), DOI `10.1038/s41598-025-27232-1` | not in the repository: CC BY-NC-ND 4.0 (no adapted material may be shared); build it locally (README.md). Full text: `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12680680/fullTextXML`. Supplementary: `.../PMC12680680/supplementaryFiles` → `41598_2025_27232_MOESM1_ESM.xlsx` (216 rows). Note `www.nature.com` redirects to an auth page for automated fetches; Europe PMC does not. |
| `vaglio2020_ti64_targets.json` | Vaglio et al., Mendeley Data, DOI `10.17632/s9438vb5xd` | download the dataset zip; it nests a second zip containing `Allegati/Data.xlsx` with sheets *Track width (W)* / *Track depth (D)* / *Track height (H)* / *Contact angle* / *Microhardness*. |
| `gan_points.json` | Gan et al. 2021 universal keyhole scaling, dataset SD1 | see `python/data/gan_scaling_reference.json` for the derived curve. |
| `hawk304l_targets.json` | Hawk et al., *Weld. World* **66**, 895-906 (2022), doi:10.1007/s40194-022-01265-w: 304L spot welds (Table 5) and coupling efficiencies (Tables 3, 4) | not in the repository: subscription article (© IIW 2022), no licence grants redistribution; build it locally (README.md). The accepted manuscript is readable at https://www.osti.gov/biblio/1868245 (tables A-1, A-2, 3). |
| `amb718_targets.json` | NIST AM-Bench `mds2-2718`, IN718 cross-sections | NIST AM-Bench portal. |
| `appsci_cocr_targets.json`, `wang_cocr_targets.json` | *Appl. Sci.* 16:5850 Table 6 (CoCr, EOS M280, r₀ 50 µm, means of 8 cross-sections); Wang et al. CoCr | journal. |

## Regenerating the two slow-speed anchors

Both xlsx files are read with the standard library only (`zipfile` + `xml.etree`):
an xlsx is a zip of XML, so no `openpyxl` is required, which matters because the `pino`
env does not have it.

Two parsing traps, each of which yields wrong numbers without an error:

1. **Empty cells are absent from the XML, not blank.** Iterating `<c>` elements
   in order silently shifts every later column left. Map each cell through its
   `r` attribute (`"N29"` → column 13) instead. Getting this wrong moves
   *Meltpool Depth* into the *Keyhole Frequency* column for exactly the rows
   where frequency was not measured.
2. **Rows with no melt carry `0`**, not an empty string; filter them out before
   computing an aspect ratio or a log-space fit.

Sketch:

```python
import re, zipfile, xml.etree.ElementTree as ET
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
z = zipfile.ZipFile(xlsx_path)
shared = ["".join(t.text or "" for t in si.iter(NS + "t"))
          for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(NS + "si")]

def col(ref):                      # "N29" -> 13
    n = 0
    for ch in re.match(r"([A-Z]+)", ref).group(1):
        n = n * 26 + ord(ch) - 64
    return n - 1

for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(NS + "row"):
    cells = {}
    for c in row.iter(NS + "c"):
        v = c.find(NS + "v")
        if v is None or v.text is None:
            continue
        cells[col(c.get("r"))] = shared[int(v.text)] if c.get("t") == "s" else v.text
```

## Provenance status

`source.provenance_verified` in each file is the authority:

- `true`: every field was read from the source or quoted from it. Only
  `galiegue2025_ti64_targets.json` currently qualifies.
- `"PARTIAL"`: the measured values were read from the source but some setup
  metadata was not. `vaglio2020_ti64_targets.json`: depths/widths/heights come
  from `Data.xlsx`, but the **25 µm beam radius and the 25 µm powder-layer
  thickness are not in the archive** and remain unsourced. Its geometry matching
  Galiègue's verified 2.4×-larger spot to 2% is unexplained until that radius is
  confirmed, so this anchor must not be used to score spot-size response.
- `false`: transcribed from a summary, not yet checked. A verdict from one of
  these is provisional and must say so.
