# Experimental anchors

Every published measurement the calibration is scored against, tracked here so
that the targets behind every validation claim are under version control.

## Reading an anchor file

Two fields decide whether a comparison is meaningful at all:

- **`depth_kind`**: `fusion` (cross-section metallography: the melted boundary)
  or `depression` (X-ray or similar: the vapour cavity). These are different
  physical surfaces and the fusion depth is always the larger. Comparing a
  simulated depression against a metallographic fusion depth inflates the
  apparent gap (in one case from ~2× to "4×"), so the scorer matches kinds
  before it compares numbers.
- **`beam.spot_convention`**: radius vs diameter, and 1/e², D4σ or FWHM. At
  fixed power the spot size is the *dominant* variable, so an unresolved
  convention makes a scored depth meaningless. `UNRESOLVED` means exactly that:
  do not quote a verdict from that anchor.

`source.provenance_verified: false` means the values were transcribed from a
literature search and not yet checked against a quoted sentence in the source.
A verdict from an unverified anchor is provisional.

## Two anchor files not distributed

`galiegue2025_ti64_targets.json` and `hawk304l_targets.json` are read by the scorer and the validation notebooks
but are not in this repository, because their sources' licences do not permit redistributing their values:

| file | source | licence |
|---|---|---|
| `galiegue2025_ti64_targets.json` | Galiègue et al., Acoustic emission monitoring reveals keyhole formation and evolution, *Sci. Rep.* **15**, 43243 (2025), doi:10.1038/s41598-025-27232-1, Supplementary Information 1 | CC BY-NC-ND 4.0, which withholds permission to share adapted material |
| `hawk304l_targets.json` | Hawk et al., Laser spot welding of additive manufactured 304L stainless steel, *Weld. World* **66**, 895-906 (2022), doi:10.1007/s40194-022-01265-w, Table 5 | subscription article (© International Institute of Welding 2022), no licence grants redistribution |

To use them, build each file locally from the publication and place it here: a JSON object with `source`
(citation), `depth_kind` (`fusion` for both) and a `points` list, one entry per condition with `power_w`,
`speed_m_s`, `r0_m`, `depth_um` and `width_um` (the schema `score_anchors.normalize_points` reads; the
`conditions` schema of `appsci_cocr_targets.json` works too). Both names are in
`.gitignore`, so a local copy is never committed. `SOURCES.md` lists the exact table and file each one comes
from and the conventions the scorer expects (beam radius r0 = 60 µm, half the D4σ diameter, for Galiègue; a
top-hat of 303 µm diameter for Hawk).

## Raw data not tracked here

Too large for the repo; re-fetchable from the cited sources:

| data | size | where |
|---|---|---|
| NIST AMB2018-02 Ti64 scan absorptance | 6.4 MB CSV | NIST mds2-2525 |
| Ti64 X-ray radiographs | 8.3 MB TIFF | NIST mds2-2525 |

## Known cross-anchor conflict

At the radii quoted in `galiegue2025_ti64_targets.json` and
`vaglio2020_ti64_targets.json`, Vaglio's condition carries 6.4× the intensity of
Galiègue's and reports the smaller depth (110.9 vs 128.2 µm), where higher intensity
would be expected to give a deeper pool. Both anchors are kept.
