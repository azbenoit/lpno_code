# Sources for the solver-calibration section

Every source the calibration section cites, with a DOI, an access route, what it
supports, and whether we verified it against the original. They are kept here,
beside the manuscript's `.bib`, so that the *provenance status* travels with the
citation: a number transcribed from a summary and a number read from a source look
identical in LaTeX.

**Rule.** No claim in the section may rest on a source marked `UNVERIFIED`, and any
number from a `PARTIAL` source must carry its caveat in the text.

## Measurement anchors

| source | DOI / locator | access | status | what it supports |
|---|---|---|---|---|
| **Gan Z, Kafka OL, Parab N, et al.** *Universal scaling laws of keyhole stability and porosity in 3D printing of metals.* Nat Commun **12**, 2379 (2021) | `10.1038/s41467-021-22704-0` | open access | **VERIFIED**: 87 (Ke, e/r₀) points derived from Supplementary Data 1, 4 alloy/spot groups | The Ti-6Al-4V keyhole depth ladder (the headline agreement figure); the keyhole-number scaling used to predict cavity depth. **Its `eta` column is a ray-tracing model output, not a measurement**; never cite as experimental absorptance. |
| **Galiègue L, Briffod F, Ito K, Masuo H, Watanabe M, Shiraiwa T, Enoki M.** *Acoustic emission monitoring reveals keyhole formation and evolution.* Sci Rep **15**, 43243 (2025) | `10.1038/s41598-025-27232-1` · PMC12680680 | open access. **`nature.com` redirects automated fetches to an auth page; Europe PMC does not**; use `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12680680/fullTextXML` | **VERIFIED**: full text + Supplementary Data 1 (216 rows) read directly | The transition-regime ladder (depth *and* width) and the defocus spot-size series. r₀ = 60 µm is a **radius**, D4σ/2 of a stated 120 µm D4σ; bare 3 mm plate; depth is the metallographic fusion boundary (XCT was for porosity only). |
| **Vaglio E, et al.** *Single Tracks Data obtained by Selective Laser Melting of Ti6Al4V with a Small Laser Spot Diameter.* Mendeley Data | `10.17632/s9438vb5xd` | open data | **PARTIAL**: depth/width/height tables read from `Allegati/Data.xlsx`; **the 25 µm beam radius and 25 µm powder layer are NOT in the archive** | A slow-speed cross-check only. **Must not be used to score spot-size response** until the radius is sourced from the paper. |
| **Hawk & Simonds**, 304L stationary spot welds (2022) | see `anchors/hawk304l_targets.json` | — | **VERIFIED against its own tables** | SS304L depth and width at 4 powers × 4 material variants, 10 ms dwell, 303 µm spot **diameter**. Depth measured on *etched cross-sections* (fusion). Variants differ up to 2× at identical conditions, so a variant must be named. |
| CoCr cross-section series, *Appl. Sci.* **16**, 5850 | see `anchors/appsci_cocr_targets.json` | — | **VERIFIED against Tables 4 + 6** | CoCr depth and width, EOS M280, r₀ 50 µm, means of 8 cross-sections. |
| **NIST AM-Bench** `mds2-2525` (Ti-6Al-4V absorptance + X-ray), `mds2-2718` (IN718 sections), `mds2-2923` (IN625 slow speed) | NIST AM-Bench portal | open data | absorptance **VERIFIED**; `mds2-2923` **NOT YET USED** | The absorptance diagnostic (50.8 % **gross**) and the IN718 depth/width anchors. |

## Tv1 literature-bracket sources

| source | locator | status | what it supports |
|---|---|---|---|
| **Pantawane MV, Ho Y-H, Joshi SS, Dahotre NB.** Sci Rep **10**, 7579 (2020) | `10.1038/s41598-020-63281-4` · PMC7200685 | **VERIFIED**: full text read via Europe PMC; states verbatim "At the vaporization temperature (3315 K)" for Ti6Al4V | The 3315 K endpoint of the Tv1 bracket, as the value used across LPBF simulation studies |
| **CRC Handbook of Chemistry and Physics**, 97th ed., Haynes ed. | ISBN 978-1-4987-5428-6 | **VERIFIED (standard reference)**: Ti boiling point 3287 °C = 3560 K is the handbook value; not re-read for this entry | The 3560 K pure-titanium endpoint of the bracket |

Note: Boivineau 2006 (Int J Thermophys 27, 507) is often cited for Ti64 thermo-
physical data, but whether it states a boiling point could not be verified (the PDF
was not retrievable), so it is not cited for the bracket.

Note on Vaglio in the manuscript: the 12%-cross-lab-depth-difference claim cites
Vaglio and Galiègue as *scatter evidence* for the acceptance band; neither is scored
there as an anchor. The missing beam radius (the PARTIAL flag above) is part of why the
sources scatter, so it does not undermine that use.

## Open provenance questions

- **Vaglio beam radius.** Needed before that anchor can carry any spot-size claim.
- **Galiègue defocus radii.** The paper gives r₀ = 60/64/69/83/130 µm against defocus
  0/2/3/5/10 mm but does not state how the defocused values were measured, nor
  whether they are also D4σ/2. Affects only the spot-size arm.
- **Galiègue depth datum.** Substrate surface or bead top is not stated.
- **Wavelength and beam profile** are absent from Galiègue entirely; we did not
  substitute vendor values.

## Mesh resolution and refinement

Retrieval ran through Crossref, OpenAlex (incl. full-text),
Semantic Scholar, Unpaywall, Europe PMC, arXiv and direct PDF fetches; every quote
below came from a document actually retrieved. **The method misses paywalled
Elsevier/Wiley content, so the negative results are negatives for the reachable
corpus, not proofs of absence.**

### What cell sizes the field actually uses

Cells per beam radius, computed from each paper's stated numbers (none report the
ratio themselves):

| study | code | cell | beam radius | cells/radius |
|---|---|---|---|---|
| Lai 2024, Nat Commun 15:9546 | FLOW-3D | 10 µm | 21.5 µm | **2.2** |
| Kumar 2026 (preprint) | laserbeamFoam | 10 µm | 36 µm | 3.6 |
| Guo/Leung 2023, IJMTM 184 | FLOW-3D v11.2 | 6 µm | 25 µm | 4.2 |
| **this work** | FLOW-3D WELD | **10 µm** | **50 µm** | **5.0** |
| Zenz & Cook 2025 | FLOW-3D + OpenFOAM | 6.25 µm | 47.5 / 70 µm | 7.6 / 11.2 |
| Khairallah 2016, Acta Mater 108 | ALE3D | 3 µm | 27 µm | 9.0 |
| Wang & Yan 2022, npj Comput Mater 8:22 | in-house VOF | 4 µm | 50 µm | 12.5 |
| Ebrahimi 2022, Mater Des 214 | Fluent | 3 µm | 55 µm | 18.3 |
| Much 2024, AMSES 11:16 | CutFEM + AMR | 1.56 µm | 70 µm | 45 |

**Spread is ~2 to 45 (a factor of 20), with no correlation to venue or year.**
Our 5.0 sits in the low-middle, above three of the studies (Lai 2024 in Nature
Communications, the Kumar 2026 preprint and Guo/Leung 2023 in IJMTM), which supports
10 µm as a normal choice in this field.

### Is there an established criterion? Essentially no

The only explicit numeric criterion found: Ebrahimi et al. 2022, *Mater. Des.*
**214**:110385, DOI `10.1016/j.matdes.2022.110385` (gold OA; preprint
arXiv:2201.02614): *"the numerical grid cell spacing should be chosen to have at
least **35 cells in the melt pool region along its width**"*. The unit is the
melt-pool **width** (neither beam radius nor depth). **Our ~200 µm pools at 10 µm give
~20 cells, so we do not meet it**; state that openly.

Full-text search of OpenAlex for `"cells per beam radius"`, `"cells across the melt
pool"`, `"cells across the laser spot"`, `"elements across the beam diameter"`
returns **zero hits in laser processing**. Papers assert adequacy ("to ensure
numerical accuracy", "after the mesh-sensitivity analysis") without a rule. That
absence is itself citable.

Mechanistic reason the beam-radius ratio matters in ray-traced solvers: Flint et al.
2023, *SoftwareX* **21**:101299, DOI `10.1016/j.softx.2022.101299` (gold OA): ray
count is set by *"the number of computational cells that are within the laser-beam
radius"*.

### Depth converges far more slowly than width, which corroborates our result

Ebrahimi, Kleijn & Richardson 2021, *J. Phys. D* **54**:105304, DOI
`10.1088/1361-6463/abca62`, Appendix A Table A1: refining 60 → 20 → 10 µm moves
melt-pool **width by 0.2 %** and **depth by 11.9 %**. Cite the Table A1 values. The
study is in conduction mode, which the citing text should state.
This independently reproduces the asymmetry we measure (width 4/4 in band at
−12 %; depth 0/4 at −54 %). Khairallah 2016 gives the physical reason: *"melt pool
depth is sensitive to laser absorptivity, whereas the width … depends mostly on beam
size."*

### Precedent for reporting an unconverged production mesh

- **Benoit et al. 2026, arXiv:2602.06241** (our own prior work) already states it:
  *"mesh convergence has not yet been achieved at 10 μm resolution in the keyhole
  regime"*, while the conduction regime *"is effectively achieved at 10 μm"*. Its
  cost table gives 6 min 7 s at 10 µm vs 1 h 6 min 54 s at 5 µm (**10.9×**). Theory
  gives **~16×** (8× cells × 2× CFL steps); a dwell pair here gave 16.17× but also
  differs in recoil model, pv0 and tv1. That the 10.9× is below theory may mean the
  timestep did not halve in that study.
- Kumar et al. 2026 (preprint) run 10 µm for cost and assess convergence by
  comparison with experiment. Our mesh pair compares two resolutions of the same case,
  which is a different test.
- Much et al. 2024, *AMSES* **11**:16, DOI `10.1186/s40323-024-00270-6` (OA), state
  the limits of their own production run directly: *"the results of this simulation
  do not claim high accuracy … the recoil pressure is likely to be very inaccurate."*

### Mechanism for why a coarse grid suppresses the cavity

Much, Koch, Meier & Schreter-Fleischhacker 2026, *CMAME* **457**:119023, DOI
`10.1016/j.cma.2026.119023` (preprint arXiv:2606.19556). A dedicated 2-D
laser-induced vapour-depression benchmark, Ti-6Al-4V, r_L = 70 µm: *"only the result
obtained with the finest spatial resolution … resembles the behaviour of the
reference solution"*, and the mechanism we need: recoil pressure depends
**exponentially** on interface temperature, so under-resolution under-predicts
surface temperature and therefore recoil. Their 1-D benchmark: a 1.1 % temperature
error produced a 16.7 % recoil-pressure error.

**Gap:** no located paper reports the *binary* failure: a coarse
grid producing **no** depression where a finer one produces a shallow one. Much et
al. give monotone degradation plus the mechanism. If we claim the binary behaviour
it is our own finding and must be presented as such.

### FLOW-3D meshing capability

**Multi-block refinement is documented and supported.** Flow Science product page
(https://www.flow3d.com/products/flow-3d/): *"linked, nested, conforming, and/or
partially overlapping mesh blocks. … Nested mesh blocks can be used to enhance the
resolution around an area of interest."* Conforming blocks exist at least since
v12.0. Structured Cartesian + FAVOR, not body-fitted (CFD-101, *Free Gridding*).

In practice for this problem: Ransenigo et al. 2022, *Lasers Manuf. Mater. Process.*
**9**:481, DOI `10.1007/s40516-022-00185-3` (OA): FLOW-3D WELD with 5 µm inner /
25 µm outer static blocks. Zenz & Cook 2025 use 50 µm base refined to 12.5 µm.

**Local refinement around the beam is therefore available in the solver.** The
generator supports it through `mesh.refinement` (one nested fine block, single rank
only); the runs behind the manuscript use one uniform block.

**Moving-source AMR: not documented in any FLOW-3D release v12.0 → 2025R1.** The
search enumerated the product page, the modeling-capabilities index, all release
notes including WELD and AM, and the CFD-101 gridding articles; "adaptive mesh
refinement"/"AMR" appears on none. **No vendor or third-party source states the
absence**, so phrase as *"no adaptive or moving refinement is listed in any release
from v12.0 through 2025R1"* and cite the pages, and never write "FLOW-3D does not
support AMR". A moving mesh block / moving reference frame could not be confirmed or
denied (manual is behind the customer portal). Codes that do have it: Ansys Fluent; Jia et
al. 2024 *CMAME* **419**:116673 (moving thermal-fluid zone); Hu et al. 2017 *IJHMT*
**115**:258 (octree AMR for keyhole welding, paywalled, unverified).
**Do not attribute AMR to laserbeamFoam or ALE3D**: the LLNL AMR reference is to
Diablo, a different code.

### Cost-saving strategies, with verdicts

- **Moving/co-moving window: the one with a quantified depth verdict.** Jia et al.
  2024 (free postprint https://hal.science/hal-04800567/document), Table 4: melt-pool
  **depth identical (0.927 mm) at every box size**, width +1.2 %, length −2.6 % for
  the smallest box; 62 h vs ~3 weeks. The window clips the trailing tail and leaves
  the depth unchanged. FE thermal-fluid welding, not keyhole L-PBF.
- **Local refinement**: used by three FLOW-3D/OpenFOAM studies above; **none
  quantifies the far-field coarsening error.**
- **Reduced domain: cautionary.** NIST AM-Bench 2024 (DOI
  `10.1007/s40192-023-00336-0`, OA): *"At least one participant admitted that
  infinite boundary conditions were used … to increase computational efficiency"*,
  flagged as a possible source of discrepancy.
- **Symmetry planes: nobody in modern keyhole CFD uses them.** Absent from every
  full text retrieved; Much et al. deliberately tilt the beam 5° *"to break
  symmetry"*.
- **Adaptive timestep**: universally a *stability* argument (Co < 0.2–0.25); **no
  paper varies the timestep and reports depth.**

### Precedent for an effective, calibration-dependent parameter

**Zenz, Cook, Vörös & Otto 2025**, *Discover Materials* **5**:266, DOI
`10.1007/s43939-025-00434-0` (OA, CC-BY) is the closest precedent to our position,
and FLOW-3D-specific:
*"Reducing the evaporative recoil factor in the one-fluid model by 70 % gave the best
match to experiment in a deep keyhole scenario. However, such one-fluid calibrations
must be repeated when process conditions alter."*
And directly parallel to our spot-size problem: *"**It is not possible to match both
the 95 and 140 μm spot size simulations to experiment just by varying recoil
pressure.**"*

A calibrated constant shown to be **grid-dependent**: Katagiri et al. 2024, *Results
in Engineering* **24**:103567 (OA): the mushy-zone constant *"should be varied
depending on … the grid size of the simulation"*. And Kidess et al. 2016, *Int. J.
Thermal Sci.* **104**:412, DOI `10.1016/j.ijthermalsci.2016.01.015` (preprint
arXiv:1603.07124): a calibrated enhancement factor reproduces experiment *only* on
the coarse mesh it was fitted on; on a finer mesh the pool is *"much less deep"*.
**Our claim that a tv1 calibrated at 10 µm is resolution-bound is an extension of
Kidess, and should cite it as such.**

Naming precedent: Whalen, Ma & Balachandran 2021, *IMMI* **10**:597, DOI
`10.1007/s40192-021-00238-z` (OA): *"we refer to it as apparent absorptivity."*

Best single framing sentence, Much et al. 2026: *"existing studies consistently show
that quantitatively reliable predictions either require prohibitively high
computational costs or rely on extensive (artificial) calibration of poorly
constrained parameters."*

### Could not be sourced: do not fill from memory

1. No paper states literally "our production mesh is not converged for keyhole
   depth" **except our own** arXiv:2602.06241.
2. No source for a coarse grid producing **no** depression where a fine grid produces
   a shallow one (our binary observation).
3. No source stating a calibrated absorptivity must be **recalibrated when the mesh
   changes**; Kidess demonstrates the adjacent case.
4. No quantified error for a symmetry plane in keyhole CFD.
5. No timestep-sensitivity study of keyhole depth.
6. No domain-truncation study reporting keyhole depth.
7. No statement that FLOW-3D lacks AMR; only documented absence.

Three paywalled papers are summarised in entries 20-22 below: Bayat et al. 2019
*Addit. Manuf.* 30:100835 (FLOW-3D Ti64 keyhole mesh numbers); Cook & Murphy 2020
*Addit. Manuf.* 31:100909 (the standard review, for the criterion question); Ross et
al. 2022 *Addit. Manuf.* 60:103267 (effective heat-source framing).

**Preprints, not to be cited as peer-reviewed:** arXiv:2606.19556 (Much 2026; the
journal version is CMAME 457:119023), arXiv:2604.07359 (Kumar), arXiv:2602.06241
(Benoit, ours).

## References not yet in the manuscript bibliography (22 papers)

Checked against the 67 DOIs in the manuscript's bibliography.
Ransenigo (`10.1007/s40516-022-00185-3`) and Flint (`10.1016/j.softx.2022.101299`)
are already there and are omitted below.

`[R]` = retrieved and quoted during the search. `[U]` = unread, bibliographic
record only; do not cite its content until someone opens it.

### Open access

| # | citation | link | brings |
|---|---|---|---|
| 1 | Gan Z et al. 2021, *Nat Commun* 12:2379 | https://doi.org/10.1038/s41467-021-22704-0 | `[R]` Our keyhole ladder anchor and the Ke scaling law. **Its η column is a model output, not a measurement.** |
| 2 | Galiègue L et al. 2025, *Sci Rep* 15:43243 | https://doi.org/10.1038/s41598-025-27232-1 · full text via https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12680680/fullTextXML | `[R]` The transition-regime ladder (depth **and** width) and the defocus spot-size series. |
| 3 | Lai Y et al. 2024, *Nat Commun* 15:9546 | https://doi.org/10.1038/s41467-024-53888-w | `[R]` FLOW-3D at **10 µm with a 21.5 µm beam radius (2.2 cells/radius)** over an 8 mm track; the strongest evidence our 10 µm is not a lax choice. |
| 4 | Wang & Yan 2022, *npj Comput Mater* 8:22 | https://doi.org/10.1038/s41524-022-00699-6 | `[R]` 12.5 cells/radius yet **15 % under-prediction** of the deepest keyhole; high resolution is not sufficient for depth. |
| 5 | Ebrahimi A et al. 2021, *J Phys D* 54:105304 | https://doi.org/10.1088/1361-6463/abca62 | `[R]` **Table A1: width converges to 0.2 %, depth moves 11.9 %.** Independent corroboration of our width-passes/depth-fails asymmetry. |
| 6 | Ebrahimi A et al. 2022, *Mater Des* 214:110385 | https://doi.org/10.1016/j.matdes.2022.110385 · preprint https://arxiv.org/pdf/2201.02614 | `[R]` **The only explicit resolution criterion in the field**: ≥35 cells across melt-pool width. We give ~20 and do not meet it. |
| 7 | Much N et al. 2024, *AMSES* 11:16 | https://doi.org/10.1186/s40323-024-00270-6 | `[R]` Precedent for declaring a production run under-resolved; calibration compensating for model-form error. |
| 8 | Zenz C, Cook P, Vörös P, Otto A 2025, *Discover Materials* 5:266 | https://doi.org/10.1007/s43939-025-00434-0 | `[R]` **The closest precedent to our position, and FLOW-3D-specific**: recoil calibration cannot match two spot sizes at once. |
| 9 | Katagiri J et al. 2024, *Results in Engineering* 24:103567 | https://doi.org/10.1016/j.rineng.2024.103567 | `[R]` A calibrated constant shown to be **grid-size dependent**; Table 1 is a ready-made survey of L-PBF grid sizes. |
| 10 | Simonds B et al. 2024, *IMMI* 13:175 (NIST AM-Bench) | https://doi.org/10.1007/s40192-023-00336-0 | `[R]` Community calibration data; the truncated-domain cautionary quote. |
| 11 | Whalen S, Ma X, Balachandran P 2021, *IMMI* 10:597 | https://doi.org/10.1007/s40192-021-00238-z · https://www.osti.gov/servlets/purl/1976761 | `[R]` Naming precedent: *"apparent absorptivity"*. |
| 12 | Vaglio E et al. 2020, Mendeley Data | https://doi.org/10.17632/s9438vb5xd | `[R]` Slow-speed cross-check. **Beam radius not in the archive: PARTIAL provenance.** |

### Version of record paywalled, free copy listed alongside

Cite the DOI (the peer-reviewed version of record); the second link is only for
reading access. Both are given so Zotero can hold the published record.

| # | citation | version of record (DOI) | free copy | brings |
|---|---|---|---|---|
| 13 | Khairallah S, Anderson A, Rubenchik A, King W 2016, *Acta Mater* 108:36 | https://doi.org/10.1016/j.actamat.2016.02.014 | https://arxiv.org/pdf/1512.02593 | 9 cells/radius; *"melt pool depth is sensitive to absorptivity, width depends mostly on beam size"*; absorptivity 0.35 as a calibrated constant. |
| 14 | Guo Q et al. 2023, *Int J Mach Tools Manuf* 184:103977 | https://doi.org/10.1016/j.ijmachtools.2022.103977 | https://discovery.ucl.ac.uk/10159796/1/Leung_revised%20Understanding%20keyhole%20induced.pdf | FLOW-3D v11.2 at 6 µm / 4.2 cells per radius. |
| 15 | Kidess A, Kenjeres S, Righolt B, Kleijn C 2016, *Int J Therm Sci* 104:412 | https://doi.org/10.1016/j.ijthermalsci.2016.01.015 | https://arxiv.org/abs/1603.07124 | **A calibrated factor reproduces experiment only on the mesh it was fitted on**; our tv1-is-resolution-bound claim extends this. |
| 16 | Jia X et al. 2024, *Comput Methods Appl Mech Eng* 419:116673 | https://doi.org/10.1016/j.cma.2023.116673 | https://hal.science/hal-04800567/document | The only quantified depth verdict for a moving window: depth identical at every box size, 62 h vs 3 weeks. |
| 17 | Much N et al. 2026, *Comput Methods Appl Mech Eng* 457:119023 | https://doi.org/10.1016/j.cma.2026.119023 | https://arxiv.org/abs/2606.19556 | **The mechanism**: recoil exponential in interface temperature; 1.1 % temperature error → 16.7 % recoil error. |
| 6 | Ebrahimi A et al. 2022, *Mater Des* 214:110385 (listed above as OA) | https://doi.org/10.1016/j.matdes.2022.110385 | https://arxiv.org/pdf/2201.02614 | The ≥35-cells-across-melt-pool-width criterion. |

### Preprints with no version of record

Not yet peer-reviewed as far as we can establish: cite as preprints, and check for a
published version before submission.

| # | citation | link | brings |
|---|---|---|---|
| 18 | Kumar et al. 2026 | https://arxiv.org/abs/2604.07359 | 10 µm chosen for cost; convergence asserted against experiment rather than a refined mesh. |
| 19 | Benoit A et al. 2026 (**ours**) | https://arxiv.org/abs/2602.06241 | Our own prior statement that convergence is not achieved at 10 µm in the keyhole regime; cost table 10.9×. |

### Paywalled papers, read from the published PDFs

Two of these three change the argument.

**20. Bayat M, Thanki A, Mohanty S, Witvrouw A, Yang S, Thorborg J, Tiedje NS,
Hattel JH (2019)**, *Keyhole-induced porosities in Laser-based Powder Bed Fusion
(L-PBF) of Ti6Al4V: High-fidelity modelling and experimental validation*,
*Addit. Manuf.* 30:100835. https://doi.org/10.1016/j.addma.2019.100835

The closest prior work to ours that exists: **same solver (FLOW-3D), same alloy
(Ti-6Al-4V), same regime (keyhole), same journal.** What it reports:

| quantity | Bayat 2019 | ours |
|---|---|---|
| cell size | **3.3 µm** ("average size of 3.3 μm with an aspect ratio of very close to one, which is ideal for surface tracking algorithms such as VOF") | 10 µm |
| cells | **6,288,368** | 0.23 M (`spot60`) |
| beam radius | `rb (1/e²)` = **25 µm** (Table 1) | 60 µm |
| **cells per beam radius** | **7.6** | **5.0** |
| cost | **145 h on a 14-core machine** for one single track | 0.6 h |
| mesh independence study | **none** | 2-point, quantified |

Two things follow. First, 7.6 against our 5.0 cells per beam radius is the same order,
within the same class of model: their finer cell goes into a proportionally smaller
beam, so their beam is resolved to the same order as ours. Second, the paper reports
results on a single mesh, with no mesh-independence study and no numerical-error
estimate. Their 145 h per track, on a different machine and case, is consistent with
our compute-cost argument.

**21. Cook PS & Murphy AB (2020)**, *Simulation of melt pool behaviour during additive
manufacturing: Underlying physics and progress*, *Addit. Manuf.* 31:100909.
https://doi.org/10.1016/j.addma.2019.100909

This answers the resolution-criterion question. The standard review of the field,
in a section headed *"weaknesses in overall scientific approach"*, states verbatim:

> "Another aspect absent from virtually all publications is any discussion of the mesh
> sensitivity or convergence criteria of their modelling results. Instead, results for a
> single mesh size are routinely presented with no indication of numerical error."

The sourced claim is therefore: **the field's standard review reports that no such
criterion is applied, and that single-mesh results without error estimates are the
norm.** Our 2-point study adds a quantified two-resolution comparison to that
practice. Bayat 2019 above also reports a single mesh.

The same passage also supports the reproducibility framing of `validation/`:

> "Very few authors present a complete description of their model setup, omitting a few
> key items of information regarding boundary conditions, model dimensions or process
> parameters, meaning that reproduction of their work is not possible."

Cook & Murphy also name the NIST AM-Bench series as the corrective initiative, which is
the dataset we already anchor to (entry 10).

**22. Ross AJ, Bitharas I, Perkins KG, Moore AJ (2022)**, *Volumetric heat source
calibration for laser powder bed fusion*, *Addit. Manuf.* 60:103267.
https://doi.org/10.1016/j.addma.2022.103267

A conduction-only model with a double-ellipsoid source whose parameters are fitted by
an inverse heat conduction problem against melt-pool solidification boundaries. It is
a conduction model with no CFD keyhole, so it sets no resolution precedent. Its value is
as the precedent for *calibrating an effective parameter instead of resolving the
physics that produces it*, which is what our tv1/absorptivity calibration does:

> "...enable a heat source to be selected for conduction-only models that incorporates
> experimentally calibrated effects such as the laser absorption and the penetration of
> the vapour depression into the melt pool, **which are computationally expensive to
> calculate from first principles**."

Fitted values for SS316L at 200 W, d = 50 µm: absorptivity **η = 0.73**, depth
2σ_z = **93 µm**, rear/front length ratio 8.85. The η = 0.73 bears on our own
absorptivity choice: it is a *fitted effective* value, high because it absorbs
multiple-reflection and depression effects the conduction model cannot represent, and it
is reported as such. It belongs in the same category as our tv1: a calibrated stand-in
for a quantity nobody measured.

Also unread, lower priority: Hu C et al. 2017, *IJHMT* 115:258,
https://doi.org/10.1016/j.ijheatmasstransfer.2017.07.061 `[U]`: octree AMR for
keyhole welding, title only, would support the "other codes have AMR" contrast.
Kennedy M & O'Hagan A 2001, *JRSS-B* 63:425,
https://doi.org/10.1111/1467-9868.00294 `[U]`: canonical
calibration/model-discrepancy confounding reference, if the section needs a
statistics anchor.

## Access notes

- Papers and datasets are kept outside the repository (licences differ; one archive
  is 105 MB). Point `F3D_SOURCES_DIR` at wherever they live. Only the extracted
  numeric targets are tracked, in `validation/anchors/`.
- Extraction procedure and the two xlsx parsing traps: `validation/anchors/SOURCES.md`.

### NIST AM-Bench `mds2-2525`: "Spot on Bare Metal" (stationary pulse), VERIFIED

Second configuration of the same record that supplies the scanned case
(`t2_nist195`). Read from the source: `2525_README_v200.txt` and
`Spot on Bare Metal_Calibrated Absorption Data.csv`, both downloaded to
`lpno/data/experimental/spot/`.

| quantity | value | source |
|---|---|---|
| laser power | **101.9 W** (plateau median; mean 101.6, sd 3.3) | CSV `InputLaser`, largest contiguous on-run |
| pulse duration | **2.008 ms** (t = 0 to +2.0085 ms) | CSV, same run; README says "nominally 2.0 ms" |
| beam spot | **122.5 um 1/e^2 diameter** at the sample, i.e. **r0 = 61.25 um** | README: measured at focus, sample 2.8 mm below focal plane |
| substrate | Ti-6Al-4V, NIST SRM 654b | README |
| camera | **50 kfps, 19.987 us frames**, 227 frames; frame 80 at t = -52.48 us | CSV `FrameNumber`/`Time` |
| absorptance at onset | **30.5 %** (first 50 us) | CSV `RelativeAbsorption` |
| absorptance, pulse median | 53.2 % | same |
| radiograph pixel size | **2.01 um/px**, FOV 1286 x 1206 um | transferred from the scanned run; see below |
| surface tilt in frame | **-6.16 deg** (straight-line fit, sd **0.34 px** over all 640 columns) | fitted on the pre-pulse mean |
| **keyhole onset** | **867 us** (depth) / **890 us** (absorptance) | two independent channels, below |

Three things worth carrying forward:

* **The beam radius is identical to the scanned case** (61.25 um 1/e^2), and the
  cold-surface absorptance is also 30.5 %, the same figure the scan starts from.
  So the spot and the scan differ in motion and power and share their optics, which
  makes the pair a clean test of whether the drilling-transient deficit is
  geometry-specific.
* **The scan's radiograph scale cannot be re-derived here, so it was transferred
  and the transfer was measured.** 2.01 um/px came from beam travel (0.7 m/s
  against the frame interval), and a stationary pulse has no travel. Neither the
  README nor the captioned images carry a pixel size or a scale bar; the only
  dimension in `Sample Diagram.pdf` is the 300 um sample thickness, which lies
  along the X-ray path and is therefore invisible in the image plane. What
  settles it is that the record ships both representations of the scan, so the
  two can be compared directly: the distributed "processed" images are a
  **mirrored crop of the raw at identical sampling**, with no rebinning. Measured on
  the same frames, the cavity travels **-7.26 px/frame in the raw against
  +7.94 px/frame in the processed** (ratio 0.91; a 2x rebin would give 0.5, and
  the sign flip is the mirror). The spot frames are distributed raw only, at
  640 x 600 (a taller, narrower sensor ROI than the scan's 1024 x 400, at unchanged
  magnification), so **2.01 um/px applies to them**.
  Everything downstream scales linearly with this number; if it is ever revised,
  every depth in the spot comparison moves with it.
* **The keyhole switches on abruptly at ~870 us, and two independent channels
  agree on when.** The cavity first exceeds 20 um at **867 us** in the
  radiographs; the absorbed fraction sits flat at **0.331** from 0 to 810 us,
  then half-rises at **890 us** and climbs to **0.64** by the end of the pulse.
  One frame apart. Before onset the surface shows only a dark lens straddling
  the surface line (displaced melt piled above the original surface, adding
  material to the X-ray path) and no sub-surface cavity at all. This makes the
  spot a sharper test than the scan: it dates an event as well as a depth.

`RelativeAbsorption` reaches 182 % in places, which is unphysical and reflects
division by a near-zero input during the ramp; any absorptance comparison must
window to the plateau and ignore the raw maximum. The binned recomputation
in `spot_radiographs.absorptance` does this.

**There is no measured Ti-6Al-4V melt-pool depth or width anywhere in this
record.** Every tabulated geometry result is `Al_*`:
`Al_Spot_TDW_Results.csv` (melt width), `Al_Spot_AA_ASR_Results.csv`
(solidification rate), `Al_Scan_AA_MWD_ASR_Results.csv` (max depth and width).
The README is explicit that "all A-AMB2022-01 challenge data was obtained from
experiments on solid aluminum". The Ti64 half of the record is absorptance plus
raw X-ray images.

Consequences, both of which bind:

* The spot's usable measured quantities are the **vapour-depression depth** (read
  from the radiographs) and the **absorbed fraction**. A simulated fusion depth
  has nothing in this record to score against.
* The melt extent of 200-240 um quoted from the flank-excess decay
  (`spot_radiographs`, and a parked melt-line analysis) is
  **our own extraction with no measured counterpart**, so it must not be printed
  as a measurement. It serves only as a plausibility check on the simulation.
  A metallographic boundary and "where a few-percent transmission excess decays
  into noise" are different surfaces, and only the former is what this campaign
  scores fusion against everywhere else.

**Where a real number might exist, UNVERIFIED:** the record's own supplement
paper is Simonds et al. 2021, *Appl. Mater. Today* **23**, 101049, "The causal
relationship between melt pool geometry and energy absorption measured in real
time during laser-based manufacturing" (`10.1016/j.apmt.2021.101049`). The title
implies measured Ti64 melt-pool geometry from these very radiographs. It has not
been retrieved or checked (Elsevier; the Europe PMC route that worked for
Galiegue does not apply). Until someone reads it, the spot has no fusion anchor;
do not assume the paper supplies one.

Reader: `lpno/src/notebooks/spot_radiographs.py`.
