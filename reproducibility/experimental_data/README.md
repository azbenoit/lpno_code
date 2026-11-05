# Published experimental measurements used in the LP-NO manuscript

This folder holds the published measurements that the manuscript compares its simulations and surrogate
models against: the solver validation (Supplementary Note `sec:calibration`, summarized in the main text in
`sec:expval_main` and `fig:calib-materials`) and the comparison of the surrogate with experiment
(`sec:closing`, `fig:closing`, `tab:closing-conditions`). Labels in this file are the LaTeX labels of the
manuscript.

Every value here was measured by the authors of the cited source, except the vapor-depression depth series
in `nist_mds2-2525/`, which the manuscript authors segmented from the NIST radiographs (stated where it
applies). No simulated or model-derived value is included. Each section states whether its values were read
from a downloaded file or transcribed from a table.

## Conventions shared by all files

- CSV: comma-separated, UTF-8, one header row, no index column, one row per measurement. An empty cell
  means the source reports no value.
- Units are in the column names: `W`, `m_per_s`, `um` (micrometre), `J`, `ms`, `us` (microsecond), `pct`,
  `kHz`.
- `used_in_manuscript` names the figure, table or section labels where the row is used, separated by `; `.
  Empty means the row was extracted with its source but is not used. `sec:calibration (text)` means the value
  is quoted or used in the text of that section.
- Two different lengths are called "depth". **Melt-pool depth** (fusion depth) is the depth of the melted
  and resolidified region, measured on metallographic cross-sections after the track solidified. **Vapor-depression
  depth** (keyhole depth) is the depth of the vapor cavity, measured on in-situ X-ray images during melting.
  The manuscript compares each only with its own kind.
- Beam size is kept in each source's own convention (radius or diameter, and its definition where the source
  gives one). The manuscript matches simulations on the beam radius r0 and takes r0 as half the D4σ diameter
  (four times the standard deviation of the intensity profile, equal to the 1/e² diameter for a Gaussian
  beam). The conversion used for each source is stated in its section.

## Files

| file | source | rows | content |
|---|---|---|---|
| `gan2021_keyhole_depth.csv` | Gan et al. 2021 | 87 | vapor-depression depth, Ti-6Al-4V, SS316, Al6061 |
| `nist_amb2022-03_in718_conditions.csv` | NIST AM Bench AMB2022-03 | 7 | IN718 melt-pool depth and width, condition means |
| `nist_amb2022-03_in718_sections.csv` | NIST AM Bench AMB2022-03 | 42 | the individual cross-sections behind those means |
| `soares2026_cocr_single_tracks.csv` | Soares et al. 2026 | 10 | CoCr melt-pool depth and width |
| `vaglio2020_ti64_single_tracks.csv` | Vaglio et al. 2020 | 80 | Ti-6Al-4V single-track depth, width, height |
| `nist_mds2-2525/scan_absorptance.csv` | NIST mds2-2525 | 49096 | absorptance against time, scanned track |
| `nist_mds2-2525/spot_absorptance.csv` | NIST mds2-2525 | 50217 | absorptance against time, stationary spot |
| `nist_mds2-2525/scan_cavity_depth.csv` | NIST mds2-2525 | 8 | vapor-depression depth against time, scanned track |
| `nist_mds2-2525/spot_cavity_depth.csv` | NIST mds2-2525 | 101 | vapor-depression depth against time, stationary spot |
| `nist_mds2-2525/radiographs/*.png`, `radiographs.csv` | NIST mds2-2525 | 5 | the radiographs shown in `fig:calib-xray` and `fig:closing-xray` |

Two sources the manuscript compares against, Galiègue et al. 2025 and Hawk et al. 2022, are cited below with what the manuscript uses from them, but their values are not redistributed here: their licences do not permit it (see each section). Their values are read from the publications themselves.

Bibliographic details below were taken on 2026-10-29 from the DOI registration records (Crossref for
articles, DataCite for datasets), the repository landing pages, and the title pages of the published articles.

## Licences at a glance

| source | licence | redistribution of the extracted values |
|---|---|---|
| Gan et al. 2021 | CC BY 4.0 | allowed with attribution and a notice of changes (given below) |
| Galiègue et al. 2025 | CC BY-NC-ND 4.0 | not redistributed: the licence withholds permission to share adapted material |
| NIST mds2-2718, mds2-2525 | NIST data, public | allowed; modified works should say what was changed (given below) and acknowledge NIST |
| Hawk et al. 2022 | subscription article, no open licence | not redistributed: no licence grants it |
| Soares et al. 2026 | CC BY 4.0 | allowed with attribution and a notice of changes |
| Vaglio et al. 2020 (dataset) | CC BY 4.0 (the Data in Brief article is CC BY-NC-ND 4.0) | allowed: the values come from the CC BY dataset |

---

## Gan et al. 2021: keyhole depth by in-situ X-ray imaging

**Citation.** Gan, Z., Kafka, O. L., Parab, N., Zhao, C., Fang, L., Heinonen, O., Sun, T. & Liu, W. K.
Universal scaling laws of keyhole stability and porosity in 3D printing of metals. *Nature Communications*
**12**, 2379 (2021), published 22 April 2021. DOI: https://doi.org/10.1038/s41467-021-22704-0.
Manuscript key: `gan_universal_2021`.

**Licence.** Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/. Read in the article's licence statement (Europe PMC full-text
record PMC8062476, which reproduces the publisher's statement, and the published article) and in the Crossref
licence record of the DOI.

**What was taken.** Supplementary Data 1 (`41467_2021_22704_MOESM4_ESM.xlsx`, downloaded from the publisher's
supplementary-file server, sha256 `b52d9173eb3a63983dd87c56b2720abd419a200e9594b863218de13da6a7597a`), all
87 rows. Kept: laser power `P`, scan speed `V`, spot size `d`, beam radius `r0`, keyhole depth `e`, keyhole
inlet length `Length (µm)`, tangent of the keyhole front angle `tan(theta)`, and the group label. Not kept:
`eta`, an absorptivity computed by the authors' model (a model output), and the
material-property columns (`K`, `rho`, `cp`, `alpha`, `Tv-T0`, `Lv`, `Tl-T0`, `Lm`), which are inputs to the
authors' scaling analysis.

**How.** Read cell by cell from the downloaded file. Changes: `d`, `r0` and `e` converted from m to µm;
`e` rounded to 0.1 nm; `tan(theta)` rounded to six decimals; columns removed as listed; rows unchanged.

**Units and conventions.** Depth, inlet length and front angle are defined on the X-ray image in the
paper's Supplementary Fig. 1. The paper measures the depth at 20 µs intervals and shows its minimum and
maximum over the middle 2 mm of the track as error bars in its Fig. 1a; Supplementary Data 1 gives one value
per condition and no spread. The laser is described as single-mode with a Gaussian beam profile; the paper
does not say whether the spot size is a 1/e² or a D4σ diameter. The radius is the tabulated `r0 = d/2`
(47.5 µm for the 95 µm group), used directly by the manuscript. The group label "Ti6, spot size 140 μm" is
read as Ti-6Al-4V (its property columns are identical to those of the "Ti64" group). Powers are as tabulated
(for example 103.198 W).

**Credit.** Gan et al. credit the Ti-6Al-4V keyhole data to their reference 2 (caption of their Fig. 1):
Cunningham, R. et al. Keyhole threshold and morphology in laser melting revealed by ultrahigh-speed x-ray
imaging. *Science* **363**, 849–852 (2019), https://doi.org/10.1126/science.aav4687.
Credit line: *Data from Gan et al., Nat. Commun. 12, 2379 (2021), Supplementary Data 1, CC BY 4.0
(Ti-6Al-4V points originally from Cunningham et al., Science 363, 849 (2019)); reformatted, units converted,
model-output and property columns removed.*

## Galiègue et al. 2025: melt-pool depth and width on cross-sections

**Citation.** Galiègue, L., Briffod, F., Ito, K., Masuo, H., Watanabe, M., Shiraiwa, T. & Enoki, M.
Acoustic emission monitoring reveals keyhole formation and evolution. *Scientific Reports* **15**, 43243
(2025), published 5 December 2025. DOI: https://doi.org/10.1038/s41598-025-27232-1. Europe PMC: PMC12680680.
Manuscript key: `galiegue_acoustic_2025`.

**Licence, and why no file is here.** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
(CC BY-NC-ND 4.0), https://creativecommons.org/licenses/by-nc-nd/4.0/, per the article's licence statement
(Europe PMC PMC12680680 and the published article) and its Crossref record: "You do not have permission under
this licence to share adapted material derived from this article or parts of it." A reformatted extract of
its supplementary table would be adapted material, so none is distributed.

**What the manuscript uses, and where to get it.** The Ti-6Al-4V melt-pool depths and widths of Supplementary
Information 1 of the article (`41598_2025_27232_MOESM1_ESM.xlsx`, sheet `Supplementary_data`, from the
article's page), at r0 = 60 µm and 0.15 to 0.9 m/s and at the defocused radii, in `sec:calibration`,
`fig:calib-materials`, `fig:calib-transition`, `fig:calib-spot`, `fig:calib-mesh`, `fig:closing` and
`tab:closing-conditions`.

**Units and conventions.** Bare plates (Ti-6Al-4V 3 mm, 316L 5 mm, IN625 2 mm thick), single tracks with a
continuous-wave fiber laser (Raycus RFL-C500W, wavelength and beam profile not stated), argon or no shielding
gas. Depth and width were measured on cross-sections perpendicular to the track (Kroll's etch for
Ti-6Al-4V); the paper does not say whether depth runs from the original plate surface or from the top of the
bead. Beam: the focused beam has a D4σ diameter of about 120 µm, and the tabulated `r0 = 60 µm` is its half,
which is the manuscript's convention directly. Defocused radii (64, 69, 83 and 130 µm at 2, 3, 5 and 10 mm
negative defocus) are as tabulated; the paper does not say how they were determined. `sections_averaged`:
the paper states that low-speed tracks were sectioned at three locations (0.5, 0.7 and 0.9 mm) and averaged,
and other conditions at one location; its Fig. 4 caption gives the three-point average for the focused
Ti-6Al-4V tracks at 150 to 300 mm/s. The column is 3 for those rows, 1 for speeds of 500 mm/s and above, and
empty where the paper leaves it open (400 mm/s, the defocused series, 316L and IN625 below 500 mm/s). Along
the track, depth varies by up to about 10 % for deep pools (the paper's Supplementary Fig. S9). Keyhole
frequency comes from airborne acoustic emission; empty where none is reported.

## NIST AM Bench AMB2022-03: IN718 single-track cross-sections

**Citation (dataset).** Weaver, J., Deisenroth, D., Mekhontsev, S., Lane, B., Levine, L. & Yeung, H. (2022).
*AM Bench 2022 Measurement Results Data: Optical Microscopy of Laser-scanned Single Tracks and Pads
(AMB2022-03)*. National Institute of Standards and Technology. DOI: https://doi.org/10.18434/mds2-2718,
landing page https://data.nist.gov/od/id/mds2-2718, record version 1.0.3, accessed 2026-10-29. (The README
inside the record suggests a citation with a different author list; the list above is that of the DOI
record and the landing page.) Measurement and challenge description, version 1.01:
https://www.nist.gov/document/amb2022-03-measurement-and-challenge-descriptions-version-101.
Manuscript key: `weaver_am_2022`.

**Licence.** NIST data, public access, https://www.nist.gov/open/license (read 2026-10-29) and the "Data Use
Notes" of the record's README: works of NIST employees are not subject to copyright in the United States;
NIST grants a worldwide royalty-free right to copy, modify and distribute; modified works should carry a
notice of the change, and NIST should be acknowledged as the source of the data.

**What was taken.** `AMB2022-718-SH1-MeltPool_Cross-Section_Measurement_Results.xlsx` (downloaded from the
record, sha256 `2cfaac96aaca3dabb77b7029f842cdcc7e75c5a2cf3577d0734823246364a931`): the 42 rows whose
"Case and Line No." starts with "Line", that is the 21 single tracks of plate BP1 (7 conditions, 3 tracks
each), each sectioned at two positions along the track (4.9 and 6 mm). The 190 rows of the pad scans (plates
BP2 and BP3) are not included.

**How.** Read cell by cell from the downloaded file. `nist_amb2022-03_in718_sections.csv` keeps the 42
section measurements as tabulated (speed converted from mm/s to m/s). `nist_amb2022-03_in718_conditions.csv`
gives, per condition, the mean and the sample standard deviation over its 6 sections (3 tracks × 2
positions), rounded to 0.1 µm. `case_number` is the case of Table 2 of the challenge description.
Notice of change: reformatted and aggregated on 2026-10-29 as described here.

**Units and conventions.** Spot diameter is the D4σ diameter of a rotationally symmetric Gaussian beam
(Table 2 of the description: "Spot size, D4σ [μm]"; the spreadsheet header reads "Beam diameter (gauss, avg)
(µm)"). The manuscript uses `r0 = D4σ/2` (33.5, 24.5 and 41 µm). Depth and width were measured in ImageJ on
dark-field optical micrographs (0.069 µm per pixel) with a bounding rectangle whose top is the plate surface
(record README). Bare IN718 plates cut from 3.17 mm rolled sheet, annealed and ground; argon; chamber
95 ± 5 kPa and 23.5 ± 1 °C; laser incidence 5°; 10 mm tracks (Table 1 of the description).

**Credit line.** *Data from NIST AM Bench 2022, AMB2022-03 (Weaver et al. 2022,
https://doi.org/10.18434/mds2-2718); reformatted and aggregated.*

## NIST mds2-2525: Ti-6Al-4V absorptance and X-ray imaging (AM Bench 2022 asynchronous challenge)

**Citation (dataset).** Simonds, B. J., Tanner, J., Artusio-Glimpse, A., Williams, P. A., Parab, N., Zhao, C.
& Sun, T. (2022). *Asynchronous AM Bench 2022 Challenge Data: Real-time, simultaneous absorptance and
high-speed Xray imaging*. National Institute of Standards and Technology. DOI:
https://doi.org/10.18434/mds2-2525, landing page https://data.nist.gov/od/id/mds2-2525, record version 1.3.2
(DataCite; the README file is `2525_README_v200.txt`), accessed 2026-10-29.

**Associated articles.** The manuscript cites the record through Simonds, B. J., Tanner, J.,
Artusio-Glimpse, A., Parab, N., Zhao, C., Sun, T. & Williams, P. A. Ability to Simulate Absorption and Melt
Pool Dynamics for Laser Melting of Bare Aluminum Plate: Results and Insights from the 2022 Asynchronous
AM-Bench Challenge. *Integrating Materials and Manufacturing Innovation* **13**, 175–184 (2024),
https://doi.org/10.1007/s40192-023-00336-0 (CC BY 4.0; manuscript key `simonds_ability_2024`). The record
names as its primary supplement: Simonds, B. J., Tanner, J., Artusio-Glimpse, A., Williams, P. A., Parab, N.,
Zhao, C. & Sun, T. The causal relationship between melt pool geometry and energy absorption measured in real
time during laser-based manufacturing. *Applied Materials Today* **23**, 101049 (2021),
https://doi.org/10.1016/j.apmt.2021.101049.

**Licence.** NIST data, public access; same terms as mds2-2718 above (https://www.nist.gov/open/license and
the record README, which points to https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software).

**Experiment.** Ti-6Al-4V machined from NIST SRM 654b, thinned to about 300 µm along the X-ray path, laser
surface polished; argon at room pressure; Yb fiber laser at 1070 nm, 7° incidence; beam approximately
Gaussian with a 1/e² diameter of 122.5 ± 3.0 µm at the sample surface (2.8 mm below the waist). The manuscript
uses r0 = 61.25 µm. Two configurations: a track scanned at 700 mm/s, and a stationary spot with a pulse of
about 2.0 ms. Absorptance is measured with an integrating sphere and an oscilloscope at 40 ns resolution;
X-ray images at 50,000 frames per second with 2.5 µs exposure. All from the record README.

### Absorptance time series

`scan_absorptance.csv` and `spot_absorptance.csv` come from the record's
`Scan on Bare Metal_Calibrated Absorption Data.csv` (sha256
`1c64f24e84c274d9f9ae27fb09e79b86cda2fda5bee4b67da3567c8a59ca499d`) and
`Spot on Bare Metal_Calibrated Absorption Data.csv` (sha256
`0e96b220852d762fde846e406cc44c6fc874cef22e7e41db7f4025dbcc9ca274`), downloaded 2026-10-29. Rows are the
samples with input power above 50 W (laser on): 0.12 to 1964.04 µs for the scan (median input 196.2 W) and
0 to 2008.76 µs for the spot (median input 101.9 W). Columns:

- `nist_sample_index`: the numerical index of the NIST file.
- `time_us`: time from laser on, `first printed time + 0.04 µs × sample index` (the 40 ns sampling). The
  scan file prints its `Time` column rounded to 1 µs between 100 and 999 µs; this reconstruction agrees with
  every printed value to within its printed precision, and is identical to the printed time in the spot file.
- `time_us_as_published`: the `Time` column converted from s to µs.
- `input_laser_power_W`, `absorbed_power_W`, `absorbed_power_expanded_uncertainty_W`: the NIST columns
  `InputLaser`, `AbsoluteAbsorption`, `AbsAbsorptionUncertainty`, as printed (the scan file prints an
  uncertainty of 0 for 12601 of its laser-on samples; the spot file prints `--` for 54, left empty here).
  The NIST README notes that the sharp peak of `InputLaser` in the first few hundred nanoseconds is an
  artifact.
- `relative_absorption_pct_as_published`: the NIST column `RelativeAbsorption`.
- `absorbed_fraction_pct`: `100 × AbsoluteAbsorption / InputLaser`, the measured absorbed fraction plotted in
  `fig:calib-history-abs` and `fig:calib-spot-abs`. It differs from `RelativeAbsorption` by about one
  percentage point on average for the scan.
- `absorbed_fraction_running_median_40us_pct`: centred running median over 1001 samples (40 µs), edge values
  held, computed over the rows marked as used; this is the bold measured line of those panels.
- `used_in_manuscript`: rows inside the plotted window (0 to 849.5 µs for the scan, 0 to 2008.4 µs for the
  spot).

### Vapor-depression depth time series

These depths are **not tabulated by NIST**. The manuscript authors segmented them from the NIST radiographs
(as the manuscript states). They are the measured series of `fig:calib-history-depth` and
`fig:calib-spot-depth`.

- `scan_cavity_depth.csv`: every fifth frame from frame 105 to frame 140 (127 to 827 µs) of
  `Scan on Bare Metal_XrayImages_processed_captioned.zip`. Depth is the depth below the substrate surface
  line of the deepest pixel of the largest bright (cavity) region near the beam, in whole pixels
  (`vapor_depression_depth_px`) times 2.01 µm.
- `spot_cavity_depth.csv`: frames 83 to 183 (7.5 to 2007.5 µs, the beam-on interval) of
  `Spot on Bare Metal_XrayImages_Raw.zip`. Each frame is divided by the mean of the frames before laser on;
  the cavity is the set of pixels above 1.10 times that reference that touches the fitted substrate surface
  line (at least 8 pixels); depth is its largest perpendicular distance below that line, times 2.01 µm.
  Frames 83 to 112 show no cavity (depth 0).
- Pixel size: the NIST record states none. 2.01 µm per pixel was determined from the beam travel between
  frames of the scanned track at 0.7 m/s, and the same scale is applied to the spot frames, which have the
  same sampling. Every depth scales with this number.
- `time_us`: exposure start of the frame, the rising edge of `FrameTrigger` at which the NIST CSV writes the
  frame number, on the `time_us` axis above.

### Radiograph images

`radiographs/` holds the five X-ray images as shown in `fig:calib-xray-transient`, `fig:calib-xray-settled`,
`fig:calib-xray-spot-early`, `fig:calib-xray-spot-late` and `fig:closing-xray`. `radiographs.csv` gives,
for each image: the NIST archive and frame number, the frame time (exposure start), the time printed in the
figure, the extent of the image in µm, its size in pixels and the µm per image pixel, and the processing
applied to the NIST frame. Axes: `x` is the distance from the beam axis in µm (for the scan, at that frame
and positive in the scan direction); `z` is the height above the original substrate surface in µm (negative
below it). The images are 8-bit grey levels after the stated processing (flat-field division and a linear
grey window between the 2nd and 99.5th percentiles), resampled for display, except
`scan_frame130_t627us_closing_crop.png`, which is at the detector sampling of 2.01 µm per pixel. Draw them
with a grey colour map from 0 to 255 over the given extent.

Notice of change for all mds2-2525 files: selected, reformatted, recomputed (absorbed fraction and running
median) and, for the images, rotated, flat-fielded, cropped and contrast-mapped, on 2026-10-29, as
described here. The depth series are the manuscript authors' own analysis of the NIST images.

**Credit line.** *Contains data from NIST AM Bench 2022 (Simonds et al. 2022,
https://doi.org/10.18434/mds2-2525); selected, reformatted and processed as described.*

## Hawk et al. 2022: 304L stationary spot welds

**Citation.** Hawk, C., Simonds, B., Tanner, J., Pacheco, R., Brand, M., Vigil, G., Javernick, D. & Liu, S.
Laser spot welding of additive manufactured 304L stainless steel. *Welding in the World* **66**(5), 895–906
(2022), published online 21 February 2022. DOI: https://doi.org/10.1007/s40194-022-01265-w. Accepted
manuscript: report LA-UR-21-30444, OSTI record https://www.osti.gov/biblio/1868245.
Manuscript key: `hawk_laser_2022`.

**Licence, and why no file is here.** A subscription article without open access: it carries "© International
Institute of Welding 2022", and its Crossref record lists only the publisher's text-and-data-mining licence.
No licence grants redistribution of its tables, so none is distributed.

**What the manuscript uses, and where to get it.** The weld depth and width of three conditions from Table 5
of the published article (Table A-2 of the accepted manuscript on OSTI), in `sec:calibration` and
`fig:calib-materials`.

**Units and conventions.** Stationary spot welds of 10 ms with a Yb fiber laser at 1070 nm under sharp
focus, "producing a spot size of 303 µm and a top-hat beam profile" (the paper does not use the word
diameter; the value is read as the diameter of the top-hat). The manuscript's validation simulations of these
welds use a top-hat beam of 303 µm diameter (radius 151.5 µm). `laser_power_W` in the spot-weld file is the
applied energy divided by 10 ms; the coupling tables print the same powers, except 355 W (Table 4) against
356 W (Table 3) for 3.55 J, and the file gives each table's own value. Depth and width were measured on
etched cross-sections, one weld per condition, with no uncertainty reported. Coupling efficiency was measured
with an integrating sphere recalibrated after every weld; its uncertainty is 1σ.

## Soares et al. 2026: CoCr single-track cross-sections

**Citation.** De Araujo Soares, M., Campion, D., Leclercq, A., Kreitcberg, A. & Brailovski, V. Reliability of
Thermal Conduction-Based Melt Pool Simulations Using In-Process Thermal Camera and Post-Process Single-Track
Measurements. *Applied Sciences* **16**(12), 5850 (2026), published 10 June 2026. DOI:
https://doi.org/10.3390/app16125850. (Crossref records the family name as "Soares" and the given names as
"Matheus De Araujo".) Manuscript key: `soares_reliability_2026`.

**Licence.** Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/: "© 2026 by the authors. Licensee MDPI, Basel, Switzerland ...
Creative Commons Attribution (CC BY) license" in the published article, and CC BY 4.0 in the Crossref
licence record.

**What was taken.** The CoCr columns of Table 4 (power, speed, linear energy density) and Table 6 (width and
depth with their uncertainty) for the ten parameter sets P0 to P9. The 316L and IN625 columns of the same
tables are not included.

**How.** Transcribed from the tables. Changes: speed converted from mm/s to m/s.

**Units and conventions.** EOS M280 with a 400 W Yb fiber laser, "beam radius r_f = 50 µm" (the paper does
not define the radius as 1/e² or D4σ); the manuscript uses r0 = 50 µm. EOS Cobalt Chrome MP1 powder, single
tracks on stainless-steel baseplates without preheating, argon; the printed zones used a 40 µm layer
thickness. Width and depth are means of 8 measurements on cross-sections (Keyence VHX-7000, 500× and 1000×).
The ± values are combined uncertainties: the paper states that they include track-to-track process
variability and the operator-dependent identification of the melt-pool boundary.

**Credit line.** *Data from De Araujo Soares et al., Appl. Sci. 16, 5850 (2026), Tables 4 and 6, CC BY 4.0;
transcribed, units converted.*

## Vaglio et al. 2020: Ti-6Al-4V single tracks with a 50 µm spot

**Citation (article).** Vaglio, E., De Monte, T., Lanzutti, A., Totis, G., Sortino, M. & Fedrizzi, L. Single
tracks data obtained by selective laser melting of Ti6Al4V with a small laser spot diameter. *Data in Brief*
**33**, 106443 (2020). DOI: https://doi.org/10.1016/j.dib.2020.106443. Manuscript key: `vaglio_single_2020`.

**Citation (dataset).** Totis, G., Vaglio, E., De Monte, T., Lanzutti, A., Sortino, M. & Fedrizzi, L. (2021).
*Data for: Single Tracks Data obtained by Selective Laser Melting of Ti6Al4V with a Small Laser Spot
Diameter*. Mendeley Data, version 1, published 26 April 2021. DOI: https://doi.org/10.17632/s9438vb5xd.1.
(Contributor order as on the Mendeley Data page; DataCite lists Totis only.)

**Licence.** Dataset: CC BY 4.0 (Mendeley Data page and DataCite record). Article: CC BY-NC-ND 4.0 (article
licence statement, Europe PMC record PMC7642809, Crossref). The values come from the dataset.

**What was taken.** The sheets *Track width (W)*, *Track depth (D)* and *Track height (H)* of `Data.xlsx`
in the dataset archive: 80 conditions (8 powers from 50 to 400 W × 10 speeds from 0.25 to 2.5 m/s).

**How.** Read from the dataset spreadsheet; the archive (about 105 MB) was not downloaded again for this
folder, so these values were not re-checked against it here. Speed is in m/s (the dataset uses mm/s in its
file names).

**Units and conventions.** From the article: Concept Laser M2 Cusing with a 400 W single-mode
continuous-wave Yb fiber laser at 1070 nm, Gaussian beam with M² = 1.08, spot size 50 µm "according to the
1/e² classical definition" (so r0 = 25 µm); one track per condition on a 25 µm Ti6Al4V powder layer over
previously built structures; each geometric feature averaged over three locations along the track where
possible. Width W, height H and depth D are defined in the article's Fig. 2(b). The manuscript uses one row
(100 W, 0.25 m/s) for the scatter between laboratories.

**Credit line.** *Data from Totis et al., Mendeley Data (2021), https://doi.org/10.17632/s9438vb5xd.1,
CC BY 4.0, described in Vaglio et al., Data in Brief 33, 106443 (2020); reformatted.*
