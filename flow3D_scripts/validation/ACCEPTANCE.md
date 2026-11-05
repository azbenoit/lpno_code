# What this validation campaign has to prove, and what it does not

## Scope

The surrogate work downstream needs FLOW-3D melt/keyhole fields that an
experimentalist or AM audience will accept as physically plausible and
traceably calibrated. A single parameter set that reproduces every alloy to
measurement precision would be a modelling paper in its own right, and training
neural operators on this data does not require one.

So the campaign succeeds when all three hold:

1. **Every physics choice traces to an experimental anchor or is declared as a
   fitted scalar.** No unexplained switch settings, no parameters tuned to make a
   plot look right without saying so.
2. **Agreement sits inside the error norm the measurements themselves support**
   (see below) for the conditions the training data actually occupies.
3. **Where it does not, the gap is quantified, attributed, and bounded**: a
   stated limitation with a number.

Explicit non-goals: matching all five alloys simultaneously; matching outside the
anchored process window; reproducing absolute absorptance (a separate diagnostic
that currently fails and is reported as failing).

## The acceptance band comes from the experiments' own scatter

±30% on a length scale for campaign-wide scoring (`melt_metrics.DEFAULT_BAND`,
what `score_anchors.py` and the manuscript report). It sits at or inside the
scatter the source measurements themselves quote:

- **Galiègue 2025** states the along-track variability directly: *"This
  fluctuation gets significant for deep melt pools (around 10% relative change
  was measured at most)."* Most of their conditions are **single-location**
  measurements (*"For the other conditions, melt pool depths were measured at
  only one location"*); only the low-speed set is averaged over three locations.
- In the same dataset, at 100 W / 0.20 m/s, r₀ = 69 µm gives 103.7 µm and the
  larger r₀ = 83 µm gives 134.8 µm: depth rises by 30% with spot size, where a
  larger spot would be expected to give a shallower pool. That spread, within one
  lab and one method, is larger than our band.
- **Gan 2021** reports the same depth (62 µm) at 0.7 and 0.9 m/s at 156 W, a speed
  step over which depth would be expected to fall. All points are kept and scored.
- Cross-lab, at matched power and speed, Vaglio and Galiègue differ by 12% on
  depth (110.9 vs 98.7 µm) and 0.1% on width.

A model scored to tighter than ~10–30% against these anchors would be fitting
measurement noise. The failure mode to guard against is claiming better agreement
than the data supports.

**What this band does not license:** a *systematic* offset of one sign across a
whole series is a finding even when every individual point is inside ±30%. Report
the mean signed error alongside the pass count, always.

**The calibration gates used tighter bands, per anchor.** Before the cases were
solved, the gates were registered at ±15% on two points of the Gan Ti64 power
series (129 W and 206 W) and ±20% on the IN718 keyhole blind test
(`rep4_in718_s049_blind`, 285 W / 0.96 m/s); SS316 and the first-contact anchors
used ±30%. Every verdict names its band: the IN718 keyhole depth error of −25.2%
is an absolute error inside 30% and outside the stricter 20% gate.

## Metric definitions: these must be identical everywhere

Every scorer and every table in this campaign uses these definitions.

| metric | definition |
|---|---|
| **fusion depth** | time-**union** of `Melt Region > 0.5` over a ±5-cell band at mid-track; deepest cell below z = 0. Matches cross-section metallography, which sees the maximum extent ever melted. |
| **fusion width** | same time-union, same band: full y-extent. |
| **depression depth** | per-frame deepest *interior* gas cell (`FoF < 0.5`) below z = 0, x/y edge cells excluded; mean ± sd over the steady window. Matches in-situ X-ray. |
| **depression width** | **the same steady window as the depth**; never a separate frame. |
| **steady window** | laser-on = first 85% of frames; steady = final 30% of that. Report window sensitivity (50/30/20/10%) whenever a claim rests on the window. |
| **aspect ratio** | depth ÷ width from **one** window for both. |
| **melting mode** | `depression / r₀`: keyhole ≳ 0.5, conduction ≲ 0.25. A power-law fit must not span the transition. |
| **domain validity** | floor temperature < 400 K **and** melt y-span < 0.8 of the domain. A clipped case counts for nothing, in either direction. |

**A feature under ~5 cells may be unformed as well as unmeasurable.**
This is the deeper version of the quantization rule below. A matched mesh pair
(identical physics, cell size alone differing) gave a 2.3-cell vapour cavity at
10 µm and a 24.7-cell cavity at 5 µm: **+164% fusion depth, +439% cavity**. Beyond
failing to *resolve* the cavity, the coarse grid prevented the solver from
*sustaining* one. Sorting every scored Ti64 case by cavity cell count then showed the campaign's pass/fail split running
along resolution, not along melting regime: every case under ~5 cells failed,
every well-resolved case passed.

The consequence for interpretation: when a simulated feature is a few cells across,
"the model under-predicts it" and "the mesh cannot support it" are the same
observation, and no amount of comparison against measurements separates them. Only
a matched mesh pair does. Never attribute a physical cause to a deficit in a feature
spanning fewer than ~5 cells without one.

**A metric at its quantization floor is not a measurement.** A length resolved by
only 1–3 cells cannot be compared to anything: report it as "≤ N cells" and draw
no conclusion from it. The tell is a standard deviation of exactly zero across the
steady window: a real vapour cavity fluctuates, so `20.0 ± 0.0 µm` on a 10 µm
grid means "2 cells, pinned". Example: `spot60_p90_v020_std`, where the
depression was at the floor while the fusion depth (6 cells) and width (17 cells)
were properly resolved. Rule of thumb: quote
a length only if it spans ≥ 5 cells, and state the cell count next to any length
below 10 cells.

**Depth-kind matching is mandatory.** Metallography anchors report *fusion*;
X-ray anchors report *vapour depression*. Fusion is always the larger. Comparing
across kinds inflates the apparent gap (in one case from ~2× to "4×"). Every anchor
file declares `depth_kind`; a scorer that compares across kinds is a bug.

**Spot convention matching is mandatory.** Radius vs diameter, 1/e² vs D4σ vs
FWHM. At fixed power the spot is the dominant variable. An unresolved convention
makes a scored depth meaningless, and an *assumed* one can produce a wrong
radius (in one case 40 µm).

## How a result gets reported

Per condition: our value, the measured value, signed % error, the depth kind,
the melting mode, and the domain-validity verdict. Per series: pass count, mean
signed error, and whether the error trends with power or speed. Never a pass
count alone.

## Pre-registered: what `spot60_p90_mesh5` can and cannot settle

Fixed 2026-07-30, **before the run completes**. Cost ~6-10 h (10 µm case 2115 s x a refinement
factor expected to be ~16x; the run's own elapsed time will fix the true ratio, since
its prepin differs from the 10 µm case in cell size alone). It is a 2-point comparison at 10 and 5 µm, matched in every other
respect, against Galiegue's 90 W / 0.20 m/s condition where 10 µm scores **depth -53 %,
width -16 %**.

Outcomes, decided in advance:

- **Depth deficit largely closes** (|signed| < 20 %): the transition-regime deficit is
  resolution-driven. The manuscript must drop the cavity-formation *physics* claim,
  and the 10 µm training set carries a documented depth caveat in this regime.
- **Deficit shrinks but stays out of band** (20-40 %): both contribute, neither
  dominates. Report as jointly attributed; keep the hedged wording.
- **Deficit essentially unchanged** (> 40 %): resolution is not the driver at this
  step. The physics attribution stands and is then *earned* rather than assumed.

**What it cannot do.** Two points are not a convergence study. Even if the deficit
closes at 5 µm we cannot claim convergence: Wang & Yan 2022 report 15 % depth
under-prediction at 12.5 cells per beam radius, twice our refined resolution. The
third mesh needed to demonstrate convergence (2.5 µm) costs another 16.2x, ~6.4 days
for one case, and is not affordable. So the strongest available claim remains
*bounded*: the direction and rough size of the residual, never "converged".

Width is expected to move little either way (Ebrahimi 2021 Table A1: 0.2 % across a
6x refinement), so a large width shift would point to a setup error.

## Measured: every transition-regime depth is 4-7 cells (2026-07-30)

Applied to the Ti64 transition series at 10 um, the ">=5 cells or it may be unformed"
rule binds on every point:

| case | r0 (um) | P (W) | sim depth | cells | meas depth | err | sim width | cells |
|---|---|---|---|---|---|---|---|---|
| `spot60_p80_v020_std` | 60 | 80 | 50.0 | **5.0** | 97.2 | -49 % | 160.0 | 16.0 |
| `spot60_p90_v020_std` | 60 | 90 | 60.0 | **6.0** | 128.2 | -53 % | 170.0 | 17.0 |
| `spot60_p100_v020_std` | 60 | 100 | 70.0 | **7.0** | 148.0 | -53 % | 180.0 | 18.0 |
| `spot60_p110_v020_std` | 60 | 110 | 70.0 | **7.0** | 190.8 | -63 % | 200.0 | 20.0 |
| `spot83_p100_v020_std` | 83 | 100 | 50.0 | **5.0** | 134.8 | -63 % | 200.0 | 20.0 |
| `spot130_p100_v020_std` | 130 | 100 | 40.0 | **4.0** | 52.0 | -23 % | 200.0 | 20.0 |

Three consequences, each a separate claim:

**1. The depth/width asymmetry has a resolution reading.** Width is measured with 16-20
cells and passes (4/4 in band on the r0 60 series); depth is measured with 4-7 and
fails (0/4). The metric that fails is the one carrying ~3x fewer cells. This does not
prove resolution *causes* the deficit, but it removes "why would resolution hit depth
and not width?" as an objection; it is also what Ebrahimi 2021 Table A1 reports
independently (width 0.2 %, depth 11.9 % across a 6x refinement).

**2. Quantization uncertainty is two-thirds of the acceptance band.** Depth is counted
in whole cells, so every simulated depth is an exact multiple of 10 um. On a 50 um pool
one cell is +/-20 % against a +/-30 % band. The depth measurement is barely
resolved enough to score at all, and no depth agreement inside ~20 % should be read as
meaningful at this cell size.

**3. `spot130_p100_v020_std` is below our own floor and must be qualified.** At **4
cells** it sits under the >=5-cell rule, and it is the *only* depth point in band
(-23 %). Its 12 um miss is 1.2 cells, inside quantization noise. So the three-point
spot-size series shows width degrading monotonically with spot size (-11, -22, -35 %),
but the apparent *improvement* in depth at r0 130 rests on the least reliable point in
the campaign and must not be reported as a spot-size trend in depth.

What this does not say: quantization sets a precision floor and introduces no
systematic bias. A systematic -54 % is far larger than +/-20 % noise, so quantization
cannot explain the deficit by itself. The candidate bias mechanism is Much et al.
2026's: recoil is exponential in interface temperature, so an under-resolved surface under-predicts
recoil and suppresses the cavity. Both point the same way, which is what
`spot60_p90_mesh5` (5 um, same physics, cell size the only difference) is running to
separate. At 5 um these depths become 8-14 cells and the quantization floor halves.

## OUTCOME: the transition-regime depth deficit is physics, not resolution (2026-07-31)

`spot60_p90_mesh5` completed and was scored against the bands fixed above **before**
it landed. The comparison is controlled: the two prepins differ in `size` and
`project` alone (verified by diffing the rendered inputs, not the manifests).

| | cell | frames | depth | cells | depth err | width | cells | width err |
|---|---|---|---|---|---|---|---|---|
| `spot60_p90_v020_std` | 10 µm | 689 | 60.0 µm | 6.0 | **-53 %** | 170.0 µm | 17.0 | **-16 %** |
| `spot60_p90_mesh5` | 5 µm | 173 | 65.0 µm | 13.0 | **-49 %** | 170.0 µm | 34.0 | **-16 %** |

Measured (Galiegue 2025, r0 60 µm, 90 W, 0.20 m/s): depth 128.25 µm, width 203.0 µm.
Both cases valid, floor 300 K, melt span 0.34, neither depth-truncated.

**Verdict: PHYSICS.** Halving the cell closed **4 points of a 53-point deficit**. That
falls in the pre-registered ">40 % = the physics attribution is earned rather than
assumed" band, so the manuscript's cavity-formation reading stands, now backed by a
controlled test.

Three things this settles that argument alone could not:

1. **The quantization worry is answered.** At 10 µm the depth was 6 cells, close enough
   to the >=5-cell floor that "the feature may be unformed rather than unmeasured" was a
   live alternative. At 5 µm it is **13 cells**, comfortably resolved, and still
   -49 %. Under-resolution does not suppress the cavity here.
2. **Width converges; depth does not converge to the measurement.** Width is *170.0 µm
   at both resolutions* (17 and 34 cells) and stays -16 %. The model reproduces the
   lateral extent and misses the depth, and refining does not move either. This matches
   Ebrahimi 2021 Table A1 (width 0.2 %, depth 11.9 % across a 6x refinement) in kind,
   though our depth moves less.
3. **The deficit is not in the direction refinement would fix.** Much et al. 2026's
   mechanism (recoil exponential in interface temperature, so a coarse grid
   under-predicts recoil) predicts the deficit should shrink substantially on
   refinement. It shrank by 4 points. That mechanism is therefore not the dominant one
   at this cell size.

**Cost, finally measured on a controlled pair: 11.99x CPU** (20 183.6 -> 242 069 s).
Take it from CPU time: the run shared the box. This is *below* the theoretical
16x (8x cells x 2x CFL) and close to our prior paper's 10.9x
(arXiv:2602.06241). It replaces the 31.5x and 16.2x estimates used before this pair
was measured, including the 16.2x in the pre-registration above.

**Caveat on the export.** The 5 um NPZ is a **stride-4** export (173 of 689 frames);
the full-frame export needs ~100 GB and was killed at frame 655/689. Striding is
metric-neutral here: subsampling the 10 um case's own 689-frame NPZ changes fusion
depth and width by **0.0 % at stride 2, 4 and 8**, because the metric is a time-union.
Anything reading *time evolution* rather than the union must use a finer stride.

**Still not a convergence study.** Two points. The third mesh (2.5 µm) costs another
~12x (roughly 34 CPU-days for one case), so "converged" remains unclaimable, as
pre-registered.

## Speed axis, on the shipped recipe (2026-08-02): the last open series

The speed-exponent targets belong to the **vapour depression** metric. A fit on
**fusion** depths gives -1.21 at 208 W and -1.17 at 156 W, which would make 156 W
"not distinguishable"; it compares across depth kinds, the comparison
`melt_metrics.score_point` refuses with `DepthKindMismatch`, and is reachable only by
calling `fusion_geometry` directly, which bypasses the guard. An earlier version of
this section reported that fusion fit; the depression fit below replaces it.

The targets are anchored to Gan's in-situ X-ray data, i.e. depression (the recipe's
depression exponent is -0.90 at 90 W, within 8 % of Gan's -0.98). Depression is
therefore the metric that must be fitted.

| case | P (W) | V (m/s) | depression (um) | cells | fusion (um) | cells |
|---|---|---|---|---|---|---|
| `vsw3_p208_v06` | 208 | 0.60 | 144.0 | 14.4 | 170.0 | 17.0 |
| `vsw3_p208_v07` | 208 | 0.70 | 118.6 | 11.9 | 140.0 | 14.0 |
| `vsw3_p209_v09` | 209 | 0.90 | 86.7 | 8.7 | 110.0 | 11.0 |
| `vsw3_p209_v11` | 209 | 1.10 | 63.4 | 6.3 | 80.0 | 8.0 |
| `vsw32_p156_v06` | 156 | 0.60 | 80.5 | 8.1 | 110.0 | 11.0 |
| `vsw32_p156_v07` | 156 | 0.70 | 62.0 | 6.2 | 90.0 | 9.0 |
| `vsw32_p158_v09` | 158 | 0.90 | 40.5 | 4.1 | 60.0 | 6.0 |
| `vsw32_p157_v12` | 157 | 1.20 | 29.3 | 2.9 | 50.0 | 5.0 |

| group | fitted (depression) | pre-registered | excess |
|---|---|---|---|
| ~208 W | **-1.34** | -0.67 | 2.0x |
| ~156 W | **-1.47** | -1.04 | 1.4x |

**Both groups are too steep.** The fusion fit would have cleared the 156 W group;
the depression fit does not. An independent derivation for the manuscript gives the
same figures.

Depression is an interface measure, so it does not carry the +/-1-cell quantization
envelope of the whole-cell fusion count. The resolution caveat that
remains is coverage: the two fastest 156 W points sit at **4.1 and 2.9 cells** across the
depression, below the >=5-cell floor, so they support the trend but not an absolute claim.

The defensible position is the manuscript's: use these cases for **absolute keyhole
depth** (6/6 resolved cases in band), and **do not claim agreement in the fitted speed
slope**.

## Where the training data is experimentally bracketed (2026-08-02)

The 84-case straight-path training set spans P 40-190 W, V 0.101-0.960 m/s. The
conditions this campaign actually scored against experiment are:

| arm | P (W) | V (m/s) |
|---|---|---|
| Galiegue transition ladder | 80-110 | 0.20 |
| Vaglio slow-speed check | 100 | 0.25 |
| Gan keyhole ladder | 103-206 | 0.40 |
| speed sweep | 156-209 | 0.60-1.20 |

Union envelope: P 80-209 W, V 0.20-1.20 m/s.

| | cases | share |
|---|---|---|
| inside the scored P-V box | 56/84 | **67 %** |
| near a scored condition in *both* P and V (+/-30 %) | 60/84 | **71 %** |
| below any scored power (< 80 W) | 25/84 | 30 % |
| below any scored speed (< 0.20 m/s) | 16/84 | 19 % |
| both at once | 13/84 | 15 % |

**Two-thirds of the training set is bracketed by scored conditions; the uncovered third
sits in one corner**: low power and low speed, i.e. conduction mode. That is the
extrapolation to declare.

**Bracketed does not mean accurate, and the text must keep the two apart.** Being inside
the envelope means a measured comparison exists nearby, which says nothing about whether
it agreed. The transition series inside this box scores depth 0/4 in band at -53 %.
The defensible phrasing is that ~two-thirds of the training set lies in a region where the
model's error against experiment has been *characterised*, and the remaining third lies
where it has not been measured at all.

## Depth-kind audit of every anchor comparison (2026-08-02)

Checked every anchor comparison made in this campaign against the anchor's declared
`depth_kind`:

| anchor | depth_kind | scoring-plan `sim_kind` | match |
|---|---|---|---|
| `galiegue2025_ti64_targets` | fusion | fusion (x3 entries) | yes |
| `gan_points` (+ sidecar) | depression | depression | yes |
| `vaglio2020_ti64_targets` | fusion | fusion | yes |
| `appsci_cocr_targets` | fusion | fusion | yes |
| `hawk304l_targets` | fusion | fusion | yes |

All seven scoring-plan entries agree with their anchor. **The transition-regime mesh
verdict is unaffected**: Galiegue is fusion and it was scored with `fusion_geometry`,
so the -53 % -> -49 % result and its PHYSICS attribution stand.

**The structural finding: the speed-exponent comparison had no scoring-plan entry.**
Everything routed through `validation/scoring_plan.json` is guarded, because the plan
carries `sim_kind` and `score_point` refuses a mismatch. The speed fit was an ad-hoc
comparison against exponent targets, outside the plan, so nothing checked its metric
and a fusion-vs-depression comparison could pass unchecked.
Adding the speed sweep to the plan with `sim_kind: depression` would close it.

Two anchors declare **no** `depth_kind` at all: `amb718_targets.json` and
`wang_cocr_targets.json`. Neither is in the scoring plan today, so neither is live, but an
undeclared kind is the same trap waiting: declare it before either is scored.
