# Material property provenance

Citations for every material in `materials/`. Each material is a directory:
`materials/<name>/<name>.json` holds scalars + the `optical` block; the
temperature-dependent values are one CSV each (`<property>.csv`, columns
`temperature_k,<property>`). This file records **where each value comes from** and
how much to trust it; parameters from different sources are listed separately.

## Two tiers of confidence

1. **Thermophysical (density, specific heat, thermal conductivity, viscosity,
   surface tension, latent heats, transition temperatures)**: starter
   engineering values assembled for reproducible scripted tests and not
   individually traced to a measurement. The alloys named `*_mills`/`Mills_data`
   derive from Mills 2002 (also the source behind the solver's material database); the
   rest are representative literature/datasheet values. **Verify against the target
   feedstock and temperature range before any scientific use** (see `assumptions.md`
   A11). The `source` field inside each `<name>.json` states this too.
2. **Optical (`electrical_resistivity_ohm_m.csv`, `optical.normal_absorptivity`)**:
   added for the Fresnel laser-absorption model and cited per value below.

## Electrical resistivity ρ(T): `electrical_resistivity_ohm_m.csv`

Stored as a multi-point table from room temperature (≈293 K) to a
near-liquidus/liquid value (linear between points). The full ρ(T) curve drives the
temperature-dependent Fresnel ε(T) (`ereflct(T)`) when the resistivity route is
used. Units Ω·m. The endpoints in the table below are the cited handbook/measurement
values; the **intermediate solid points follow the established temperature trend for
each alloy class** (austenitic and carbon steel: near-linear rise toward the melt;
Ti‑6Al‑4V: weak solid rise then a drop on melting, Boivineau 2006; IN718: nearly
flat, Pottlacher 2002; pure Cu: CRC linear-in-T); they were not individually
digitised from a figure. Treat the curve **shape** as correlation-grade and the
**endpoints** as cited; intermediate points are one confidence tier below the
endpoint values listed here.

| material | ρ(293 K) | ρ(liquid) | sources | confidence |
|---|---|---|---|---|
| ti_imi318_mills (Ti‑6Al‑4V) | 1.7e‑6 | 1.5e‑6 | RT: ASM/handbook for Ti‑6Al‑4V (~1.7 µΩ·m); liquid: Boivineau et al. 2006 | firm |
| ss304, ss304l | 0.72e‑6 | 1.28e‑6 | RT: AISI 304 handbook (~0.72 µΩ·m); liquid: Mills 2002 | medium |
| ss316l | 0.74e‑6 | 1.28e‑6 | RT: AISI 316 handbook (~0.74 µΩ·m); liquid: Mills 2002 | firm |
| inconel718 | 1.25e‑6 | 1.28e‑6 | RT: Special Metals IN718 datasheet (1.25 µΩ·m); liquid: Mills 2002 / Pottlacher | firm |
| cocr (Co‑Cr‑Mo) | 0.85e‑6 | 1.20e‑6 | estimate by analogy to Ni/Co superalloys; no direct datum | low |
| alsi10mg | 0.05e‑6 | 0.30e‑6 | RT: AlSi10Mg AM literature; liquid: liquid-Al + Si estimate | low |
| al6061 | 0.04e‑6 | 0.27e‑6 | RT: ASM Al‑6061 (~0.04 µΩ·m); liquid: liquid-Al estimate | low |
| pure_copper | 0.017e‑6 | 0.21e‑6 | RT: CRC handbook (Cu 1.7e‑8 Ω·m); liquid: handbook/Mills | firm (ρ); the Drude route is *not* used for Cu; see below |

## Normal-incidence absorptivity at ~1 µm: `optical.normal_absorptivity`

Used directly for materials with `optical.prefer_resistivity=false`, and as a
cross-check otherwise.

| material | A | sources | confidence |
|---|---|---|---|
| Ti‑6Al‑4V | 0.35 | Trapp et al. 2017 (316L/Ti64 in-situ, 1070 nm); Bergström 2008 | medium |
| 304/304L/316L | 0.35 | Trapp et al. 2017 (316L bulk ≈0.3, →~0.7 in keyhole) | medium |
| inconel718, cocr | 0.35 | analogy to transition-metal alloys at ~1 µm | low |
| alsi10mg | 0.10 | Al alloys reflective at ~1 µm (≈0.05–0.15) | low |
| al6061 | 0.09 | Al alloys at ~1 µm | low |
| pure_copper | 0.03 | Cu reflective at ~1 µm (≈0.02–0.05) | medium |

`optical.prefer_resistivity` is `false` for `pure_copper`, `al6061`, `alsi10mg`
(the DC-resistivity Drude form overestimates absorption for these at ~1 µm, so the
measured absorptivity is used) and for `ti64_v4_measured` / `ti64_v5_tv3420_measured`,
which exist to carry the measured A=0.33. It is `true` for the other seven.

**A = 0.35 is a sentinel value with no measurement behind it; do not "improve" it.** The seven materials
carrying exactly 0.35 (`cocr`, `inconel718`, `ss304`, `ss304l`, `ss316l`,
`ti64_v3_calibrated`, `ti_imi318_mills`) are *exactly* the seven with
`prefer_resistivity=true`, and the five with distinct values (0.03–0.33) are exactly the
five with it false. The correspondence has no exceptions. For the seven, the
solver computes Fresnel from ε(T) off the resistivity table and **never reads
`normal_absorptivity` except as a cross-check**, so 0.35 means "ignore me".

Writing real per-alloy absorptivities into those seven files therefore changes nothing
physically while making the files look like they carry measured optics they do not, so
such a cleanup, however plausible it looks, should be rejected. If a per-alloy
optical difference is genuinely wanted for one of the seven, it belongs in that material's
`electrical_resistivity_ohm_m.csv`, which is what the solver actually reads.

Consumers should note the same thing: a `mat_A` model channel fed from
`normal_absorptivity` is **degenerate across all seven** and carries zero discriminative
signal between them, even though their ρ(T), and hence their simulated absorption,
genuinely differs (Ti64 1.70e-6 vs IN718 1.25e-6 Ω·m at 293 K, ~1.36×).

**Worse than degenerate for anyone pre-training on the synthetic material sweep**
(610 manifests: 600 sampled synthetic materials and 10 real alloys): across the 600
synthetic cases `mat_A` takes **292 distinct values spanning 0.1–0.3999**, while
across the real alloys it is the 0.35 sentinel. So a model pre-trained on the synthetic
sweep *learns to rely on that channel* and then meets it **pinned at a constant** on every
real alloy: an out-of-distribution input on a channel it depends on, which is a stronger
reason to drop `mat_A` than degeneracy alone. This applies to anyone fine-tuning a
model trained on that sweep onto real alloys, whatever the recipe.

**And the resistivity channel is not a drop-in replacement in mixed recipes:** only **10 of
those 610 manifests** carry `electrical_resistivity_table` (the 10 real-alloy cases);
sampled synthetic materials have no resistivity table at all. So a resistivity-derived
channel is usable in a real-alloy-only recipe and missing for ~98 % of cases in anything
that mixes the synthetic sweep. Of the six `mat_*` channels, five separate Ti64 from IN718
(`mat_rho` 1.85×, `mat_k` 1.27×, `mat_Tl` 0.84×, `mat_cp` 0.80×, `mat_latent` 0.61×) and
`mat_A` is 1.00×, so dropping `mat_A` and deferring resistivity leaves ample separation.

**Check which Ti64 variant a manifest names before comparing it to a JSON.** A staged Ti64
manifest reading `vaporization_temperature_k` 3123.15 is not a discrepancy against
`ti64_v3_calibrated`'s 3520: it means those cases used `ti_imi318_mills`. Three Ti64
material files are in use across the record and they differ in more than optics.

The material-derived Fresnel coupling is the **default** whenever a material has this
optical data (a resistivity CSV or `normal_absorptivity`): the runner uses ε(T) from
ρ(T) (or constant ε from absorptivity) automatically. Set
`laser.fresnel_from_material=false` in a config to fall back to the fixed scalar
`reflectivity` (the legacy line-track reproduction config does this to stay faithful).

## References

- K. C. Mills, *Recommended Values of Thermophysical Properties for Selected
  Commercial Alloys*, Woodhead (2002). [Source of the "Mills_data" entry in the solver's material database.]
- M. Boivineau et al., "Thermophysical Properties of Solid and Liquid Ti‑6Al‑4V
  (TA6V) Alloy", Int. J. Thermophys. 27, 507 (2006). doi:10.1007/s10765-005-0006-1
- J. Trapp et al., "In situ absorptivity measurements of metallic powders during
  laser powder-bed fusion", Appl. Mater. Today 9, 341 (2017).
  doi:10.1016/j.apmt.2017.08.006
- D. Bergström, "The Absorption of Laser Light by Rough Metal Surfaces", PhD
  thesis, Luleå (2008).
- Special Metals Corp., Inconel 718 datasheet; ASM Handbook Vol. 2; CRC Handbook of
  Chemistry and Physics (room-temperature resistivities).

Update this file whenever a value changes; keep one row per value with its source
and confidence. Replacing a starter thermophysical value with a properly cited
measurement is encouraged; note the source here.

## Vaporization molar mass

`vaporization_molar_mass_kg_mol` is the molar mass of the dominant evaporating
species (Al for AlSi10Mg, Fe for SS316L, Ni for IN718, Co for CoCr), standard
atomic weights. It feeds the per-material Clausius-Clapeyron exponent
`tvexp = R/(M*clhv1)` (the solver's own saturation-curve recipe). `ti_imi318_mills`
does not carry the field: its Ti64 cases render the legacy template
`tvexp=5.989e-05` (assumptions.md A30), and adding it would change the physics of
every case that uses this material. For a new Ti64 series on this material, the value
to add is M(Ti)=0.047867.
