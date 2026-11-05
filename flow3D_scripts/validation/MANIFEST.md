# Validation campaign manifest

The reproducibility inventory for the experimental validation campaign.
Acceptance band and metric definitions: `ACCEPTANCE.md`; outcomes per case group:
the table below.
Scoring: `python3 python/score_anchors.py --plan validation/scoring_plan.json
--runs <root>` in this repository. The manuscript's supplementary notebook
(`src/notebooks/experimental_validation.ipynb` in the `lpno` sibling
repository) presents the same rows; point it at your run root via
`F3D_RUNS_ROOT`.

Metric everywhere: steady-window keyhole depth = per-frame deepest interior
gas cell below z=0, laser-on window = first 85% of frames, steady = tail 30%,
mean ± sd. Fusion zone = time-union of Melt Region at mid-track. Band for
campaign-wide scoring (`score_anchors.py`, the manuscript) = ±30%
of the measurement (`melt_metrics.DEFAULT_BAND`), justified in `ACCEPTANCE.md`.
The calibration gates were registered before solving with tighter bands, per anchor:
±15% on two points of the Gan Ti64 power series (`q_129w_bundle`, `q_206w_bundle`), ±20% on
the IN718 keyhole blind test (`rep4_in718_s049_blind`), ±30% on SS316
(`q9_ss316_bundle`) and every first-contact anchor. A verdict copied from a gate
carries the gate's band, so name the band beside it. 10 µm cells: half-cell
band-edge results are indeterminate.

## Final recipes (what the campaign concluded)

- **Ti64** (`materials/ti64_tv3520`): per-material Fresnel, `recoil_model: 2`,
  `void_vapor_pressure_pa: 101325`, per-material tvexp, tv1 = 3520 ± ~95 K
  (the one calibrated scalar, fitted on Gan 129 W only), no plume.
- **IN718** (`materials/in718_full`): same recipe family, database tv1 = 3186,
  no sigma extension.
- **SS316** (`materials/ss316l`, + `ss316l_raoult` under test): same family;
  not yet in the validated set.

## Case groups (config stem -> anchor)

| group | configs | anchor | recorded outcome |
|---|---|---|---|
| Ti64 tv1 scan | `p3c_ti64_129w_tv3315_noplume`, `t1_129w_tv3420_noplume`, `t1_129w_tv3520_noplume` | Gan Ti64 129.36 W/0.4 m/s = 85.3 µm | 107/99/93 µm (3520 in band) |
| Ti64 final-recipe replay | `t2_103w`, `t2_156w`, `t2_206w`, `t2_dwell90`, `t2_dwell105`, `t2_dwell130`, `t2_dwell90_big`, `t2_nist195` | Gan ladder ±15%; Simonds onsets; NIST 61 µm diagnostic | 5/5 falsifiable legs pass; 103 W floor-indeterminate; 105 W between bands; NIST out-of-scope (44.3% vs 50.8%) |
| Ti64 sink-arm (superseded) | `q_129w_bundle`, `q_206w_bundle`, `gan103w_bundle`, `gan156w_bundle`, `q3/q4/q5_dwell*`, `q7_dwell90_bigdomain`, `q1_nist195_bundle` | same anchors | the retired aplume=0.20 arm; kept for the matched-pair audits |
| Plume/σ/acondens ablations | `p2a/p2b/p2c`, `p3a/p3b`, `p4_129w_acondens0` | matched controls | plume = beam-power sink; acondens null; σ depth-inert |
| IN718 final table | `p3a_in718_noplume`, `rep3_in718_*` | NIST AMB2022-03 (n=6/condition) | depth 6/6 and width 5/6 with an absolute error within 30%; 3 edge-indeterminate |
| IN718 plume arm (superseded) | `rep2_in718_*` | same | uniform 20-30 µm plume cut |
| IN718 blind keyhole | `rep4_in718_s049_blind`, `rep4b_in718_s049_plume` | NIST 7th condition 227.2/106.2 µm | −25.2% (inside 30%, outside the stricter 20% gate) / −38.4% (outside both); out of scope, reported verbatim |
| SS316 arms | `q9_ss316_bundle`, `p2a_ss316_noplume`, `q9c_ss316_noplume_nosigma`, `t3_ss316_raoult` | Gan SS316 93.6 µm ±30% | 30/59/57 µm; Raoult arm pre-registered ~65 |
| Al6061 blind first contact | `al6061_p{416,520}_v{045,06,075,09}` | 8 Gan Al6061 anchors ±30% | (outcome not recorded in this manifest) |

## Reproducing

```bash
# route 1, no license: render + verify all prepin inputs
python3 validation/reproduce.py

# route 2, with a configured FLOW-3D license: solve everything (sequential,
# ~30-50 solver-hours) and export NPZs, then score the tree
python3 validation/reproduce.py --solve
python3 python/score_anchors.py --plan validation/scoring_plan.json --runs runs/validation --no-aiida
# optional: the manuscript's supplementary notebook over the same tree
F3D_RUNS_ROOT=$(pwd)/runs/validation jupyter nbconvert --execute ../lpno/src/notebooks/experimental_validation.ipynb

# route 3, from already-solved NPZs: pass their root as --runs / F3D_RUNS_ROOT
# (layout: <root>/<case_name>/npz/<case_name>.npz). The solved NPZs are solver
# output and are not distributed with this repository.
```

Experimental reference data is fetched from the original public repositories
by `lpno/src/notebooks/fetch_experimental_data.py` (NIST
mds2-2525, mds2-2718; Gan et al. 2021 SD1; Simonds et al. 2021 values).
