# Data deposit for "LP-NO" (laser melt-pool neural operators): open record

This record holds every trained model and evaluation behind the results tables
and figures of the manuscript A. Benoit and E. Iseli, *Provenance-tracked, experimentally validated neural operator surrogates of
three-dimensional laser melting with keyholes and free-form paths* (2026; preprint: **[TO FILL: arXiv link]**), the configuration of every simulation they
trained and were scored on, and the full AiiDA provenance graph that links
them. A reader can trace any printed number to the model, the evaluation and
the simulations it was computed from. With the companion restricted-access
record, which holds the simulated fields, a reader can also re-run any
evaluation and re-train any model on the same data.

Code repository: https://github.com/azbenoit/lpno_code, release **[TO FILL: tag]**.
This record: doi:10.5281/zenodo.23325086. Restricted-access record with the FLOW-3D WELD
output: doi:10.5281/zenodo.23325088. Licence of both: CC BY-NC-SA 4.0 (see `LICENSE.txt`).

## What is in the record

| file | size | contents |
| --- | ---: | --- |
| `lpno_provenance.aiida` | 6.20 GB | AiiDA archive (24,628 nodes, aiida-core 2.8.0): 46 model nodes with weights, 82 evaluation nodes with every metric, 1,147 simulation nodes with all metadata (array payloads removed, see below), 1,245 dataset collections, and every process node and link between them |
| `lpno_provenance_rounds.aiida` | 9.46 GB | addition: the FNO models of rounds 1 to 4 of the acquisition loop, plotted in the paper's acquisition-rounds figure, three seeds each: 36 model nodes with weights, their 72 evaluations, the training collections they were rebuilt from, and the processes that made them (see "The two-alloy models") |
| `lpno_provenance_evals.aiida` | 0.83 GB | addition: 10 evaluations of deposited models, the per-simulation evaluations behind the metrics printed under the weld-path, variable-speed, multi-beam and preheat prediction figures, with the one-simulation collections they score and the processes that made them (see "Which node backs which result"; the three of the preheat figure are re-scored evaluations, see "Re-scored evaluations") |
| `provenance_log.csv` | small | one row per evaluation behind a printed number or a comparison the manuscript draws (72 rows; an evaluation behind two results has a row for each, and the row with table `superseded` names a replaced evaluation kept for its model's provenance, see "Re-scored evaluations"): table or figure, the evaluation, model, scored dataset, training dataset and training workflow UUIDs, the model's seed, and the headline metrics in kelvin |
| `provenance_log_rounds.csv` | small | the same for the 72 evaluations of the per-round models |
| `provenance_log_evals.csv` | small | the same for the 10 evaluations of `lpno_provenance_evals.aiida` |
| `lpno_web_demo_models.tar`, `lpno_web_demo_models.tar.sha256` | 0.93 GB | weights-only bundle of the code repository's web demo (`self_hosted_web_demo/`, `scripts/assets.py fetch zenodo`): each served model's `u_network.0.pth` and `.hydra/config.yaml`, read from the model nodes of `lpno_provenance.aiida`, with a `trained_grid.json` per model folder. Steady page: `fno_spkh123_steady_32k_s44`, `gaot_spkh123_steady_32k`, `rigno_spkh123_steady_32k_fill3520`; straight-track page: `uni_fno_lab_b8_32k_s42`, `uni_gaot_lab_b8_32k_s42`, `uni_rigno_lab_b8_32k_s42`; free-form path page: `uni_fno_rand150_b2_12ch_35k_s43`, `uni_gaot_rand150_b2_12ch_35k_s43`, `rigno_rand150_newphys_35k_e8_12ch`. No FLOW-3D output; `lpno_web_demo_models.tar.sha256` lists the sha256 of every member |
| `experimental_data.zip` | small | the published measurements the manuscript compares against, as CSV files, with each source's citation, DOI, licence and credit in its README (the code repository's `reproducibility/experimental_data/`); two sources are cited there without their values, which their licences do not let us redistribute |
| `payload_manifest.csv` | small | one row per simulation: uuid, family, split, case_name, bytes, sha256; the index of the payload tarballs in the restricted record |
| `restricted_files_main.csv`, `restricted_files_rounds.csv`, `restricted_files_evals.csv` | small | one row per file moved out of each archive into the restricted record: node UUID, node type, process, path in the node, object sha256, bytes, reason |
| `SHA256SUMS` | small | checksums of every file in the record |
| `README.md`, `README_AiiDA.md`, `files_description.md`, `LICENSE.txt` | | this file, the AiiDA how-to, per-file one-line descriptions, and the CC BY-NC-SA 4.0 licence |

The archives here and the payload tarballs of the restricted record are two
views of one set. The archives hold the graph and the metadata; the tarballs
hold the arrays. They join on the simulation UUID, which is the file stem
inside every tarball and an attribute of every simulation node. The tools below
read both from one directory: put the restricted record's payload tarballs, or links
to them, beside the archives.

## What was simulated

FLOW-3D WELD (2025 R1 Update 1; the exact build string is recorded in each
case manifest where the solver wrote one) on a rectangular substrate under a
moving laser, 10 µm cells, one case per (power, scan speed, path, beam radius,
preheat, material) design point. Temperature, metal/gas fraction, liquid
fraction and the other fields below are stored at every exported time step on
the full grid. Two materials: Ti-6Al-4V (`ti64_v3_calibrated`, the validated
settings of the manuscript's Methods) and Inconel 718 (`inconel718`; the case
manifests name the file `in718_full`, which is the same file byte for byte).

The 1,096 simulations fall into 24 families. Every family backs at least one
printed number; the last column says which.

| family | sims | GB | grid | median frames | material | used by |
| --- | ---: | ---: | --- | ---: | --- | --- |
| ti64_multibeam40_capped | 79 | 25.3 | 150×100×40 | 214 | Ti-6Al-4V | paths-accuracy table (multi-beam rows) |
| ti64_rand150_newphys | 150 | 19.1 | 150×100×40 | 100 | Ti-6Al-4V | paths-accuracy table (free-form rows) |
| ti64_anchorbox_lhs150 | 150 | 16.1 | 90×60×55 | 187 | Ti-6Al-4V | preheat table, two-alloy table, closing figure |
| ti64_varspeed_newphys | 140 | 14.7 | 150×100×40 | 169 | Ti-6Al-4V | paths-accuracy table (variable-speed rows) |
| ti64_edge75_highenthalpy | 75 | 13.8 | 90×60×55 | 315 | Ti-6Al-4V | two-alloy table (Ti-6Al-4V anchor training set and test) |
| ti64_preheat60 | 107 | 10.2 | 90×40×50 | 245 | Ti-6Al-4V | preheat table |
| original_straight_path_data_calibrated_20260731 | 83 | 9.9 | 90×40×40 | 295 | Ti-6Al-4V | architecture table (steady and laboratory frames), frame-difficulty table |
| ti64_keyhole40, ti64_keyhole40_capped | 40 | 7.4 | 90×40×40 | 394–443 | Ti-6Al-4V | architecture table, frame-difficulty table |
| in718_loop_* (14 sub-families: 5 rounds × 2 acquisition strategies, plus replicate and top-up sets) | 248 | 37.7 | 90×60×55 | 146–246 | Inconel 718 | two-alloy table (training sets of the acquisition loop) |
| in718_testset_twoalloy | 24 | 2.7 | 90×60×55 | 146 | Inconel 718 | two-alloy table (frozen test set) |

Splits as recorded on each simulation (`split` in `payload_manifest.csv`):
889 train, 152 test, 48 val, 7 `duplicate_excluded` (members of a consumed
split collection that a de-duplication pass flagged; kept because the model
saw them).

## What is left out, and why

- **The FLOW-3D WELD output, which is in the restricted record.** Under the
  terms agreed with Flow Science, the solver's vendor, the solver output is
  shared for non-commercial use under an access agreement, in a separate
  restricted-access record. It holds the simulated fields (the payload
  tarballs) and the archive files that carry solver output: the evaluation
  previews and dumps (`lazy_preview_batch<n>.npz`, `eval_frames_dump.npz`)
  and the training snapshots (`constraints/data-<n>.vti`), which store the
  solver's fields next to the predictions, and the solver's console logs
  (`runhyd_tail.log`, and `aiida.out` of the solver jobs). Every node that
  lost files carries an `lpno_restricted_files` extra listing them, and
  `restricted_files_<archive>.csv` lists them all. Everything derived from
  the output as a number (metrics, validation curves, run summaries) stays
  here.
- **Site-specific names.** Working paths read `/home/user/...` and
  `/tmp/aiida-user/...`, hosts `workstation` or `localhost`, and the scheduler
  job owner `user`, in the database and in every stored text file. Nodes
  whose content changed this way carry no `_aiida_hash` extra.
- **Array payloads inside the archive.** Each of the 1,147 simulation nodes in
  `lpno_provenance.aiida` carries a `data_evicted` extra and no repository
  files; the 1,096 whose arrays a model trained on or was scored on are in the
  tarballs, and the 51 others (members of the same collections that the
  pipeline had flagged out of training) are metadata only. Keeping the arrays
  in the tarballs leaves the archive small enough to import, browse in a
  Jupyter session, or open without AiiDA's plugins. Nothing else about the
  node was changed: attributes, extras, links
  and UUID are the ones the paper's numbers were computed against.
- **Optimizer states.** Model nodes ship the final weights, `u_network.0.pth`;
  `fno_steady_edge75` also carries its best-validation weights,
  `u_network.best.pth`. `optim_checkpoint.0.pth` was removed from
  every model; it serves only to resume training. The same two file kinds were
  also removed from the evaluation jobs' uploaded inputs and the training jobs'
  retrieved folders, which held copies; a `deposit_trimmed` extra on every
  affected node lists what was removed. This is what takes the archive from
  18 GB to 6.8 GB; moving the solver output to the restricted record and
  replacing the two-alloy models (below) brings it to 6.19 GB, and the uniform-drawing re-run (below) to 6.20 GB.
- **Everything the paper does not print.** The source store holds about 5,500
  simulations, 640 models and 1,100 evaluations. This record is the subgraph
  reachable from the 45 models of the results tables (46 model nodes with one
  fine-tune parent) and the 68 evaluations
  of `provenance_log.csv`: the simulations those models trained on and were scored on, and nothing else.
  Superseded runs, ablations that back no printed number, and the 58
  simulations flagged out of training by the pipeline are absent. Fourteen
  further nodes are by-products of those models' training workflows and could
  only go together with their models; see "By-products of the training
  workflows".

## Reading the arrays without AiiDA

The tarballs are in the restricted record. Each `payloads_<family>.tar`
member `<uuid>.npz` is a `numpy.savez_compressed`
file. Member names and shapes, with `nt` frames and grid `(nx, ny, nz)`:

| member | dtype, shape | meaning |
| --- | --- | --- |
| `x`, `y`, `z` | float32 `(nx,)`, `(ny,)`, `(nz,)` | cell-centre coordinates, m; z increases upward, the metal surface is at the `surface_z_m` of the manifest |
| `time_s` | float64 `(nt,)` | time of each stored frame, s |
| `power_w`, `scan_speed_m_per_s` | float64 scalars | laser power, W; nominal scan speed, m/s |
| `case_name`, `frame_stride` | scalar | the solver case name; 1 unless frames were thinned at export |
| `Temperature` | float32 `(nt, nx, ny, nz)` | K |
| `Fraction Of Fluid` | float32 `(nt, nx, ny, nz)` | metal volume fraction (1 metal, 0 gas) |
| `Solid Fraction` | float32 `(nt, nx, ny, nz)` | solid fraction of the metal; liquid fraction is `1 - Solid Fraction` where `Fraction Of Fluid > 0` |
| `Melt Region`, `Melt Region Contour` | float32 `(nt, nx, ny, nz)` | solver melt-region flags |
| `Pressure` | float32 `(nt, nx, ny, nz)` | Pa |
| `Velocity` | float32 `(nt, nx, ny, nz, 3)` | m/s |
| `Cell Volume Fraction`, `Heat Absorption Rate`, `Heat Flux Spatial Distribution`, `Volume Fraction After AVRCK Adjustment` | float32 `(nt, nx, ny, nz)` | further solver outputs, kept as written |

```python
import numpy as np, tarfile, json
with tarfile.open("payloads_ti64_rand150_newphys.tar") as t:
    z = np.load(t.extractfile("<uuid>.npz"))
    manifest = json.load(t.extractfile("<uuid>.manifest.json"))
T = z["Temperature"]            # (nt, nx, ny, nz), kelvin
melt = (z["Fraction Of Fluid"] > 0.5) & (z["Solid Fraction"] < 0.5)
```

`<uuid>.manifest.json` is the FLOW-3D case manifest: `input` (the full JSON
configuration the case was rendered from: geometry, laser, material, mesh,
simulation), `derived` (beam path samples and speeds, mesh bounds, Fresnel
coefficients, recoil model, ambient pressure, output cadence), `material`
(name, file, the property tables' provenance), `run` (completion status, wall
and CPU seconds, cycles), and `units` (SI throughout).

## Reproducing a number

Every table row and figure point resolves to an evaluation node in the archive, which links
to its model node and the dataset collection it scored; `provenance_log.csv`
lists them row by row with every UUID. Two routes re-run them, both from the
code repository at the tagged release above, with its environment installed by
`tools/setup_env.py` (pinned to the versions the models were trained with).

**With AiiDA** (`README_AiiDA.md`). Import the archive (about 5 minutes), put
the arrays of the simulations you need back into their nodes
(`aiida-pino restore-payloads`), and re-run an evaluation
(`aiida-pino submit eval`) or a training (`aiida-pino resubmit`). The re-run
is recorded in the same provenance graph, linked to the deposited model and
data. No message broker is needed: `aiida-pino --foreground` runs the job in
the terminal.

**Without AiiDA** (no import, no profile, no broker; `reproducibility/rerun/README.md` in the
code repository). `reproducibility/rerun/export_deposit.py` reads the archive and the tars in
place and writes the exact inputs the published models were trained and
scored on; `reproducibility/rerun/replay_overrides.py` replays a deposited model's recorded
settings; `reproducibility/rerun/run_local.sh` runs a training or an evaluation on one local
GPU and `reproducibility/rerun/submit.sh` on a Slurm cluster. The re-run itself is not
recorded in AiiDA. The two routes agree: the two-alloy FNO models were scored
both ways, on a GPU without AiiDA and through AiiDA on a CPU, and the results
match to 4e-6 K in temperature MAE and 3e-5 K in beam-box MAE over all 92
evaluations.

Either way, re-running an evaluation needs the model's weights (in the
archive) and the scored split's arrays (in the restricted record's tarballs;
2.7 GB for the IN718 test split); re-training needs the training and validation splits (42 GB for
one half of the two-alloy acquisition) and a CUDA GPU (about 40 min for a
32,000-step FNO run on a V100).

## Which node backs which result

Each table and figure has a group of its evaluations, `lpno/evaluations/<label>`,
and each table a group of its models, `lpno/models/<label>`. The labels are those of
the manuscript's LaTeX source, so they survive renumbering. `provenance_log*.csv` list
the evaluations row by row, with the model, its seed and every UUID.

| label | result | models | evaluations, scored on |
| --- | --- | --- | --- |
| `tab:arch-steady` | architecture table, straight path | 18: FNO, GAOT and RIGNO, each in the steady and the laboratory frame, three seeds each; the laboratory-frame ones are the uniform-drawing re-run | 18, the straight-path test split: the nine of the quasi-steady block (re-scored as described in "Re-scored evaluations") and the nine of the laboratory frame, every frame at full resolution |
| `tab:frame-difficulty` | supplementary frame-difficulty table | the 9 laboratory-frame models of `tab:arch-steady` | 9, the same split, each laboratory model's predictions averaged over time and scored against the quasi-steady truth; the native rows are the evaluations of `tab:arch-steady` |
| `tab:paths-accuracy` | weld-path table | 9: FNO, GAOT and RIGNO on the free-form (`rand150`), variable-speed and multi-beam path sets (all but the variable-length and multi-beam RIGNO from the uniform-drawing re-run); the group also holds the parent the multi-beam RIGNO was fine-tuned from (the multi-beam FNO and GAOT were fine-tuned from the variable-length ones); 10 nodes | 9, the three path test splits |
| `tab:preheat` | beam-radius and preheat table | 6: FNO, GAOT and RIGNO on the beam-radius sweep (`ti64_anchorbox_lhs150`) and on the preheat set (`ti64_preheat60`) | 6, the two test splits: one per model, run at inference batch size 1 and re-scored (see "Re-scored evaluations") |
| `tab:twoalloy` | two-alloy table | 12: the Ti-6Al-4V-only FNO (`fno_steady_edge75`); the alloy-label FNO trained on the maximin half, the bandit half and both halves (joint), round 5, seeds 42, 43 and 44; the joint GAOT and RIGNO, seed 42 | 24, the IN718 test split and the Ti-6Al-4V test split |
| `fig:twoalloy-pred` | two-alloy predictions figure | the joint RIGNO | 2: the IN718 extrapolation slice (its first preview batch is the IN718 panel) and the Ti-6Al-4V test split |
| `fig:closing` | depths against measurement | the joint RIGNO | 2: the seven IN718 experimental conditions and the Ti-6Al-4V test split, each with a frame dump (`eval_frames_dump.npz`, in the restricted record) from which the depths are read |
| `fig:al-rounds` | acquisition-rounds figure (in `lpno_provenance_rounds.aiida`) | 36: the three FNO series of rounds 1 to 4, three seeds each | 72, the IN718 and Ti-6Al-4V test splits |
| `fig:lineage` | the supplement's lineage figure | the variable-length FNO of `tab:paths-accuracy`, whose ancestry it draws | 1, that model's evaluation on the free-form test split (also in `tab:paths-accuracy`) |

The acquisition-rounds figure also plots round 5, which is the `tab:twoalloy` FNO rows,
and round 0, which is the Ti-6Al-4V-only FNO's `tab:twoalloy` evaluations.

The 10 evaluations of `lpno_provenance_evals.aiida` are listed in `provenance_log_evals.csv` (they are in no group): those of the tabled `tab:paths-accuracy` and `tab:preheat` models on the single simulation each prediction figure shows (`fig:rp-artifact`, `fig:varspeed-pred`, `fig:multibeam-pred`, `fig:preheat-pred`), each with the evaluation profile of that model's published evaluation; the three of `fig:preheat-pred` replace earlier evaluations (see "Re-scored evaluations"). Import the addition after the main archive; the deposited nodes it refers to enter it as copies and are skipped on import.

## Re-scored evaluations

The models take the normalized line energy H* = (P/√|V|)/H_max as an input, where H_max is the
largest value of P/√|V| over the training split. An earlier version of the standalone evaluation
recomputed H_max over the split it scored, so a single-alloy model saw H* scaled by the ratio of
the two maxima: 0.97 on the straight-path and beam-radius test splits, 1.45 on the preheat test
split, and up to 3.99 on the one-simulation evaluations of the preheat prediction figure. The
pipeline now records H_max beside the weights, and aiida-pino supplies it to the evaluation of an
older model from that model's training simulations.

The 18 evaluations behind `tab:preheat` (6), the quasi-steady block of `tab:arch-steady`, whose
evaluations `tab:frame-difficulty` also prints (9), and `fig:preheat-pred` (3, in
`lpno_provenance_evals.aiida`) were re-scored this way on a CPU, with the inputs of the
evaluations they replace: the same model, collection, evaluation profile and derivation profile.
Each new evaluation names the one it replaces in its extra `hnorm_rescore_of` and records the
denominator it used in `evaluation_metadata.metrics.h_norm`. Three of the replaced evaluations,
replayed on the same CPU with their original denominator, reproduced every value the manuscript
printed from them.

The replaced evaluations were removed, together with unprinted evaluations of the same models
that only duplicated printed ones: the inference-batch-2 twins of `tab:preheat`, and seven
laboratory-frame evaluations of `tab:arch-steady` on a subset of the frames, which the full-frame
evaluations of `lpno_provenance_evals.aiida` had already replaced. One replaced evaluation stays:
`1c322ce6`, made by the training workflow of the printed model `rigno_spkh123_steady_32k_s43`, so
it can only go with that model. It is in the group `lpno/evaluations/superseded`, carries the
extra `lpno_superseded_by` (its replacement, `563a655b`), and is the row with table `superseded`
in `provenance_log.csv`.

## The uniform-drawing re-run

The laboratory-frame models of `tab:arch-steady` (and so the laboratory rows of `tab:frame-difficulty`), and the
FNO and GAOT models and the variable-speed RIGNO model of `tab:paths-accuracy`, were trained again with each
training frame drawn uniformly from the whole training set, on the grid subsampled by two along x and y, and scored
on every frame of each test split at full resolution. The multi-beam FNO and GAOT were fine-tuned again from the new
variable-length models. These 16 models and their 31 evaluations (25 in `lpno_provenance.aiida`, the 6 behind the
weld-path, variable-speed and multi-beam prediction figures in `lpno_provenance_evals.aiida`) replace the earlier
ones in the groups, and the earlier models of those rows are not in the record. The variable-length RIGNO of
`tab:paths-accuracy` is the earlier model. The re-run models were trained and scored through AiiDA on a GPU
cluster; their working paths read `/tmp/lpno/uniform_rerun/cluster_00000001/...` and their hosts `hst` or
`localhost`. Their optimizer states and best-validation weights were removed like every other model's
(`deposit_trimmed`).

## The two-alloy models

The models of `tab:twoalloy`, `fig:twoalloy-pred`, `fig:closing` and `fig:al-rounds`
were retrained after the acquisition loop with an alloy label as an
extra input, 1 for IN718 and 0 for Ti-6Al-4V, and with the material constants of each
alloy in the normalized enthalpy H*. The loop's own models, which received no alloy
label, are not in the record. Each model replays the recorded settings of the loop's
round-5 models: three seeds (42, 43, 44) for every FNO, one (42) for GAOT and RIGNO.
Round 5 trains on the deposited round-5 training sets. Each earlier round trains on the
round-5 training set restricted to the simulation families (`batch_keyword`) acquired
up to that round, composed from the deposited simulations and derived with the round-5
settings. Round 1 includes the five replacement simulations per half, which were
simulated after the loop's own round-1 models had been trained.

The models were trained on a GPU cluster without AiiDA (`reproducibility/rerun/README.md` in the
code repository) and then registered in an AiiDA profile holding the imported archive.
Each run was imported with `aiida-pino submit import-model` (final-step weights) and
evaluated through AiiDA on the CPU. Every evaluation sets
`++custom.eval_per_case_melt_window=true`, recorded in its Hydra overrides. The liquid
fraction behind the melt-pool IoU then uses each case's own solidus and liquidus,
1523.15 and 1608.15 K for IN718 and 1873.15 and 1923.15 K for Ti-6Al-4V, read from
the case manifests. The Ti-6Al-4V-only FNO's two evaluations also set
`++custom.h_norm_denom=425.86942258945584`, its own training normalization of H*.
The joint RIGNO's `fig:closing` evaluations keep a frame dump (`PINO_DUMP_FRAMES=1`
in their recorded environment).

The round-5 models and every evaluation above are in `lpno_provenance.aiida`; rounds
1 to 4 are in `lpno_provenance_rounds.aiida`. Deposited nodes the addition uses enter
it as metadata-only copies, and the arrays stay in the payload tarballs. Import it
after the main archive:

```bash
verdi archive import lpno_provenance.aiida lpno_provenance_rounds.aiida lpno_provenance_evals.aiida
```

Three things differ for these models. Their files sit under `model/` in their
repository (`model/u_network.0.pth`, `model/.hydra/`). The training workflow column of
the provenance logs names the import workflow, since the training itself ran outside
AiiDA. And they were registered under neutral names: installation and working paths
read `/tmp/lpno`, the host `localhost` and the user `user`, and the cluster path
prefixes in the models' `.hydra` files read `/work`.

The training seed matters: the three seeds of one FNO configuration differ by up to
25 K in IN718 beam-box MAE and 0.016 in melt-pool IoU, and a retraining with the same
seed is not bit-identical either. Evaluations of a given model reproduce to a relative
1e-5.

## By-products of the training workflows

Fourteen nodes in `lpno_provenance.aiida` back no printed number. Each one is an
output of the training workflow that produced a deposited model, and AiiDA deletes an
output only together with the process that made it, so removing any of them would
remove that model too. They are in no `lpno/evaluations/<table>` group.

All fourteen are automatic evaluations. A training workflow (`TrainEvalWorkChain`)
scores its model once, right after training. Without `--full-frames` that score uses
about 25 evenly spread frames per case, a health readout, and the printed numbers come
from separate full-frame evaluations of the same models, which are in the groups.
These readouts are not comparable with the printed values.

| evaluation (UUID prefix) | training workflow of | scored split | cases |
| --- | --- | --- | ---: |
| `ef91ee83` | `fno_spkh123_steady_32k` (`tab:arch-steady`) | straight-path validation | 14 |
| `b79084ee` | `gaot_spkh123_steady_32k` (`tab:arch-steady`) | straight-path validation | 14 |
| `e7a4fac6` | `rigno_spkh123_steady_32k_fill3520` (`tab:arch-steady`) | straight-path validation | 14 |
| `993087d8` | `gaot_spkh123_steady_32k_s43` (`tab:arch-steady`) | straight-path test | 13 |
| `691a2279` | `gaot_spkh123_steady_32k_s44` (`tab:arch-steady`) | straight-path test | 13 |
| `1c322ce6` | `rigno_spkh123_steady_32k_s43` (`tab:arch-steady`) | straight-path test | 13 |
| `8e42df88` | `rigno_mb80_ft6k_inera_v5` (`tab:paths-accuracy`) | multi-beam validation | 4 |
| `43f71457` | `fno_anchorbox_steady_32k` (`tab:preheat`) | beam-radius sweep validation | 9 |
| `de383364` | `gaot_anchorbox_steady_32k` (`tab:preheat`) | beam-radius sweep validation | 9 |
| `3a802548` | `rigno_anchorbox_steady_32k` (`tab:preheat`) | beam-radius sweep validation | 9 |
| `ad905418` | `fno_preheat107_cond_32k` (`tab:preheat`) | preheat test | 20 |
| `f1ba2ab2` | `gaot_preheat107_cond_32k` (`tab:preheat`) | preheat test | 20 |
| `4225e64c` | `rigno_preheat107_cond_32k` (`tab:preheat`) | preheat test | 20 |
| `da7cd88b` | `fno_steady_edge75` (`tab:twoalloy`) | Ti-6Al-4V test (beam-radius sweep) | 38 |

`1c322ce6` differs from the other thirteen: it scored every case and was the printed quasi-steady RIGNO evaluation of seed 43 until `563a655b` replaced it (see "Re-scored evaluations"); it is in `lpno/evaluations/superseded`.


## Citation

```bibtex
@misc{benoit_lpno_2026,
  author = {Benoit, Alix and Iseli, Elia},
  title  = {Provenance-tracked, experimentally validated neural operator surrogates of
            three-dimensional laser melting with keyholes and free-form paths},
  year   = {2026},
  note   = {Preprint, arXiv:[TO FILL]}
}
```

**[TO FILL: BibTeX for this record with its DOI.]**
Please cite the paper for the method and this record for the data.
