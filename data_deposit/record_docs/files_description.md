# files_description

One line per file of the two records of the deposit: the open record (provenance
graph, models, configurations) and the restricted-access record (the FLOW-3D WELD
output). The same file is in both.

## Open record

- `lpno_provenance.aiida`: AiiDA archive (aiida-core 2.8.0, 24,628 nodes, 6.20 GB): the full provenance graph behind the manuscript's results tables and figures; 46 model nodes with weights, 82 evaluation nodes with every metric, 1,147 simulation nodes with all metadata (array payloads removed and shipped in the restricted record's tarballs; files carrying solver output moved there too), 1,245 dataset collections, process nodes and links. See `README_AiiDA.md`.
- `README.md`: what the record contains, how the simulations were produced, the array layout of every NPZ, what was left out, and how to cite.
- `README_AiiDA.md`: versions, how to install and import, how to read the archive with and without the plugins, the group layout, how to walk from a printed number to its model, evaluation and simulations, and how to put the arrays back and re-run an evaluation or a training through AiiDA.
- `payload_manifest.csv`: one row per simulation: `uuid, family, split, case_name, bytes, sha256`; the index that joins the archive's simulation nodes to the tarball members of the restricted record.
- `provenance_log.csv`: one row per evaluation behind a printed number or a comparison the manuscript draws: table or figure, the UUIDs of the evaluation, model, scored dataset, training dataset and training workflow, the model's seed, and the headline metrics in kelvin.
- `lpno_provenance_rounds.aiida`: addition (9.46 GB): the FNO models of rounds 1 to 4 of the acquisition loop, three seeds each (36 models with weights, their 72 evaluations, the training collections rebuilt from the deposited simulations); import it after `lpno_provenance.aiida`. See README.md, "The two-alloy models".
- `provenance_log_rounds.csv`: one row per evaluation of those models, same columns as `provenance_log.csv`.
- `lpno_provenance_evals.aiida`: addition (0.83 GB): 10 evaluations of deposited models, the per-simulation evaluations behind the prediction-figure metrics, with the one-simulation collections they score; import it after `lpno_provenance.aiida`.
- `provenance_log_evals.csv`: one row per evaluation of that addition, same columns as `provenance_log.csv`.
- `lpno_web_demo_models.tar`, `lpno_web_demo_models.tar.sha256`: weights-only bundle of the code repository's web demo (`self_hosted_web_demo/`, `scripts/assets.py fetch zenodo`): each served model's `u_network.0.pth` and `.hydra/config.yaml`, read from the model nodes of `lpno_provenance.aiida`, with a `trained_grid.json` per model folder. Steady page: `fno_spkh123_steady_32k_s44`, `gaot_spkh123_steady_32k`, `rigno_spkh123_steady_32k_fill3520`; straight-track page: `uni_fno_lab_b8_32k_s42`, `uni_gaot_lab_b8_32k_s42`, `uni_rigno_lab_b8_32k_s42`; free-form path page: `uni_fno_rand150_b2_12ch_35k_s43`, `uni_gaot_rand150_b2_12ch_35k_s43`, `rigno_rand150_newphys_35k_e8_12ch`. No FLOW-3D output; `lpno_web_demo_models.tar.sha256` lists the sha256 of every member.
- `experimental_data.zip`: the published measurements the manuscript compares against, as CSV files with a README giving each source's citation, DOI, licence and credit.
- `restricted_files_main.csv`, `restricted_files_rounds.csv`, `restricted_files_evals.csv`: one row per file moved out of each archive into the restricted record: node UUID, node type, process, path in the node, object sha256, bytes, reason.
- `LICENSE.txt`: CC BY-NC-SA 4.0 legal code, with the copyright notice.
- `SHA256SUMS`: sha256 of every other file in the record.

## Restricted-access record

- `README.md`: what the record holds, the conditions of access, and how to use it with the open record.
- `restricted_files_main.tar`, `restricted_files_rounds.tar`, `restricted_files_evals.tar`: the files moved out of each archive (evaluation previews and dumps and training snapshots with the solver's fields, and the solver's console logs), one member `objects/<sha256>` per distinct file; the CSVs of the same name map them to their nodes.
- `lpno_web_demo_ground_truth.tar`, `lpno_web_demo_ground_truth.tar.sha256`: the optional ground truth of the code repository's web demo: the held-out cases of its straight-track pages (14 validation, 13 test) and the 22 test paths of its free-form page, built from the payloads above by `self_hosted_web_demo/lpno_web_demo_backend/scripts/build_ground_truth_from_deposit.py`.
- `payload_manifest.csv`, `files_description.md`, `LICENSE.txt`, `SHA256SUMS`: as in the open record.
- The payload tars below that are larger than 2 GiB (20 of the 24) are stored as 2 GiB parts, `<name>.part00`, `<name>.part01`, ...; rejoin each with `cat <name>.part?? > <name>` (README.md, "Rejoining the payload tars"). `SHA256SUMS` lists the parts and the rejoined tars.
- `payloads_ti64_multibeam40_capped.tar`: 79 multi-beam Ti-6Al-4V simulations (150×100×40 grid), one `<uuid>.npz` + `<uuid>.manifest.json` per case; training and test data of the multi-beam rows of the weld-path table.
- `payloads_ti64_rand150_newphys.tar`: 150 free-form random-path Ti-6Al-4V simulations (150×100×40); training and test data of the free-form rows of the weld-path table.
- `payloads_ti64_varspeed_newphys.tar`: 140 variable-speed free-form-path Ti-6Al-4V simulations (150×100×40); training and test data of the variable-speed rows of the weld-path table.
- `payloads_ti64_anchorbox_lhs150.tar`: 150 Ti-6Al-4V simulations on a Latin-hypercube design over power, speed and beam radius (90×60×55); training and test data of the preheat/beam-radius table, the two-alloy table and the closing figure.
- `payloads_ti64_edge75_highenthalpy.tar`: 75 high-enthalpy Ti-6Al-4V simulations extending the anchor box (90×60×55); training and test data of the two-alloy table's Ti-6Al-4V side.
- `payloads_ti64_preheat60.tar`: 107 preheated-substrate Ti-6Al-4V simulations (90×40×50); training and test data of the preheat table.
- `payloads_original_straight_path_data_calibrated_20260731.tar`: 83 straight-track Ti-6Al-4V simulations under the validated solver settings (90×40×40); training and test data of the architecture table and the frame-difficulty table.
- `payloads_ti64_keyhole40.tar`: 16 keyhole-regime straight-track Ti-6Al-4V simulations (90×40×40); part of the architecture table's training set.
- `payloads_ti64_keyhole40_capped.tar`: 24 keyhole-regime straight-track Ti-6Al-4V simulations with capped depth (90×40×40); part of the architecture table's training set.
- `payloads_in718_testset_twoalloy.tar`: 24 Inconel 718 simulations forming the frozen test set of the two-alloy table (90×60×55).
- `payloads_in718_loop_bandit_r01.tar` … `payloads_in718_loop_bandit_r05.tar`: Inconel 718 simulations acquired by the bandit strategy in rounds 1–5 of the acquisition loop (90×60×55); training data of the two-alloy table.
- `payloads_in718_loop_maximin_r01.tar` … `payloads_in718_loop_maximin_r05.tar`: the same for the maximin strategy.
- `payloads_in718_loop_banditrepl_r02.tar`, `payloads_in718_loop_maximinrepl_r02.tar`: round-2 replicate acquisitions of each strategy.
- `payloads_in718_loop_bandittop_r01.tar`, `payloads_in718_loop_maximintop_r01.tar`: round-1 top-up acquisitions of each strategy.

Every file name carries an extension, and every large file is a `.tar` (the NPZ
members are already deflate-compressed; a second compression gains nothing).
