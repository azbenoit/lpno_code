# Figure generators

Every figure of the manuscript is drawn by a script or notebook in this folder, from
the data in `data/`, the open deposit record and, for the ground-truth panels of five
figures, the restricted record. `reproducibility/figures.ipynb` at the repository root
runs them all in order and compares the result with the published PDFs.

## Running

Python 3.10 with numpy, matplotlib, Pillow and, for the 3-D renders, pyvista with
off-screen rendering. `fonts/` ships Liberation Sans, the one typeface of every figure
(`figstyle.apply()` registers it; licence in `fonts/LICENSE-LiberationSans.txt`).

```bash
cd reproducibility/figures
LPNO_REPLACE_FIGURES=1 python make_calib_keyhole_panels.py                  # writes ../Figs/<name>.pdf
LPNO_FIG_DIR=/tmp/figs LPNO_REPLACE_FIGURES=1 python make_closing_figs.py   # elsewhere
```

`figstyle.save` refuses to overwrite an existing PDF unless `LPNO_REPLACE_FIGURES=1`
is set, and writes to `LPNO_FIG_DIR` when that is set (default `../Figs`). All style
lives in `figstyle.py`: one typeface at one size (10 pt printed), panels authored at
the width they are placed at, PDF at 600 dpi.

## What draws what

| script | figures (manuscript label) | data |
| --- | --- | --- |
| `provenance_diagram.ipynb` | `three_pillars`, `provenance_chain`, `collection_roles` (fig:pillars, fig:prov-chain, fig:prov-roles) | none (diagrams) |
| `make_lineage_fig.py` | `lineage` (fig:lineage) | none (diagram of one stored chain) |
| `make_channel_fig.py` | `channels_*` (fig:channels) | a synthetic variable-speed path; `linear_response` from `lpno/src` on `PYTHONPATH` |
| `make_calib_summary_parity.py`, `make_calib_regime_panel.py` | `calib_summary_parity`, `calib_summary_regime` (fig:calib-materials) | `data/figure_data/calib_summary_parity.json`, `data/calib_regime_points.json` |
| `make_calib_xray_overlays.py` | `calib_xray_*`, `calib_spot_overlay_*` (fig:calib-xray) | `data/figure_data/<name>.json` and `<name>_radiograph.png` |
| `make_calib_keyhole_panels.py` | `calib_keyhole_ladder`, `calib_speed_ladder` (fig:calib-keyhole) | `data/figure_data/` |
| `make_calib_transition_panels.py` | `calib_transition_ladder`, `calib_aspect_ratio` (fig:calib-transition) | `data/figure_data/` |
| `make_calib_cross_sections.py` | `calib_metric_defs*`, `calib_spot_size_*`, `calib_mesh_*` (fig:calib-metrics, fig:calib-spot, fig:calib-mesh) | `data/figure_data/` |
| `make_calib_history_panels.py` | `calib_depth_history`, `calib_absorptance_history`, `calib_spot_depth`, `calib_spot_abs` (fig:calib-history) | `data/figure_data/` |
| `calibration_figures.ipynb` | none: computes the validation panels' values from the validation simulations and the measurement files in `flow3D_scripts/validation/anchors/` | the validation simulations (not deposited) |
| `prepare_path_frames.py` | none: the ground-truth frames and laser paths of Figs 7-9 | the restricted record's payload tars (`LPNO_RESTRICTED`); writes `data/path_frames/*_gt*.npz` (git-ignored) and `tracks.json` |
| `make_path_renders.py` | `rp_artifact_*`, `varspeed_fno_{gt,pred}_*`, `multibeam_*` (fig:rp-artifact, fig:varspeed-pred, fig:multibeam-pred) | `data/path_frames/`: committed predictions, ground truth from `prepare_path_frames.py` |
| `frame_metrics.py` | none (the Figs 7-9 subcaption metrics, over the instant shown; also used by `make_gaot_smoothing.py`) | per-case evaluation dumps: writes `data/frame_metrics_figs7_9.json` |
| `make_gaot_smoothing.py` | `gaot_smoothing_{before,after,scale}` (fig:gaot-smoothing) | `data/path_frames/`: Fig 7's GAOT frame, filtered by the script; `--metrics <eval_frames_dump.npz>` prints the caption's numbers from the case's evaluation dump |
| `make_varspeed_speed_profiles.py` | `varspeed_fno_speed_profiles`, `varspeed_fno_cbar_vertical` | `data/figure_data/varspeed_speed_profiles.json`, `data/path_frames/clims.json` |
| `prepare_truth_frames.py` | none: the ground-truth frames of Figs 10 and 12 | the restricted record (`LPNO_RESTRICTED`) and, for Fig 10, an AiiDA profile with the open archives and three restored payloads; writes `data/*_truth_frames.npz` (git-ignored) |
| `preheat_prediction_panels.ipynb` | `preheat3d_{hot,small,big}_{gt,pred}` (fig:preheat-pred) | `data/preheat_prediction_frames.npz`, `data/preheat_truth_frames.npz` |
| `make_twoalloy_fig.py` | `twoalloy3d_*` (fig:twoalloy-pred) | `data/twoalloy_prediction_frames.npz`, `data/twoalloy_truth_frames.npz` |
| `forward_ti64only_fno.py` | none: panel (e) of fig:twoalloy-pred, a CPU forward of the Ti-6Al-4V-only FNO | that model's weights and one evaluation preview (see its docstring) |
| `make_colorbars.py`, `make_render_keys.py`, `project_render_overlays.py` | `preheat3d_cbar`, `twoalloy3d_cbar`, `preheat3d_scale`, `twoalloy3d_scale` | clims in the script; `data/render_overlays.json` |
| `make_al_rounds_fig.py` | `al_rounds_*` (fig:al-rounds) | `data/al_rounds_alloyflag.json`, one row per deposited evaluation |
| `make_closing_figs.py` | `closing_parity_b`, `closing_parity_c`, `closing_xray`, `closing_legend` (fig:closing) | `data/{ti64,in718}_joint_rigno_alloyflag_depths.json`, `data/figure_data/closing_*` |
| `make_dataset_regimes_fig.py` | `dataset_regimes` (fig:dataset-regimes) | `data/dataset_regimes.json`: the melting regime aiida-pino assigns each simulation (`console.viz.classify_regime`) |

`data/figure_data/` holds the values plotted in the validation figures and the closing
radiograph panel, computed from the validation simulations and the published
measurements; the radiographs are frames of the NIST AM Bench record (credits in
`reproducibility/experimental_data/README.md`). The predictions in `data/` and
`data/path_frames/` are model outputs; the FLOW-3D ground truth they are drawn beside
is never committed (`.gitignore`) and is rebuilt from the restricted record.

`preheat_frames/` records how the committed preheat prediction frames were made. Its two scripts
ran against the authors' AiiDA store and are kept as a record; they do not run elsewhere.
