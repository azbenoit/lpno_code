# Served assets (`data/`)

The model checkpoints and ground-truth cases the backend serves, one directory
per demo page. The heavy files are not committed: git tracks this layout, each
model's `meta.json` (and `flow3D_params.yaml` / `grid.npz` where present) and
these READMEs. A fresh checkout gets the rest in two parts:

- the model weights (required, 0.93 GB): `python scripts/assets.py fetch zenodo`,
  from the open Zenodo record (doi:10.5281/zenodo.23325086);
- the ground truth (optional): `python scripts/assets.py fetch zenodo-ground-truth
  --token-file <file>`, from the restricted-access record
  (doi:10.5281/zenodo.23325088), once access has been granted on its Zenodo page.
  Without it every page still predicts, and the ground-truth controls are greyed out.

`python scripts/assets.py verify` reports what is installed.

## Layout

```
data/
  steady-state/                    index.html — steady P–V melt pool  (/infer, /infer/gt_compare)
    models/{fno_cal,gaot_cal,rigno_cal}/
    ground_truth_calibrated/{val,test}/var_1_time_steps30_avg30.npz
  straight_path_time/              line.html  — straight track in time (/infer_straight*)
    models/{fno_cal,gaot_cal,rigno_cal}/
    ground_truth_calibrated/{val,test}/<case>.npz + <case>.manifest.json
  random_path/                     path.html  — free-form path in time  (/infer_path*)
    models/{fno_newphys,gaot_newphys,rigno_newphys}/
    ground_truth/rand150_newphys_test/<case>.npz + <case>.manifest.json
```

Each model folder holds `u_network.0.pth`, `.hydra/config.yaml` (the training
config the loader is built from) and `meta.json`; steady folders add
`flow3D_params.yaml` (the served geometry), and a folder may pin its own grid
with `grid.npz` (the straight RIGNO does: 90×40×30, the crop it was trained and
scored on). Any folder with a checkpoint under `data/<page>/models/` is
discovered and appears in that page's model dropdown.

## `meta.json`

```json
{
  "label": "FNO",                       // dropdown text
  "type": "fno",                        // informational; the architecture comes from the hydra config
  "order": 1, "default": true,          // dropdown order / pre-selected
  "model_name": "fno_spkh123_steady_32k_s44",  // label of the model node in the open record's AiiDA archive
  "physics": {                          // what the adapter needs to build the inputs
    "p_watts":    {"min": 40.0, "max": 215.0, "default": 150.0},
    "v_scan_mps": {"min": 0.101, "max": 0.96, "default": 0.6},
    "h_norm_denom": 275.0539301382,     // the TRAIN split's max of P/sqrt(V): the fitted H_norm denominator
    "power_weighted_dose": true,        // path_dose channels scaled by P, as in training
    "presets_dir": "rand150_newphys_test"   // free-form page only: this model's held-out cases
  }
}
```

`h_norm_denom` and `power_weighted_dose` are not decoration: the first is what
the training pipeline fitted (not `ref_scales.H` from the config, which is
inert unless `h_norm_mode: ref_scale`), and a missing second silently serves
unweighted dose to a model trained on weighted dose. Both were read from the
training runs when the folders were built.

## Ground truth

- **Steady and straight** share one catalog: the 27 held-out cases (14 val / 13 test)
  of `ti64_straight_path_calibrated_with_keyholes`, a straight 0.6 mm track on Ti-6Al-4V at P = 40–215 W, V = 0.10–0.96 m/s,
  simulated with FLOW-3D WELD using the paper's validated simulation settings. The steady set
  stores one 30-frame time-averaged 61×40×40 beam-following window per case
  (each case's own window; the server shifts its prediction to it); the straight
  set stores 20 frames per case on the full 90×40×40 grid, plus the FLOW-3D case
  manifest the conditioning channels are built from.
- **Free-form**: the 22 held-out test paths of `ti64_rand150_newphys`, every stored
  frame (14 to 400 per path) on the 75×50×40 training grid (the source cases are 150×100×40;
  the stride is applied at build time so prediction and ground truth compare
  cell for cell).

Ground truth is served for a request that lands within 1 W / 0.005 m/s of a
catalog case (steady, straight) or that names a preset (free-form).

Without ground truth, the two time pages take their grid from
`data/straight_path_time/grid.npz` and `data/random_path/grid.npz` (the x, y, z
cell-centre axes of the training mesh, in metres); the steady page's grid comes
from each model's `flow3D_params.yaml`.

## Rebuilding the bundles (authors only)

The served sets are exports from an AiiDA profile holding the deposited archive
and, for the ground truth, the restored payloads of the restricted record; the
scripts are the record of how:

- `scripts/build_calibrated_steady_gt.py` and `scripts/build_calibrated_straight_gt.py`
  take a model and read every derivation knob (stride, z-crop, moving frame,
  averaging) from that model's own evaluation node, because the hydra config's
  `geometry.resolution` is nominal.
- `scripts/build_newphys_assets.py` exports the 22 free-form test cases.
- `scripts/build_ground_truth_from_deposit.py` builds all three ground-truth sets from the
  restricted record's payload tars and `payload_manifest.csv`, with no AiiDA store, through
  the same per-case code. Run it where there is disk to spare: it extracts each case in turn.
  `--splits train val test` also builds the 96 training cases, which the published bundle
  leaves out; the pages list whichever split folders exist.
- Model folders are the node's `u_network.0.pth` + `.hydra/config.yaml`, with a
  `meta.json` written by hand from the training config (see above).

Then `python scripts/assets.py bundle --part models` and `--part ground_truth`
tar each part and write the `.sha256` manifest next to it.
