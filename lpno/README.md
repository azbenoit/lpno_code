# lpno: neural-operator surrogates for laser melting

## Overview

`lpno` trains and evaluates neural-operator surrogates that predict the
temperature field, the metal-gas interface and the melt pool of a laser
melting process from its process parameters and laser path. Three
architectures are implemented on NVIDIA PhysicsNeMo and PyTorch:

- the Fourier neural operator (FNO), which works on the uniform grid in
  spectral space;
- the region interaction graph neural operator (RIGNO), a graph network on the
  grid points, ported to PyTorch from the JAX reference;
- the geometry-aware operator transformer (GAOT), which encodes the grid onto
  a fixed latent lattice, mixes it with a vision transformer and decodes it back.

The training data are thermo-fluid-dynamic FLOW-3D WELD simulations, exported
by `../flow3D_scripts` as one consolidated NPZ file per case with a JSON
manifest beside it. The package provides the NPZ datasource, the
conditioning channels built from the manifest (beam position and footprint,
accumulated dose, a linear conduction response, material and preheat inputs),
the train and evaluation pipeline driven by Hydra configs, and the evaluation
metrics (whole-domain, substrate, beam-box and melt-pool geometry).

The module names (`NS_Energy_laser_PINO*`) and the environment variable
prefix (`PINO_*`) are historical; the package no longer trains with a physics
loss (see "Deprecated physics loss").

Detailed practical guidance and a short roadmap are in
[`docs/training_and_evaluation.md`](docs/training_and_evaluation.md), and known
issues with their workarounds in [`potential_bugs.md`](potential_bugs.md). The
conventions behind the data pipeline (axes, normalisation, fills, channel
order) are listed with their source locations in
[`assumptions.md`](assumptions.md).

## Repository layout

```
lpno/
├── src/lpno/
│   ├── conf/hydra/      # Hydra configs (config_flow3D_* for FLOW-3D data)
│   ├── custom_utils/    # NPZ and VTK datasources, conditioning channels, losses
│   ├── equations/       # post-process nodes: liquid fraction, alpha mask, warmup
│   ├── models/          # entrypoints, RIGNO and GAOT architectures, pipeline/
│   ├── plotting/        # validators and plots
│   └── gpu_spool.py     # per-GPU job queue and memory ledger
├── src/notebooks/       # FLOW-3D validation, random-path inference, rollout summary
├── scripts/             # split builders, ensemble eval, timing, push checks
├── tests/               # pytest suite (CPU, no licence needed)
├── docs/                # training and evaluation guide
├── environment.yml      # pinned conda environment
└── pyproject.toml
```

## Installation

From the repository root, with a Python 3.10 interpreter:

```bash
python3.10 tools/setup_env.py --prefix ~/envs/lpno --torch cu128   # Ampere and newer
python3.10 tools/setup_env.py --prefix ~/envs/lpno --torch cu126   # Volta (V100)
python3.10 tools/setup_env.py --prefix ~/envs/lpno --torch cpu --with-aiida
```

This creates a virtual environment with torch 2.8.0 and every other package at
the pins of `environment.yml`, adds `torch-scatter` and `nvidia-physicsnemo-sym`,
installs `lpno` (and `aiida-pino` with `--with-aiida`) and checks that the
entrypoints import. A CPU build with AiiDA needs about 3 GB of disk.

The manual route, from this directory:

1. `conda env create -f environment.yml`, then `pip install -e .`.
2. `python scripts/install_physicsnemo_sym.py` installs `nvidia-physicsnemo-sym`.
   It builds the optional CUDA kernel only when `CUDA_HOME` is set; the kernel
   serves the fp16 loss scaler, and the published configs train in fp32.
3. `pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.8.0+cu128.html`
   (use `cu126` on Volta GPUs and older, together with torch from the CUDA 12.6
   index). RIGNO and GAOT fall back to native torch scatter operations without it,
   with identical results.
4. To use AiiDA, install `../aiida-pino` into the same environment
   (`pip install -e '../aiida-pino[web,dev]'`).

`../reproducibility/rerun/build_env.sbatch` builds the environment on a Slurm cluster.
Logging uses Weights & Biases: run `wandb login`, or pass
`custom.wandb_offline=true`.

## Data

The simulations and trained models are in the data deposit:

```bash
python3 ../data_deposit/scripts/download.py open                   # models, evaluations, configs
python3 ../data_deposit/scripts/download.py restricted --token-file <file> --join   # simulated fields, on request
```

`../reproducibility/` rebuilds the manuscript's tables and figures from these
records, and its `train_evaluate.ipynb` tutorial trains and evaluates a model on
a small synthetic dataset on a CPU, with and without AiiDA.

Each case is a consolidated NPZ with explicit `x`/`y`/`z` axes, `time_s`,
`power_w`, `scan_speed_m_per_s` and fields shaped `(nt, nx, ny, nz)`, plus a
`<case>.manifest.json` that records the laser path, beam radius, timing and
material. The grid and domain are read from the file. Point `custom.data_dir`
and `custom.val_data_dir` at a split directory (absolute paths: Hydra changes the
working directory), set `PINO_SIMULATIONS_DATA_ROOT`, or use an `aiida-pino`
dataset. Direct VTK directories are refused unless `PINO_ALLOW_VTK=1` (A36).

`scripts/build_random_path_split.py` builds a stratified train/val/test split
from `flow3D_scripts` batch outputs:

```bash
python scripts/build_random_path_split.py \
  --source /abs/path/to/batch_a --source /abs/path/to/batch_b \
  --dest /abs/path/to/split --frames 0 --seed 0
```

## Training and evaluation

Submit real trainings and evaluations through `../aiida-pino`, which records
every model and evaluation with its dataset and settings and books the GPU
through the same ledger as the spooler:

```bash
aiida-pino submit train-eval --train-dataset <dataset> --python-code <code label> \
  --training-profile @profile.json
aiida-pino submit eval --model <model> --dataset <dataset> --python-code <code label>
```

`../aiida-pino/README.md` lists the flags, and `aiida-pino profile-keys training`
the keys a training profile accepts.

The module entrypoints run the same pipeline directly, for smoke tests and debugging:

```bash
python -m lpno.models.NS_Energy_laser_PINO \
  --config-name=config_flow3D_RIGNO_randompath \
  custom.name=smoke training.max_steps=10 custom.wandb_offline=true \
  custom.data_dir=/abs/split/train custom.val_data_dir=/abs/split/val

python -m lpno.models.NS_Energy_laser_PINO_eval \
  --config-name=config_flow3D_RIGNO_randompath \
  custom.name=smoke_eval network_dir=outputs/<run dir> \
  custom.data_dir=/abs/split/test
```

An evaluation loads its defaults from `network_dir/.hydra/config.yaml`; explicit
command-line overrides still win. Always pass `--config-name`: all entrypoints
default to `config_NS_energy_PINO` (A20). Training aborts without CUDA unless
`PINO_ALLOW_CPU=1` is set.

The configs follow `config_flow3D_<ARCH>_<task>`, with `<ARCH>` one of `FNO`,
`RIGNO`, `GAOT` and the main tasks:

| suffix | data and frame |
| --- | --- |
| `_randompath` | arbitrary laser paths, lab frame, one frame per sample |
| `_pvlhs` | straight-path and process-space sweeps, lab frame |
| `_pvlhs_steady` | the same sweeps, time-averaged in a beam-centred frame |
| (none) | straight-path set, time-averaged in a moving reference frame (`config_flow3D_RIGNO`, `config_flow3D_GAOT`; the FNO counterpart is `config_NS_energy_PINO_flow3D`) |

The other configs are variants of these (schedules, kernels, lattice sizes,
loss terms); each states its purpose in its header comment. The configuration of
every model in the manuscript is stored with it in the deposit's AiiDA archive.

## Architectures

All three consume the grid layout and the same input and output key lists
(`custom.FNO_input_keys`, `custom.FNO_output_keys`), so loss, post-processing and
metrics are shared.

**FNO** (`custom.model_family=fno`) is the PhysicsNeMo FNO. It needs a uniform
grid. `PINO_FNO_SUPERRES` controls the evaluation-time adaptation that lets an
FNO trained on a strided grid score a finer one.

**RIGNO** (`custom.model_family=rigno`, `models/rigno_arch.py`) is a port of
[arXiv:2501.19205](https://arxiv.org/abs/2501.19205). The grid points are the
physical nodes, a subset are regional nodes, and an encode-process-decode graph
network passes messages between them. The graph depends only on the grid, so it
is built once per grid shape and cached under `arch.rigno.cache_dir`. Main
`arch.rigno.*` options:

| option | meaning |
| --- | --- |
| `processor_steps` | regional message-passing steps (18 in the base config) |
| `num_graph_variants` | pre-built regional graphs drawn per training forward; evaluation uses variant 0 |
| `grad_checkpointing` | recompute activations in backward; required with 18 steps and 8 variants on a 24 GB card |
| `node_latent_size`, `edge_latent_size` | latent widths (96) |
| `subsample_factor`, `rnode_count` | regional-node density, or a fixed count |
| `k_p2r`, `k_r2p`, `rmesh_levels` | encoder/decoder neighbours and multi-scale regional edges |
| `p_edge_masking`, `inference_edge_masking` | edge dropout in training, and optionally at evaluation for an ensemble |
| `film_keys` | broadcast scalars (`P`, `V_scan`, `H_norm`) used for feature-wise modulation of the regional latents |

**GAOT** (`custom.model_family=gaot`, `models/gaot_arch.py`) is a port of
[arXiv:2505.18781](https://arxiv.org/abs/2505.18781). An attentional graph
encoder lifts the grid onto a fixed latent lattice, a patched vision
transformer mixes the tokens, and a graph decoder returns to every physical
node. The layers in `models/gaot_layers/` are vendored from GAOT-3D under the
MIT licence (see `../NOTICE`). The phys-latent graphs are built once with
`scipy.spatial.cKDTree` and cached under `arch.gaot.cache_dir`. Coordinates are
normalised per axis, so size `latent_tokens` against the index aspect of the
grid the model receives (A41). Main `arch.gaot.*` options: `latent_tokens`,
`patch_size`, `lifting_channels`, `hidden_size`, `ffn_hidden_size`,
`num_layers`, `num_heads`, `k_encoder`, `k_decoder`, `neighbor_strategy`
(`knn`, `radius`, or `bidirectional`, the union of the two as in GAOT-3D),
`decoder_stencil` (`knn`, or `trilinear`: each node reads the 8 corner tokens of
its latent cell with trilinear weights, so the decoded field is continuous across
latent cell walls), `decoder_kernel` (`agno`, or `none` with the trilinear
stencil: plain interpolation of the token features), `neighbor_metric` (per-axis
scales for the neighbour searches only, e.g. the physical half-extents over the
largest one for isotropic searches), `use_pointwise_skip`,
`use_geoembed_encoder`, `use_geoembed_decoder`,
`positional_embedding` (`rope` needs `rotary-embedding-torch`), and
`output_bounds`.

Train RIGNO and GAOT in fp32 (`amp.enabled=false`): fp16 autocast can overflow
on this data, and the RIGNO random-path config still enables it (see the guide).

## Conditioning channels

On arbitrary paths the model runs in the lab frame
(`custom.moving_ref_frame=false`) with one frame per sample
(`custom.time_as_batch=true`), and the laser trajectory enters as input
channels computed from the manifest only, never from solver output (A28, A29):

- `beam_x`, `beam_y`: beam centre at the frame time;
- `beam_dist`: the instantaneous Gaussian beam footprint (peak 1);
- `path_dose`, `path_dose_recent`: accumulated and exponentially decaying
  exposure, the thermal-memory channels; `custom.power_weighted_dose` scales
  them by power;
- `beam_s`, `beam_n`, `beam_speed` and their arc-length variants: along-track
  and cross-track coordinates and local speed;
- `conduction_field`: the linear conduction response to the deposited beam,
  cached per case under `PINO_CONDUCTION_CACHE`;
- `preheat_T0`, `beam_shape_uniform`, `focus_factor`, `beam_radius`: case
  constants, broadcast;
- `mat_*`: per-case material properties (density, heat capacity, conductivity,
  latent heat, liquidus, absorptivity) for multi-material datasets.

A channel is built only when its name is in `custom.FNO_input_keys`, and a
requested channel without a manifest raises an error. `H_norm = P/sqrt(|V_scan|)`
is normalised by its maximum over the training split; training writes the value
to `<network_dir>/h_norm_denom.json` and evaluation reads it back (A9). Training
also writes `<network_dir>/input_fields.json` with the effective settings that
change channel values: the `conduction_field` source width as a multiple of the
beam radius (`custom.conduction_source_sigma_factor`; 1.0 when unset, the
original channel) and whether the conduction and dose channels are scaled by
`P/P_ref` (`custom.power_weighted_conduction`, `custom.power_weighted_dose`, or
their `PINO_*` environment overrides). A model's stored config can omit a key
that took its default; this record cannot.

## Metrics

An evaluation writes `model_errors.txt` and, per sample,
`val_errors_by_sample.csv`. Temperatures are stored normalised by
`ref_scales.T = 3000 K`.

- Every output key carries `abs_err_mean`, `abs_err_std` and `rmse`, plus
  `substrate_*` and `surface_*` versions when a `z` channel is present. With
  `custom.metrics_exclude_gas=true` (the flow3D default) whole-domain means
  exclude the gas above `z = 0`.
- `T` carries relative errors normalised by the mean truth field; `fl` and
  `alpha` carry IoU instead (A12).
- `fl` is always derived from `T` (tanh centred on the melt threshold) and set
  to zero outside the metal (A13).
- A beam-box block covers the substrate within two beam radii of the beam:
  `beambox_abs_err_mean`, `beambox_rmse`, `beambox_rel_err_mean`, `beambox_iou`
  and `beambox_n` (the number of cells scored). The validation curve calls the
  same metric `beambox_mae_T`; when present it selects the best checkpoint.
- `thermal_headroom_K` and `can_reach_melt` give the margin between the
  predicted substrate peak and the melt threshold; `tail_deficit_max_K` and
  `tail_deficit_per_sqrt2lnN_K` measure the missed peak temperature.
- `surface_msd_um`, `surface_hd95_um` and `surface_hd_max_um` are interface
  distances in micrometres.
- A `[geometry]` block gives melt-pool and keyhole depth, length, width and
  volume (`fl > 0.5`, `alpha < 0.5`) for truth and prediction.
- `served_grid` records the grid that was scored, and `[h_norm]` the
  denominator used.

How to compare these numbers between runs is described in the guide.

## Package map

### 1. Configuration

Hydra configs live in `src/lpno/conf/hydra/`. Runtime options are read from
`custom.*` and the command line; `+key=value` arguments are remapped into
`custom.*`.

### 2. Model Entrypoints + Shared Pipeline

- `NS_Energy_laser_PINO.py`: training entrypoint (standard or k-fold via `custom.split_mode`)
- `NS_Energy_laser_PINO_Kfolds.py`: wrapper defaulting to k-fold mode
- `NS_Energy_laser_PINO_eval.py`: evaluation entrypoint
- `ns_energy_pipeline.py`: facade (`run_training_pipeline`, `_run_standard`, `_run_kfold`)
- `rigno_arch.py`: RIGNO architecture
- `gaot_arch.py`: GAOT architecture
- `pipeline/config_loading.py`: runtime config, `custom.*` handling, device and seed setup
- `pipeline/preprocessing_datasets.py`: loading, preprocessing, splits, datasets
- `pipeline/npz_lazy_store.py`: lazy NPZ loader (`PINO_LAZY_NPZ=1`)
- `pipeline/model_domain.py`: model, nodes, constraints and validators
- `pipeline/postprocessing.py`: inference, metrics and output files
- `pipeline/dataclasses.py`: shared typed containers

### 3. Main options

- `custom.name`, `network_dir` (required for evaluation), `custom.data_dir`,
  `custom.val_data_dir`, `custom.param_dir`.
- `custom.model_family`: `fno`, `rigno`, `gaot`, and the baselines `unet`,
  `fully_connected`, `deeponet`, `rigno_refined`.
- `custom.time_as_batch`: `true` scores one frame per sample; `false` averages
  frames in a moving reference frame (`custom.time_avg_window`,
  `custom.moving_frame_steps`).
- `custom.moving_ref_frame`: beam-following crop on or off.
- `custom.val_max_cases`, `custom.val_frames_per_case`: caps on in-training
  validation (5 cases, 10 frames); scoring runs ignore them.
- `custom.cuda_device`: `0` or `1`; leave it unset on queued runs.
- `custom.wandb_offline=true`: no network logging.

## Environment variables

| variable | effect |
|---|---|
| `PINO_LAZY_NPZ=1` | load consolidated NPZ lazily; aiida-pino stagings always do |
| `PINO_LAZY_STRIDE=sx,sy,sz` | stride the grid at read time; refused on a staging that already has a stride |
| `PINO_LAZY_CROP_Z_TOP=n` | drop the top `n` z-cells (gas) at read time |
| `PINO_LAZY_MOVING_REF=behind,ahead` | serve a beam-following x-window |
| `PINO_LAZY_CACHE=n` | cases held decompressed per store (default 3) |
| `PINO_LOADER_WORKERS=n` | DataLoader workers (2 for FNO and RIGNO, 3 for GAOT work well) |
| `PINO_EVAL_FRAME_STRIDE=n` | score every `n`-th frame at evaluation |
| `PINO_FNO_SUPERRES=0` | turn off the FNO evaluation-time grid adaptation |
| `PINO_DUMP_FRAMES=1` | write per-frame prediction and truth dumps at evaluation |
| `PINO_CONDUCTION_CACHE=dir` | cache of the `conduction_field` channel (default `~/.pino/conduction_cache`) |
| `PINO_MATERIAL_OVERLAY=file.json` | material overlay for the `mat_*` channels |
| `PINO_POWER_WEIGHTED_DOSE=1`, `PINO_POWER_WEIGHTED_CONDUCTION=1` | scale dose or conduction channels by `power_w / 100 W` |
| `PINO_ALLOW_VTK=1` | permit the deprecated VTK directory loader |
| `PINO_ALLOW_CPU=1` | run training or evaluation without CUDA |
| `PINO_SIMULATIONS_DATA_ROOT` | root for the default dataset paths |
| `PINO_RESOURCE_GUARDS` | spooler memory and CPU gates: `warn` (default) or `enforce` |
| `WANDB_ENTITY` | Weights & Biases organisation (unset: account default) |

## GPU spooler

`python -m lpno.gpu_spool` queues jobs per GPU and admits them by declared
memory. The queue lives under `~/.pino/spool` (`PINO_SPOOL_DIR`) and the
reservation ledger under `/var/tmp/pino_gpu_leases` (`PINO_GPU_LEASE_DIR`), so
users on one machine share one budget.

```bash
python -m lpno.gpu_spool add --no-aiida --vram-mb 8000 custom.name=run1
nohup python -m lpno.gpu_spool start >spool.out 2>&1 &
python -m lpno.gpu_spool status
python -m lpno.gpu_spool seat --gpu auto --vram-mb 6000 --ram-mb 16384 -- <command>
```

`add` refuses an unregistered training or evaluation unless it is a smoke test
(`training.max_steps <= 100`) or `--no-aiida` is passed. `--vram-mb` and
`--ram-mb` are bookings; size them from a measured peak. Without `--vram-mb` a
run books the whole card. With `PINO_RESOURCE_GUARDS=enforce`, size the
headrooms (`PINO_SPOOL_VRAM_HEADROOM_MB`, `PINO_SPOOL_RAM_HEADROOM_MB`), the
free-core floor (`PINO_SPOOL_MIN_FREE_CORES`) and the default RAM booking
(`PINO_SPOOL_DEFAULT_RAM_MB`) for the machine. `PINO_SPOOL_ENFORCE_RAM=1` adds a
systemd memory cap at 1.5x the declaration. A priority seat (`--priority`) gives
up after a bounded wait and suits short jobs only. Details: A44 and
`tests/test_gpu_spool.py`.

## Tests and checks

```bash
python -m pytest -q                  # CPU, no FLOW-3D licence, about a minute
python -m pytest -q tests/test_x.py  # one file
```

Install with `pip install -e ".[test]"` or `environment.yml`. Tests that need a
GPU skip when none is visible.

## What the manuscript does not cover

The package contains code that the accompanying manuscript does not use or
discuss. None of it is needed to reproduce the paper, and most of it has unit
tests.

- **Autoregressive training and rollout** (`custom.autoregressive.*`): one-step
  training with recursive rollout evaluation for `fno`, `rigno`, `gaot` and
  `deeponet`, a multi-step unrolled loss, previous-state noise, feedback
  clamping, temporal bundling, and the FNO-only `masked_patch` mode (A31, A32,
  A37). `src/notebooks/autoregressive_rollout_summary.ipynb` summarises
  rollout results.
- **Baselines**: U-Net (`config_flow3D_UNet`), pointwise fully connected
  (`config_flow3D_FullyConnected`, `config_flow3D_FCN_randompath`) and DeepONet
  (`config_flow3D_DEEPONet*`).
- **RIGNO refinement heads** (`custom.model_family=rigno_refined`,
  `config_flow3D_RIGNO_refined_*`): an FNO or U-Net residual corrector on a
  warm-started RIGNO.
- **k-fold training** (`custom.split_mode=kfold`, `NS_Energy_laser_PINO_Kfolds`).
- **2D middle-slice and time-as-dimension modes** (`custom.use_2D`,
  `custom.time_as_dim`).
- **Non-FLOW-3D readers**: the VTK/OpenFOAM datasource
  (`custom_utils/openFoam_datasource.py`, behind `PINO_ALLOW_VTK=1`) and the
  Abaqus H5 datasource (`custom_utils/dataset_loading.py`).
- **The GPU spooler** and the push-check scripts, which are infrastructure.
- **The deprecated physics loss** (below).

## Deprecated physics loss

Earlier versions supervised the FNO with residuals of the energy, momentum and
volume-of-fluid equations (`custom.pde_loss`, `custom.VOF_pde_loss`,
`custom.Diff_loss`). Every shipped config sets them `false`, and every model
in the manuscript is trained on simulation data alone. The residual modules
(`deprecated/equations/`) are not part of this repository, so enabling these
flags fails at import.
