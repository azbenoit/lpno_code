# Reproducing without AiiDA: one local GPU or a Slurm cluster

This is the route for re-running the paper's trainings and evaluations from the
data deposit without AiiDA's machinery: no `verdi archive import`, no profile,
no daemon or message broker. aiida-core is installed only as a library that
opens the deposited archive read-only. A re-run made this way is not recorded
in a provenance graph; for that, use the AiiDA route of the deposit's
`README_AiiDA.md`.

It is checked end to end: a deposited model re-scored this way on the IN718
test split reproduced all 119 metrics of its archived evaluation to within a
relative 1.6e-5, and trainings replay their published settings.

## 1. Environment

```bash
python3.10 tools/setup_env.py --prefix ~/envs/lpno --torch cu126 --with-aiida   # a GPU machine (V100)
python3.10 tools/setup_env.py --prefix ~/envs/lpno-cpu --torch cpu --with-aiida # export only, no GPU
source ~/envs/lpno/bin/activate
```

`tools/setup_env.py` installs torch 2.8.0 from the chosen PyTorch index, every
other package at the pins of `lpno/environment.yml`,
`torch-scatter`, `nvidia-physicsnemo-sym` (without its optional fp16 kernel
unless a CUDA toolkit is loaded), and with `--with-aiida` the aiida-pino package
the export uses. **Volta GPUs (V100) need `cu126`**: the CUDA 12.8 wheels of
torch 2.8 dropped sm_50 to sm_70 and fail at the first kernel with "no kernel
image is available". Use `cu128` for Ampere or newer.

## 2. Data

A training does not read the raw simulation payloads. It reads a collection
*exported* by aiida-pino: for the steady models, each case's last 30 laser-on
frames cut to a window that follows the beam and averaged, plus the
conduction-field input channel, one `.npy` per field and about 10 MB per
simulation. The raw payloads are 20-800 MB each (69 GB for the 490 simulations
of the two-alloy study), so the export runs where the data deposit is and only
its output goes to the GPU machine.

`reproducibility/rerun/export_deposit.py` runs aiida-pino's own export on the deposited
archive, opened read-only, reading every payload in place from its tar and
checking it against the sha256 in `payload_manifest.csv`.

```bash
# the script reads the archive and the payload tars from one folder: link the tars beside the archive
ln -s "$PWD"/data_deposit/restricted_record/payloads_*.tar data_deposit/open_record/
DEPOSIT=data_deposit/open_record; DATA=<data directory>
python reproducibility/rerun/export_deposit.py list --deposit $DEPOSIT twoalloy     # derived collections and sizes
python reproducibility/rerun/export_deposit.py export --deposit $DEPOSIT --out $DATA \
    09e2a9c5 0b657a95 3c7af352 25e13ac1 2aebba14 a9d6017d b28c2dc2 4c1e0e4c 8e185306 95034053
python reproducibility/rerun/export_deposit.py model --deposit $DEPOSIT --out $DATA/models \
    fno_maximin_r05_alloyflag_s42 fno_bandit_r05_alloyflag_s42 fno_joint_r05_alloyflag_s42 fno_steady_edge75
```

Collections are named by UUID (or a unique prefix), models by label or UUID;
`provenance_log.csv` in the deposit lists the UUIDs behind every printed number.

The collections behind the two-alloy results, by UUID prefix:

| prefix | collection | cases | used as |
| --- | --- | --- | --- |
| `b28c2dc2` | `in718_twoalloy_maximin_r05_TRAIN` | 295 | training, maximin half |
| `4c1e0e4c` | `in718_twoalloy_bandit_r05_TRAIN` | 295 | training, bandit half |
| `8e185306` | `in718_twoalloy_COMBINED_r05_TRAIN` | 419 | training, combined selection |
| `95034053` | `ti64_edge75_highenthalpy_train_split` | 171 | training, Ti-6Al-4V-only model |
| `0b657a95` | `ti64_anchorbox_lhs150_val_split` | 9 | validation, every model above |
| `09e2a9c5` | `in718_testset_twoalloy_test_split` | 24 | IN718 test |
| `2aebba14`, `a9d6017d` | `in718_test_extrap`, `in718_test_overlap` | 13, 11 | the two parts of the IN718 test |
| `3c7af352` | `ti64_edge75_highenthalpy_test_split` | 38 | Ti-6Al-4V test |
| `25e13ac1` | `in718_experimental_anchor_conditions` | 7 | IN718 experimental conditions |

Each seed of a model trained on its own collection node, but the three nodes of
a label hold the same members in the same order under the same derivation, so
one export per label serves every seed. The list above covers 490 simulations.
Each is derived once and hard-linked into every collection that contains it,
and an interrupted export resumes where it stopped. Members are derived one at a
time. The steady derivation keeps only the 30 frames before the laser stops, so
aiida-pino's export reads just those frames of each field, where it used to
decompress every frame first (about 8.5 GB for the Velocity of the longest,
798-frame case). The output is bit-identical (`--full-frames` restores the full
read; on 7 cases all 84 arrays and every manifest matched), and the peak falls
to the conduction field, which is computed over all frames. Each case
also writes that field (up to 0.5 GB) to a scratch file deleted right after; on a
laptop, `--cache-dir /dev/shm/lpno` keeps those writes in RAM. To keep a
too-large case from pushing the machine into swap, cap the memory:
`systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 -- python reproducibility/rerun/export_deposit.py ...`.

The archive keeps only the final training collections of the acquisition loop.
An earlier round is the subset of the same members from earlier families
(`batch_keyword`): per half, round 1 is 20 cases plus 5 top-ups, round 2 adds 16
(bandit: 17) plus 9 (bandit: 8) replicates, rounds 3 and 4 add 25 each and round
5 adds 24. For example, the maximin half after round 3, and the combined
selection after round 2:

```bash
python reproducibility/rerun/export_deposit.py subset $DATA/in718_twoalloy_maximin_r05_TRAIN__b28c2dc2 \
    $DATA/in718_twoalloy_maximin_r03_TRAIN 'ti64_*' 'in718_loop_maximin*_r0[1-3]'
python reproducibility/rerun/export_deposit.py subset $DATA/in718_twoalloy_COMBINED_r05_TRAIN__8e185306 \
    $DATA/in718_twoalloy_COMBINED_r02_TRAIN 'ti64_*' 'in718_loop_*_r0[1-2]'
```

(`*` also matches the top-up and replicate families, e.g. `in718_loop_maximintop_r01`.)

## 3. Training and evaluation on one local GPU

`reproducibility/rerun/run_local.sh` runs the same job scripts the cluster uses, with this
checkout as the code and the active environment as the Python:

```bash
RUNS=<runs directory>
# a deposited model scored on the exported test split: checks the data end to end
bash reproducibility/rerun/run_local.sh eval DATA=$DATA RUNS=$RUNS \
    MODEL=fno_maximin_r05_alloyflag_s42 TEST=in718_testset_twoalloy_test_split__09e2a9c5
# a training like a published one, here the maximin half after round 3 with seed 43
bash reproducibility/rerun/run_local.sh train DATA=$DATA RUNS=$RUNS RUN_NAME=fno_maximin_r03_s43 \
    LIKE=fno_maximin_r05_alloyflag_s42 TRAIN=in718_twoalloy_maximin_r03_TRAIN \
    VAL=ti64_anchorbox_lhs150_val_split__0b657a95 SEED=43
bash reproducibility/rerun/run_local.sh eval DATA=$DATA RUNS=$RUNS \
    MODEL=fno_maximin_r03_s43 TEST=in718_testset_twoalloy_test_split__09e2a9c5
```

`LIKE` names a model under `$DATA/models` whose archived `.hydra` the run
replays: its config name and its override list, in order, minus the entries
that name the original machine (`reproducibility/rerun/replay_overrides.py`). `SEED` replaces
its seed; `MAX_STEPS` shortens a smoke test and caps the recording intervals
(checkpoint, validation, stop criterion) to the run's length, since the pipeline
refuses a run that would never checkpoint. An evaluation runs the model's own
`.hydra/config.yaml` with the same overrides, as aiida-pino's eval adapter does,
and prints `fno_eval/model_errors.txt`. Temperature metrics there are normalized
by `ref_scales.T` (3000 K), so the beam-box MAE (BB MAE) in kelvin is `beambox_abs_err_mean`
times 3000. A run lands in `$RUNS/<RUN_NAME>/model` (weights and `.hydra`), an
evaluation in `$RUNS/evals/`.

Costs measured on a V100: a full 32,000-step FNO run takes about 40 min (71 ms
per step, the same for training sets of 196 to 419 cases), using 2.1 GB of GPU
memory on the 295-case set; an evaluation of the 24-case test split takes under
a minute.

## 4. Training and evaluation on a Slurm cluster

### Site configuration stays private

Every site-specific value (SSH alias, user, account, partition, QOS, cluster
paths) lives in one file outside the repository:

```bash
mkdir -p ~/.config/lpno && chmod 700 ~/.config/lpno
cp reproducibility/rerun/site.env.example ~/.config/lpno/site.env && chmod 600 ~/.config/lpno/site.env
$EDITOR ~/.config/lpno/site.env        # replace every <placeholder>
bash tools/install_hooks.sh            # once per clone: the leak check runs before each commit
```

`reproducibility/rerun/lib.sh` reads it (or `$LPNO_SITE_ENV`, or a git-ignored
`reproducibility/rerun/site.env`) and refuses to run while a value is missing or still a
placeholder. The job scripts carry only generic resource requests; account,
partition, QOS, GPU request and paths are added by `reproducibility/rerun/submit.sh` at
submission time, so no committed file names a site.

`tools/check_site_leaks.py` enforces it: every value of the site file, and
generic site patterns (real `/scratch/` and `/home/<name>` paths, IP addresses,
licence `PORT@HOST` endpoints, keys, host fingerprints), are refused in any
staged line, and the site file itself under any `*site.env` name. Run it over the
whole tree with `python3 tools/check_site_leaks.py --all`.

### One-time setup and data

```bash
bash reproducibility/rerun/submit.sh deploy        # committed HEAD of lpno/, reproducibility/rerun/, tools/ (~12 MB)
bash reproducibility/rerun/submit.sh build-env     # tools/setup_env.py under the build QOS; 5-10 min, rerunning resumes
bash reproducibility/rerun/submit.sh log <jobid>   # the last lines show the torch build and the entry points importing
bash reproducibility/rerun/submit.sh check-gpu     # a short job that runs torch and torch_scatter on a card
bash reproducibility/rerun/submit.sh upload $DATA  # rsync of the exported collections and models; resumes if interrupted
```

`build-env` uses `cu126` by default; set `TORCH_CUDA=cu128` in the submission
for Ampere or newer cards. Keep the local export: if the cluster data directory
is on a scratch filesystem that purges old files, rerunning `upload` restores
what went missing and sends nothing else.

### Jobs

```bash
bash reproducibility/rerun/submit.sh eval MODEL=fno_maximin_r05_alloyflag_s42 TEST=in718_testset_twoalloy_test_split__09e2a9c5
bash reproducibility/rerun/submit.sh train RUN_NAME=fno_maximin_r03_s43 LIKE=fno_maximin_r05_alloyflag_s42 \
    TRAIN=in718_twoalloy_maximin_r03_TRAIN VAL=ti64_anchorbox_lhs150_val_split__0b657a95 SEED=43
bash reproducibility/rerun/submit.sh eval MODEL=fno_maximin_r03_s43 TEST=in718_testset_twoalloy_test_split__09e2a9c5 AFTER=<train jobid>
bash reproducibility/rerun/submit.sh status
bash reproducibility/rerun/submit.sh log <jobid>
```

The keys are those of `run_local.sh`. `AFTER=<jobid>` holds a job until another
has finished successfully (Slurm `afterok`), so an evaluation can be queued with
its training. `DEBUG=1` sends a job to the debug QOS named by `SLURM_DEBUG_QOS`, which is
for short tests of the setup only, such as a short evaluation or a
training with `MAX_STEPS`; results belong on the normal QOS. `submit.sh` refuses a debug training without
`MAX_STEPS` and a second debug job while one is queued or running.

A run lands in `<runs>/<RUN_NAME>/model` and an evaluation in `<runs>/evals/`;
fetch them with rsync. To register a run in an AiiDA profile that holds the
imported archive, use
`aiida-pino submit import-model --artifacts-path <run>/model --training-source <collection pk or uuid>`.

## 5. What differs from an AiiDA-managed run

The GPU is the one the scheduler (or `CUDA_VISIBLE_DEVICES`) assigns:
`custom.cuda_device` is never passed, since it would override
`CUDA_VISIBLE_DEVICES`. Runs are single-process (physicsnemo would otherwise
read a Slurm batch job as an `srun` launch), there is no `gpu_spool` seat lock,
data loading runs in the main process (`PINO_LOADER_WORKERS=0`), the open-file
limit is raised for the memory-mapped loader, and wandb stays offline.
