# aiida-pino

`aiida-pino` runs and records a machine-learning pipeline that learns to predict
laser-welding physics. Each step of the pipeline runs through AiiDA (a workflow
engine that records what ran, with which inputs, and what came out), and a
browser console and an `aiida-pino` command-line tool let you drive it without
writing Python.

If the terms *AiiDA*, *PINO* or *FLOW-3D* are new to you, read
[Key concepts](#key-concepts) first; the rest of this README assumes those
short explanations.

## What it does

Training a neural network to imitate a physics simulation needs three things:
simulation data to learn from, a place to keep that data organized, and a way to
train and check models on it. `aiida-pino` connects the three and records how
every result was produced.

The pipeline has four stages:

1. **Generate data**: simulate a laser melting or welding a metal substrate
   with the commercial computational-fluid-dynamics (CFD) solver FLOW-3D
   (through the sibling repo `flow3D_scripts`). Each run produces gridded
   fields over time: temperature, metal/void fraction, velocity, pressure.
2. **Ingest and store**: convert those solver outputs into AiiDA "database"
   nodes, one `PinoSimulationData` node per simulation, grouped into a
   `PinoDatasetCollectionData` that serves as a training dataset.
3. **Train**: train a PINO model (physics-informed neural operator, a neural
   network that learns a fast, solver-like mapping between field states) and
   register it as a `PinoModelData` node.
4. **Evaluate**: score the model on held-out data and store the metrics as a
   `PinoModelEvaluationData` node.

Everything a run touches is stored with its *provenance*, the recorded history
linking a trained model back to the exact simulations and settings that produced
it.

## Status

Every simulation, training run and evaluation behind the paper was run and
recorded through this plugin. The provenance graph that holds them is published
with the paper, and `../data_deposit/record_docs/README_AiiDA.md` explains how
to import it; the FLOW-3D simulation arrays themselves are only in the
restricted-access record.

The plugin was built toward an automatic loop: generate FLOW-3D simulations,
file them into datasets, train, mark data as suitable for training, and retrain
a shared model on new data. The building blocks of the full loop exist and were
exercised in the IN718 acquisition experiment
(`../flow3D_scripts/batches/in718_loop/`). Its driver, `bin/round.py`, runs each
round as separate stages through the existing workchains: `acquire` picks the
round's cases (by maximin coverage, or by a regime bandit that allocates cases
from per-region model error), `solve` submits them as one
`aiida-pino submit flow3d-batch`, `collect` labels the new cases `split=train`,
`train` composes the round's training collection and submits the models with
`aiida-pino submit train-eval`, and `probe` scores them per regime with
`aiida-pino submit eval`, which sets the bandit's next allocation.
`bin/tick.py` reads the loop's state and runs the next stage, either unattended
(`--apply --every <seconds>`) or with a person approving each solved round
before it is trained on (`--apply --gate`). Both are described in
[`../flow3D_scripts/batches/README.md`](../flow3D_scripts/batches/README.md#running-the-acquisition-loop),
with the additions that would make the loop more robust.

Whether a simulation is suitable for training is its `split` label: `train`,
or `excluded` with the reason in `exclusion_reason` (the loop's `collect` sets
it from the round's boundary check), and training collections are composed of
`train` cases only. The loop is written for that experiment: its datasets, its
two halves and its five rounds. A general version, in which suitable data filed
by any batch retrains a shared model, is the next step. The remaining steps are
listed in the roadmap of
[`docs/developer_notes.md`](docs/developer_notes.md#roadmap).

## What the manuscript does not cover

Present in the code and not described in the manuscript:

- **The legacy VTK ingest path** (`flow3d_to_vtk` then `vtk_to_dataset`, which
  needs a ParaView `pvpython` Code), kept for raw FLOW-3D output without an NPZ
  export.
- **k-fold training and evaluation** (`KfoldTrainEvalWorkChain`, `pino.kfold`).
  It is withdrawn from the console and the CLI; the workchain still runs when
  submitted directly (`docs/developer_notes.md`).
- **The web console** (beta, see [Web console (beta)](#web-console-beta)).
- **Melt-pool statistics and regime labels** (`case-stats`, `dataset-stats`,
  `console/viz.classify_regime`). `classify_regime` labels a case keyhole when its
  time-averaged cavity aspect ratio exceeds one and its cavity reaches 2.1 beam
  radii, lack of fusion when the cavity is under two cells deep, and shallow cavity
  otherwise. The paper's regime labels come from it (`assumptions.md` A18).
- **Storage and staging maintenance**: metadata-only eviction (`evict`),
  `reclaim`, `compress`, and the runtime-cache tools (`warm`, `stage-remote`,
  `unstage`, `derive-fields`).
- **Dataset unions** (`PinoDatasetUnionData`) and the single-tensor
  `PinoDatasetData`, kept for older nodes.
- **Nodes from the removed OpenFOAM importer** (`source_type` `openfoam` or
  `openfoam_raw`), which stay readable. The importer and its steady sweep
  generator are no longer in the code.


## Key concepts

A few AiiDA and CFD terms recur in this README.

**The workflow engine**

- **AiiDA**: an open-source Python engine for computational science. It runs
  your calculations, stores every input and output in a database, and records how
  each result was produced. This repo is an AiiDA *plugin* that adds the
  simulate-to-model steps.
- **provenance**: the automatically recorded history of a result (which inputs,
  code and settings produced it, and what it fed into), stored as a graph.
- **WorkChain**: a multi-step AiiDA workflow (e.g. "convert data, then train,
  then evaluate"). These are the `pino.*` workflow entry points.
- **CalcJob**: a single external calculation, i.e. one program run in a sandboxed
  folder with inputs staged in and outputs parsed back (e.g. one training run).
- **Data node**: a single stored object in the AiiDA database (a number, a
  dict, a folder, or a custom array bundle). This repo's `pino.*` node types are
  the "rows" of the pipeline's database (see [Data model](#data-model)).
- **Code / Computer**: AiiDA's abstractions for *where and with what to run*. A
  Computer is a machine (here, `localhost`, or a remote GPU box over SSH); a
  Code is a specific executable on it (a Python interpreter with
  `lpno` installed). You register these once before running
  anything.
- **verdi**: AiiDA's command-line tool (`verdi status`, `verdi code list`,
  `verdi daemon restart`).
- **daemon / RabbitMQ**: AiiDA's background worker (daemon) runs submitted
  workflows and needs a message broker (RabbitMQ). Browsing works without
  either; submitting a workflow does not (see [Install](#install)).

**The science and the model**

- **PINO (Physics-Informed Neural Operator)**: the model being trained. A
  *neural operator* learns a mapping between whole fields (e.g. from the
  temperature field now to the field a step later), so once trained it
  approximates a slow CFD solver far faster. *Physics-informed* means the
  training loss can include physics (PDE) terms besides the data misfit.
- **FLOW-3D**: the commercial (license-gated) CFD package that runs the laser
  welding simulations, and so the pipeline's data source. Its runs are ingested
  directly from the compact NPZ export written by `flow3D_scripts`.
- **conduction mode / keyhole mode**: the two regimes a laser weld runs in, and
  the words the datasets are labelled with. In **conduction** mode the beam melts
  the metal and the heat spreads by conduction: the pool is shallow, roughly
  hemispherical, and the surface stays nearly flat. Raise the power density and
  the metal under the beam starts to vaporise; the recoil pressure of that vapour
  pushes open a narrow, deep cavity into the melt, the **keyhole**, which the
  beam then shines down, so it penetrates several times deeper. Keyhole welds are
  deeper and faster but prone to porosity when the cavity collapses. Which regime
  a case lands in follows from power, scan speed and beam radius; datasets here
  are tagged `keyhole` or `conduction`, and some are selected by a normalised
  enthalpy band (`H`) that separates the two.
- **lpno**: the sibling Python package (installed separately) that holds the
  simulation-conversion and PINO training and evaluation code. `aiida-pino`
  only orchestrates and records; it calls into this package.
- **source_type**: a provenance tag on each stored node recording where it came
  from: `flow3d` (the supported path), or `openfoam` / `openfoam_raw` on nodes
  imported by the removed OpenFOAM path. The ML stage is source-agnostic, so this
  tag is how origin is remembered.

## How the pipeline fits together

Every data source ends at the same dataset node, which the ML stage consumes
without needing to know its origin:

```text
FLOW-3D run dir ── raw_flow3d_to_dataset (NPZ) ──→ PinoDatasetCollectionData
                                                          │
                                                    train / eval
                                                          │
                                    PinoModelData ──▶ PinoModelEvaluationData
```

The FLOW-3D path is one step (`raw_flow3d_to_dataset`): it reads the compact
`npz/<case>.npz` export and `case_manifest.json` that every `flow3D_scripts` run
directory carries, and needs no ParaView and no `lpno` in the Code
environment. See
[Importing FLOW-3D runs](#importing-flow-3d-runs-recommended-path). The stored
node types and how they link are documented in [`database.md`](database.md).

## Companion repositories

`aiida-pino` is one part of a pipeline; the others are referenced as sibling
directories beside this one (`../<name>`), never by absolute path. In the
public repository (https://github.com/azbenoit/lpno_code) all of them
are directories of one checkout, beside `tools/setup_env.py` (one pinned
environment for both packages), `data_deposit/` (download scripts, and the
record documentation in `data_deposit/record_docs/`) and `reproducibility/`
(notebooks that rebuild the paper's tables and figures, and
`train_evaluate.ipynb`, a tutorial on training and evaluating with and without
AiiDA).

| Directory | Role |
|---|---|
| `lpno` | the model, training and evaluation code every train/eval CalcJob calls into; installed into the Code's Python environment |
| `flow3D_scripts` | the FLOW-3D run generator and NPZ exporter; its `python/run_rectangular_case.py` is what `pino.flow3d_run` launches, and its `runs/` tree is what the NPZ importer reads |

Neither is vendored: `lpno` is a normal `pip install -e` into the
same environment, and the `flow3D_scripts` root is resolved at run time (see
"Runner resolution").

## Requirements

- **Python** and the sibling `lpno` package installed in the same
  environment (it provides the actual conversion/training code).
- **AiiDA 2.x** with a configured profile. A message broker (RabbitMQ) and the
  AiiDA daemon are needed to submit workflows; browsing needs neither.
- **A GPU** (CUDA) for training of any real size; training exposes a
  `cuda_device` setting. Weights & Biases logging is optional (`wandb_offline`).

Paths below are placeholders; substitute your own. Do not commit machine-specific
absolute paths or any license/endpoint values.

## Install

This package installs on stock Ubuntu with the system Python (3.10+); no
compiler and no conda are needed. The web console needs the `web` extra:
without it the server has no FastAPI to start and no matplotlib, so every plot
endpoint answers 500:

```bash
cd /path/to/aiida-pino && python3 -m pip install -e '.[web,dev]'
```

That is enough for the console, the CLI and the tests (the CLI alone needs no
extra). `verdi presto` then gives you a working SQLite profile with no
PostgreSQL to install. Without RabbitMQ that profile has no broker and so no
daemon: browsing works, and submissions run in the terminal with
`aiida-pino --foreground ...` until they finish. Background submission needs
RabbitMQ 3.8.14 (see [`web/README.md`](web/README.md)) before `verdi presto` (or `verdi profile configure-rabbitmq` after).

For a pinned environment with both packages, torch, `torch-scatter` and
`nvidia-physicsnemo-sym` in one step, use `../tools/setup_env.py`
(`--with-aiida` adds this package); the instructions in
`../data_deposit/record_docs/` use it.

The sibling package is needed for training and evaluation, and for exporting
derived datasets (the beam-following window and the conduction field come from
it); install it into the same environment when you get there:

```bash
cd /path/to/lpno && python3 -m pip install -e .
python scripts/install_physicsnemo_sym.py   # lpno.models imports nvidia-physicsnemo-sym
```

`nvidia-physicsnemo-sym` is source-only and its `setup.py` compiles a CUDA
kernel, so a plain `pip install` (or the `[train]` extra) fails on any machine
without a CUDA toolkit; the script installs it with the kernel only when
`CUDA_HOME` is set (the kernel serves fp16 training, which no published config
uses).

In practice, if you will train or evaluate here, use one environment for both
packages so they import in one interpreter: `../tools/setup_env.py --with-aiida`,
or the `pino` conda env of `../lpno/environment.yml` with this
package installed into it (`pip install -e '.[web,dev]'` from that env). On Ubuntu 23.04 and later the system Python refuses `pip install`
outright (PEP 668, "externally-managed-environment"); use that env or a venv.
To open the published provenance archive, `aiida-core` 2.8.x is the version that
wrote it; this package's pin admits it, and
`../data_deposit/record_docs/README_AiiDA.md` gives the import steps.

## One-time AiiDA setup

You need a profile, a local Computer, and a Code registered. If you do not
already have a profile, create one with `verdi presto` (or `verdi profile
setup`), then:

```bash
# Register localhost as a Computer
verdi computer setup -n -L local -H localhost -T core.local -S core.direct -w /tmp/aiida-<user>
verdi computer configure core.local -n local
```

### Codes to register

Register the executable the pipeline calls:

```bash
# Required: the Python interpreter that has lpno installed.
# Used for dataset conversion, training and eval.
verdi code create core.code.installed -n -L pino-python \
  -Y local -X /path/to/pino-env/bin/python -D 'PINO Python environment'
```


Verify the setup:

```bash
verdi status                 # profile / daemon / broker health
verdi code list              # your registered Codes
verdi plugin list aiida.calculations | grep pino   # plugin entry points are visible
```

## Web console (beta)

The browser console under `web/` is useful for monitoring jobs, cleaning up and
annotating datasets, and following provenance from a model back to the
simulations it was trained on. It is beta software with rough edges; the CLI
below drives the same service layer. It runs on the machine that holds the
AiiDA profile and can be opened from other machines on the local network.
`web/README.md` has the execution model, the broker and daemon setup, and the
endpoint list.

```bash
# loopback only
web/dev/run_local.sh start          # http://127.0.0.1:8180/index.html

# reachable from other machines on the LAN
web/dev/run_local.sh start lan

web/dev/run_local.sh status
web/dev/run_local.sh stop
```

`run_local.sh` runs the backend with the interpreter named by
`AIIDA_PINO_WEB_PYTHON` (default: a `pino` conda env under `$HOME/miniconda3` if
one exists, else `python3` on `PATH`) and exits with a message unless it imports
`aiida`, `fastapi` and `uvicorn` (the `web` extra). Pick a named
profile with `AIIDA_PINO_WEB_AIIDA_PROFILE`.

It serves tabs for browsing and monitoring: Monitor, Environment, Datasets,
Compose new dataset, Models, Runs, and Logs. Browsing and monitoring need only a
loaded AiiDA profile.


If dropdowns for Codes or datasets look empty, see
[Troubleshooting](#troubleshooting).

## Command line: the `aiida-pino` tool

The `aiida-pino` console script (entry point `aiida_pino.cli:main`, backed by
the same service layer in `src/aiida_pino/console/` that the web console uses)
is the terminal counterpart of the web console: browse, annotate, and submit
without a notebook or a Python session.

Three global options go before the subcommand: `--profile NAME` picks the AiiDA
profile (default: the configured default profile), `--json` prints raw JSON
instead of tables, and `--foreground` runs a submission (`submit ...`,
`resubmit`) in the terminal until it finishes, which needs no daemon and no
RabbitMQ. Nodes can be given by pk or by UUID wherever a command takes a model,
dataset or training run; pks change on `verdi archive import`, UUIDs do not.

### Subcommands

Generated from `aiida-pino --help`; every subcommand has its own `--help`.

| Subcommand | What it does |
|---|---|
| `env` | environment status (version, profile, daemon, codes, configs) |
| `health` | service health and daemon reachability |
| `codes` | installed AiiDA codes by role |
| `configs` | available training config names |
| `training-defaults [--config NAME]` | default training profile values |
| `datasets [--search S] [--role R] [--keyword K] [--limit N] [--superseded] [--all]` | list dataset nodes |
| `dataset PK` | dataset detail |
| `models [--trained-on PK] [--search S] [--evaluated-only] [--keyword K] [--limit N] [--all]` | list model nodes; `--trained-on` follows splits and layout variants |
| `model PK` | model detail |
| `compare-models PK_A PK_B` | every difference between two models, classified confound / caution / benign |
| `sims [--source S] [--folder F] [--split S] [--keyword K] [--limit N]` | list simulation nodes |
| `case-stats PK [--recompute]` | melt-pool statistics for one simulation (cached in extras) |
| `dataset-stats PK [--compute]` | aggregate melt-pool statistics for a dataset |
| `plot PK [--field F] [--view long\|surface\|cross] [--frame N] [--no-contour] [-o FILE]` | render one case slice frame to a PNG file |
| `ps [--limit N] [--recent] [--label S] [--all-processes] [--smokes] [--state all\|queued\|finished\|failed]` | list processes (pino-only by default) |
| `show PK` | process detail with outputs and metrics |
| `logs PK` | process report and captured console output |
| `tail PK [--lines N] [--path P]` | one-shot tail of a running CalcJob's live log |
| `tag PK [--add K ...] [--remove K ...] [--set K ...]` | edit node keywords (no flags: show current) |
| `split PK VALUE [--reason TEXT]` | assign a simulation's split; `--reason` is recorded in the node's `split_audit` |
| `presubmit --train-dataset PK [...]` | dry-run a training submission: resolve the derived collection, check remote staging, report every refusal; submit nothing |
| `staged --computer C [--verify] [--check DATASET] [--derivation JSON\|@FILE] [--fields F ...]` | list the runtime caches on a computer, or `--check` whether the collection a submission would consume is staged |
| `unstage DATASET --computer C [--derivation ...] [--fields ...] [--yes]` | delete a staged runtime cache from a computer and retract its record (dry run unless `--yes`) |
| `compare PK [PK ...]` | check whether evaluations measure the same thing (dataset, split, derivation, grid, strides) |
| `stage-remote PK_OR_UUID --computer C [--fields F ...] [--remote-root DIR] [--dry-run] [--for-training]` | copy a dataset's runtime cache to a remote computer (needed before any job there) |
| `warm PK [--field NAME ...]` | build a collection's local runtime cache now, before a job needs it |
| `describe PK TEXT [--title T]` | set a node's description (and optionally its short title) |
| `folder PATH PK ...` / `folders` / `folder-rename OLD NEW` / `folder-remove NAME` | file datasets under a browser folder; list, rename (merges), or remove (unfiles, deletes nothing) |
| `profile-keys [training\|evaluation\|derivation\|flow3d_run] [--config NAME]` | list every accepted key of a profile, with its default |
| `resubmit PK [--override K=V ...] [--run-name N] [--mode train\|train_eval] [--title T] [--description D]` | re-run a past training with its own inputs, optionally overridden |
| `file-orphans PK [--yes] [--name N] [--cases a,b] [--description D]` | file a killed batch's completed cases into a base dataset (dry run unless `--yes`) |
| `evict PK [--yes] [--reason TEXT]` | delete a dataset's array data, keeping metadata + provenance (dry run unless `--yes`) |
| `restore-payloads --deposit DIR [--collection ID ...] [--for ID ...] [--all] [--yes]` | put the arrays of an imported data deposit back into the deposited simulation nodes, same UUIDs, sha256-checked against `payload_manifest.csv`; `--for` takes an evaluation (what it scored) or a training workflow or model (what it trained and validated on) (dry run unless `--yes`) |
| `derive-fields PK [--stride sx,sy,sz] [--limit N] [--seed-only]` | precompute expensive conditioning fields (`conduction_field`) into the loader cache (a speed-up only, never what makes a channel available) |
| `reclaim [--yes]` | reclaim disk from evicted/unreferenced objects, repacking only affected packs (dry run unless `--yes`) |
| `compress PK [--yes]` | convert legacy arrays-mode members to compressed `npz_file`, verified lossless (dry run unless `--yes`) |
| `preview collection PK ...` | validate simulations as collection members (broker-free) |
| `collection create NAME [PK ...] [--dataset-pk PK ...] [--intersect-fields] --description D \| --no-description [--keyword K]` | compose a collection from simulation pks and/or whole datasets (role: composed) |
| `collection eval-only PK` | reserve a dataset for evaluation: every member becomes `split=test`, and training refuses it |
| `collection assign-splits PK [--train F] [--val F] [--test F] [--reshuffle] [--seed N]` | randomly assign train/val/test splits to a collection's members |
| `flow3d-queue` | FLOW-3D spool queue and seat-lock status |
| `paused` | everything deliberately parked, with how long it has been parked |
| `timing [--solo]` | how long each operation takes: median and IQR, normalised by work done |
| `delete PK ... [--force]` | delete nodes (dry run unless `--force`) |
| `kill PK` | kill a running process (needs the daemon) |
| `submit KIND ...` | submit a workflow (needs the daemon); kinds below |


Browse and inspect:

```bash
aiida-pino env                      # profile / daemon / registered Codes
aiida-pino datasets                 # list dataset collections
aiida-pino dataset PK               # one collection in detail
aiida-pino models
aiida-pino model PK
aiida-pino compare-models PK_A PK_B # every difference between two
                                    # models, classified confound / caution / benign, with
                                    # member-level training-data overlap and shared
                                    # evaluations. "A beat B" is only evidence about the
                                    # changed thing when the verdict is SINGLE-FACTOR.
aiida-pino sims --split train --keyword "random path"   # filter simulations
aiida-pino ps                       # running / recent processes
aiida-pino ps --state failed        # excepted, killed, or a non-zero exit — filtered in the query
aiida-pino show PK                  # any node
aiida-pino logs PK
aiida-pino tail PK
aiida-pino --json datasets --all    # raw JSON, for scripts
```

Page caps: `--limit N` caps a listing and `--limit 0` means no cap, on every
listing (`datasets`, `models`, `sims`, `ps`). `--all` is a
convenience for `--limit 0` on `datasets` and `models`; on `ps` it means
"include non-pino processes" and does *not* lift the cap, so it is also spelled
`--all-processes`.

Annotate (see [Annotations](#annotations-split-keywords-description)):

```bash
aiida-pino tag PK --add keyhole --add "random path"   # or --remove / --set
aiida-pino collection eval-only PK                    # reserve for evaluation: all members -> split=test
aiida-pino split PK train           # train / val / test / eval / excluded / duplicate_excluded / unassigned
aiida-pino split PK excluded --reason "centre-point replicate of <PK>"   # the reason lands in split_audit
aiida-pino describe PK "coarse-mesh smoke batch; not for training"
aiida-pino describe PK "600-case LHS sweep" --title "material LHS"  # title = short name shown bold in the console Runs list
aiida-pino tag PK --add core   # protected reference set: deletion is refused until the keyword is removed
aiida-pino folder "Core reference" <PK> <PK> <PK>   # file datasets into a browser folder ('' unfiles them)
aiida-pino folders                 # list folders
aiida-pino folder-rename "refernce" "reference"   # merges when the target exists
aiida-pino folder-remove "scratch"                # unfiles its members; nothing is deleted
aiida-pino evict PK                               # dry run: how much data eviction would free
aiida-pino evict PK --yes --reason "free disk"     # delete the arrays, keep metadata + provenance
aiida-pino collection create my_set 101 102 103 \
  --description "keyhole subset" --keyword keyhole
```

Manage and submit:

```bash
aiida-pino delete PK                # dry-run preview; add --force to delete
aiida-pino kill PK
aiida-pino preview collection PK PK # broker-free: validate simulations as collection members
aiida-pino profile-keys training    # every accepted key of a profile, with its default
aiida-pino presubmit --train-dataset <PK> --python-code pino-python@local \
  --training-profile @profile.json  # dry run: every refusal a submit would hit, submits nothing
aiida-pino submit train-eval --train-dataset <PK> --python-code pino-python@local \
  --training-profile @profile.json
```

### `submit` kinds and their flags

Every `submit` kind takes `--description TEXT` / `--submitted-by TEXT` where
listed (both recorded on the node) and profile inputs as `@file.json` or
`key=value` (values parsed as JSON, with a string fallback). Each flag takes one
value, so repeat the flag for several keys (`--training-profile @base.json
--training-profile max_steps=79822`). `presubmit` takes the same repeated
flags, so a submit command can be checked as it stands, and also accepts
several values after one flag. Profiles are validated
by the same `aiida_pino.utils.validate_*` functions the workflows use;
`aiida-pino profile-keys <kind>` lists the accepted keys. Submissions need the
broker and daemon up (`web/README.md`).

- `submit train-eval --train-dataset PK --python-code CODE [--val-dataset PK]
  [--mode train|train_eval] [--init-model PK] [--training-profile ...]
  [--derivation-profile ...] [--evaluation-profile ...] [--description D]
  [--submitted-by S]`: train (and optionally evaluate) a model.
- `submit eval --dataset PK --model PK --python-code CODE [--full-frames]
  [--allow-domain-change] [--allow-training-data] [--evaluation-profile ...]
  [--derivation-profile ...] [--description D] [--submitted-by S]`: evaluate an
  existing model on a dataset. **`--full-frames` matters for any number you
  publish**: without it an automatic eval subsamples to ~25 evenly spread frames
  per case, which is good for a health readout only (the stride is stamped into
  the metrics either way). `--allow-domain-change` is the
  documented remedy for the exit-306 refusal; `--allow-training-data` scores
  cases the model trained on and stamps the record `held_out=False`. The eval
  scales H_norm by the model's training maximum: for an older model that carries
  no `h_norm_denom.json`, the calculation rebuilds it from
  the model's training-source simulations and stages it with the weights; the
  metrics' `[h_norm]` block records the value used (`scored_split: 1.0` means it
  had to fall back to the scored split).
- `submit raw-flow3d --source-path DIR --python-code CODE [--dataset-profile
  ...] [--materials-root DIR]`: import raw FLOW-3D NPZ runs as simulations
  (the CLI form of the workflow below).
- `submit flow3d --python-code CODE [--run-config ...] [--execution-profile
  ...] [--dataset-profile ...] [--materials-root DIR] [--target-dataset PK]`:
  run one FLOW-3D case through `flow3D_scripts`, ingest its NPZ and file it.
- `submit flow3d-batch --python-code CODE --config @FILE [--config @FILE ...]
  [--execution-profile ...] [--dataset-profile ...] [--materials-root DIR]
  [--max-concurrent N] [--target-dataset PK]`: a FLOW-3D config sweep, one
  solve at a time.
- `submit import-model --python-code CODE --training-source PK [--artifacts-pk
  PK | --artifacts-path DIR] [--validation-source PK] [--training-profile ...]
  [--derivation-profile ...] [--model-name NAME] [--dry-run]`: register
  externally trained artifacts as a model. `--training-source` takes the
  dataset the model was trained on; a training-job pk does not belong there.
- `submit derive --dataset PK [--derivation-profile ...]`: derive a dataset
  from a collection.

### Warm start: continue training from a checkpoint

`--init-model PK` seeds the trainer with an existing model's checkpoints and
resumes from them, so a run that stopped short can be extended without starting
over:

```bash
aiida-pino submit train-eval --mode train \
  --train-dataset <PK> --val-dataset <PK> --python-code pino-python@local \
  --init-model <MODEL_PK> \
  --training-profile @profile.json --training-profile max_steps=79822
```

The adapter picks one of two paths per run and prints which, on a `[warm-start]`
line:

- **Resume.** When the source `optim_checkpoint.0.pth` will load cleanly, its
  files are copied into `network_dir` and physicsnemo restores the optimizer,
  the LR schedule and the step counter. `max_steps` is then a total across
  runs: continuing a step-6000 checkpoint with `max_steps=32000` trains 26000
  more steps.
- **Weights only.** Otherwise, physicsnemo's `initialization_network_dir` loads
  the weights and a fresh optimizer starts at step 0, so the LR schedule
  restarts at warmup and `max_steps` counts *this* run's steps.

Resuming is refused, with the reason printed, when `amp.enabled` is true and
the checkpoint's `scaler_state_dict`/`deriv_scalers_state_dict` are empty (a
checkpoint written with AMP off). An enabled `GradScaler` raises on an empty
state dict, and `Trainer._load_optimizer` swallows that in a bare `except:` and
silently falls back to `step = 0` with a half-loaded optimizer; the weights-only
path reaches the same outcome explicitly. `train_summary.json` records
`resumed_optimizer` so the choice can be checked after the fact.

Scale `scheduler_decay_steps` with `max_steps`, or the extra steps are wasted:
the schedule is `decay_rate ** (step / decay_steps)`, so a value tuned for a
shorter run drives the learning rate to near zero early. Keeping the
0.98/250-at-32000 trajectory over 79822 steps means `decay_steps=624`.

**Short runs.** The train adapter lowers every interval longer than `max_steps`
to `max_steps`: `rec_results_freq`, `rec_validation_freq`, `stop_criterion.freq`
and physicsnemo's weights interval `save_network_freq` (default 1000, set by no
shipped config). It reads the value the run will use, the profile's
`hydra_overrides` included, and leaves a finer interval set there alone. A
200-step run therefore writes the weights of step 200 as well as those of step 0.

Sizing note: set `gpu_vram_mb` and `host_ram_mb` from a measured peak. The two
behave differently.

**`gpu_vram_mb` is measured automatically.** Leave it unset
and `TrainModelWorkChain` sizes the reservation itself: every train run
records its measured GPU peak (extras `vram_key` / `gpu_vram_peak_mb` on the
calcjob), a new submission with a known footprint reserves
`peak + max(1 GiB, 10%)`, and an unknown footprint first runs a 100-step
exclusive smoke run (`smoke` keyword; hidden from the Runs list by default,
always visible in the Monitor's GPU panel) to measure one. A failed smoke run
shows within minutes that the configuration does not fit the card. Set
`gpu_vram_mb` to force a manual booking, or `gpu_exclusive: true` for a
whole-card run; both bypass the automation. Without nvidia-smi the automation
falls back to the old exclusive booking.

`gpu_vram_mb` is a ledger booking only. Over-declaring wastes packing capacity:
a 13000 MiB eval that actually used 3470 MiB blocked a queued training for hours.
An undeclared value books the whole card exclusively (once the smoke run above
has had its chance).

**`host_ram_mb` is a packing declaration.** `gpu_spool` places a run beside
co-tenants by the figure it declares, so an under-declared run is admitted onto
a machine that cannot hold it, and an over-declared one starves the queue. The
declaration is not enforced by default. Enforcement, in which
`gpu_spool.memory_scope_prefix` wraps a declaring run in a systemd scope with
`MemoryMax` at 1.5× the declaration, is opt-in through
`PINO_SPOOL_ENFORCE_RAM=1`. With it set, under-declaring is fatal and silent:
`host_ram_mb=16384` (a 24 GB ceiling) got a RIGNO training SIGKILLed seconds
after step 0 with no traceback, while the same job had already run 15 h
uncapped. A cgroup counts every process in the scope, so 8 dataloader workers
each transiently decompressing a case cost far more than the main process on
its own.

Measured for RIGNO at 75×50×40, batch 1: eval ~3.5 GB VRAM; training ~14.7 GB VRAM and 8.4 GiB
peak RSS for the main process, needing `host_ram_mb` ≈ 48 GB once 8 loader
workers are counted.

## Importing FLOW-3D runs (recommended path)

`RawFlow3DImportWorkChain` (`pino.raw_flow3d_import`) imports a tree of
`flow3D_scripts` run directories (`runs/<case>/` with `case_manifest.json` and
`npz/<case>.npz`) into one `PinoDatasetCollectionData` in a single step. The
adapter needs `aiida-pino` and `aiida-core` importable in the Code's
environment, plus numpy; it needs neither `pvpython` nor `lpno`:

```python
from aiida import orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

submit(
    WorkflowFactory("pino.raw_flow3d_import"),
    source_path=orm.Str("/path/to/flow3D_scripts/runs"),
    dataset_profile=orm.Dict(dict={"time_target_frames": 100}),
    python_code=orm.load_code("pino-python@local"),
    # optional: resolve manifest material files against the flow3D_scripts repo
    materials_root=orm.Str("/path/to/flow3D_scripts"),
)
```

The dataset profile keys are all optional:

- `npz_fields`: list of NPZ field names to import (e.g. `["Temperature",
  "Melt Region", "Velocity"]`); default is every physical field in the NPZ.
- `time_target_frames`: integer ≥ 2; when a run stores more frames, they are
  evenly subsampled to this count, always keeping the first and last frame.
  Default keeps all frames.
- `description`: free-text note stored on every imported simulation (where the
  run came from, how it was made) and mirrored onto the created dataset's
  `node.description`, which is the field the console displays. Default `""`.
- `keywords`: a list of tags for the dataset itself, unioned with anything
  already set (`{"keywords": ["multi-beam", "ti64"]}`). Filter with
  `aiida-pino datasets --keyword multi-beam`.
- `batch_keyword`: despite its name, this sets the dataset's name. It is the
  regroup tag stored on every imported simulation, and the batch resolves its
  base dataset by it. The dataset's keyword list is `keywords` above. Default
  `""`.
- `usable_for_training`: boolean marking whether the runs are fit for training.
  Default `True`; it is a node extra, so you can flip it per simulation later
  (`node.base.extras.set("usable_for_training", False)`) without re-importing.
- `exclude_globs`: list of glob patterns relative to `source_path` (e.g.
  `["smoke/*", "*_twin.npz"]`); a case whose NPZ matches one is skipped.
  Default none.
- `trust_npz_payload`: boolean, default `False`. When a run is stored verbatim
  (no `time_target_frames`), the importer reads every selected field once to
  reject non-finite values; `true` skips that pass and stores the file as-is.

Each stored simulation serves the `x`/`y`/`z`/`time_s` grids, constant `P` and
`V_scan` conditioning channels taken from the case manifest (never the NPZ
filename), the selected fields (`Velocity` as three channels), and full
provenance: the verbatim `case_manifest.json`, laser scalars
(`laser_power`, `V_scan`, `beam_radius`) as queryable extras, and the material
block (plus the parsed material JSON when `materials_root` is given). A case
whose manifest lacks a positive laser power or scan speed fails loudly instead
of importing zero conditioning; a field with non-finite values fails at import;
manifest-only cases (no NPZ export) are skipped and recorded in the summary.

Storage is always the compact NPZ itself (`storage_mode="npz_file"`): without
`time_target_frames` the source file is stored verbatim from its realpath, with
it a frame-sliced compact copy is stored. The conditioning channels are
synthesized at read time, never materialized to disk.

## Running FLOW-3D simulations through AiiDA

### Where each run is filed

A FLOW-3D run submitted through AiiDA does three things without further
instruction: it solves, it ingests the compact NPZ into a `PinoSimulationData`
node, and it files that simulation into a base dataset. This holds for a
single run as much as for a batch, so a one-off case also ends up in a dataset.

Where it lands, in order of precedence:

1. `target_dataset`: an existing base dataset you name explicitly. The case
   joins that dataset and nothing else is consulted.
2. `dataset_profile.batch_keyword`: a base dataset of that display name is
   extended if one exists, created if not.
3. `flow3d_runs`: the fallback for an unnamed single run.

```bash
# single run, filed into an existing dataset
aiida-pino submit flow3d --python-code pino-python@local \
    --run-config @case.json --target-dataset <PK>

# a whole sweep into the same dataset
aiida-pino submit flow3d-batch --python-code pino-python@local \
    --config @a.json --config @b.json --target-dataset <PK>
```

A named target also overrides the rule that a batch child defers filing to its
parent: if you named the dataset, that is where the case goes. A superseded
dataset is refused at submit time: every listing reads its replacement, so
filing into the retired node would hide the new cases.

`Flow3DRunCalculation` (`pino.flow3d_run`) launches the sibling
`flow3D_scripts` runner (`python/run_rectangular_case.py`) as a tracked AiiDA
job, and two workchains wire it into the dataset pipeline:

- `Flow3DGenerateWorkChain` (`pino.flow3d_generate`) takes one config, runs the
  case, then ingests its `npz/<case>.npz` export with the raw FLOW-3D NPZ
  importer into a `PinoDatasetCollectionData`.
- `Flow3DBatchWorkChain` (`pino.flow3d_batch`) runs a list of configs through
  the generate workchain, one dataset per case.

```python
from aiida import orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

run_config = {  # the FULL flow3D_scripts config JSON, passed through verbatim
    "laser": {"power_w": 250.0, "scan_speed_m_per_s": 1.2, "beam_radius_m": 6.5e-5},
    "simulation": {"finish_time_s": 2.0e-3},
    # ... geometry, mesh, material, everything the runner's schema supports
}

submit(
    WorkflowFactory("pino.flow3d_generate"),
    run_config=orm.Dict(dict=run_config),
    execution_profile=orm.Dict(dict={"case_name": "my_case_v1", "cores": 2}),
    dataset_profile=orm.Dict(dict={"time_target_frames": 100}),
    python_code=orm.load_code("pino-python@local"),
)
```

`run_config` is not interpreted by aiida-pino: it is written to disk and handed
to the runner verbatim, so every config field `flow3D_scripts` supports works
without plugin changes. The `execution_profile` carries the runner options
(all optional): `flow3d_scripts_root`, `out_root`, `case_name`, `cores`,
`threads_per_rank`, `prepare_only`, `overwrite`,
`export` (`npz`/`vtk`/`vtk-surface`/`vtk-legacy60`/`both`/`none`, default
`npz`), `flow3d_env`, `post_home`, `timeout_s`, `watchdog`, `stall_s`,
`resource_scope`, `memory_max`, `keep_flsgrf`, `seat_priority`. For
batches, `run_configs` is an `orm.List` of config dicts and each child gets a
unique case name (`<case_name>_0000`, `_0001`, ...) so no run directory is
ever overwritten.

**Watchdog and wall timeout.** Solves (not prepare-only runs) launch the
runner repo's stall/divergence monitor (`python/monitor_run.py`) as a sidecar:
it watches the solver's `hd3msg.<case>` progress and SIGKILLs only that
solve's process subtree when simulated time stops advancing for `stall_s` wall
seconds (default 900) or the time step collapses/goes non-finite, so a stuck
solve frees the FLOW-3D seat and fails its CalcJob without holding a daemon
worker. Disable with `watchdog: false`; waiting for the machine-wide
seat lock produces no progress rows and is never counted as a stall.
`timeout_s` additionally wraps the runner in a hard wall-clock cap (coreutils
`timeout`, TERM then KILL). That clock includes any seat-lock wait, so leave it
unset unless you control what else holds the seat.

**Runner resolution.** The runner repo is never copied (it imports sibling
modules from its own directory). Its root is resolved in order from the
profile's `flow3d_scripts_root`, the `F3D_SCRIPTS_ROOT` environment variable,
and a `flow3D_scripts` checkout sitting beside this one (`../flow3D_scripts`
in this repository). A vendored `external/flow3D_scripts` directory is consulted
last, with a warning, if one exists; this repository ships none.

**License.** The FLOW-3D license endpoint comes only from the
environment the job runs in: `F3D_LICENSE='PORT@HOST'` (or
`F3DTKNUX_LICENSE_FILE` / `LM_LICENSE_FILE`). The execution profile rejects any `license` key: endpoints are
never stored in AiiDA nodes, logs, or documentation.

**One solve at a time, big files.** FLOW-3D solves are serialized machine-wide by a seat
lock (`~/.flow3d/spool/.seat.lock`); a second concurrent run blocks until the
seat frees up, which is why `max_concurrent_runs` on the batch workchain
defaults to 1.

**Queue order: `seat_priority`.** An integer in the `execution_profile`, higher
first, default 0. It orders queued waiters only. A solve already holding the
seat is never interrupted, because killing a run that is already using licence
time to start a more important one wastes both. Waiters announce themselves in
`~/.flow3d/spool/waiting/<pid>.json`, and a run stands aside for as long as any
live higher-priority run is queued. A steady stream of urgent work therefore
keeps a low-priority run waiting indefinitely; the queue is visible in
`aiida-pino paused` and `aiida-pino flow3d-queue`.

The only bound is ticket validity: a ticket stops counting once its owner dies
or it exceeds `TICKET_MAX_AGE_S` (6 h), so a crashed, zombied or stopped waiter
cannot block the seat. There is no separate time limit on deference. An
earlier one-hour cap let a `seat_priority=9` run lose the seat to a priority-0
run that had been queuing for an hour, inverting the priority.

If the bookkeeping fails, everything falls back to the plain first-come lock.
The runner's own lock and the task spooler do not read priorities at all, so a
direct run can still take a free seat ahead of a queued one. Priority
orders a queue and cannot *park* work; holding work back until further notice
is what pausing is for. `seat_priority` covers the FLOW-3D licence seat only;
GPU trainings and evaluations use the separate boolean `gpu_seat_priority` in
their own profiles.

The large artifacts (`flsgrf1.<case>` and the NPZ export) stay on disk under
`out_root` (default `<runner root>/runs/`); provenance records their absolute paths in
`run_summary`, and only small curated files (case manifest, `summary.md`,
`path.png`, `prepin.<case>`, the last 200 lines of `runhyd.log`) are retrieved
into the `run_artifacts` folder node. Prepare-only runs (`prepare_only: true`,
no license needed) and non-NPZ exports skip the dataset ingest; the generate
workchain still finishes ok with just the run outputs.

## Steady (time-averaged, beam-centred) runs

A steady run states the steady layout twice, and the two must agree: once in
the training profile and once in an explicit `derivation_profile`. Passing the
derivation profile declares that the layout substitution is intended. Without
it a layout guard refuses the submission, and with only half of the layout
stated the derivation validator refuses it. Neither refusal mentions the other
check, so state both.

`steady_training.json` (the first five keys are required in every training
profile; the last five state the steady layout):

```json
{"config_name": "<steady-config>", "run_name": "<name>", "cuda_device": "auto",
 "wandb_offline": true, "hydra_overrides": [],
 "include_time": true, "time_as_batch": false, "moving_ref_frame": true,
 "time_avg_window": 30, "moving_frame_steps": 30}
```

`steady_derivation.json`, the same layout:

```json
{"include_time": true, "time_as_batch": false, "moving_ref_frame": true,
 "time_avg_window": 30, "moving_frame_steps": 30}
```

```bash
aiida-pino presubmit --train-dataset <PK> --val-dataset <PK> --python-code <code> \
  --training-profile @steady_training.json --derivation-profile @steady_derivation.json
aiida-pino submit train-eval --train-dataset <PK> --val-dataset <PK> --python-code <code> \
  --training-profile @steady_training.json --derivation-profile @steady_derivation.json
```

`time_as_batch: false` belongs in the derivation profile too. Left out there, it
is inferred from `moving_ref_frame: true` with a logged warning, since the moving
frame exists only as the beam-centred average of this layout; stating
`time_as_batch: true` together with `moving_ref_frame: true` is refused.

The averaging and beam-centring happen offline, once: the result is a
materialised derived collection (`dataset_role: derived`, a consolidated
`var_1_time_steps30_avg30.npz`) whose arrays already have a single time frame.
Later submissions with the same derivation reuse it without rebuilding.

A pre-averaged NPZ folder cannot be ingested directly; `A-NO-DERIVED-INGEST` in
`assumptions.md` explains why that path is refused.

## Running on a remote GPU computer

A second machine is an AiiDA `Computer` with an SSH transport and a code on it.
Everything below uses placeholders: `<BOX>` is the computer label, `<PK>` a
dataset pk. No real hostnames, addresses or usernames belong in this file.

**Register the Computer with the `pino.direct_pgid` scheduler**
(`verdi computer setup ... -S pino.direct_pgid`). It is `core.direct` plus
process-group kill: with the stock scheduler `aiida-pino kill` on a remote job
stops the wrapper shell and leaves the training process running on the box.

**Datasets do not travel with the job.** A job's dataset is staged from a
runtime cache, and that cache is built from this machine's object store, which
the compute box cannot reach. The copy is therefore put there once, by hand:

```bash
aiida-pino stage-remote <PK> --computer <BOX>
aiida-pino stage-remote <PK> --computer <BOX> --fields T fl   # a field subset
aiida-pino stage-remote <PK> --computer <BOX> --dry-run       # measure, transfer nothing
aiida-pino stage-remote <PK> --computer <BOX> --remote-root /scratch/aiida-pino/runtime  # not the box's $HOME/.cache
aiida-pino stage-remote <TRAINING_PK> --computer <BOX> --for-training  # every collection that past training's jobs consumed
```

One-time per (dataset, field set, machine); every later submission to that box
reuses it. Expect tens of minutes for a large collection on 1 GbE. `--fields`
must match the training profile's runtime fields, or the job looks for a
different cache. A job reads a derived collection built from the dataset you
passed. `--for-training` stages what a past training actually consumed, and
`presubmit` (below) answers the question for a new one.

The copy is verified on the box before it is recorded, against an exact set of
relative paths and sizes. The exact set is needed because the transport reports
nothing back, creates missing destination parents silently, and on a re-put
nests a duplicate tree inside the destination. That tree has more files and
more bytes than expected, so a count or size check would pass on the most
likely corruption.

**Check before you submit.**

```bash
aiida-pino presubmit --train-dataset <PK> --python-code <code> \
  --training-profile code_label=<BOX> ...    # every refusal the submit would hit, nothing submitted
aiida-pino staged --computer <BOX>                          # the runtime caches on the box
aiida-pino staged --computer <BOX> --check <PK> --derivation @derive.json --fields T fl
                                                            # is what THIS submission consumes staged?
aiida-pino staged --computer <BOX> --verify                 # also ask the box whether each path is still there
aiida-pino unstage <PK> --computer <BOX>                    # dry run; --yes deletes the cache and retracts its record
```

The bare `staged` listing shows base datasets, while the job consumes whatever
the derivation resolves to, so seeing your uuid listed confirms nothing.
`--check` resolves the derivation the same way the submission will.

**Submitting there.** Name the machine in the training profile:

```bash
aiida-pino submit train-eval --train-dataset <PK> --python-code <code> \
  --training-profile code_label=<BOX> --training-profile run_name=my-run
```

`cuda_device` is promoted to `auto` for a remote submission unless you set it
yourself. The shipped default is `"0"`, which on a two-card box pins every run
to card 0 and serialises them without erroring.

**If a job reports a missing runtime manifest**, the dataset was never staged to
that box, or was staged and later removed. `staged --check` shows which; re-run
`stage-remote` and resubmit. There is no repair path on the remote side, because
there is no object store there to rebuild from. `unstage` is the only thing that
reclaims a box's disk: retracting a record alone leaves the bytes there.

**Limits.** Dataset *unions* cannot run remotely: the union exporter writes
members with pack-index entries pointing into this machine's object store, so a
remote copy would resolve to nothing and staging refuses it. Compose
the members into a base or composed dataset and stage that.

## AiiDA entry points

Registered under the `pino.*` namespace in `pyproject.toml`:

- **Data nodes**: `pino.simulation`, `pino.dataset`, `pino.dataset_collection`,
  `pino.dataset_union`, `pino.model`, `pino.model_evaluation`.
- **CalcJobs**: `pino.flow3d_run`, `pino.raw_flow3d_to_dataset`, `pino.train`,
  `pino.eval`, `pino.warm_runtime_cache`.
- **WorkChains**: `pino.flow3d_generate`, `pino.flow3d_batch`,
  `pino.raw_flow3d_import`, `pino.derive_dataset`, `pino.train_model`,
  `pino.evaluate_model`, `pino.train_eval`, `pino.eval_only`,
  `pino.import_model`.
- **Schedulers**: `pino.direct_pgid` (see "Running on a remote GPU computer").

Load any of them with AiiDA's factories, e.g.
`WorkflowFactory("pino.train_eval")`.

## Data model

The "database" is AiiDA's provenance store holding the `pino.*` nodes above.
[`database.md`](database.md) is the reference: what each node stores, how
simulations compose into collections and unions, how models link to their
datasets and evaluations, and which relationships are real provenance edges
versus by-UUID references.

The short version: `raw_flow3d_to_dataset` stores one `PinoSimulationData` per
simulation and wraps them in a `PinoDatasetCollectionData` (the training
dataset). `PinoDatasetData` and `PinoDatasetUnionData` are the older
single-tensor kinds and are kept for compatibility. Derived datasets
(moving-frame crops, super-resolution, etc.) are reconstructed on demand from a
canonical base, so their arrays are not stored twice.

### Annotations: split, keywords, description

Stored nodes can be annotated after the fact. Annotations live in AiiDA extras
and the node description, both mutable, so nothing needs re-importing:

- **Simulations** carry a `split` extra, plus a `keywords` list and a free-text
  description. The split vocabulary (`aiida_pino.split_values.SPLIT_VALUES`) has
  seven values:
  - `train`, `val`, `test`: the usual three; the console composes subset
    datasets from them.
  - `eval`: held out like `test`, for cases that live in an evaluation-only
    set (`HELD_OUT_SPLIT_VALUES` is `test` + `eval`; training refuses both).
  - `excluded`: kept out of every split on purpose (a replicate, a bad case);
    the value new work should use.
  - `duplicate_excluded`: the same meaning, kept writable because it was
    already live on existing nodes; do not use it for new exclusions.
  - `unassigned`: the default.
  Set it with `aiida-pino split PK VALUE --reason "why"`; the reason is
  appended to the node's `split_audit` extra with the actor and time, so an
  exclusion carries a record of who did it and why.
- **Datasets and models** carry a `keywords` list and a description.
- Keywords are free-form; the suggested vocabulary is `keyhole`, `conduction`,
  `random path`, `variable material`, `variable path length`, `straight path`,
  `even timesteps`, `steady`, `downsampled`, `smoke`.

Both the web console and the CLI filter on them (`aiida-pino sims --split train
--keyword keyhole`; `GET /simulations?split=&keyword=`), and a filtered
selection can be composed into a new collection (`aiida-pino collection
create`). This is how training sets are carved out of the store.

### Storage: FLOW-3D NPZ kept verbatim

Every FLOW-3D simulation is stored with `storage_mode="npz_file"`: the
compressed compact `npz/<case>.npz` is kept as one repository object, about 20x
smaller than unpacked per-array `.npy` files (the broadcast
`x/y/z/P/V_scan/time_s` channels are synthesized at read time and never stored).
Training materializes those members as flat NPZ folders using hardlinks where
the filesystem allows, so array data is not duplicated on disk.

Older imports stored uncompressed per-array `.npy` objects
(`storage_mode="arrays"`). Convert legacy nodes
in place:

```bash
aiida-pino compress PK          # dry run: members and bytes that would convert
aiida-pino compress PK --yes    # de-broadcast, verify byte-identical reads, swap
```

The conversion is lossless and verified per array before the swap; pk/uuid,
attributes, links and extras are untouched. The old uncompressed objects
become unreferenced; `aiida-pino reclaim` returns the disk.

Storing an arrays-mode simulation is refused at `store()` level. The escape
hatch `AIIDA_PINO_ALLOW_ARRAYS_STORAGE=1` exists for tests and deliberate
legacy imports (e.g. the ParaView VTK path, which has no compact NPZ source);
anything stored through it should be `compress`ed right after.

### Metadata-only eviction: reclaim disk, keep the record

A simulation's metadata lives in DB attributes; its arrays are one repository
object (`payload.npz`, a few hundred MB for a typical case).
The two are independent, and provenance is link-based (a trained model links
to the *collection*, which references its simulations by UUID), so the array
bytes can be deleted while every attribute, link, and node identity (pk/uuid)
stays intact. After eviction the node is "metadata-only": it still lists in
every dataset, still carries its P/V/shape metadata, and a model trained on it
still resolves its training source; only reading the arrays raises a clear
"metadata-only" error.

```bash
aiida-pino evict PK                                  # dry run: members, and MiB it would free
aiida-pino evict PK --yes --reason "free disk"       # delete the arrays for real
```

In the console it is the **Evict data…** button on a dataset's detail pane (it
dry-runs and confirms first), and evicted datasets show a `metadata only` /
`partly evicted` chip in the list.

Guards and mechanics:

- **Eviction is per simulation, and data is shared.** A simulation belongs to
  one base dataset but can be unioned into composed/derived sets, so evicting
  here empties it everywhere it is referenced (the metadata survives in all of
  them). The object store is content-addressed, so evicting one simulation
  never touches bytes another simulation shares.
- **`core` datasets are refused**, and so is any dataset whose members are also
  in a core set: reference data cannot be evicted through a non-core view.
- **It is one-way.** The arrays are gone; restoring them means re-importing from
  the original source. Derived/reconstructable datasets can be re-materialized
  from their base, but an evicted base import is final.
- **Disk is reclaimed by a repack.** Eviction drops each node's *reference* to
  its bytes; on a packed store the bytes stay in their pack file until a
  repack. Run `aiida-pino reclaim` afterwards to shrink `~/.aiida`: it deletes
  the unreferenced objects and repacks only the packs that held them, so the
  transient space (about one pack) and the exclusive-lock window scale with
  what was evicted. `verdi storage maintain --full`, by comparison, repacks
  every pack.

```bash
aiida-pino reclaim         # dry run: unreferenced objects, GB freed, packs touched
aiida-pino reclaim --yes   # delete + repack the affected packs (stop the daemon and console first)
```

Reclaim needs exclusive access, so stop the AiiDA daemon and the web console
before `--yes`; it fails with a clear message otherwise.

### What is in the provenance graph, and what predates it

An import claims the simulations it creates (`simulations.*` outputs on the
importing CalcJob), and dataset filing, the split path included, composes
through a calcfunction that takes the members as inputs. So for anything created
under this rule:

- every simulation has a creating process,
- `simulation -> compose_dataset -> dataset` is a real path, which is what makes
  `verdi archive create -N <dataset>` export the data and `verdi node delete` on
  a member warn about the datasets holding it.

Datasets created before the rule do not have this and cannot be given it:
AiiDA provenance is immutable, and a fabricated edge would be worse than an
absent one. For simulations and collections that predate it, membership lives
only in the `member_simulation_uuids` attribute, and
`verdi archive create -N <old dataset>` exports the collection node alone.
`python -m aiida_pino.utils.integrity [--profile NAME]` reports which nodes are
in that state (expect 2-5 minutes; it walks every simulation and collection).

### What every run records about itself

Besides its inputs and outputs, each job stamps queryable extras on its calcjob
so "what did this cost, and what produced it" is answered by a query:

| Extra | On | Meaning |
|---|---|---|
| `runtime_wall_seconds` | train, eval and FLOW-3D run jobs | wall time of the payload, seat wait included |
| `runtime_cpu_seconds`, `runtime_max_rss_mb` | train, eval and FLOW-3D run jobs | CPU time and peak RSS (rusage) |
| `env_hostname`, `env_gpu_name` | train, eval and FLOW-3D run jobs | where it ran |
| `gpu_vram_peak_mb` | GPU jobs | measured peak, feeds automatic VRAM reservation |
| `solver_elapsed_seconds`, `solver_cpu_seconds`, `solver_cores` | FLOW-3D | the solver's own clock, excluding seat wait and export |
| `solver_version`, `solver_build` | FLOW-3D calcjobs and simulations | the FLOW-3D build, from the run's `runhyd.log` banner |

Dataset-import jobs are not timed: `raw_flow3d_to_dataset_adapter.py` never
wraps its payload in a `RuntimeMeter`, so `stamp_runtime_extras` never fires
for them. Their total wall time is still recoverable from process timestamps;
only the payload-versus-overhead split is lost.

The full detail (library versions, GPU driver, git SHAs of the checkouts) is
in the job's `runtime.environment` block inside its summary Dict.

Two limits:

- **`env_*` cannot be backfilled.** It describes the machine at the moment of
  the run; nothing in an old job's artifacts records it. Jobs that finished
  before this capture existed have compute times and no environment record.
- **`solver_version` comes only from the run's own `runhyd.log`**, never inferred from a sibling run
  or from the version installed today. A node whose run has no readable log stays blank.

## Configuration and units

Settings are passed around as plain dictionaries called *profiles*:

- a **dataset profile** controls what an import keeps (fields, frames, the
  annotations above),
- a **derivation profile** controls the layout a job consumes (moving frame,
  time averaging, strides, derived fields),
- a **training profile** controls the model run (learning rate, steps, losses,
  etc.), and an **evaluation profile** the scoring,
- an **execution profile** carries the FLOW-3D runner options.

Profiles are validated against fixed key sets, so a typo is caught early;
`aiida-pino profile-keys <kind>` prints every accepted key with its default.
Physical inputs use SI units. One *sample* is one simulation. Profiles also
accept `hydra_overrides` (`key=value` strings passed
straight to the underlying `lpno` config system).

### Environment variables

Each variable is read by the process named in the third column. For a
daemon-side variable that is the daemon worker, so set it where the daemon is
started; setting it in the shell that submits has no effect.

| Variable | Effect | Read by |
|---|---|---|
| `AIIDA_PINO_RUNTIME_CACHE` | root of the local runtime cache (default `~/.cache/aiida-pino/runtime`) | daemon, CLI |
| `AIIDA_PINO_REMOTE_READ_TIME` | comma-separated computer labels whose `lpno` checkout supports the read-time crop; a strided submission to a box not listed is refused (exit 303) instead of training the full extent against a profile claiming a crop | daemon |
| `AIIDA_PINO_RESOURCE_GUARDS` | `enforce` or `warn`: whether the memory and disk guards (the export pool's memory clamp, the disk-fit check before a runtime-cache build, the `host_ram_mb` floor on a training profile) refuse and clamp, or print the same numbers as a warning and proceed. The thresholds were measured on the development machine; the default in `src/aiida_pino/utils/guards.py` is what this checkout ships | daemon, CLI, console |
| `AIIDA_PINO_EXPORT_WORKERS` | cap on the parallel export pool that builds a runtime cache (clamped by the memory estimate under `enforce`; under `warn` the value is used as given and the estimate is printed beside it) | daemon |
| `AIIDA_PINO_BUILD_WAIT_S` | how long a runtime-cache build waits for enough free disk before giving up, in seconds (default 1800; concurrent builds each pass a stateless check, so waiting is what keeps two of them from filling the disk together) | daemon, CLI |
| `AIIDA_PINO_PACK_INDEX` | `0` disables staging-side pack-index entries for local jobs (opt-out; the loader-side switch is `PINO_PACK_INDEX_DISABLE=1`) | daemon |
| `AIIDA_PINO_ALLOW_ARRAYS_STORAGE` | `1` lets `store()` accept an arrays-mode simulation (tests and deliberate legacy imports only) | any process storing nodes |
| `PINO_SPOOL_ENFORCE_RAM` | `1` restores the cgroup `MemoryMax` enforcement of `host_ram_mb` (default off); read by `lpno`' gpu_spool on the training box, mentioned here because the declaration's meaning depends on it | gpu_spool (sibling package) |
| `PINO_CONDUCTION_CACHE` | where `derive-fields` and the loader cache precomputed conduction fields (default `~/.pino/conduction_cache`) | CLI, training job |
| `PINO_GPU_LEASE_DIR` | gpu_spool's shared VRAM ledger, read by the Monitor tab (default `/var/tmp/pino_gpu_leases`) | console |
| `F3D_SPOOL_DIR` | the FLOW-3D spool directory holding the seat lock and queue (default `~/.flow3d/spool`) | FLOW-3D run adapter, console |
| `F3D_SCRIPTS_ROOT` | the `flow3D_scripts` checkout, second candidate in "Runner resolution" | FLOW-3D run adapter |
| `F3D_LICENSE` (or `F3DTKNUX_LICENSE_FILE` / `LM_LICENSE_FILE`) | the FLOW-3D licence endpoint, `PORT@HOST`; never stored | FLOW-3D run adapter |
| `AIIDA_PINO_WEB_PYTHON`, `AIIDA_PINO_WEB_PORT`, `AIIDA_PINO_WEB_FRONTEND_PORT`, `AIIDA_PINO_WEB_AIIDA_PROFILE`, `AIIDA_PINO_WEB_LAN_IP`, `AIIDA_PINO_WEB_PUBLIC`, `AIIDA_PINO_WEB_FLOW3D_SCRIPTS_ROOT` | the web console launcher; any other `AIIDA_PINO_WEB_*` maps onto a `Settings` field (`web/README.md`) | `web/dev/run_local.sh`, backend |

### Daemon

Daemon-side variables (the table's "daemon" rows) only take effect on the
worker that is running, so restart the daemon (`verdi daemon restart`) after
changing one, and start the daemon from a shell that has them set.


## Tests

Run the suite in the environment that has `aiida` installed. Two tiers:

```bash
python3 -m pytest -m "not slow" -q     # the fast tier — run this after any change
python3 -m pytest -q                   # everything — before a commit that matters
```

The `slow` marker is applied automatically in `tests/conftest.py` from each
test's fixture closure: anything that resets the AiiDA storage or builds nodes
through the factories is slow. Nothing is marked by hand, so a new test is
classified correctly without anyone remembering to do it.

The suite uses an in-memory AiiDA profile and fake Codes, so it needs no
daemon, GPU or license. Two modules
(`tests/test_beam_following_derivation.py`, `tests/test_laser_overrun_window.py`)
import `lpno` at module level and fail at collection without it;
the rest of the suite does not need the sibling package. In an environment
without `aiida` the whole suite fails at collection, which points to the wrong
Python environment.

If a run appears to hang, check the FLOW-3D seat lock first: the adapter holds
`<spool>/.seat.lock` for its process lifetime, and a test that takes it twice in
one process blocks on itself. With the `pytest-timeout` plugin (in the `dev`
extra), `--timeout=300` turns such a hang into a named failure.

## Known limits

- The IN718 acquisition loop runs unattended or with a person approving each
  round; a general loop that retrains a shared model whenever suitable data is
  filed is not built yet (see [Status](#status)).
- **FLOW-3D is a first-class source**: laser power and scan speed come from each
  run's `case_manifest.json`, never the filename; an all-zero or metadata-less
  FLOW-3D member raises instead of importing; and `flip_axes` defaults by
  `source_type` (`flow3d` → identity), so FLOW-3D geometry is not transposed.
- Design assumptions live in [`assumptions.md`](assumptions.md), known issues
  with their workarounds in [`potential_bugs.md`](potential_bugs.md), and
  developer practice and the roadmap in
  [`docs/developer_notes.md`](docs/developer_notes.md).

## Repository layout

From `git ls-files`, two levels deep:

```text
aiida-pino/
├── web/                   # Browser console: FastAPI backend (app/) + static front end (public/) + dev/run_local.sh + logo_aiida.svg
├── src/aiida_pino/
│   ├── cli.py             # The `aiida-pino` command (entry point aiida_pino.cli:main)
│   ├── split_values.py    # The split vocabulary, importable without AiiDA
│   ├── calculations/      # CalcJobs (script-backed AiiDA jobs)
│   ├── console/           # The service layer shared by the CLI and the web console
│   ├── data/              # Native Data node types (simulation, dataset, model, ...)
│   ├── parsers/           # Turn calcjob output into Data nodes
│   ├── schedulers/        # pino.direct_pgid (process-group kill for remote boxes)
│   ├── scripts/           # Standalone adapters run inside the job sandbox
│   ├── utils/             # Dataset derivation, registries, import helpers, maintenance tools
│   └── workflows/         # WorkChains wiring calcjobs together
├── tests/                 # License-free, daemon-free test suite
├── bin/                   # daemon_restart.sh (restart and verify the AiiDA daemon)
├── docs/                  # developer_notes.md: practice and roadmap
├── pyproject.toml         # Package metadata and every pino.* entry point
├── daemon.env.example     # Template for the untracked daemon environment file
├── LICENSE
├── database.md            # Data-model reference
├── assumptions.md         # Non-trivial design choices
└── potential_bugs.md      # Known issues: symptoms, causes, workarounds
```


## Troubleshooting

- **Empty Code dropdowns in the console**: no Codes are registered, or the
  console can't see them. Register them (see [setup](#one-time-aiida-setup))
  and reload the Environment tab.
- **A submission is refused with `daemon_unavailable`**: the RabbitMQ broker
  or the AiiDA daemon is down; submitting needs both. Check `verdi status`.
- **Changes to plugin code don't take effect**: restart the daemon
  (`verdi daemon restart`) and the web console, so the new classes are picked
  up; a running worker keeps serving the code it imported.
- **`pytest` fails at collection with "No module named 'aiida'"**: you're in an
  environment without AiiDA; use the one that has `lpno` + `aiida`.
