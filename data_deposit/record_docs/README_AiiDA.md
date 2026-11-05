# How to open `lpno_provenance.aiida`, and re-run what it records

The archive is an AiiDA export (`sqlite_zip` format, written by aiida-core
2.8.0). It contains the complete provenance graph behind the manuscript's
results tables: model nodes with their weights, evaluation nodes with every
metric, simulation nodes with all metadata, the dataset collections that group
them, and every process node and link. The simulation arrays themselves are
in the `payloads_<family>.tar` files of the restricted-access record (FLOW-3D
WELD output, shared on request for non-commercial use); download both records
into one directory, and §4 says how the two join. Files that carried solver
output were also moved there from the archive; a node that lost files lists
them in its `lpno_restricted_files` extra.

This file is the AiiDA route: import the archive, trace any number to its
sources, and re-run an evaluation or a training with the re-run recorded in
the same graph. To re-run numbers without AiiDA (no import, no profile, no
message broker), see "Reproducing a number" in `README.md`.

## 1. Versions

The nodes were produced with, and open cleanly under:

| package | version | needed for |
| --- | --- | --- |
| Python | 3.10 | |
| `aiida-core` | 2.8.0 | reading the archive. Newer 2.x reads it after an automatic migration; an older aiida-core than the one that wrote it cannot (archive migration is forward-only) |
| `aiida-pino` | 0.1.0, from the code repository at **[TO FILL: tag]** | the `PinoSimulationData`, `PinoDatasetCollectionData`, `PinoModelData` and `PinoModelEvaluationData` classes, the workflows, and the `aiida-pino` command |
| `lpno` | 0.1.0, same repository | imported by `aiida-pino`; the datasource, models and metrics |
| `nvidia-physicsnemo` | 1.2.0, `torch` 2.8.0 | to load weights or re-run an evaluation or a training |

The archive carries no record of these plugin versions itself; this file is
that record. `tools/setup_env.py` in the code repository installs exactly
these versions (§2).

## 2. Install and import

```bash
git clone https://github.com/azbenoit/lpno_code lpno-code       # at the tag of §1
cd lpno-code
python3.10 tools/setup_env.py --prefix ../lpno-env --torch cpu --with-aiida
source ../lpno-env/bin/activate
export AIIDA_PATH=$PWD/../aiida              # optional: keeps this profile out of ~/.aiida
verdi presto                                 # a zero-configuration local profile (SQLite)
verdi archive import <deposit>/lpno_provenance.aiida   # 6.2 GB; about 5 minutes
verdi archive import <deposit>/lpno_provenance_rounds.aiida   # optional: the per-round retrains (README.md)
verdi archive import <deposit>/lpno_provenance_evals.aiida    # the 10 added evaluations (README.md)
```

`tools/setup_env.py` installs torch 2.8.0 from the PyTorch index of the chosen
build (`cpu`; `cu126` for a Volta GPU such as the V100; `cu128` for Ampere or
newer), every other package at the versions the models were trained with,
`nvidia-physicsnemo-sym` and `torch-scatter`, and `aiida-pino` with its
command-line tool. A plain `pip install` of the two packages resolves to newer,
untested versions and leaves `lpno.models` unimportable. Any
Python 3.10 works as the interpreter, e.g. `micromamba create -p py310 python=3.10`.
With a conda or micromamba Python on Red Hat 9 and similar, `verdi` can fail at
`import ssl` with "version `OPENSSL_3.3.0' not found" (the system's user lookup
loads the older system OpenSSL first); the environment `setup_env.py` builds
imports `ssl` at start-up to prevent it, and in any other environment
`python -c "import site, pathlib; pathlib.Path(site.getsitepackages()[0], '00-load-own-openssl.pth').write_text('import ssl')"` does the same.

**The broker.** `verdi presto` uses RabbitMQ if one is running and otherwise
makes a profile without a broker, which has no daemon. Everything in this file
works without one: browsing and tracing need nothing, and the re-runs of §7 run
in the terminal with `aiida-pino --foreground`. A broker only adds background
submission (`verdi daemon start`); install RabbitMQ 3.8.14, the newest version
AiiDA supports with a default server configuration (`docker run -d -p 5672:5672
rabbitmq:3.8.14`), before `verdi presto`, or point an existing profile at one with
`verdi profile configure-rabbitmq`. AiiDA warns on newer versions, which need the
server's `consumer_timeout` disabled.

**Refer to nodes by UUID.** Node primary keys (`pk`) are reassigned on import;
UUIDs are stable. `provenance_log.csv` lists every UUID behind every printed
number (§6), `payload_manifest.csv` every simulation's, and every `aiida-pino`
command below takes a UUID wherever it takes a pk. Labels need not be unique,
so always resolve a model through its UUID.

## 3. Reading without the plugins

`aiida-core` alone opens everything. Without `aiida-pino` installed, the
custom data nodes load as plain `aiida.orm.Data`: their attributes, extras,
links and repository files are all there; only the plugin's convenience
methods are missing. Process nodes load normally, so the graph walks; only
re-running a workflow needs the plugin.

```python
from aiida import load_profile, orm
load_profile()
sim = orm.load_node("<simulation uuid>")
sim.base.attributes.all["laser_power"], sim.base.attributes.all["V_scan"]   # W, m/s
sim.base.attributes.all["simulation_metadata"]["flow3d_case_manifest"]      # the full case manifest
sim.base.extras.all["split"], sim.base.extras.all["material_name"]
model = orm.load_node("<model uuid>")
model.base.repository.list_object_names()          # ['.hydra', 'best_checkpoint.json', 'u_network.0.pth', ...]
weights = model.base.repository.get_object_content("u_network.0.pth", mode="rb")
ev = orm.load_node("<evaluation uuid>")
ev.base.attributes.all["evaluation_metadata"]["metric_units"]["kelvin"]     # substrate, surface and whole-domain T metrics, in kelvin
```

## 4. Joining the archive to the arrays

Simulation nodes in the archive carry no `payload.npz` (the `data_evicted`
extra says so, with the reason). The arrays are in the restricted record's
tarballs, one member per simulation named by UUID, and can be read directly
(`<deposit>` below is the directory holding both records):

```python
import tarfile, numpy as np, json
sim = orm.load_node("<simulation uuid>")
family = sim.base.attributes.all["batch_keyword"]
with tarfile.open(f"payloads_{family}.tar") as t:
    z = np.load(t.extractfile(f"{sim.uuid}.npz"))
    manifest = json.load(t.extractfile(f"{sim.uuid}.manifest.json"))
```

To re-run anything (§7) the arrays have to be back in the nodes themselves,
which `aiida-pino restore-payloads` does. It reads each member in place from its
tar, checks it against the sha256 in `payload_manifest.csv`, and writes it
into the deposited node, keeping its UUID, so every collection, model and
evaluation that names the node works as it did before the deposit was made.
Without `--yes` it is a dry run that prints how many simulations, how many bytes
it would add to the profile, and which tars it needs:

```bash
aiida-pino restore-payloads --deposit <deposit> --for <evaluation uuid>        # what an evaluation scored
aiida-pino restore-payloads --deposit <deposit> --for <training workflow uuid> # its training + validation sets
aiida-pino restore-payloads --deposit <deposit> --collection <collection uuid> --yes
aiida-pino restore-payloads --deposit <deposit> --all --yes                    # everything: 157 GB
```

Only the tars named in the dry run need to be present. For scale: the IN718
test split of the two-alloy table is 24 simulations and 2.7 GB; one two-alloy
half's training set with its validation split is 304 simulations and 42 GB.
(`PinoSimulationData.from_npz_file` would instead create a new node with a new
UUID, which no deposited collection references.)

## 5. Groups

The archive is organised in AiiDA groups, which `verdi group list` shows and
`verdi group show <label>` lists:

| group label | holds |
| --- | --- |
| `lpno/models/tab:arch-steady` | the 18 models of the architecture table (9 steady-frame, 9 laboratory-frame from the uniform-drawing re-run) |
| `lpno/models/tab:paths-accuracy` | the 9 weld-path models plus the multi-beam RIGNO's fine-tune donor (the other two donors are among the 9); 10 nodes |
| `lpno/models/tab:preheat` | the 6 preheat and beam-radius models |
| `lpno/models/tab:twoalloy` | the 12 two-alloy models: the Ti-6Al-4V-only FNO, the alloy-label FNO on each half and on both (round 5, three seeds each), and the joint GAOT and RIGNO |
| `lpno/models/fig:lineage` | the variable-length FNO of the weld-path table, whose ancestry the supplement's lineage figure draws |
| `lpno/evaluations/tab:arch-steady` etc. | the evaluations behind each table, one group per table, plus `fig:lineage`, `fig:twoalloy-pred` and `fig:closing` (README.md, "Which node backs which result"); `lpno/evaluations/superseded` holds the one replaced evaluation kept (README.md, "Re-scored evaluations") |
| `lpno/models/fig:al-rounds`, `lpno/evaluations/fig:al-rounds` | in `lpno_provenance_rounds.aiida`: the 36 FNO models of rounds 1 to 4 and their 72 evaluations |
| `lpno/simulations/<family>` | the simulations of each of the 24 families named in `README.md` |
| `lpno/datasets` | every dataset collection (train, validation and test splits, and the base collections they derive from) |

Model nodes also carry a label (`run_name`, e.g. `fno_spkh123_steady_32k`) and
evaluation nodes a label of the form `<model>_on_<dataset>`.

**Input-field settings.** Every model that takes the `conduction_field` input carries an extra `input_fields`
with the settings that shaped its conduction and dose inputs: `conduction_source_sigma_factor` (the standard
deviation of the conduction field's heat source, in beam radii), `power_weighted_conduction`,
`power_weighted_dose`, and `source`. `source` is `stored config` when the model's archived `.hydra/config.yaml`
states the conduction settings, and `code default (key absent from the stored config)` for models trained before the
configuration stated them, which used the code defaults (1.0 and false). Models trained with the current code also
store these settings in `input_fields.json` in their repository, written by the training.

## 6. From a printed number to its sources

Every table cell is an evaluation node. `provenance_log.csv` has one row per
evaluation: the table or figure, the evaluation, the model it scored and that
model's seed, the scored dataset, the training dataset, the training workflow
(the node §7 re-runs), and the headline metrics in kelvin. Take an evaluation's
UUID from it, then:

```python
ev = orm.load_node("<evaluation uuid>")
md = ev.base.attributes.all["evaluation_metadata"]
md["metric_units"]["kelvin"]["T.substrate_abs_err_mean"]   # the printed substrate MAE, K
md["metrics"]["fl"]["iou_mean"]                            # melt-pool IoU
md["metrics"]["T"]["beambox_abs_err_mean"], md["metrics"]["T"]["beambox_n"]   # normalised; n cells
model = orm.load_node(md["model_uuid"])                    # the scored model
test  = orm.load_node(md["dataset_source_uuid"])          # the split it scored
test.base.attributes.all["member_simulation_uuids"]        # its simulations
train = orm.load_node(model.base.extras.all["train_dataset_uuid"])   # what the model trained on
```

`metrics.T.*` values are stored normalised by the model's temperature scale.
`metric_units.kelvin` holds the substrate, surface and whole-domain MAE and
RMSE already converted, and is what the tables print. The beam-box and tail
metrics exist only in normalised form: multiply by `ref_scales.T` from the
model's archived `.hydra/config.yaml`, which is 3000 K for every model in this
record (`provenance_log.csv` has the product).

The same walk is available on the command line:

```bash
verdi node show <evaluation uuid>            # inputs: model, dataset; outputs: metrics
verdi node graph generate <evaluation uuid>  # the provenance graph as a PDF
```

A QueryBuilder over a whole table:

```python
from aiida.orm import QueryBuilder, Group, Node
qb = QueryBuilder().append(Group, filters={"label": "lpno/evaluations/tab:arch-steady"}, tag="g")
qb.append(Node, with_group="g", project=["label", "attributes.evaluation_metadata.metric_units.kelvin"])
for label, kelvin in qb.all():
    print(label, kelvin["T.substrate_abs_err_mean"])
```

## 7. Re-running an evaluation or a training

Both re-runs are recorded in your profile, linked to the deposited nodes they
used. Register the Python of §2 as a code on this machine once:

```bash
verdi code create core.code.installed -n --label pino-python --computer localhost \
  --filepath-executable "$(which python)"
```

**An evaluation** (UUIDs from `provenance_log.csv`):

```bash
aiida-pino restore-payloads --deposit <deposit> --for <evaluation uuid> --yes
export PINO_ALLOW_CPU=1          # only on a machine without a CUDA GPU
aiida-pino --foreground submit eval --model <model uuid> --dataset <scored dataset uuid> \
  --python-code pino-python@localhost --full-frames
```

The metrics land on a new evaluation node in the same form as the deposited
one, so the two compare field by field. `--full-frames` scores every frame, as
the published evaluations did. Without a CUDA GPU the pipeline refuses to run
unless `PINO_ALLOW_CPU=1` is set, which is meant for evaluations and short
trainings: the steady evaluation of the IN718 test split took 2 min 15 s on
8 CPU cores, where a training run would take days.

**A training**:

```bash
aiida-pino restore-payloads --deposit <deposit> --for <training workflow uuid> --yes
aiida-pino --foreground resubmit <training workflow uuid> --override code_label=pino-python@localhost
```

`resubmit` clones the recorded workflow's inputs (training profile, datasets,
derivation) and re-runs them; `--override key=value` changes one setting, e.g.
`max_steps=200` for a quick check. The recording, validation and weights-saving
intervals are shortened with it, so the new model holds the weights of step 200.
A full training needs a CUDA GPU (about 40 min for a 32,000-step FNO run on a
V100); install with `--torch cu126` or `cu128` for it, and see "Running on a
remote GPU computer" in the aiida-pino README to train on another machine.

`--foreground` runs the workflow in the terminal until it finishes, without a
daemon. With a broker and `verdi daemon start`, leave it out and the job runs
in the background.

## 8. Browsing the graph in a browser

The archive can be served read-only, without importing it, and explored with
the AiiDA explorer, the interface Materials Cloud uses for its own databases:

```bash
verdi profile setup core.sqlite_zip -n --profile-name lpno --filepath lpno_provenance.aiida
verdi -p lpno restapi                       # serves http://127.0.0.1:5000/api/v4
# then open https://aiidateam.github.io/aiida-explorer/?api_url=http://127.0.0.1:5000/api/v4
```

## 9. What each node type holds

| node type | repository files | attributes and extras worth knowing |
| --- | --- | --- |
| `PinoSimulationData` | none in the deposit (evicted; `aiida-pino restore-payloads` puts back `payload.npz` and `_time_values_s.npy`, §4) | `laser_power` (W), `V_scan` (m/s), `beam_radius` (m), `spatial_shape`, `num_timesteps`, `batch_keyword`, `simulation_metadata.flow3d_case_manifest` (the full case manifest), `array_shapes`, `requested_fields`; extras `split`, `material_name`, `material_file`, `cell_size_m`, `wall_seconds`, `cores`, `normal_completion`, `meltpool_stats` |
| `PinoDatasetCollectionData` | none | `member_simulation_uuids`, `derivation_profile` (stride, time window, frame handling the models saw), `dataset_role`, `base_collection_uuid`, `simulation_count`; extras `folder`, `superseded` |
| `PinoModelData` | `u_network.0.pth` (weights), `.hydra/config.yaml` (the resolved training configuration), `.hydra/overrides.yaml`, `best_checkpoint.json`, `validators/validation_metrics.csv` (the training curve), `resource_usage.txt`, `trained_grid.json`; `u_network.best.pth` on one model | `model_metadata` (`config_name`, `run_name`, `training_profile`, `training_summary`, `training_source_uuid`); extras `train_dataset_uuid`, `training_seed`, `steps_trained`, `gpu_vram_peak_mb`, `host_ram_peak_mb`, `ms_per_iter`, `deposit_trimmed` |
| `PinoModelEvaluationData` | none | `evaluation_metadata` (`model_uuid`, `dataset_source_uuid`, `metrics.{T,alpha,fl,geometry}`, `metric_units.kelvin`, `served_grid`, `eval_stride`, `training_stride`, `scored_n`, `evaluation_profile`) |
| process nodes | the adapters each job uploaded, its `aiida.out` | `process_type` names the workflow (`pino.train_model`, `pino.evaluate_model`, `pino.flow3d_generate`, ...); `aiida-pino resubmit` re-runs a training workflow (§7) |
