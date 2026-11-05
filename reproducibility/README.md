# Reproducibility

Three notebooks rebuild the manuscript's results from the public store, the two Zenodo records of the deposit:

* the **open record**, doi:10.5281/zenodo.23325086: the AiiDA archives (every trained model with its weights, every
  evaluation with its metrics, every simulation's configuration and the provenance graph that links them), the
  provenance logs and `payload_manifest.csv`;
* the **restricted record**, doi:10.5281/zenodo.23325088: the FLOW-3D WELD output (the simulated fields), available on
  request under the terms stated there.

| notebook | reproduces | needs |
| --- | --- | --- |
| `tables.ipynb` | every printed cell of the model-evaluation tables (`tab:arch-steady`, `tab:paths-accuracy`, `tab:preheat`, `tab:twoalloy`, `tab:frame-difficulty`), the cost table (`tab:cost`) and the dataset table (`tab:datasets`) | the open record; no GPU |
| `figures.ipynb` | all 81 figure files of the manuscript (22 figures), compared with the published PDFs | committed data for 16 figures; the restricted record for the ground truth of the other five |
| `train_evaluate.ipynb` | how to train and evaluate a model, with and without AiiDA, and how to choose the architecture and hyperparameters; a synthetic dataset runs it on a CPU, and an optional section re-runs a deposited evaluation on the real data | CPU; the restricted record for the optional section |

[`rerun/`](rerun/README.md) re-runs the paper's trainings and evaluations from the deposit without an AiiDA installation, on one local GPU or a Slurm cluster.

## Setting up

From the repository root:

```bash
python3.10 tools/setup_env.py --prefix ../lpno-env --torch cpu --with-aiida
source ../lpno-env/bin/activate
pip install jupyter pandas pyvista                     # notebooks, tables, 3-D renders
python3 data_deposit/scripts/download.py open          # -> data_deposit/open_record/
python3 data_deposit/scripts/download.py restricted --token-file <file> --join   # optional, on request
export PINO_ALLOW_CPU=1
cd reproducibility && jupyter lab
```

The open record holds `lpno_provenance.aiida`, `lpno_provenance_rounds.aiida`, `lpno_provenance_evals.aiida` and
`payload_manifest.csv` (`--only lpno_provenance payload_manifest.csv provenance_log` skips the demo weights). The
tables notebook opens the archives read-only and needs no AiiDA profile. The figures that show FLOW-3D ground truth
need the restricted record; `--join` rejoins each payload tar from its 2 GiB parts and checks it against the record's
`SHA256SUMS`. The notebooks look for the records in `data_deposit/open_record/` and
`data_deposit/restricted_record/`; set `LPNO_DEPOSIT` and `LPNO_RESTRICTED` to use copies elsewhere.

## What is here

* `train_evaluate.ipynb` and `synthetic_data.py`: the tutorial and the twelve synthetic straight-track cases it
  trains on. It runs in about 11 minutes on four CPU threads and works in `train_evaluate_work/` (git-ignored). It
  needs no GPU: with `PINO_ALLOW_CPU=1` the pipeline runs on the CPU, a CPU-only torch build included, and RIGNO
  runs without `torch-scatter`.
* `lpno_repro.py`: the helpers the notebooks share. It opens the archives read-only, defines the metric conventions of
  every table and formats cells the way the manuscript prints them.
* `data/table_cells.csv`: every printed metric cell, with the evaluation UUIDs behind it, how they combine (a single
  evaluation, or the mean and sample standard deviation over three training seeds) and the value as printed.
* `data/figure_files.csv`: every figure file the manuscript includes, with its figure label.
* `experimental_data/`: the published measurements the manuscript compares against, as clean CSV files, with full
  citations, DOIs, licences and credits in its README.
* The figure generators themselves are in `figures/`, with their data.

