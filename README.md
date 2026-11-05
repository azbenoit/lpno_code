# LP-NO: neural-operator surrogates for laser melting

Code for the manuscript A. Benoit and E. Iseli, *Provenance-tracked, experimentally validated neural operator
surrogates of three-dimensional laser melting with keyholes and free-form paths* (2026; preprint, arXiv link to follow).

The repository covers the whole pipeline behind the paper. It generates FLOW-3D WELD
simulations of laser melting from JSON configurations, trains and evaluates neural-operator
surrogates on them (the Fourier neural operator, FNO, the region interaction graph neural
operator, RIGNO, and the geometry-aware operator transformer, GAOT), and records every
simulation, dataset, model and evaluation in an AiiDA provenance graph. The trained models
can be explored in a web demo you host yourself.

## Quick start

1. Install one environment for the Python packages (Python 3.10; torch, PhysicsNeMo and
   `torch-scatter` pinned to versions that work together), and the notebook tools:

   ```bash
   python3.10 tools/setup_env.py --prefix <env dir> --torch cpu|cu126|cu128 --with-aiida
   source <env dir>/bin/activate
   pip install jupyter pandas pyvista
   ```

2. Train and evaluate a small model, with nothing to download:
   [`reproducibility/train_evaluate.ipynb`](reproducibility/train_evaluate.ipynb) generates twelve
   synthetic cases and walks through training and evaluation, with and without AiiDA, in about
   11 minutes on a CPU.

3. Download the data from Zenodo (see [Data](#data) below; about 18 GB for the open record, about
   157 GB for the restricted one):

   ```bash
   python3 data_deposit/scripts/download.py open        # models, evaluations, configurations
   python3 data_deposit/scripts/download.py restricted --token-file <file> --join   # simulated fields, on request
   ```

4. Reproduce the paper: the notebooks in [`reproducibility/`](reproducibility/README.md)
   rebuild the results tables (`tables.ipynb`) and every figure (`figures.ipynb`) from the
   downloaded records.

5. Try the demo: [`self_hosted_web_demo/`](self_hosted_web_demo/README.md) runs on your
   own machine (a GPU is recommended) with the weights from the open record.

## Repository layout

| directory | contents |
|---|---|
| [`flow3D_scripts/`](flow3D_scripts/README.md) | Programmatic use of FLOW-3D WELD: a JSON configuration becomes a complete solver input deck, the solve is launched through the vendor's launchers, and every stored time step is exported to one compact NPZ per simulation. Also the batch generator, the task spooler and the validation of the solver settings against published measurements (`validation/`). It needs an installed FLOW-3D WELD and an active licence of your own; it automates the solver but does not replace it. |
| [`lpno/`](lpno/README.md) | The machine-learning package: FNO, RIGNO and GAOT on NVIDIA PhysicsNeMo and PyTorch with Hydra configurations, the NPZ data loading and conditioning channels, the training and evaluation pipeline and its metrics. |
| [`aiida-pino/`](aiida-pino/README.md) | An AiiDA plugin that runs the two above as provenance-tracked jobs: simulation ingest, datasets and splits, training and evaluation workflows, the `aiida-pino` command line and a browser console. |
| [`reproducibility/`](reproducibility/README.md) | The notebooks that rebuild the paper's tables and figures, the figure generators (`reproducibility/figures/`), the scripts that re-run the paper's trainings and evaluations from the deposit without an AiiDA installation, on one GPU or a Slurm cluster ([`reproducibility/rerun/`](reproducibility/rerun/README.md)), and the published measurements the paper compares against (`experimental_data/`), with their sources and licences. |
| [`data_deposit/`](data_deposit/README.md) | The README and file lists shipped with the two Zenodo records, the provenance logs that map every printed number to its evaluation, and the download script. |
| [`self_hosted_web_demo/`](self_hosted_web_demo/README.md) | The web demo: a FastAPI backend serving model predictions as images, and a static browser front end. |
| `tools/` | `setup_env.py` (the environment above), and `check_site_leaks.py` with `install_hooks.sh`, a pre-commit check that keeps host names, user names and paths of your own machines out of commits. |

The three packages (`flow3D_scripts/`, `lpno/`, `aiida-pino/`) each have their own README, an `assumptions.md` that records the modelling and
implementation choices with their effect and source location, a `potential_bugs.md` that lists
known issues and traps (symptom, cause, workaround, where in the code), and a `docs/` folder with
developer notes. Some code in these directories goes beyond what the paper discusses; each
README says which parts.

## Data

The data are in two Zenodo records, both under CC BY-NC-SA 4.0.

- **Open record**, [doi:10.5281/zenodo.23325086](https://doi.org/10.5281/zenodo.23325086):
  the AiiDA archives with every trained model behind the paper (weights included), every
  evaluation with its metrics, and the configuration of every simulation; the provenance logs;
  the published measurements as CSV files; and the weights of the web demo.
- **Restricted-access record**, [doi:10.5281/zenodo.23325088](https://doi.org/10.5281/zenodo.23325088):
  the FLOW-3D WELD output, i.e. the simulated fields of every simulation (about 157 GB) and the
  solver logs, and the ground truth of the web demo. These are solver outputs, which are
  distributed for non-commercial research through this record only. To request access, use
  the request button on the record's Zenodo page, or write to
  alix.benoit@alumni.polytechnique.org with your name, affiliation and intended use.

`data_deposit/scripts/download.py` downloads either record into `data_deposit/`, where the
notebooks and scripts look by default. Each record's README (in
[`data_deposit/record_docs/`](data_deposit/record_docs/)) lists its files and how to open them.

### Using simulations from another solver

Nothing downstream of the export depends on FLOW-3D itself. The surrogates read one NPZ per
simulation (cell-centre axes, stored times and the field arrays on a structured grid, as in
[`flow3D_scripts/README.md`, "NPZ layout"](flow3D_scripts/README.md)) and a case manifest
with the laser parameters and path. Output from any other solver can be used once it is
written in that layout; the field names a model reads are set in its Hydra configuration.

## Web demo

The demo shows the trained models in a browser: the quasi-steady melt pool at any laser power
and scan speed, a straight track in time, and a path you draw. It runs on your own machine, so a GPU is recommended for a fluid experience. The model weights come from the open record. The
FLOW-3D ground truth that the pages can compare against is optional and comes from the
restricted record.

## Provenance console

`aiida-pino` includes a browser console for the AiiDA database: monitoring running
simulations, trainings and evaluations, browsing and cleaning up datasets, and following a
model or an evaluation back through its provenance. It is in beta: useful for monitoring
and navigation, but may contain slight bugs and misbehaving or unoptimized features. The [`aiida-pino` README](aiida-pino/README.md) and
`aiida-pino/web/docs/SETUP.md` describe the setup.

## Citation

If you use this code or the data, please cite the paper:

```bibtex
@misc{benoit_lpno_2026,
  author = {Benoit, Alix and Iseli, Elia},
  title  = {Provenance-tracked, experimentally validated neural operator surrogates of
            three-dimensional laser melting with keyholes and free-form paths},
  year   = {2026},
  note   = {Preprint}
}
```

and the data records ([doi:10.5281/zenodo.23325086](https://doi.org/10.5281/zenodo.23325086),
[doi:10.5281/zenodo.23325088](https://doi.org/10.5281/zenodo.23325088)).

## Acknowledgments

This work builds on open-source software, in particular:

- [AiiDA](https://www.aiida.net), the workflow and provenance framework behind `aiida-pino`;
- [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo), on which the models are trained;
- [RIGNO](https://github.com/camlab-ethz/rigno) (arXiv:2501.19205), ported to PyTorch in
  `lpno/src/lpno/models/rigno_arch.py`;
- [GAOT](https://github.com/camlab-ethz/GAOT) (arXiv:2505.18781) and
  [GAOT-3D](https://github.com/Shizheng-Wen/GAOT-3D), whose layers are vendored in
  `lpno/src/lpno/models/gaot_layers/`.

The licences of the third-party code are listed in `NOTICE`.

FLOW-3D and FLOW-3D WELD are products of Flow Science, Inc. This repository contains no part of
them.

## Licence

The code is licensed under the PolyForm Noncommercial License 1.0.0 (`LICENSE`, copyright
notice in `NOTICE`); the same licence applies to every directory. Any noncommercial use is
permitted, including by universities and public research organizations. The data records are
licensed under CC BY-NC-SA 4.0 ([`data_deposit/record_docs/LICENSE_data.txt`](data_deposit/record_docs/LICENSE_data.txt)):
use and adaptation for non-commercial purposes, with attribution, and adaptations under the
same licence. Third-party components keep their own notices in place.

## Reporting problems

If you find a bug or another problem with the code, please open an issue at
<https://github.com/azbenoit/lpno_code/issues> or email
alix.benoit@alumni.polytechnique.org.
