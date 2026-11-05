# Tools

## `setup_env.py`: one pinned environment

Builds a Python 3.10 environment with `lpno` (and, with `--with-aiida`, aiida-core 2.8.0 and `aiida-pino`) at the
versions the published models were trained and scored with: torch 2.8.0 from the PyTorch index of the chosen build,
every dependency held to the pins of `lpno/environment.yml`, then `torch-scatter` and `nvidia-physicsnemo-sym`, and
an import of everything at the end.

```bash
python3.10 tools/setup_env.py --prefix ~/envs/lpno --torch cpu --with-aiida   # new venv
python tools/setup_env.py --in-place --torch cu126                             # the active environment
```

`--torch` takes `cpu`, `cu126` (Volta GPUs such as the V100) or `cu128` (Ampere and newer). A CUDA build downloads
6-8 GB of wheels, the CPU build about 3 GB. The web demo has its own environment script,
`self_hosted_web_demo/lpno_web_demo_backend/scripts/setup_env.sh`.

## `check_site_leaks.py` and `install_hooks.sh`: keep your machines out of commits

Running simulations and trainings on a cluster needs host names, user names, accounts and scratch paths. These belong
in a local, git-ignored site file (`reproducibility/rerun/site.env`, see `reproducibility/rerun/site.env.example`), never in tracked files.
`check_site_leaks.py` refuses a commit that would add any value of that file, or text that is site-specific by
construction (a real path under `/home/<name>/` or `/scratch/`, an IPv4 address, a licence endpoint `PORT@HOST`, a
private key). Extra words to refuse go in the site file's `SITE_DENY` list.

```bash
bash tools/install_hooks.sh                   # run the check before every commit
python3 tools/check_site_leaks.py --all       # check every tracked file
python3 tools/check_site_leaks.py --self-test
```

A line that must carry such a pattern on purpose (a documented placeholder) can end with the marker
`site-leak: allow`.
