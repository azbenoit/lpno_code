# Data deposit for "LP-NO" (laser melt-pool neural operators): FLOW-3D WELD output, restricted access

This record holds the FLOW-3D WELD simulation output behind the manuscript
A. Benoit and E. Iseli, *Provenance-tracked, experimentally validated neural operator surrogates of
three-dimensional laser melting with keyholes and free-form paths* (2026; preprint: **[TO FILL: arXiv link]**): the simulated fields of the 1,096 simulations its models were
trained and scored on, and the files carrying solver output that were moved
out of the AiiDA archives of the companion open record. The open record
doi:10.5281/zenodo.23325086 holds the provenance graph, the trained models, every
evaluation and the configuration of every simulation; its `README.md`
describes the whole deposit.

Code repository: https://github.com/azbenoit/lpno_code, release **[TO FILL: tag]**.
This record: doi:10.5281/zenodo.23325088. Licence: CC BY-NC-SA 4.0 (see `LICENSE.txt`).

## Access

Under the terms agreed with Flow Science, the vendor of FLOW-3D WELD, the
solver output is shared for non-commercial use only, under an access
agreement. Request access on this page, with your name, affiliation and
intended use, and confirm the conditions below. The record owner reviews each
request.

Conditions of access:

1. The data are used for non-commercial research or teaching only. Using them
   to develop, train or validate a commercial product or service is not
   permitted.
2. The data, and anything adapted from them, are shared only under CC BY-NC-SA
   4.0 and these conditions, and not made openly downloadable.
3. Work that uses the data cites the manuscript and both records.

## What is in the record

| file | size | contents |
| --- | ---: | --- |
| `payloads_<family>.tar` (24 tars), stored as `payloads_<family>.tar.part00`, `.part01`, ... when larger than 2 GiB | 156.8 GB total | one compressed NPZ and one JSON manifest per simulation, named by the simulation's AiiDA UUID |
| `payload_manifest.csv` | small | one row per simulation: uuid, family, split, case_name, bytes, sha256 |
| `restricted_files_main.tar`, `restricted_files_rounds.tar` | 0.54 GB, 0.46 GB | the files moved out of `lpno_provenance.aiida` and `lpno_provenance_rounds.aiida`, one member `objects/<sha256>` per distinct file |
| `restricted_files_evals.tar` | 0.12 GB | the files moved out of `lpno_provenance_evals.aiida` (evaluation previews and frame dumps) that no other file of this record already holds, one member `objects/<sha256>` each |
| `restricted_files_main.csv`, `restricted_files_rounds.csv`, `restricted_files_evals.csv` | small | one row per moved file: node UUID, node type, process, path in the node, object sha256, bytes, reason; `restricted_files_evals.csv` also names, in `held_in`, the file of this record that holds each object (most are simulation payloads already in a `payloads_<family>.tar`) |
| `lpno_web_demo_ground_truth.tar`, `lpno_web_demo_ground_truth.tar.sha256` | 0.94 GB | the optional ground truth of the code repository's web demo, held-out cases only |
| `files_description.md`, `SHA256SUMS`, `LICENSE.txt` | | per-file descriptions of both records, checksums of every file here, the CC BY-NC-SA 4.0 licence |

## Rejoining the payload tars

Zenodo does not reliably accept single uploads of several GB, so the 20 tars
larger than 2 GiB are stored as numbered 2 GiB parts (80 parts in all; the 4
smaller tars are stored whole). Rejoin each tar and check it before use:

```bash
for f in payloads_*.tar.part00; do t=${f%.part00}; cat "$t".part?? > "$t"; done
sha256sum -c SHA256SUMS --ignore-missing
```

`SHA256SUMS` lists both the parts and the rejoined tars. The parts can be
deleted once every tar checks out.

## Using it

Put the payload tarballs of this record, or links to them, in the directory that
holds the open record. The tools described in the open record's `README.md` and
`README_AiiDA.md` then find both halves: `aiida-pino restore-payloads --deposit <dir>` puts the arrays
back into an imported profile, and `reproducibility/rerun/export_deposit.py` in the code
repository writes training and evaluation inputs without AiiDA. The member
layout of every NPZ is in the open record's `README.md`, "Reading the arrays
without AiiDA".

The moved files are the evaluation previews and dumps
(`lazy_preview_batch<n>.npz`, `eval_frames_dump.npz`) and the training
snapshots (`constraints/data-<n>.vti`), which store the solver's fields next
to the model's predictions, and the solver's console logs (`runhyd_tail.log`,
and `aiida.out` of the solver jobs). The logs carry the same site-name
replacements as the archives (`/home/user`, `workstation`). To read the files
of one node:

```python
import csv, io, tarfile
import numpy as np

rows = [r for r in csv.DictReader(open("restricted_files_main.csv")) if r["node_uuid"] == "<node uuid>"]
with tarfile.open("restricted_files_main.tar") as t:
    for r in rows:
        data = t.extractfile(f"objects/{r['object_sha256']}").read()
        if r["path_in_node"].endswith(".npz"):
            z = np.load(io.BytesIO(data))      # pred_T, true_T, pred_alpha, true_alpha, ...
```

## Citation

```bibtex
@misc{benoit_lpno_2026,
  author = {Benoit, Alix and Iseli, Elia},
  title  = {Provenance-tracked, experimentally validated neural operator surrogates of
            three-dimensional laser melting with keyholes and free-form paths},
  year   = {2026},
  note   = {Preprint, arXiv:[TO FILL]}
}
```

**[TO FILL: BibTeX for both records with their DOIs.]**
