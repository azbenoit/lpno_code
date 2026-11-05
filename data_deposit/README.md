# Data deposit

The documents shipped with the two Zenodo records, and the scripts around them.

| path | contents |
| --- | --- |
| `scripts/download.py` | downloads either record into `open_record/` or `restricted_record/` here; the one script most readers need |
| `record_docs/` | the README, file descriptions and licence of each record |
| `provenance_logs/` | one row per evaluation behind a printed number, with every UUID |
| `restricted_file_lists/` | one row per file moved from the AiiDA archives to the restricted record |
| `scripts/sync.py`, `scripts/split_restricted.py`, `scripts/make_provenance_log.py` | the authors' tools that built the records, kept as a record of how the deposit was made |
