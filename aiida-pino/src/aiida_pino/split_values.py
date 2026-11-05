"""The split vocabulary, kept importable without pulling in AiiDA.

`aiida_pino.utils.__init__` imports `data_nodes`, which imports `aiida`, so
anything reaching these constants through `aiida_pino.utils.annotations` drags
the whole framework in. The CLI builds its argument parser at import time and
must answer `--help` without that (`tests/test_cli.py::
test_help_runs_without_heavy_imports`), so the tuples live here, in a leaf
module, and `utils.annotations` re-exports them. One source of truth, reachable
from both sides.
"""

from __future__ import annotations

#: `eval` and `duplicate_excluded` were already live in production before they
#: were sanctioned here (4 and 5 simulations on the authors' database), written by ad-hoc
#: `extras.set` scripts because this tuple rejected them — which is how the
#: deliberate, correct exclusion of 38427's five centre-point replicas ended up
#: with no record of who did it or why. `excluded` is the value new work should
#: use; `duplicate_excluded` stays writable so the existing five can be restated
#: rather than migrated.
SPLIT_VALUES = (
    "train",
    "val",
    "test",
    "eval",
    "excluded",
    "duplicate_excluded",
    "unassigned",
)

#: Values that mean "held out" — present in an evaluation-only set, never in
#: training data.
HELD_OUT_SPLIT_VALUES = ("test", "eval")

#: Values that mean "this case is deliberately in no split at all".
EXCLUDED_SPLIT_VALUES = ("excluded", "duplicate_excluded")
