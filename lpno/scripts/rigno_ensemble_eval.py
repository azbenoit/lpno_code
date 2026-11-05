"""Ensemble evaluation for RIGNO (or any grid-model) test runs.

Averages the per-model saved predictions (from each run's `fno_eval/`) and scores
them with the pipeline's own metric functions, so the numbers are directly
comparable to a single-model eval. fl is re-derived from the averaged fl fields
(re-deriving fl from averaged *masked* T is invalid — the boiling-temperature fill
outside the melt region corrupts the ramp).

Usage:
    python scripts/rigno_ensemble_eval.py OUTPUTS_DIR RUN_GLOB [RUN_GLOB ...]
      OUTPUTS_DIR : the Hydra outputs root (e.g. ./outputs)
      RUN_GLOB    : glob of an eval run dir under OUTPUTS_DIR (its fno_eval/ holds
                    predictions.npz + ground_truth.npz), e.g. 'rigno_E_test_*'
"""
import glob
import sys

import numpy as np

from lpno.models.pipeline.postprocessing import _compute_error_stats


def _load(outputs_dir: str, run_glob: str):
    matches = sorted(glob.glob(f"{outputs_dir}/{run_glob}/fno_eval"))
    if not matches:
        raise FileNotFoundError(f"no fno_eval dir for {outputs_dir}/{run_glob}")
    d = matches[-1]
    return np.load(f"{d}/predictions.npz"), np.load(f"{d}/ground_truth.npz")


def ensemble_metrics(outputs_dir: str, run_globs: list[str]) -> dict:
    """Average the member predictions and score them with the pipeline metric."""
    preds_list, truth = [], None
    for rg in run_globs:
        p, g = _load(outputs_dir, rg)
        preds_list.append({k: p[k] for k in p.files})
        truth = {k: g[k] for k in g.files}
    ens = {k: np.mean([p[k] for p in preds_list], axis=0) for k in ("T", "alpha", "fl")}
    return _compute_error_stats(ens, truth, ["T", "alpha", "fl"])


def main(argv: list[str]) -> None:
    if len(argv) < 3:
        sys.exit(__doc__)
    outputs_dir, run_globs = argv[1], argv[2:]
    err = ensemble_metrics(outputs_dir, run_globs)
    print(f"=== ENSEMBLE of {run_globs} ===")
    for k in ("T", "alpha", "fl"):
        s = err[k]
        iou = f"  IoU {s['iou_mean']:.5f}" if "iou_mean" in s else ""
        # fl/alpha carry no relative metric: they are already fractions in [0,1].
        rel = f"rel_err {s['rel_err_mean']:.5f}  " if "rel_err_mean" in s else ""
        print(f"{k:5s}: {rel}mae {s['abs_err_mean']:.6f}  rmse {s['rmse']:.6f}{iou}")


if __name__ == "__main__":
    main(sys.argv)
