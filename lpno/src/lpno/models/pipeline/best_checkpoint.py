"""Keep the checkpoint with the best validation T MAE, never overwritten.

physicsnemo writes `u_network.0.pth` on a fixed schedule and overwrites it every time,
so a run that trains well and then diverges ends with the diverged weights and the good
ones are gone. That is not hypothetical here: `gaot_cond_32k_ctrl` (model 26734) reached
val T MAE 22.4 K at step 12000, spiked at 14000, and from step 16000 diverged monotonically
to 3.8e8 K by step 28000, with alpha collapsing to a constant. The registered model was the
wreckage; the good weights from step 12000 had been overwritten sixteen times. The whole
run -- ~13 GPU hours -- was unsalvageable for want of one extra file.

This keeper snapshots the LIVE model whenever validation improves, so:

  * `u_network.best.pth`      -- weights at the best validation T MAE
  * `best_checkpoint.json`    -- the step, the metric, and which key was tracked

It saves from the in-memory model rather than copying `u_network.0.pth`, because the
checkpoint on disk at validation time belongs to the last *save* step, which need not be
the last *validation* step -- copying it would silently store the wrong weights.

Hooked in the same place and style as `amp_collapse_guard` (BUG-43): patch the class the
solver looks up, since there is no instance to reach after the fact. The repo already
patches `_l2_relative_error` on these same validator classes to add `mae_*`/`rmse_*`, which
is what puts the tracked metric in the returned dict.

Off is not an option worth having -- the cost is one file write per improvement -- but
`custom.best_checkpoint=false` disables it for a run that wants byte-identical output.
"""
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Preference order. `beambox_mae_T` first (2026-08-13): domain mae_T
# cannot separate a cold run from a healthy one -- on gaot_pvlhs_base_32k it kept
# the step-26000 checkpoint (mae 0.0345, tail_deficit 0.368, no hot tail) over
# step 22000 (mae 0.0604, tail_deficit 0.019, healthy tail), because the ambient
# far field dominates the domain average. The beam box is where the melt physics
# lives, so its MAE ranks the healthy basin above the cold one. The others are
# fallbacks for validators that do not expose the earlier keys.
_METRIC_PREFERENCE = ("beambox_mae_T", "mae_T", "rmse_T", "l2_relative_error_T")

BEST_WEIGHTS_NAME = "u_network.best.pth"
BEST_META_NAME = "best_checkpoint.json"


def _pick_metric(losses: dict) -> tuple[str, float] | None:
    """First available tracked metric, as (key, value). None if none are present."""
    if not isinstance(losses, dict):
        return None
    for key in _METRIC_PREFERENCE:
        value = losses.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        # A diverged run reports inf/nan; those must never count as an improvement.
        if value != value or value in (float("inf"), float("-inf")):
            continue
        return key, value
    return None


@contextlib.contextmanager
def best_val_checkpoint(cfg: Any, model: Any, network_dir: str | Path):
    """Snapshot `model` whenever validation T MAE improves, for the enclosed block."""
    enabled = True
    try:
        from omegaconf import OmegaConf

        configured = OmegaConf.select(cfg, "custom.best_checkpoint", default=None)
        if configured is not None:
            enabled = bool(configured)
    except Exception:  # pragma: no cover - config shape is not worth failing a run over
        pass

    if not enabled or model is None:
        yield
        return

    import torch
    from physicsnemo.sym.domain.validator import GridValidator, PointwiseValidator

    out_dir = Path(network_dir)
    state: dict[str, Any] = {"best": None, "key": None, "step": None}
    originals: list[tuple[type, Any]] = []

    def _make(original):
        def save_results(self, name, results_dir, writer, save_filetypes, step):
            losses = original(self, name, results_dir, writer, save_filetypes, step)
            picked = _pick_metric(losses)
            if picked is None:
                if not state.get("warned_no_metric"):
                    state["warned_no_metric"] = True
                    logger.warning(
                        "best-checkpoint keeper is armed but none of %s appear in the "
                        "validation results (keys: %s) -- no best checkpoint will be kept",
                        _METRIC_PREFERENCE, sorted(losses) if isinstance(losses, dict) else type(losses),
                    )
                return losses
            key, value = picked
            if key != _METRIC_PREFERENCE[0] and not state.get("warned_fallback"):
                # The silent half of the same failure the no-metric warning below covers.
                # `_pick_metric` takes the first AVAILABLE preference, so a run whose
                # validator cannot produce `beambox_mae_T` selects on whole-domain `mae_T`
                # -- the bulk-diluted metric that rewards a cold field -- while sibling arms
                # select on the melt region, and nothing said so. GAOT AR k=8 (model 86604)
                # was ranked that way against arms that were not (session 067cdc63,
                # 2026-09-02). Its `best_checkpoint.json` records the metric, so the fact
                # was on disk; what was missing is anyone being told.
                state["warned_fallback"] = True
                logger.warning(
                    "checkpoint selection is using %r, not %r: the preferred metric is not "
                    "among this validator's results (have: %s). This arm is NOT ranked on "
                    "the same criterion as one that reports %s -- do not compare their "
                    "best checkpoints without saying so. best_checkpoint.json records the "
                    "metric actually used.",
                    key, _METRIC_PREFERENCE[0],
                    ", ".join(sorted(k for k in losses if isinstance(losses, dict))) or "none",
                    _METRIC_PREFERENCE[0],
                )
            # The CURRENT reading, every validation -- not only when it improves. For a
            # REMOTE job the log tail is the ONLY live channel: validation_metrics.csv
            # stays on the compute box until retrieval, and the console scrapes this exact
            # line shape (`<metric>=<value> at step <n>`) from a 16 KB tail. Logging solely
            # on improvement means a plateaued run shows no beam-box number at all, which
            # is precisely when one wants to look: 82438 (gaot_crossds_ph_w2_90k) last
            # improved at step 10000, and by step 29800 that line sat 34969 bytes back in a
            # 61274-byte log -- two tail windows out of reach -- so the monitor showed
            # nothing while the CSV carried beambox_mae_T at every step from 2000 on.
            # It also makes `last_beambox_mae` honest: it documents returning the LAST
            # reading, but a best-only line makes the last reading a stale best.
            logger.info("validation: %s=%.6g at step %d", key, value, step)
            if state["best"] is not None and not (value < state["best"]):
                return losses
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out_dir / BEST_WEIGHTS_NAME)
                state.update(best=value, key=key, step=int(step))
                (out_dir / BEST_META_NAME).write_text(
                    json.dumps(
                        {"metric": key, "value": value, "step": int(step),
                         "weights": BEST_WEIGHTS_NAME},
                        indent=1,
                    )
                    + "\n"
                )
                logger.info("best checkpoint: %s=%.6g at step %d", key, value, step)
            except Exception as exc:  # never let bookkeeping kill a training run
                logger.warning("best-checkpoint save failed at step %s: %s", step, exc)
            return losses

        return save_results

    # Base classes AND every subclass that overrides save_results in its own
    # __dict__. Patching only the bases is how this keeper silently did nothing
    # for every grid run: solve_domain builds StreamingGridValidator (A35),
    # whose own save_results shadows the patched base method, so neither
    # rigno2579 fp32 arm ever wrote u_network.best.pth (2026-08-10). A subclass
    # that does NOT override save_results is covered by its patched base --
    # patching it too would double-wrap.
    def _walk(cls):
        yield cls
        for sub in cls.__subclasses__():
            yield from _walk(sub)

    seen: set[type] = set()
    for base in (GridValidator, PointwiseValidator):
        for cls in _walk(base):
            if cls in seen or "save_results" not in cls.__dict__:
                continue
            seen.add(cls)
            originals.append((cls, cls.__dict__["save_results"]))
            wrapper = _make(cls.__dict__["save_results"])
            wrapper._best_checkpoint_patched = True
            cls.save_results = wrapper

    try:
        yield
    finally:
        for cls, original in originals:
            cls.save_results = original
        if state["best"] is not None:
            logger.info(
                "kept best checkpoint: %s=%.6g at step %s -> %s",
                state["key"], state["best"], state["step"], out_dir / BEST_WEIGHTS_NAME,
            )
