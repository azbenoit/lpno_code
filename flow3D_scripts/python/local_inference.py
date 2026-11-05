"""Run a trained PINO/FNO surrogate locally, outside an AiiDA eval.

Two things an eval cannot do, both needed by the experiment comparison:

1. **Score a window an eval will not take.** The steady derivation averages the LAST
   `time_avg_window` frames of an export. For the 15 anchor-condition replicates the
   laser stays on past the declared path end, so that window is an overrun regime no
   training case occupies. `derivation_profile.time_frames` cannot express "the frames
   BEFORE the path end" -- it is a subsample COUNT that always keeps the last frame
   (`aiida-pino/.../dataset_derivation.py:421-427`).
2. **Predict where there is no simulation at all.** Measured experimental conditions
   sit inside the training envelope with no FLOW-3D solve behind them, so there is no
   dataset for an eval to consume.

Nothing here reimplements the pipeline. The channels come from the model's own archived
Hydra config, the preprocessing is the shipped `_read_processed_frame`, and the store is
the shipped `ArrayDirCollectionStore`, so a change in the ML repo reaches this file
instead of silently diverging from it.

**Do not route this through the website inference adapter.** `adapter_pino.py` ends its
if/elif over `input_keys` with `else: fill zeros` and only warns, which is 5 of this
model's 11 channels; and it reads the H_norm denominator from `ref_scales["H"]` (203),
which is not this model's denominator (425.87 -- see `h_norm_denom_from_store`).

Validated against the eval it must reproduce: on the 15 replicates of eval 86786, this
path returns the eval's own prediction to a mean |dT| of 2.49 K in metal, 0.40 K in gas,
and a mean |d alpha| of 0.000647. That is the tolerance this comparison has always
carried (the notebook records the earlier pipeline at |d alpha| = 0.001), and it sits
well inside the model's own 14.94 K substrate error. `validate_against_dump` re-measures
it so a regression shows up as a number rather than as a quiet drift.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

#: Where the sibling `lpno` checkout lives. `PINO_SIMULATIONS_ROOT`
#: overrides; otherwise the walk-up below finds `../lpno` from the
#: repository root, including when this file runs from a git worktree nested
#: two levels inside the repository, where the sibling is three levels further
#: up and the obvious `parents[N]` form points at a directory that does not exist.
#: If the package is installed (`pip install -e ../lpno`, as it is in
#: the `pino` env) nothing here is needed and the import simply succeeds.
PINO_ROOT_ENV = "PINO_SIMULATIONS_ROOT"

# `alpha_masking` fills T with the vaporisation temperature wherever alpha says gas. The
# eval applies it to the PREDICTION as well as the target, so a raw network output
# compared against an eval dump is wrong by ~575 K on average -- the whole gas
# half-space. Reproducing an eval means masking the prediction the same way.
ALPHA_GATE = 0.5


@dataclass
class LoadedModel:
    model: Any
    run_cfg: Any
    input_keys: list[str]
    output_keys: list[str]
    ref_t: float
    device: str


def lpno_root() -> pathlib.Path:
    """The sibling `lpno` checkout: env override, else the walk-up."""
    override = os.environ.get(PINO_ROOT_ENV)
    if override:
        return pathlib.Path(override).expanduser().resolve()
    here = pathlib.Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        cand = base.parent / "lpno"
        if (cand / "src" / "lpno").is_dir():
            return cand
    return here.parents[1].parent / "lpno"


def _ensure_pino_on_path() -> None:
    """Make `lpno` importable when it is not installed.

    An installed package wins: this only adds the sibling's `src/` to `sys.path`
    when the import would otherwise fail, so a `pip install -e` is never shadowed
    by a different checkout found on disk.
    """
    try:
        import lpno  # noqa: F401
        return
    except ImportError:
        pass
    src = lpno_root() / "src"
    if not (src / "lpno").is_dir():
        raise ImportError(
            "lpno is not installed and no sibling checkout was found. "
            f"Either `pip install -e ../lpno` or set {PINO_ROOT_ENV} "
            f"to the checkout (looked for {src})."
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _quiet(fn, *args, **kwargs):
    """Call `fn` with stdout swallowed; the pipeline is chatty at import and load."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def h_norm_denom_from_store(train_cache: str | pathlib.Path) -> float:
    """max(P / sqrt(|V|)) over a staged TRAIN collection, radius-blind.

    The same quantity `lazy_runtime.compute_h_norm_denom` fits, recomputed from the
    staged arrays so it can be checked rather than trusted. Radius-blind because
    `custom.h_norm_include_radius` defaults to False and none of these models set it;
    including the radius on one side only is the asymmetry that put H_norm = 2.5e6 into
    a whole campaign (see `utils.h_norm_raw`).
    """
    root = pathlib.Path(train_cache) / "members"
    best = 0.0
    for member in sorted(root.iterdir()):
        p_file, v_file = member / "P.npy", member / "V_scan.npy"
        if not (p_file.is_file() and v_file.is_file()):
            continue
        power = float(np.load(p_file, mmap_mode="r").flat[0])
        speed = abs(float(np.load(v_file, mmap_mode="r").flat[0]))
        if speed <= 0.0:
            continue
        best = max(best, power / math.sqrt(speed))
    if best <= 0.0:
        raise ValueError(f"no usable (P, V_scan) pairs under {root}")
    return best


def load_model(model_dir: str | pathlib.Path, *, checkpoint: str = "u_network.best.pth",
               device: str | None = None) -> LoadedModel:
    """Rebuild a model from its archived Hydra config and load its checkpoint STRICTLY.

    Strict on purpose. A tolerated `load_state_dict` mismatch is not a warning, it is a
    silently wrong model: anything in `missing` keeps its random init and the network
    returns confident nonsense. The website adapter raises here for the same reason.
    """
    _ensure_pino_on_path()
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.sym.key import Key
    from physicsnemo.sym.hydra.utils import instantiate_arch
    from lpno.models import register_custom_hydra_inputs
    from lpno.models.pipeline.config_loading import load_cfg_and_paths

    register_custom_hydra_inputs()
    model_dir = pathlib.Path(model_dir)
    cfg = OmegaConf.load(model_dir / ".hydra" / "config.yaml")
    input_keys = list(cfg.custom.FNO_input_keys)
    output_keys = list(cfg.custom.FNO_output_keys)
    run_cfg = _quiet(load_cfg_and_paths, cfg)

    in_k = [Key(k) for k in input_keys]
    out_k = [Key(k) for k in output_keys]
    decoder = instantiate_arch(cfg=cfg.arch.decoder, output_keys=out_k)
    model = instantiate_arch(cfg=cfg.arch.fno, decoder_net=decoder,
                             input_keys=in_k, output_keys=out_k)

    raw = torch.load(model_dir / checkpoint, map_location="cpu")
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state, strict=True)
    model.eval()

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    return LoadedModel(model=model, run_cfg=run_cfg, input_keys=input_keys,
                       output_keys=output_keys, ref_t=float(cfg.custom.ref_scales.T),
                       device=dev)


def open_collection(cache_root: str | pathlib.Path):
    """A staged runtime collection (`aiida_arraydir_v2`) as the shipped lazy store."""
    _ensure_pino_on_path()
    from lpno.models.pipeline.lazy_runtime import ArrayDirCollectionStore
    return ArrayDirCollectionStore(str(cache_root))


def predict_case(loaded: LoadedModel, store, case_idx: int, *, h_norm_denom: float,
                 time_idx: int | None = None) -> dict[str, np.ndarray]:
    """Predict one case. Returns normalised fields plus Kelvin temperatures.

    `pred_T` is alpha-masked exactly as the eval masks it, so it is directly comparable
    to an eval dump. `pred_T_raw` is the unmasked network output, kept because the mask
    destroys the gas-side field and a diagnosis sometimes needs it.
    """
    _ensure_pino_on_path()
    import torch
    from lpno.models.pipeline.preprocessing_datasets import (
        _read_processed_frame, _fill_temperature)
    from lpno.custom_utils.utils import fill_where_mask_smooth_np

    processed = _quiet(_read_processed_frame, store=store, run_cfg=loaded.run_cfg,
                       h_norm_denom=h_norm_denom, case_idx=case_idx, time_idx=time_idx)
    fill_norm = float(_quiet(_fill_temperature, processed, loaded.run_cfg)) / loaded.ref_t

    invar = {}
    for key in loaded.input_keys:
        arr = np.asarray(processed[key], dtype=np.float32)
        invar[key] = torch.as_tensor(arr if arr.ndim == 5 else arr[None], device=loaded.device)

    with torch.no_grad():
        out = loaded.model(invar)
    pred_t_raw = out["T"].cpu().numpy()
    pred_a = out["alpha"].cpu().numpy()
    pred_t = fill_where_mask_smooth_np(pred_t_raw, pred_a, threshold=ALPHA_GATE,
                                       d=fill_norm, channel_axis=1, inplace=False)
    true_t = np.asarray(processed["T"], dtype=np.float32)

    return {
        "pred_T": pred_t,
        "pred_T_raw": pred_t_raw,
        "pred_alpha": pred_a,
        "true_T": true_t,
        "true_alpha": np.asarray(processed["alpha"], dtype=np.float32),
        "pred_T_K": pred_t * loaded.ref_t,
        "true_T_K": true_t * loaded.ref_t,
    }


def validate_against_dump(loaded: LoadedModel, store, dump_path: str | pathlib.Path, *,
                          h_norm_denom: float, cases: Iterable[int] | None = None
                          ) -> dict[str, float]:
    """Re-measure agreement with the eval dump this path is supposed to reproduce.

    Returns metal/gas mean |dT| in Kelvin and mean |d alpha|. Metal is what matters: the
    gas half-space is a constant fill on both sides, so a gas-only difference cannot move
    any melt measurement.
    """
    dump = np.load(dump_path)
    indices = list(cases) if cases is not None else range(dump["pred_T"].shape[0])
    metal, gas, dalpha = [], [], []
    for idx in indices:
        got = predict_case(loaded, store, idx, h_norm_denom=h_norm_denom)
        pred = np.squeeze(got["pred_T"])
        ref = np.squeeze(dump["pred_T"][idx])
        mask = np.squeeze(got["pred_alpha"]) > ALPHA_GATE
        err = np.abs(pred - ref) * loaded.ref_t
        metal.append(float(err[mask].mean()) if mask.any() else 0.0)
        gas.append(float(err[~mask].mean()) if (~mask).any() else 0.0)
        dalpha.append(float(np.abs(np.squeeze(got["pred_alpha"])
                                   - np.squeeze(dump["pred_alpha"][idx])).mean()))
    return {"n_cases": len(metal), "metal_mae_k": float(np.mean(metal)),
            "metal_worst_k": float(np.max(metal)), "gas_mae_k": float(np.mean(gas)),
            "alpha_mae": float(np.mean(dalpha))}


def _demo() -> int:
    """Self-check: reproduce eval 86786 on the 15 anchor replicates.

    Needs the artefacts under `runs/anchorbox_model_validation/` and the staged runtime
    caches, so it runs from the main checkout rather than a worktree.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    work = root / "runs/anchorbox_model_validation"
    # The staged runtime caches this check reads are instance-specific artefacts of
    # the machine that trained model 86651 (the UUIDs name its derived train and eval
    # collections); they do not ship. Point the two variables at your own staged
    # copies, or at any train/eval pair whose members carry the same channels.
    cache_root = pathlib.Path(os.environ.get(
        "AIIDA_PINO_RUNTIME_CACHE", pathlib.Path.home() / ".cache/aiida-pino/runtime"))
    train_cache = os.environ.get("LOCAL_INFERENCE_TRAIN_CACHE",
                                 str(cache_root / "95034053-e13d-4fbf-aae3-569b58fbabe5"))
    eval_cache = os.environ.get("LOCAL_INFERENCE_EVAL_CACHE",
                                str(cache_root / "f00ef8b6-f14d-4bd1-b715-f534c485924c"))

    denom = h_norm_denom_from_store(train_cache)
    assert abs(denom - 425.8694) < 1e-3, f"H_norm denominator drifted: {denom}"
    print(f"H_norm denominator: {denom:.4f}")

    loaded = load_model(work / "models/steady_86651")
    store = open_collection(eval_cache)
    got = validate_against_dump(loaded, store,
                                work / "dumps/steady_86651_edge75_replicates_dump.npz",
                                h_norm_denom=denom)
    print(f"vs eval 86786 over {got['n_cases']} cases: "
          f"metal {got['metal_mae_k']:.3f} K (worst {got['metal_worst_k']:.3f}), "
          f"gas {got['gas_mae_k']:.3f} K, alpha {got['alpha_mae']:.6f}")
    assert got["metal_mae_k"] < 6.0, "metal agreement regressed"
    assert got["alpha_mae"] < 0.005, "alpha agreement regressed"
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
