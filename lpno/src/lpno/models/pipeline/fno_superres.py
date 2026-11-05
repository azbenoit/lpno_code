"""Scoring an FNO on a finer grid than it trained on (strided training, full-resolution evaluation).

An FNO trained on a read-time-strided grid (``PINO_LAZY_STRIDE=2,2,1``: 45x20x40 served from 90x40x40) and scored on
the full grid used to read about six times worse at full resolution than on its own grid (2026-11-01, Table 3 FNO
s43 on validation: 10.3 K substrate MAE on 45x20x40, 57.0 K on 90x40x40, with every input channel identical at the
shared points). Two defects, both repaired here at evaluation time:

1. **Padding is set in cells.** The encoder zero-pads ``pad`` cells per axis before its FFTs, so the padded domain
   -- the period of every Fourier mode -- is ``(n + pad) * dx``. Kept at 9 cells on a twice-finer grid it is shorter
   in physical length, and every mode lands on a different frequency. The padding is scaled by the per-axis grid
   factor (9 -> 18, 18, 9), which keeps the periods the model trained with. ``config_loading`` already does this for
   ``super_res_factor``; it did not for a stride the model trained on.

2. **Modes the trained grid could not hold.** physicsnemo's ``SpectralConv3d`` writes four corner blocks
   (weights1 [:m1, :m2], weights2 [-m1:, :m2], weights3 [:m1, -m2:], weights4 [-m1:, -m2:], in that order). On a
   padded axis shorter than ``2 m`` the low and high slices overlap and the later write wins: y padded to 29 with
   ``m2 = 18`` overlaps on indices 11..17, so weights1/weights2 at y-index 11..17 never received a gradient (checked on
   the checkpoint: 100 % of those entries still in the init range [0, 1/(in*out)), against 1 % elsewhere). On the
   finer grid the blocks separate and those untrained weights act on real signal. The fix rebuilds each block so that
   every fine-grid frequency gets the weight the trained grid actually applied to that same frequency, and zero for a
   frequency the trained grid did not hold -- the trained operator, evaluated on the finer grid.

With both repairs a spectral layer on the finer grid reproduces, at the shared points, what it computes on its trained
grid for a band-limited input (``tests/test_fno_superres.py``). On the Table 3 FNO s43 (validation, 156 frames) the
full-resolution score went from 57.0 K substrate / 286.3 K beam box / IoU 0.400 to 14.3 / 65.7 / 0.825, against
10.3 / 46.4 / 0.877 on its own grid; a feature dump put the encoder output 3.7 % from the trained-grid one at the
shared points, against 28.5 % unrepaired. The pad list of physicsnemo.sym's encoder runs z, y, x (`pad_axis_order`).
On by default for any FNO whose stamped training grid differs from the scored one; ``PINO_FNO_SUPERRES=0`` turns it off.
``PINO_DEBUG_FNO_DUMP=<file.npz>`` saves the first sample's encoder features stage by stage.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

LOW, HIGH = 0, 1
# physicsnemo SpectralConv3d block order and the (x, y) half each block covers; a later block overwrites an earlier
# one wherever their slices overlap.
_BLOCKS = (("weights1", LOW, LOW), ("weights2", HIGH, LOW), ("weights3", LOW, HIGH), ("weights4", HIGH, HIGH))


def superres_factors(trained: Sequence[int], served: Sequence[int]) -> Optional[tuple[int, ...]]:
    """Per-axis integer factor served/trained; None when the grids agree. Refuses a coarser or non-integer served
    grid, which this repair does not describe."""
    trained, served = tuple(int(v) for v in trained), tuple(int(v) for v in served)
    if len(trained) != len(served):
        raise ValueError(f"grid rank mismatch: trained {trained}, served {served}")
    if trained == served:
        return None
    factors = []
    for t, s in zip(trained, served):
        if t <= 0 or s % t or s < t:
            raise ValueError(f"[fno-superres] served grid {served} is not an integer refinement of the trained "
                             f"grid {trained}")
        factors.append(s // t)
    return tuple(factors)


def scaled_padding(padding: Any, factors: Sequence[int]) -> list[int]:
    """Encoder padding per axis, scaled so the padded physical length matches the trained one."""
    pads = [int(padding)] * len(factors) if isinstance(padding, (int, float)) else [int(p) for p in padding]
    pads = (pads + [0] * len(factors))[: len(factors)]
    return [p * int(f) for p, f in zip(pads, factors)]


def _freq(index: int, n: int) -> int:
    """numpy fftfreq convention: index -> signed mode number on an axis of length n."""
    return index if index < (n + 1) // 2 else index - n


def _index(freq: int, n: int) -> Optional[int]:
    """Signed mode number -> index on an axis of length n, or None when the axis cannot hold it."""
    if -(n // 2) <= freq <= (n - 1) // 2:
        return freq % n
    return None


def _covers(i: int, n: int, m: int, half: int) -> Optional[int]:
    """Weight index of block-half `half` (LOW = [:m], HIGH = [-m:]) at axis index i, or None if not covered."""
    if half == LOW:
        return i if i < min(m, n) else None
    start = n - m
    return i - start if 0 <= start <= i < n else None


def _trained_writer(ix: int, iy: int, nx: int, ny: int, m1: int, m2: int):
    """(block name, wx, wy) whose weight the trained grid applied at coarse index (ix, iy): the last block written
    there, or None where no block wrote (the output mode was left zero)."""
    found = None
    for name, hx, hy in _BLOCKS:
        wx, wy = _covers(ix, nx, m1, hx), _covers(iy, ny, m2, hy)
        if wx is not None and wy is not None:
            found = (name, wx, wy)
    return found


def remap_spectral_conv3d(layer: Any, trained_padded: Sequence[int], factors: Sequence[int]) -> dict:
    """Rebuild a physicsnemo SpectralConv3d's four weight blocks for a grid `factors` times finer (padding scaled the
    same way), so each fine-grid mode gets the weight the trained grid applied at that frequency. In place; returns a
    summary for the log."""
    import torch

    nx, ny, nz = (int(v) for v in trained_padded)
    fx, fy, fz = (int(v) for v in factors)
    Nx, Ny = nx * fx, ny * fy  # z is a real FFT axis: its modes keep their index under refinement
    m1, m2, m3 = int(layer.modes1), int(layer.modes2), int(layer.modes3)
    if Nx < 2 * m1 or Ny < 2 * m2:
        raise ValueError(f"[fno-superres] modes ({m1}, {m2}) still overlap on the fine padded grid ({Nx}, {Ny})")
    old = {name: getattr(layer, name).detach().clone() for name, _, _ in _BLOCKS}
    new = {name: torch.zeros_like(w) for name, w in old.items()}
    moved = zeroed = 0
    for name, hx, hy in _BLOCKS:
        for jx in range(m1):
            Ix = jx if hx == LOW else Nx - m1 + jx
            ix = _index(_freq(Ix, Nx), nx)
            for jy in range(m2):
                Iy = jy if hy == LOW else Ny - m2 + jy
                iy = _index(_freq(Iy, Ny), ny)
                writer = None if ix is None or iy is None else _trained_writer(ix, iy, nx, ny, m1, m2)
                if writer is None:
                    zeroed += 1
                    continue
                src, wx, wy = writer
                new[name][:, :, jx, jy] = old[src][:, :, wx, wy]
                moved += int((src, wx, wy) != (name, jx, jy))
    # z is a real FFT: only [:m3] exists. Modes the trained grid's rfft did not reach never trained.
    nz_rfft = nz // 2 + 1
    if m3 > nz_rfft:
        for w in new.values():
            w[:, :, :, :, nz_rfft:] = 0
    with torch.no_grad():
        for name, w in new.items():
            getattr(layer, name).copy_(w)
    return {"moved": moved, "zeroed": zeroed, "blocks": 4 * m1 * m2}


def _encoders(model: Any):
    """physicsnemo FNO encoders inside `model`: modules carrying padding (`pad`, `ipad`) and `spconv_layers`."""
    for module in model.modules():
        if hasattr(module, "spconv_layers") and hasattr(module, "pad") and hasattr(module, "ipad"):
            yield module


def pad_axis_order(enc: Any) -> str:
    """Axis order of an encoder's `pad` list. physicsnemo.sym's FNO3DEncoder (the class the pino FNO arch builds)
    pads `(0, pad[0], 0, pad[1], 0, pad[2])` -- pad[0] on the LAST axis -- and crops `[..., :ipad[2], :ipad[1],
    :ipad[0]]`, so its list runs z, y, x; physicsnemo.models.fno's own encoder runs x, y, z. A symmetric padding hides
    the difference; the first version of this repair wrote [18, 18, 9] meaning x, y, z and padded z by 18, x by 9
    (feature dump, 2026-11-01: 99x58x58 instead of 108x58x49)."""
    return "zyx" if type(enc).__module__.startswith("physicsnemo.sym") else "xyz"


def apply_fno_superres(model: Any, trained_grid: Sequence[int], served_grid: Sequence[int]) -> Optional[dict]:
    """Repair every FNO encoder in `model` for scoring on `served_grid` after training on `trained_grid`. Returns a
    summary (paddings in x, y, z order), or None when nothing applies (same grid, no FNO encoder)."""
    factors = superres_factors(trained_grid, served_grid)
    if factors is None:
        return None
    summary = None
    for enc in _encoders(model):
        order = pad_axis_order(enc)
        raw = (list(enc.pad) + [0] * 3)[:3]
        pads_xyz = raw[::-1] if order == "zyx" else raw
        trained_padded = [int(t) + int(p) for t, p in zip(trained_grid, pads_xyz)]
        layers = [m for m in enc.spconv_layers if type(m).__name__ == "SpectralConv3d"]
        if not layers:
            continue
        stats = [remap_spectral_conv3d(m, trained_padded, factors) for m in layers]
        new_xyz = scaled_padding(pads_xyz, factors)
        new_raw = new_xyz[::-1] if order == "zyx" else new_xyz
        enc.pad = new_raw
        enc.ipad = [-p if p > 0 else None for p in new_raw]
        summary = {"trained_grid": list(trained_grid), "served_grid": list(served_grid), "factors": list(factors),
                   "padding_cells_xyz": {"trained": pads_xyz, "served": new_xyz}, "pad_list_order": order,
                   "spectral_layers": len(layers),
                   "weights_moved_per_layer": stats[0]["moved"], "modes_zeroed_per_layer": stats[0]["zeroed"]}
    return summary


def attach_fno_feature_dump(model: Any, path: str) -> None:
    """Diagnostic: on the first forward call, save sample 0's features at every stage of each FNO encoder (lift
    output, every spectral and pointwise layer output, encoder output) as float16 to `path` (.npz), with the
    encoder's padding. Lets a coarse-grid and a fine-grid evaluation of the same frame be compared stage by stage."""
    import numpy as np

    store: dict = {}
    done = {"saved": False}

    def keep(name):
        def hook(module, args, out):
            if done["saved"] or name in store:
                return
            t = out[0] if isinstance(out, (tuple, list)) else out
            store[name] = t[0].detach().float().cpu().numpy().astype(np.float16)
        return hook

    for e, enc in enumerate(_encoders(model)):
        # forward hooks only, each returning None: a forward PRE-hook that returns a value replaces the module's input
        if hasattr(enc, "lift_network"):
            enc.lift_network.register_forward_hook(keep(f"enc{e}_lift"))
        for k, layer in enumerate(enc.spconv_layers):
            layer.register_forward_hook(keep(f"enc{e}_spec{k}"))
        for k, layer in enumerate(getattr(enc, "conv_layers", [])):
            layer.register_forward_hook(keep(f"enc{e}_conv{k}"))

        def finish(module, args, out, e=e):
            if done["saved"]:
                return
            t = out[0] if isinstance(out, (tuple, list)) else out
            store[f"enc{e}_out"] = t[0].detach().float().cpu().numpy().astype(np.float16)
            store[f"enc{e}_pad"] = np.asarray(list(module.pad), dtype=np.int32)
            done["saved"] = True
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **store)
            print(f"[fno-dump] saved {len(store)} arrays to {path}", flush=True)

        enc.register_forward_hook(finish)


def maybe_apply_fno_superres(model: Any, network_dir: Any, served_grid: Sequence[int], run_cfg: Any = None,
                             output_dir: Any = None) -> Optional[dict]:
    """Evaluation hook: apply the repair when the model is an FNO whose stamped training grid (``trained_grid.json``
    in the network dir) differs from the grid being scored. Off with ``PINO_FNO_SUPERRES=0``; skipped for a
    ``super_res_factor`` study, whose padding config_loading already scales."""
    if os.environ.get("PINO_FNO_SUPERRES", "1").strip() == "0":
        return None
    if run_cfg is not None and int(getattr(run_cfg, "super_res_factor", 1) or 1) != 1:
        return None
    if not any(True for _ in _encoders(model)):
        return None
    stamp = Path(str(network_dir)) / "trained_grid.json"
    if not stamp.is_file():
        print("[fno-superres] no trained_grid.json beside the checkpoint: grid unknown, no repair applied", flush=True)
        return None
    try:
        trained = tuple(int(v) for v in json.loads(stamp.read_text())["grid"])
    except (ValueError, KeyError, TypeError, OSError):
        print(f"[fno-superres] unreadable {stamp}: no repair applied", flush=True)
        return None
    summary = apply_fno_superres(model, trained, tuple(int(v) for v in served_grid)[:3])
    if summary is None:
        return None
    print(f"[fno-superres] trained on {'x'.join(map(str, summary['trained_grid']))}, scoring on "
          f"{'x'.join(map(str, summary['served_grid']))} (factors {summary['factors']}): padding (x, y, z) "
          f"{summary['padding_cells_xyz']['trained']} -> {summary['padding_cells_xyz']['served']} cells "
          f"(pad list order {summary['pad_list_order']}); "
          f"{summary['spectral_layers']} spectral layers remapped onto the frequencies the trained grid held "
          f"({summary['weights_moved_per_layer']} mode weights moved, {summary['modes_zeroed_per_layer']} zeroed "
          f"per layer)", flush=True)
    if output_dir is not None:
        try:
            out = Path(str(output_dir))
            out.mkdir(parents=True, exist_ok=True)
            (out / "fno_superres.json").write_text(json.dumps(summary, indent=1))
        except OSError:
            pass
    return summary
