#!/usr/bin/env python
"""Measure per-frame inference wall time for trained models, with provenance.

Produces the `Inference, per frame` column of the manuscript cost table
(`tab:cost`). Every number it emits is accompanied by enough provenance to
reproduce it: the AiiDA model pk/uuid, the checkpoint file actually loaded, the
dataset collection and the exact case names timed, the frame count per forward
pass, and the host/GPU/library versions.

WHY A SCRIPT AND NOT THE EVAL NODES. An evaluation's `eval_summary.runtime.
wall_seconds` divided by its frame count is ~24 s/frame, because that wall clock
also covers dataset staging, model construction, metric computation and the
geometry extraction (IoU, Hausdorff). None of that is inference. This script
times the forward pass alone.

METHOD (prototyped in a notebook; this script is the tracked form):
  1. materialise the model node's repository -- `.hydra/config.yaml` plus the
     checkpoints -- into a temporary directory;
  2. point its config at the dataset directory and load through the SAME
     pipeline functions production inference uses (`load_cfg_and_paths`,
     `load_and_preprocess_data`, `build_datasets`, `build_model_and_nodes`);
  3. assert the checkpoint actually loads into the built model, so a silently
     re-initialised network cannot be timed;
  4. run `--warmup` untimed forward passes, `cuda.synchronize()`, then `--runs`
     timed passes, `cuda.synchronize()`, and divide.

Per-frame time is the average forward time divided by the number of frames in
one forward pass; that frame count is reported, never assumed.

Usage:
  python scripts/measure_inference_time.py --models 78585,80753,70803 \
      --data-dir ~/.cache/aiida-pino/runtime/<uuid> \
      --dataset 55017 --cases 1 --warmup 2 --runs 5 \
      --out inference_timing.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lpno.models import register_custom_hydra_inputs  # noqa: E402
from lpno.models.ns_energy_pipeline import (  # noqa: E402
    _assert_eval_checkpoint_loadable,
)
import lpno.models.pipeline.config_loading as config_loading  # noqa: E402
from lpno.models.pipeline import (  # noqa: E402
    build_datasets,
    build_model_and_nodes,
    get_splits,
    load_and_preprocess_data,
    load_cfg_and_paths,
    sample_count,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def host_provenance() -> dict:
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
            "count": torch.cuda.device_count(),
        }
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "numpy": np.__version__,
        "device": str(DEVICE),
        "gpu": gpu,
        "lpno_git_sha": git_sha(REPO_ROOT),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pino_lazy_crop_z_top": os.environ.get("PINO_LAZY_CROP_Z_TOP"),
        "pino_loader_workers": os.environ.get("PINO_LOADER_WORKERS"),
    }


def materialise_model(model_pk: int, dest: Path) -> dict:
    """Copy an AiiDA PinoModelData repository to `dest`; return its identity."""
    from aiida import orm

    node = orm.load_node(model_pk)
    dest.mkdir(parents=True, exist_ok=True)

    def _copy(rel: str) -> None:
        for name in node.base.repository.list_object_names(rel or None):
            child = f"{rel}/{name}" if rel else name
            try:
                data = node.base.repository.get_object_content(child, mode="rb")
            except (IsADirectoryError, OSError, TypeError):
                _copy(child)
                continue
            out = dest / child
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)

    _copy("")
    return {
        "model_pk": node.pk,
        "model_uuid": str(node.uuid),
        "model_label": str(node.label),
        "objects": sorted(p.name for p in dest.iterdir()),
    }


def build_context(run_root: Path, data_dir: Path, max_cases: int):
    """Load config + data + model exactly as production inference does."""
    register_custom_hydra_inputs()
    cfg = OmegaConf.load(run_root / ".hydra" / "config.yaml")
    with open_dict(cfg):
        cfg.run_mode = "eval"
        cfg.network_dir = str(run_root)
        cfg.custom.data_dir = str(data_dir)
        cfg.custom.val_data_dir = str(data_dir)
        cfg.custom.wandb_offline = True

    explicit = {"custom.data_dir", "custom.val_data_dir", "custom.wandb_offline"}
    original = config_loading._parse_explicit_cli_override_paths
    config_loading._parse_explicit_cli_override_paths = lambda argv=None: (set(explicit), set())
    try:
        run_cfg = load_cfg_and_paths(cfg)
    finally:
        config_loading._parse_explicit_cli_override_paths = original

    var = load_and_preprocess_data(
        run_cfg,
        input_dir=run_cfg.val_directory_path,
        h_norm_mode="max",
        set_runtime_params=False,
    )
    n_all = sample_count(var)
    # A lazy RuntimeStore (npz arraydir) is not a dict and cannot be sliced here;
    # restrict the case count by pointing --data-dir at a directory holding only
    # the cases you want timed, which also makes them explicit in the provenance.
    if isinstance(var, dict) and 0 < max_cases < n_all:
        keep = max_cases
        var = {
            k: (np.asarray(v)[:keep] if getattr(v, "ndim", 0) > 0
                and np.asarray(v).shape[0] == n_all else v)
            for k, v in var.items()
        }
    keep = sample_count(var)
    n = keep
    split = get_splits(mode="standard", n_samples=n, val_n_samples=n, seed=run_cfg.seed)[0]
    datasets = build_datasets(
        var=var, key_cfg=run_cfg.keys, split=split, pde_loss=False,
        vof_pde_loss=False, diff_loss=False, var_val_source=var, run_cfg=run_cfg,
    )
    bundle = build_model_and_nodes(
        cfg=run_cfg.cfg, key_cfg=run_cfg.keys, material_params=run_cfg.material_params,
        laser_params=run_cfg.laser_params, ref_scales=run_cfg.ref_scales,
        alpha_masking=run_cfg.alpha_masking, eval_only=True,
        model_family=run_cfg.model_family,
        deeponet_branch_keys=run_cfg.deeponet_branch_keys,
        deeponet_trunk_keys=run_cfg.deeponet_trunk_keys,
        pde_loss=False, vof_pde_loss=False, diff_loss=False,
        dx=run_cfg.dx, dy=run_cfg.dy, dz=run_cfg.dz, diff_weight=run_cfg.diff_weight,
    )
    # refuses to time a network whose weights did not actually load
    _assert_eval_checkpoint_loadable(bundle.model, run_root)
    model = bundle.model.to(DEVICE).eval()
    return run_cfg, model, datasets.ds_input_val, n_all, keep


def time_forward(model, ds_input, run_cfg, warmup: int, runs: int):
    out_keys = list(run_cfg.keys.str_fno_output_keys)
    invar = {
        k: torch.from_numpy(np.asarray(v)).to(device=DEVICE, dtype=torch.float32)
        for k, v in ds_input.items()
    }
    n_frames = int(next(iter(invar.values())).shape[0])

    def once():
        with torch.inference_mode():
            out = model(invar)
        return {k: out[k] for k in out_keys}

    for _ in range(warmup):
        once()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        once()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    total = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 1024**2) if DEVICE.type == "cuda" else None
    return {
        "frames_per_forward": n_frames,
        "timed_runs": runs,
        "warmup_runs": warmup,
        "total_seconds": total,
        "avg_forward_seconds": total / max(1, runs),
        "per_frame_ms": (total / max(1, runs)) / max(1, n_frames) * 1000.0,
        "gpu_peak_mb": peak_mb,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="comma-separated AiiDA model pks (needs the AiiDA DB)")
    ap.add_argument("--model-dirs", help="comma-separated already-exported model dirs "
                                         "(each holding _aiida_identity.json); use on a compute "
                                         "box that cannot reach the AiiDA database")
    ap.add_argument("--export-only", metavar="DIR",
                    help="materialise --models into DIR (with identity json) and exit, "
                         "so the dirs can be copied to the box that does the timing")
    ap.add_argument("--data-dir", help="runtime cache dir holding the cases")
    ap.add_argument("--dataset", type=int, default=None, help="AiiDA collection pk, recorded for provenance")
    ap.add_argument("--cases", type=int, default=1, help="number of cases to time (0 = all)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="inference_timing.json")
    args = ap.parse_args()

    if args.export_only:
        from aiida import load_profile
        load_profile()
        out_root = Path(args.export_only)
        for pk in [int(x) for x in args.models.split(",") if x.strip()]:
            d = out_root / f"m{pk}"
            ident = materialise_model(pk, d)
            ident["exported_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            ident["exported_from_host"] = platform.node()
            (d / "_aiida_identity.json").write_text(json.dumps(ident, indent=2))
            print(f"exported m{pk} {ident['model_label']} -> {d}")
        return 0

    if not args.data_dir:
        print("ERROR: --data-dir is required for timing", file=sys.stderr)
        return 2
    if not (args.models or args.model_dirs):
        print("ERROR: pass --models or --model-dirs", file=sys.stderr)
        return 2
    if args.models:
        from aiida import load_profile
        load_profile()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: data dir not found: {data_dir}", file=sys.stderr)
        return 2

    dataset_info = None
    case_names = None
    if args.dataset and args.models:
        from aiida import orm
        ds = orm.load_node(args.dataset)
        a = ds.base.attributes.all
        dataset_info = {"pk": ds.pk, "uuid": str(ds.uuid), "label": str(ds.label),
                        "simulation_count": a.get("simulation_count"),
                        "derivation_profile": a.get("derivation_profile")}
    manifests = sorted(p.stem.replace(".manifest", "") for p in data_dir.glob("*.manifest.json"))
    if manifests:
        case_names = manifests

    report = {
        "measurement": "per-frame inference wall time, forward pass only",
        "script": f"scripts/{Path(__file__).name}",
        "script_git_sha": git_sha(REPO_ROOT),
        "host": host_provenance(),
        "data_dir": str(data_dir),
        "dataset": dataset_info,
        "cases_available": case_names,
        "cases_requested": args.cases,
        "results": [],
    }

    if args.model_dirs:
        sources = [("dir", Path(x.strip())) for x in args.model_dirs.split(",") if x.strip()]
    else:
        sources = [("pk", int(x)) for x in args.models.split(",") if x.strip()]

    for kind, src in sources:
        with tempfile.TemporaryDirectory(prefix="mt_") as tmp:
            if kind == "pk":
                root = Path(tmp)
                ident = materialise_model(src, root)
                pk = src
            else:
                root = src
                idf = root / "_aiida_identity.json"
                ident = json.loads(idf.read_text()) if idf.exists() else {"model_dir": str(root)}
                pk = ident.get("model_pk", root.name)
            print(f"\n=== m{pk} {ident['model_label']} ===", flush=True)
            try:
                run_cfg, model, ds_input, n_all, kept = build_context(root, data_dir, args.cases)
                timing = time_forward(model, ds_input, run_cfg, args.warmup, args.runs)
                ckpt = "u_network.0.pth" if (root / "u_network.0.pth").exists() else None
                row = {**ident, "model_family": str(run_cfg.model_family),
                       "checkpoint_used": ckpt,
                       "cases_in_data_dir": n_all, "cases_timed": kept, **timing}
                print(f"  frames/forward={timing['frames_per_forward']}  "
                      f"avg forward={timing['avg_forward_seconds']*1000:.1f} ms  "
                      f"per frame={timing['per_frame_ms']:.2f} ms", flush=True)
            except Exception as exc:  # record the failure rather than losing the run
                row = {**ident, "error": f"{type(exc).__name__}: {exc}"}
                print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            report["results"].append(row)
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
