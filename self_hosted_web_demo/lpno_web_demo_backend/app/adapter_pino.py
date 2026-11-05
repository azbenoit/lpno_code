from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .arch_build import build_gaot_kwargs, build_rigno_net, is_fno_config, is_gaot
from .config import Settings
from . import submodule

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InferenceContext:
    infer: Any
    input_keys: list[str]
    output_keys: list[str]
    template_input: dict[str, np.ndarray]
    ref_scales: dict[str, Any]
    material_params: dict[str, Any]
    alpha_masking: bool
    h_norm_denom: float
    results_dir: Path
    bounds: list[list[float]]
    x_phys: np.ndarray
    y_phys: np.ndarray
    z_phys: np.ndarray
    norm_summary: str
    # Folder-level "serving" block of flow3D_params.yaml (may be empty): lets a
    # model folder override the grid policy (super_res_factor, crop_x,
    # include_time) that the settings otherwise decide.
    serving: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PinoRuntime:
    ctx: InferenceContext
    device: str
    submodule_commit: str | None
    submodule_src: str



class PinoRuntimeAdapter:
    def __init__(self) -> None:
        self._imports: dict[str, Any] | None = None

    def build_runtime(
        self,
        settings: Settings,
        *,
        run_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        sim_params_path: str | Path | None = None,
    ) -> PinoRuntime:
        """Build a steady-state runtime. With no overrides it loads the model
        configured by ``settings.runtime_*`` (the FNO); pass ``run_dir`` etc. to
        load another steady model folder by path (FNO or RIGNO — the architecture
        is read from the folder's Hydra config)."""
        imports = self._ensure_imports()
        torch = imports["torch"]

        from .gpu import pick_device

        device = pick_device(torch, settings.cuda_device)
        ctx = self._build_context(
            settings=settings,
            imports=imports,
            device=device,
            run_dir=Path(run_dir) if run_dir is not None else Path(settings.runtime_run_dir),
            checkpoint_path=(
                Path(checkpoint_path) if checkpoint_path is not None else Path(settings.runtime_checkpoint_path)
            ),
            sim_params_path=(
                Path(sim_params_path) if sim_params_path is not None else Path(settings.runtime_sim_params_path)
            ),
        )
        return PinoRuntime(
            ctx=ctx,
            device=str(device),
            submodule_commit=submodule.submodule_commit(),
            submodule_src=str(submodule.submodule_src()),
        )

    def infer(
        self,
        runtime: PinoRuntime,
        p_watts: float,
        v_scan_mps: float,
        x_shift_m: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """``x_shift_m`` translates the served x window, in metres.

        The steady models are trained on a beam-following 61-cell window of the
        ABSOLUTE domain, and that window is not in the same place for every
        case: over the 13 calibrated test cases its x[0] runs
        -1.877e-4..-1.550e-4. The runtime is built on one window, so comparing
        against a case that used another one measures the offset, not the model
        (6.23 K substrate on the matching cases against 21.00 K on the offset
        ones, 2026-09-03). Passing the matched case's own x[0] minus the
        runtime's puts both on the same window. 0.0 = the runtime's own window,
        which is right whenever there is no case to match (free P/V).
        """
        imports = self._ensure_imports()
        return self._run_inference_for_p_v(
            runtime.ctx, p_watts, v_scan_mps, imports=imports, x_shift_m=x_shift_m,
        )

    def runtime_summary(self, runtime: PinoRuntime) -> dict[str, Any]:
        first_key = runtime.ctx.input_keys[0] if runtime.ctx.input_keys else None
        grid_shape = None
        if first_key and first_key in runtime.ctx.template_input:
            grid_shape = tuple(int(x) for x in runtime.ctx.template_input[first_key].shape[-3:])
        return {
            "device": runtime.device,
            "submodule_commit": runtime.submodule_commit,
            "submodule_src": runtime.submodule_src,
            "input_keys": list(runtime.ctx.input_keys),
            "output_keys": list(runtime.ctx.output_keys),
            "grid_shape_xyz": grid_shape,
        }

    def _ensure_imports(self) -> dict[str, Any]:
        if self._imports is not None:
            return self._imports

        submodule.ensure_submodule_on_path()

        import torch
        import yaml
        from omegaconf import OmegaConf

        from lpno.custom_utils.inferencer import FNOGridInferencer
        from lpno.custom_utils.utils import coerce_numbers, fill_where_mask_smooth_np, smooth_liquid_fraction_numpy
        from lpno.models import register_custom_hydra_inputs
        from physicsnemo.sym.hydra import instantiate_arch
        from physicsnemo.sym.key import Key

        self._imports = {
            "torch": torch,
            "yaml": yaml,
            "OmegaConf": OmegaConf,
            "FNOGridInferencer": FNOGridInferencer,
            "coerce_numbers": coerce_numbers,
            "fill_where_mask_smooth_np": fill_where_mask_smooth_np,
            "smooth_liquid_fraction_numpy": smooth_liquid_fraction_numpy,
            "register_custom_hydra_inputs": register_custom_hydra_inputs,
            "instantiate_arch": instantiate_arch,
            "Key": Key,
        }
        return self._imports

    # Kept as a class attribute: tests reach _find_hydra_config_path here.
    _find_hydra_config_path = staticmethod(submodule.find_hydra_config_path)

    def _prepare_cfg_and_template(
        self,
        *,
        settings: Settings,
        imports: dict[str, Any],
        run_dir: Path,
        sim_params_path: Path,
    ) -> tuple[Any, dict[str, Any], dict[str, np.ndarray], float, list[list[float]], dict[str, np.ndarray], str]:
        yaml = imports["yaml"]
        OmegaConf = imports["OmegaConf"]
        coerce_numbers = imports["coerce_numbers"]
        register_custom_hydra_inputs = imports["register_custom_hydra_inputs"]

        register_custom_hydra_inputs()
        hydra_config_path = self._find_hydra_config_path(run_dir)
        if hydra_config_path is None:
            raise FileNotFoundError(
                "Hydra config not found under "
                f"{run_dir / '.hydra' / 'config.yaml'}, "
                f"{run_dir.parent.parent / '.hydra' / 'config.yaml'}, "
                f"or {run_dir.parent / '.hydra' / 'config.yaml'}"
            )
        cfg = OmegaConf.load(hydra_config_path)

        if not sim_params_path.exists():
            raise FileNotFoundError(f"sim_params not found: {sim_params_path}")
        with sim_params_path.open("r", encoding="utf-8") as f:
            yaml_params = coerce_numbers(yaml.safe_load(f))

        grid_params = copy.deepcopy(yaml_params["geometry"])
        ref_scales = copy.deepcopy(yaml_params["ref_scales"])
        material_params = copy.deepcopy(yaml_params.get("material_params", {}))
        software = yaml_params.get("software", "openFoam")
        # Optional folder-level serving overrides: the calibrated steady folders
        # were pre-averaged/cropped offline, so they serve at native resolution
        # (sr=1, crop_x=false, include_time=false) instead of the settings defaults.
        serving = copy.deepcopy(yaml_params.get("serving", {}) or {})

        include_time = serving.get("include_time", settings.runtime_include_time_override)
        if include_time is None:
            include_time = software == "flow3D"

        (xmin, xmax) = grid_params["x_range"]
        (ymin, ymax) = grid_params["y_range"]
        (zmin, zmax) = grid_params["z_range"]
        (nx, ny, nz) = grid_params["resolution"]

        sr = int(serving.get("super_res_factor", settings.runtime_super_res_factor))
        taw = int(settings.runtime_time_avg_window)
        nx_sr, ny_sr, nz_sr = (nx * sr, ny * sr, nz * sr)

        nx_effective = nx_sr - taw * int(bool(include_time))
        xmin += taw * 1e-5 / sr * int(bool(include_time)) / 2
        xmax -= taw * 1e-5 / sr * int(bool(include_time)) / 2
        # FNO spectral modes are capped below Nyquist on the (super-res) grid and
        # its zero-padding scales with the super-res factor. RIGNO is graph-based
        # and grid-agnostic, so this block only applies to the FNO architecture.
        if is_fno_config(cfg):
            nyquist_x = max(1, nx_effective // 2)
            nyquist_y = max(1, ny_sr // 2)
            nyquist_z = max(1, nz_sr // 2)
            fno_modes = list(cfg.arch.fno.fno_modes)
            fno_modes[0] = min(fno_modes[0], nyquist_x)
            fno_modes[1] = min(fno_modes[1], nyquist_y)
            fno_modes[2] = min(fno_modes[2], nyquist_z)
            cfg.arch.fno.fno_modes = tuple(fno_modes)
            cfg.arch.fno.padding = cfg.arch.fno.padding * sr

        input_keys = list(yaml_params.get("FNO_input_keys", []))
        output_keys = list(yaml_params.get("FNO_output_keys", []))

        gx_full = np.linspace(xmin, xmax, nx_sr, dtype=np.float32)
        gy = np.linspace(ymin, ymax, ny_sr, dtype=np.float32)
        gz = np.linspace(zmin, zmax, nz_sr, dtype=np.float32)

        x_start = 0
        if bool(serving.get("crop_x", settings.runtime_crop_x_to_effective_domain)) and bool(include_time):
            crop = taw
            if crop >= nx_sr:
                raise ValueError(
                    f"Invalid crop: TIME_AVG_WINDOW={crop} >= nx_sr={nx_sr}. Reduce TIME_AVG_WINDOW or disable x-crop."
                )
            x_start = crop

        gx = gx_full[x_start:]
        nx_in = int(gx.size)
        bounds = [[float(gx[0]), float(gx[-1])], [float(ymin), float(ymax)], [float(zmin), float(zmax)]]

        x = gx.reshape(1, 1, -1, 1, 1)
        y = gy.reshape(1, 1, 1, -1, 1)
        z = gz.reshape(1, 1, 1, 1, -1)

        template: dict[str, np.ndarray] = {}
        shape = (1, 1, nx_in, ny_sr, nz_sr)
        p0 = float(settings.runtime_initial_p_watts)
        v0 = float(settings.runtime_initial_v_mps)
        p_norm = p0 / float(ref_scales["P"])
        v_norm = v0 / float(ref_scales["V"])
        h_norm_denom = float(ref_scales.get("H", 1.0))
        if np.isclose(h_norm_denom, 0.0):
            h_norm_denom = 1.0
        h_norm_val = p0 / np.sqrt(max(abs(v0), 1e-12)) / h_norm_denom

        for key in input_keys:
            if key == "x":
                template[key] = (np.broadcast_to(x, shape) / ref_scales["L"]).astype(np.float32)
            elif key == "y":
                template[key] = (np.broadcast_to(y, shape) / ref_scales["L"]).astype(np.float32)
            elif key == "z":
                template[key] = (np.broadcast_to(z, shape) / ref_scales["L"]).astype(np.float32)
            elif key == "P":
                template[key] = np.full(shape, p_norm, dtype=np.float32)
            elif key == "V_scan":
                template[key] = np.full(shape, v_norm, dtype=np.float32)
            elif key == "H_norm":
                template[key] = np.full(shape, h_norm_val, dtype=np.float32)
            elif key == "time_s":
                template[key] = np.zeros(shape, dtype=np.float32)
            else:
                logger.warning("Unhandled input key '%s'; filling zeros", key)
                template[key] = np.zeros(shape, dtype=np.float32)

        axes = {
            "x": gx.astype(np.float32, copy=False),
            "y": gy.astype(np.float32, copy=False),
            "z": gz.astype(np.float32, copy=False),
        }
        norm_summary = (
            "Input normalization: x,y,z are divided by L; "
            "P is divided by Pref; V_scan is divided by Vref; "
            "H_norm = (P/sqrt(|V|)) / Href"
        )

        meta = {
            "ref_scales": ref_scales,
            "material_params": material_params,
            "software": software,
            "input_keys": input_keys,
            "output_keys": output_keys,
            "serving": serving,
        }
        return cfg, meta, template, h_norm_denom, bounds, axes, norm_summary

    def _build_context(
        self,
        *,
        settings: Settings,
        imports: dict[str, Any],
        device: Any,
        run_dir: Path,
        checkpoint_path: Path,
        sim_params_path: Path,
    ) -> InferenceContext:
        torch = imports["torch"]
        FNOGridInferencer = imports["FNOGridInferencer"]
        instantiate_arch = imports["instantiate_arch"]
        Key = imports["Key"]

        cfg, meta, template_input, h_norm_denom, bounds, axes, norm_summary = self._prepare_cfg_and_template(
            settings=settings,
            imports=imports,
            run_dir=run_dir,
            sim_params_path=sim_params_path,
        )

        software = meta["software"]
        alpha_masking = settings.runtime_alpha_masking_override
        if alpha_masking is None:
            alpha_masking = software == "flow3D"

        input_keys = meta["input_keys"]
        output_keys = meta["output_keys"]
        fno_input_keys = [Key(k) for k in input_keys]
        fno_output_keys = [Key(k) for k in output_keys]

        if is_fno_config(cfg):
            decoder_net = instantiate_arch(cfg=cfg.arch.decoder, output_keys=fno_output_keys)
            model = instantiate_arch(
                cfg=cfg.arch.fno,
                decoder_net=decoder_net,
                input_keys=fno_input_keys,
                output_keys=fno_output_keys,
            )
        elif is_gaot(cfg):
            # GAOT and RIGNO are grid-agnostic, so they serve on the grid the
            # adapter already builds for the FNO; see app/arch_build.py.
            from lpno.models.gaot_arch import GAOTArch

            model = GAOTArch(**build_gaot_kwargs(
                cfg=cfg,
                cache_dir=submodule.graph_cache_dir(settings),
                input_keys=fno_input_keys,
                output_keys=fno_output_keys,
            ))
        else:
            model = build_rigno_net(
                cfg=cfg, cache_dir=submodule.graph_cache_dir(settings),
                input_keys=fno_input_keys, output_keys=fno_output_keys,
            )

        submodule.load_checkpoint_strict(model, checkpoint_path, torch)
        model.to(device)
        model.eval()

        results_dir = Path(settings.runtime_results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        infer = FNOGridInferencer(
            model=model,
            input_keys=input_keys,
            output_names=output_keys,
            results_dir=results_dir / "vti",
            device=str(device),
            requires_grad=False,
            mask_fn=None,
            log_iter=False,
            bounds=bounds,
        )

        return InferenceContext(
            infer=infer,
            input_keys=input_keys,
            output_keys=output_keys,
            template_input=template_input,
            ref_scales=meta["ref_scales"],
            material_params=meta["material_params"],
            alpha_masking=bool(alpha_masking),
            h_norm_denom=float(h_norm_denom),
            results_dir=results_dir,
            bounds=bounds,
            x_phys=axes["x"],
            y_phys=axes["y"],
            z_phys=axes["z"],
            norm_summary=norm_summary,
            serving=meta["serving"],
        )

    def _run_inference_for_p_v(
        self,
        ctx: InferenceContext,
        p_watts: float,
        v_scan_m_per_s: float,
        *,
        imports: dict[str, Any],
        x_shift_m: float = 0.0,
    ) -> dict[str, np.ndarray]:
        smooth_liquid_fraction_numpy = imports["smooth_liquid_fraction_numpy"]
        fill_where_mask_smooth_np = imports["fill_where_mask_smooth_np"]

        if np.isclose(v_scan_m_per_s, 0.0):
            raise ValueError("Vscan must be non-zero to compute H_norm.")

        invar = {k: v.copy() for k, v in ctx.template_input.items()}
        # Translate the window before anything else reads x. The channel is
        # stored normalised by ref_scales["L"], so the shift is too.
        if x_shift_m and "x" in invar:
            invar["x"] = invar["x"] + np.float32(float(x_shift_m) / float(ctx.ref_scales["L"]))
        p_norm = p_watts / float(ctx.ref_scales["P"])
        v_norm = v_scan_m_per_s / float(ctx.ref_scales["V"])

        if "P" in invar:
            invar["P"][:] = p_norm
        if "V_scan" in invar:
            invar["V_scan"][:] = v_norm
        if "H_norm" in invar:
            h_phys = p_watts / np.sqrt(np.abs(v_scan_m_per_s))
            invar["H_norm"][:] = h_phys / ctx.h_norm_denom

        preds = ctx.infer.run(invar, name="preds_widget", step=0, save_filetypes=())

        if "T" in preds:
            preds["T"] = preds["T"] * float(ctx.ref_scales["T"])

        # fl from the folder's own T_s/T_l (flow3D_params.yaml material_params).
        if all(k in ctx.material_params for k in ("T_s", "T_l")):
            fl_bounds: tuple[float, float] | None = (float(ctx.material_params["T_s"]), float(ctx.material_params["T_l"]))
        else:
            fl_bounds = None
        if "T" in preds and fl_bounds is not None:
            preds["fl"] = smooth_liquid_fraction_numpy(preds["T"], fl_bounds[0], fl_bounds[1])

        if ctx.alpha_masking and "alpha" in preds:
            # Gate every field on the RAW metal fraction, captured before the loop,
            # so the result is independent of output-key order (if a model listed
            # "alpha" before "T", filling T on the already-masked alpha would break
            # the renderer's raw-alpha fill inversion). Matches the path adapter.
            alpha_raw = preds["alpha"]
            for key in list(preds.keys()):
                if key == "T":
                    preds[key] = fill_where_mask_smooth_np(
                        preds[key], alpha_raw, threshold=0.5, d=3123, k=20, inplace=False
                    )
                elif key not in ("x", "y", "z", "P", "V_scan", "Deposition"):
                    preds[key] = fill_where_mask_smooth_np(
                        preds[key], alpha_raw, threshold=0.5, d=0.0, k=20, inplace=False
                    )

        return preds
