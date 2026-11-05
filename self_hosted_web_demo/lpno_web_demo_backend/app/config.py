from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="LPFNO_", case_sensitive=False)

    app_name: str = "lpfno-inference-server"
    host: str = "0.0.0.0"
    port: int = 8000
    uvicorn_workers: int = 1
    log_level: str = "INFO"
    startup_load_runtime: bool = True
    use_stub_runtime: bool = False

    # Which CUDA device to load models on. "auto" picks the GPU with the most
    # free memory at load time (skips a full GPU0); or force "cpu"/"1"/"cuda:1".
    cuda_device: str = "auto"

    pages_origin: str = ""

    request_max_bytes: int = 16 * 1024
    preview_timeout_ms: int = 10_000
    final_timeout_ms: int = 45_000

    # Server-side clamp for the steady route: the served catalog's own extremes
    # (ti64_straight_path_calibrated_with_keyholes: P 40..215 W, V 0.101..0.960
    # m/s). A 200 W ceiling here left the three catalog cases above it
    # unreachable from the map (2026-09-14). The pages narrow this further to
    # each model's meta.json envelope.
    p_min_watts: float = 40.0
    p_max_watts: float = 215.0
    v_min_mps: float = 0.101
    v_max_mps: float = 0.96

    render_min_width_px: int = 320
    render_max_width_px: int = 1600
    render_min_height_px: int = 200
    render_max_height_px: int = 1000

    mesh_size_um: float = 10.0

    # Served models + ground truth live under data/<page>/ (see data/README.md;
    # populate with `python scripts/assets.py fetch <bundle>`). Defaults are repo-relative so a
    # populated checkout runs self-contained without a .env.
    runtime_run_dir: str = "data/steady-state/models/fno_cal"
    runtime_checkpoint_path: str = "data/steady-state/models/fno_cal/u_network.0.pth"
    runtime_sim_params_path: str = "data/steady-state/models/fno_cal/flow3D_params.yaml"
    # Per-page model folders (dynamic dropdown discovery; see model_registry.py).
    steady_models_dir: str = "data/steady-state/models"
    runtime_results_dir: str = "/tmp/lpfno_outputs"
    runtime_super_res_factor: int = 2
    runtime_time_avg_window: int = 30
    runtime_include_time: str = "auto"
    runtime_alpha_masking: str = "auto"
    runtime_crop_x_to_effective_domain: bool = True
    runtime_initial_p_watts: float = 50.0
    runtime_initial_v_mps: float = 0.26

    # /infer_path (time-dependent random-path inference)
    path_data_dir: str = "data/random_path/ground_truth"
    path_models_dir: str = "data/random_path/models"
    path_rigno_graph_cache_dir: str = "outputs/rigno_graphs"
    path_timeout_ms: int = 60_000
    path_request_max_bytes: int = 128 * 1024

    # /infer_straight (time-dependent straight-track inference at free P/V): the
    # calibrated lab-frame models (16 conditioning channels, 90x40x40 grid, per
    # -model grid.npz override) and their ground truth below.
    straight_data_dir: str = "data/straight_path_time/ground_truth_calibrated"
    straight_models_dir: str = "data/straight_path_time/models"

    # Steady-state ground-truth catalog + comparison (/infer + /gt_catalog).
    # Calibrated steady GT (ti64_straight_path_calibrated_with_keyholes, split 55017),
    # built by scripts/build_calibrated_steady_gt.py. MUST match the served models'
    # physics era: the legacy cache under ground_truth/ is from
    # original_straight_path_data, and comparing a calibrated prediction against it
    # renders a T_FILL_K band and a displaced boundary that look like model error.
    # Point back at "data/steady-state/ground_truth" to serve the legacy pair.
    steady_gt_data_dir: str = "data/steady-state/ground_truth_calibrated"

    # Frames-per-forward-pass for /infer_path (both models). Frames are
    # independent samples, so this trades GPU memory for speed only. Measured
    # on the shared RTX 24 GB card with training resident: RIGNO batch 2 was
    # only 6% faster but doubled peak memory (2.5 -> 4.6 GiB) and left <4 GiB
    # free, so the safe default is 1. Raise via LPFNO_PATH_INFER_BATCH when
    # the GPU is otherwise idle.
    path_infer_batch: int = 1

    # How many (model, path/PV) predictions to keep per time-dependent route so
    # comparing several models on one path/PV re-renders without re-inferring.
    # ~70 MB each on the path grid; 8 covers all models on ~2 inputs.
    prediction_cache_entries: int = 8
    # Simultaneously RESIDENT models per adapter (steady runtimes, and path and
    # straight each keep their own). A model pins its weights plus its lazily
    # built neighbour graphs, so an unbounded cache grew the backend to 11.4 GiB
    # from an 886 MiB baseline and starved a training sharing the card. Evicting
    # is cheap here because the graphs are also cached on disk under
    # path_rigno_graph_cache_dir, so a reload is far cheaper than a cold start.
    model_cache_entries: int = 2
    # Before a forward, if the DEVICE has less than this free, drop every cached
    # model except the one about to run and return the allocator's pool. Capping
    # the caches bounds growth but does not guarantee room to RUN: on 2026-09-02
    # a bounded ~9 GiB of cached models sat on a card also holding an 11.9 GiB
    # training, leaving 2.28 GiB free against a steady forward that allocates
    # 4.64 GiB, and every /infer returned 500. 0 disables the guard.
    min_free_vram_gib: float = 6.0
    # Drop every cached model after this many seconds with no inference, so an
    # IDLE demo hands its GPU back to trainings sharing the card instead of
    # sitting on several GiB indefinitely. The headroom guard above only fires
    # when a request arrives, which does nothing for an idle backend. 0 disables.
    model_idle_ttl_s: float = 600.0
    # How often the background sweeper checks the idle TTL.
    cache_sweep_interval_s: float = 60.0

    hstar_eta: float = 0.35
    hstar_rho: float = 4420.0
    hstar_cp: float = 750.0
    hstar_delta_tm: float = 1573.0
    hstar_d: float = 8.1e-6
    hstar_sigma: float = 50e-6

    @field_validator("uvicorn_workers")
    @classmethod
    def _validate_workers(cls, value: int) -> int:
        if value != 1:
            raise ValueError("uvicorn_workers must be 1 to preserve in-process busy lock semantics")
        return value

    @field_validator("runtime_include_time", "runtime_alpha_masking")
    @classmethod
    def _validate_optional_bool_str(cls, value: str) -> str:
        lowered = value.strip().lower()
        if lowered not in {"auto", "true", "false"}:
            raise ValueError("must be one of: auto, true, false")
        return lowered

    @property
    def cors_allow_origins(self) -> list[str]:
        origin = self.pages_origin.strip().rstrip("/")
        if not origin:
            return []
        return [origin]

    def infer_timeout_seconds(self, mode: Literal["preview", "final"]) -> float:
        if mode == "preview":
            return self.preview_timeout_ms / 1000.0
        return self.final_timeout_ms / 1000.0

    def clamp_inputs(self, p_watts: float, v_scan_mps: float) -> tuple[float, float]:
        p = min(self.p_max_watts, max(self.p_min_watts, float(p_watts)))
        v = min(self.v_max_mps, max(self.v_min_mps, float(v_scan_mps)))
        return p, v

    def clamp_render_size(self, width_px: int, height_px: int) -> tuple[int, int]:
        w = min(self.render_max_width_px, max(self.render_min_width_px, int(width_px)))
        h = min(self.render_max_height_px, max(self.render_min_height_px, int(height_px)))
        return w, h

    def parse_optional_bool(self, raw: str) -> bool | None:
        lowered = raw.strip().lower()
        if lowered == "auto":
            return None
        return lowered == "true"

    @property
    def runtime_include_time_override(self) -> bool | None:
        return self.parse_optional_bool(self.runtime_include_time)

    @property
    def runtime_alpha_masking_override(self) -> bool | None:
        return self.parse_optional_bool(self.runtime_alpha_masking)

    def compute_hstar(self, p_watts: float, v_scan_mps: float) -> float:
        import math

        v_abs = max(abs(float(v_scan_mps)), 1e-12)
        denom = self.hstar_rho * self.hstar_cp * self.hstar_delta_tm * math.sqrt(
            math.pi * self.hstar_d * (self.hstar_sigma**3) * v_abs
        )
        if abs(denom) < 1e-30:
            return float("nan")
        return self.hstar_eta * float(p_watts) / denom
