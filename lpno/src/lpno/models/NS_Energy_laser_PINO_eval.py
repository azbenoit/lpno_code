from pathlib import Path

HYDRA_CONFIG_DIR = Path(__file__).resolve().parents[1] / "conf" / "hydra"

from .cuda_bootstrap import bootstrap_cuda_visible_devices_from_argv, ensure_single_process_launch_env

bootstrap_cuda_visible_devices_from_argv()
ensure_single_process_launch_env()

from .argv_custom_remap import remap_plus_args_to_custom

remap_plus_args_to_custom()

import physicsnemo.sym
from physicsnemo.sym.hydra import PhysicsNeMoConfig

from . import register_custom_hydra_inputs
from .ns_energy_pipeline import run_training_pipeline
# pyright: reportAttributeAccessIssue=false
# pyright: reportUnboundVariable=false
# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportOptionalSubscript=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportOptionalMemberAccess=false


register_custom_hydra_inputs()


@physicsnemo.sym.main(config_path=str(HYDRA_CONFIG_DIR), config_name="config_NS_energy_PINO")
def run(cfg: PhysicsNeMoConfig) -> None:
    # This wrapper is eval-only by design.
    cfg.run_mode = "eval"
    run_training_pipeline(cfg, default_split_mode="standard")


if __name__ == "__main__":
    run()
