"""Model package helpers and entrypoint registration.

Keep this module free of torch/physicsnemo imports: the training/eval
entrypoints import this package BEFORE bootstrap_cuda_visible_devices_from_argv
sets CUDA_VISIBLE_DEVICES, and an early torch import breaks custom.cuda_device
routing (rigno_rp5/rp6 OOM'd on the wrong GPU this way). The physicsnemo_ext
op registration lives in ns_energy_pipeline.py, which loads after bootstrap.
"""


def register_custom_hydra_inputs() -> None:
    """Register custom Hydra inputs used by the training entrypoints."""

    from dataclasses import dataclass, field
    from typing import List

    from hydra.core.config_store import ConfigStore
    from physicsnemo.sym.hydra.optimizer import OptimizerConf
    from physicsnemo.sym.hydra.scheduler import SchedulerConf

    @dataclass
    class LionConf(OptimizerConf):
        _target_: str = "lion_pytorch.Lion"
        lr: float = 1e-4
        betas: List[float] = field(default_factory=lambda: [0.9, 0.99])
        weight_decay: float = 0.0

    @dataclass
    class OneCycleLRConf(SchedulerConf):
        """torch OneCycleLR, stepped once per training step by physicsnemo.

        Warmup to max_lr over pct_start, cosine-anneal to max_lr/div_factor/
        final_div_factor. total_steps must be set to training.max_steps per run
        (no interpolation here). cycle_momentum off: Lion has betas, not momentum.
        """
        _target_: str = "torch.optim.lr_scheduler.OneCycleLR"
        max_lr: float = 1e-4
        total_steps: int = 32000
        pct_start: float = 0.05
        div_factor: float = 10.0
        final_div_factor: float = 10.0
        cycle_momentum: bool = False

    cs = ConfigStore.instance()
    cs.store(group="optimizer", name="lion", node=LionConf)
    cs.store(group="scheduler", name="one_cycle", node=OneCycleLRConf)
