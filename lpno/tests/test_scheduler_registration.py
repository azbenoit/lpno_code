"""The one_cycle scheduler group option is registered and instantiable.

Motivated by the RIGNO recipe: the shipped tf_exponential_lr decays LR to
~4e-7 by 16k steps on long runs; upstream RIGNO trains with a one-cycle
schedule. physicsnemo instantiates torch-target scheduler configs generically
and steps them once per training step, so registering the conf is enough.
"""
import torch

from lpno.models import register_custom_hydra_inputs


def test_one_cycle_scheduler_registered_and_instantiable():
    register_custom_hydra_inputs()
    from hydra.core.config_store import ConfigStore

    node = ConfigStore.instance().load("scheduler/one_cycle.yaml").node
    assert node._target_ == "torch.optim.lr_scheduler.OneCycleLR"

    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    total = 100
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=float(node.max_lr), total_steps=total,
        pct_start=float(node.pct_start), div_factor=float(node.div_factor),
        final_div_factor=float(node.final_div_factor))
    lrs = []
    for _ in range(total):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    # warmup then anneal: peak near max_lr is reached, then LR decays well below it
    assert max(lrs) <= float(node.max_lr) + 1e-12
    assert max(lrs) > 0.9 * float(node.max_lr)
    assert lrs[-1] < 0.05 * max(lrs)
