"""A param group added after OneCycleLR init must not kill the first step.

physicsnemo builds the scheduler (`trainer.py:501`), then instantiates the
aggregator, then — only if the aggregator has parameters —
`optimizer.add_param_group(...)` (`trainer.py:525`). OneCycleLR's __init__
stamps `initial_lr`/`max_lr`/`min_lr` onto the groups that exist at that
moment; `get_lr` then reads them per group, so the late group raises
`KeyError: 'initial_lr'` on the very first `scheduler.step()`.

Sum/ReLoBRaLo have no parameters, so every run to date dodged this. The
homoscedastic aggregator (learned log-variance per key) is the first with
parameters — measured 2026-08-10: `gaot_spcal_hom_20k` and `gaot_spcal_vel_20k`
both died at step 0 with exactly that KeyError.

The repair copies the scheduler bookkeeping keys from a scheduled group onto
any group missing them, so the late group follows the same curve. Upstream is
read-only; the patch lives in the pipeline right after Solver construction.
"""

from __future__ import annotations

import pytest
import torch

from lpno.custom_utils.utils import ensure_late_param_group_scheduler_keys


def _setup():
    p1 = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.Adam([p1], lr=3e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=100)
    late = torch.nn.Parameter(torch.zeros(2))   # the aggregator's log_var
    opt.add_param_group({"params": [late]})
    return opt, sched


def test_reproduces_the_upstream_keyerror():
    opt, sched = _setup()
    opt.step()
    with pytest.raises(KeyError):
        sched.step()


def test_repair_makes_the_late_group_schedulable():
    opt, sched = _setup()
    ensure_late_param_group_scheduler_keys(opt)
    opt.step()
    for _ in range(5):
        sched.step()
    # the late group follows the same one-cycle curve as the first
    assert opt.param_groups[1]["lr"] == pytest.approx(opt.param_groups[0]["lr"])


def test_repair_is_a_noop_when_all_groups_are_scheduled():
    p1 = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.Adam([p1], lr=3e-4)
    torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=100)
    before = {k: v for k, v in opt.param_groups[0].items() if k != "params"}
    ensure_late_param_group_scheduler_keys(opt)
    after = {k: v for k, v in opt.param_groups[0].items() if k != "params"}
    assert before == after


def test_repair_is_a_noop_without_a_scheduler():
    """Plain optimizer (no scheduler keys anywhere): nothing to copy, no crash."""
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(3))], lr=3e-4)
    opt.add_param_group({"params": [torch.nn.Parameter(torch.zeros(2))]})
    ensure_late_param_group_scheduler_keys(opt)
    assert "initial_lr" not in opt.param_groups[1]


def test_attach_repairs_at_the_load_network_seam():
    """The optimizer exists only inside _train_loop; the repair must fire there.

    Reproduces the wiring bug that killed the first resubmit: reading
    `solver.optimizer` right after Solver() raises AttributeError, because
    physicsnemo creates it inside _train_loop. The repair wraps load_network,
    which runs after add_param_group and before the first scheduler step.
    """
    from lpno.custom_utils.utils import attach_late_param_group_repair

    class FakeSolver:
        def load_network(self):
            # by the time physicsnemo calls this, the optimizer exists and the
            # aggregator group has been added (trainer.py:499/525/563)
            opt, _ = _setup()
            self.optimizer = opt
            return 0

    solver = FakeSolver()
    attach_late_param_group_repair(solver)   # must not touch .optimizer yet
    assert not hasattr(solver, "optimizer")
    step = solver.load_network()
    assert step == 0
    assert "initial_lr" in solver.optimizer.param_groups[1], "late group not repaired"
