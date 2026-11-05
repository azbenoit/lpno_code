"""Unit tests for the autoregressive delta / residual-in-time wrapper.

DeltaArch reparametrizes an autoregressive model as ``u_{t+1} = u_t + f(x)``:
for each fed-back state key it adds the previous field (``*_prev`` input, present
in AR mode) to the base network output, so the network predicts only the per-step
increment. See rigno_arch.DeltaArch.
"""
import torch
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Arch

from lpno.models.rigno_arch import DeltaArch


class _ConstBase(Arch):
    """Minimal base Arch returning a constant field for each output key."""

    def __init__(self, input_keys, output_keys, const):
        super().__init__(input_keys=input_keys, output_keys=output_keys)
        self._const = float(const)

    def forward(self, in_vars):
        ref = next(iter(in_vars.values()))
        return {name: torch.full_like(ref, self._const) for name in self.output_key_dict}


def test_delta_adds_prev_to_output():
    base = _ConstBase([Key("T_prev"), Key("beam_x")], [Key("T")], const=2.0)
    wrapped = DeltaArch(base, state_keys=["T"], state_prev_suffix="_prev")
    out = wrapped.forward({"T_prev": torch.ones(1, 1, 4), "beam_x": torch.zeros(1, 1, 4)})
    # u_{t+1} = u_t (prev=1) + f (const=2) = 3
    assert torch.allclose(out["T"], torch.full((1, 1, 4), 3.0)), out["T"]


def test_delta_only_applies_to_keys_with_a_prev_input():
    # alpha has no alpha_prev input -> passes through; T gets the delta.
    base = _ConstBase([Key("T_prev"), Key("x")], [Key("T"), Key("alpha")], const=5.0)
    wrapped = DeltaArch(base, state_keys=["T", "alpha"], state_prev_suffix="_prev")
    assert wrapped._delta_keys == ["T"]
    out = wrapped.forward({"T_prev": torch.ones(1, 1, 2), "x": torch.zeros(1, 1, 2)})
    assert torch.allclose(out["T"], torch.full((1, 1, 2), 6.0))       # 5 + 1
    assert torch.allclose(out["alpha"], torch.full((1, 1, 2), 5.0))   # unchanged


def test_delta_requires_at_least_one_prev_input():
    base = _ConstBase([Key("x")], [Key("T")], const=1.0)
    try:
        DeltaArch(base, state_keys=["T"], state_prev_suffix="_prev")
    except ValueError:
        return
    raise AssertionError("DeltaArch must reject a config with no *_prev inputs")


def test_wrapper_keeps_base_params_and_output_keys():
    base = _ConstBase([Key("T_prev"), Key("x")], [Key("T")], const=0.0)
    wrapped = DeltaArch(base, state_keys=["T"], state_prev_suffix="_prev")
    assert set(wrapped.output_key_dict) == {"T"}
    assert "T_prev" in wrapped.input_key_dict
