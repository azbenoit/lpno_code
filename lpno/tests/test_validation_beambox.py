"""Beam-box error must be visible DURING training, not only at eval.

Whole-domain val MAE describes the quiescent ~99% of the substrate; a model can look fine
there while the melt region is wrong. The beam box (within 2 beam radii of the laser,
inside the metal) is the metric that tracks it, and it previously existed only on the eval
path -- so an arm's melt-region behaviour was invisible for its entire run.

The plumbing is the interesting part: `_l2_relative_error` sees the concatenated outputs
but never the inputs, while `save_results` sees the inputs but discards them. The inputs
are therefore stashed off `self.forward`, which costs no extra forward passes and no extra
data loading.
"""
import torch

from lpno.models.pipeline import validation_beambox as vb


def _stash(beam, z):
    vb._STASH.clear()
    vb._STASH["beam_dist"] = [beam]
    vb._STASH["z"] = [z]


def test_restricts_error_to_the_beam_box():
    # `beam_dist` is a peak-1 Gaussian footprint since e91a1b3 (in-box means
    # footprint >= exp(-2) ~ 0.135); these fixtures predate that and used
    # distance-in-radii values (5.0 is an impossible footprint, and the new
    # predicate reads it as dead-center). 4 cells: two inside (strong footprint,
    # z<=0), one far from the beam, one above the surface.
    beam = torch.tensor([0.9, 0.5, 0.01, 0.9])
    z = torch.tensor([-1.0, -1.0, -1.0, 3.0])   # last is above the surface
    _stash(beam, z)
    true = {"T": torch.tensor([1.0, 1.0, 1.0, 1.0])}
    pred = {"T": torch.tensor([0.0, 0.0, 9.0, 9.0])}  # huge errors OUTSIDE the box
    out = vb.beam_box_validation_metrics(true, pred)
    assert out["beambox_n"] == 2
    # only the two in-box cells count, each with error 1.0
    assert abs(float(out["beambox_mae_T"]) - 1.0) < 1e-6


def test_out_of_box_error_is_excluded_entirely():
    """The point of the metric: a model wrong only far from the beam scores clean."""
    beam = torch.tensor([0.9, 0.01])  # footprint semantics: 0.01 = far away
    z = torch.tensor([-1.0, -1.0])
    _stash(beam, z)
    true = {"T": torch.tensor([1.0, 1.0])}
    pred = {"T": torch.tensor([1.0, -50.0])}
    out = vb.beam_box_validation_metrics(true, pred)
    assert out["beambox_n"] == 1
    assert abs(float(out["beambox_mae_T"])) < 1e-6


def test_gas_above_the_surface_is_excluded():
    """z > 0 is the free gas gap; including it is the dilution this whole metric avoids."""
    beam = torch.tensor([0.5, 0.5])
    z = torch.tensor([-1.0, 5.0])
    _stash(beam, z)
    true = {"T": torch.tensor([1.0, 1.0])}
    pred = {"T": torch.tensor([1.0, 0.0])}
    out = vb.beam_box_validation_metrics(true, pred)
    assert out["beambox_n"] == 1
    assert abs(float(out["beambox_mae_T"])) < 1e-6


def test_missing_inputs_skip_silently():
    """A config with no laser-trajectory channel is legitimate, not an error."""
    vb._STASH.clear()
    assert vb.beam_box_validation_metrics({"T": torch.zeros(3)}, {"T": torch.zeros(3)}) == {}


def test_empty_box_returns_nothing_rather_than_nan():
    _stash(torch.tensor([0.0, 0.0]), torch.tensor([-1.0, -1.0]))  # footprint 0 = nowhere near the beam
    assert vb.beam_box_validation_metrics({"T": torch.zeros(2)}, {"T": torch.zeros(2)}) == {}


def test_shape_mismatch_is_skipped_not_raised():
    _stash(torch.tensor([0.5, 0.5]), torch.tensor([-1.0, -1.0]))
    out = vb.beam_box_validation_metrics({"T": torch.zeros(5)}, {"T": torch.zeros(5)})
    assert "beambox_mae_T" not in out


def test_context_manager_restores_the_originals():
    """It patches two class attributes on two classes; a leak would follow the process."""
    from physicsnemo.sym.domain.validator import GridValidator

    before_save = GridValidator.__dict__.get("save_results")
    before_l2 = GridValidator._l2_relative_error
    with vb.beam_box_validation(enabled=True):
        pass
    assert GridValidator.__dict__.get("save_results") is before_save
    assert GridValidator._l2_relative_error is before_l2


def test_disabled_is_a_no_op():
    from physicsnemo.sym.domain.validator import GridValidator

    before = GridValidator._l2_relative_error
    with vb.beam_box_validation(enabled=False):
        assert GridValidator._l2_relative_error is before


# --------------------------------------------------------------------------- #
# Device: the stash holds references to tensors the validator already computed,
# which live wherever it computed them -- on the GPU during training. `np.asarray`
# raises `TypeError: can't convert cuda:0 device type tensor to numpy` on those,
# which killed the FNO AR arm (calc 69851) at its FIRST validation, step 0.
# --------------------------------------------------------------------------- #
class _FakeCudaTensor:
    """Stands in for a CUDA tensor: refuses `np.asarray` until `.cpu()` is called.

    A real CUDA tensor cannot be built on a CPU-only test runner, and the box this
    runs on may have both GPUs busy, so the CONTRACT is tested rather than the
    hardware: array conversion must go through detach().cpu().
    """

    def __init__(self, values):
        self._values = torch.as_tensor(values)
        self.detached = False
        self.moved = False

    def detach(self):
        self.detached = True
        return self

    def cpu(self):
        self.moved = True
        return self._values

    def __array__(self, *args, **kwargs):
        raise TypeError(
            "can't convert cuda:0 device type tensor to numpy. "
            "Use Tensor.cpu() to copy the tensor to host memory first."
        )


def test_beam_box_accepts_device_resident_tensors():
    """`_beam_box_from_stash` is the frame that raised in production.

    It is called from TWO places: `beam_box_validation_metrics`, which swallows and
    returns {}, and `model_domain.save_results`, which does not -- so the same
    TypeError is invisible on one path and fatal on the other. Test the shared
    function directly, which is what both depend on.
    """
    beam = _FakeCudaTensor([0.9, 0.5, 0.01, 0.9])
    z = _FakeCudaTensor([-1.0, -1.0, -1.0, 3.0])

    box = vb._beam_box_from_stash(beam, z)

    assert box is not None, "device-resident inputs produced no beam box"
    assert beam.moved and z.moved, "conversion did not route through .cpu()"


def test_cpu_tensors_still_take_the_zero_copy_path():
    """The docstring's claim -- a CPU tensor round-trips as a view -- must hold."""
    beam = torch.tensor([0.9, 0.5, 0.01, 0.9])
    z = torch.tensor([-1.0, -1.0, -1.0, 3.0])

    assert vb._beam_box_from_stash(beam, z) is not None
