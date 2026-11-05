"""FNO scoring on a finer grid than training (models/pipeline/fno_superres.py) and the training-time mode guard."""
import copy

import pytest

from lpno.models.pipeline.fno_superres import (
    apply_fno_superres,
    remap_spectral_conv3d,
    scaled_padding,
    superres_factors,
)

torch = pytest.importorskip("torch")
spectral = pytest.importorskip("physicsnemo.models.layers.spectral_layers")


def test_factors_and_padding():
    assert superres_factors((45, 20, 40), (90, 40, 40)) == (2, 2, 1)
    assert superres_factors((45, 20, 40), (45, 20, 40)) is None
    with pytest.raises(ValueError):
        superres_factors((45, 20, 40), (60, 40, 40))
    with pytest.raises(ValueError):
        superres_factors((90, 40, 40), (45, 20, 40))
    assert scaled_padding(9, (2, 2, 1)) == [18, 18, 9]
    assert scaled_padding([9, 8], (2, 2, 1)) == [18, 16, 0]


def _upsample_xy(x, fx, fy):
    """Exact trigonometric interpolation of a real field with ODD x/y extents onto fx, fy times finer x/y."""
    nx, ny = x.shape[-3], x.shape[-2]
    X = torch.fft.fftn(x, dim=(-3, -2))
    Nx, Ny = nx * fx, ny * fy
    out = torch.zeros(*x.shape[:-3], Nx, Ny, x.shape[-1], dtype=X.dtype)
    hx, hy = (nx - 1) // 2, (ny - 1) // 2
    for kx in range(-hx, hx + 1):
        for ky in range(-hy, hy + 1):
            out[..., kx % Nx, ky % Ny, :] = X[..., kx % nx, ky % ny, :]
    return torch.fft.ifftn(out, dim=(-3, -2)).real * (Nx * Ny) / (nx * ny)


@pytest.mark.parametrize("m1, m2", [(4, 5), (3, 2)])
def test_remap_reproduces_the_trained_operator_on_a_finer_grid(m1, m2):
    """(4, 5) on a 9x7 trained grid overlaps in y (7 < 10), the failure seen on the Table 3 FNOs; (3, 2) does not."""
    torch.manual_seed(0)
    c, nx, ny, nz = 3, 9, 7, 6
    # SpectralConv3d allocates its spectrum as complex64 whatever the input dtype, so the comparison is float32-tight
    layer = spectral.SpectralConv3d(c, c, m1, m2, 3)
    with torch.no_grad():
        for name in ("weights1", "weights2", "weights3", "weights4"):
            getattr(layer, name).normal_()
    x = torch.randn(2, c, nx, ny, nz, dtype=torch.float64)
    y_coarse = layer(x.float()).detach()

    fine = copy.deepcopy(layer)
    remap_spectral_conv3d(fine, (nx, ny, nz), (2, 2, 1))
    y_fine = fine(_upsample_xy(x, 2, 2).float()).detach()
    err = (y_fine[..., ::2, ::2, :] - y_coarse).abs().max().item()
    assert err < 1e-4 * y_coarse.abs().max().item(), err

    # without the remap, the overlapping case goes wrong at the shared points
    if 2 * m2 > ny:
        y_plain = layer(_upsample_xy(x, 2, 2).float()).detach()
        assert (y_plain[..., ::2, ::2, :] - y_coarse).abs().max().item() > 1e-2 * y_coarse.abs().max().item()


def test_apply_scales_padding_and_skips_matching_grids():
    torch.manual_seed(0)

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pad, self.ipad = [9, 9, 9], [-9, -9, -9]
            self.spconv_layers = torch.nn.ModuleList([spectral.SpectralConv3d(2, 2, 20, 18, 8)])

    enc = Enc()
    assert apply_fno_superres(enc, (45, 20, 40), (45, 20, 40)) is None
    before = enc.spconv_layers[0].weights1.detach().clone()
    summary = apply_fno_superres(enc, (45, 20, 40), (90, 40, 40))
    assert enc.pad == [18, 18, 9] and enc.ipad == [-18, -18, -9]
    assert summary["factors"] == [2, 2, 1] and summary["weights_moved_per_layer"] > 0
    after = enc.spconv_layers[0].weights1
    # y-index 11..17 of the low block were overwritten during training on 29 padded points: replaced or zeroed now
    assert not torch.equal(after[:, :, :, 11:], before[:, :, :, 11:])
    assert torch.equal(after[:, :, :, :11], before[:, :, :, :11])


def test_real_sym_encoder_is_padded_along_the_right_axes():
    """The pino FNO arch builds physicsnemo.sym's FNO3DEncoder, whose pad list runs z, y, x. After the repair the
    spectral layers must see x and y padded by twice the trained cells and z unchanged, and the output must come back
    at the served grid."""
    sym = pytest.importorskip("physicsnemo.sym.models.fno")
    torch.manual_seed(0)
    enc = sym.FNO3DEncoder(in_channels=2, nr_fno_layers=2, fno_layer_size=4, fno_modes=[3, 4, 2], padding=[2, 3, 4])
    assert getattr(enc, "coord_features", False) in (True, False)
    seen = []
    enc.spconv_layers[0].register_forward_hook(lambda m, a, o: seen.append(tuple(a[0].shape[-3:])))
    enc(torch.zeros(1, 2, 9, 7, 6))
    trained_padded = seen[-1]                                     # learn the class's own axis order from its output
    summary = apply_fno_superres(enc, (9, 7, 6), (18, 14, 6))
    out = enc(torch.zeros(1, 2, 18, 14, 6))
    fine_padded = seen[-1]
    assert out.shape[-3:] == (18, 14, 6)
    pads_xyz = [p - n for p, n in zip(trained_padded, (9, 7, 6))]
    assert fine_padded == (18 + 2 * pads_xyz[0], 14 + 2 * pads_xyz[1], 6 + pads_xyz[2]), (trained_padded, fine_padded)
    assert summary["padding_cells_xyz"]["trained"] == pads_xyz


def test_training_guard_refuses_modes_the_strided_grid_cannot_hold(monkeypatch):
    from lpno.models.pipeline.config_loading import _refuse_fno_modes_beyond_served_grid

    monkeypatch.setenv("PINO_LAZY_STRIDE", "2,2,1")
    with pytest.raises(ValueError, match="fno-modes"):
        _refuse_fno_modes_beyond_served_grid([20, 18, 8], (90, 40, 40), 9)   # y: 20 + 9 = 29 < 36
    _refuse_fno_modes_beyond_served_grid([20, 18, 8], (150, 100, 40), 9)      # free-form: y 50 + 9 = 59, fine
    monkeypatch.delenv("PINO_LAZY_STRIDE")
    _refuse_fno_modes_beyond_served_grid([20, 18, 8], (90, 40, 40), 9)        # unstrided: y 49 >= 36
    monkeypatch.setenv("PINO_LAZY_STRIDE", "2,2,1")
    monkeypatch.setenv("PINO_ALLOW_FNO_MODE_OVERLAP", "1")
    _refuse_fno_modes_beyond_served_grid([20, 18, 8], (90, 40, 40), 9)
