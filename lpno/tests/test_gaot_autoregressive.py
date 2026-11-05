"""GAOT must be admissible for autoregressive rollout, like the other grid families.

`autoregressive.enabled=True` was gated to fno/rigno/deeponet. GAOT was left out
because it arrived after the gate was written, not because anything in the AR path
is architecture-specific: every stage downstream of the gate branches on
`run_cfg.data_layout` ("grid" for fno/unet/rigno/rigno_refined/gaot), never on
`model_family` --

  * pair building        preprocessing_datasets.py:1486, 1530, 1581, 1604
  * unrolled constraint  ar_constraints.UnrolledGridConstraint, a SupervisedGridConstraint
  * rollout eval         postprocessing.py:1107 (`pointwise_layout`)

and GAOTArch implements the same Arch contract as RIGNOArch: concatenated input
channels on a fixed structured grid in, output channels on that grid out. The
`*_prev` feedback inputs are ordinary extra channels to it.

Probed 2026-08-13 at the arch level on a 16x12x8 grid: k=3 unroll with feedback,
gradients reach all 44 parameter tensors both with and without `detach_grad`, and
the output moves with the fed-back state (max|dT| = 2.3e-4 between an all-zero and
an all-one `T_prev`), so the channels are read rather than ignored.
"""
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import lpno.models.pipeline.config_loading as config_loading
from lpno.models.ns_energy_pipeline import load_cfg_and_paths


@pytest.fixture(autouse=True)
def _default_data_root_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_root = tmp_path / "data_root"
    for relative_path in (
        config_loading._DEFAULT_TRAIN_DATA_SUBDIR,
        config_loading._DEFAULT_VAL_DATA_SUBDIR,
        config_loading._DEFAULT_TEST_DATA_SUBDIR,
        config_loading._DEFAULT_PARAM_DATA_SUBDIR,
    ):
        (data_root / relative_path).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PINO_SIMULATIONS_DATA_ROOT", str(data_root))
    return data_root


def _ar_cfg(tmp_path: Path, model_family: str, **ar_overrides):
    custom = {
        "name": "test",
        "software": "flow3D",
        "model_family": model_family,
        "geometry": {
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
            "z_range": [0.0, 1.0],
            "resolution": [16, 12, 8],
        },
        "ref_scales": {"L": 1.0, "T": 3000.0, "V": 3.0, "P": 4.0, "Rho": 5.0, "H": 6.0},
        "material_params": {"T_s": 1878, "T_l": 1928, "K": {"coeffs": [1.0, 2.0]}, "K_g": 0.1},
        "laser": {"x0": 0.0, "z0": 0.0},
        "vtk_fields": ["Temperature", "Fraction Of Fluid", "Pressure", "Velocity"],
        "FNO_input_keys": ["x", "y", "z"],
        "FNO_output_keys": ["T", "alpha"],
        "Energy_input_keys": ["u", "v", "w"],
        "Energy_output_keys": ["energy"],
        "VOF_input_keys": ["u", "v", "w"],
        "VOF_output_keys": ["VOF"],
        "Diff_input_keys": [],
        "Diff_output_keys": ["T_diffTerm"],
        "include_time": True,
        "time_as_batch": True,
        "infer_velocities": False,
        "inference_batch_size": 1,
        "moving_ref_frame": False,
        "moving_frame_steps": 30,
    }
    autoregressive = {"enabled": True, "mode": "rollout"}
    autoregressive.update(ar_overrides)
    return OmegaConf.create(
        {
            "custom": custom,
            "autoregressive": autoregressive,
            "network_dir": str(tmp_path / "net"),
            "run_mode": "train",
            "batch_size": {"validation": 1, "grid": 1},
            "arch": {
                "fno": {"padding": 1, "fno_modes": [3, 3, 3]},
                "gaot": {"latent_tokens": [8, 6, 4], "patch_size": 2},
                "rigno": {},
            },
        }
    )


def test_gaot_is_admitted_for_autoregressive_rollout(tmp_path: Path) -> None:
    run_cfg = load_cfg_and_paths(_ar_cfg(tmp_path, "gaot"))

    assert run_cfg.autoregressive_enabled is True
    assert run_cfg.autoregressive_mode == "rollout"
    # The gate must not silently downgrade GAOT to the direct formulation.
    assert run_cfg.data_layout == "grid"


def test_gaot_autoregressive_builds_prev_state_inputs(tmp_path: Path) -> None:
    """The *_prev feedback channels are what make the run autoregressive at all."""
    run_cfg = load_cfg_and_paths(_ar_cfg(tmp_path, "gaot"))

    suffix = run_cfg.autoregressive_state_prev_suffix
    assert run_cfg.autoregressive_state_keys == ["T", "alpha"]
    for state_key in run_cfg.autoregressive_state_keys:
        assert f"{state_key}{suffix}" in run_cfg.keys.str_fno_input_keys


def test_gaot_matches_rigno_input_keys(tmp_path: Path) -> None:
    """Same AR wiring for both grid families; a divergence here means a hidden gate."""
    gaot_cfg = load_cfg_and_paths(_ar_cfg(tmp_path, "gaot"))
    rigno_cfg = load_cfg_and_paths(_ar_cfg(tmp_path, "rigno"))

    assert gaot_cfg.keys.str_fno_input_keys == rigno_cfg.keys.str_fno_input_keys
    assert gaot_cfg.keys.str_fno_output_keys == rigno_cfg.keys.str_fno_output_keys
    assert gaot_cfg.autoregressive_context_keys == rigno_cfg.autoregressive_context_keys


def test_gaot_still_rejected_for_masked_patch(tmp_path: Path) -> None:
    """masked_patch is genuinely FNO-only (it needs the patch validator); keep it shut."""
    with pytest.raises(ValueError, match="masked_patch"):
        load_cfg_and_paths(_ar_cfg(tmp_path, "gaot", mode="masked_patch"))


def test_unsupported_family_is_still_rejected(tmp_path: Path) -> None:
    """The gate must stay a gate -- unet has no AR support."""
    with pytest.raises(ValueError, match="autoregressive"):
        load_cfg_and_paths(_ar_cfg(tmp_path, "unet"))


# --------------------------------------------------------------------------- #
# The arch-level claim the gate change rests on: GAOT trains through an unroll.
# --------------------------------------------------------------------------- #
_GRID = (16, 12, 8)
_CTX = ["P", "V_scan", "t_hat", "beam_dist"]
_STATE = ["T", "alpha"]
_PREV = "_prev"


def _tiny_gaot(cache_dir: Path):
    from physicsnemo.sym.key import Key

    from lpno.models.gaot_arch import GAOTArch

    return GAOTArch(
        input_keys=[Key(k) for k in _CTX + [s + _PREV for s in _STATE]],
        output_keys=[Key(k) for k in _STATE],
        dimension=3,
        latent_tokens=(8, 6, 4),
        patch_size=2,
        lifting_channels=16,
        hidden_size=32,
        ffn_hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        mlp_hidden_layers=(16, 16),
        projection_channels=32,
        cache_dir=str(cache_dir),
    )


@pytest.mark.parametrize("detach", [False, True])
def test_gaot_backprops_through_a_three_step_unroll(tmp_path: Path, detach: bool) -> None:
    """Mirror UnrolledGridConstraint.forward/.loss: k forwards, feedback, one backward.

    Both settings of `unroll_detach_grad` must reach every parameter with finite
    gradients, or the unrolled constraint would train a subset of the network.
    """
    import torch

    torch.manual_seed(0)
    model = _tiny_gaot(tmp_path / "graphs")
    assert model.num_inputs == len(_CTX) + len(_STATE), "prev-state channels not accepted"

    ctx = {key: torch.randn(1, 1, *_GRID) for key in _CTX}
    running = {s: torch.rand(1, 1, *_GRID) for s in _STATE}
    k = 3
    targets = [{s: torch.rand(1, 1, *_GRID) for s in _STATE} for _ in range(k)]

    loss = 0.0
    for i in range(k):
        invar = dict(ctx, **{s + _PREV: running[s] for s in _STATE})
        pred = model(invar)
        loss = loss + sum((pred[s] - targets[i][s]).pow(2).mean() for s in _STATE) / k
        running = {s: (pred[s].detach() if detach else pred[s]) for s in _STATE}
    loss.backward()

    params = list(model.parameters())
    grads = [p.grad for p in params if p.grad is not None]
    assert len(grads) == len(params), "some parameters got no gradient through the unroll"
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(g.abs().sum() > 0 for g in grads)
    assert pred["T"].shape == (1, 1, *_GRID)


def test_gaot_output_moves_with_the_fed_back_state(tmp_path: Path) -> None:
    """If the *_prev channels did not reach the output, the rollout would be a no-op."""
    import torch

    torch.manual_seed(0)
    model = _tiny_gaot(tmp_path / "graphs").eval()
    ctx = {key: torch.randn(1, 1, *_GRID) for key in _CTX}
    zeros = {s + _PREV: torch.zeros(1, 1, *_GRID) for s in _STATE}
    ones = {s + _PREV: torch.ones(1, 1, *_GRID) for s in _STATE}

    with torch.no_grad():
        delta = (model(dict(ctx, **zeros))["T"] - model(dict(ctx, **ones))["T"]).abs().max()

    assert delta > 1e-6, "output ignores the *_prev channels"


def test_gaot_trains_through_the_real_unrolled_constraint(tmp_path: Path) -> None:
    """End-to-end: the pipeline's own UnrolledGridConstraint driving a GAOT node.

    The arch test above proves GAOT backprops through an unroll; this proves the
    constraint the training path actually builds can drive it -- ride-along
    `__ar{i}` context, `__tgt_ar{i}` targets, feedback, and an optimizer step.
    """
    import numpy as np
    import torch

    from lpno.custom_utils.ar_constraints import UnrolledGridConstraint
    from lpno.models.pipeline.preprocessing_datasets import DictGridDataset

    batch, k = 2, 3
    rng = np.random.default_rng(0)
    sample = lambda: rng.random((batch, 1, *_GRID), dtype=np.float32)  # noqa: E731

    invar = {key: sample() for key in _CTX}
    for i in range(1, k):
        invar.update({f"{key}__ar{i}": sample() for key in _CTX})
        invar.update({f"{s}__tgt_ar{i}": sample() for s in _STATE})
    invar.update({f"{s}{_PREV}": sample() for s in _STATE})
    outvar = {s: sample() for s in _STATE}

    arch = _tiny_gaot(tmp_path / "graphs")
    constraint = UnrolledGridConstraint(
        nodes=[arch.make_node(name="gaot")],
        dataset=DictGridDataset(invar, outvar),
        batch_size=batch,
        context_keys=_CTX,
        state_keys=_STATE,
        state_prev_suffix=_PREV,
        unroll_steps=k,
        detach_grad=True,
    )

    optimizer = torch.optim.Adam(arch.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in arch.parameters()]
    losses = []
    for step in range(3):
        constraint.load_data()
        constraint.forward()
        loss = sum(constraint.loss(step).values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())

    assert len(constraint._unroll_preds) == k, "constraint did not keep k predictions"
    assert all(np.isfinite(losses))
    assert all(not torch.equal(a, b.detach()) for a, b in zip(before, arch.parameters()))
