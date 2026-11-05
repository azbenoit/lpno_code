"""Architecture dispatch and construction for served checkpoints.

One definition shared by the steady adapter (`adapter_pino`) and the random-path
adapter (`adapter_randompath`), so the two cannot drift apart the way a copied kwargs
block would: which family a Hydra config declares (`model_family`, `is_fno_config`,
`is_gaot`), the GAOT constructor kwargs (`build_gaot_kwargs`) and the RIGNO /
RIGNORefined net (`build_rigno_net`).

GAOT is grid-agnostic like RIGNO: it builds its encoder/decoder neighbour graphs lazily
from whatever grid shape arrives, and its latent lattice is fixed by config rather than
by the input. So a served model needs no per-resolution adjustment, unlike the FNO path
which has to respect Nyquist caps on `fno_modes`.

Defaults mirror `GAOTArch`'s own signature rather than upstream GAOT:
  * `use_geoembed_encoder` / `use_geoembed_decoder` default OFF. The statistical
    geometry descriptors are near-degenerate on a uniform Cartesian lattice, measured
    slightly better off on this task, and turning them off avoids a silent
    `torch.linalg.eigvalsh` failure mode.
  * `clamp_fraction_outputs` defaults ON, bounding `alpha`/`fl`/`VOF` to [0, 1] with a
    straight-through gradient. Nothing else in the pipeline enforces the physical
    range, and an out-of-range `alpha` corrupts the masking that the eval and the
    serving path both apply.

Reading every value off the checkpoint's own Hydra config (with these defaults as
fallback) means a folder loads as what it actually is, not as what its name suggests.
"""
from __future__ import annotations

from typing import Any


def model_family(cfg: Any) -> str:
    """`custom.model_family` if the config has one, else "".

    Guarded: OmegaConf raises MissingMandatoryValue on a `???` node rather than
    returning a default, so a plain getattr here took down startup for every
    config whose `custom` block is not fully resolved.
    """
    try:
        return str(cfg.custom.model_family or "").strip().lower()
    except Exception:  # noqa: BLE001
        return ""


def is_fno_config(cfg: Any) -> bool:
    """True when a checkpoint is an FNO, by its own declaration.

    `custom.model_family` is what the checkpoint says it is; the arch-block
    heuristic below is only a fallback for configs predating that field. The
    calibrated FNOs ship arch = [rigno, decoder, fno] -- inert blocks
    inherited from the base config -- so the heuristic alone read them as RIGNO,
    skipped the FNO Nyquist mode cap, and then loaded an FNO into RIGNOArch.
    """
    family = model_family(cfg)
    if family:
        return family == "fno"
    return ("fno" in cfg.arch) and ("rigno" not in cfg.arch)


def is_gaot(cfg: Any) -> bool:
    """True when the config declares GAOT (family first, arch block as fallback)."""
    family = model_family(cfg)
    return family == "gaot" if family else is_gaot_config(cfg)


def build_rigno_net(*, cfg: Any, cache_dir: Any, input_keys: list, output_keys: list) -> Any:
    """Instantiate the RIGNO net a Hydra config describes: base `RIGNOArch`, or
    `RIGNORefinedArch` (RIGNO + residual FNO/U-Net refinement head, trained
    end-to-end) when the config carries `arch.refine_head` or declares
    `model_family: rigno_refined` -- every refined checkpoint on disk does both.
    RIGNO is graph-based and grid-agnostic, so it serves on whatever grid the
    caller builds. Inference only, so `base_init_ckpt=None`; the checkpoint
    carries both base and head weights."""
    from lpno.models.rigno_arch import RIGNOArch, RIGNORefinedArch

    rc = cfg.arch.rigno
    rigno_kwargs = dict(
        input_keys=input_keys,
        output_keys=output_keys,
        dimension=int(getattr(rc, "dimension", 3)),
        node_latent_size=int(getattr(rc, "node_latent_size", 64)),
        edge_latent_size=int(getattr(rc, "edge_latent_size", 64)),
        processor_steps=int(getattr(rc, "processor_steps", 6)),
        mlp_hidden_layers=int(getattr(rc, "mlp_hidden_layers", 1)),
        subsample_factor=float(getattr(rc, "subsample_factor", 8.0)),
        rmesh_levels=int(getattr(rc, "rmesh_levels", 4)),
        k_p2r=int(getattr(rc, "k_p2r", 4)),
        k_r2p=int(getattr(rc, "k_r2p", 4)),
        p_edge_masking=float(getattr(rc, "p_edge_masking", 0.5)),
        graph_seed=int(getattr(rc, "graph_seed", 0)),
        # Keys routed through FiLM rather than node features. RIGNOArch sizes
        # its encoder as (inputs - len(film_keys)), so omitting this builds a
        # net 3 features too WIDE for a FiLM checkpoint: (96, 6) against (96, 9).
        film_keys=[str(k) for k in (getattr(rc, "film_keys", []) or [])],
        cache_dir=str(cache_dir),
    )
    if model_family(cfg) == "rigno_refined" or "refine_head" in cfg.arch:
        rh = cfg.arch.refine_head
        return RIGNORefinedArch(
            head_type=str(getattr(rh, "type", "fno")),
            head_width=int(getattr(rh, "width", 24)),
            head_modes=tuple(int(m) for m in rh.modes),
            head_layers=int(getattr(rh, "layers", 4)),
            head_padding=int(getattr(rh, "padding", 0)),
            base_init_ckpt=None,
            **rigno_kwargs,
        )
    return RIGNOArch(**rigno_kwargs)


def build_gaot_kwargs(
    *,
    cfg: Any,
    cache_dir: Any,
    input_keys: list,
    output_keys: list,
) -> dict:
    """`GAOTArch` constructor kwargs read off a checkpoint's Hydra config."""
    gc_ = cfg.arch.gaot

    def g(name: str, default: Any) -> Any:
        return getattr(gc_, name, default)

    return dict(
        input_keys=input_keys,
        output_keys=output_keys,
        dimension=int(g("dimension", 3)),
        latent_tokens=tuple(int(n) for n in g("latent_tokens", (32, 20, 20))),
        patch_size=int(g("patch_size", 2)),
        lifting_channels=int(g("lifting_channels", 32)),
        hidden_size=int(g("hidden_size", 256)),
        ffn_hidden_size=int(g("ffn_hidden_size", 1024)),
        num_layers=int(g("num_layers", 5)),
        num_heads=int(g("num_heads", 8)),
        num_kv_heads=int(g("num_kv_heads", 8)),
        atten_dropout=float(g("atten_dropout", 0.0)),
        positional_embedding=str(g("positional_embedding", "absolute")),
        use_long_range_skip=bool(g("use_long_range_skip", True)),
        mlp_hidden_layers=tuple(int(n) for n in g("mlp_hidden_layers", (64, 64, 64))),
        projection_channels=int(g("projection_channels", 256)),
        neighbor_strategy=str(g("neighbor_strategy", "knn")),
        # Displacement-aware decoder kernel (77c2ff5). It widens the AGNO kernel
        # input from 2*dim to 3*dim, so omitting it builds a net 3 features too
        # narrow and load_state_dict fails on (64, 9) vs (64, 6). Off by default
        # upstream, so pre-W33 checkpoints are unaffected.
        kernel_use_displacement=bool(g("kernel_use_displacement", False)),
        k_encoder=int(g("k_encoder", 4)),
        k_decoder=int(g("k_decoder", 8)),
        gno_radius=float(g("gno_radius", 0.15)),
        max_neighbors=int(g("max_neighbors", 0)),
        # Trilinear decoder stencil (2026-11-03); "knn" for every checkpoint before it, whose configs lack the key.
        decoder_stencil=str(g("decoder_stencil", "knn")),
        decoder_kernel=str(g("decoder_kernel", "agno")),
        neighbor_metric=(None if g("neighbor_metric", None) is None
                         else [float(v) for v in g("neighbor_metric", None)]),
        use_attention=bool(g("use_attention", True)),
        attention_type=str(g("attention_type", "cosine")),
        transform_type=str(g("transform_type", "linear")),
        use_geoembed_encoder=bool(g("use_geoembed_encoder", False)),
        use_geoembed_decoder=bool(g("use_geoembed_decoder", False)),
        embedding_method=str(g("embedding_method", "statistical")),
        use_pointwise_skip=bool(g("use_pointwise_skip", False)),
        pointwise_skip_channels=int(g("pointwise_skip_channels", 64)),
        clamp_fraction_outputs=bool(g("clamp_fraction_outputs", True)),
        cache_dir=str(cache_dir),
    )


def is_gaot_config(cfg: Any) -> bool:
    """True when a checkpoint's Hydra config describes a GAOT model.

    Checked by architecture block, not by `custom.model_family` or the folder name, to
    match how the adapters already distinguish FNO from RIGNO.
    """
    try:
        return "gaot" in cfg.arch
    except Exception:
        return False
