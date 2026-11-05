"""Which architecture a checkpoint is built as, and the flags that change its shape.

Both bugs here were found the same way: a state_dict size mismatch at load time,
i.e. the served net was a different shape from the trained one. Neither is
catchable by inspecting a config alone -- only by building and loading.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from app.arch_build import build_gaot_kwargs

MODELS = Path(__file__).resolve().parents[1] / "data" / "random_path" / "models"


def _cfg(model_id: str):
    return yaml.safe_load((MODELS / model_id / ".hydra" / "config.yaml").read_text())


def _served_ids() -> list[str]:
    if not MODELS.is_dir():
        return []
    return sorted(d.name for d in MODELS.iterdir() if (d / ".hydra" / "config.yaml").is_file())


def test_build_gaot_kwargs_threads_kernel_use_displacement() -> None:
    """77c2ff5 widens the AGNO kernel input from 2*dim to 3*dim.

    Dropping the flag builds a net 3 features too narrow, and load_state_dict
    fails with (64, 9) vs (64, 6) -- which is exactly how gaot_newphys failed.
    """
    gaot = SimpleNamespace(kernel_use_displacement=True)
    kwargs = build_gaot_kwargs(
        cfg=SimpleNamespace(arch=SimpleNamespace(gaot=gaot)),
        cache_dir="/tmp", input_keys=[], output_keys=[],
    )
    assert kwargs["kernel_use_displacement"] is True

    off = build_gaot_kwargs(
        cfg=SimpleNamespace(arch=SimpleNamespace(gaot=SimpleNamespace())),
        cache_dir="/tmp", input_keys=[], output_keys=[],
    )
    assert off["kernel_use_displacement"] is False  # pre-W33 checkpoints unaffected


@pytest.mark.parametrize("model_id", _served_ids())
def test_every_served_model_declares_its_family(model_id: str) -> None:
    """Dispatch reads custom.model_family, so a served model must carry one.

    The arch-block heuristic it replaced ("fno in arch and rigno not in arch")
    misread the calibrated FNOs, whose configs ship arch = [rigno, decoder, fno]:
    inert blocks inherited from the base config sent an FNO checkpoint into the
    RIGNO branch.
    """
    family = str((_cfg(model_id).get("custom") or {}).get("model_family") or "").lower()
    assert family in {"fno", "rigno", "gaot", "rigno_refined"}, (
        f"{model_id} declares model_family={family!r}; dispatch would fall back to "
        "guessing from arch blocks"
    )


@pytest.mark.parametrize("model_id", _served_ids())
def test_declared_family_matches_the_served_label(model_id: str) -> None:
    """A GAOT folder must not quietly hold an FNO, and vice versa."""
    meta = json.loads((MODELS / model_id / "meta.json").read_text())
    family = str((_cfg(model_id).get("custom") or {}).get("model_family") or "").lower()
    declared = str(meta.get("type") or "").lower()
    if declared and family:
        assert family.startswith(declared) or declared.startswith(family), (
            f"{model_id}: meta.json type={declared!r} but config model_family={family!r}"
        )


def test_family_dispatch_prefers_declaration_over_arch_blocks() -> None:
    """The calibrated FNOs ship arch = [rigno, decoder, fno]; only the declared
    family keeps them out of the RIGNO branch. Without a declaration the arch
    blocks decide, as for the pre-field checkpoints."""
    from app.arch_build import is_fno_config, is_gaot, model_family

    def cfg(family, *blocks):
        custom = SimpleNamespace(model_family=family) if family is not None else SimpleNamespace()
        return SimpleNamespace(custom=custom, arch={b: SimpleNamespace() for b in blocks})

    assert model_family(cfg("FNO ", "rigno", "fno")) == "fno"
    assert model_family(cfg(None, "fno")) == ""
    assert is_fno_config(cfg("fno", "rigno", "decoder", "fno"))
    assert not is_fno_config(cfg("rigno", "rigno", "decoder", "fno"))
    assert is_fno_config(cfg(None, "decoder", "fno"))
    assert not is_fno_config(cfg(None, "rigno", "fno"))         # legacy heuristic
    assert is_gaot(cfg("gaot", "rigno", "gaot"))
    assert not is_gaot(cfg("rigno", "rigno", "gaot"))
    assert is_gaot(cfg(None, "rigno", "gaot"))                 # arch-block fallback
