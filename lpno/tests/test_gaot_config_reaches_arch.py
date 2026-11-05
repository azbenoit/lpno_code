"""Every `arch.gaot` key in a shipped config must actually be passed to GAOTArch.

A config knob that Hydra resolves but the dispatch never reads is silent: the run
proceeds, the value appears in the job's `.hydra/config.yaml`, and nothing happens.

That is exactly how `output_bounds` failed on 2026-07-30. The bound was added to
`GAOTArch`, declared in both material configs, covered by unit tests on the arch and
by a test that the configs declare it -- and it still had no effect, because
`model_domain.py` never forwarded it. Eval 23560 carried
`output_bounds: {T: [0.0, 1.3]}` in its resolved config and still reported
`T.pred_max = 21804.6`.
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / "src" / "lpno"
CONF = ROOT / "conf" / "hydra"
DOMAIN = ROOT / "models" / "pipeline" / "model_domain.py"

# Rewritten by config_loading from use_2D rather than read through the dispatch helper.
_REWRITTEN = {"dimension", "latent_tokens"}


def _gaot_branch() -> str:
    src = DOMAIN.read_text()
    start = src.index('elif model_family == "gaot"')
    return src[start : src.index("\n    elif ", start + 1) if "\n    elif " in src[start:] else len(src)]


def _config_gaot_keys() -> dict[str, set[str]]:
    out = {}
    for path in sorted(CONF.glob("config_flow3D_GAOT*.yaml")):
        cfg = yaml.safe_load(path.read_text()) or {}
        keys = set(((cfg.get("arch") or {}).get("gaot") or {}).keys())
        if keys:
            out[path.name] = keys
    return out


def test_some_gaot_configs_exist():
    assert _config_gaot_keys(), "no GAOT configs found; the guard below would be vacuous"


def test_every_config_key_is_read_by_the_dispatch():
    branch = _gaot_branch()
    unwired = {}
    for name, keys in _config_gaot_keys().items():
        missing = sorted(k for k in keys - _REWRITTEN if f'"{k}"' not in branch)
        if missing:
            unwired[name] = missing
    assert not unwired, (
        "these arch.gaot config keys are never read by model_domain's gaot branch, so "
        f"setting them does nothing: {unwired}"
    )


def test_output_bounds_specifically_is_forwarded():
    """The regression that motivated this file."""
    assert '"output_bounds"' in _gaot_branch()
