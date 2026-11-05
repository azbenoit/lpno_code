"""A measured VRAM peak must not size a run whose config has changed since.

2026-08-23. `vram_footprint_key` covers `config_name`,
`hydra_overrides`, the memory-relevant env and the dataset signature -- never the CONTENTS
of the named config. Raising `arch.rigno.processor_steps` 8 -> 18 and
`num_graph_variants` 1 -> 8 INSIDE the config file left the key byte-identical, so the
lookup returned a depth-8 peak to size a depth-18 job.

Two things make it worse than an ordinary mis-size. The number is presented as
"auto-sized from measured peak", so nobody re-checks it; and under-booking is the
direction that costs a card AND the result, because the job is admitted and then OOMs.

The key's own `hydra_overrides` comment anticipates this exact hazard -- "a peak measured
amp-on must never size an amp-off reservation" -- but a config-file edit reaches the same
capacity knobs while bypassing it.
"""

from __future__ import annotations

import pytest

from aiida_pino.utils.vram import (
    VRAM_CONFIG_DIGEST_EXTRA,
    config_capacity_digest,
    vram_footprint_key,
)


@pytest.mark.requires_lpno
def test_the_digest_is_stable_and_short():
    a = config_capacity_digest("config_flow3D_RIGNO_pvlhs")
    assert a and len(a) == 12 and a == config_capacity_digest("config_flow3D_RIGNO_pvlhs")


@pytest.mark.requires_lpno
def test_different_configs_digest_differently():
    assert config_capacity_digest("config_flow3D_RIGNO_pvlhs") != config_capacity_digest(
        "config_NS_energy_PINO_flow3D"
    )


def test_an_unreadable_config_is_None_not_a_fake_digest():
    """None must not collide with a real digest, or it would silently pass the check."""
    assert config_capacity_digest("no_such_config_exists") is None
    assert config_capacity_digest(None) is None


def test_editing_the_config_changes_the_digest_but_not_the_key(tmp_path, monkeypatch):
    """The reported case, in miniature: the key cannot see a capacity edit.

    This is the whole bug -- if the key DID change, no digest would be needed.
    """
    import lpno
    from pathlib import Path as _P

    conf = _P(lpno.__file__).parent / "conf" / "hydra"
    name = "config_flow3D_RIGNO_pvlhs"
    original = (conf / f"{name}.yaml").read_bytes()
    profile = {"config_name": name, "hydra_overrides": []}

    before_key = vram_footprint_key(profile, {})
    before_digest = config_capacity_digest(name)
    try:
        (conf / f"{name}.yaml").write_bytes(original + b"\n# capacity edit\n")
        after_key = vram_footprint_key(profile, {})
        after_digest = config_capacity_digest(name)
    finally:
        (conf / f"{name}.yaml").write_bytes(original)

    assert after_key == before_key, "the key is blind to config contents -- that IS the bug"
    assert after_digest != before_digest, "the digest must see what the key cannot"


def test_a_peak_measured_under_a_different_config_is_not_used(aiida_profile_clean, monkeypatch):
    """End of the chain: the lookup must actually drop it."""
    from aiida import orm

    from aiida_pino.utils import vram
    from aiida_pino.utils.vram import VRAM_KEY_EXTRA, VRAM_PEAK_EXTRA, lookup_vram_peak_mb

    def _stored(peak, digest):
        node = orm.CalcJobNode()
        node.set_process_state("finished")
        node.store()
        node.base.extras.set(VRAM_KEY_EXTRA, "k")
        node.base.extras.set(VRAM_PEAK_EXTRA, peak)
        if digest is not None:
            node.base.extras.set(VRAM_CONFIG_DIGEST_EXTRA, digest)
        return node

    monkeypatch.setattr(vram, "config_capacity_digest", lambda name: "current00cfg1")

    _stored(4000, "stale0000cfg0")
    assert lookup_vram_peak_mb("k", "any") is None, "a peak from a changed config must not size a run"

    _stored(9000, "current00cfg1")
    assert lookup_vram_peak_mb("k", "any") == 9000


def test_a_peak_with_no_digest_is_grandfathered(aiida_profile_clean, monkeypatch):
    """Deliberate: refusing them would re-smoke every footprint on the box at once.

    A peak recorded before this field existed cannot be judged -- there is no way to tell
    a stale one from a fine one -- and each re-measurement costs an exclusive card.
    """
    from aiida import orm

    from aiida_pino.utils import vram
    from aiida_pino.utils.vram import VRAM_KEY_EXTRA, VRAM_PEAK_EXTRA, lookup_vram_peak_mb

    node = orm.CalcJobNode()
    node.set_process_state("finished")
    node.store()
    node.base.extras.set(VRAM_KEY_EXTRA, "legacy")
    node.base.extras.set(VRAM_PEAK_EXTRA, 7000)

    monkeypatch.setattr(vram, "config_capacity_digest", lambda name: "current00cfg1")
    assert lookup_vram_peak_mb("legacy", "any") == 7000
