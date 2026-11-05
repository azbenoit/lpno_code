"""Per-machine VRAM keys (item D) and eviction that knows what it cannot see (F).

D: a measured VRAM peak describes one architecture on one data layout on ONE
   GPU. TITAN RTX (Turing, sm_75) and RTX 4090 (Ada, sm_89) are different
   silicon, so pooling their peaks under one key sizes a booking from the wrong
   machine. Local keys must not move, or every peak already recorded becomes
   unreachable and the ledger re-measures from scratch.

F: `_live_workdir_targets` pins a cache entry by walking a live job's workdir
   with `Path.is_dir()` and `iterdir()`. Those are LOCAL operations. For a job on
   a remote computer the path belongs to that box -- and on a box sharing
   /home/<user> with the primary it can EXIST here, belonging to something else
   entirely, so the walk does not merely fail to pin, it reads a stranger's
   directory. This module has a body count (seven runs lost to a cache cleared
   mid-flight), so the change must only ever skip remote jobs, never narrow what
   a local job pins.
"""

from __future__ import annotations

from aiida_pino.utils.vram import vram_footprint_key

PROFILE = {"config_name": "NS_energy_PINO", "hydra_overrides": ["amp.enabled=false"], "env": {}}
LAYOUT = {"compatibility_signature": "sig-1", "canonical_layout": "xyzt", "time_steps_max": 100}


def test_a_local_key_is_unchanged_by_the_existence_of_remote_keys():
    """Byte-identical to the pre-remote key, or every recorded peak is orphaned."""
    baseline = vram_footprint_key(PROFILE, LAYOUT)
    assert vram_footprint_key(PROFILE, LAYOUT, computer_label=None) == baseline
    assert vram_footprint_key(PROFILE, LAYOUT, computer_label="local") == baseline


def test_a_remote_computer_gets_its_own_key():
    """Different silicon, different peak: sm_75 must not size an sm_89 booking."""
    local = vram_footprint_key(PROFILE, LAYOUT)
    remote = vram_footprint_key(PROFILE, LAYOUT, computer_label="gpu-box")
    assert remote != local


def test_two_remote_machines_do_not_share_a_key():
    a = vram_footprint_key(PROFILE, LAYOUT, computer_label="box-a")
    b = vram_footprint_key(PROFILE, LAYOUT, computer_label="box-b")
    assert a != b


def test_the_role_discriminator_still_applies_per_machine():
    train = vram_footprint_key(PROFILE, LAYOUT, computer_label="box-a")
    eval_ = vram_footprint_key(PROFILE, LAYOUT, role="eval", computer_label="box-a")
    assert train != eval_


def test_a_remote_workdir_is_never_walked_as_a_local_path(tmp_path, monkeypatch):
    """The same path can exist here and mean something else entirely."""
    from aiida_pino.utils import runtime_cache_evict

    assert runtime_cache_evict.workdir_is_readable_here("core.local") is True
    assert runtime_cache_evict.workdir_is_readable_here("core.ssh_async") is False
    assert runtime_cache_evict.workdir_is_readable_here("core.ssh") is False


def test_an_unknown_transport_is_treated_as_unreadable_here():
    """Fail safe: an unrecognised transport must not be walked locally."""
    from aiida_pino.utils import runtime_cache_evict

    assert runtime_cache_evict.workdir_is_readable_here("") is False
    assert runtime_cache_evict.workdir_is_readable_here(None) is False
