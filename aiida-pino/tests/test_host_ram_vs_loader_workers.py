"""A host-RAM declaration must cover the profile's loader settings.

Regression origin (2026-08-05): `gaot_varspeed_multitau_v2` cloned a working RIGNO
profile, correctly dropped its `PINO_LAZY_STRIDE=2,2,1` (GAOT runs full resolution),
and kept its `PINO_LOADER_WORKERS=4` and `host_ram_mb=32768`. Those two were only safe
BECAUSE of the stride: full resolution costs ~4x per cached case, each worker holds its
own cache, and the kernel OOM-killed a DataLoader worker at the first validation pass.

The gate books DECLARED host RAM, so under-declaring is invisible rather than tight --
the job is admitted and then exceeds its claim, endangering co-tenants. These tests pin
the three real configurations so the combination cannot come back.
"""
import pytest

from aiida_pino.utils.profiles import validate_training_profile

BASE = {
    'config_name': 'config_flow3D_GAOT_varspeed_multitau',
    'run_name': 'x',
    'cuda_device': 'auto',
    'wandb_offline': True,
    'hydra_overrides': [],
}


def _profile(**over):
    prof = dict(BASE)
    prof.update(over)
    return prof


def _superseded__proven_config_accepted():
    """4 workers at full resolution with 32 GB is PROVEN to work.

    gaot_cond_disp_tail (calc 30444) trained 20k steps on the same 150x100x40 grid with
    exactly these settings at 1911 ms/iter. An earlier version of the guard demanded
    49152 and rejected it -- a false positive that blocked the only configuration with
    direct evidence behind it, because the threshold was derived from an OOM whose real
    cause (two concurrent graph builds) the guard cannot see.
    """
    out = validate_training_profile(_profile(
        host_ram_mb=32768, env={'PINO_LOADER_WORKERS': '4'}))
    assert out['host_ram_mb'] == 32768


def test_a_grossly_undersized_declaration_is_still_rejected():
    """The guard still catches the real under-declaration case: 8 workers on 16 GB."""
    with pytest.raises(ValueError, match='too small for PINO_LOADER_WORKERS=8'):
        validate_training_profile(_profile(
            host_ram_mb=16384, env={'PINO_LOADER_WORKERS': '8'}))


def test_serial_loading_needs_no_extra_declaration():
    """v4's fix: workers=0, so the main process holds one cache and 16 GB is fine.

    workers=0 is a valid DECLARATION (one cache, small booking) and that is all this
    asserts. It is NOT the recommended resolution of a rejection: that advice cited a
    2026-08-05 measurement taken before `case_coherent_sampling` existed, so it compared
    workers-without-coherence -- the mode that duplicates whole-case decompression per
    worker. The standing recipe is coherence + workers=2 + host_ram_mb=40960 (0.67 s/step
    against 4.70 s/step, measured 2026-08-08). Corrected 2026-08-26 after the guard steered
    one submission into both documented failure modes in sequence.
    """
    out = validate_training_profile(_profile(
        host_ram_mb=16384,
        env={'PINO_LOADER_WORKERS': '0', 'PINO_LAZY_CACHE': '1'},
    ))
    assert out['host_ram_mb'] == 16384


def test_the_donor_rigno_profile_still_validates():
    """The profile v2 was cloned FROM must stay legal: strided grids are ~4x cheaper,
    so 4 workers at 32 GB is fine when the stride is present."""
    out = validate_training_profile(_profile(
        config_name='config_flow3D_RIGNO_randompath_onecycle',
        host_ram_mb=32768,
        env={'PINO_LOADER_WORKERS': '4', 'PINO_LAZY_STRIDE': '2,2,1'},
    ))
    assert out['host_ram_mb'] == 32768


def test_full_resolution_with_workers_passes_when_declared_honestly():
    out = validate_training_profile(_profile(
        host_ram_mb=49152, env={'PINO_LOADER_WORKERS': '4'}))
    assert out['host_ram_mb'] == 49152


def test_identity_stride_counts_as_full_resolution():
    """`1,1,1` is not a real stride and must not buy the cheaper allowance.

    Checked at 8 workers, because 4 workers at full resolution is now legal (32768 is
    the proven declaration) and would not discriminate.
    """
    validate_training_profile(_profile(          # strided: 8 workers fit in 32 GB
        host_ram_mb=32768,
        env={'PINO_LOADER_WORKERS': '8', 'PINO_LAZY_STRIDE': '2,2,1'}))
    with pytest.raises(ValueError):              # identity stride must not
        validate_training_profile(_profile(
            host_ram_mb=32768,
            env={'PINO_LOADER_WORKERS': '8', 'PINO_LAZY_STRIDE': '1,1,1'},
        ))


def test_no_declaration_is_left_alone():
    """Nothing booked, nothing to check -- the gate skips undeclared host RAM."""
    out = validate_training_profile(_profile(env={'PINO_LOADER_WORKERS': '4'}))
    assert not out.get('host_ram_mb')


def test_full_resolution_declaration_must_clear_the_enforced_cgroup_cap():
    """gpu_spool turns the declaration into a MemoryMax at 1.5x (BUG-43), so a
    full-resolution run must declare at least peak/1.5.

    A full-res GAOT graph build peaks at ~57 GB (measured VmHWM, 600k nodes / 7.2M
    edges). Declaring 32768 yields a 48 GB ceiling and the kernel kills the run at graph
    construction -- which happened to the varspeed multi-tau arm four times. It never
    named memory: gpu_spool's seat propagates the child's -15 and Python masks it to
    exit status 241, so it presented as an unexplained SIGTERM four hours after
    submission (the delay being the seat queue, not the cause).

    This supersedes the earlier test asserting 32768 is fine at 4 workers: that was
    calibrated on a run which never reached this peak.
    """
    with pytest.raises(ValueError):
        validate_training_profile(_profile(
            host_ram_mb=32768, env={'PINO_LOADER_WORKERS': '4'}))
    out = validate_training_profile(_profile(
        host_ram_mb=49152, env={'PINO_LOADER_WORKERS': '4'}))
    assert out['host_ram_mb'] == 49152


def test_strided_runs_keep_the_cheaper_allowance():
    """The cap floor is full-resolution only -- a strided grid is ~4x smaller and its
    build never approaches 57 GB, so RIGNO's 32768 stays legal."""
    out = validate_training_profile(_profile(
        host_ram_mb=32768,
        env={'PINO_LOADER_WORKERS': '4', 'PINO_LAZY_STRIDE': '2,2,1'}))
    assert out['host_ram_mb'] == 32768


def test_launch_check_does_not_block_reading_a_finished_profile():
    """A pre-launch resource check must not veto reading a training that already ran.

    Origin (2026-08-21): the beam-box eval of one random-path model failed twice with exit 300.
    The refusal came from the model's own FROZEN training profile -- host_ram_mb=32768
    with PINO_LOADER_WORKERS=4 -- being pushed through this launch-time check by
    `merge_model_training_with_evaluation_profile`. That training had already completed
    successfully; the floor it was measured against is calibrated for a 600k-node GAOT
    graph build that a random-path model never performs. No evaluation-side override
    could clear it, because the check runs on the frozen profile BEFORE the merge.

    The asymmetry is the point, so both halves are pinned here: launching this profile
    is still refused, reading it is not.
    """
    frozen = _profile(host_ram_mb=32768, env={'PINO_LOADER_WORKERS': '4'})

    with pytest.raises(ValueError, match='host_ram_mb'):
        validate_training_profile(dict(frozen), expected_split_mode='standard')

    ok = validate_training_profile(
        dict(frozen), expected_split_mode='standard', for_launch=False
    )
    assert ok['host_ram_mb'] == 32768, 'reading must not silently rewrite the declaration'


def test_a_pinned_gpu_is_announced_at_launch(capsys):
    """Three arms lost ~7 hours on 2026-08-21 to a cuda_device inherited from a donor profile.

    Each looked alive in every process listing and never logged a step, because a pinned job can
    only wait for its own card. A warning rather than a refusal: pinning one card on purpose is
    legitimate, doing it by accident is what cost the time.
    """
    validate_training_profile(_profile(cuda_device='1'), expected_split_mode='standard')
    assert 'pins this run to ONE gpu' in capsys.readouterr().out

    validate_training_profile(_profile(cuda_device='auto'), expected_split_mode='standard')
    assert 'pins this run' not in capsys.readouterr().out


def test_reading_a_finished_profile_does_not_warn(capsys):
    """for_launch=False is a read; a pin in history is not actionable."""
    validate_training_profile(
        _profile(cuda_device='1'), expected_split_mode='standard', for_launch=False
    )
    assert capsys.readouterr().out == ''
