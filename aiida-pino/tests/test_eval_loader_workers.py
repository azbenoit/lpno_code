"""The eval adapter must not force dataloader workers on.

History, kept because the wrong version of this file shipped first. The adapter
briefly forced PINO_LOADER_WORKERS=4 on the theory that evals silently serialised
while trainings did not, and that serial decompression (13.56 s/case against
~1.65 s of compute) explained multi-hour evals. An A/B on one model over the same
301 frames measured the opposite:

    4 workers: loader 144.6s (46%), forward  97.5s (31%), metrics 71.1s (23%)
    0 workers: loader 129.2s (41%), forward 113.7s (36%), metrics 71.9s (23%)

Workers made loading slower -- the eval loader reads cases in order
(shuffle=False) and a cache of 3 already gives full within-case reuse, so IPC and
tensor serialisation cost more than parallel decompression recovers. Totals
matched within noise. The real cost is frame count: ~2 h for 6016 frames of GAOT
forward over a 600k-node grid, addressed by PINO_EVAL_FRAME_STRIDE, not workers.

These tests pin the corrected behaviour so the plausible-but-wrong version does
not come back.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "aiida_pino" / "scripts" / "eval_adapter.py"


def test_eval_does_not_force_loader_workers():
    """Measured slower. Leave the library default (serial) alone."""
    text = SRC.read_text()
    assert 'setdefault("PINO_LOADER_WORKERS"' not in text
    assert 'eval_env["PINO_LOADER_WORKERS"]' not in text


def test_cache_size_is_still_set():
    """Cache 3 is what gives full within-case reuse on a sequential eval read."""
    assert 'setdefault("PINO_LAZY_CACHE", "3")' in SRC.read_text()


def test_env_falls_back_to_the_real_environment_not_none():
    """profile_environment returns None for a profile with no env mapping; we need a
    real dict to set defaults on, so the fallback must copy os.environ, not {}."""
    assert "profile_environment(profile) or dict(os.environ)" in SRC.read_text()


def test_the_subprocess_receives_the_augmented_env():
    # NB: a [^)]* regex fails here -- Path.cwd() contains a closing paren.
    assert "run_streamed_subprocess(command, cwd=Path.cwd(), env=eval_env)" in SRC.read_text()


def test_the_measured_ab_is_recorded_next_to_the_code():
    """The next person to 'fix' this needs the numbers, not just the conclusion."""
    text = SRC.read_text()
    assert "144.6s" in text and "129.2s" in text
