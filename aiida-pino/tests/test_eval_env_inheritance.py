"""An evaluation must inherit the training env, and the silent case is why.

2026-08-27, found on one evaluation run. The read-time demotion
wrote the two demoted keys onto `ctx.evaluation_profile["env"]` BEFORE
`merge_model_training_with_evaluation_profile` ran. That merge deep-copies the model's
TRAINING profile and lets the evaluation profile override it, and it never merges `env`
key-by-key -- so an evaluation profile carrying an `env` replaced the inherited one entirely.
Every `train_eval` then evaluated with `env: null`.

Half of that is loud: `PINO_MATERIAL_OVERLAY` going missing raises, because `_overlay_value`
refuses to default rather than broadcast one constant to every case. The other half is
silent. `PINO_LAZY_MOVING_REF` going missing scores moving-reference weights on lab-frame
data and returns a plausible number that reads as a model failure.
`PINO_LAZY_STRIDE` and `PINO_LAZY_CROP_Z_TOP` have the same
shape: they change what the eval READS without saying so.

So the property is a superset, not equality: an eval may ADD keys (`PINO_EVAL_FRAME_STRIDE`
is added deliberately, and the demoted transforms describe how the dataset was staged), but
it may never DROP one the training relied on.
"""

from __future__ import annotations

def test_the_workchain_holds_the_demoted_keys_aside():
    """Structural: the fix is that these never reach `ctx.evaluation_profile["env"]`."""
    from pathlib import Path

    import aiida_pino

    src = (Path(aiida_pino.__file__).parent / "workflows" / "evaluate_model.py").read_text()
    assert 'self.ctx.demoted_env = {' in src
    assert '_ep["env"] = _env' not in src, "writing the demoted env onto the eval profile is the bug"
    # and they must be applied after the merge, not before it
    assert src.index("merge_model_training_with_evaluation_profile(") < src.index('merged_profile["env"] = env')
