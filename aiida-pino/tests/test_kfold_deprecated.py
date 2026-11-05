"""K-fold submission is deprecated from the console (2026-08-19).

Not removed — the machinery works. `KfoldTrainEvalWorkChain`, its CalcJob,
parser and adapter all stay, and a deliberate submission through the workchain
is still possible. What is withdrawn is the CONSOLE offering it as a
one-click path, because it is not bug-tested to the standard the other
submission paths now are: it had no split guard at all until 2026-08-19 (it
would train k folds on a dataset reserved for evaluation, silently), which is
the kind of defect that only surfaces under review.

The refusal must therefore say "deprecated, not broken", name where the
capability went, and keep the workchain reachable — otherwise the next person
concludes the feature was deleted and rebuilds it.
"""

from __future__ import annotations



def test_the_workchain_itself_is_untouched():
    """Deprecating a UI path must not remove the capability."""
    from aiida_pino.workflows.kfold_train_eval import KfoldTrainEvalWorkChain

    assert hasattr(KfoldTrainEvalWorkChain, "validate_inputs")
    # The split guard added the same day stays in force for direct submissions.
    import inspect

    source = inspect.getsource(KfoldTrainEvalWorkChain.validate_inputs)
    assert "is_evaluation_only_split_set" in source
