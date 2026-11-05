"""TrainModelWorkChain refuses to train on test/excluded members or overlapping splits.

Regression for 2026-08-11: one unfiltered 229-member dataset was passed as both
train_dataset and val_dataset, and four arms trained on their own 29-case test
split. The eval-side guard (allow_training_data, exit 305) caught it 11.6
GPU-hours late; split_violation_message is the train-side twin, checked at
validate_inputs before any GPU time is spent.
"""
from aiida_pino.workflows.train_model import split_violation_message


def _uuids(n, prefix="u"):
    return [f"{prefix}{i}" for i in range(n)]


def test_clean_disjoint_splits_pass():
    train, val = _uuids(5, "t"), _uuids(3, "v")
    splits = {u: "train" for u in train} | {u: "val" for u in val}
    assert split_violation_message(train, val, splits) is None


def test_unlabeled_members_are_allowed():
    """Legacy collections carry no split extras; positive evidence only."""
    assert split_violation_message(_uuids(5, "t"), _uuids(3, "v"), {}) is None


def test_identical_train_and_val_refused():
    """The exact 2026-08-11 submission: same node passed twice."""
    members = _uuids(10)
    msg = split_violation_message(members, members, {})
    assert msg is not None and "identical sets" in msg


def test_partial_overlap_refused():
    train, val = _uuids(5), _uuids(5)[3:] + ["x1", "x2"]
    msg = split_violation_message(train, val, {})
    assert msg is not None and "share 2" in msg


def test_test_members_in_train_refused():
    train = _uuids(5)
    splits = {train[0]: "test", train[1]: "test", train[2]: "train"}
    msg = split_violation_message(train, ["v0"], splits)
    assert msg is not None and "split='test'" in msg and "2 " in msg


def test_excluded_members_in_train_refused_by_suffix():
    train = _uuids(3)
    msg = split_violation_message(train, ["v0"], {train[0]: "duplicate_excluded"})
    assert msg is not None and "duplicate_excluded" in msg


def test_test_members_in_val_are_not_the_train_guards_business():
    """Scoring val on test-labeled members is the EVAL guard's decision."""
    val = _uuids(3, "v")
    assert split_violation_message(_uuids(2, "t"), val, {val[0]: "test"}) is None


def test_message_names_the_override():
    members = _uuids(4)
    msg = split_violation_message(members, members, {members[0]: "test"})
    assert "allow_split_violation" in msg
