import numpy as np
import pytest

from lpno.models.ns_energy_pipeline import get_splits


def test_get_splits_standard_mode_uses_full_train_and_val() -> None:
    splits = get_splits(mode="standard", n_samples=5, val_n_samples=3)

    assert len(splits) == 1
    split = splits[0]
    assert split.fold == 0
    assert np.array_equal(split.train_idx, np.arange(5))
    assert np.array_equal(split.val_idx, np.arange(3))


def test_get_splits_kfold_returns_all_folds() -> None:
    splits = get_splits(mode="kfold", n_samples=8, k_folds=4, fold_index=-1, seed=7)

    assert len(splits) == 4
    assert {split.fold for split in splits} == {0, 1, 2, 3}

    all_val_idx = np.concatenate([split.val_idx for split in splits])
    assert np.array_equal(np.sort(all_val_idx), np.arange(8))


def test_get_splits_kfold_returns_selected_fold_only() -> None:
    splits = get_splits(mode="kfold", n_samples=10, k_folds=5, fold_index=2, seed=11)

    assert len(splits) == 1
    assert splits[0].fold == 2
    assert len(splits[0].val_idx) == 2


def test_get_splits_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported split mode"):
        get_splits(mode="bad_mode", n_samples=4)
