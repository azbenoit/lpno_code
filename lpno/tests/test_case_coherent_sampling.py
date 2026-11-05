"""Case-coherent streaming in LazySampleDataset.

Regression origin (2026-08-06): a global frame shuffle over 120 varspeed cases with a
1-case LRU decompressed a whole ~600 MB case (5.9 s, 1.9 GB raw) for every 19 MB frame
served -- a ~99% miss rate that starved full-resolution training to 14 s/iter and
OOM-killed 4-worker loaders, each worker privately re-decompressing cases. Five failed
runs and two wrong published causes later, the access pattern was the root.

The fix is two-level shuffling. These tests pin the properties that make it a FIX
rather than a sampling bug:

  * every entry is served exactly once per epoch (a permutation, not sampling with
    replacement);
  * the order is case-blocked, so a 1-case cache misses exactly once per case;
  * workers shard BY CASE, so no two workers hold the same case decompressed;
  * epochs reshuffle (different permutations), deterministically from the seed;
  * the default (case_coherent=False) is byte-identical to the old behaviour.
"""
import numpy as np
import pytest

from lpno.models.pipeline.lazy_runtime import LazySampleDataset


class _CountingStore:
    """Fake store that counts case 'decompressions' like the LRU would see them."""

    def __init__(self):
        self.loads = []


def _make(entries, case_coherent, seed=42):
    store = _CountingStore()

    def builder(case_idx, time_idx):
        store.loads.append(case_idx)
        arr = np.zeros((1, 2, 2, 2), dtype=np.float32)
        return {"x": arr}, {"T": arr}

    ds = LazySampleDataset(
        store=store, entries=entries, sample_builder=builder,
        invar_keys=["x"], outvar_keys=["T"],
        case_coherent=case_coherent, seed=seed,
    )
    return ds, store


def _entries(n_cases=5, frames=8):
    return [(c, t) for c in range(n_cases) for t in range(frames)]


def _served_entries(ds, n):
    served = []
    orig_builder = ds.sample_builder

    def spy(case_idx, time_idx):
        served.append((case_idx, time_idx))
        return orig_builder(case_idx=case_idx, time_idx=time_idx)

    ds.sample_builder = spy
    for i in range(n):
        ds[i % len(ds)]
    ds.sample_builder = orig_builder
    return served


def test_every_entry_served_exactly_once_per_epoch():
    entries = _entries()
    ds, _ = _make(entries, case_coherent=True)
    served = _served_entries(ds, len(entries))
    assert sorted(served) == sorted(entries), "coherent mode must be a permutation"


def test_order_is_case_blocked():
    """A 1-case LRU must miss exactly n_cases times per epoch, not ~n_entries."""
    entries = _entries(n_cases=6, frames=10)
    ds, _ = _make(entries, case_coherent=True)
    served = _served_entries(ds, len(entries))
    switches = sum(1 for a, b in zip(served, served[1:]) if a[0] != b[0])
    assert switches == 5, f"expected 5 case switches (6 blocks), got {switches}"


def test_frames_are_shuffled_within_a_case():
    entries = _entries(n_cases=2, frames=32)
    ds, _ = _make(entries, case_coherent=True)
    served = _served_entries(ds, len(entries))
    first_block = [t for c, t in served if c == served[0][0]]
    assert first_block != sorted(first_block), "frames within a case must be shuffled"


def test_epochs_reshuffle_deterministically():
    entries = _entries()
    ds1, _ = _make(entries, case_coherent=True, seed=7)
    ds2, _ = _make(entries, case_coherent=True, seed=7)
    epoch1_a = _served_entries(ds1, len(entries))
    epoch2_a = _served_entries(ds1, len(entries))
    epoch1_b = _served_entries(ds2, len(entries))
    assert epoch1_a == epoch1_b, "same seed must give the same stream"
    assert epoch1_a != epoch2_a, "successive epochs must reshuffle"
    assert sorted(epoch2_a) == sorted(entries), "epoch 2 is still a permutation"


def test_workers_shard_by_case_not_by_index():
    """No case may appear in two workers' streams -- per-worker duplicate
    decompression is what made 4 workers OOM."""
    entries = _entries(n_cases=7, frames=4)
    streams = {}
    for wid in range(3):
        ds, _ = _make(entries, case_coherent=True)
        ds._worker_id, ds._num_workers = wid, 3
        ds._rebuild_stream()
        streams[wid] = {entries[i][0] for i in ds._stream}
    for a in range(3):
        for b in range(a + 1, 3):
            assert not (streams[a] & streams[b]), f"workers {a} and {b} share cases"
    assert streams[0] | streams[1] | streams[2] == set(range(7))


def test_default_mode_is_unchanged():
    """case_coherent=False must serve exactly the indexed entry -- old behaviour."""
    entries = _entries(n_cases=3, frames=4)
    ds, _ = _make(entries, case_coherent=False)
    served = _served_entries(ds, 5)
    assert served == [entries[i % len(entries)] for i in range(5)]
