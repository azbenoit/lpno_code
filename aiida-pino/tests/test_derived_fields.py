"""Precomputed derived fields: stored in AiiDA, seeded into the loader cache.

The loader's persistent conduction cache is per-host and outside provenance;
these tests pin the storage/seeding contract that closes that gap:

- stored files round-trip byte-identically through SinglefileData;
- seeding materialises exactly the missing files and never rewrites existing
  ones (a training run may hold one open);
- the seeding entry point used by the automatic staging hook tolerates sims
  without the extra (every pre-existing dataset).
"""

from __future__ import annotations

import numpy as np
import pytest

from aiida import orm

from aiida_pino.utils.derived_fields import (
    EXTRA_KEY,
    seed_local_cache_for_sims,
)


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    d = tmp_path / "conduction_cache"
    monkeypatch.setenv("PINO_CONDUCTION_CACHE", str(d))
    return d


def _stored_field(tmp_path, name="conduction_abc.npy", value=7):
    payload = np.full((3, 4), value, dtype=np.float16)
    src = tmp_path / name
    np.save(src, payload)
    node = orm.SinglefileData(file=str(src), filename=name)
    node.store()
    return node, payload


def _sim_with_entry(node, filename):
    sim = orm.Data()
    sim.store()
    sim.base.extras.set(EXTRA_KEY, [{"file": filename, "uuid": node.uuid}])
    return sim


def test_seeding_materialises_the_stored_bytes(cache_dir, tmp_path):
    node, payload = _stored_field(tmp_path)
    sim = _sim_with_entry(node, "conduction_abc.npy")
    assert seed_local_cache_for_sims([sim]) == 1
    out = np.load(cache_dir / "conduction_abc.npy")
    assert out.dtype == np.float16
    assert np.array_equal(out, payload)


def test_seeding_never_rewrites_an_existing_file(cache_dir, tmp_path):
    node, _ = _stored_field(tmp_path)
    sim = _sim_with_entry(node, "conduction_abc.npy")
    cache_dir.mkdir(parents=True)
    marker = b"do-not-touch"
    (cache_dir / "conduction_abc.npy").write_bytes(marker)
    assert seed_local_cache_for_sims([sim]) == 0
    assert (cache_dir / "conduction_abc.npy").read_bytes() == marker


def test_sims_without_the_extra_are_a_noop(cache_dir):
    sim = orm.Data()
    sim.store()
    assert seed_local_cache_for_sims([sim]) == 0


def test_seeding_is_idempotent(cache_dir, tmp_path):
    node, _ = _stored_field(tmp_path)
    sim = _sim_with_entry(node, "conduction_abc.npy")
    assert seed_local_cache_for_sims([sim]) == 1
    assert seed_local_cache_for_sims([sim]) == 0


def test_compressed_store_round_trips_through_the_seeder(cache_dir, tmp_path):
    """npz-stored fields must seed back as byte-identical raw .npy."""
    import io

    payload = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    buf = io.BytesIO()
    np.savez_compressed(buf, field=payload)
    buf.seek(0)
    node = orm.SinglefileData(file=buf, filename="conduction_xyz.npy.npz")
    node.store()
    sim = orm.Data(); sim.store()
    sim.base.extras.set(EXTRA_KEY, [{"file": "conduction_xyz.npy", "uuid": node.uuid,
                                     "stored_format": "npz"}])
    assert seed_local_cache_for_sims([sim]) == 1
    out = np.load(cache_dir / "conduction_xyz.npy")
    assert out.dtype == np.float16 and np.array_equal(out, payload)
