"""Tests for the legacy dataset-migration helpers."""

from __future__ import annotations

import numpy as np
from aiida import orm

from aiida_pino.data import PinoDatasetCollectionData, PinoSimulationData
from aiida_pino.data.pino_dataset_collection import collection_compatibility_signature
from aiida_pino.utils.dataset_migration import (
    _legacy_group_entries,
    migrate_all_legacy_datasets,
    migrate_legacy_dataset_node,
    migrate_legacy_union_node,
)


def _simulation_count() -> int:
    builder = orm.QueryBuilder()
    builder.append(PinoSimulationData)
    return builder.count()


def _batch_arrays(batch: int, *, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "Temperature": rng.random((batch, 2, 1, 4, 4, 4)).astype(np.float32),
        "alpha": rng.random((batch, 2, 1, 4, 4, 4)).astype(np.float32),
    }


def _grouped_files(batch: int, *, prefix: str) -> list[dict]:
    return [
        {
            "group": f"{prefix}_{index}",
            "files": [
                {
                    "relative_path": f"{prefix}_{index}/t{step}.vtk",
                    "params": {"time_s": step * 1.0e-4},
                }
                for step in range(2)
            ],
        }
        for index in range(batch)
    ]


# --- _legacy_group_entries: every branch returns exactly batch_size entries ---


def test_legacy_group_entries_uses_matching_grouped_files() -> None:
    grouped = _grouped_files(3, prefix="case")
    entries = _legacy_group_entries({"grouped_files": grouped, "sample_metadata": [{"group": "x"}]}, 3)
    assert entries == grouped
    assert len(entries) == 3


def test_legacy_group_entries_uses_matching_sample_metadata() -> None:
    samples = [{"group": f"g{index}", "params": {"time_s": float(index)}} for index in range(3)]
    entries = _legacy_group_entries({"grouped_files": [], "sample_metadata": samples}, 3)
    assert len(entries) == 3
    assert [entry["group"] for entry in entries] == ["g0", "g1", "g2"]
    assert [entry["files"] for entry in entries] == [[sample] for sample in samples]


def test_legacy_group_entries_synthetic_fallback() -> None:
    entries = _legacy_group_entries({}, 4)
    assert entries == [{"group": f"simulation_{index:04d}", "files": []} for index in range(4)]


def test_legacy_group_entries_mismatched_grouped_files_falls_back_to_synthetic() -> None:
    """A grouped_files list of the wrong length is ignored, not an error.

    All three branches of ``_legacy_group_entries`` return exactly
    ``batch_size`` entries, which is what keeps the
    ``len(groups) != batch_size`` guard in ``migrate_legacy_dataset_node``
    unreachable. Pin that invariant: a mismatched grouped_files list must
    fall through to the synthetic branch rather than propagate a short list.
    """
    entries = _legacy_group_entries({"grouped_files": _grouped_files(2, prefix="case")}, 3)
    assert len(entries) == 3
    assert all(entry["files"] == [] for entry in entries)
    assert [entry["group"] for entry in entries] == [f"simulation_{index:04d}" for index in range(3)]


# --- migrate_legacy_dataset_node: base role ---


def test_migrate_base_legacy_dataset_slices_batch_axis(legacy_pino_dataset_factory) -> None:
    arrays = _batch_arrays(3, seed=1)
    legacy = legacy_pino_dataset_factory(
        "mig-base",
        arrays=arrays,
        metadata={"grouped_files": _grouped_files(3, prefix="mig-base-case")},
    )

    collection = migrate_legacy_dataset_node(legacy)

    assert isinstance(collection, PinoDatasetCollectionData)
    members = collection.iter_member_simulations()
    assert len(members) == 3
    assert collection.get_simulation_count() == 3
    for index, member in enumerate(members):
        for name, batch_array in arrays.items():
            np.testing.assert_array_equal(member.get_array_by_original_name(name), batch_array[index])
    signatures = {collection_compatibility_signature([member]) for member in members}
    assert len(signatures) == 1
    assert collection.get_compatibility_signature() == signatures.pop()
    assert collection.get_dataset_metadata()["composition_kind"] == "migration"


def test_migration_marks_legacy_node(legacy_pino_dataset_factory) -> None:
    legacy = legacy_pino_dataset_factory(
        "mig-mark",
        arrays=_batch_arrays(2, seed=2),
        metadata={"grouped_files": _grouped_files(2, prefix="mig-mark-case")},
    )

    collection = migrate_legacy_dataset_node(legacy)

    extras = legacy.base.extras
    assert extras.get("legacy_format") is True
    assert extras.get("superseded") is True
    assert extras.get("hidden_from_selectors") is True
    assert extras.get("migrated_to_uuid") == collection.uuid
    assert extras.get("migrated_to_pk") == collection.pk


def test_migration_is_idempotent(legacy_pino_dataset_factory) -> None:
    legacy = legacy_pino_dataset_factory(
        "mig-idem",
        arrays=_batch_arrays(2, seed=3),
        metadata={"grouped_files": _grouped_files(2, prefix="mig-idem-case")},
    )

    first = migrate_legacy_dataset_node(legacy)
    simulations_before = _simulation_count()
    second = migrate_legacy_dataset_node(legacy)

    assert second.uuid == first.uuid
    assert _simulation_count() == simulations_before


def test_migrate_base_legacy_dataset_with_one_sample_entry_per_simulation(legacy_pino_dataset_factory) -> None:
    """Per-simulation sample_metadata must not crash the batch-axis split.

    The factory default is the repo's canonical legacy base shape: batch 1,
    two time steps per simulation, and a sample_metadata list with a single
    file entry (one time_s value) per simulation. The migration must not
    forward that 1-entry time list to PinoSimulationData.from_arrays, which
    rejects it against the 2-step sliced array.
    """
    legacy = legacy_pino_dataset_factory("mig-sample-times")

    collection = migrate_legacy_dataset_node(legacy)

    members = collection.iter_member_simulations()
    assert len(members) == 1
    assert members[0].get_num_timesteps() == 2


# --- migrate_legacy_dataset_node: derived role ---


def test_migrate_derived_legacy_dataset_chains_to_migrated_base(legacy_pino_dataset_factory) -> None:
    base_legacy = legacy_pino_dataset_factory(
        "mig-derived-base",
        arrays=_batch_arrays(2, seed=4),
        metadata={"grouped_files": _grouped_files(2, prefix="mig-derived-case")},
    )
    derived_legacy = legacy_pino_dataset_factory(
        "mig-derived",
        role="derived",
        metadata={
            "base_dataset_uuid": base_legacy.uuid,
            "base_dataset_pk": base_legacy.pk,
            # A derivation that actually derives something. Collections are
            # content-addressed (members + derivation profile), so a "derived"
            # node with an empty profile is byte-for-byte its base and migrates
            # onto the same node — see test_a_noop_derivation_returns_the_base
            # _itself in test_dataset_lineage.py.
            "derivation_profile": {"super_res_factor": 2},
        },
    )

    derived_collection = migrate_legacy_dataset_node(derived_legacy)

    # Migrating the derived node first must recursively migrate its base.
    base_collection = orm.load_node(base_legacy.base.extras.get("migrated_to_uuid"))
    assert isinstance(base_collection, PinoDatasetCollectionData)

    metadata = derived_collection.get_dataset_metadata()
    assert metadata["composition_kind"] == "migration"
    assert metadata["dataset_role"] == "derived"
    assert metadata["base_collection_uuid"] == base_collection.uuid
    # The legacy derivation profile (time-as-batch layout) is carried over.
    assert metadata["derivation_profile"]["time_as_batch"] is True
    assert metadata["derivation_profile"]["time_as_dim"] is False
    # The derived collection wraps the migrated base's simulations.
    assert derived_collection.get_member_simulation_uuids() == base_collection.get_member_simulation_uuids()
    assert derived_legacy.base.extras.get("migrated_to_uuid") == derived_collection.uuid


# --- migrate_legacy_union_node ---


def test_migrate_legacy_union_node_covers_all_members(
    legacy_pino_dataset_factory, pino_dataset_union_factory
) -> None:
    member_a = legacy_pino_dataset_factory(
        "mig-union-a",
        arrays=_batch_arrays(1, seed=5),
        metadata={"grouped_files": _grouped_files(1, prefix="mig-union-a-case")},
    )
    member_b = legacy_pino_dataset_factory(
        "mig-union-b",
        arrays=_batch_arrays(1, seed=6),
        metadata={"grouped_files": _grouped_files(1, prefix="mig-union-b-case")},
    )
    union = pino_dataset_union_factory(member_a, member_b, union_name="mig-union")

    collection = migrate_legacy_union_node(union)

    assert isinstance(collection, PinoDatasetCollectionData)
    assert collection.get_simulation_count() == 2
    member_uuids = set(collection.get_member_simulation_uuids())
    for member in (member_a, member_b):
        migrated_member = orm.load_node(member.base.extras.get("migrated_to_uuid"))
        assert set(migrated_member.get_member_simulation_uuids()) <= member_uuids
    assert collection.get_dataset_metadata()["composition_kind"] == "union_migrated"
    assert union.base.extras.get("migrated_to_uuid") == collection.uuid
    assert union.base.extras.get("superseded") is True


# --- migrate_all_legacy_datasets ---


def test_migrate_all_legacy_datasets_counts_and_idempotency(
    aiida_profile_clean, legacy_pino_dataset_factory, pino_dataset_union_factory
) -> None:
    base_a = legacy_pino_dataset_factory(
        "all-a",
        arrays=_batch_arrays(2, seed=7),
        metadata={"grouped_files": _grouped_files(2, prefix="all-a-case")},
    )
    legacy_pino_dataset_factory(
        "all-b",
        role="derived",
        metadata={
            "base_dataset_uuid": base_a.uuid,
            "base_dataset_pk": base_a.pk,
            "derivation_profile": {"super_res_factor": 2},  # see the derived-chain test
        },
    )
    base_c = legacy_pino_dataset_factory(
        "all-c",
        arrays=_batch_arrays(1, seed=8),
        metadata={"grouped_files": _grouped_files(1, prefix="all-c-case")},
    )
    union = pino_dataset_union_factory(base_c, union_name="all-union")

    summary = migrate_all_legacy_datasets()
    assert summary["legacy_dataset_count"] == 3
    assert summary["legacy_union_count"] == 1
    # base A, derived B, base C — and NOT a fourth node for the union: a union
    # of one dataset holds exactly base C's members with no derivation, so
    # content-addressed reuse migrates it onto base C's collection rather than
    # minting a twin. Every legacy node still maps somewhere (asserted below);
    # what collapses is duplicate content, not the migration record.
    assert summary["migrated_collection_count"] == 3
    assert len(summary["migrated_collection_pks"]) == 3
    assert union.base.extras.get("migrated_to_uuid") == base_c.base.extras.get("migrated_to_uuid")

    simulations_before = _simulation_count()
    second = migrate_all_legacy_datasets()
    assert second["legacy_dataset_count"] == 3
    assert second["legacy_union_count"] == 1
    assert second["migrated_collection_count"] == 3
    assert sorted(second["migrated_collection_pks"]) == sorted(summary["migrated_collection_pks"])
    assert _simulation_count() == simulations_before
