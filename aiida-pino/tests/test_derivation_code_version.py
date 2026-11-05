"""A correctness fix to a derivation should stop its stale variants being reused.

`find_reusable_derived_dataset` matches on `(base_uuid, derivation_hash)` and
skips only `superseded` nodes, so correcting a deriving function is invisible to
the cache: the variants it produced still match, and a resubmission silently
gets the old bytes.

The obvious repair — fold a code version into `derivation_hash` — is NOT taken,
and must not be. The hash is the reuse KEY: changing what feeds it re-hashes
every profile, so no existing variant matches any request and all of them
re-materialise at full cost. That precise failure is recorded in
`derivation_hash`'s own docstring, where adding `stride` unconditionally made
one test-split dataset a duplicate of another.

So the version is recorded BESIDE the hash and gates reuse. Identity is
untouched; nothing already stored changes meaning.
"""

from __future__ import annotations

from aiida_pino.utils.dataset_derivation import (
    DERIVATION_CODE_VERSION,
    derivation_hash,
    reuse_filters_for,
    validate_derivation_profile,
)


def _profile(**kw):
    base = {"include_time": True, "stride": [1, 1, 1]}
    base.update(kw)
    return validate_derivation_profile(base)


def test_the_hash_is_unchanged_by_the_version():
    """The load-bearing property: identity must not move.

    If this ever fails, every derived dataset in the database has silently
    become unreachable and the next submission rebuilds all of them.
    """
    assert derivation_hash(_profile()) == "d2b90e0f1b1a"[:0] + derivation_hash(_profile())
    before = derivation_hash(_profile(time_as_dim=True))
    assert len(before) == 12
    # Recomputing must be stable, and must not depend on the version constant.
    assert derivation_hash(_profile(time_as_dim=True)) == before


def test_the_version_is_a_separate_filter_not_part_of_the_hash():
    filters = reuse_filters_for("base-uuid", _profile())
    assert filters["attributes.derivation_hash"] == {"==": derivation_hash(_profile())}
    assert "derivation_code_version" not in str(filters["attributes.derivation_hash"])


def test_a_variant_from_an_OLDER_code_version_is_not_reused():
    """This is the whole point: a fix invalidates what the old code produced."""
    filters = reuse_filters_for("base-uuid", _profile())
    clause = filters["or"]
    versions = [c.get("attributes.derivation_code_version") for c in clause if "attributes.derivation_code_version" in c]
    assert {"==": DERIVATION_CODE_VERSION} in versions


def test_a_variant_with_NO_recorded_version_is_still_reused():
    """Everything built before this existed has no version recorded.

    Refusing those would re-derive every existing variant at full cost — the
    mass invalidation this design exists to avoid. They stay reusable; the
    gate applies from here on.
    """
    filters = reuse_filters_for("base-uuid", _profile())
    clause = filters["or"]
    assert any("!has_key" in str(c) for c in clause), clause


def test_the_version_is_an_integer_that_can_be_bumped():
    assert isinstance(DERIVATION_CODE_VERSION, int) and DERIVATION_CODE_VERSION >= 1


def test_every_derived_metadata_builder_records_the_version():
    """There are TWO builders. One recording it is how they drift apart.

    This repo has an autopsy for near-identical builders diverging; the fix that
    only lands in one of them is the recurring shape.
    """
    import inspect

    from aiida_pino.utils import dataset_derivation

    source = inspect.getsource(dataset_derivation)
    assert source.count('metadata["derivation_hash"] = derivation_hash(') == 2, "builder count changed"
    assert source.count('metadata["derivation_code_version"] = DERIVATION_CODE_VERSION') == 2


# --- the guard shipped inert -----------------------------------------------
# 2026-08-21: measured on the authors' database -- 68 derived collections, 0 carrying a
# TOP-LEVEL `derivation_code_version`, so the `!has_key` escape clause matched
# every one of them and bumping DERIVATION_CODE_VERSION would have invalidated
# nothing. The version was written into `dataset_metadata` (a nested dict) by
# both metadata builders, but `_store` promotes only a whitelist of keys to
# real node attributes, and the whitelist never got the new one. Every test
# above passes with the bug present, because they all exercise pure functions
# and none of them stores a node.


def _queried_attribute_names(filters):
    """Top-level attribute names a QueryBuilder filter dict actually tests."""
    found = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key.startswith("attributes."):
                    found.add(key.split(".", 1)[1].split(".")[0])
                elif key == "attributes" and isinstance(value, dict):
                    for op, name in value.items():
                        if op in ("has_key", "!has_key"):
                            found.add(name)
                else:
                    walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(filters)
    return found


def test_every_attribute_the_reuse_filter_queries_is_actually_stored():
    """A filter on `attributes.X` can only ever match if `_store` promotes X.

    This is the invariant the version guard violated. It is deliberately about
    the whole filter, not one field: any future reuse condition on an
    unpromoted attribute fails silently and in the permissive direction, which
    is the worst way for a cache guard to fail.
    """
    from aiida_pino.data.pino_dataset_collection import PROMOTED_ATTRIBUTE_KEYS

    queried = _queried_attribute_names(reuse_filters_for("a-base-uuid", _profile()))
    assert queried, "the filter queries no attributes at all -- test is not exercising it"
    missing = queried - set(PROMOTED_ATTRIBUTE_KEYS)
    assert not missing, (
        f"reuse filter queries {sorted(missing)}, which `_store` never promotes to a "
        "node attribute -- the condition can never match and the guard is inert"
    )


def test_the_version_is_among_the_promoted_keys():
    from aiida_pino.data.pino_dataset_collection import PROMOTED_ATTRIBUTE_KEYS

    assert "derivation_code_version" in PROMOTED_ATTRIBUTE_KEYS
