"""The staging stamp hashes the syntax tree of the staging sources, not their text.

Three full runtime-cache rebuilds on 2026-09-14 were caused by edits that could not change a
staged byte: a memory factor, then a comment, then a docstring, in a file on the list. The
pool sizing has since left that file (`utils/export_pool.py`), and the digest now ignores
comments, formatting and docstrings. Stamps written by the byte digest over the same
derivation code are listed as equivalent so the caches built that day are not rebuilt.
"""

from __future__ import annotations

from aiida_pino.utils import runtime_cache as rc


def test_comments_docstrings_and_formatting_do_not_change_the_digest():
    a = b'def f(x):\n    """doc"""\n    # comment\n    return x + 1\n'
    b = b'def f(x):\n\n    return x+1  # other comment\n'
    assert rc._staging_source_digest_bytes(a) == rc._staging_source_digest_bytes(b)


def test_a_change_to_code_does_change_the_digest():
    a = b"def f(x):\n    return x + 1\n"
    b = b"def f(x):\n    return x + 2\n"
    assert rc._staging_source_digest_bytes(a) != rc._staging_source_digest_bytes(b)


def test_unparseable_source_is_hashed_verbatim_not_dropped():
    a = b"def f(:\n"
    assert rc._staging_source_digest_bytes(a) == a


def test_an_equivalent_stamp_is_accepted_as_current(tmp_path, monkeypatch):
    import json

    (tmp_path / "manifest.json").write_text(json.dumps({"staging_code_version": "6f31a4505743"}))
    monkeypatch.setattr(rc, "staging_code_version", lambda: "045c27ef3462")
    usable, reason = rc.staged_cache_is_current(tmp_path)
    assert usable and "equivalent" in reason
    (tmp_path / "manifest.json").write_text(json.dumps({"staging_code_version": "000000000000"}))
    usable, reason = rc.staged_cache_is_current(tmp_path)
    assert not usable and "different" in reason


def test_the_pool_sizing_is_not_in_a_hashed_file():
    assert "utils/export_pool.py" not in rc.STAGING_CODE_FILES
    assert "utils/guards.py" not in rc.STAGING_CODE_FILES


def test_a_stale_cache_is_sidestepped_to_an_existing_equivalent_sibling(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(rc, "staging_code_version", lambda: "bdad67243746")
    monkeypatch.setattr(rc, "EQUIVALENT_STAGING_CODE_VERSIONS", frozenset({"6f31a4505743"}))
    # A finished build under an equivalent stamp is found and reused ...
    done = tmp_path / "abc-s6f31a4505743"
    done.mkdir()
    (done / "manifest.json").write_text(json.dumps({"staging_code_version": "6f31a4505743"}))
    assert rc.sidestep_cache_name(tmp_path, "abc") == "abc-s6f31a4505743"
    # ... the current stamp's own directory wins when it exists ...
    cur = tmp_path / "abc-sbdad67243746"
    cur.mkdir()
    (cur / "manifest.json").write_text(json.dumps({"staging_code_version": "bdad67243746"}))
    assert rc.sidestep_cache_name(tmp_path, "abc") == "abc-sbdad67243746"
    # ... and with nothing usable beside it, the fresh name is the current stamp's.
    assert rc.sidestep_cache_name(tmp_path, "xyz") == "xyz-sbdad67243746"
