"""Tests for the ``aiida-pino`` CLI (`aiida_pino.cli`).

Browse/annotate/delete tests run against the AiiDA test profile with seeded
nodes, with `cli._service` monkeypatched to a directly constructed service (no
`load_profile` call). Profile-token parsing is unit-tested. `--help` is proven
to work in a subprocess where aiida/fastapi/pydantic imports are blocked.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from aiida_pino import cli


@pytest.fixture
def svc(monkeypatch):
    from aiida_pino.console.config import Settings
    from aiida_pino.console.service import AiidaConsoleService

    service = AiidaConsoleService(Settings())
    monkeypatch.setattr(cli, "_service", lambda args: service)
    return service


def _cli_stderr(text: str) -> str:
    """The CLI's own stderr, from its `error:` line onward.

    `capsys` captures the whole stream, and SQLAlchemy emits warnings during
    node access after an earlier delete in the same process. They landed above
    the CLI's error line and broke `err.startswith("error:")` — the assertion
    was measuring SQLAlchemy, and it passed or failed depending on which other
    tests had run.

    Filtering by the `warnings` module's line format was tried first and is not
    robust: the message can itself span lines, so a header-plus-indent rule
    leaves the tail behind (measured: "Identity map already had an identity
    for ..." survived and broke the same assertion a second time). Anchoring on
    what the CLI writes is exact.

    The trade-off, stated plainly: this no longer proves stderr is *only* the
    CLI's line. It still proves the CLI writes exactly one line, that it starts
    with `error:`, and what it says — which is the contract these tests exist
    for. Text with no `error:` line is returned unchanged, so assertions about
    empty stderr are unaffected.
    """
    marker = "error:"
    for index, line in enumerate(lines := text.splitlines(keepends=True)):
        if line.startswith(marker):
            return "".join(lines[index:])
    return text


def run_cli(capsys, *argv):
    rc = cli.main(list(argv))
    captured = capsys.readouterr()
    return rc, captured.out, _cli_stderr(captured.err)


def test_cli_stderr_survives_library_noise():
    """Both shapes that broke `test_submit_gated_on_daemon_exits_1` in a full run."""
    single = "/x/aiida/orm/nodes.py:224: SAWarning: boom\n  self._flush()\nerror: nope\n"
    assert _cli_stderr(single) == "error: nope\n"
    # A warning whose own message wraps onto further lines — this is the one a
    # header-plus-indent filter let through.
    wrapped = (
        "/x/models.py:1: SAWarning: Identity map already had an identity for\n"
        "(DbNode, (98,), None), replacing it with newly flushed object.\n"
        "error: nope\n"
    )
    assert _cli_stderr(wrapped) == "error: nope\n"
    # No error line: untouched, so assertions about empty stderr still hold.
    assert _cli_stderr("") == ""
    assert _cli_stderr("just a warning\n") == "just a warning\n"


# --- help without heavy imports ----------------------------------------------
def test_help_runs_without_heavy_imports():
    code = textwrap.dedent(
        """
        import sys

        blocked = ("aiida", "fastapi", "pydantic", "pydantic_settings", "aiida_pino.console")

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if any(name == b or name.startswith(b + ".") for b in blocked):
                    raise ImportError("blocked import: " + name)
                return None

        sys.meta_path.insert(0, Blocker())
        from aiida_pino.cli import main

        main(["--help"])
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "usage: aiida-pino" in proc.stdout


# --- profile-token parsing ---------------------------------------------------
def test_parse_profile_none_passthrough():
    assert cli.parse_profile(None) is None


def test_parse_profile_key_val_json_values():
    out = cli.parse_profile(["a=true", "b=3", "c=[1,2]", "d=plain text", "e=0.5"])
    assert out == {"a": True, "b": 3, "c": [1, 2], "d": "plain text", "e": 0.5}


def test_parse_profile_at_file_and_override(tmp_path):
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")
    assert cli.parse_profile([f"@{profile_file}"]) == {"a": 1, "b": 2}
    assert cli.parse_profile([f"@{profile_file}", "b=3"]) == {"a": 1, "b": 3}


def test_presubmit_accumulates_repeated_profile_flags_like_submit():
    """`presubmit` kept only the LAST `--training-profile` (nargs="*"), so the exact
    `submit` command, which takes one token per flag, was checked with one key.
    Found writing reproducibility/train_evaluate.ipynb."""
    flags = ["--training-profile", "config_name=c", "--training-profile", "run_name=r",
             "--derivation-profile", "include_time=true", "--derivation-profile", "time_as_batch=false"]
    pre = cli.build_parser().parse_args(["presubmit", "--train-dataset", "1", *flags])
    sub = cli.build_parser().parse_args(
        ["submit", "train-eval", "--train-dataset", "1", "--python-code", "p", *flags])
    for args in (pre, sub):
        assert cli.parse_profile(args.training_profile) == {"config_name": "c", "run_name": "r"}
        assert cli.parse_profile(args.derivation_profile) == {"include_time": True, "time_as_batch": False}
    # presubmit still takes several tokens after one flag, as the README says
    many = cli.build_parser().parse_args(
        ["presubmit", "--train-dataset", "1", "--training-profile", "a=1", "b=2", "--training-profile", "c=3"])
    assert cli.parse_profile(many.training_profile) == {"a": 1, "b": 2, "c": 3}
    assert cli.build_parser().parse_args(["presubmit", "--train-dataset", "1"]).derivation_profile is None


def test_parse_profile_rejects_bad_tokens(tmp_path):
    with pytest.raises(ValueError):
        cli.parse_profile(["not-a-pair"])
    not_object = tmp_path / "list.json"
    not_object.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError):
        cli.parse_profile([f"@{not_object}"])


# --- sims listing with filters ----------------------------------------------
def test_sims_split_and_keyword_filters(svc, capsys, pino_simulation_factory):
    tagged = pino_simulation_factory("cli-flt-tagged")
    plain = pino_simulation_factory("cli-flt-plain")
    svc.update_annotations(tagged.pk, split="train", keywords_set=["keyhole", "smoke"])
    svc.update_annotations(plain.pk, keywords_set=["keyhole"])

    rc, out, _ = run_cli(capsys, "sims", "--split", "train", "--json")
    assert rc == 0
    pks = {row["pk"] for row in json.loads(out)}
    assert tagged.pk in pks and plain.pk not in pks

    rc, out, _ = run_cli(capsys, "sims", "--split", "unassigned", "--json")
    assert rc == 0
    pks = {row["pk"] for row in json.loads(out)}
    assert plain.pk in pks and tagged.pk not in pks

    # AND semantics on repeated --keyword
    rc, out, _ = run_cli(capsys, "sims", "--keyword", "keyhole", "--keyword", "smoke", "--json")
    assert rc == 0
    rows = json.loads(out)
    pks = {row["pk"] for row in rows}
    assert tagged.pk in pks and plain.pk not in pks
    row = next(row for row in rows if row["pk"] == tagged.pk)
    assert row["split"] == "train" and row["keywords"] == ["keyhole", "smoke"]

    # default output is a human table with the same rows
    rc, out, _ = run_cli(capsys, "sims", "--keyword", "keyhole")
    assert rc == 0
    assert "pk" in out.splitlines()[0]
    assert str(tagged.pk) in out and str(plain.pk) in out


# --- tag/split/describe round-trip -------------------------------------------
def test_core_tagged_node_refuses_deletion(svc, pino_simulation_factory):
    from aiida_pino.console.errors import InvalidRequestError

    sim = pino_simulation_factory("core-protected-sim")
    svc.update_annotations(sim.pk, keywords_add=["core"])
    with pytest.raises(InvalidRequestError, match="core"):
        svc.delete_nodes([sim.pk], dry_run=True)
    svc.update_annotations(sim.pk, keywords_remove=["core"])
    out = svc.delete_nodes([sim.pk], dry_run=True)  # untagged: dry-run passes again
    assert sim.pk in out["affected"]


def test_tag_split_describe_roundtrip(svc, capsys, pino_simulation_factory):
    sim = pino_simulation_factory("cli-ann-sim")
    pk = str(sim.pk)

    rc, out, _ = run_cli(capsys, "tag", pk, "--add", "keyhole", "smoke", "--json")
    assert rc == 0
    assert json.loads(out)["keywords"] == ["keyhole", "smoke"]
    rc, out, _ = run_cli(capsys, "tag", pk, "--remove", "keyhole", "--json")
    assert rc == 0
    assert json.loads(out)["keywords"] == ["smoke"]

    rc, out, _ = run_cli(capsys, "split", pk, "test", "--json")
    assert rc == 0
    assert json.loads(out)["split"] == "test"

    rc, out, _ = run_cli(capsys, "describe", pk, "cli notes", "--title", "cli title", "--json")
    assert rc == 0
    body = json.loads(out)
    assert body["description"] == "cli notes"
    assert body["title"] == "cli title"

    # bare `tag PK` reads current state; verify the full round-trip landed
    rc, out, _ = run_cli(capsys, "tag", pk, "--json")
    assert rc == 0
    state = json.loads(out)
    assert state == {
        "pk": sim.pk,
        "kind": "simulation",
        "title": "cli title",
        "label": state["label"],
        "description": "cli notes",
        "keywords": ["smoke"],
        "split": "test",
    }
    # description lands on the native column and the extras mirror
    from aiida.orm import load_node

    node = load_node(sim.pk)
    assert node.description == "cli notes"
    assert node.base.extras.get("description") == "cli notes"

    # invalid split value -> single-line error, exit 1
    rc, out, err = run_cli(capsys, "split", pk, "bogus")
    assert rc == 1 and out == ""
    assert err.startswith("error:") and "\n" not in err.strip()


# --- delete dry-run vs --force -----------------------------------------------
def test_delete_dry_run_default_then_force(svc, capsys, pino_simulation_factory):
    from aiida.orm import Node, QueryBuilder, load_node

    def node_exists(pk: int) -> bool:
        return QueryBuilder().append(Node, filters={"id": pk}).count() > 0

    sim_pk = pino_simulation_factory("cli-del-sim").pk

    rc, out, _ = run_cli(capsys, "delete", str(sim_pk), "--json")
    assert rc == 0
    result = json.loads(out)
    assert result["dry_run"] is True and result["deleted"] is False
    assert sim_pk in result["affected"]
    assert load_node(sim_pk).pk == sim_pk  # still there

    rc, out, _ = run_cli(capsys, "delete", str(sim_pk), "--force", "--json")
    assert rc == 0
    result = json.loads(out)
    assert result["dry_run"] is False and result["deleted"] is True
    assert not node_exists(sim_pk)


# --- submit gating and CLI-level errors --------------------------------------
def test_submit_gated_on_daemon_exits_1(svc, capsys, local_code_factory):
    local_code_factory("/bin/bash", label="pino-cli-gate")
    rc, out, err = run_cli(
        capsys, "submit", "raw-flow3d", "--source-path", "/tmp/does-not-matter", "--python-code", "pino-cli-gate"
    )
    assert rc == 1 and out == ""
    assert err.startswith("error:")
    assert "docker start aiida-rabbitmq && verdi daemon start" in err
    assert "\n" not in err.strip()


def test_bad_profile_token_exits_1(svc, capsys):
    rc, out, err = run_cli(capsys, "submit", "derive", "--dataset", "1", "--derivation-profile", "junk")
    assert rc == 1 and out == ""
    assert "bad profile token" in err


def test_unknown_pk_exits_1(svc, capsys):
    rc, _, err = run_cli(capsys, "show", "99000123")
    assert rc == 1
    assert err.startswith("error:") and "99000123" in err
