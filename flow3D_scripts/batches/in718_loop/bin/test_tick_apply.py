#!/usr/bin/env python3
"""Checks on the loop's executing mode: what `tick.py --apply` runs at each stage, how it
decides a round's solves are finished, the per-family approval gate, the retry limits, the
training-member profiles `round.py train` submits, and the boundary drops
`check_round.py --write-drops` records. No AiiDA, no solver, no GPU: AiiDA nodes are
stand-ins and the loop and runs directories are temporary.

Run: python bin/test_tick_apply.py
"""
import copy, importlib.util, json, pathlib, sys, tempfile, zipfile
from types import SimpleNamespace

BIN = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BIN))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, BIN / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tick = _load("loop_tick", "tick.py")
rnd = _load("loop_round", "round.py")          # not `round`: that would shadow the builtin
chk = _load("loop_check", "check_round.py")


def tails(cmds):
    """Each command without its interpreter and script path: the part that names the step."""
    return [[pathlib.Path(c[1]).name] + c[2:] for c in cmds]


def test_commands_per_stage():
    t = lambda stage, det, auto=True: tails(tick.commands_for(stage, det, "bandit", autonomous=auto))
    # Autonomous: the boundary check records its drops BEFORE collect, because collect
    # assigns splits once, at ingest, and never re-assigns them.
    assert t("NEED_COLLECT", {"round": 3}) == [
        ["check_round.py", "--arm", "bandit", "--round", "3", "--write-drops"],
        ["round.py", "collect", "--arm", "bandit", "--round", "3"]]
    # Gated: the person who approved the round has already decided the drops.
    assert t("NEED_COLLECT", {"round": 3}, auto=False) == [
        ["round.py", "collect", "--arm", "bandit", "--round", "3"]]
    assert t("NEED_TRAIN", {"round": 3}) == [["round.py", "train", "--arm", "bandit", "--round", "3"]]
    for stage in ("NEED_PROBE_EVAL", "NEED_PROBE_UPDATE"):
        assert t(stage, {"round": 3}) == [["round.py", "probe", "--arm", "bandit", "--round", "3"]]
    # NEED_ACQUIRE's detail carries the NEXT round number.
    assert t("NEED_ACQUIRE", {"round": 4}) == [
        ["round.py", "acquire", "--arm", "bandit", "--n", str(tick.MAX_SOLVES_PER_ROUND)],
        ["round.py", "solve", "--arm", "bandit", "--round", "4"]]
    assert t("START", {"round": 1}) == t("NEED_ACQUIRE", {"round": 1})
    assert t("NEED_SOLVE", {"round": 2, "suffix": ""}) == [["round.py", "solve", "--arm", "bandit", "--round", "2"]]
    assert t("NEED_SOLVE", {"round": 2, "suffix": "repl"}) == [
        ["round.py", "solve", "--arm", "bandit", "--round", "2", "--suffix", "repl"]]
    for stage in ("SOLVING", "TRAINING", "EVALUATING", "DONE", "TRAIN_FAILED", "SOLVE_FAILED",
                  "NO_ROUNDS", "ERROR"):
        assert tick.commands_for(stage, {"round": 1}, "bandit", autonomous=True) == [], stage


def test_gate():
    approvals = {"maximin": {"2": ["round02_maximin"], "3": ["round03_maximin"]}}
    hold = lambda stage, rr, fams, arm="maximin", gated=True: tick.gate_hold(
        stage, {"round": rr}, arm, gated=gated, approvals=approvals, families=fams)
    assert hold("NEED_COLLECT", 4, ["round04_maximin"]) is not None      # solved, not approved
    assert "approve --arm maximin --round 4" in hold("NEED_COLLECT", 4, ["round04_maximin"])
    assert hold("NEED_COLLECT", 2, ["round02_maximin"]) is None          # approved
    # a replacement acquired after the approval is held until it is checked too
    assert "round03_maximinrepl" in hold("NEED_COLLECT", 3, ["round03_maximin", "round03_maximinrepl"])
    assert hold("NEED_COLLECT", 4, ["round04_maximin"], gated=False) is None   # autonomous
    assert hold("NEED_COLLECT", 2, ["round02_bandit"], arm="bandit") is not None   # per arm
    # The gate sits only between solving and collecting.
    for stage in ("NEED_TRAIN", "NEED_PROBE_EVAL", "NEED_PROBE_UPDATE", "NEED_ACQUIRE", "NEED_SOLVE"):
        assert hold(stage, 4, ["round04_maximin"]) is None, stage


def test_approvals_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "approvals.json"
        assert rnd.read_approvals(p) == {}
        rnd.add_approval(p, "bandit", 2, ["round02_bandit"])
        rnd.add_approval(p, "bandit", 2, ["round02_bandit"])           # idempotent
        rnd.add_approval(p, "bandit", 2, ["round02_banditrepl"])
        rnd.add_approval(p, "bandit", 1, ["round01_bandit"])
        assert rnd.read_approvals(p) == {"bandit": {"1": ["round01_bandit"],
                                                    "2": ["round02_bandit", "round02_banditrepl"]}}


def test_member_profiles():
    base = {"run_name": "fno_twoalloy_steady",
            "hydra_overrides": ["++custom.seed=42", "custom.super_res=1"]}
    keep = copy.deepcopy(base)
    p3 = rnd.member_profile(base, "maximin", 4, 3)
    assert base == keep, "the shared profile must not be mutated"
    assert p3["run_name"] == "fno_twoalloy_maximin_r04_m3"
    seeds = [o for o in p3["hydra_overrides"] if "custom.seed=" in o]
    assert seeds == [f"++custom.seed={rnd.MEMBER_SEEDS[2]}"], seeds
    assert "custom.super_res=1" in p3["hydra_overrides"]
    p1 = rnd.member_profile({"run_name": "x", "hydra_overrides": []}, "bandit", 1, 1)
    assert p1["hydra_overrides"] == [f"++custom.seed={rnd.MEMBER_SEEDS[0]}"]
    assert len(set(rnd.MEMBER_SEEDS)) == len(rnd.MEMBER_SEEDS) >= tick.MAX_TRAIN_PER_ROUND


def test_boundary_reasons():
    ok = dict(melted=True, peakT=2500.0, depth=200, floor=200, span=0.4, xmarg=60)
    assert chk.boundary_reasons(ok) == []
    # No melt is a lack-of-fusion case: valid training data, reported but never dropped.
    assert chk.boundary_reasons(dict(melted=False, peakT=1200.0)) == []
    assert [r.split(":")[0] for r in chk.boundary_reasons(dict(ok, floor=50))] == ["FLOOR"]
    assert [r.split(":")[0] for r in chk.boundary_reasons(dict(ok, xmarg=10))] == ["XWALL"]
    assert [r.split(":")[0] for r in chk.boundary_reasons(dict(ok, span=0.8))] == ["SPAN"]


def test_merge_drops():
    old = {"maximin": ["in718_maximin_r02_001"]}              # a list-shaped entry
    new = chk.merge_drops(old, "bandit", {"in718_bandit_r02_004": "XWALL: ..."})
    assert old == {"maximin": ["in718_maximin_r02_001"]}, "input must not be mutated"
    assert new == {"maximin": ["in718_maximin_r02_001"],
                   "bandit_health_check": {"in718_bandit_r02_004": "XWALL: ..."}}
    assert chk.merge_drops(new, "bandit", {"in718_bandit_r02_004": "XWALL: ..."}) == new
    assert chk.merge_drops({}, "maximin", {}) == {"maximin_health_check": {}}


def test_retry_limit():
    f = {}
    for _ in range(tick.MAX_RETRIES - 1):
        assert not tick.record_result(f, "bandit|NEED_COLLECT|3", 1)
    assert tick.record_result(f, "bandit|NEED_COLLECT|3", 1), "should give up"
    f = {}
    tick.record_result(f, "bandit|NEED_COLLECT|3", 1)
    tick.record_result(f, "bandit|NEED_COLLECT|3", 0)            # success resets the count
    assert f["bandit|NEED_COLLECT|3"] == {"fail": 0, "wait": 0}
    # "not ready yet" is allowed longer, but not forever
    f = {}
    for _ in range(tick.MAX_WAITS - 1):
        assert not tick.record_result(f, "bandit|NEED_COLLECT|3", 2)
    assert tick.record_result(f, "bandit|NEED_COLLECT|3", 2)
    # counts are per round: a failure in round 3 does not carry into round 4
    f = {}
    tick.record_result(f, "bandit|NEED_COLLECT|3", 1)
    assert not tick.record_result(f, "bandit|NEED_COLLECT|4", 1)
    assert json.loads(json.dumps(f)) == f, "the counts must survive a JSON round trip (state file)"


class FakeOrm:
    def __init__(self, nodes):
        self.nodes = nodes

    def load_node(self, pk):
        n = self.nodes[int(pk)]
        n.pk = int(pk)
        return n


def _node(terminated, ok=True):
    return SimpleNamespace(is_terminated=terminated, is_finished_ok=terminated and ok, label="")


def _npz(runs, name, complete=True):
    d = runs / name / "npz"
    d.mkdir(parents=True, exist_ok=True)
    if complete:
        with zipfile.ZipFile(d / f"{name}.npz", "w") as z:
            z.writestr("T.npy", b"0")
    else:
        (d / f"{name}.npz").write_bytes(b"PK\x03\x04 half written")


def test_stage_detection():
    """How a round's solves are judged done: from the recorded batch, never from the folders alone."""
    saved = (tick.LOOP, tick.RUNS, tick._solver_running)
    try:
        with tempfile.TemporaryDirectory() as d:
            loop, runs = pathlib.Path(d) / "loop", pathlib.Path(d) / "runs"
            (loop / "state").mkdir(parents=True); runs.mkdir()
            tick.LOOP, tick.RUNS, tick._solver_running = loop, runs, (lambda: False)
            orm = FakeOrm({})
            assert tick.stage_of(orm, "bandit")[0] == "NO_ROUNDS"
            (loop / "state" / "bandit.json").write_text(json.dumps({"round": 0, "taken": []}))
            assert tick.stage_of(orm, "bandit") == ("START", {"round": 1})

            fam = loop / "round01_bandit"
            fam.mkdir()
            # a render that died before CASE_ORDER.txt is redone by acquire
            assert tick.stage_of(orm, "bandit")[0] == "NEED_ACQUIRE"
            (fam / "CASE_ORDER.txt").write_text("c0\nc1\n")
            # rendered, never submitted (a refused or failed submission): solve it
            assert tick.stage_of(orm, "bandit") == ("NEED_SOLVE", {"round": 1, "family": "round01_bandit", "suffix": ""})

            (loop / "round01_solve_pks.json").write_text(json.dumps({"round01_bandit": 7}))
            orm.nodes[7] = _node(terminated=False)
            # batch running, c0 exported, c1's run directory exists without an NPZ (it is
            # solving): the round is still solving, not finished one case short
            _npz(runs, "c0"); (runs / "c1").mkdir()
            assert tick.stage_of(orm, "bandit") == ("SOLVING", {"round": 1, "solved": 1, "of": 2})
            # a half-written NPZ is not an exported case
            _npz(runs, "c1", complete=False)
            assert tick.stage_of(orm, "bandit")[1]["solved"] == 1
            # every case exported but the batch still running (ingesting): still solving
            _npz(runs, "c1")
            assert tick.stage_of(orm, "bandit") == ("SOLVING", {"round": 1, "solved": 2, "of": 2})
            (runs / "c1" / "npz" / "c1.npz").unlink()
            # batch terminated: c1 will never export, the round is done with the case it has
            orm.nodes[7] = _node(terminated=True, ok=False)
            assert tick.stage_of(orm, "bandit")[0] == "NEED_COLLECT"
            # batch terminated with nothing exported: stop, do not collect an empty round
            (runs / "c0" / "npz" / "c0.npz").unlink()
            assert tick.stage_of(orm, "bandit") == ("SOLVE_FAILED", {"round": 1, "families": ["round01_bandit"]})
    finally:
        tick.LOOP, tick.RUNS, tick._solver_running = saved


def test_settings_are_recorded_and_checked():
    env = {"IN718_LOOP_BASE_DS": "1234", "IN718_LOOP_TAG": "run2"}
    s = rnd.resolve_settings(env=env, recorded={})
    assert s["base_ds"] == "1234" and s["tag"] == "run2" and s["val_ds"] == "0b657a95"
    # an initialised run keeps its values when a timer's environment lacks the variables
    assert rnd.resolve_settings(env={}, recorded=s) == s
    for bad_env, bad_rec in (({"IN718_LOOP_TAG": "run3"}, s),          # changed mid-run
                             ({"IN718_LOOP_TAG": "run.2"}, {})):       # not a case-name character
        try:
            rnd.resolve_settings(env=bad_env, recorded=bad_rec)
        except SystemExit:
            continue
        raise AssertionError(f"{bad_env} should be refused")


def test_batch_simulations_walks_called_processes():
    """Simulations hang off processes the batch CALLED, which a with_ancestors query misses."""
    import types
    links = types.ModuleType("aiida.common.links")
    links.LinkType = SimpleNamespace(CREATE="create", RETURN="return")
    saved = {k: sys.modules.get(k) for k in ("aiida", "aiida.common", "aiida.common.links")}
    sys.modules.update({"aiida": types.ModuleType("aiida"), "aiida.common": types.ModuleType("aiida.common"),
                        "aiida.common.links": links})
    try:
        def proc(outputs, called=()):
            out = {lt: [SimpleNamespace(node=n) for n in ns] for lt, ns in outputs.items()}
            return SimpleNamespace(called_descendants=list(called), base=SimpleNamespace(links=SimpleNamespace(
                get_outgoing=lambda link_type: SimpleNamespace(all=lambda: out.get(link_type, [])))))
        sim = lambda pk, label: SimpleNamespace(pk=pk, label=label, node_type="data.pino.PinoSimulationData.")
        calc = proc({"create": [sim(11, "case_a | case_a | flow3d"),
                                SimpleNamespace(pk=12, label="", node_type="data.core.dict.Dict.")]})
        gen = proc({"return": [sim(11, "case_a | case_a | flow3d")]})
        batch = proc({"return": [sim(13, "case_b | case_b | flow3d")]}, called=[gen, calc])
        orm = FakeOrm({5: batch})
        assert rnd.batch_simulations(orm, [5]) == [(11, "case_a | case_a | flow3d"),
                                                   (13, "case_b | case_b | flow3d")]
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_limits_checked_before_running():
    f = {"bandit|NEED_TRAIN|2": {"fail": tick.MAX_RETRIES, "wait": 0}}
    assert tick.at_limit(f, "bandit|NEED_TRAIN|2")
    assert not tick.at_limit(f, "bandit|NEED_TRAIN|3") and not tick.at_limit({}, "x")
    assert tick.at_limit({"k": {"fail": 0, "wait": tick.MAX_WAITS}}, "k")


def test_timeout_kills_the_process_group():
    import time
    t0 = time.time()
    rc = tick.run_command(["sh", "-c", "sleep 30 & sleep 30"], timeout=1)
    assert rc == 124 and time.time() - t0 < 10, (rc, time.time() - t0)


def test_retire_keeps_the_failed_pks_on_record():
    saved = rnd.LOOP
    try:
        with tempfile.TemporaryDirectory() as d:
            rnd.LOOP = pathlib.Path(d)
            rnd.retire(3, "train", {"bandit_m2": 41})
            rnd.retire(3, "train", {"bandit_m2": 57})
            assert json.loads((rnd.LOOP / "round03_retired_pks.json").read_text()) == \
                {"train": {"bandit_m2": [41, 57]}}
    finally:
        rnd.LOOP = saved


def test_failed_jobs_detected():
    # an excepted or killed job has no exit status, and is a failure all the same
    orm = FakeOrm({1: _node(True, ok=True), 2: _node(True, ok=False), 3: _node(False)})
    assert tick._terminated(orm, [1, 2, 3]) == (2, 3, [2])


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
