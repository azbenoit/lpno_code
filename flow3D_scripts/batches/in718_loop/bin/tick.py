#!/usr/bin/env python3
"""State machine for the two-alloy loop, in three modes.

  tick.py                          propose: print and log the next stage of each arm
  tick.py --apply [--every S]      autonomous: run the next stage of each arm (every S s)
  tick.py --apply --gate [...]     gated: the same, but each solved round waits for
                                   `round.py approve --arm A --round N` before collect

Each call advances each arm by AT MOST ONE stage and stops on anything unexpected (a failed
training or evaluation, a command failing MAX_RETRIES ticks in a row) rather than guessing.
The stage is derived from observed state (run directories, AiiDA node states, the state
files), never from a stored pointer, so it cannot drift from reality, and a run that was
interrupted resumes at the right stage.

The guards in round.py come from five bugs found in the experiment's rounds (probe overlap,
duplicate acquisition, arm[:3] name collisions, exit 306 twice, non-idempotent b_r history).
`--gate` keeps a person between each round's solves and its training, for the checks that
are not automated (see the README).
"""
import argparse, fcntl, json, os, subprocess, sys, time, zipfile
from datetime import datetime
from pathlib import Path

LOOP = Path(__file__).resolve().parent.parent
BIN = Path(__file__).resolve().parent
from _repo import RUNS
LOG = LOOP / "tick_log.jsonl"
LOCK = LOOP / "state" / ".tick.lock"
STOPPED = LOOP / "state" / "STOPPED"   # written when the loop stops; every tick honours it
ARMS = ("maximin", "bandit")
MAX_RETRIES = 3      # consecutive failed ticks of one stage before the loop stops
MAX_WAITS = 12       # consecutive "not ready yet" exits (2) of one stage before it stops
COMMAND_TIMEOUT_S = 3 * 3600   # a stage command running longer than this is killed

# hard caps -- a bug must not be able to spend two rounds' budget
MAX_SOLVES_PER_ROUND = 25
MAX_TRAIN_PER_ROUND = 3
# Free-disk floor PER STAGE. One blanket 60 GB floor was calibrated for a solve and
# stalled the loop at NEED_PROBE_EVAL with 57.7 GB free -- for a stage whose entire
# persisted output is 4.0 MB (measured: round-3's 42 eval nodes stored 252 objects
# totalling 4.0 MB in the object store). Measured costs of the other stages, from this
# loop's own tick log and run dirs:
#   solve  -- flsgrf reaches ~13 GB before the NPZ export drops it; runner floor is 30 GB
#   train  -- ~7 GB per run (disk went 99.5 -> 57.7 GB across six round-4 trainings)
#   eval   -- 4.0 MB persisted. The transient working set is NOT measured, so the floor
#             below is set far above anything plausible rather than close to 4 MB.
MIN_FREE_GB = 60
MIN_FREE_GB_BY_STAGE = {
    "NEED_PROBE_EVAL": 20,
    "NEED_PROBE_UPDATE": 20,
    "NEED_COLLECT": 20,
    "NEED_TRAIN": 60,
    "NEED_ACQUIRE": 60,
}
MIN_RAM_GB = 20

# The loop stops after this round: the experiment ends at round 5.
# Without this the tick proposes NEED_ACQUIRE forever, and an autonomous run would keep
# buying solves past the point the experiment was declared finished.
FINAL_ROUND = 5


def _orm():
    from aiida import load_profile, orm
    load_profile()
    return orm


def _disk_paths():
    """Where the loop writes: the runs directory and, when it can be resolved, the AiiDA
    profile's repository."""
    paths = [RUNS if RUNS.exists() else RUNS.parent]
    try:
        from aiida.manage import get_manager
        repo = get_manager().get_profile().repository_path
        if repo and Path(repo).exists():
            paths.append(Path(repo))
    except Exception:
        pass
    return paths


def _resources():
    """Gate on the two things that actually stop work here: free disk where the loop writes
    (the smallest of its locations) and available RAM."""
    free_gb = min(s.f_bavail * s.f_frsize / 1e9 for s in map(os.statvfs, _disk_paths()))
    avail_gb = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            avail_gb = int(line.split()[1]) / 1e6
    return {"free_gb": round(free_gb, 1), "ram_avail_gb": round(avail_gb, 1)}


def _resources_ok(res, stage):
    """Is there room for THIS stage? A 4 MB eval must not wait on a solve-sized floor."""
    floor = MIN_FREE_GB_BY_STAGE.get(stage, MIN_FREE_GB)
    return (res["free_gb"] >= floor and res["ram_avail_gb"] >= MIN_RAM_GB), floor


FAMILY_SUFFIXES = ("", "top", "repl")


def _families(arm, rnd):
    """The rendered families of one round (directories holding a CASE_ORDER.txt)."""
    return [d for s in FAMILY_SUFFIXES
            if ((d := LOOP / f"round{rnd:02d}_{arm}{s}") / "CASE_ORDER.txt").is_file()]


def _npz_ok(path):
    """An exported case: the NPZ exists and is a complete zip archive (not one mid-write)."""
    return path.is_file() and zipfile.is_zipfile(path)


def family_progress(orm, fam, rnd):
    """(solved, can_still_solve, state) for one family.

    `round.py solve` records each family's batch workflow in roundNN_solve_pks.json, and
    the batch's own state decides: while it runs, every case without an NPZ may still
    solve; once it has terminated, a case without an NPZ never will. State is
    'unsubmitted' (no batch recorded and no run started), 'in_flight', 'finished' or
    'failed' (terminated with nothing exported). A family solved without a recorded batch
    is judged from the run directories: unfinished cases count as pending while a solver
    runs on this machine, and as failed once none does.
    """
    names = (fam / "CASE_ORDER.txt").read_text().split()
    done = sum(1 for n in names if _npz_ok(RUNS / n / "npz" / f"{n}.npz"))
    pk = (_json(f"round{rnd:02d}_solve_pks.json") or {}).get(fam.name)
    if pk is not None:
        if not orm.load_node(int(pk)).is_terminated:
            return done, len(names), "in_flight"
        return (done, done, "finished") if done else (0, 0, "failed")
    if not done and not any((RUNS / n).is_dir() for n in names):
        return 0, len(names), "unsubmitted"
    if _solver_running():
        return done, len(names), "in_flight"
    return done, done, "finished"


def _solver_running():
    import subprocess
    r = subprocess.run(["pgrep", "-f", "run_rectangular_case.py"],
                       capture_output=True, text=True)
    # pgrep -f matches this pgrep's own argv in some shells; require a real pid line
    return any(line.strip().isdigit() for line in r.stdout.splitlines())


def _json(name):
    p = LOOP / name
    return json.loads(p.read_text()) if p.is_file() else None


def _terminated(orm, pks):
    """(terminated, total, failed pks); failed covers a non-zero exit, an exception and a kill."""
    ns = [orm.load_node(p) for p in pks]
    return (sum(1 for n in ns if n.is_terminated), len(ns),
            [n.pk for n in ns if n.is_terminated and not n.is_finished_ok])


def _uncollected(orm, arm, rnd, st):
    """Exported case names of this round that `collect` has not taken ownership of."""
    names = []
    for f in _families(arm, rnd):
        names += (f / "CASE_ORDER.txt").read_text().split()
    exported = {n for n in names if _npz_ok(RUNS / n / "npz" / f"{n}.npz")}
    owned = set()
    for pk in st.get("owned_sim_pks", []):
        try:
            owned.add((orm.load_node(pk).label or "").split(" | ")[0])
        except Exception:
            pass
    return exported - owned


def stage_of(orm, arm):
    """Observed stage for one arm, plus the evidence behind it."""
    rounds = sorted(int(d.name[5:7]) for d in LOOP.glob(f"round*_{arm}") if d.is_dir())
    if not rounds:
        # `round.py init` writes the arm's state (its exclusion list); round 1 can start then.
        if (LOOP / "state" / f"{arm}.json").is_file():
            return "START", {"round": 1}
        return "NO_ROUNDS", {}
    rnd = max(rounds)
    if not (LOOP / f"round{rnd:02d}_{arm}" / "CASE_ORDER.txt").is_file():
        # A render that died before writing CASE_ORDER.txt: acquire removes and redoes it.
        return "NEED_ACQUIRE", {"round": rnd, "incomplete_render": True}
    prog = {f.name: family_progress(orm, f, rnd) for f in _families(arm, rnd)}
    unsub = [name for name, p in prog.items() if p[2] == "unsubmitted"]
    if unsub:
        return "NEED_SOLVE", {"round": rnd, "family": unsub[0],
                              "suffix": unsub[0][len(f"round{rnd:02d}_{arm}"):]}
    failed = [name for name, p in prog.items() if p[2] == "failed"]
    if failed:
        return "SOLVE_FAILED", {"round": rnd, "families": failed}
    done, total = sum(p[0] for p in prog.values()), sum(p[1] for p in prog.values())
    # A batch that has not terminated may still be ingesting its last export.
    if done < total or any(p[2] == "in_flight" for p in prog.values()):
        return "SOLVING", {"round": rnd, "solved": done, "of": total}

    st = _json(f"state/{arm}.json") or {}
    if st.get("round", 0) < rnd:
        return "NEED_COLLECT", {"round": rnd, "solved": done}
    # `state.round` alone is NOT enough. A replacement acquired AFTER its round was
    # collected solves into round0N_<arm>repl while state.round is already N, so this
    # check passed and the stage jumped straight to NEED_TRAIN -- the replacement was
    # exported, ingested, and silently never trained on (measured: maximin would have
    # trained on 74 cases with a 75th sitting collected-but-unowned). Compare the round's
    # actual case names against what collect has taken ownership of.
    uncollected = sorted(_uncollected(orm, arm, rnd, st))
    if uncollected:
        return "NEED_COLLECT", {"round": rnd, "solved": done,
                                "uncollected": uncollected[:4],
                                "n_uncollected": len(uncollected)}

    tp = _json(f"round{rnd:02d}_train_pks.json") or _json(f"round{rnd:02d}v3_train_pks.json")
    mine = {k: v for k, v in (tp or {}).items() if k.startswith(arm)}
    if len(mine) < MAX_TRAIN_PER_ROUND:
        return "NEED_TRAIN", {"round": rnd, "submitted": len(mine)}
    nterm, ntot, failed = _terminated(orm, list(mine.values()))
    if failed:
        return "TRAIN_FAILED", {"round": rnd, "failed": failed}
    if nterm < ntot:
        return "TRAINING", {"round": rnd, "terminated": nterm, "of": ntot}

    ep = _json(f"round{rnd:02d}_probe_eval_pks.json") or {}
    mine_ev = {k: v for k, v in ep.items() if k.startswith(arm)}
    if not mine_ev:
        return "NEED_PROBE_EVAL", {"round": rnd}
    nterm, ntot, failed = _terminated(orm, list(mine_ev.values()))
    if failed:
        return "EVAL_FAILED", {"round": rnd, "failed": failed}
    if nterm < ntot:
        return "EVALUATING", {"round": rnd, "terminated": nterm, "of": ntot}

    hist = (st.get("history") or {})
    if any(len(v) < rnd for v in hist.values()) or not hist:
        return "NEED_PROBE_UPDATE", {"round": rnd}
    if rnd >= FINAL_ROUND:
        return "DONE", {"round": rnd, "final_round": FINAL_ROUND}
    return "NEED_ACQUIRE", {"round": rnd + 1}


ACTIONS = {
    "SOLVING":            ("wait",   "solves in flight; nothing to do"),
    "TRAINING":           ("wait",   "trainings in flight; nothing to do"),
    "EVALUATING":         ("wait",   "evals in flight; nothing to do"),
    "START":              ("act",    "round.py acquire --arm {arm} --n {nsolve} ; round.py solve --arm {arm} --round 1"),
    "NEED_COLLECT":       ("act",    "check_round.py --arm {arm} --round {round} [--write-drops] ; "
                                     "round.py collect --arm {arm} --round {round}"),
    "NEED_TRAIN":         ("act",    "round.py train --arm {arm} --round {round}  (train-only collection, {ntrain} members)"),
    "NEED_PROBE_EVAL":    ("act",    "round.py probe --arm {arm} --round {round}  (submits per-regime probe evals)"),
    "NEED_PROBE_UPDATE":  ("act",    "round.py probe --arm {arm} --round {round}  (collect E_r, append history, fit b_r)"),
    "NEED_ACQUIRE":       ("act",    "round.py acquire --arm {arm} --n {nsolve} ; round.py solve --arm {arm} --round {round}"),
    "NEED_SOLVE":         ("act",    "round.py solve --arm {arm} --round {round} --suffix '{suffix}'  ({family} has no batch yet)"),
    "SOLVE_FAILED":       ("stop",   "batches of {families} terminated with no case exported -- STOP, needs diagnosis"),
    "ERROR":              ("hold",   "could not read the loop state: {error}"),
    "DONE":               ("stop",   "round {round} is the final round -- the loop ends here"),
    "TRAIN_FAILED":       ("stop",   "training failures {failed} -- STOP, needs diagnosis"),
    "EVAL_FAILED":        ("stop",   "eval failures {failed} -- STOP, needs diagnosis"),
    "NO_ROUNDS":          ("stop",   "no rounds and no state: run `round.py init` first"),
}


def commands_for(stage, det, arm, *, autonomous):
    """The commands `--apply` runs for `stage`, in order; [] for a stage that only waits or stops.

    Autonomous mode runs the boundary check with --write-drops before collect, so a
    domain-truncated case is excluded rather than trained on (collect assigns splits once, at
    ingest). Gated mode leaves the drops to the person who approves the round.
    """
    py = sys.executable
    rnd_py, chk_py = str(BIN / "round.py"), str(BIN / "check_round.py")
    r = str(det.get("round", ""))
    if stage == "NEED_COLLECT":
        cmds = [[py, chk_py, "--arm", arm, "--round", r, "--write-drops"]] if autonomous else []
        return cmds + [[py, rnd_py, "collect", "--arm", arm, "--round", r]]
    if stage == "NEED_TRAIN":
        return [[py, rnd_py, "train", "--arm", arm, "--round", r]]
    if stage in ("NEED_PROBE_EVAL", "NEED_PROBE_UPDATE"):
        return [[py, rnd_py, "probe", "--arm", arm, "--round", r]]
    if stage in ("NEED_ACQUIRE", "START"):
        return [[py, rnd_py, "acquire", "--arm", arm, "--n", str(MAX_SOLVES_PER_ROUND)],
                [py, rnd_py, "solve", "--arm", arm, "--round", r]]
    if stage == "NEED_SOLVE":
        sfx = det.get("suffix") or ""
        return [[py, rnd_py, "solve", "--arm", arm, "--round", r] + (["--suffix", sfx] if sfx else [])]
    return []


def gate_hold(stage, det, arm, *, gated, approvals, families):
    """The hold message when `--gate` keeps a solved round from collect, else None.

    `families` are the round's family directory names; each needs approval, so a
    replacement acquired after the round was approved is held until it is checked too."""
    if not gated or stage != "NEED_COLLECT":
        return None
    rnd = int(det["round"])
    missing = sorted(set(families) - set(approvals.get(arm, {}).get(str(rnd), [])))
    if not missing:
        return None
    return (f"round {rnd} solved, {missing} not approved yet; waiting for a person: check it with "
            f"`check_round.py --arm {arm} --round {rnd}` (add --write-drops to record its boundary "
            f"drops, or edit round{rnd:02d}_drops.json), then `round.py approve --arm {arm} --round {rnd}`")


def at_limit(failures, key):
    """True when this stage of this arm and round has already used up its retries or waits."""
    c = failures.get(key) or {}
    return c.get("fail", 0) >= MAX_RETRIES or c.get("wait", 0) >= MAX_WAITS


def run_command(cmd, timeout):
    """Run one stage command in its own process group; on timeout kill the whole group, so a
    hung `aiida-pino` call started by it cannot submit after the stage has been given up."""
    import signal
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return 124


def record_result(failures, key, rc):
    """Track consecutive results of one stage of one arm and round; True to give up.

    Exit 0 resets the counts. Exit 2 is a step saying "not ready yet" (`collect` while a solved
    case is still being ingested, `probe` after submitting evaluations): it is allowed
    MAX_WAITS times in a row. Any other exit counts as a failure, allowed MAX_RETRIES times.
    `failures` is kept in state/ between ticks, so the limits hold under a timer too.
    """
    c = failures.setdefault(key, {"fail": 0, "wait": 0})
    if rc == 0:
        c["fail"] = c["wait"] = 0
        return False
    if rc == 2:
        c["wait"] += 1
        return c["wait"] >= MAX_WAITS
    c["fail"] += 1
    return c["fail"] >= MAX_RETRIES


def _read_approvals():
    p = LOOP / "state" / "approvals.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def _load_failures():
    return _json("state/tick_failures.json") or {}


def _save_failures(failures):
    (LOOP / "state").mkdir(exist_ok=True)
    (LOOP / "state" / "tick_failures.json").write_text(json.dumps(failures, indent=1) + "\n")


def tick_once(orm, a, failures):
    """One pass over both arms. Returns 'continue', 'done' or 'stop'."""
    res = _resources()
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"  tick {ts}   disk {res['free_gb']} GB   RAM avail {res['ram_avail_gb']} GB")
    approvals = _read_approvals()
    entries, kinds, give_up = [], [], False
    for arm in ARMS:
        try:
            stage, det = stage_of(orm, arm)
            failures.pop(f"{arm}|ERROR|", None)
        except Exception as exc:          # a transient database error, an unreadable file
            stage, det = "ERROR", {"error": f"{type(exc).__name__}: {exc}"}
        kind, tmpl = ACTIONS.get(stage, ("stop", "unknown stage"))
        det2 = dict(det, arm=arm, ntrain=MAX_TRAIN_PER_ROUND, nsolve=MAX_SOLVES_PER_ROUND)
        try: action = tmpl.format(**det2)
        except Exception: action = tmpl
        ok, floor = _resources_ok(res, stage)
        if kind == "act" and not ok:
            kind, action = "hold", (f"free {res['free_gb']} GB / RAM {res['ram_avail_gb']} GB "
                                    f"below the {floor} GB floor for {stage} -- would defer: {action}")
        fams = [f.name for f in _families(arm, int(det["round"]))] if "round" in det else []
        held = (gate_hold(stage, det, arm, gated=a.gate, approvals=approvals, families=fams)
                if kind == "act" else None)
        if held:
            kind, action = "hold", held
        print(f"  [{arm:<8}] {stage:<18} {det}")
        print(f"             -> {kind.upper()}: {action}")
        rcs = []
        key = f"{arm}|{stage}|{det.get('round', '')}"
        if stage == "ERROR" and record_result(failures, key, 1):
            print(f"  [{arm}] the loop state could not be read {MAX_RETRIES} ticks in a row -- stopping")
            give_up = True
        if a.apply and kind == "act" and at_limit(failures, key):
            print(f"  [{arm}] {stage} has already failed {MAX_RETRIES} times or reported not ready "
                  f"{MAX_WAITS} times -- stopping instead of running it again")
            kind, give_up = "stop", True
        if a.apply and kind == "act":
            for cmd in commands_for(stage, det, arm, autonomous=not a.gate):
                rc = run_command(cmd, COMMAND_TIMEOUT_S)
                if rc == 124:
                    print(f"  [{arm}] {Path(cmd[1]).name} {cmd[2:4]} ran over {COMMAND_TIMEOUT_S} s; killed")
                rcs.append(rc)
                if rc not in (0, 2):
                    break
            if record_result(failures, key, rcs[-1] if rcs else 0):
                print(f"  [{arm}] {stage} did not advance after {MAX_RETRIES} failures or "
                      f"{MAX_WAITS} not-ready exits in a row -- stopping")
                give_up = True
        kinds.append((stage, kind))
        entries.append({"ts": ts, "arm": arm, "stage": stage, "detail": det,
                        "kind": kind, "action": action, "resources": res,
                        "applied": bool(a.apply and kind == "act"), "rcs": rcs})
    with open(LOG, "a") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    _save_failures(failures)
    print(f"  logged {len(entries)} entries -> {LOG}")
    if give_up or any(k == "stop" and s != "DONE" for s, k in kinds):
        if a.apply:
            why = "; ".join(f"{e['arm']}: {e['stage']} {e['action']}" for e in entries
                            if e["kind"] == "stop" or e["stage"] == "ERROR" or e["rcs"])
            STOPPED.write_text(f"{ts}  {why}\n")
            print(f"  wrote {STOPPED}: later ticks do nothing until a person removes it")
        return "stop"
    if all(s == "DONE" for s, _ in kinds):
        return "done"
    return "continue"


def main():
    global FINAL_ROUND
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="run the next stage of each arm")
    ap.add_argument("--gate", action="store_true",
                    help="hold each solved round until `round.py approve` records a person's check")
    ap.add_argument("--every", type=float, default=0,
                    help="with --apply: keep ticking every this many seconds until done or stopped")
    ap.add_argument("--final-round", type=int, default=FINAL_ROUND,
                    help=f"last round of the experiment (default {FINAL_ROUND})")
    ap.add_argument("--max-ticks", type=int, default=0, help="stop after this many ticks (0: no limit)")
    a = ap.parse_args()
    FINAL_ROUND = a.final_round
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("  another tick is running (state/.tick.lock); not starting a second one")
        return 0
    if a.apply and STOPPED.is_file():
        print(f"  the loop is stopped ({STOPPED}):\n    {STOPPED.read_text().strip()}\n"
              f"  fix the cause (see round.py retry), then remove the file to resume")
        return 1
    orm = _orm()
    failures, ticks = _load_failures(), 0
    while True:
        ticks += 1
        status = tick_once(orm, a, failures)
        if status != "continue" or not (a.apply and a.every) or (a.max_ticks and ticks >= a.max_ticks):
            break
        time.sleep(a.every)
    print(f"  loop {status}")
    return 1 if status == "stop" else 0


if __name__ == "__main__":
    sys.exit(main())
