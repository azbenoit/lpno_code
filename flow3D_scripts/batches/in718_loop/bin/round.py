#!/usr/bin/env python3
"""Round driver for the two-alloy IN718 loop.

Subcommands rather than a monolith, so each step is separately runnable and
verifiable -- the P-V experiment's driver was a single pass and a failure in a
late stage meant re-running the early ones.

  init     seed both arms' exclusion list (the frozen test set and the probe)
  acquire  pick this round's cases (arm = maximin | bandit)
  solve    render configs and submit the FLOW-3D batch through AiiDA
  collect  assign the round's new cases split=train (or excluded) and record them as owned
  train    compose the round's train-only collection and submit its training members
  probe    out-of-fold per-region error -> state, for the bandit's b_r
  approve  record that a person checked a solved round (the gate of `tick.py --gate`)
  retry    set aside a round's failed solves, trainings or probe evaluations to resubmit them
  state    show the current loop state

`tick.py` chains these: it reads the observed stage and either proposes the next one or,
with `--apply`, runs it.

Invariants this driver enforces, each from a documented failure:
  * The frozen IN718 test set is never in a training collection, and is excluded
    from the acquisition lattice by construction (not checked afterwards).
  * Each round SNAPSHOTS its member list into its own content-addressed
    collection. Splits are per-simulation, global and MUTABLE -- ds 62921 moved
    92/6/21 -> 85/6/20 within an hour and silently split a three-way comparison
    across different member sets while every arm cited the same pk.
  * Member counts echoed by the submission report are recorded, because that
    report is the only tell that a dataset's membership changed.
"""
import argparse, copy, csv, fcntl, glob, json, os, re, shutil, subprocess, sys, zipfile
from pathlib import Path

LOOP = Path(__file__).resolve().parent.parent
from _repo import REPO, RUNS
PY = sys.executable                       # acquire.py and make_configs.py run in the same env
_CLI = Path(sys.executable).with_name("aiida-pino")
AIIDA_PINO = str(_CLI) if _CLI.is_file() else "aiida-pino"   # the CLI of this env, else PATH
CONFIG = LOOP / "state" / "config.json"

# The run's settings: node references in the AiiDA store (a pk or a UUID), the AiiDA Code and a
# name prefix. `init` records them in state/config.json; later calls read them from there, so a
# timer whose environment lacks a variable cannot switch a running loop to other values. The
# defaults name nodes of the authors' store; on another installation set the variables before init.
SETTINGS = {
    "base_ds": ("IN718_LOOP_BASE_DS", "86529"),     # Ti-6Al-4V anchor dataset (171 train cases)
    "val_ds": ("IN718_LOOP_VAL_DS", "0b657a95"),    # ti64_anchorbox_lhs150_val_split (9 cases)
    "python_code": ("IN718_LOOP_PYTHON_CODE", "1292"),
    "tag": ("IN718_LOOP_TAG", ""),                  # prefix of the run's case and dataset names
}
TAG_RE = re.compile(r"[A-Za-z0-9-]*")                # the runner's case names: letters, digits, _ and -


def resolve_settings(env=os.environ, recorded=None):
    """{name: value}: the recorded value of an initialised run, else the environment, else the
    default. A variable set to a value other than the recorded one is refused."""
    recorded = recorded if recorded is not None else (
        json.loads(CONFIG.read_text()) if CONFIG.is_file() else {})
    out = {}
    for name, (var, default) in SETTINGS.items():
        value = str(env.get(var, "")).strip()
        if name == "tag":
            value = value.strip("_")
        if name in recorded:
            if value and value != recorded[name]:
                raise SystemExit(f"  refusing: {var}={value!r} differs from the value this run was "
                                 f"initialised with ({recorded[name]!r}, state/config.json)")
            value = recorded[name]
        out[name] = value or default
    if not TAG_RE.fullmatch(out["tag"]):
        raise SystemExit(f"  refusing: IN718_LOOP_TAG={out['tag']!r}; use letters, digits and '-'")
    return out


_S = resolve_settings()
BASE_DS, VAL_DS, PYTHON_CODE, RUN_TAG = _S["base_ds"], _S["val_ds"], _S["python_code"], _S["tag"]
BASE_MODEL = 86651       # FNO steady material-blind (cited in dataset descriptions only)
TESTSET_BATCH = 86840    # frozen IN718 test set (cited in dataset descriptions only)
TAG = f"{RUN_TAG}_" if RUN_TAG else ""
MEMBER_SEEDS = (42, 43, 44)   # member m1, m2, m3 of each arm and round
APPROVALS = LOOP / "state" / "approvals.json"
FAMILY_SUFFIXES = ("",) + ("top", "repl")   # the families of one round: main, top-up, replacement


def _ref(value):
    """A node reference as load_node wants it: an int pk, or a UUID string."""
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def npz_ok(path):
    """An exported case: the NPZ exists and is a complete zip archive (not one mid-write)."""
    return path.is_file() and zipfile.is_zipfile(path)


def families(arm, rnd):
    """The rendered families of one round (directories holding a CASE_ORDER.txt)."""
    return [d for s in FAMILY_SUFFIXES
            if ((d := LOOP / f"round{rnd:02d}_{arm}{s}") / "CASE_ORDER.txt").is_file()]


def solve_pks(rnd):
    p = LOOP / f"round{rnd:02d}_solve_pks.json"
    return json.loads(p.read_text()) if p.is_file() else {}


CLI_TIMEOUT_S = 900
# Submitted copies of the configs carry absolute paths, so they are written under the untracked
# runs/ directory, never into the tracked loop directory.
SUBMIT_ROOT = RUNS / "_in718_loop_submissions" / LOOP.name


_HELD = []


def submit_lock():
    """Held while a stage submits and records its pks, so a person's `round.py` and a tick's
    cannot both read a pk file, both submit and both write it."""
    (LOOP / "state").mkdir(exist_ok=True)
    fh = open(LOOP / "state" / ".submit.lock", "w")
    fcntl.flock(fh, fcntl.LOCK_EX)
    _HELD.append(fh)                     # released when the process exits


def batches_terminated(orm, pks):
    return all(orm.load_node(int(p)).is_terminated for p in pks)


def train_members_recorded(arm, rnd):
    p = LOOP / f"round{rnd:02d}_train_pks.json"
    tp = json.loads(p.read_text()) if p.is_file() else {}
    return [k for k in tp if k.startswith(arm)]


def sh(*cmd, **kw):
    print(f"  $ {' '.join(map(str, cmd))}", flush=True)
    return subprocess.run(list(map(str, cmd)), **kw)


def cli_json(*cmd):
    """Run an `aiida-pino --json ...` command and return its parsed result, or None."""
    print(f"  $ {' '.join(map(str, cmd))}", flush=True)
    try:
        r = subprocess.run(list(map(str, cmd)), capture_output=True, text=True, timeout=CLI_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"  no answer from the AiiDA CLI within {CLI_TIMEOUT_S} s\n")
        return None
    if r.returncode:
        sys.stderr.write(r.stdout[-1500:] + r.stderr[-1500:])
        return None
    out = r.stdout
    start = 0 if out.lstrip().startswith("{") else out.find("\n{") + 1
    try:
        return json.loads(out[start:])
    except json.JSONDecodeError:
        sys.stderr.write(out[-1500:])
        return None


def read_approvals(path=APPROVALS):
    """{arm: {round (str): [approved family directory names]}}."""
    return json.loads(path.read_text()) if path.is_file() else {}


def add_approval(path, arm, rnd, fams):
    """Approve the families of `arm`'s round `rnd` that exist now. A family acquired later
    (a replacement) needs its own approval, so it is checked before it is trained on."""
    ap = read_approvals(path)
    per = ap.setdefault(arm, {})
    per[str(int(rnd))] = sorted(set(per.get(str(int(rnd)), [])) | set(fams))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ap, indent=1) + "\n")
    return ap


def member_profile(base, arm, rnd, k):
    """Training profile of member k (1-based) of `arm` in round `rnd`: the shared profile
    with its own run name and seed. Any seed override already in the profile is replaced."""
    p = copy.deepcopy(base)
    kept = [o for o in p.get("hydra_overrides", []) if not o.lstrip("+").startswith("custom.seed=")]
    p["hydra_overrides"] = [f"++custom.seed={MEMBER_SEEDS[k - 1]}"] + kept
    p["run_name"] = f"fno_twoalloy_{TAG}{arm}_r{rnd:02d}_m{k}"
    return p


def load_state(path):
    return json.loads(path.read_text()) if path.is_file() else {
        "round": 0, "taken": [], "region_err": {}, "region_n": {}, "history": {},
        "owned_sim_pks": [], "rounds": []}


def cmd_acquire(a):
    st_path = LOOP / "state" / f"{a.arm}.json"
    st_path.parent.mkdir(parents=True, exist_ok=True)
    st = load_state(st_path)
    # Default: the next round. `--round N --suffix repl` instead acquires INTO an already
    # collected round, as a replacement for cases its health check dropped. Deriving the
    # round from state with no override wrote a round0(N+1) directory holding one case and
    # consumed the next round's slot; recovering that took a rename, and the rename left a
    # stale absolute materials path that only `--prepare-only` caught.
    suffix = getattr(a, "suffix", "") or ""
    rnd = a.round if getattr(a, "round", None) else st["round"] + 1
    if suffix and not (LOOP / f"round{rnd:02d}_{a.arm}").is_dir():
        print(f"  refusing: --suffix {suffix!r} names a family of round {rnd}, but "
              f"round{rnd:02d}_{a.arm} does not exist")
        return 1
    fam = LOOP / f"round{rnd:02d}_{a.arm}{suffix}"
    if suffix and train_members_recorded(a.arm, rnd):
        print(f"  refusing: {a.arm} round {rnd} already has training members submitted, so a "
              f"replacement would not be trained on this round; acquire it before `train`")
        return 1
    if (fam / "CASE_ORDER.txt").is_file():
        print(f"  refusing: {fam.name} is already acquired; a further replacement for round "
              f"{rnd} goes in the other family suffix ('top' or 'repl')")
        return 1
    if fam.is_dir():                     # a render that died before CASE_ORDER.txt: redo it
        print(f"  removing the incomplete render {fam.name}")
        shutil.rmtree(fam)
    out = LOOP / "out" / f"r{rnd:02d}{suffix}_{a.arm}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # The frozen test points and the probe are part of `taken`, so drop_near excludes them.
    # Families rendered but not collected yet (this round's main family, when a replacement
    # is acquired before collect) are added here, so a replacement cannot re-pick them.
    extra_taken = []
    for r in range(st["round"] + 1, rnd + 1):
        for f in families(a.arm, r):
            for row in csv.DictReader(open(f / "design.csv")):
                extra_taken.append({"H": float(row["H"]), "V": float(row["V"]),
                                    "r0": float(row["r0_um"]) * 1e-6, "P": float(row["P"]),
                                    "src": f.name})
    acq_state = st_path
    if extra_taken:
        acq_state = LOOP / "out" / f"acquire_state_{a.arm}.json"
        acq_state.write_text(json.dumps(dict(st, taken=st["taken"] + extra_taken), indent=1))
    extra = ["--h-max", str(a.h_max)] if getattr(a, "h_max", None) else []
    rc = sh(PY, LOOP / "bin" / "acquire.py", "--arm", a.arm, "--n", a.n,
            "--state", acq_state, "--out", out, *extra).returncode
    if rc:
        return rc
    rc = sh(PY, LOOP / "bin" / "make_configs.py", "--picks", out, "--arm", a.arm,
            "--round", rnd, "--suffix", suffix,
            env=dict(os.environ, IN718_LOOP_TAG=RUN_TAG)).returncode   # the run's recorded tag
    if rc or not (fam / "CASE_ORDER.txt").is_file():
        print(f"  make_configs failed (exit {rc}); removing the incomplete {fam.name}")
        shutil.rmtree(fam, ignore_errors=True)
        return rc or 1
    print(f"  round {rnd}{suffix} picks -> {out}")
    return 0


def cmd_solve(a):
    # `--suffix repl` solves round0N_<arm>repl, the replacement family for cases dropped
    # by the health check. cmd_collect already reads the top/repl families; solve did not,
    # so a replacement had to be submitted by hand.
    rd = LOOP / f"round{a.round:02d}_{a.arm}{getattr(a, 'suffix', '') or ''}"
    submit_lock()                        # held until the pks are recorded
    recorded = solve_pks(a.round)
    if rd.name in recorded:
        print(f"  {rd.name} was already submitted as batch {recorded[rd.name]}; not resubmitting")
        return 0
    order = (rd / "CASE_ORDER.txt").read_text().split()
    # The batch receives each config as data, not as a file, so a material path relative to
    # the config's directory would no longer resolve: submit it absolute. The tracked
    # configs keep the relative path.
    sub = SUBMIT_ROOT / rd.name
    sub.mkdir(parents=True, exist_ok=True)
    cfgs = []
    for n in order:
        src = rd / "configs" / (n + ".json")
        cfg = json.loads(src.read_text())
        mat = cfg.get("material")
        if isinstance(mat, dict) and mat.get("file") and not Path(mat["file"]).is_absolute():
            mat["file"] = str((src.parent / mat["file"]).resolve())
        (sub / src.name).write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
        cfgs += ["--config", f"@{sub / src.name}"]
    prof = rd / "dataset_profile.json"
    prof.write_text(json.dumps({
        "batch_keyword": f"in718_loop_{TAG}{a.arm}_r{a.round:02d}",
        "description": (
            f"Two-alloy IN718 self-improvement loop, arm={a.arm}, round {a.round}: "
            f"{len(order)} acquired training cases. Solved in the ds86529 domain "
            f"(900x600x450+100 um @ 10 um) so they compose with the Ti64 anchor set and are "
            f"evaluable by base model {BASE_MODEL}. TRAINING cases -- the frozen IN718 test "
            f"set is batch {TESTSET_BATCH} and is never mixed in. Acquisition: "
            + ("greedy maximin over the full candidate lattice (baseline)."
               if a.arm == "maximin" else
               "diminishing-returns bandit -- region chosen by w_r*(E_r*b_r+c)/(n_r+1), "
               "point chosen by maximin within it.")),
        "keywords": ["in718", "two-alloy", "self-improve", "straight path", "steady",
                     "training", a.arm, f"round{a.round:02d}", "experimentally_anchored"],
    }, indent=1))
    exec_prof = sub / "execution_profile.json"
    # The runs land in this checkout's runs/, which is where tick.py and collect look.
    exec_prof.write_text(json.dumps({"cores": 10, "threads_per_rank": 10, "export": "npz",
                                     "flow3d_scripts_root": str(REPO), "out_root": str(RUNS)}))
    res = cli_json(AIIDA_PINO, "--json", "submit", "flow3d-batch", "--python-code", PYTHON_CODE,
                   *cfgs, "--execution-profile", f"@{exec_prof}",
                   "--dataset-profile", f"@{prof}",
                   "--materials-root", rd / "materials")
    if not res or "pk" not in res:
        print(f"  submission of {rd.name} failed; nothing recorded, so the next tick retries")
        return 1
    recorded = solve_pks(a.round)
    recorded[rd.name] = int(res["pk"])
    (LOOP / f"round{a.round:02d}_solve_pks.json").write_text(json.dumps(recorded, indent=1) + "\n")
    print(f"  {rd.name}: batch {res['pk']} submitted ({len(order)} cases)")
    return 0


def cmd_init(a):
    """Seed both arms' state with the points they must NEVER acquire.

    `taken` is what `drop_near` excludes, so seeding it with the 25 frozen test
    points is what makes the test set excluded BY CONSTRUCTION rather than
    checked for overlap after the fact -- the P-V experiment did the latter and
    got lucky. The 10 pre-existing IN718 cases go in too: they are the shared
    training seed, so neither arm should spend budget re-buying them.
    """
    # EVERY held-out set must be seeded here, not just the test set. The probe was
    # created after `init` had already run and was never added, so round 2 drew from a
    # lattice still containing it and picked 9/25 (maximin) and 8/25 (bandit) probe
    # points -- most at distance exactly 0.0000. A probe measured on cases the model
    # trained on is in-sample, which corrupts E_r and therefore b_r, the bandit's whole
    # mechanism.
    tests = list(csv.DictReader(open(LOOP.parent / "in718_testset_20260904" / "design.csv")))
    # Every probe design present (round00_probe*/design.csv) is excluded, including the one
    # `probe_regime_collections.json` evaluates on. Without a probe the bandit has no error
    # signal, so init refuses.
    probe_csvs = sorted(glob.glob(str(LOOP / "round00_probe*" / "design.csv")))
    if not probe_csvs:
        print("  refusing: no probe design (round00_probe*/design.csv); solve the probe set first")
        return 1
    probes = [r for f in probe_csvs for r in csv.DictReader(open(f))]
    seed_pts = [{"H": float(r["H"]), "V": float(r["V"]),
                 "r0": float(r["r0_um"]) * 1e-6, "P": float(r["P"]),
                 "src": src} for rows, src in ((tests, "frozen_test"), (probes, "probe"))
                for r in rows]
    # the 10 existing IN718 cases (pk 30068), r0 fixed at 50 um
    import math
    mat = json.loads((LOOP.parent / "in718_testset_20260904" / "materials" /
                      "in718_full" / "in718_full.json").read_text())
    sys.path.insert(0, str(LOOP / "bin"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("acq", LOOP / "bin" / "acquire.py")
    acq = importlib.util.module_from_spec(spec); spec.loader.exec_module(acq)
    SEED10 = [(61.0, 0.65), (81.5, 0.75), (98.8, 0.95), (114.1, 0.82), (108.5, 0.59),
              (130.3, 0.67), (156.5, 0.88), (139.8, 0.55), (174.0, 0.71), (223.0, 0.92)]
    k = acq.k1(mat, 5e-5)
    seed_pts += [{"H": P / (k * math.sqrt(V)), "V": V, "r0": 5e-5, "P": P,
                  "src": "seed10"} for P, V in SEED10]
    key = lambda t: (round(t["H"], 6), round(t["V"], 6), round(t["r0"], 9))
    for arm in ("maximin", "bandit"):
        sp = LOOP / "state" / f"{arm}.json"
        sp.parent.mkdir(parents=True, exist_ok=True)
        st = load_state(sp)
        # Add to `taken`, never replace it: a rerun of init mid-experiment must keep the
        # earlier rounds' picks excluded.
        seen = {key(t) for t in st["taken"]}
        st["taken"] = st["taken"] + [t for t in seed_pts if key(t) not in seen]
        sp.write_text(json.dumps(st, indent=1) + "\n")
    print(f"  seeded both arms: {len(tests)} frozen test points + {len(probes)} probe points "
          f"({len(probe_csvs)} probe design(s)) + 10 IN718 seed cases excluded from acquisition")
    if not CONFIG.is_file():
        CONFIG.write_text(json.dumps(_S, indent=1) + "\n")
    print(f"  run settings (state/config.json): {json.loads(CONFIG.read_text())}")
    return 0


def cmd_collect(a):
    """Snapshot this round's training collection, and assign the new cases split=train.

    Composition: ds86529 in full (its own 171/38/9/7 splits are inherited, never
    re-assigned -- it is `core`) UNION this arm's IN718 cases so far. The trainer
    uses split=train and validates on split=val, so Ti64's 38-member test split
    rides along and gives a non-regression read for free; IN718 is scored
    separately on the frozen ds 87432.

    Snapshotting matters because splits are per-simulation and MUTABLE: ds 62921
    moved 92/6/21 -> 85/6/20 within an hour and silently split a three-way
    comparison while every arm cited the same pk. A round that cites a frozen
    member list cannot drift under it.
    """
    from aiida import load_profile, orm
    load_profile()
    # An arm's round can span several families: the main acquisition plus a top-up
    # (round 1, restoring 25 usable after boundary exclusions) or replacements (round 2,
    # after the probe-overlap drops). Collecting only `round0N_<arm>` would silently
    # leave those out of training and leave the round short of the 25/arm design.
    _train_members(orm, a.arm, load_state(LOOP / "state" / f"{a.arm}.json"))  # refuses a bad anchor first
    fams = families(a.arm, a.round)
    names = set()
    for f in fams:
        names |= set((f / "CASE_ORDER.txt").read_text().split())
    print(f"  families: {[f.name for f in fams]} -> {len(names)} case names")
    batch_pks = [pk for fam, pk in solve_pks(a.round).items() if fam in {f.name for f in fams}]
    mine = _ingested(orm, names, batch_pks)
    # Compare against what actually SOLVED, not against every name in the family. A case
    # that failed permanently never produces a PinoSimulationData, so an exact
    # names-vs-ingested test can never pass and collect blocks forever. Round 5 hit this:
    # a licence outage killed 10 of 25 cases per arm, and the guard then reported
    # "only 25 of 35 ingested" on a round where 25 was the final answer.
    solved = {n for n in names if npz_ok(RUNS / n / "npz" / f"{n}.npz")}
    if not solved:
        print(f"  refusing: no case of {a.arm} round {a.round} has an exported NPZ under {RUNS}")
        return 1
    if len(mine) != len(solved):
        if batch_pks and batches_terminated(orm, batch_pks):
            # A finished batch has ingested everything it will: a case with an NPZ and no
            # simulation failed at ingest (non-finite fields, or the run failed after its
            # export). It is not trained on; name it and go on.
            have = {(orm.load_node(pk).label or "").split(" | ")[0] for pk in mine}
            print(f"  {len(solved - have)} exported case(s) have no simulation in AiiDA and are left "
                  f"out: {sorted(solved - have)}")
        else:
            print(f"  only {len(mine)} of {len(solved)} SOLVED cases ingested yet -- "
                  f"rerun when the batch finishes")
            return 2                     # not ready yet: tick.py waits and retries
    # Cases recorded in round0N_drops.json are excluded rather than trained on. Round 2
    # acquired 10 (maximin) and 8 (bandit) points that duplicate the OLD probe set,
    # because the probe was never in the acquisition exclusion list. Training on a case
    # the probe then measures makes E_r in-sample and biases b_r.
    # name -> reason. A drops file may list plain names (the round-2 shape, all of which
    # were probe duplicates) or map name -> reason. The reason is written to the node's
    # `exclusion_reason` extra, so it has to be per-case: round 3 drops a WALL-BOUNDED
    # case, and recording that as "duplicates the probe set" would be a false record of
    # why a simulation was excluded.
    PROBE_DUP = ("duplicates a point in the superseded probe set ds 89701; training on "
                 "it would make the probe measurement in-sample and bias E_r/b_r")
    drops = {}
    dpath = LOOP / f"round{a.round:02d}_drops.json"
    if dpath.is_file():
        for v in json.loads(dpath.read_text()).values():
            drops.update(v if isinstance(v, dict) else {n: PROBE_DUP for n in v})
    for pk in mine:                      # assign ONCE, at ingest; never re-assign later
        n = orm.load_node(pk)
        nm = (n.label or "").split(" | ")[0]
        if n.base.extras.all.get("split") in (None, "unassigned"):
            if nm in drops:
                n.base.extras.set("split", "excluded")
                n.base.extras.set("exclusion_reason", drops[nm])
            else:
                n.base.extras.set("split", "train")
    st_path = LOOP / "state" / f"{a.arm}.json"
    st = load_state(st_path)
    owned = sorted(set(st.get("owned_sim_pks", [])) | set(mine))
    st["owned_sim_pks"] = owned
    # CRITICAL: also add this round's points to `taken`, which is what drop_near reads.
    # `owned_sim_pks` is bookkeeping; omitting the taken update let round 2 re-buy EXACT
    # duplicates of round 1 (min normalized distance 0.0000, 11/25 and 8/25 across the
    # arms). The P-V experiment hit the softer version of this -- picks a median 0.034
    # apart -- and it is why EXCLUDE_RADIUS was raised to 0.06 there.
    seen = {(round(t["H"], 6), round(t["V"], 6), round(t["r0"], 9)) for t in st["taken"]}
    for f in fams:
      for r in csv.DictReader(open(f / "design.csv")):
        key = (round(float(r["H"]), 6), round(float(r["V"]), 6),
               round(float(r["r0_um"]) * 1e-6, 9))
        if key not in seen:
            st["taken"].append({"H": float(r["H"]), "V": float(r["V"]),
                                "r0": float(r["r0_um"]) * 1e-6, "P": float(r["P"]),
                                "src": f"{a.arm}_r{a.round:02d}"})
            seen.add(key)
    # region_n is the number of the arm's simulations each region has in training: the
    # owned cases labelled split=train, so dropped, failed and never-solved cases do not
    # count. It is the bandit's (n_r + 1) denominator and the x axis of its b_r fit.
    sys.path.insert(0, str(LOOP / "bin"))
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("acq", LOOP / "bin" / "acquire.py")
    _acq = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_acq)
    design = {}
    for r0 in range(1, a.round + 1):
        for fam in families(a.arm, r0):
            design.update({row["case_name"]: row for row in csv.DictReader(open(fam / "design.csv"))})
    counts = {}
    for pk in owned:
        n = orm.load_node(pk)
        row = design.get((n.label or "").split(" | ")[0])
        if row is None or n.base.extras.all.get("split") != "train":
            continue
        reg = _acq.region_of({"H": float(row["H"]), "V": float(row["V"]),
                              "r0": float(row["r0_um"]) * 1e-6})
        counts[reg] = counts.get(reg, 0) + 1
    st["region_n"] = counts
    st["round"] = a.round
    st_path.write_text(json.dumps(st, indent=1) + "\n")
    # Report the TRAIN-split count, not the raw collected count: cases in the drops file
    # were just marked `excluded` and are not trainable, so "+25" would overstate it.
    trainable = [pk for pk in owned
                 if orm.load_node(pk).base.extras.all.get("split") == "train"]
    excl = len(owned) - len(trainable)
    print(f"  {a.arm} round {a.round}: +{len(mine)} collected, "
          f"{len(trainable)} train / {excl} excluded, {len(owned)} owned total")
    # The compose hint MUST enumerate split=train members. Passing `--dataset-pk` unions
    # every split of the base dataset (test/val/excluded included) -- that is exit 306,
    # hit twice already -- and passing `owned` verbatim would train on the cases this
    # very function just excluded. tick.py's NEED_TRAIN text says the same thing; this
    # line used to contradict it.
    # The compose hint MUST enumerate split=train members as POSITIONAL pks. `--dataset-pk`
    # unions every split of a dataset (test/val/excluded included) and is refused as exit
    # 306 -- hit twice already. Passing `owned` verbatim would also train on the cases this
    # very function just excluded. tick.py's NEED_TRAIN text says the same; this line used
    # to print the --dataset-pk form and contradict it.
    anchor, _ = _train_members(orm, a.arm, st)
    print(f"  next: round.py train --arm {a.arm} --round {a.round} composes the train-only "
          f"collection ({len(anchor)} Ti64 anchor train + {len(trainable)} IN718 train = "
          f"{len(anchor) + len(trainable)} members) and submits the training members")
    return 0


def _ingested(orm, names, batch_pks):
    """pks of the PinoSimulationData nodes whose case name is in `names`.

    Taken from the outputs of the round's recorded batches when there are any, so that a
    simulation of the same name from another run in the store (for example an imported
    archive) cannot be counted; by label across the store for a family solved without a
    recorded batch."""
    if batch_pks:
        return sorted(pk for pk, lab in batch_simulations(orm, batch_pks)
                      if (lab or "").split(" | ")[0] in names)
    sim = {"node_type": {"like": "%PinoSimulationData%"}}
    rows = orm.QueryBuilder().append(orm.Data, filters=sim, project=["id", "label"]).all()
    return sorted({pk for pk, lab in rows if (lab or "").split(" | ")[0] in names})


def batch_simulations(orm, batch_pks):
    """(pk, label) of every PinoSimulationData the batch workflows and the processes they called
    created or returned. Process provenance runs through CALL links, which a QueryBuilder
    `with_ancestors` does not follow, so the called processes are walked explicitly."""
    from aiida.common.links import LinkType
    out = {}
    for bpk in batch_pks:
        batch = orm.load_node(int(bpk))
        for proc in [batch] + list(batch.called_descendants):
            for lt in (LinkType.CREATE, LinkType.RETURN):
                for triple in proc.base.links.get_outgoing(link_type=lt).all():
                    n = triple.node
                    if "PinoSimulationData" in (n.node_type or ""):
                        out[n.pk] = n.label
    return sorted(out.items())


def _train_members(orm, arm, st):
    """(anchor, own): the Ti64 anchor's split=train simulations and this arm's usable IN718
    ones so far. Enumerated as pks: `--dataset-pk` would union the anchor's test/val/excluded
    members too (refused as exit 306), and the arm's owned list includes excluded cases."""
    anchor = [n.pk for n in _dataset_members(orm, BASE_DS)
              if n.base.extras.all.get("split") == "train"]
    if not anchor:
        raise SystemExit(f"  refusing: the anchor dataset {BASE_DS} has no split=train "
                         "members; set IN718_LOOP_BASE_DS to the Ti-6Al-4V anchor's pk or UUID")
    own = [pk for pk in st.get("owned_sim_pks", [])
           if orm.load_node(pk).base.extras.all.get("split") == "train"]
    return anchor, own


def cmd_train(a):
    """Compose this round's train-only collection and submit its training members.

    Idempotent: the collection pk goes to roundNN_train_collections.json and each member's
    workflow pk to roundNN_train_pks.json as it is submitted, so a rerun submits only the
    members still missing. Every member trains on the same collection with its own seed and
    validates on VAL_DS; `probe` reads the trained models once they finish.
    """
    tp_path = LOOP / f"round{a.round:02d}_train_pks.json"
    tp = json.loads(tp_path.read_text()) if tp_path.is_file() else {}
    submit_lock()                        # held until the pks are recorded
    tp = json.loads(tp_path.read_text()) if tp_path.is_file() else {}
    todo = [k for k in range(1, a.members + 1) if f"{a.arm}_m{k}" not in tp]
    if not todo:
        print(f"  {a.arm} round {a.round}: all {a.members} members already submitted")
        return 0
    st = load_state(LOOP / "state" / f"{a.arm}.json")
    if st.get("round", 0) < a.round:
        print(f"  refusing: {a.arm} round {a.round} is not collected yet "
              f"(run round.py collect --arm {a.arm} --round {a.round})")
        return 1
    from aiida import load_profile, orm
    load_profile()
    cp = LOOP / f"round{a.round:02d}_train_collections.json"
    cols = json.loads(cp.read_text()) if cp.is_file() else {}
    if a.arm not in cols:
        anchor, own = _train_members(orm, a.arm, st)
        res = cli_json(
            AIIDA_PINO, "--json", "collection", "create",
            f"in718_twoalloy_{TAG}{a.arm}_r{a.round:02d}_TRAIN", *(anchor + own),
            "--description",
            f"TRAIN-ONLY, two-alloy {a.arm} round {a.round}: {len(anchor)} Ti64 "
            f"(ds{BASE_DS} split=train, enumerated) + {len(own)} usable IN718.",
            "--keyword", "in718", "--keyword", "two-alloy", "--keyword", "self-improve",
            "--keyword", "training", "--keyword", a.arm, "--keyword", f"round{a.round:02d}")
        if not res or "pk" not in res:
            print("  collection create failed; nothing submitted")
            return 1
        cols[a.arm] = res["pk"]
        cp.write_text(json.dumps(cols, indent=1) + "\n")
    base = json.loads((LOOP / "training_profile_twoalloy.json").read_text())
    (LOOP / "out").mkdir(exist_ok=True)
    for k in todo:
        prof = member_profile(base, a.arm, a.round, k)
        pf = LOOP / "out" / f"tp_{a.arm}_r{a.round:02d}_m{k}.json"
        pf.write_text(json.dumps(prof, indent=1) + "\n")
        res = cli_json(
            AIIDA_PINO, "--json", "submit", "train-eval", "--mode", "train",
            "--train-dataset", cols[a.arm], "--val-dataset", VAL_DS,
            "--python-code", PYTHON_CODE, "--training-profile", f"@{pf}",
            # the layout the profile trains on (moving frame, 30-frame window); without it
            # AiiDA refuses the composed collection's raw layout as an undeclared substitution
            "--derivation-profile", f"@{LOOP / 'derivation_86551.json'}",
            "--description", f"two-alloy loop, {a.arm} round {a.round}, member m{k} "
                             f"(seed {MEMBER_SEEDS[k - 1]}); its probe errors steer the next round",
            "--submitted-by", "in718_loop round.py train")
        if not res or "pk" not in res:
            print(f"  submission of m{k} failed; rerun to submit the missing members")
            return 1
        tp[f"{a.arm}_m{k}"] = int(res["pk"])
        tp_path.write_text(json.dumps(tp, indent=1) + "\n")
        try:
            orm.load_node(int(res["pk"])).label = prof["run_name"]
        except Exception:
            pass
    print(f"  {a.arm} round {a.round}: submitted members {todo} -> {tp_path.name}")
    return 0


def _models_from_trainings(orm, rnd, arm):
    """{member: model pk} for `arm`'s finished trainings of round `rnd`, or {} if any is not."""
    tp = json.loads((LOOP / f"round{rnd:02d}_train_pks.json").read_text())
    out = {}
    for k, pk in tp.items():
        if not k.startswith(arm):
            continue
        n = orm.load_node(pk)
        if not n.is_finished_ok:
            print(f"  training {k} (pk {pk}) has not finished cleanly")
            return {}
        out[k] = n.outputs.model.pk
    return out


def retire(rnd, stage, entries):
    """Move failed entries out of the round's pk file into roundNN_retired_pks.json, so the
    stage submits them again; the retired pks stay on record."""
    p = LOOP / f"round{rnd:02d}_retired_pks.json"
    ret = json.loads(p.read_text()) if p.is_file() else {}
    for k, pk in entries.items():
        ret.setdefault(stage, {}).setdefault(k, []).append(pk)
    p.write_text(json.dumps(ret, indent=1) + "\n")


def cmd_retry(a):
    """Set aside the FAILED solve batches, training members or probe evaluations of one arm and
    round, so the next tick (or the matching round.py step) submits them again. Entries that
    are running or finished cleanly are never touched."""
    from aiida import load_profile, orm
    load_profile()
    submit_lock()                        # held until the pks are recorded
    failed = lambda pk: (lambda n: n.is_terminated and not n.is_finished_ok)(orm.load_node(int(pk)))
    if a.stage == "solve":
        path = LOOP / f"round{a.round:02d}_solve_pks.json"
        rec = solve_pks(a.round)
        mine = {f.name: rec[f.name] for f in families(a.arm, a.round) if f.name in rec}
        # A batch that terminated with nothing exported is the SOLVE_FAILED case.
        gone = {}
        for fam, pk in mine.items():
            names = (LOOP / fam / "CASE_ORDER.txt").read_text().split()
            if orm.load_node(int(pk)).is_terminated and not any(
                    npz_ok(RUNS / n / "npz" / f"{n}.npz") for n in names):
                gone[fam] = pk
                for n in names:                  # the runner refuses an existing case directory
                    if (RUNS / n).is_dir() and not npz_ok(RUNS / n / "npz" / f"{n}.npz"):
                        shutil.rmtree(RUNS / n)
    else:
        path = LOOP / f"round{a.round:02d}_{'train' if a.stage == 'train' else 'probe_eval'}_pks.json"
        rec = json.loads(path.read_text()) if path.is_file() else {}
        gone = {k: pk for k, pk in rec.items()
                if k.split("|")[0].startswith(a.arm) and failed(pk)}
    if not gone:
        print(f"  nothing failed to retry for {a.arm} round {a.round} ({a.stage})")
        return 0
    retire(a.round, a.stage, gone)
    path.write_text(json.dumps({k: v for k, v in rec.items() if k not in gone}, indent=1) + "\n")
    print(f"  set aside {sorted(gone)} ({a.stage}); the next tick submits them again")
    return 0


def cmd_approve(a):
    fams = [f.name for f in families(a.arm, a.round)]
    if not fams:
        print(f"  refusing: {a.arm} round {a.round} has no rendered family to approve")
        return 1
    # Approve only what can be checked: every family submitted and its batch finished.
    from aiida import load_profile, orm
    load_profile()
    rec = solve_pks(a.round)
    waiting = [f for f in fams if f not in rec or not orm.load_node(int(rec[f])).is_terminated]
    if waiting:
        print(f"  refusing: {waiting} not submitted or still solving; approve once every case "
              f"of the round has finished and been checked")
        return 1
    add_approval(APPROVALS, a.arm, a.round, fams)
    print(f"  approved {a.arm} round {a.round}: {fams}. `tick.py --apply --gate` continues "
          f"with collect; a family acquired later needs its own approval.")
    return 0


def _dataset_members(orm, ds_pk):
    """Simulation nodes of a base dataset, by UUID (member pks go stale after migration)."""
    ds = orm.load_node(_ref(ds_pk))
    if "PinoDatasetCollectionData" not in (ds.node_type or ""):
        raise SystemExit(f"  refusing: node {ds_pk} is a {ds.node_type}, not a dataset collection; "
                         "set IN718_LOOP_BASE_DS to the Ti-6Al-4V anchor dataset")
    out = []
    for u in ds.base.attributes.all.get("member_simulation_uuids") or []:
        try:
            out.append(orm.load_node(u))
        except Exception:
            pass
    return out


def cmd_probe(a):
    """Score this round's models on the probe set per regime, update E_r and history.

    The probe (ds 89701, 16 cases, 4 per regime) is bought ONCE and re-evaluated every
    round. It is disjoint from the frozen test set, the acquisition lattice and every
    arm's training picks, which is what keeps the bandit from ever being steered on the
    scoring set -- that would hand it an advantage maximin does not have.

    Appends (n_r, E_r) per regime so `learning_rates` can fit the exponent b_r on the
    TREND. One history point per regime is not enough: b_r needs >=2, so it stays 1.0
    until round 3.
    """
    import statistics as st
    from aiida import load_profile, orm
    load_profile()
    from aiida_pino.console.model_metrics import recorded_kelvin_factor

    cols = json.loads((LOOP / "probe_regime_collections.json").read_text())
    if not cols:
        print("  refusing: probe_regime_collections.json names no probe collection")
        return 1
    submit_lock()                        # held until the pks are recorded
    mpath = LOOP / f"round{a.round:02d}_models.json"
    models = json.loads(mpath.read_text()) if mpath.is_file() else {}
    if not any(k.startswith(a.arm) for k in models):
        found = _models_from_trainings(orm, a.round, a.arm)
        if found:
            models.update(found)
            mpath.write_text(json.dumps(models, indent=1) + "\n")
    mine = {k: v for k, v in models.items() if k.startswith(a.arm)}
    if not mine:
        print(f"  no {a.arm} models for round {a.round}")
        return 1

    ev_path = LOOP / f"round{a.round:02d}_probe_eval_pks.json"
    evals = json.loads(ev_path.read_text()) if ev_path.is_file() else {}
    todo = [(mk, reg) for mk in mine for reg in cols if f"{mk}|{reg}" not in evals]
    refused = 0
    for mk, reg in todo:
        res = cli_json(AIIDA_PINO, "--json", "submit", "eval", "--model", str(mine[mk]),
            "--dataset", str(cols[reg]), "--python-code", PYTHON_CODE,
            "--derivation-profile", f"@{LOOP / 'derivation_86551.json'}", "--full-frames",
            "--description", f"PROBE eval round {a.round}, {mk}, {reg}. Feeds E_r and b_r. "
                             f"NOT the scoring set (that is the frozen IN718 test set).")
        if not res or "pk" not in res:
            refused += 1
            continue
        evals[f"{mk}|{reg}"] = int(res["pk"])
        ev_path.write_text(json.dumps(evals, indent=1) + "\n")   # after each, so a crash loses none
    if todo and refused == len(todo):
        print(f"  all {refused} probe evaluations were refused (see the errors above)")
        return 1

    mine_ev = {k: pk for k, pk in evals.items() if k.split("|")[0].startswith(a.arm)}
    pending = [k for k, pk in mine_ev.items() if not orm.load_node(pk).is_terminated]
    if pending:
        print(f"  {len(pending)} probe evals still running; rerun when they finish")
        return 2

    by = {}
    for k, pk in evals.items():
        mk, reg = k.split("|")
        if not mk.startswith(a.arm):
            continue
        n = orm.load_node(pk)
        if not n.is_finished_ok:             # failed, excepted or killed: no metrics to read
            continue
        m = n.outputs.metrics.get_dict(); ev = n.outputs.evaluation
        kf, _ = recorded_kelvin_factor(
            ev.base.attributes.all["evaluation_metadata"], ev.base.extras.all)
        by.setdefault(reg, []).append(m["T"]["beambox_abs_err_mean"] * kf)

    st_path = LOOP / "state" / f"{a.arm}.json"
    stt = load_state(st_path)
    # Idempotent per round: rerunning `probe` for the same round must not append a
    # second history point. Measured -- it did, and two points at the SAME n_r fit a
    # vertical line, producing a degenerate b_r pinned at the floor with a RankWarning.
    # History entries are [round, n_r, E_r]; a re-run overwrites its own round's entry.
    for reg, vals in by.items():
        E = st.median(vals)
        stt["region_err"][reg] = E
        h = stt.setdefault("history", {}).setdefault(reg, [])
        h[:] = [e for e in h if not (len(e) == 3 and e[0] == a.round)]
        h.append([a.round, stt["region_n"].get(reg, 0), E])
    st_path.write_text(json.dumps(stt, indent=1) + "\n")

    sys.path.insert(0, str(LOOP / "bin"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("acq", LOOP / "bin" / "acquire.py")
    acq = importlib.util.module_from_spec(spec); spec.loader.exec_module(acq)
    rates = acq.learning_rates(stt["history"])
    print(f"  {a.arm} round {a.round} probe:")
    for reg in sorted(by):
        pts = len(stt["history"][reg])
        b = rates.get(reg)
        print(f"    {reg:<12} E={stt['region_err'][reg]:6.1f} K  n={stt['region_n'].get(reg,0):<3} "
              f"history={pts}  b_r={'-' if b is None else f'{b:.3f}'}")
    if not rates:
        print("    b_r not fitted yet (needs >=2 history points per regime)")
    return 0


def cmd_state(a):
    for arm in ("maximin", "bandit"):
        p = LOOP / "state" / f"{arm}.json"
        if not p.is_file():
            print(f"  {arm}: no state yet"); continue
        st = json.loads(p.read_text())
        print(f"  {arm}: round {st['round']} | owned {len(st['owned_sim_pks'])} sims | "
              f"regions with error {len(st['region_err'])}")
        for r in sorted(st["region_n"]):
            e = st["region_err"].get(r)
            print(f"     {r:<22} n={st['region_n'][r]:<3} E={'-' if e is None else f'{e:.2f}'}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("acquire"); p.add_argument("--arm", required=True,
        choices=("maximin", "bandit")); p.add_argument("--n", type=int, default=25)
    p.add_argument("--round", type=int, default=None,
                   help="acquire INTO this round (default: state round + 1)")
    p.add_argument("--suffix", default="", help="family suffix, e.g. 'repl' or 'top'")
    p.add_argument("--h-max", type=float, default=None,
                   help="confine this acquisition to H <= this")
    p.set_defaults(fn=cmd_acquire)
    p = sub.add_parser("solve"); p.add_argument("--arm", required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--suffix", default="", help="family suffix, e.g. 'repl' or 'top'")
    p.set_defaults(fn=cmd_solve)
    p = sub.add_parser("collect"); p.add_argument("--arm", required=True)
    p.add_argument("--round", type=int, required=True); p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("train"); p.add_argument("--arm", required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--members", type=int, default=len(MEMBER_SEEDS), choices=range(1, len(MEMBER_SEEDS) + 1))
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("probe"); p.add_argument("--arm", required=True)
    p.add_argument("--round", type=int, required=True); p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("approve"); p.add_argument("--arm", required=True, choices=("maximin", "bandit"))
    p.add_argument("--round", type=int, required=True); p.set_defaults(fn=cmd_approve)
    p = sub.add_parser("retry"); p.add_argument("--arm", required=True, choices=("maximin", "bandit"))
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--stage", required=True, choices=("solve", "train", "probe"))
    p.set_defaults(fn=cmd_retry)
    p = sub.add_parser("init"); p.set_defaults(fn=cmd_init)
    p = sub.add_parser("state"); p.set_defaults(fn=cmd_state)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
