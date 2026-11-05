# Tracked batch records

`runs/` is untracked by design (solver outputs, gigabytes per case), which would leave
the *method* of these batches (their drivers, design tables, per-round configs and
round state) visible only on the machine that ran them. This directory is the
tracked copy of that method, without any solver output: no NPZ, no `flsgrf`, no
case directories. Each subdirectory is a batch that ran from
`runs/batches/<same name>/`; the live copy there stays the working one, this copy
is the record.

| batch | what it is | drivers |
| --- | --- | --- |
| `in718_testset_20260904/` | the frozen 24-case IN718 test set (25 generated, one excluded) (`generate.py`, `design.csv`, `configs/`, the frozen material under `materials/in718_full/`, `EXCLUSIONS.md`) | `bin/generate.py`, `bin/check_cases.py` |
| `in718_loop/` | the two-alloy active-learning loop: five rounds of two acquisition arms (`bandit`, `maximin`) plus probes and replacements, each `roundNN_<arm>/` holding the 25 rendered configs, `design.csv`, `CASE_ORDER.txt`, its frozen material copy and the AiiDA submission profiles; `state/` (arm state), `roundNN_*_pks.json` / `roundNN_models.json` (the AiiDA node ids of each round's solves, trainings and evaluations), `regime_groups.json`, `training_profile_twoalloy.json`, `tick_log.jsonl`, `out/` (acquisition picks), `figs/points.json` | `bin/acquire.py` (pick), `make_configs.py` (render), `round.py` (the stages of a round), `tick.py` (chains them, see below), `check_round.py` (boundary check), `label.py`, `compare.py`, `plot_zeroshot.py`; `test_domain_bounds.py`, `test_family_naming.py` and `test_tick_apply.py` are its own checks |

## What changed against the copies under `runs/`

The files were written with the authors' absolute paths; the only change here
makes them portable:

- every rendered config names its frozen material as
  `../materials/in718_full/in718_full.json` (the runner resolves `material.file`
  against the config's own directory first), instead of an absolute path. The
  rendered `prepin` is byte-identical either way; both generators now write the
  relative form. An AiiDA batch receives each config as data rather than as a file,
  so `round.py solve` submits the path resolved to an absolute one;
- the drivers find the checkout through `bin/_repo.py` (a walk-up to the directory
  holding `python/run_rectangular_case.py`) rather than by counting parent
  directories, because that count differs between `runs/batches/<name>/` and
  `batches/<name>/`. `_repo.RUNS` is the repository's `runs/`, so a re-run from
  here still writes its solves to the untracked place.

`tests/test_no_hardcoded_home_paths.py` sweeps every text file under `batches/`
for a home path. Left out: `figs/*.png` (regenerable by
`plot_zeroshot.py`), `__pycache__`, and `.bak`/`.dead` files.

## Running a driver from here

```bash
cd batches/in718_loop
python3 bin/test_domain_bounds.py           # no AiiDA needed
python3 bin/make_configs.py --help          # renders a round from a picks file
```

`acquire.py`, `round.py`, `tick.py`, `label.py` and `compare.py` talk to an AiiDA
store. The `roundNN_*.json` files record the dataset, model and workflow pks of the
experiment in the authors' store; AiiDA node ids change from one installation to
another, so these files are a record of that run, and a new run writes its own.

## Running the acquisition loop

`tick.py` chains the stages of `round.py` and runs the loop either unattended or with a
person checking each round's simulations before the loop trains on them. Each round, for
each half (`maximin` and `bandit`):

1. **acquire, solve**: pick 25 cases (maximin coverage, or the regime bandit) and submit
   them as one `aiida-pino submit flow3d-batch`, whose workflow pk goes to
   `roundNN_solve_pks.json`. The round counts as solved once that workflow has
   terminated; a case without an exported NPZ by then is a failed solve;
2. **check, collect**: run the boundary check, then label the round's simulations (the
   outputs of its batches) `split=train`, or `excluded` for the cases in
   `roundNN_drops.json`;
3. **train**: compose the train-only collection (the Ti-6Al-4V anchor's train split and
   the half's IN718 cases so far) and submit three FNO members (seeds 42, 43 and 44)
   with `training_profile_twoalloy.json`;
4. **probe**: evaluate the members on the probe set per regime, which updates the
   bandit's per-regime error and learning-rate fit for the next round.

**Before the first round.** The loop needs an AiiDA store holding the Ti-6Al-4V anchor
dataset and the solved probe set, a FLOW-3D licence for the solves and a GPU for the
trainings. The round files in `in718_loop/` record the published run, so a new run
starts in a copy of the directory beside it under `batches/` (the drivers read the frozen
test set from `../in718_testset_20260904/`), keeping only `bin/`,
`training_profile_twoalloy.json`, `derivation_86551.json`, `probe_regime_collections.json`
and the probe design (`round00_probe*/design.csv`, which `init` excludes from
acquisition). Point the loop at the store with environment variables, each a pk or a UUID:

| variable | what it names | default |
| --- | --- | --- |
| `IN718_LOOP_BASE_DS` | the Ti-6Al-4V anchor dataset | the authors' pk |
| `IN718_LOOP_VAL_DS` | the validation set of every training member | `0b657a95`, `ti64_anchorbox_lhs150_val_split` |
| `IN718_LOOP_PYTHON_CODE` | the AiiDA Code that runs the jobs | the authors' pk |
| `IN718_LOOP_TAG` | a prefix for the run's case and dataset names | none |

Set `IN718_LOOP_TAG` (for example `run2`) whenever the store or the `runs/` directory
already holds a run of the loop, such as the imported deposit: case names are otherwise
the same as the published ones. `probe_regime_collections.json` maps each regime to its
probe collection in the store. The tag takes letters, digits and `-`. Then run
`python3 bin/round.py init`, which writes each half's exclusion list (the frozen test set
and every probe design present) and records the four settings in `state/config.json`.
Later calls read them from there and refuse a variable set to a different value, so a
timer whose environment lacks them cannot switch the run to the defaults. Run every
command from the loop directory with the Python of the environment that has aiida-pino
installed; the drivers call that environment's `aiida-pino`.

**Unattended.**

```bash
python3 bin/tick.py --apply --every 1800      # one tick every 30 minutes until done
```

A cron or systemd timer works the same way, for example
`cd <loop directory> && <env>/bin/python bin/tick.py --apply` every 30 minutes. A lock
(`state/.tick.lock`) keeps two ticks from overlapping. In this mode the boundary check
records in `roundNN_drops.json` every case whose melt pool reaches the domain boundary:
within 50 µm of the plate floor, within 10 µm of an x wall, or across 0.8 of the domain
width. `collect` then excludes those cases. Cases that do not melt are lack-of-fusion
samples and stay in training.

**With a person checking each round.**

```bash
python3 bin/tick.py --apply --gate --every 1800
```

When a half's solves finish, that half holds and the tick prints what to do. Check the
round with `python3 bin/check_round.py --arm <arm> --round <N>`, which prints each case's
peak temperature, melt depth, floor and wall margins and width fraction, and look at the
simulations themselves if needed. Record the cases to drop, either with
`check_round.py ... --write-drops` or by editing `roundNN_drops.json` (name to reason). To
replace dropped cases, run `round.py acquire --arm <arm> --round <N> --suffix repl --n <k>`
and `round.py solve --arm <arm> --round <N> --suffix repl` (a second replacement for the
same round uses `--suffix top`); the replacement excludes the round's own picks, and the
loop waits for its solves. Replacements are acquired before the round is trained on;
`acquire` refuses one once the round's training members are submitted. Then approve the
round, which `approve` refuses while any of its families is still solving:

```bash
python3 bin/round.py approve --arm <arm> --round <N>
```

The approval covers the round's families that exist at that moment, so a replacement
acquired afterwards holds the round again until it is checked and approved. The next
ticks collect, train, probe and acquire the next round, which holds again once it is
solved.

**Without `--apply`**, `tick.py` only prints and logs the next stage of each half, and
the stages can be run by hand with the `round.py` subcommands it names.

The loop stops after the final round (`--final-round`, default 5); on a failed training,
evaluation or solve batch (a batch that terminated with nothing exported); and when a
stage fails three ticks in a row or reports "not ready yet" twelve ticks in a row. A stage
command running longer than three hours is killed and counts as one failure. The counts
are kept in `state/tick_failures.json`, and a stop writes `state/STOPPED` with its reason;
every later tick does nothing while that file exists, so a timer stops too. To resume,
fix the cause, run `round.py retry --arm <arm> --round <N> --stage solve|train|probe` to
set aside the failed batch, members or evaluations (they are resubmitted, and their pks
kept in `roundNN_retired_pks.json`), and remove `state/STOPPED`. A case whose NPZ was
exported but never ingested is named by `collect` and left out of training. The loop
defers a stage while free disk (where the runs and the AiiDA repository are) or available
RAM is below that stage's floor. Submitted copies of the configs, which carry absolute
paths, go to `runs/_in718_loop_submissions/`, outside the tracked directory. Every stage is
derived from what is on disk and in AiiDA, so an interrupted loop resumes where it was,
and each tick is appended to `tick_log.jsonl`. `python3 bin/test_tick_apply.py` checks the
commands each stage runs, how a round is judged solved, the gate, the retry limits and
the drop rules without AiiDA.

Possible additions that would make the loop more robust:

- acquire replacements for dropped or failed cases automatically, so that each round
  keeps 25 usable cases (an unattended round now goes short by the cases it drops);
- wait and retry when the licence server is down, as the runner already does for disk
  space (in round 5 of the experiment one outage failed 20 solves within two minutes);
- render each case with `--prepare-only` before the batch is submitted;
- notify a person when a half holds or the loop stops;
- check the probe collections and the anchor's training-case count when `init` runs, so
  a wrong pk fails before the first solve (the anchor's type is checked at `collect`).
