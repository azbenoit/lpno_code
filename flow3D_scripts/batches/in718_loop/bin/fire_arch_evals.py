"""Detached watcher: when the GAOT/RIGNO joint trainings finish, fire the comparison evals.

Same slices as the FNO joint's closing evaluation (fire_joint_evals.sh) minus the Ti64
experimental anchor (its beam-box measurement reads 0.000 for every model; withheld).
Evals inherit the training's hydra_overrides (model_registry.py: APPENDED), so 103774's plain
arch.rigno.* pins reach the eval and no eval-time pin is needed.
Run detached: setsid nohup python bin/fire_arch_evals.py > out/arch_evals.log 2>&1 &
"""
import json, pathlib, subprocess, sys, time, importlib.util
from aiida import load_profile, orm
load_profile()

HERE = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("lab", HERE / "bin/label.py")
lab = importlib.util.module_from_spec(spec); spec.loader.exec_module(lab)
LEDGER = HERE / "round05_archcompare_train_pks.json"
OUT = HERE / "round05_archcompare_eval_pks.json"
SL = {"all": 87436, "overlap": 87534, "extrap": 87537, "ti64": 86539, "in718_exp": 102329}
DERIV = str((HERE / "derivation_86551.json").resolve())
SIGN = "self-improve [46ad75]"

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def fire(arm, mpk):
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    for sl, ds in SL.items():
        key = f"{arm}_{sl}"
        if key in done:
            continue
        r = subprocess.run(["aiida-pino", "submit", "eval", "--model", str(mpk), "--dataset", str(ds),
                            "--python-code", "1292", "--derivation-profile", f"@{DERIV}", "--full-frames",
                            "--submitted-by", SIGN,
                            "--description", f"ARCH COMPARE {arm.upper()} joint model (419 members = 171 Ti64 + "
                                             f"248 IN718, both arms pooled; tabled steady {arm.upper()} hyperparameters), "
                                             f"slice={sl}. Same slices as the FNO joint's closing evaluation."],
                           capture_output=True, text=True)
        pk = [l.split()[1] for l in r.stdout.splitlines() if l.startswith("pk:")]
        if pk:
            done[key] = int(pk[0]); lab.describe(int(pk[0]), f"eval[{sl}]", f"{arm}-joint", 5, 1)
            log(f"  {key}: eval {pk[0]} on ds{ds}")
        else:
            log(f"  {key}: SUBMIT FAILED: {(r.stdout + r.stderr)[-300:]}")
        OUT.write_text(json.dumps(done, indent=1) + "\n")

fired = set()
while True:
    tr = json.loads(LEDGER.read_text())
    live = {k.split("_")[0]: pk for k, pk in tr.items() if not (k.endswith("RETIRED") or k.endswith("KILLED"))}
    for arm, pk in live.items():
        if arm in fired:
            continue
        n = orm.load_node(pk)
        if not n.is_terminated:
            continue
        if n.exit_status == 0 and "model" in n.outputs:
            mpk = n.outputs.model.pk
            log(f"{arm} train {pk} finished -> model {mpk}; firing {len(SL)} evals")
            fire(arm, mpk)
        else:
            log(f"{arm} train {pk} TERMINAL exit={n.exit_status} {str(n.exit_message)[:80]} -- no evals")
        fired.add(arm)
    if fired >= set(live):
        log("all arms handled; exiting"); sys.exit(0)
    time.sleep(300)
