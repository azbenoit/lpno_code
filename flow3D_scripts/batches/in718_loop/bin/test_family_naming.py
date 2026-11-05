#!/usr/bin/env python3
"""One check on the family-naming contract shared by acquire/make_configs/solve/collect.

Round 3's replacement had to be recovered by hand because these four disagreed:
`acquire` could only write round N+1, `make_configs` could only name `_rNN_`, `solve`
only looked at `roundNN_<arm>`, while `collect` already read the `top`/`repl` families.
The rename that recovered it left a stale absolute materials path behind.

Run: python bin/test_family_naming.py
"""
import json, pathlib, re, subprocess, sys, tempfile

LOOP = pathlib.Path(__file__).resolve().parent.parent
PY_ = sys.executable


def main():
    src = LOOP / "bin" / "round.py"
    txt = src.read_text()
    mk = (LOOP / "bin" / "make_configs.py").read_text()

    # 1. all four sites build the family name the same way: roundNN_<arm><suffix>
    assert 'f"round{a.round:02d}_{a.arm}{getattr(a, \'suffix\', \'\') or \'\'}"' in txt, \
        "cmd_solve no longer honours --suffix"
    assert 'f"round{a.round:02d}_{a.arm}{a.suffix}"' in mk, \
        "make_configs no longer honours --suffix"
    assert '("top", "repl")' in txt, "cmd_collect family list changed"

    # 2. acquire accepts the overrides and defaults to state round + 1
    for flag in ("--round", "--suffix"):
        out = subprocess.run([PY_, str(src), "acquire", "--help"],
                             capture_output=True, text=True).stdout
        assert flag in out, f"acquire lost {flag}"

    # 3. make_configs names cases with the suffix INSIDE the round token, so a repl case
    #    can never collide with a plain case of the same index (17 solves died on that).
    with tempfile.TemporaryDirectory() as td:
        picks = pathlib.Path(td) / "p.json"
        picks.write_text(json.dumps([
            {"H": 9.0, "V": 0.5, "P": 100.0, "r0": 30e-6,
             "region": "conduction", "regime": "overlap"}]))
        r = subprocess.run([PY_, str(LOOP / "bin" / "make_configs.py"),
                            "--picks", str(picks), "--arm", "maximin",
                            "--round", "99", "--suffix", "selftest"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-400:]
        fam = LOOP / "round99_maximinselftest"
        try:
            assert fam.is_dir(), "suffix did not reach the directory name"
            name = (fam / "CASE_ORDER.txt").read_text().strip()
            assert name == "in718_maximin_r99selftest_000", name
            cfg = json.loads((fam / "configs" / f"{name}.json").read_text())
            assert cfg["case_name"] == name
            # 4. every path in the config points INSIDE its own family -- the rename bug
            blob = json.dumps(cfg)
            stray = [m for m in re.findall(r'"(/[^"]+)"', blob)
                     if "in718_loop" in m and "round99_maximinselftest" not in m]
            assert not stray, f"config references another family: {stray}"
        finally:
            import shutil; shutil.rmtree(fam, ignore_errors=True)

    print("OK  family naming consistent across acquire / make_configs / solve / collect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
