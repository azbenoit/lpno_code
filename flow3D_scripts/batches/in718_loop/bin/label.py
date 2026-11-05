#!/usr/bin/env python3
"""Systematic labels for every node this loop creates.

The gap this closes: `aiida-pino describe` was a separate manual step done from memory
after submitting, so it survived exactly as long as attention did. Six live round-1
trainings went out bare, and someone had to notice. Labels are DERIVED from (kind, arm,
round, member) so submission and description cannot come apart.

Call `describe(pk, **spec)` from every submission path, and `audit()` to catch anything
that slipped.
"""
import json, subprocess
from pathlib import Path

LOOP = Path(__file__).resolve().parent.parent
OWNER = "self-learning session [d0780f6d]"

CONTEXT = (
    "Two-alloy IN718 self-improvement loop. Question: does error-driven allocation "
    "(bandit) beat space-filling (maximin) at teaching a Ti64-trained steady FNO a "
    "second alloy? Round-0 zero-shot baseline: 96.9 K substrate / fl IoU 0.203 in the "
    "overlap regime, against a Ti64 reference of 11.46 K / 0.9006."
)
FLAGS = (
    "custom.per_case_t_boil=true (per-case mask fill; both alloys carry mat_Tv). "
    "PINO_LOADER_WORKERS=0 is CORRECT here and must not be 'fixed': the w=0 cliff scales "
    "with frames per case and a steady config is one frame per case. "
    "Val set ds82782 is 9 Ti64 members, shared by both arms so validation is identical."
)


def title(kind, arm, rnd, member=None):
    m = f" m{member}" if member else ""
    return f"twoalloy r{rnd:02d} {arm}{m} {kind}"


def description(kind, arm, rnd, member=None, **extra):
    bits = [f"{CONTEXT}", f"Arm: {arm}. Round: {rnd}." +
            (f" Ensemble member {member} (seed {41 + member})." if member else ""),
            f"Kind: {kind}. Owner: {OWNER}."]
    for k, v in extra.items():
        bits.append(f"{k}: {v}")
    if kind in ("train", "eval"):
        bits.append(FLAGS)
    return " ".join(bits)


def describe(pk, kind, arm, rnd, member=None, **extra):
    t = title(kind, arm, rnd, member)
    d = description(kind, arm, rnd, member, **extra)
    subprocess.run(["aiida-pino", "describe", str(pk), d, "--title", t],
                   capture_output=True, text=True)
    return t


def audit():
    """Every node this loop owns that has no label -- the check that catches a miss."""
    from aiida import load_profile, orm
    load_profile()
    pks = set()
    for f in LOOP.glob("round*_pks.json"):
        pks |= set(json.loads(f.read_text()).values())
    for f in LOOP.glob("round*_eval_pks.json"):
        pks |= set(json.loads(f.read_text()).values())
    bare = []
    for pk in sorted(pks):
        try:
            n = orm.load_node(pk)
        except Exception:
            continue
        if not (n.label or "").strip() or not (n.description or "").strip():
            bare.append(pk)
    return bare


if __name__ == "__main__":
    b = audit()
    print(f"  unlabelled nodes owned by this loop: {len(b)}")
    for pk in b[:20]:
        print(f"    {pk}")
