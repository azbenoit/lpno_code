#!/usr/bin/env python3
"""The exploration domain must cover every requirement -- and H must be computed with the
per-case K1, not a constant.

On 2026-09-10 an anchor table built with a CONSTANT K1 = 18.1249 put the experimental
IN718 conditions at H 13.8-18.3, "inside the Ti64 overlap region", and the box was
narrowed to H <= 28 on that basis. King H has K1 proportional to r0^1.5; with the
per-case k1(mat, r0) the same anchors are at H 21.6-46.8, ABOVE the Ti64 ceiling of 20.8,
and the narrowed box cut off every validation target. Both halves are pinned below.

Run: python bin/test_domain_bounds.py
"""
import importlib.util, json, math, pathlib

LOOP = pathlib.Path(__file__).resolve().parent.parent
from _repo import MATERIALS as MATS
# measured 2026-09-10 with the per-case k1, and cross-checked against Materials
# variations' independent sweep of 287 solves (their King H / this one = 1.00)
EXP_H = (21.6, 46.8)          # validation/configs/*in718*, n=16
TI64_TRAIN_H_MAX = 20.8       # ds86529 split=train, n=171
IN718_TEST_H_MAX = 46.8       # ds87436, n=24
EXP_R_UM_MIN = 24.5
IN718_TEST_R_UM_MIN = 23.0


def main():
    spec = importlib.util.spec_from_file_location("acq", LOOP / "bin" / "acquire.py")
    acq = importlib.util.module_from_spec(spec); spec.loader.exec_module(acq)
    mat = json.loads((MATS / "inconel718" / "inconel718.json").read_text())

    # 1. K1 must depend on r0. A constant K1 is the bug this file exists for.
    k_small, k_big = acq.k1(mat, 22.0e-6), acq.k1(mat, 65.0e-6)
    assert k_big > k_small * 2, (
        f"k1 barely varies with r0 ({k_small:.3f} -> {k_big:.3f}); King H needs K1 ~ r0^1.5")
    ratio = k_big / k_small
    expect = (65.0 / 22.0) ** 1.5
    assert abs(ratio - expect) / expect < 0.02, f"k1 scales {ratio:.2f}, expected {expect:.2f}"

    # 2. the box must reach every validation target
    assert acq.H_HI >= EXP_H[1], (
        f"H_HI={acq.H_HI} cuts off the experimental anchors, which reach {EXP_H[1]}")
    assert acq.H_HI >= IN718_TEST_H_MAX, (
        f"H_HI={acq.H_HI} cuts off the frozen test set, which reaches {IN718_TEST_H_MAX}")
    assert acq.H_LO <= min(EXP_H[0], TI64_TRAIN_H_MAX)
    assert acq.R_LO * 1e6 <= min(EXP_R_UM_MIN, IN718_TEST_R_UM_MIN), \
        "r0 must reach the experimental and test minima"

    # 3. the anchors are ABOVE the Ti64 ceiling -- the fact that inverted the argument
    assert EXP_H[0] > TI64_TRAIN_H_MAX, (
        "the experimental anchors are above the Ti64 training ceiling; any 'confine to the "
        "Ti64 region' step excludes every validation target and must be a deliberate choice")

    # 4. and the lattice respects the box
    L = acq.lattice(mat)
    assert not [c for c in L if c["H"] > acq.H_HI + 1e-9]
    print(f"OK  H {acq.H_LO}-{acq.H_HI}, r0 {acq.R_LO*1e6:.0f}-{acq.R_HI*1e6:.0f} um, "
          f"k1 scales {ratio:.2f}x over the r0 range, {len(L)} lattice points; "
          f"covers anchors to {EXP_H[1]} and test to {IN718_TEST_H_MAX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
