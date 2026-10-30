"""Fixtures for the WRF v4.7.1 WRF-Solar EPS hook oracle (run_eps.F90).

    python make_inputs.py OUT_DIR [--seed N]

Writes OUT_DIR/inputs.bin in run_eps.F90's layout.  Deterministic (numpy
default_rng seeded); every value is a float32 word.

Perturbation fields are the oracle's inputs (the pattern generator is graded
separately): Gaussian draws at STOCHPERT.TBL's standard deviations, clipped
at its 3-sigma cutoff as RAND_PERT_UPDATE clips them, plus a stress share
far outside that range so 1 + pert*factor crosses zero and goes negative.
State covers every clamp: zero, signed zero, subnormal and tiny mixing
ratios against QX_MIN = 1e-12 and NI_MIN = 1e-6, negative ones (WRF's moist
arrays carry them), albedo against both ends of [0, 1], asymmetry across
[-1, 1], vertical velocity of both signs.  Cases vary the namelist factors:
WRF-Solar's (1.0), zero, large, negative and mixed.
"""
import argparse
import pathlib

import numpy as np

F = np.float32
# STOCHPERT.TBL (WRF v4.7.1 run/): std per field, cutoff 3.
STD = {"albedo": 0.10, "aod": 0.25, "angstrom": 0.10, "assymfac": 0.05,
       "qvapor": 0.05, "qcloud": 0.10, "qice": 0.10, "qsnow": 0.10,
       "ni": 0.05, "th": 0.001, "w": 0.10}
ORDER = ("albedo", "aod", "angstrom", "assymfac", "qvapor", "qcloud", "qice",
         "qsnow", "ni", "th", "w")


def pert(rng, shape, std):
    p = np.clip(rng.normal(0.0, std, shape), -3 * std, 3 * std)
    stress = rng.random(shape) < 0.08
    p[stress] = rng.uniform(-4.0, 4.0, stress.sum())
    p[rng.random(shape) < 0.01] = 0.0
    neg0 = rng.random(shape) < 0.005
    p = p.astype(F)
    p[neg0] = F(-0.0)
    return p


def mixing(rng, shape, scale, tiny):
    q = rng.lognormal(np.log(scale), 2.0, shape)
    r = rng.random(shape)
    q[r < 0.15] = 0.0
    q[(r >= 0.15) & (r < 0.2)] = rng.uniform(0.0, tiny * 10, ((r >= 0.15) & (r < 0.2)).sum())
    q[(r >= 0.2) & (r < 0.23)] = -rng.uniform(0.0, tiny * 10, ((r >= 0.2) & (r < 0.23)).sum())
    q = q.astype(F)
    q[(r >= 0.23) & (r < 0.24)] = F(-0.0)
    sub = (r >= 0.24) & (r < 0.25)
    q[sub] = (rng.integers(1, 1 << 23, sub.sum()).astype(np.uint32)).view(F)
    q[(r >= 0.25) & (r < 0.26)] = F(tiny)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=20261030)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    ni, nk = 128, 24
    kme = nk + 1
    shape = (kme, ni)            # written transposed: Fortran (ni, kme)

    cases = [
        # farms: albedo aod angexp aerasy qv qc qs | deng: qv qc t w |
        # thom: qv qc qi qs ni | cld3: qv t
        [1.0] * 18,
        [0.0] * 18,
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        [5.0, 3.0, 7.5, 9.0, 4.0, 6.0, 2.5, 3.0, 8.0, 50.0, 0.5, 6.0, 7.0, 8.0, 9.0, 3.0, 5.0, 40.0],
        [-1.0, -2.0, -0.5, -3.0, -1.5, -0.25, -4.0, -2.0, -1.0, -10.0, -0.75, -2.0, -3.0, -0.5, -1.0, -2.5, -1.0, -20.0],
        [0.3, 1.7, 0.1, 2.2, 0.9, 1.1, 0.6, 1.3, 0.2, 0.7, 1.9, 0.45, 1.25, 0.8, 2.75, 0.05, 0.33, 0.66],
    ]
    fac = np.array(cases, F)

    perts = [pert(rng, shape, STD[name]) for name in ORDER]

    qv = mixing(rng, shape, 5e-3, 1e-12)
    moist = rng.random(shape) < 0.5
    qv[moist] = rng.uniform(1e-6, 0.025, moist.sum()).astype(F)
    qc =mixing(rng, shape, 1e-4, 1e-12)
    qi = mixing(rng, shape, 3e-5, 1e-12)
    qs = mixing(rng, shape, 1e-4, 1e-12)
    qn = mixing(rng, shape, 1e4, 1e-6)
    th = rng.uniform(250.0, 450.0, shape).astype(F)
    t = rng.uniform(190.0, 315.0, shape).astype(F)
    w = rng.normal(0.0, 2.0, shape).astype(F)
    w[rng.random(shape) < 0.05] = F(0.0)
    w[rng.random(shape) < 0.02] = F(-0.0)
    w[0] = F(0.0)                                   # the surface: w = 0

    def edge2(lo, hi, special):
        x = rng.uniform(lo, hi, ni).astype(F)
        r = rng.random(ni)
        for i, value in enumerate(special):
            x[(r >= i * 0.03) & (r < (i + 1) * 0.03)] = F(value)
        return x
    albedo = edge2(0.0, 1.0, (0.0, 1.0, -0.0, 0.08, 0.99999994))
    aod = edge2(0.0, 3.0, (0.0, -0.0, 1e-3, 1e-40))
    angexp = edge2(0.0, 2.5, (0.0, -0.0, 1.3))
    aerasy = edge2(-1.0, 1.0, (0.0, -0.0, 1.0, -1.0, 0.7))

    # Post-microphysics stand-ins: the input moved by a factor in [0, 2),
    # zero, or replaced outright.
    def post(x):
        y = (x * rng.uniform(0.0, 2.0, x.shape)).astype(F)
        r = rng.random(x.shape)
        y[r < 0.1] = F(0.0)
        y[(r >= 0.1) & (r < 0.15)] = rng.permutation(x.ravel()).reshape(x.shape)[(r >= 0.1) & (r < 0.15)]
        return y
    mp = [post(x) for x in (qv, qc, qi, qs, qn)]

    for arr in [*perts, qv, qc, qi, qs, qn, th, t, *mp]:
        arr[nk] = F(0.0)                           # mass arrays' unused top

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "inputs.bin", "wb") as f:
        np.array([ni, nk, len(cases)], "<i4").tofile(f)
        fac.astype("<f4").tofile(f)              # (ncase, 18) == Fortran (18, ncase)
        for arr in [*perts, qv, qc, qi, qs, qn, th, t, w]:
            np.ascontiguousarray(arr).astype("<f4").tofile(f)   # (kme, ni) == Fortran (ni, kme)
        for arr in (albedo, aod, angexp, aerasy):
            arr.astype("<f4").tofile(f)
        for arr in mp:
            np.ascontiguousarray(arr).astype("<f4").tofile(f)
    print("wrote", out / "inputs.bin", ni, nk, len(cases))


if __name__ == "__main__":
    main()
