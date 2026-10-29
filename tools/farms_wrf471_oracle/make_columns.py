"""Column fixtures for the FARMS oracle (tools/farms_wrf471_oracle).

    python make_columns.py BUILD_DIR

Writes ``<scenario>.in.bin`` (run_farms.F90's stream layout) for each
scenario and ``columns.npz`` holding the same inputs, keyed
``<scenario>/<name>``.  Deterministic (fixed seeds), so the fixture is a
pure function of this file.

A scenario is one Farms_driver call: its scalars (has_reqc/has_reqi/has_reqs,
aer_opt, julian) apply to every column of the call.  Columns span the sky
classes FARMS branches on -- clear; liquid, ice and snow alone; liquid+ice;
all three; optically thin (tau < 1) and thick (tau > TAU_MAX = 300) clouds;
cloud optical depths around TAU_MIN; partial cloud fraction; condensate in
the top model layer only (Integrate_1var stops at kte - 1); radii under the
climatology thresholds with and without cloud fraction -- crossed with
aerosol off (WRF's aer_opt = 0 zeros) and on, surface albedo, and the sun
from below the horizon through grazing to overhead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

F = np.float32
NZ = 30
#: (name, has_reqc, has_reqi, has_reqs, aer_opt, julian, seed)
SCENARIOS = (
    ("req1_summer", 1, 1, 1, 1, 172.6, 11),
    ("req0_jan", 0, 0, 0, 0, 0.2, 12),
    ("req1_dec", 1, 1, 1, 0, 364.97, 13),
    ("req0_feb", 0, 0, 0, 1, 59.5, 14),
)
SKIES = ("clear", "liquid", "ice", "snow", "liquid_ice", "mixed3",
         "thin_liquid", "thick", "tau_min", "partial", "top_only",
         "small_re", "small_re_clear", "huge_re")
COSZEN_SET = (-0.5, -1.0e-3, 0.0, 1.0e-6, 1.0e-4, 1.0e-3, 0.01, 0.03, 0.05,
              0.1, 0.17, 0.25, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97, 0.995,
              1.0)
N_RANDOM = 1500
FIELDS3 = ("p8w", "rho", "dz8w", "qv", "qi", "qs", "qc", "re_cloud",
           "re_ice", "re_snow", "cldfra")
FIELDS2 = ("albedo", "coszen_loc", "aerssa2d", "aerasy2d", "aod5502d",
           "angexp2d")


def _profile(rng):
    """A moist atmosphere on NZ layers plus the kme row (nz + 1 levels)."""
    kme = NZ + 1
    psfc = rng.uniform(60000.0, 103500.0)
    ptop = 5000.0
    eta = np.linspace(1.0, 0.0, kme) ** rng.uniform(1.1, 1.6)
    p8w = ptop + (psfc - ptop) * eta
    pmid = 0.5 * (p8w[:-1] + p8w[1:])
    tsfc = rng.uniform(235.0, 310.0)
    z_approx = 7000.0 * np.log(psfc / pmid)
    t = np.maximum(tsfc - 0.0065 * z_approx, 195.0)
    qv = rng.uniform(1.0e-4, 0.02) * (pmid / psfc) ** 3.0
    rd, g = 287.0, 9.81
    dz = rd * t * (1.0 + 0.61 * qv) / g * np.log(p8w[:-1] / p8w[1:])
    rho = pmid / (rd * t * (1.0 + 1.608 * qv)) * (1.0 + qv)
    pad = lambda a: np.concatenate([a, a[-1:]])  # the kme row
    return dict(p8w=p8w, rho=pad(rho), dz8w=pad(dz), qv=pad(qv),
                pmid=pmid, t=t)


def _cloud(rng, sky, col):
    """Fill qc/qi/qs, radii (metres, as WRF's re_* carry them) and cldfra."""
    kme = NZ + 1
    z = {n: np.zeros(kme) for n in ("qc", "qi", "qs", "cldfra")}
    re_c = rng.uniform(2.5e-6, 50.0e-6, kme)
    re_i = rng.uniform(5.0e-6, 125.0e-6, kme)
    re_s = rng.uniform(10.0e-6, 999.0e-6, kme)
    lo = int(rng.integers(1, NZ // 2))
    hi = int(min(NZ - 1, lo + rng.integers(1, 8)))
    sl = slice(lo, hi)
    warm = slice(lo, min(hi, lo + 3))
    cold = slice(min(hi, lo + 2), hi + 4)
    frac = 1.0
    if sky in ("liquid", "liquid_ice", "mixed3", "partial", "small_re",
               "small_re_clear", "huge_re"):
        z["qc"][warm] = rng.uniform(1.0e-5, 1.5e-3)
    if sky in ("ice", "liquid_ice", "mixed3", "partial"):
        z["qi"][cold] = rng.uniform(1.0e-6, 3.0e-4)
    if sky in ("snow", "mixed3"):
        z["qs"][cold] = rng.uniform(1.0e-6, 8.0e-4)
    if sky == "thin_liquid":
        z["qc"][lo] = rng.uniform(1.0e-7, 2.0e-6)
    if sky == "thick":
        z["qc"][sl] = rng.uniform(2.0e-3, 6.0e-3)
        z["qi"][sl] = rng.uniform(5.0e-4, 2.0e-3)
        re_c[sl] = 2.6e-6
    if sky == "tau_min":
        # cloud optical depths of order TAU_MIN = 1e-4: liquid path ~1e-7
        z["qc"][lo] = 10.0 ** rng.uniform(-11.0, -8.5)
        z["qi"][lo + 1] = 10.0 ** rng.uniform(-11.0, -8.5)
    if sky == "top_only":
        z["qc"][NZ - 1] = 1.0e-4
        z["qi"][NZ - 1] = 1.0e-4
        z["qs"][NZ - 1] = 1.0e-4
    if sky == "partial":
        frac = rng.uniform(0.05, 0.95)
    if sky in ("small_re", "small_re_clear"):
        re_c[:] = rng.uniform(1.0e-7, 2.4e-6)
        re_i[:] = rng.uniform(1.0e-7, 4.9e-6)
        re_s[:] = rng.uniform(1.0e-7, 9.9e-6)
        z["qi"][warm] = rng.uniform(1.0e-6, 1.0e-4)
        z["qs"][warm] = rng.uniform(1.0e-6, 1.0e-4)
    if sky == "huge_re":
        re_c[:] = 300.0e-6
        re_i[:] = 900.0e-6
    cond = (z["qc"] + z["qi"] + z["qs"]) > 0.0
    if sky != "small_re_clear":
        z["cldfra"][cond] = frac
        # a few unsaturated cloud-free condensate levels, as cal_cldfra1's
        # 0.01 truncation leaves
        z["cldfra"][cond & (rng.uniform(size=kme) < 0.15)] = 0.0
    if col % 7 == 3:
        # cloud fraction without condensate (BL clouds, Thompson icloud=3)
        z["cldfra"][rng.integers(0, NZ)] = rng.uniform(0.0, 1.0)
    return dict(qc=z["qc"], qi=z["qi"], qs=z["qs"], cldfra=z["cldfra"],
                re_cloud=re_c, re_ice=re_i, re_snow=re_s)


def _surface(rng, aerosol_on):
    if aerosol_on:
        return dict(aerssa2d=rng.uniform(0.7, 1.0),
                    aerasy2d=rng.uniform(0.5, 0.85),
                    aod5502d=10.0 ** rng.uniform(-2.5, 0.4),
                    angexp2d=rng.uniform(0.0, 2.3))
    return dict(aerssa2d=0.0, aerasy2d=0.0, aod5502d=0.0, angexp2d=0.0)


def scenario_columns(seed):
    rng = np.random.default_rng(seed)
    cols = []
    col = 0
    for sky in SKIES:
        for aerosol_on in (False, True):
            for cz in COSZEN_SET:
                prof = _profile(rng)
                cols.append((prof, _cloud(rng, sky, col),
                             _surface(rng, aerosol_on),
                             rng.uniform(0.03, 0.85), cz))
                col += 1
    for _ in range(N_RANDOM):
        sky = SKIES[int(rng.integers(0, len(SKIES)))]
        prof = _profile(rng)
        cz = (rng.uniform(-0.2, 1.0) if rng.uniform() < 0.8
              else 10.0 ** rng.uniform(-7.0, -0.5))
        cols.append((prof, _cloud(rng, sky, col),
                     _surface(rng, rng.uniform() < 0.6),
                     rng.uniform(0.03, 0.9), cz))
        col += 1
    ncol = len(cols)
    out = {n: np.zeros((NZ + 1, ncol), F) for n in FIELDS3}
    out.update({n: np.zeros(ncol, F) for n in FIELDS2})
    for i, (prof, cloud, sfc, albedo, cz) in enumerate(cols):
        for n in ("p8w", "rho", "dz8w", "qv"):
            out[n][:, i] = prof[n]
        for n, v in cloud.items():
            out[n][:, i] = v
        for n, v in sfc.items():
            out[n][i] = v
        out["albedo"][i] = albedo
        out["coszen_loc"][i] = cz
    return out


def write_scenario(path, scalars, cols):
    ncol = cols["albedo"].size
    has_reqc, has_reqi, has_reqs, aer_opt, julian = scalars
    with open(path, "wb") as fh:
        np.array([ncol, NZ, has_reqc, has_reqi, has_reqs, aer_opt],
                 np.int32).tofile(fh)
        np.array([julian], F).tofile(fh)
        for n in FIELDS2:
            cols[n].astype(F).tofile(fh)
        for n in FIELDS3:
            # (k, i) C order is Fortran's (i, k): i fastest
            np.ascontiguousarray(cols[n], F).tofile(fh)


def main(build_dir):
    build = Path(build_dir)
    build.mkdir(parents=True, exist_ok=True)
    bundle = {}
    for name, rc, ri, rs, aer, julian, seed in SCENARIOS:
        cols = scenario_columns(seed)
        write_scenario(build / f"{name}.in.bin",
                       (rc, ri, rs, aer, julian), cols)
        bundle[f"{name}/scalars"] = np.array([rc, ri, rs, aer], np.int32)
        bundle[f"{name}/julian"] = np.array([julian], F)
        for n, v in cols.items():
            bundle[f"{name}/{n}"] = v
        print(f"{name}: {cols['albedo'].size} columns, nz={NZ}")
    bundle["scenarios"] = np.array([s[0] for s in SCENARIOS])
    np.savez_compressed(build / "columns.npz", **bundle)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: make_columns.py BUILD_DIR")
    main(sys.argv[1])
