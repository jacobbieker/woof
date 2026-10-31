"""Fixtures for the WRF v4.7.1 MAD-WRF oracle (run_madwrf.F90).

    python make_columns.py OUT_DIR [--seed N]

Writes OUT_DIR/madwrf_columns.bin.  Deterministic.

Columns: the icloud = 3 oracle's profiles (tools/cldfra3_wrf471_oracle/
make_columns.py: tropical, mid-latitude, arctic, no-tropopause and deep
stress columns, liquid/ice/snow decks across every threshold, tiny
negatives, land and sea, 0.5-27 km grids), recast as real.exe holds them
(perturbation theta, hydrostatic pressure, geopotential at w levels over
terrain 0-2.5 km), each with one of real.exe's five cloud-initialization
inputs: none (RH only), CLDMASK alone, CLDMASK + BRTEMP, CLDTOPZ, and
CLDTOPZ + CLDBASEZ -- clear, cloudy and missing (-999.9) values, observed
tops below, inside and above the analysis cloud, bases below and above
the top.  Every observed height stays below the model top (see
woof/core/kernels/madwrf.cu on find_k_from_z_agl's last iteration).
Columns WRF stops on (find_cloudLayers' -12 C / -40 C screen) are left out.

Nudge cases: random blocks with a spec-zone tile, forecast times before,
at and after madwrf_dt_nudge, relaxation times shorter and longer than dt.
"""
import argparse
import importlib.util
import pathlib

import numpy as np

F = np.float32
HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "cldfra3_columns", HERE.parent / "cldfra3_wrf471_oracle" / "make_columns.py")
c3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c3)
MISSING = F(-999.9)


def screened(t, p, nz):
    """find_cloudLayers' stop (kte = nz + 1), with a level of margin."""
    m12 = [k + 1 for k in range(nz) if t[k] - 273.16 > -12.0 and p[k] > 10100.0]
    m40 = [k + 1 for k in range(nz) if t[k] - 273.16 > -40.0 and p[k] > 7000.0]
    return (m12 and max(m12) > nz - 3) or (m40 and max(m40) > nz - 2)


def recast(rng, col, mode):
    qv, qc, qi, qs, dz, p, t = (np.asarray(a, np.float64) for a in col["arrays"])
    nz = col["nz"]
    ht = float(rng.choice([0.0, 0.0, 150.0, 900.0, 1600.0, 2500.0]))
    theta = t * (1.0e5 / p) ** (2.0 / 7.0)
    t2 = theta - 300.0
    zw = np.concatenate([[0.0], np.cumsum(dz)])
    split = rng.uniform(0.0, 1.0, nz + 1)
    phb = 9.81 * (ht + zw) * split
    ph = 9.81 * (ht + zw) - phb
    top = zw[-1]
    flags = dict(cldmask=0, cldtopz=0, cldbasez=0, brtemp=0)
    cldmask = cldtopz = cldbasez = brtemp = float(MISSING)
    r = rng.random()
    if mode == 1:
        flags["cldmask"] = 1
        cldmask = 0.0 if r < 0.4 else (1.0 if r < 0.85 else float(MISSING))
    elif mode == 2:
        flags["cldmask"] = flags["brtemp"] = 1
        cldmask = 0.0 if r < 0.25 else (1.0 if r < 0.9 else float(MISSING))
        brtemp = (float(MISSING) if rng.random() < 0.1
                  else float(rng.uniform(t.min() - 5.0, t.max() + 5.0)))
    elif mode in (3, 4):
        flags["cldtopz"] = 1
        if r < 0.2:
            cldtopz = 0.0
        elif r < 0.3:
            cldtopz = float(MISSING)
        else:
            cldtopz = ht + float(rng.uniform(150.0, 0.85 * top))
        if mode == 4:
            flags["cldbasez"] = 1
            s = rng.random()
            if cldtopz > 0 and s < 0.6:
                cldbasez = float(rng.uniform(50.0, cldtopz - ht))
            elif cldtopz > 0 and s < 0.8:
                cldbasez = float(rng.uniform(cldtopz - ht, 0.9 * top))
            elif s < 0.9:
                cldbasez = 0.0
    return dict(nz=nz, dx=col["dx"], dy=col["dy"], xland=col["xland"], ht=ht,
                flags=flags, cldmask=cldmask, cldtopz=cldtopz, cldbasez=cldbasez,
                brtemp=brtemp, arrays=[a.astype(F) for a in (t2, p, ph, phb, qv, qc, qi, qs)],
                t=t, p=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=471096)
    ap.add_argument("--per-kind", type=int, default=60)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cols = []
    mode = 0
    for kind in ("tropical", "midlat", "arctic", "notrop", "stress"):
        n = a.per_kind if kind != "stress" else 10
        while n:
            base = c3.stress_column(rng) if kind == "stress" else c3.column(rng, kind)
            col = recast(rng, base, mode % 5)
            if screened(col["t"], col["p"], col["nz"]):
                continue
            cols.append(col)
            mode += 1
            n -= 1
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "madwrf_columns.bin", "wb") as f:
        np.array([len(cols)], "<i4").tofile(f)
        for c in cols:
            fl = c["flags"]
            np.array([c["nz"], fl["cldmask"], fl["cldtopz"], fl["cldbasez"], fl["brtemp"]],
                     "<i4").tofile(f)
            np.array([c["dx"], c["dy"], c["xland"], c["ht"], c["cldmask"], c["cldtopz"],
                      c["cldbasez"], c["brtemp"]], "<f4").tofile(f)
            for arr in c["arrays"]:
                arr.astype("<f4").tofile(f)
        cases = []
        for dt, relax, nudge, xtime in ((18.0, 60.0, 60.0, 0.0), (18.0, 60.0, 60.0, 59.7),
                                        (18.0, 60.0, 60.0, 60.0), (60.0, 900.0, 120.0, 30.0),
                                        (7.5, 7.5, 1.0e9, 5000.0), (30.0, 10.0, 60.0, 0.5)):
            ni, nk, nj = 24, 10, 18
            its, ite, jts, jte = 3, ni - 2, 2, nj - 3
            q = rng.lognormal(np.log(1e-4), 2.0, (6, nj, nk + 1, ni))
            q[rng.random(q.shape) < 0.3] = 0.0
            q[rng.random(q.shape) < 0.02] *= -1e-6
            cases.append((ni, nk, nj, its, ite, jts, jte, dt, relax, nudge, xtime, q.astype(F)))
        np.array([len(cases)], "<i4").tofile(f)
        for ni, nk, nj, its, ite, jts, jte, dt, relax, nudge, xtime, q in cases:
            np.array([ni, nk, nj, its, ite, jts, jte], "<i4").tofile(f)
            np.array([dt, relax, nudge, xtime], "<f4").tofile(f)
            q.astype("<f4").tofile(f)     # (6, nj, nk+1, ni) == Fortran (ni, nk+1, nj, 6)
    print(len(cols), "columns")


if __name__ == "__main__":
    main()
