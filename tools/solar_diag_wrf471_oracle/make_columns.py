"""Column fixtures for the WRF v4.7.1 solar_diag oracle.

    python make_columns.py OUT_DIR [--seed N]

Writes OUT_DIR/columns.bin in run_solar_diag.F90's layout and
OUT_DIR/columns_meta.json (which columns are WRF-shaped and which are
MPAS-shaped) and OUT_DIR/mpas_zgrid.bin (the MPAS columns' zgrid).  Deterministic; every value is a float32 word.

WRF-shaped columns: a hydrostatic sounding on a stretched height ladder;
geopotential split as WRF splits it (PHB the base state, PH a small
perturbation), physics density (1+qv)/ALT, dz8w the w-level differences.

MPAS-shaped columns: the inputs woof/hex/mpasphys/solar_diagnostics_v842.py
hands the same routine from MPAS fields -- PH = 0, PHB = 9.81*zgrid (the
w-level heights above sea level), rho = rho_p*(1+qv) with rho_p MPAS's dry
physics density, dz8w = zgrid(k+1)-zgrid(k) -- all formed in float32 exactly
as that adapter forms them.

Coverage: clear and cloudy columns (liquid, ice, snow; single and multiple
decks; condensate across WC_MIN = 1e-5), a deck only at the top level
(WRF's MISSING base) and only at the lowest level (MISSING top), negative
column paths (SIGN(MAX(.,0),1) and the zero radius), the moist species
absent in every combination WRF's packages allow, schemes without radii
(has_req* = 0, outputs left as they were), day and night (sza >= 90, a zero
TOA flux, coszen exactly -1, 0 and 1), resolved vs radiation (_tot)
condensate that differ, and grid steps from 2 s to 90 s.
"""
import argparse
import json
import pathlib

import numpy as np

F = np.float32
G = F(9.81)


def sounding(rng, nz):
    ztop = rng.uniform(15000.0, 30000.0)
    eta = np.linspace(0.0, 1.0, nz + 1) ** rng.uniform(1.2, 1.8)
    zw = (eta * ztop + rng.uniform(0.0, 1500.0)).astype(np.float64)
    zc = 0.5 * (zw[1:] + zw[:-1])
    t = np.maximum(rng.uniform(270, 305) - 6.5e-3 * (zc - zw[0]), 200.0)
    p = 101325.0 * np.exp(-zc / 8000.0)
    qv = np.maximum(rng.uniform(0.2, 0.9) * 0.622 * 611.2 * np.exp(
        17.67 * (t - 273.15) / (t - 29.65)) / p, 1e-6)
    rho_dry = p / (287.0 * t * (1 + 0.608 * qv))
    return zw, zc, t, p, qv, rho_dry


def condensate(rng, nz, t, kind):
    qc, qi, qs = (np.zeros(nz) for _ in range(3))
    if kind == "clear":
        return qc, qi, qs
    menu = [2e-6, 9e-6, 1e-5, 1.1e-5, 5e-5, 3e-4, 1.5e-3]
    for _ in range(rng.integers(1, 4)):
        k0 = int(rng.integers(0, nz - 1))
        sl = slice(k0, min(nz - 1, k0 + int(rng.integers(1, 6))))
        warm = t[sl] > 258.0
        qc[sl] = np.where(warm, rng.choice(menu, warm.size), 0.0)
        qi[sl] = np.where(~warm, rng.choice(menu, warm.size), qi[sl])
        if rng.random() < 0.5:
            qs[sl] = rng.choice(menu, warm.size) * 0.5
    if kind == "top_only":
        qc[:] = qi[:] = qs[:] = 0.0
        qi[nz - 1] = 4e-5
    elif kind == "bottom_only":
        qc[:] = qi[:] = qs[:] = 0.0
        qc[0] = 4e-4
    elif kind == "negative":
        qc[:] = -np.abs(rng.uniform(1e-9, 1e-7, nz))
        qi[:] = 0.0
        qi[nz // 2] = -1e-8
    elif kind == "trace":
        qc *= 1e-3
        qi *= 1e-3
        qs *= 1e-3
    return qc, qi, qs


FLAG_SETS = [  # f_qv f_qc f_qi f_qs has_reqc has_reqi has_reqs
    (1, 1, 1, 1, 1, 1, 1),      # Thompson / WSM6 class
    (1, 1, 1, 1, 0, 0, 0),      # Morrison: no scheme radii
    (1, 1, 1, 0, 1, 1, 0),      # P3: qi, no qs
    (1, 1, 0, 0, 0, 0, 0),      # Kessler / WSM3
    (1, 0, 0, 0, 0, 0, 0),      # passive qv
    (1, 1, 1, 1, 1, 0, 1),
]


def column(rng, mpas, kind, flags, coszen_override=None):
    nz = int(rng.choice([30, 41, 50, 60]))
    zw, zc, t, p, qv, rho_dry = sounding(rng, nz)
    qc, qi, qs = condensate(rng, nz, t, kind)
    qc_tot = qc.copy()
    qi_tot = qi.copy()
    # the radiation driver's own cloud: made-up condensate in partial layers
    extra = rng.random(nz) < 0.2
    qc_tot = np.where(extra & (t > 258), qc_tot + rng.uniform(1e-6, 8e-5, nz), qc_tot)
    qi_tot = np.where(extra & (t <= 258), qi_tot + rng.uniform(1e-7, 2e-5, nz), qi_tot)
    re_c = np.clip(rng.normal(8e-6, 3e-6, nz), 2.49e-6, 50e-6)
    re_i = np.clip(rng.normal(30e-6, 15e-6, nz), 4.99e-6, 125e-6)
    re_s = np.clip(rng.normal(300e-6, 100e-6, nz), 9.99e-6, 999e-6)
    cldfra = np.where((qc + qi) > 1e-6, 1.0,
                      np.clip(rng.uniform(-0.3, 0.9, nz), 0.0, 0.9))
    f32 = lambda a: np.asarray(a, np.float64).astype(F)
    qv, qc, qi, qs, qc_tot, qi_tot = map(f32, (qv, qc, qi, qs, qc_tot, qi_tot))
    zwf = f32(zw)
    if mpas:
        # woof/hex/mpasphys/solar_diagnostics_v842.py's mapping, float32
        rho_p = f32(rho_dry)
        rho = (rho_p * (F(1.0) + qv)).astype(F)
        dz = (zwf[1:] - zwf[:-1]).astype(F)
        ph = np.zeros(nz + 1, F)
        phb = (G * zwf).astype(F)
    else:
        alt = f32(1.0 / rho_dry)
        rho = ((F(1.0) / alt) * (F(1.0) + qv)).astype(F)
        phb = f32(9.81 * zw * (1.0 - 1e-4 * rng.random()))
        ph = f32(9.81 * zw * 1e-4 * rng.normal(0.0, 1.0, nz + 1))
        zgeo = ((ph + phb) / G).astype(F)
        dz = (zgeo[1:] - zgeo[:-1]).astype(F)
    if coszen_override is not None:
        coszen = F(coszen_override)
    else:
        coszen = F(rng.uniform(-0.6, 1.0))
    swdnt = F(1361.0 * max(float(coszen), 0.0)) if rng.random() > 0.05 else F(0.0)
    swdnb = F(float(swdnt) * rng.uniform(0.05, 0.85))
    dt = F(rng.choice([2.0, 6.0, 12.0, 18.0, 36.0, 60.0, 90.0]))
    prior = F(rng.choice([0.0, 1.25e7, 3.0e5]))
    arrays = [rho, dz, qv, qc, qi, qs, qc_tot, qi_tot, f32(re_c), f32(re_i),
              f32(re_s), f32(cldfra)]
    return dict(nz=nz, flags=flags, scalars=[dt, coszen, swdnb, swdnt, prior],
                arrays=arrays, ph=ph, phb=phb, zgrid=zwf if mpas else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=471017)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cols, meta = [], []
    kinds = ["clear", "cloudy", "cloudy", "cloudy", "top_only", "bottom_only",
             "negative", "trace"]
    for mpas in (False, True):
        for flags in FLAG_SETS:
            for kind in kinds:
                for rep in range(6 if not mpas else 3):
                    override = None
                    if rep == 0 and kind == "cloudy":
                        override = [-1.0, 0.0, 1.0][len(cols) % 3]
                    cols.append(column(rng, mpas, kind, flags, override))
                    meta.append({"mpas": mpas, "kind": kind, "flags": flags})
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "columns.bin", "wb") as f:
        f.write(np.int32(len(cols)).tobytes())
        for c in cols:
            f.write(np.int32(c["nz"]).tobytes())
            f.write(np.array(c["flags"], np.int32).tobytes())
            f.write(np.array(c["scalars"], F).tobytes())
            for arr in c["arrays"]:
                f.write(np.ascontiguousarray(arr, F).tobytes())
            f.write(c["ph"].tobytes())
            f.write(c["phb"].tobytes())
    (out / "columns_meta.json").write_text(json.dumps(meta))
    # The MPAS columns' zgrid (w-level heights, float32), in column order:
    # the adapter's own input, from which it forms PHB = 9.81*zgrid.
    with open(out / "mpas_zgrid.bin", "wb") as f:
        for c in cols:
            if c["zgrid"] is not None:
                f.write(np.ascontiguousarray(c["zgrid"], F).tobytes())
    print(len(cols), "columns")


if __name__ == "__main__":
    main()
