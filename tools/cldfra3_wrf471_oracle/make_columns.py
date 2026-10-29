"""Column fixtures for the WRF v4.7.1 cal_cldfra3 oracle.

    python make_columns.py OUT_DIR [--seed N]

Writes OUT_DIR/columns.bin in run_cldfra3.F90's layout.  Deterministic
(numpy default_rng seeded); every value is a float32 word.

Coverage, by construction:
  * clear and cloudy columns: liquid, ice and snow decks at every height,
    condensate across cal_cldfra3's thresholds (1e-10, 1e-7, 1e-6, the
    log10 arm between them), tiny negative mixing ratios (WRF's moist
    arrays carry them), and RH from dry to supersaturated so the
    RH-based arm runs warm (tc >= -12), cold (tc < -12) and hot (>= 25 C);
  * land (XLAND = 1) and ocean (XLAND = 2);
  * tropopause cases: tropical (17 km), mid-latitude (11 km), arctic
    (6-8 km, where WRF's 700 hPa guard binds) and no clear tropopause;
    model tops from 10 to 50 hPa so the 70 hPa cut, the 101 hPa -12 C
    guard and the 150 hPa search start all move;
  * thin layers (20 m near the surface) and thick ones, a well-mixed
    boundary layer and a stable one (find_cloudLayers' kbot search);
  * stress columns with very deep layers whose made-up condensate path
    exceeds 1 kg m-2, so adjust_cloudFinal rescales;
  * grid spacings 0.5 to 27 km, square and not.
Columns WRF would stop on (the -12 C level within three levels of the top,
wrf_error_fatal) are screened out here; the port's status flag for them is
tested separately.
"""
import argparse
import pathlib

import numpy as np

F = np.float32
G, RD = 9.81, 287.04


def esat_liq(t):
    x = np.maximum(-80.0, t - 273.16)
    return 611.2 * np.exp(17.67 * x / (x + 243.5))


def column(rng, kind):
    nz = int(rng.choice([35, 45, 50, 61]))
    if kind == "stress":
        nz = 16
    ptop = float(rng.choice([1000.0, 2000.0, 5000.0]))
    # Stretched dz: thin near the surface, growing upward.
    dz0 = float(rng.choice([20.0, 40.0, 80.0]))
    if kind == "stress":
        dz0 = float(rng.choice([3000.0, 3500.0, 4000.0]))
    psfc = float(rng.uniform(85000.0, 103000.0))
    t0 = {"tropical": rng.uniform(296, 306), "midlat": rng.uniform(270, 300),
          "arctic": rng.uniform(235, 262), "notrop": rng.uniform(260, 290),
          "stress": rng.uniform(303, 310)}[kind]
    ztrop = {"tropical": rng.uniform(15e3, 17.5e3), "midlat": rng.uniform(9e3, 13e3),
             "arctic": rng.uniform(5.5e3, 8.5e3), "notrop": 60e3,
             "stress": 60e3}[kind]
    gamma = rng.uniform(5.5e-3, 7.5e-3) if kind != "stress" else rng.uniform(3.3e-3, 3.8e-3)
    strat = rng.uniform(0.0, 2.5e-3) if kind != "stress" else 0.0
    # Integrate the hydrostatic column upward until ptop; then rescale dz so
    # nz levels exactly span it.
    for _ in range(40):
        z = np.zeros(nz)
        dz = dz0 * (1.0 + np.arange(nz)) ** 1.0
        if kind == "stress":
            dz = np.full(nz, dz0)
        dz = np.minimum(dz, 1200.0 if kind != "stress" else 5000.0)
        zc = np.cumsum(dz) - 0.5 * dz
        t = np.where(zc < ztrop, t0 - gamma * zc, t0 - gamma * ztrop + strat * (zc - ztrop))
        t = np.maximum(t, 185.0)
        if kind != "stress" and rng.random() < 0.5:
            # well-mixed boundary layer: dry adiabat to ~1.2 km
            h = rng.uniform(600, 2000)
            t = np.where(zc < h, t0 - 9.8e-3 * zc, t)
        elif kind != "stress":
            # surface inversion
            t = t + np.where(zc < 300.0, rng.uniform(0, 6) * (zc / 300.0 - 1.0) * -1.0 * 0.5, 0.0)
        p = np.empty(nz)
        pw = psfc
        for k in range(nz):
            p[k] = pw * np.exp(-G * 0.5 * dz[k] / (RD * t[k]))
            pw = pw * np.exp(-G * dz[k] / (RD * t[k]))
        if kind == "stress":
            break
        if p[-1] > ptop * 1.05:
            dz0 *= 1.08
        elif p[-1] < ptop * 0.95:
            dz0 /= 1.06
        else:
            break
    t = t + rng.normal(0.0, 0.3, nz)
    # Humidity: layered RH.
    rh = np.clip(rng.uniform(0.2, 0.7) + 0.25 * np.sin(zc / rng.uniform(1500, 5000)
                                                       + rng.uniform(0, 6)), 0.02, 1.0)
    for _ in range(rng.integers(0, 4)):
        k0 = rng.integers(0, nz - 3)
        rh[k0:k0 + rng.integers(1, 6)] = rng.uniform(0.85, 1.25)
    es = np.minimum(esat_liq(t), 0.15 * p)
    qv = np.maximum(rh * 0.622 * es / (p - es), 1e-7)
    qv = np.where(p < 10000.0, 3e-6, qv)
    qc = np.zeros(nz)
    qi = np.zeros(nz)
    qs = np.zeros(nz)
    menu = [0.0, 1e-11, 5e-11, 1e-10, 3e-10, 1e-9, 1e-8, 5e-8, 9.9e-8, 1e-7,
            2e-7, 6e-7, 9.99e-7, 1e-6, 1.01e-6, 3e-6, 1e-5, 2e-4, 1e-3]
    for _ in range(rng.integers(0, 5)):
        k0 = int(rng.integers(0, nz - 2))
        depth = int(rng.integers(1, 8))
        sl = slice(k0, min(nz, k0 + depth))
        warm = t[sl] > 253.0
        qc[sl] = np.where(warm, rng.choice(menu, size=warm.size), qc[sl])
        qi[sl] = np.where(~warm | (rng.random(warm.size) < 0.3),
                          rng.choice(menu, size=warm.size), qi[sl])
        if rng.random() < 0.4:
            qs[sl] = rng.choice(menu, size=warm.size)
    # tiny negatives, as advection leaves them
    neg = rng.random(nz) < 0.04
    qc = np.where(neg, -rng.choice([1e-12, 1e-9, 3e-8]), qc)
    neg = rng.random(nz) < 0.04
    qi = np.where(neg, -rng.choice([1e-12, 1e-9]), qi)
    if kind == "stress":
        # Isolated near-saturated levels in very deep layers: single-level
        # fractional clouds take 0.05*qvs*cfr each (find_cloudLayers), and
        # over kilometres of depth their path passes 1 kg m-2, which is
        # what adjust_cloudFinal rescales.
        qc[:] = 0.0
        qi[:] = 0.0
        sat = 0.622 * es / (p - es)
        wet = (np.arange(nz) % 2 == 0)
        qv = np.where(p < 10000.0, 3e-6, np.where(wet, rng.uniform(0.95, 0.995) * sat, 0.3 * sat))
    xland = F(rng.choice([1.0, 2.0]))
    dx = F(rng.choice([500.0, 1000.0, 3000.0, 4000.0, 9000.0, 12000.0, 27000.0]))
    dy = dx if rng.random() < 0.8 else F(dx * rng.choice([0.5, 1.5]))
    return dict(nz=nz, dx=dx, dy=dy, xland=xland,
                arrays=[a.astype(F) for a in (qv, qc, qi, qs, dz, p, t)])


def stress_column(rng):
    """Isolated near-saturated levels in 3 km layers of a warm, stable
    column: the one at level 3 is a single-level fractional cloud that
    find_cloudLayers fills with 0.05*qvs*cfr, and in 3 km of warm air its
    path passes 1 kg m-2 -- what adjust_cloudFinal rescales.
    """
    nz = 16
    dz = np.full(nz, float(rng.choice([2800.0, 3000.0, 3200.0])))
    zc = np.cumsum(dz) - 0.5 * dz
    t = np.maximum(rng.uniform(303, 310) - rng.uniform(1.5e-3, 2.2e-3) * zc, 185.0)
    p = np.empty(nz)
    pw = rng.uniform(98000.0, 102000.0)
    for k in range(nz):
        p[k] = pw * np.exp(-G * 0.5 * dz[k] / (RD * t[k]))
        pw = pw * np.exp(-G * dz[k] / (RD * t[k]))
    es = np.minimum(esat_liq(t), 0.15 * p)
    sat = 0.622 * es / (p - es)
    wet = np.arange(nz) % 2 == 0
    qv = np.where(wet, rng.uniform(0.95, 0.995, nz) * sat, 0.3 * sat)
    qv = np.where(p < 10000.0, 3e-6, qv)
    zero = np.zeros(nz)
    xland = F(rng.choice([1.0, 2.0]))
    dx = F(rng.choice([1000.0, 3000.0, 9000.0]))
    return dict(nz=nz, dx=dx, dy=dx, xland=xland,
                arrays=[a.astype(F) for a in (qv, zero, zero, zero, dz, p, t)])


def fatal(col):
    """WRF's wrf_error_fatal screen: k_m12C > kte-3 (find_cloudLayers)."""
    t, p = col["arrays"][6], col["arrays"][5]
    nz = col["nz"]
    hits = [k + 1 for k in range(nz) if (t[k] - F(273.16)) > F(-12.0) and p[k] > F(10100.0)]
    return bool(hits) and max(hits) > nz - 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=471003)
    ap.add_argument("--per-kind", type=int, default=120)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cols = []
    for kind in ("tropical", "midlat", "arctic", "notrop", "stress"):
        n = a.per_kind if kind != "stress" else max(8, a.per_kind // 6)
        while n:
            c = stress_column(rng) if kind == "stress" else column(rng, kind)
            if not fatal(c):
                cols.append(c)
                n -= 1
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "columns.bin", "wb") as f:
        f.write(np.int32(len(cols)).tobytes())
        for c in cols:
            f.write(np.int32(c["nz"]).tobytes())
            f.write(np.array([c["dx"], c["dy"], c["xland"]], F).tobytes())
            for arr in c["arrays"]:
                f.write(np.ascontiguousarray(arr, F).tobytes())
    print(len(cols), "columns")


if __name__ == "__main__":
    main()
