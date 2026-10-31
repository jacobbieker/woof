"""Tiles for the WRF v4.7.1 km_opt = 5 (SMS-3DTKE) oracle.

    python make_cases.py OUT_DIR

Writes OUT_DIR/cases.bin in run_sms3dtke.F90's layout and
OUT_DIR/cases.json (the per-case summary).  Deterministic (numpy
default_rng seeded); every array is float32 words, rounded once.

Each case is a 3-D tile of nx x ny columns, nz mass levels, one boundary
layer regime over flat ground or a hill, at one grid spacing.  The grid
spacings run from 50 m (LES: pthl = pthnl = pu = 0, the d <= 100 m cut) to
9 km (mesoscale: pu -> 1), so the scale-adaptive partition spans its whole
range.  Regimes:

  convective  superadiabatic surface layer, well-mixed theta (exactly
              equal float words over several levels, BouLac's bbb = 0 and
              theta(izz) = theta(k) arms), a capping inversion, HFX > 0,
              br < 0, rmol < 0: nonlocal_flux's convective column
              (pblflg) with entrainment, GAMU/GAMV written
  neutral     near-zero heat flux, sheared wind, deep neutral layer
  stable      surface inversion and low-level jet, HFX < 0, br > 0,
              rmol > 0: sfcflg false, GAMU/GAMV keep their previous words
  edge        calm levels (du = dv = 0 at the PBL top: rigs = rigsmax),
              zero, tiny and subnormal TKE (never negative: bound_tke
              clamps it after every RK pass), a column whose theta never
              exceeds thv(1) + 0.5 (hpbl = 0: the partial functions' h = 0
              arm), and very large TKE

and isfflx 1 (surface-layer HFX), 0 and 2 (namelist tke_heat_flux, which
nonlocal_flux and meso_length_scale write into HFX), periodic and
specified lateral boundaries, flat and hill terrain (the slope limiter
alpha > 1 on the hill), pure and hybrid vertical coordinates.

cases.bin layout (little endian):
  int ncase; per case
    int nx, ny, nz, bnd (0 periodic, 1 specified), isfflx, mix_full_fields,
        use_theta_m
    real dx, dy, dt, c_s, c_k, tke_heat_flux, tke_drag_coefficient,
         cf1, cf2, cf3, p_top
    real fnm, fnp, dn, dnw, c1h, c2h, c1f, c2f, znw, t_base, qv_base,
         u_base, v_base           (nz+1 each, Fortran k = 1..kde)
    real msf, mut, ust, ustm, hfx, qfx, br, rmol, wspd, u10, v10, gamu,
         gamv, pblh               (ny, nx)
    real u (nz, ny, nx+1), v (nz, ny+1, nx), w, ph, phb (nz+1, ny, nx),
         t, p, pb, alt, qv, qc, qi, tke (nz, ny, nx)
  3-D arrays go out with i fastest, then k, then j (Fortran (i, k, j)).
"""
import argparse
import json
import pathlib

import numpy as np

F = np.float32
G, RD, CP, RV = 9.81, 287.0, 1004.5, 461.6
P0 = 1.0e5

CASES = (
    # name, regime, dx, terrain, bnd, isfflx, hybrid, nx, ny, nz
    ("les50-cbl-flat-per", "convective", 50.0, "flat", 0, 1, False, 10, 9, 40),
    ("les100-cbl-hill-per", "convective", 100.0, "hill", 0, 1, False, 9, 10, 40),
    ("les50-sbl-flat-per", "stable", 50.0, "flat", 0, 1, False, 8, 8, 44),
    ("gz250-cbl-flat-per", "convective", 250.0, "flat", 0, 1, True, 10, 10, 40),
    ("gz250-neu-hill-spec", "neutral", 250.0, "hill", 1, 1, False, 11, 9, 38),
    ("gz500-cbl-hill-per", "convective", 500.0, "hill", 0, 1, False, 9, 9, 42),
    ("gz500-sbl-hill-spec", "stable", 500.0, "hill", 1, 1, True, 10, 11, 40),
    ("gz1k-cbl-flat-spec", "convective", 1000.0, "flat", 1, 1, False, 10, 10, 40),
    ("gz1k-neu-flat-per", "neutral", 1000.0, "flat", 0, 1, False, 9, 9, 36),
    ("gz1k-sbl-hill-per", "stable", 1000.0, "hill", 0, 1, False, 9, 10, 42),
    ("meso3k-cbl-hill-per", "convective", 3000.0, "hill", 0, 1, True, 10, 9, 40),
    ("meso3k-sbl-flat-spec", "stable", 3000.0, "flat", 1, 1, False, 9, 9, 40),
    ("meso9k-cbl-flat-per", "convective", 9000.0, "flat", 0, 1, False, 9, 9, 45),
    ("meso9k-neu-hill-spec", "neutral", 9000.0, "hill", 1, 1, False, 10, 10, 40),
    ("gz500-cbl-isfflx0-per", "convective", 500.0, "flat", 0, 0, False, 8, 9, 40),
    ("gz1k-cbl-isfflx2-hill", "convective", 1000.0, "hill", 0, 2, False, 9, 8, 40),
    ("les50-edge-flat-per", "edge", 50.0, "flat", 0, 1, False, 9, 9, 36),
    ("gz1k-edge-hill-spec", "edge", 1000.0, "hill", 1, 1, False, 10, 10, 36),
    ("meso3k-edge-flat-per", "edge", 3000.0, "flat", 0, 0, False, 9, 9, 36),
)


def eta_levels(nz, rng):
    """WRF-like eta: thin near the surface, stretched aloft (kde = nz+1)."""
    s = np.linspace(0.0, 1.0, nz + 1)
    znw = 1.0 - (0.75 * s + 0.25 * s ** 2.2)
    znw[0], znw[-1] = 1.0, 0.0
    return znw


def vertical_coefficients(znw, hybrid):
    nz = znw.size - 1
    znu = 0.5 * (znw[1:] + znw[:-1])
    dnw = np.zeros(nz + 1)
    dnw[:nz] = znw[1:] - znw[:-1]
    dn = np.zeros(nz + 1)
    dn[1:nz] = znu[1:] - znu[:-1]
    fnm = np.zeros(nz + 1)
    fnp = np.zeros(nz + 1)
    fnm[1:nz] = 0.5 * dnw[1:nz] / dn[1:nz]
    fnp[1:nz] = 0.5 * dnw[0:nz - 1] / dn[1:nz]
    cof1 = (2.0 * dn[1] + dn[2]) / (dn[1] + dn[2]) * dnw[0] / dn[1]
    cof2 = dn[1] / (dn[1] + dn[2]) * dnw[0] / dn[2]
    cf1 = fnp[1] + cof1
    cf2 = fnm[1] - cof1 - cof2
    cf3 = cof2
    if hybrid:
        etac = 0.2
        bf = np.where(znw > etac,
                      ((znw - etac) / (1.0 - etac)) ** 2 * (1 + 2 * (1 - znw)), 0.0)
        bf = np.clip(bf, 0.0, 1.0)
        c1f = bf
        c2f = (1.0 - bf) * (znw - 0.0)
        bh = 0.5 * (bf[1:] + bf[:-1])
        c1h = np.zeros(nz + 1)
        c2h = np.zeros(nz + 1)
        c1h[:nz] = bh
        c2h[:nz] = (1.0 - bh) * znu
    else:
        c1f = np.ones(nz + 1)
        c2f = np.zeros(nz + 1)
        c1h = np.zeros(nz + 1)
        c2h = np.zeros(nz + 1)
        c1h[:nz] = 1.0
    return dict(znw=znw, dnw=dnw, dn=dn, fnm=fnm, fnp=fnp, cf1=cf1, cf2=cf2,
                cf3=cf3, c1f=c1f, c2f=c2f, c1h=c1h, c2h=c2h)


def theta_profile(z, regime, zi, rng, th0):
    """Potential temperature (K) at heights z above ground."""
    if regime == "convective":
        th = np.where(z < 0.08 * zi, th0 + 1.5 - 1.5 * (z / (0.08 * zi)) ** 0.5, th0)
        th = np.where(z > zi, th0 + 3.0 + 0.0045 * (z - zi), th)
        th = np.where((z > zi - 150) & (z <= zi), th0 + 3.0 * (z - zi + 150) / 150, th)
    elif regime == "stable":
        th = th0 - 6.0 + 6.0 * (1 - np.exp(-z / 180.0)) + 0.004 * z
    elif regime == "neutral":
        th = np.where(z < zi, th0 + 0.0002 * z, th0 + 0.0002 * zi + 0.004 * (z - zi))
    else:  # edge: a cold top keeps theta below thv(1) + 0.5 in some columns
        th = th0 + 0.001 * np.minimum(z, 300.0) + 0.0035 * np.maximum(z - zi, 0.0)
    return th


def make_case(spec, rng):
    name, regime, dx, terrain, bnd, isfflx, hybrid, nx, ny, nz = spec
    dy = dx
    znw = eta_levels(nz, rng)
    vc = vertical_coefficients(znw, hybrid)
    ztop = 14000.0 + rng.uniform(-500, 500)
    p_top = 5000.0
    # Terrain.
    xs = (np.arange(nx) - nx / 2 + 0.5) * dx
    ys = (np.arange(ny) - ny / 2 + 0.5) * dy
    X, Y = np.meshgrid(xs, ys)
    if terrain == "hill":
        width = max(3.0 * dx, 400.0)
        amp = min(600.0, 0.6 * width)
        hgt = amp * np.exp(-((X / width) ** 2 + (Y / (1.3 * width)) ** 2))
    else:
        hgt = np.zeros((ny, nx))
    # Heights of w levels: terrain-following, ztop at the lid.
    zeta = (1.0 - znw) ** 1.35 * ztop
    zw = hgt[None] + zeta[:, None, None] * (ztop - hgt[None]) / ztop
    zw = zw + rng.normal(0.0, 0.05, zw.shape) * (np.arange(nz + 1) > 0)[:, None, None]
    zbase = hgt[None] + zeta[:, None, None] * (ztop - hgt[None]) / ztop
    phb = G * zbase
    ph = G * (zw - zbase)
    ph[0] = 0.0
    zm = 0.5 * (zw[1:] + zw[:-1]) - hgt[None]
    # Thermodynamics.
    zi = {"convective": rng.uniform(700, 1600), "stable": 200.0,
          "neutral": rng.uniform(500, 900), "edge": rng.uniform(400, 800)}[regime]
    th0 = rng.uniform(288, 302)
    theta = theta_profile(zm, regime, zi, rng, th0)
    if regime in ("convective", "neutral"):
        # Small resolved turbulence; mixed layer words stay exactly equal
        # in some columns (BouLac's bbb = 0 arm).
        noise = rng.normal(0.0, 0.08 if dx < 600 else 0.02, theta.shape)
        keep = rng.random((ny, nx)) < 0.3
        mixed = (zm > 0.1 * zi) & (zm < 0.8 * zi)
        noise = np.where(keep[None] & mixed, 0.0, noise)
        theta = theta + noise * (zm < 1.2 * zi)
    if regime == "edge":
        cold = rng.random((ny, nx)) < 0.35
        theta = np.where(cold[None], th0 + 0.0002 * zm - 0.00002 * zm, theta)
    qv = np.clip(0.014 * np.exp(-zm / 2500.0), 1e-6, None)
    if regime == "stable":
        qv = 0.6 * qv
    qv = qv * (1.0 + rng.normal(0, 0.03, qv.shape))
    qc = np.where((zm > 0.85 * zi) & (zm < zi) & (rng.random(zm.shape) < 0.5),
                  rng.uniform(1e-6, 4e-4, zm.shape), 0.0)
    if regime == "edge":
        qc = np.where(rng.random(zm.shape) < 0.05, 1.0e-5, qc)
    qi = np.where(zm > 7000.0, rng.uniform(0, 2e-5, zm.shape), 0.0)
    psfc = 101000.0 - 11.5 * hgt + rng.normal(0, 30, hgt.shape)
    # Hydrostatic pressure through the column from theta.
    pfull = np.empty((nz, ny, nx))
    pi_sfc = (psfc / P0) ** (RD / CP)
    pi_k = pi_sfc.copy()
    zprev = np.zeros((ny, nx))
    for k in range(nz):
        dz = zm[k] - zprev
        pi_k = pi_k - G * dz / (CP * theta[k] * (1 + 0.61 * qv[k]))
        pfull[k] = P0 * pi_k ** (CP / RD)
        zprev = zm[k]
    pb = pfull.mean(axis=(1, 2), keepdims=True) * np.ones_like(pfull)
    pb = pb * (1.0 + 0.0 * pb)
    p = pfull - pb
    temp = theta * (pfull / P0) ** (RD / CP)
    alt = RD * temp / pfull * (1 + 0.0 * qv)
    mut = (psfc - p_top) * (1.0 + rng.normal(0, 2e-4, psfc.shape))
    # Winds.
    ug = rng.uniform(4, 12)
    vg = rng.uniform(-4, 4)
    zfu = np.zeros((nz, ny, nx + 1))
    zfu[:, :, 1:nx] = 0.5 * (zm[:, :, 1:] + zm[:, :, :-1])
    zfu[:, :, 0] = zm[:, :, 0]
    zfu[:, :, nx] = zm[:, :, nx - 1]
    prof = np.log1p(np.maximum(zfu, 1.0) / 0.1) / np.log1p(1000.0 / 0.1)
    u = ug * np.clip(prof, 0, 1.2)
    if regime == "stable":
        u = u + 5.0 * np.exp(-((zfu - 250.0) / 120.0) ** 2)
    zfv = np.zeros((nz, ny + 1, nx))
    zfv[:, 1:ny] = 0.5 * (zm[:, 1:] + zm[:, :-1])
    zfv[:, 0] = zm[:, 0]
    zfv[:, ny] = zm[:, ny - 1]
    v = vg * np.clip(np.log1p(np.maximum(zfv, 1.0) / 0.1) / np.log1p(1000.0 / 0.1), 0, 1.2)
    turb = 1.2 if dx <= 250 else (0.5 if dx <= 1000 else 0.15)
    u = u + rng.normal(0, turb, u.shape) * (zfu < 1.3 * zi)
    v = v + rng.normal(0, turb, v.shape) * (zfv < 1.3 * zi)
    if bnd == 0:
        u[:, :, nx] = u[:, :, 0]
        v[:, ny, :] = v[:, 0, :]
    w = rng.normal(0, 0.6 * turb, (nz + 1, ny, nx)) * (np.arange(nz + 1) > 0)[:, None, None]
    w[nz] = 0.0
    if regime == "edge":
        # Calm levels: du = dv = 0 between some levels near the PBL top.
        kcal = int(np.argmin(np.abs(zm[:, 0, 0] - zi)))
        for kk in (kcal - 1, kcal, kcal + 1):
            if 0 <= kk < nz:
                u[kk] = 0.0
                v[kk] = 0.0
    # TKE.
    if regime == "convective":
        e = 0.4 + 1.2 * np.exp(-((zm / zi - 0.35) / 0.4) ** 2)
        e = np.where(zm > 1.2 * zi, 0.02 * np.exp(-(zm - 1.2 * zi) / 400), e)
    elif regime == "stable":
        e = 0.25 * np.exp(-zm / 120.0)
    elif regime == "neutral":
        e = 0.6 * np.exp(-zm / zi)
    else:
        e = 0.3 * np.exp(-zm / 300.0)
    e = e * np.exp(rng.normal(0, 0.15, e.shape))
    if regime == "edge":
        r = rng.random(e.shape)
        e = np.where(r < 0.08, 0.0, e)
        e = np.where((r >= 0.08) & (r < 0.12), 1.0e-9, e)
        e = np.where((r >= 0.12) & (r < 0.14), 1.0e-30, e)
        e = np.where((r >= 0.15) & (r < 0.17), 30.0, e)
    else:
        e = np.where(zm > 6000.0, 0.0, e)
    # Surface layer.
    u1 = 0.5 * (u[0, :, :nx] + u[0, :, 1:])
    v1 = 0.5 * (v[0, :ny] + v[0, 1:])
    wspd = np.maximum(np.sqrt(u1 ** 2 + v1 ** 2), 0.1)
    if regime == "convective":
        hfx = rng.uniform(120, 320, (ny, nx))
        br = -rng.uniform(0.05, 2.0, (ny, nx))
        rmol = -rng.uniform(0.002, 0.05, (ny, nx))
        ust = rng.uniform(0.2, 0.55, (ny, nx))
    elif regime == "stable":
        hfx = -rng.uniform(5, 45, (ny, nx))
        br = rng.uniform(0.02, 0.5, (ny, nx))
        rmol = rng.uniform(0.002, 0.08, (ny, nx))
        ust = rng.uniform(0.05, 0.3, (ny, nx))
    elif regime == "neutral":
        hfx = rng.uniform(-3, 8, (ny, nx))
        br = rng.uniform(-0.01, 0.01, (ny, nx))
        rmol = rng.uniform(-1e-4, 1e-4, (ny, nx))
        ust = rng.uniform(0.3, 0.7, (ny, nx))
    else:
        hfx = rng.uniform(-30, 250, (ny, nx))
        br = np.where(hfx > 0, -0.3, 0.2) + rng.normal(0, 0.05, (ny, nx))
        br = np.where(rng.random((ny, nx)) < 0.1, 0.0, br)
        rmol = np.where(hfx > 0, -0.01, 0.02) * rng.uniform(0.2, 2, (ny, nx))
        rmol = np.where(rng.random((ny, nx)) < 0.1, 0.0, rmol)
        # The surface layer floors u* (sfclay's 0.001, MYNN's 0.005); a
        # zero u* under a positive flux is 0/0 in nonlocal_flux's brint.
        ust = rng.uniform(0.001, 0.5, (ny, nx))
        ust = np.where(rng.random((ny, nx)) < 0.1, 0.001, ust)
    ustm = ust * rng.uniform(0.95, 1.05, ust.shape)
    qfx = rng.uniform(0.0, 1.5e-4, (ny, nx))
    gamu = rng.normal(0, 1e-4, (ny, nx))
    gamv = rng.normal(0, 1e-4, (ny, nx))
    pblh = np.full((ny, nx), zi) * rng.uniform(0.8, 1.2, (ny, nx))
    msf = np.full((ny, nx), 1.0 if terrain == "flat" else rng.uniform(0.97, 1.03))
    t2 = theta - 300.0
    dt = {50.0: 0.3, 100.0: 0.5, 250.0: 1.2, 500.0: 2.5, 1000.0: 5.0,
          3000.0: 15.0, 9000.0: 45.0}[dx]
    c_k = 0.15 if dx > 300 else 0.10
    heat = 0.24 if isfflx in (0, 2) else 0.0
    if regime == "edge" and isfflx == 0:
        heat = 0.0
    nzp = nz + 1
    t_base = np.zeros(nzp)
    qv_base = np.zeros(nzp)
    u_base = np.zeros(nzp)
    v_base = np.zeros(nzp)
    mixfull = 1 if name.endswith("per") else 0
    return dict(
        name=name, regime=regime, dx=dx, terrain=terrain, bnd=bnd, isfflx=isfflx,
        hybrid=hybrid,
        ints=np.array([nx, ny, nz, bnd, isfflx, mixfull, 0], "<i4"),
        reals=np.array([dx, dy, dt, 0.25, c_k, heat, 0.0, vc["cf1"], vc["cf2"],
                        vc["cf3"], p_top], F),
        prof=[F(a) for a in (vc["fnm"], vc["fnp"], vc["dn"], vc["dnw"], vc["c1h"],
                             vc["c2h"], vc["c1f"], vc["c2f"], vc["znw"], t_base,
                             qv_base, u_base, v_base)],
        planes=[F(a) for a in (msf, mut, ust, ustm, hfx, qfx, br, rmol, wspd,
                               u1, v1, gamu, gamv, pblh)],
        u=F(u), v=F(v), w=F(w), ph=F(ph), phb=F(phb),
        mass=[F(a) for a in (t2, p, pb, alt, qv, qc, qi, e)],
    )


def fortran_order(a):
    """(k, j, i) woof order -> bytes with i fastest, then k, then j."""
    return np.ascontiguousarray(np.asarray(a, F).transpose(1, 0, 2)).astype("<f4").tobytes()


def write(cases, out):
    with open(out / "cases.bin", "wb") as f:
        f.write(np.array([len(cases)], "<i4").tobytes())
        for c in cases:
            f.write(c["ints"].tobytes())
            f.write(c["reals"].astype("<f4").tobytes())
            for a in c["prof"]:
                f.write(a.astype("<f4").tobytes())
            for a in c["planes"]:
                f.write(np.ascontiguousarray(a).astype("<f4").tobytes())
            for key in ("u", "v", "w", "ph", "phb"):
                f.write(fortran_order(c[key]))
            for a in c["mass"]:
                f.write(fortran_order(a))
    summary = [{k: c[k] for k in ("name", "regime", "dx", "terrain", "bnd",
                                  "isfflx", "hybrid")}
               | {"shape": [int(x) for x in c["ints"][:3]]} for c in cases]
    (out / "cases.json").write_text(json.dumps(summary, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--seed", type=int, default=471005)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cases = [make_case(spec, rng) for spec in CASES]
    write(cases, args.out)
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
