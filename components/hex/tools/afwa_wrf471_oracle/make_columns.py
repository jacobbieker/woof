#!/usr/bin/env python3
"""Deterministic synthetic mesh columns for the AFWA oracle and the AFWA tests (no data from any case).

``columns()`` is the one source of the columns: components/hex/tests/test_mpas_afwa_diagnostics.py loads it from here
and hands the same words to the hex module, to the WRF route's afwa_diagnostics kernels and (through ``write``) to the
Fortran oracle (run_afwa.F90).  ``write`` writes ``columns.bin``: little endian int32 (nc, nl), float32 ccn, melt, then
float32 arrays C-contiguous: zgrid (nl+1, nc); the (nl, nc) fields rho_dry T p qv qc qi qr qs qg u v; the (nc,) fields
psfc t2 q2 u10 v10 swdown raincv rainncv rainc rainnc.  ``severe_columns()`` adds what WRF's severe package reads
besides: ``severe_columns.bin`` is float32 dt, then w (nl+1, nc) (WRF's w_2; layer k reads face k) and the step's
2-5 km updraft helicity uh (nc), little endian, C-contiguous.

    make_columns.py OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np

NL, NC = 30, 24
CCN, MELT = 263.15, 45.0


def columns(seed=11, nl=NL, nc=NC):
    """Deterministic columns spanning the branches: warm/cold/near-freezing surfaces, saturated and dry layers,
    hydrometeor columns, precipitating and dry cells, terrain."""
    rng = np.random.default_rng(seed)
    f = np.float32
    terrain = rng.uniform(0.0, 1800.0, nc).astype(f)
    top = 17000.0
    eta = np.linspace(0.0, 1.0, nl + 1)[:, None] ** 1.25
    zgrid = (terrain[None, :] + (top - terrain[None, :]) * eta).astype(f)
    zm = 0.5 * (zgrid[:-1] + zgrid[1:])
    t_sfc = rng.uniform(262.0, 303.0, nc).astype(f)
    lapse = rng.uniform(5.5e-3, 7.2e-3, nc).astype(f)
    T = (t_sfc[None, :] - lapse[None, :] * (zm - terrain[None, :])).clip(205.0).astype(f)
    # an elevated warm layer over a cold surface (the melt-and-refreeze, ice pellet, branch)
    warm = ((zm - terrain[None, :]) > 1200.0) & ((zm - terrain[None, :]) < 3200.0)
    thin = ((zm - terrain[None, :]) > 1200.0) & ((zm - terrain[None, :]) < 1950.0)
    T[:, 8:11] = np.where(thin[:, 8:11], f(274.5), T[:, 8:11])     # a thin warm layer: partial melt, refreeze, ice pellets
    T[:, 11:14] += (warm[:, 11:14] * 11.0).astype(f)    # complete melt: rain, or freezing rain at a cold surface
    t_sfc[8:14] = 270.5
    p = (100000.0 * (T / t_sfc[None, :]) ** 5.2 * np.exp(-terrain[None, :] / 8000.0)).astype(f)
    exner = ((p / f(100000.0)) ** f(0.2854)).astype(f)
    theta = (T / exner).astype(f)
    # the module forms T = theta * exner: hand it words whose product is the temperature the WRF side is given
    T = (theta * exner).astype(f)
    es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    rh = rng.uniform(0.25, 1.05, (nl, nc))
    rh[:6] = rng.uniform(0.7, 1.0, (6, nc))
    rh[:16, 8:14] = rng.uniform(0.86, 1.0, (16, 6))        # a deep moist column under the cloud top
    rh[:6, 18:20] = 0.97                                    # calm, dry-aloft, near-saturated surface: radiation fog / haze
    qv = (0.622 * rh * es / (p - rh * es)).clip(1e-8, 0.03).astype(f)
    cloud = rng.uniform(0, 1.0, (nl, nc)) < 0.35
    qc = (cloud * rng.uniform(0, 1.5e-3, (nl, nc))).astype(f)
    qi = (cloud * (T < 268) * rng.uniform(0, 5e-4, (nl, nc))).astype(f)
    qr = ((zm - terrain < 4000) * rng.uniform(0, 2e-3, (nl, nc)) * (rng.uniform(size=nc) < 0.6)).astype(f)
    qs = ((T < 275) * rng.uniform(0, 1e-3, (nl, nc)) * (rng.uniform(size=nc) < 0.6)).astype(f)
    qg = ((T < 280) * rng.uniform(0, 1.5e-3, (nl, nc)) * (rng.uniform(size=nc) < 0.4)).astype(f)
    qr[:, 18:20] = qs[:, 18:20] = qg[:, 18:20] = 0.0
    qv[6:, 18:20] *= f(0.05)
    # no echo in the top four layers: WRF's echo top reads the extra mass row when an echo reaches the model top
    qr[nl - 4:] = qs[nl - 4:] = qg[nl - 4:] = 0.0
    rho_dry = (p / (287.0 * T * (1 + 0.608 * qv))).astype(f)
    mass = dict(
        zgrid=zgrid, zz=np.ones((nl, nc), f), rho_zz=rho_dry, theta=theta, exner=exner,
        pressure_base=(p * f(0.98)).astype(f), pressure_p=(p - (p * f(0.98)).astype(f)).astype(f),
        u=rng.uniform(-25, 35, (nl, nc)).astype(f), v=rng.uniform(-30, 25, (nl, nc)).astype(f),
        qv=qv, qc=qc, qi=qi, qr=qr, qs=qs, qg=qg, T=T, p=(p * f(0.98)).astype(f) + (p - (p * f(0.98)).astype(f)).astype(f))
    mass["u"][:6, 18:20] = mass["v"][:6, 18:20] = f(0.5)
    surface = dict(
        surface_pressure=(p[0] * rng.uniform(1.0, 1.01, nc)).astype(f), t2m=t_sfc + rng.uniform(-1.5, 1.5, nc).astype(f),
        q2=(qv[0] * rng.uniform(0.9, 1.1, nc)).astype(f), u10=rng.uniform(-12, 12, nc).astype(f),
        v10=rng.uniform(-12, 12, nc).astype(f), swdnb=rng.uniform(0, 900, nc).astype(f),
        raincv=(rng.uniform(size=nc) < 0.3) * rng.uniform(0, 1.2, nc).astype(f),
        rainncv=(rng.uniform(size=nc) < 0.7) * rng.uniform(0, 3.0, nc).astype(f),
        rainc=rng.uniform(0, 40, nc).astype(f), rainnc=rng.uniform(0, 90, nc).astype(f))
    surface = {k: np.asarray(v, f) for k, v in surface.items()}
    # near-freezing and cold surfaces exist among the precipitating cells
    surface["t2m"][:6] = np.linspace(268.0, 276.5, 6).astype(f)
    surface["rainncv"][:14] = np.maximum(surface["rainncv"][:14], f(0.4))
    surface["swdnb"][8:14] = 0.0                       # no insolation correction: a cold surface stays cold
    return mass, surface


SEVERE_DT = 12.0


def severe_columns(seed=7, nl=NL, nc=NC):
    """Vertical velocity and updraft helicity for the severe package: strong and weak updrafts and downdrafts below
    400 hPa, helicity on both sides of the tornado ramp (25-75 m2 s-2) and the hail scale (100 m2 s-2)."""
    rng = np.random.default_rng(seed)
    f = np.float32
    shape = np.sin(np.linspace(0.0, np.pi, nl + 1))[:, None]
    peak = rng.uniform(-6.0, 45.0, nc)
    w = (shape * peak[None, :] + rng.uniform(-1.5, 1.5, (nl + 1, nc))).astype(f)
    w[0] = 0.0
    uh = rng.uniform(0.0, 220.0, nc).astype(f)
    uh[rng.uniform(size=nc) < 0.25] = 0.0
    return w, uh


def write_severe(out):
    w, uh = severe_columns()
    with open(Path(out) / "severe_columns.bin", "wb") as h:
        h.write(np.array([SEVERE_DT], "<f4").tobytes())
        h.write(w.astype("<f4").tobytes())
        h.write(uh.astype("<f4").tobytes())


def write(out):
    mass, surface = columns()
    with open(Path(out) / "columns.bin", "wb") as h:
        h.write(np.array([NC, NL], "<i4").tobytes())
        h.write(np.array([CCN, MELT], "<f4").tobytes())
        h.write(mass["zgrid"].astype("<f4").tobytes())
        for name in ("rho_zz", "T", "p", "qv", "qc", "qi", "qr", "qs", "qg", "u", "v"):
            h.write(mass[name].astype("<f4").tobytes())
        for name in ("surface_pressure", "t2m", "q2", "u10", "v10", "swdnb", "raincv", "rainncv", "rainc", "rainnc"):
            h.write(surface[name].astype("<f4").tobytes())


if __name__ == "__main__":
    write(sys.argv[1])
    write_severe(sys.argv[1])
