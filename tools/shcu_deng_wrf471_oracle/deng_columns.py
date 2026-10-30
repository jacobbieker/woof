"""Column fixtures for the WRF v4.7.1 Deng shallow cumulus oracle.

Writes the stream ``run_deng_oracle.F90`` reads and reads the stream it
writes (both documented there).  Every float is float32, the exact word both
sides read.

The columns are hydrostatically built boundary-layer soundings: a
superadiabatic surface layer, a well-mixed layer whose top sits at a chosen
relative humidity, a capping inversion and a free troposphere with a
tropopause.  Cases vary over what moves the scheme -- grid length, time step
(and so TIMEC and the NBC leapfrog), vertical resolution, MYJ versus MYNN
TKE, adaptive stepping, warm versus freezing cloud layers, the local-time
6 AM PBLMAX reset -- and some start from an injected NBC/updraft history
(CLDAREAA/B, CLDLIQA/B, CLDDPTHB >= 4 km so the downdraft and drizzle arms
run, LTOPB/CLDTOPB/WUB, CAPESAVE/RADSAVE and AINCKFSA) so the dissipation
code is graded from the first step.

    python deng_columns.py write OUT.bin [--set structured]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

F = np.float32
I = np.int32

G = 9.81
R_D = 287.0
CP = 7.0 * R_D / 2.0
RCP = R_D / CP
NSH = 100   # AINCKFSA's second extent (Registry: i{nsh}j)

STATE_2D = ("rainsh", "rainshv", "rainshvb", "pblmax", "capesave", "xtime1",
            "radsave", "clddpthb", "cldtopb", "mavail", "pblhavg", "pratesh")
STATE_INT = ("ltopb", "kdcldtop", "kdcldbas")
STATE_3D = ("w0avg", "tkeavg", "cldareaa", "cldareab", "cldliqa", "cldliqb",
            "wub")
STEP_3D = ("u", "v", "th", "t", "qv", "qc", "qr", "p", "pi", "rho", "dz8w",
           "tke_pbl", "qke", "el_pbl", "exch_h", "rthratenlw", "rthratensw")
STEP_W = ("z_at_w", "w")
OUT_3D = ("rthshten", "rqvshten", "rqcshten", "rqrshten", "rushten",
          "rvshten", "rdcashten", "rqcdcshten", "cldfra_sh", "ca_rad",
          "cw_rad", "w0avg", "tkeavg", "cldareaa", "cldareab", "cldliqa",
          "cldliqb", "wub")
OUT_2D = STATE_2D
OUT_INT = STATE_INT


@dataclass
class Case:
    """One horizontal tile of ``ncol`` columns run for ``nsteps`` steps."""
    name: str
    nz: int
    dt: float
    dx: float
    gmt: float
    bl_pbl: int          # 2 MYJ, 5 MYNN
    adapt: bool
    dnw: np.ndarray      # (nz,)
    ht: np.ndarray       # (ncol,)
    xlong: np.ndarray    # (ncol,)
    state: dict          # STATE_2D (ncol,), STATE_INT (ncol,), ainckfsa (100, ncol), STATE_3D (nz, ncol)
    steps: list          # per step: itimestep, xtime, pblh (ncol,), STEP_3D (nz, ncol), STEP_W (nz+1, ncol)
    meta: list = field(default_factory=list)   # per column record

    @property
    def ncol(self) -> int:
        return int(self.ht.shape[0])


# ---------------------------------------------------------------------------
# Thermodynamics used only to build the soundings (any consistent choice
# works: the oracle and WOOF read the same words).
# ---------------------------------------------------------------------------
def _qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / np.maximum(p - es, 1.0)


def _levels(rng, nz, top):
    """Full-level heights above ground, stretched from a ~25-80 m first layer."""
    dz0 = rng.uniform(25.0, 80.0)
    # geometric stretch r with dz0 * (r**nz - 1)/(r - 1) = top
    lo, hi = 1.0001, 1.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if dz0 * (mid ** nz - 1.0) / (mid - 1.0) > top:
            hi = mid
        else:
            lo = mid
    r = 0.5 * (lo + hi)
    dz = dz0 * r ** np.arange(nz)
    dz = np.minimum(dz, 1200.0)
    dz *= top / dz.sum()
    return np.concatenate([[0.0], np.cumsum(dz)])


def _sounding(rng, nz, regime):
    """One column: returns dict of mid-level / full-level profiles."""
    top = rng.uniform(15000.0, 20000.0)
    zf = _levels(rng, nz, top)
    zm = 0.5 * (zf[1:] + zf[:-1])
    p_sfc = rng.uniform(85000.0, 102000.0) if regime != "highland" else \
        rng.uniform(70000.0, 80000.0)
    h = {"shallow": rng.uniform(600.0, 1800.0),
         "deep": rng.uniform(1200.0, 2800.0),
         "cold": rng.uniform(300.0, 1200.0),
         "stable": rng.uniform(60.0, 400.0),
         "saturated": rng.uniform(300.0, 1500.0),
         "highland": rng.uniform(800.0, 2500.0),
         "dry": rng.uniform(1500.0, 3500.0),
         "fog": rng.uniform(20.0, 120.0),
         "tiny": rng.uniform(2.0, 12.0)}[regime]
    theta_m = {"shallow": rng.uniform(292.0, 302.0),
               "deep": rng.uniform(298.0, 306.0),
               "cold": rng.uniform(262.0, 272.0),
               "stable": rng.uniform(280.0, 295.0),
               "saturated": rng.uniform(285.0, 300.0),
               "highland": rng.uniform(300.0, 312.0),
               "dry": rng.uniform(305.0, 315.0),
               "fog": rng.uniform(278.0, 290.0),
               "tiny": rng.uniform(285.0, 300.0)}[regime]
    zs = rng.uniform(30.0, 120.0)
    dsl = rng.uniform(0.0, 2.0) if regime != "stable" else -rng.uniform(1.0, 6.0)
    jump = rng.uniform(0.2, 4.0)
    dinv = rng.uniform(50.0, 300.0)
    gamma = rng.uniform(2.0, 6.5) * 1.0e-3
    if regime == "deep":
        gamma = rng.uniform(1.0, 3.5) * 1.0e-3
        jump = rng.uniform(0.0, 1.0)
    trop = rng.uniform(10000.0, 14000.0)

    def theta(z):
        th = np.where(z < zs, theta_m + dsl * (1.0 - z / zs), theta_m)
        if regime == "stable":
            th = theta_m + dsl * np.maximum(0.0, 1.0 - z / h) + 0.0
        above = np.clip(z - h, 0.0, dinv) / dinv * jump
        th = th + above + gamma * np.maximum(z - h - dinv, 0.0)
        th = th + 12.0e-3 * np.maximum(z - trop, 0.0)
        return th

    th_m = theta(zm)
    rh_top = {"shallow": rng.uniform(0.80, 1.0), "deep": rng.uniform(0.85, 1.0),
              "cold": rng.uniform(0.80, 1.0), "stable": rng.uniform(0.5, 0.95),
              "saturated": rng.uniform(1.0, 1.03), "highland": rng.uniform(0.6, 0.95),
              "dry": rng.uniform(0.25, 0.6), "fog": rng.uniform(1.0, 1.04),
              "tiny": rng.uniform(0.7, 1.0)}[regime]
    ft_frac = rng.uniform(0.15, 0.9)
    hq = rng.uniform(1500.0, 3500.0)
    # first pass with a dry hydrostatic estimate, then two moist passes
    qv = np.zeros(nz)
    for _ in range(3):
        tv_guess = th_m * (1.0 + 0.608 * qv)
        pf = np.empty(nz + 1)
        pf[0] = p_sfc
        for k in range(nz):
            ex = (pf[k] / 1.0e5) ** RCP
            tvk = tv_guess[k] * ex
            pf[k + 1] = pf[k] * np.exp(-G * (zf[k + 1] - zf[k]) / (R_D * tvk))
        pm = 0.5 * (pf[1:] + pf[:-1])
        tm = th_m * (pm / 1.0e5) ** RCP
        qs = _qsat(tm, pm)
        kh = int(np.searchsorted(zm, h)) - 1
        kh = int(np.clip(kh, 0, nz - 2))
        qm = rh_top * qs[kh]
        qv = np.where(zm <= h, qm, ft_frac * qm * np.exp(-(zm - h) / hq))
        qv = np.where(zm < zs, qv * (1.0 + rng.uniform(0.0, 0.08)), qv)
        cap = 1.02 if regime in ("saturated", "fog") else 0.98
        qv = np.minimum(qv, cap * qs)
        qv = np.maximum(qv, 1.0e-7)
    return dict(zf=zf, zm=zm, pf=pf, pm=pm, th=th_m, t=tm, qv=qv, qs=qs, h=h,
                p_sfc=p_sfc, top=top)


def _column(rng, nz, regime, cfg):
    s = _sounding(rng, nz, regime)
    zm, zf, h = s["zm"], s["zf"], s["h"]
    pi = (s["pm"] / 1.0e5) ** RCP
    rho = s["pm"] / (R_D * s["t"] * (1.0 + 0.608 * s["qv"])) * (1.0 + s["qv"])
    rh = s["qv"] / s["qs"]
    qc = np.where(rh > 0.97, rng.uniform(0.0, 4.0e-4) * (rh - 0.97) / 0.06, 0.0)
    if rng.random() < 0.3:
        band = (zm > h) & (zm < h + rng.uniform(200.0, 1500.0))
        qc = qc + np.where(band, rng.uniform(1.0e-6, 3.0e-4), 0.0)
    qr = np.where(zm < h + 2000.0, rng.uniform(0.0, 2.0e-4), 0.0) \
        if rng.random() < 0.4 else np.zeros(nz)
    tke0 = rng.uniform(0.05, 4.0) if regime != "stable" else rng.uniform(0.01, 0.3)
    zr = np.clip(zm / h, 0.0, 1.0)
    tke = tke0 * (1.0 - zr) ** 1.5 + np.where(zm < h, 0.02, 0.005)
    el = np.where(zm < h, 0.4 * zm / (1.0 + 0.4 * zm / rng.uniform(40.0, 150.0)),
                  rng.uniform(1.0, 20.0))
    kth = np.where(zm < h, rng.uniform(5.0, 150.0) * zr * (1.0 - zr) ** 2 + 0.1, 0.1)
    u0 = rng.uniform(-12.0, 12.0)
    v0 = rng.uniform(-12.0, 12.0)
    shear = rng.uniform(-3.0, 6.0) * 1.0e-3
    u = u0 + shear * zm + rng.normal(0.0, 0.3, nz)
    v = v0 + 0.5 * shear * zm + rng.normal(0.0, 0.3, nz)
    if rng.random() < 0.05:
        u[:] = 0.0
        v[:] = 0.0
    wamp = rng.uniform(-0.3, 0.6)
    w = wamp * np.sin(np.pi * np.clip(zf / (h + 3000.0), 0.0, 1.0)) \
        + rng.normal(0.0, 0.05, nz + 1)
    w[0] = 0.0
    lw = np.full(nz, -rng.uniform(1.0e-5, 4.0e-5))
    ktop = int(np.clip(np.searchsorted(zm, h), 1, nz - 2))
    lw[ktop] -= rng.uniform(0.0, 2.0e-4)
    sw = np.full(nz, rng.uniform(0.0, 3.0e-5) if cfg["day"] else 0.0)
    sw[ktop] += rng.uniform(0.0, 5.0e-5) if cfg["day"] else 0.0
    return dict(sounding=s, u=u, v=v, th=s["th"], t=s["t"], qv=s["qv"], qc=qc,
                qr=qr, p=s["pm"], pi=pi, rho=rho, dz8w=np.diff(zf),
                tke=tke, el=el, kth=kth, w=w, z_at_w=zf, lw=lw, sw=sw,
                pblh=h, ktop=ktop)


def _inject_state(rng, col, nz, state, i, regime="shallow"):
    """An NBC/updraft history for column i, consistent with its sounding."""
    zm = col["sounding"]["zm"]
    h = col["pblh"]
    k1 = int(np.clip(np.searchsorted(zm, h * rng.uniform(0.6, 1.0)), 2, nz - 4))
    k2 = int(np.clip(k1 + rng.integers(1, 8), k1, nz - 3))
    area = np.zeros(nz)
    liq = np.zeros(nz)
    peak = rng.uniform(0.002, 0.95)
    for k in range(k1, k2 + 1):
        area[k] = peak * rng.uniform(0.2, 1.0)
        liq[k] = rng.uniform(1.0e-6, 1.5e-3)
    if rng.random() < 0.15:          # near-overcast, the >0.99999 clamp
        area[k1] = 1.0
    state["cldareab"][:, i] = area
    state["cldliqb"][:, i] = liq
    state["cldareaa"][:, i] = area * rng.uniform(0.5, 1.2, nz)
    state["cldliqa"][:, i] = liq * rng.uniform(0.5, 1.5, nz)
    depth = rng.choice([0.0, rng.uniform(100.0, 3900.0), rng.uniform(4000.1, 7000.0)],
                       p=([0.05, 0.15, 0.8] if regime == "deep"
                          else [0.2, 0.45, 0.35]))
    state["clddpthb"][i] = depth
    ltop = int(np.clip(k2 + rng.integers(0, 6), 1, nz - 1))
    state["ltopb"][i] = ltop + 1 if rng.random() < 0.85 else 1
    state["cldtopb"][i] = zm[ltop]
    state["wub"][:, i] = np.where(np.arange(nz) <= ltop, rng.uniform(0.0, 6.0, nz), 0.0)
    state["capesave"][i] = rng.choice([rng.uniform(0.0, 99.0), rng.uniform(100.0, 600.0)])
    state["radsave"][i] = rng.uniform(150.0, 1500.0)
    state["pblmax"][i] = rng.choice([0.0, h * rng.uniform(0.5, 1.5)])
    state["rainshvb"][i] = rng.choice([0.0, rng.uniform(0.0, 0.2)])
    state["xtime1"][i] = rng.choice([0.0, rng.uniform(0.0, 600.0)])
    nfill = int(rng.integers(0, NSH))
    state["ainckfsa"][:nfill, i] = rng.uniform(0.0, 3.0, nfill)
    state["w0avg"][:, i] = rng.normal(0.0, 0.2, nz)
    state["tkeavg"][:, i] = rng.uniform(0.0, 1.0, nz)


CASES = (
    # name, nz, dt, dx, bl_pbl, adapt, ncol, nsteps, regimes, inject, day, gmt
    ("warm-mynn-cold-start", 40, 30.0, 3000.0, 5, False, 64, 6,
     ("shallow", "deep", "saturated", "stable", "dry"), 0.0, True, 18.0),
    ("warm-myj-history", 50, 60.0, 9000.0, 2, False, 64, 5,
     ("shallow", "deep", "saturated", "highland"), 0.8, True, 15.5),
    ("cold-mynn-history", 45, 45.0, 4000.0, 5, False, 48, 5,
     ("cold", "shallow", "saturated"), 0.8, False, 3.0),
    ("fine-adaptive", 35, 20.0, 1000.0, 5, True, 48, 6,
     ("shallow", "deep", "dry", "highland"), 0.5, True, 21.25),
    ("coarse-myj", 60, 90.0, 27000.0, 2, False, 48, 4,
     ("deep", "shallow", "cold", "saturated"), 0.6, True, 12.0),
    ("dawn-reset", 32, 36.0, 12000.0, 5, False, 32, 4,
     ("shallow", "deep", "stable"), 0.5, False, 0.0),
    ("edges", 40, 30.0, 5000.0, 2, False, 48, 4,
     ("fog", "tiny", "deep", "cold"), 0.7, True, 9.0),
    ("fine-deep", 45, 10.0, 1500.0, 5, False, 48, 4,
     ("deep",), 0.9, True, 19.0),
)


def build_cases(seed: int = 20261029) -> list[Case]:
    rng = np.random.default_rng(seed)
    cases = []
    for (name, nz, dt, dx, bl, adapt, ncol, nsteps, regimes, inject, day,
         gmt) in CASES:
        cfg = {"day": day}
        cols = [_column(rng, nz, regimes[i % len(regimes)], cfg)
                for i in range(ncol)]
        # dnw: one eta increment profile per tile (WRF's dnw is per domain),
        # taken from the first column's pressure.
        pf = cols[0]["sounding"]["pf"]
        eta = (pf - pf[-1]) / (pf[0] - pf[-1])
        dnw = np.diff(eta)
        ht = rng.uniform(0.0, 600.0, ncol)
        ht = np.where([c["sounding"]["p_sfc"] < 81000.0 for c in cols],
                      rng.uniform(1800.0, 2600.0, ncol), ht)
        if name == "dawn-reset":
            # TIMELOC = GMT + XTIME/60 + XLONG/15 = 6 at the first step for
            # half the tile: the PBLMAX reset of deng_shcu.
            xlong = np.where(np.arange(ncol) % 2 == 0, 90.0,
                             rng.uniform(-180.0, 180.0, ncol))
        else:
            xlong = rng.uniform(-180.0, 180.0, ncol)
        state = {k: np.zeros(ncol) for k in STATE_2D}
        state.update({k: np.ones(ncol, dtype=np.int64) for k in STATE_INT})
        state["ainckfsa"] = np.zeros((NSH, ncol))
        state.update({k: np.zeros((nz, ncol)) for k in STATE_3D})
        state["mavail"][:] = rng.uniform(0.05, 1.0, ncol)
        meta = []
        for i, c in enumerate(cols):
            injected = bool(rng.random() < inject)
            if injected:
                _inject_state(rng, c, nz, state, i,
                              regimes[i % len(regimes)])
            meta.append({"case": name, "column": i,
                         "regime": regimes[i % len(regimes)],
                         "injected_history": injected})
        steps = []
        it0 = int(rng.integers(1, 2000)) if name != "dawn-reset" else 1
        xt0 = 0.0 if name == "dawn-reset" else float(rng.uniform(0.0, 1440.0))
        for s in range(nsteps):
            fields = {}
            # Small step-to-step evolution of the forcing the dycore would
            # bring; the Deng state itself is carried by the Fortran.
            for key, src in (("u", "u"), ("v", "v"), ("th", "th"), ("t", "t"),
                             ("qv", "qv"), ("qc", "qc"), ("qr", "qr"),
                             ("p", "p"), ("pi", "pi"), ("rho", "rho"),
                             ("dz8w", "dz8w"), ("el_pbl", "el"),
                             ("exch_h", "kth"), ("rthratenlw", "lw"),
                             ("rthratensw", "sw")):
                arr = np.stack([c[src] for c in cols], axis=1).astype(float)
                if s and key in ("u", "v"):
                    arr = arr + rng.normal(0.0, 0.2, arr.shape)
                if s and key in ("qv",):
                    arr = arr * (1.0 + rng.normal(0.0, 0.01, arr.shape))
                if s and key in ("t",):
                    arr = arr + rng.normal(0.0, 0.05, arr.shape)
                fields[key] = arr
            tke = np.stack([c["tke"] for c in cols], axis=1)
            tke = tke * (1.0 + (0.1 * rng.normal(0.0, 1.0, tke.shape) if s else 0.0))
            tke = np.maximum(tke, 0.0)
            fields["tke_pbl"] = tke
            fields["qke"] = 2.0 * tke
            fields["z_at_w"] = np.stack([c["z_at_w"] for c in cols], axis=1) + ht
            w = np.stack([c["w"] for c in cols], axis=1)
            if s:
                w = w + rng.normal(0.0, 0.05, w.shape)
                w[0] = 0.0
            fields["w"] = w
            pblh = np.array([c["pblh"] for c in cols])
            if s:
                pblh = pblh * (1.0 + rng.normal(0.0, 0.03, ncol))
            steps.append({"itimestep": it0 + s,
                          "xtime": xt0 + s * dt / 60.0,
                          "pblh": pblh, **fields})
        cases.append(Case(name=name, nz=nz, dt=dt, dx=dx, gmt=gmt,
                          bl_pbl=bl, adapt=adapt, dnw=dnw, ht=ht,
                          xlong=xlong, state=state, steps=steps, meta=meta))
    return cases


# ---------------------------------------------------------------------------
# Stream I/O
# ---------------------------------------------------------------------------
def _f(x):
    return np.ascontiguousarray(np.asarray(x, dtype=F)).tobytes()


def _i(x):
    return np.ascontiguousarray(np.asarray(x, dtype=I)).tobytes()


def write_input(path, cases) -> None:
    out = [b"DENGIN1 ", _i([len(cases)])]
    for c in cases:
        out += [_i([c.ncol, c.nz, len(c.steps), c.bl_pbl, int(c.adapt)]),
                _f([c.dt, c.dx, c.gmt]), _f(c.dnw), _f(c.ht), _f(c.xlong)]
        out += [_f(c.state[k]) for k in STATE_2D]
        out += [_i(c.state[k]) for k in STATE_INT]
        out.append(_f(c.state["ainckfsa"]))
        out += [_f(c.state[k]) for k in STATE_3D]
        for s in c.steps:
            out += [_i([s["itimestep"]]), _f([s["xtime"]]), _f(s["pblh"])]
            out += [_f(s[k]) for k in STEP_3D]
            out += [_f(s[k]) for k in STEP_W]
    Path(path).write_bytes(b"".join(out))


def _read_bytes(path) -> bytes:
    data = Path(path).read_bytes()
    if str(path).endswith(".gz"):
        import gzip
        data = gzip.decompress(data)
    return data


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, dtype, count):
        n = np.dtype(dtype).itemsize * count
        arr = np.frombuffer(self.data, dtype=dtype, count=count,
                            offset=self.pos)
        self.pos += n
        return arr.copy()


def read_input(path) -> list[dict]:
    """The input stream back as arrays (the exact float32 words)."""
    r = _Reader(_read_bytes(path))
    assert r.take("S8", 1)[0] == b"DENGIN1 "
    ncase = int(r.take(I, 1)[0])
    cases = []
    for _ in range(ncase):
        ncol, nz, nsteps, bl, adapt = (int(x) for x in r.take(I, 5))
        dt, dx, gmt = (F(x) for x in r.take(F, 3))
        c = {"ncol": ncol, "nz": nz, "nsteps": nsteps, "bl_pbl": bl,
             "adapt": bool(adapt), "dt": dt, "dx": dx, "gmt": gmt,
             "dnw": r.take(F, nz), "ht": r.take(F, ncol),
             "xlong": r.take(F, ncol), "state": {}, "steps": []}
        for k in STATE_2D:
            c["state"][k] = r.take(F, ncol)
        for k in STATE_INT:
            c["state"][k] = r.take(I, ncol)
        c["state"]["ainckfsa"] = r.take(F, NSH * ncol).reshape(NSH, ncol)
        for k in STATE_3D:
            c["state"][k] = r.take(F, nz * ncol).reshape(nz, ncol)
        for _s in range(nsteps):
            s = {"itimestep": int(r.take(I, 1)[0]), "xtime": F(r.take(F, 1)[0]),
                 "pblh": r.take(F, ncol)}
            for k in STEP_3D:
                s[k] = r.take(F, nz * ncol).reshape(nz, ncol)
            for k in STEP_W:
                s[k] = r.take(F, (nz + 1) * ncol).reshape(nz + 1, ncol)
            c["steps"].append(s)
        cases.append(c)
    assert r.pos == len(r.data)
    return cases


def read_output(path) -> list[dict]:
    """The oracle's per-step outputs: cases[c]['steps'][s][name]."""
    r = _Reader(_read_bytes(path))
    assert r.take("S8", 1)[0] == b"DENGOU1 "
    ncase = int(r.take(I, 1)[0])
    cases = []
    for _ in range(ncase):
        ncol, nz, nsteps = (int(x) for x in r.take(I, 3))
        c = {"ncol": ncol, "nz": nz, "nsteps": nsteps, "steps": []}
        for _s in range(nsteps):
            fatal, messages = (int(x) for x in r.take(I, 2))
            s = {"fatal": fatal, "messages": messages}
            for k in OUT_3D:
                s[k] = r.take(F, nz * ncol).reshape(nz, ncol)
            for k in OUT_2D:
                s[k] = r.take(F, ncol)
            for k in OUT_INT:
                s[k] = r.take(I, ncol)
            s["ainckfsa"] = r.take(F, NSH * ncol).reshape(NSH, ncol)
            c["steps"].append(s)
        cases.append(c)
    assert r.pos == len(r.data)
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("out")
    w.add_argument("--seed", type=int, default=20261029)
    w.add_argument("--meta")
    args = ap.parse_args()
    cases = build_cases(args.seed)
    write_input(args.out, cases)
    if args.meta:
        meta = [m for c in cases for m in c.meta]
        Path(args.meta).write_text(json.dumps(meta, indent=0) + "\n",
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
