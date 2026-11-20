"""A synthetic 3D-Var case at a chosen grid size for the V1.6 timings
(docs/dev/WRFDA-3DVAR-PLAN.md, section 4.12): a random but valid ``be.dat``
(written by :func:`woof.da.var.be_dat.write_be_dat` and read back), a smooth
first guess, and conventional observations (sound, synop, airep, geoamv) at
the plan's density, about 10^5 values on 200x200x50.

The observations are built directly as innovations (``O - B``, ``qc``,
``error`` and the :class:`~woof.da.var.innov_wrfda.Geometry` WRFDA's
``da_convert_zk`` leaves), so no observation file is needed; ``H'`` is linear
and its cost does not depend on the values.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from woof.da.var import be_dat as bd
from woof.da.var import innov_wrfda as iw
from woof.da.var import obs_wrfda as ow
from woof.da.var.obsop_wrfda import FlatObsTerm

#: (reports, levels per report) per type for 200x200 columns; scaled by area.
MIX = {"sound": (300, 40), "synop": (3000, 1), "airep": (6000, 1), "geoamv": (7000, 1)}
ERRORS = {"u": 2.0, "v": 2.0, "t": 1.0, "q": 1.0e-3, "p": 100.0}
U_KEYS = ("grid_box_area", "coefx", "coefy", "qs", "q", "psac", "dnw", "rho", "p", "t", "c1h", "c2h")


def be_dat(nx: int, ny: int, nz: int, seed: int = 7) -> bd.BeDat:
    """A cv5 ``be.dat`` with one latitude bin, written and read back."""
    rng = np.random.default_rng(seed)
    bins = np.repeat(np.arange(1, nz + 1, dtype=np.int64)[None, None, :], nx, 0).repeat(ny, 1)
    bins2d = np.ones((nx, ny), dtype=np.int64)
    reg = {"psi_chi": rng.uniform(-0.3, 0.3, nz), "psi_ps": rng.uniform(-1e-3, 1e-3, (nz, 1)),
           "psi_t": rng.uniform(-1e-4, 1e-4, (nz, nz, 1))}

    def eig(name, n, scale):
        q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        ev = scale * np.exp(-0.12 * np.arange(n))  # a decaying spectrum: mz ~ 35 of 50 at 99 %
        return bd.BeEigen(name, q, ev, q[:, :, None].copy(), ev[:, None].copy())

    eigs = (eig("psi", nz, 1e7), eig("chi_u", nz, 5e6), eig("t_u", nz, 1.0), eig("rh_u", nz, 0.1))
    be = bd.BeDat(
        ni=nx, nj=ny, nk=nz, bin_type=5, eta_be=None, lat_min=-90.0, lat_max=90.0, binwidth_lat=10.0,
        hgt_min=0.0, hgt_max=20000.0, binwidth_hgt=1000.0, num_bins=nz, num_bins2d=1, bin=bins,
        bin2d=bins2d, cv_options=5, regcoeff=reg, eigen=eigs, eigen_ps=eig("ps_u", 1, 1e4),
        rf_lengthscale=tuple(rng.uniform(2.0, 8.0, nz) for _ in range(4)),
        rf_lengthscale_ps=rng.uniform(2.0, 8.0, 1), endian=">", real_size=8, sha256="")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "be.dat"
        bd.write_be_dat(path, be)
        return bd.read_be_dat(path)


def background(nx: int, ny: int, nz: int, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    eta = np.linspace(0.995, 0.005, nz)
    ii, jj = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing="ij")
    psfc = 1.0e5 - 3.0e3 * (ii + jj)
    ptop = 5000.0
    p = ptop + (psfc[:, :, None] - ptop) * eta[None, None, :]
    t = 290.0 * (p / 1.0e5) ** 0.19 + 0.3 * rng.standard_normal((nx, ny, nz))
    q = 0.012 * (p / 1.0e5) ** 3
    qs = 1.3 * q + 1e-5
    rho = p / (287.0 * t)
    dnw = -np.diff(np.concatenate([[1.0], (eta[:-1] + eta[1:]) / 2, [0.0]]))
    return {
        "grid_box_area": np.full((nx, ny), 12000.0 ** 2) * rng.uniform(0.98, 1.02, (nx, ny)),
        "coefx": np.full((nx, ny), 1.0 / 24000.0), "coefy": np.full((nx, ny), 1.0 / 24000.0),
        "qs": qs, "q": q, "psac": psfc - ptop, "dnw": dnw, "rho": rho, "p": p, "t": t,
        "c1h": eta.copy(), "c2h": np.zeros(nz),
        "u": 5.0 + rng.standard_normal((nx, ny, nz)), "v": rng.standard_normal((nx, ny, nz)),
        "psfc": psfc,
    }


def innovations(nx: int, ny: int, nz: int, seed: int = 7, scale: float = 1.0) -> ow.WrfdaObs:
    """Conventional innovations with ``scale`` times the 200x200 counts per area."""
    rng = np.random.default_rng(seed + 1)
    area = nx * ny / (200.0 * 200.0) * scale
    types = {}
    for name, (nrep0, nlev) in MIX.items():
        nrep = max(1, int(round(nrep0 * area)))
        i = rng.integers(1, nx, nrep)  # 1-based lower-left corner, i + 1 <= nx
        j = rng.integers(1, ny, nrep)
        dx, dy = rng.uniform(0, 1, nrep), rng.uniform(0, 1, nrep)
        levels = np.full(nrep, nlev, dtype=np.int64)
        info = {"i": i, "j": j, "dx": dx, "dy": dy, "dxm": 1.0 - dx, "dym": 1.0 - dy,
                "levels": levels, "proc_domain": np.ones(nrep, dtype=bool),
                "elv": np.zeros(nrep), "lat": np.full(nrep, 40.0),
                "name": np.array(["X"] * nrep), "id": np.array(["X"] * nrep)}
        m = int(levels.sum())
        if name in iw.SURFACE_TYPES:
            zk = rng.uniform(1.0, 1.9, m)
        elif nlev > 1:
            zk = np.concatenate([np.sort(rng.uniform(1.0, nz - 0.5, nlev))[::-1] for _ in range(nrep)])
        else:
            zk = rng.uniform(1.0, nz - 0.5, m)
        zk[rng.uniform(0, 1, m) < 0.02] = iw.MISSING_R  # below ground / above the top
        fields = {}
        for v in ow._FIELDS[name]:
            err = np.full(m, ERRORS[v])
            qc = np.where(rng.uniform(0, 1, m) < 0.05, -88, 0).astype(np.int64)
            fields[v] = ow.Field(rng.standard_normal(m) * err, qc, err)
        t = ow.ObsType(name, info, {}, fields)
        t.extra["zk"] = zk
        t.extra["geometry"] = iw.Geometry.of(t, nz)
        types[name] = t
    return ow.WrfdaObs(types=types, ntotal={k: v.nlocal for k, v in types.items()}, analysis_minutes=0.0)


def case(nx: int = 200, ny: int = 200, nz: int = 50, seed: int = 7, scale: float = 1.0):
    """``(b, xb_u, term)``: :class:`RegionalB`, ``U``'s background and the flat observation term."""
    b = bd.setup_be_regional(be_dat(nx, ny, nz, seed), ix=nx, jy=ny, kz=nz, ds=12000.0)
    xb = background(nx, ny, nz, seed)
    term = FlatObsTerm(innovations(nx, ny, nz, seed, scale), xb)
    return b, {k: xb[k] for k in U_KEYS}, term
