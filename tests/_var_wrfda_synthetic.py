"""A small synthetic first guess and WRFDA-shaped observation types for the
woof.da.var innovation and operator tests (no oracle data needed)."""

from __future__ import annotations

import numpy as np

from woof.da.var import obs_wrfda as ow

NX, NY, NZ = 8, 7, 40


def background(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    terr = 100.0 + 20.0 * ii + 15.0 * jj
    hf = terr[:, :, None] + np.concatenate([[0.0], np.cumsum(np.linspace(40.0, 900.0, NZ))])[None, None, :]
    h = 0.5 * (hf[:, :, :-1] + hf[:, :, 1:])
    p = 101000.0 * np.exp(-h / 8000.0)
    t = 290.0 - 0.0065 * h + 0.3 * rng.standard_normal((NX, NY, NZ))
    q = 0.012 * np.exp(-h / 2200.0)
    rho = p / (287.0 * t)
    psfc = 101000.0 * np.exp(-terr / 8000.0)
    tpw = np.zeros((NX, NY))
    for k in range(NZ):
        tpw = tpw + (hf[:, :, k + 1] - hf[:, :, k]) * q[:, :, k] * rho[:, :, k]
    tpw = 0.1 * tpw
    coeff = (3.73e-3 * 1.0e8) / 77.6
    ref = 0.776 * p / t * (1.0 + coeff * q / (t * (0.622 + 0.378 * q)))
    return dict(
        u=5.0 + rng.standard_normal((NX, NY, NZ)), v=-2.0 + rng.standard_normal((NX, NY, NZ)),
        t=t, q=q, p=p, h=h, hf=hf, rho=rho, ref=ref, reflog=np.log(ref), psfc=psfc, terr=terr,
        rough=np.full((NX, NY), 0.1), tpw=tpw, landmask=np.zeros((NX, NY)), ptop=5000.0)


def obs_type(name: str, reports: list[dict]) -> ow.ObsType:
    """``reports``: dicts with ``i, j`` (1-based), ``dx, dy``, ``elv`` and
    per-level lists ``h``/``p`` and ``fields`` {var: [(value, qc, error)...]}."""
    info = {k: [] for k in ("i", "j", "dx", "dy", "dxm", "dym", "elv", "lat", "levels",
                            "proc_domain", "name", "id")}
    coords = {k: [] for k in ow._COORDS[name]}
    fields = {v: ([], [], []) for v in ow._FIELDS[name]}
    for r in reports:
        nl = len(next(iter(r["fields"].values())))
        info["i"].append(r["i"]); info["j"].append(r["j"])
        info["dx"].append(r["dx"]); info["dy"].append(r["dy"])
        info["dxm"].append(1.0 - r["dx"]); info["dym"].append(1.0 - r["dy"])
        info["elv"].append(r.get("elv", 0.0)); info["lat"].append(r.get("lat", 40.0))
        info["levels"].append(nl); info["proc_domain"].append(True)
        info["name"].append(r.get("name", "X")); info["id"].append(r.get("id", "X"))
        for c in coords:
            coords[c] += list(r[c])
        for v in fields:
            vals = r["fields"].get(v, [(ow.MISSING_R, ow.MISSING_DATA, 1.0)] * nl)
            for a, b in zip(fields[v], zip(*vals)):
                a += list(b)
    info = {k: np.asarray(v) for k, v in info.items()}
    info["levels"] = info["levels"].astype(np.int64)
    flds = {v: ow.Field(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.int64),
                        np.asarray(c, dtype=np.float64)) for v, (a, b, c) in fields.items()}
    return ow.ObsType(name, info, {c: np.asarray(v, dtype=np.float64) for c, v in coords.items()}, flds)


def obs(types: dict) -> ow.WrfdaObs:
    return ow.WrfdaObs(types=types, ntotal={k: v.nlocal for k, v in types.items()}, analysis_minutes=0.0)
