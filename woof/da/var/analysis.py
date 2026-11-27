"""The analysis step of WRFDA's 3D-Var: first guess in, WRF analysis out
(docs/dev/WRFDA-3DVAR-PLAN.md, sections 4.1-4.3; plan item V1.5).

- :func:`read_first_guess` reads a WRF ``wrfinput``/``wrfout`` (WRFDA's
  ``fg``) into Fortran index order ``(i, j, k)``, binary64.
- :func:`wrftoxb` is ``da_transfer_wrftoxb`` (regional ARW, C grid,
  ``da_setup_firstguess``'s ``coefx``/``coefy``): the background ``xb`` the
  innovations, ``U`` and the analysis transfer read, and the WRF state
  ``grid`` with WRFDA's recomputed base state (``mub``, ``phb``, ``pb``).
- :func:`xa_from_cv` is ``da_transform_vtox`` then ``da_transform_xtoxa``.
- :func:`xatowrf` is ``da_transfer_xatowrf`` (no ``A2C``, ``var4d`` off,
  ``cloud_cv_options = 0``): the increment to WRF's prognostics. Moisture
  first (specific humidity to mixing ratio), then the dry column mass
  ``mu``, the perturbation pressure (Guo's tangent-linear ``p``), the
  optional ``check_rh``, ``theta`` and the hydrostatic ``ph``
  (``hypsometric_opt`` 1 or 2), and the A- to C-grid winds with WRFDA's
  linear extrapolation at the boundaries.
- :func:`analyse` runs the whole thing: observations, innovations, ``VarCost``,
  ``da_minimise_cg`` (outer loops), the transfer, and returns the analysed
  WRF fields and WRFDA's log.
- :func:`write_bundle` writes the analysed fields as a ``wrf-analysis-bundle/1``
  through :mod:`woof.io.analysis_bundle`, the format ``woof run --analysis``
  reads.

The arithmetic is element by element in WRFDA's statement order on the host,
binary64; the column sums (``sdmd``, ``s1md``, ``tpw``) and the vertical
recurrences (``p``, ``ph``) run level by level in WRFDA's order.  On the
tutorial :func:`wrftoxb` is the oracle's ``xb`` to a few ulps (9e-14
relative at most, in ``p`` and what derives from it), on glibc and macOS
alike; per-element glibc ``pow``/``log``/``exp`` give the same words, so
the residual is not the maths library (docs/var.md).

Derived from WRFDA 4.7.1 (``var/da/da_transfer_model/da_transfer_wrftoxb.inc``,
``da_transfer_xatowrf.inc``, ``da_transfer_xatoanalysis.inc``,
``var/da/da_setup_structures/da_setup_firstguess.inc``,
``var/da/da_physics/da_tp_to_qs.inc``, ``da_tpq_to_rh.inc``,
``da_integrat_dz.inc``, ``da_transform_xtogpsref.inc``,
``da_roughness_from_lanu.inc``, ``da_check_rh_simple.inc``,
``var/da/da_main/da_solve.inc``), public domain; see
``licenses/LICENSE-WRF-public-domain.txt``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .obsop_wrfda import COEFF

__all__ = [
    "FirstGuess",
    "VarResult",
    "analyse",
    "increment_stats",
    "obs_window",
    "read_first_guess",
    "restamp_start",
    "roughness_from_lanu",
    "wrftoxb",
    "write_analysis_file",
    "write_bundle",
    "xa_from_cv",
    "xatowrf",
    "WRF_FIELDS",
]

# da_control.f90 (binary64 under WRFDA's -fdefault-real-8)
GAS_CONSTANT = 287.0
GAS_CONSTANT_V = 461.6
CP = 7.0 * GAS_CONSTANT / 2.0
KAPPA = GAS_CONSTANT / CP
RD_OVER_RV = GAS_CONSTANT / GAS_CONSTANT_V
RD_OVER_RV1 = 1.0 - RD_OVER_RV
GRAVITY = 9.81
T0 = 300.0
T_KELVIN = 273.15
ES_ALPHA, ES_BETA, ES_GAMMA = 611.2, 17.67, 243.5
QLIMIT = 1.0e-12

#: The WRF fields :func:`xatowrf` changes (``da_update_firstguess`` writes them).
WRF_FIELDS = ("U", "V", "T", "QVAPOR", "MU", "P", "PH", "PSFC")

_FG_3D = ("U", "V", "W", "PH", "PHB", "T", "P", "PB", "QVAPOR")
_FG_2D = ("MU", "MUB", "PSFC", "HGT", "MAPFAC_M", "F", "XLAT", "XLONG", "LU_INDEX", "LANDMASK", "XLAND",
          "SST", "TMN", "TSK", "SNOWC")
_FG_1D = ("ZNU", "ZNW", "DN", "DNW", "RDNW", "RDN", "C1H", "C2H", "C3H", "C4H", "C1F", "C2F", "C3F", "C4F")
_FG_0D = ("P_TOP", "P00", "T00", "TLP", "TISO", "P_STRAT", "TLP_STRAT")


# --------------------------------------------------------------------------- the first guess


@dataclass
class FirstGuess:
    """One WRF file's first record in Fortran index order, binary64."""

    path: Path
    fields: dict
    attrs: dict
    times: str
    #: 4D-Var outer loops: the boundary-table perturbation of the loop that
    #: made this analysis (``WindowModel.boundary_increment``, its tables'
    #: names and layout), which WRFDA adds to the tables the next loop's
    #: nonlinear run reads (da_transfer_xatowrf.inc:665-727).  None
    #: elsewhere.
    lbc_increment: dict | None = None

    def __getitem__(self, k):
        return self.fields[k]


def read_first_guess(path) -> FirstGuess:
    """The fields ``da_transfer_wrftoxb`` reads, first record, ``(i, j, k)`` binary64."""
    import netCDF4

    path = Path(path)
    out = {}
    with netCDF4.Dataset(path) as nc:
        nc.set_auto_mask(False)
        attrs = {k: nc.getncattr(k) for k in nc.ncattrs()}
        for names, rank in ((_FG_3D, 3), (_FG_2D, 2), (_FG_1D, 1), (_FG_0D, 0)):
            for n in names:
                if n not in nc.variables:
                    continue
                a = np.asarray(nc.variables[n][0], dtype=np.float64)
                out[n] = a.transpose() if rank >= 2 else (a if rank == 1 else float(a))
        times = b"".join(nc.variables["Times"][0]).decode() if "Times" in nc.variables else attrs["START_DATE"]
    missing = [n for n in _FG_3D + ("MU", "MUB", "PSFC", "HGT", "MAPFAC_M", "ZNU", "ZNW", "DNW", "RDNW", "RDN",
                                     "P_TOP") if n not in out]
    if missing:
        raise ValueError(f"{path.name}: not a WRF first guess, it lacks {missing}")
    return FirstGuess(path=path, fields=out, attrs=attrs, times=times)


def roughness_from_lanu(lanu: np.ndarray, mminlu: str, date: str, table=None) -> np.ndarray:
    """``da_roughness_from_lanu``: ``SFZ0 / 100`` of the land-use category, summer or winter.

    ``table`` is a ``LANDUSE.TBL`` path (default woof's packaged copy, which
    is WRF's). WRFDA reads ``SFZ0`` as ``real(kind=4)``.
    """
    from woof.core.landuse import LANDUSE_COLUMNS, load_landuse_table

    if table is not None and Path(table).name != "LANDUSE.TBL":
        raise ValueError(f"{table}: the land-use table must be named LANDUSE.TBL")
    tbl = load_landuse_table(mminlu.strip(), None if table is None else Path(table).parent)
    col = LANDUSE_COLUMNS.index("z0_cm")
    y, m, d = int(date[0:4]), int(date[5:7]), int(date[8:10])
    julday = _dt.date(y, m, d).timetuple().tm_yday
    isn = 1 if 105 <= julday <= 288 else 2
    sfz0 = np.asarray(tbl.values[isn - 1, :, col], dtype=np.float32).astype(np.float64)
    li = (np.asarray(lanu, dtype=np.float64) + 0.001).astype(np.int64)
    return sfz0[li - 1] / 100.0


def _coefficients(fg: FirstGuess) -> dict:
    """``c1f..c4h`` as ``da_solve`` sets them (``hybrid_opt <= 0``: the terrain-following eta)."""
    f = fg.fields
    hyb = int(fg.attrs.get("HYBRID_OPT", 0))
    if hyb <= 0 or "C3F" not in f:
        nz = f["ZNU"].size
        return {"c3f": f["ZNW"].copy(), "c3h": f["ZNU"].copy(), "c4f": np.zeros(nz + 1), "c4h": np.zeros(nz),
                "c1f": np.ones(nz + 1), "c1h": np.ones(nz), "c2f": np.zeros(nz + 1), "c2h": np.zeros(nz)}
    return {k.lower(): f[k].copy() for k in ("C1F", "C2F", "C3F", "C4F", "C1H", "C2H", "C3H", "C4H")}


def wrftoxb(fg: FirstGuess, *, hypsometric_opt: int | None = None, landuse_table=None) -> tuple[dict, dict]:
    """``da_transfer_wrftoxb`` for a regional ARW first guess: returns ``(xb, grid)``.

    ``xb`` has the keys :func:`woof.da.var.innov_wrfda.innovations`,
    :class:`~woof.da.var.transforms.DeviceU` and :func:`xatowrf` read;
    ``grid`` is the WRF state WRFDA holds after it (the recomputed base
    state, ``QVAPOR`` floored at ``qlimit``) that :func:`xatowrf` updates.
    """
    f = fg.fields
    if hypsometric_opt is None:
        hypsometric_opt = int(fg.attrs.get("HYPSOMETRIC_OPT", 2))
    if hypsometric_opt not in (1, 2):
        raise NotImplementedError(f"hypsometric_opt = {hypsometric_opt}")
    c = _coefficients(fg)
    nx, ny, nz = f["T"].shape
    ptop = f["P_TOP"]
    base_pres, base_temp, base_lapse = f["P00"], f["T00"], f["TLP"]
    iso_temp = f["TISO"]
    base_pres_strat, base_lapse_strat = f.get("P_STRAT", 0.0), f.get("TLP_STRAT", 0.0)
    if base_temp < 100.0 or base_pres < 10000.0:
        raise ValueError("did not find base state parameters in the first guess (P00, T00)")
    ht = f["HGT"]
    mu_2, ph_2, t_2 = f["MU"], f["PH"], f["T"]
    qv = f["QVAPOR"].copy()
    cvpm = -(1.0 - GAS_CONSTANT / CP)
    cpovcv = CP / (CP - GAS_CONSTANT)
    no_ppb = float(np.sum(f["PB"] * f["PB"])) == 0.0
    if no_ppb:
        raise NotImplementedError("a first guess without PB (WRFDA's no_ppb branch)")
    c3h, c4h, c3f, c4f = c["c3h"], c["c4h"], c["c3f"], c["c4f"]
    c1h, c2h = c["c1h"], c["c2h"]
    # base state (the else branch: pb, t_init, alb, mub, phb recomputed)
    p_surf = base_pres * np.exp(-base_temp / base_lapse + np.sqrt(
        (base_temp / base_lapse) * (base_temp / base_lapse) - 2.0 * GRAVITY * ht / base_lapse / GAS_CONSTANT))
    pb = np.empty((nx, ny, nz))
    alb = np.empty((nx, ny, nz))
    for k in range(nz):
        pbk = c3h[k] * (p_surf - ptop) + c4h[k] + ptop
        temp = np.maximum(iso_temp, base_temp + base_lapse * np.log(pbk / base_pres))
        if base_pres_strat > 0.0:
            strat = pbk < base_pres_strat
            temp = np.where(strat, iso_temp + base_lapse_strat * np.log(pbk / np.where(strat, base_pres_strat, 1.0)),
                            temp)
        t_init = temp * (base_pres / pbk) ** (GAS_CONSTANT / CP) - T0
        alb[..., k] = (GAS_CONSTANT / base_pres) * (t_init + T0) * (pbk / base_pres) ** cvpm
        pb[..., k] = pbk
    mub = p_surf - ptop
    phb = np.empty((nx, ny, nz + 1))
    phb[..., 0] = ht * GRAVITY
    for k in range(nz):
        if hypsometric_opt == 1:
            phb[..., k + 1] = phb[..., k] - f["DNW"][k] * (c1h[k] * mub + c2h[k]) * alb[..., k]
        else:
            pfu = mub * c3f[k + 1] + c4f[k + 1] + ptop
            pfd = mub * c3f[k] + c4f[k] + ptop
            phm = mub * c3h[k] + c4h[k] + ptop
            phb[..., k + 1] = phb[..., k] + alb[..., k] * phm * np.log(pfd / pfu)
    mu = mub + mu_2
    al = np.empty((nx, ny, nz))
    for k in range(nz):
        if hypsometric_opt == 1:
            al[..., k] = (-1.0 / (c1h[k] * mu + c2h[k])
                          * (alb[..., k] * (c1h[k] * mu_2) + f["RDNW"][k] * (ph_2[..., k + 1] - ph_2[..., k])))
        else:
            pfu = mu * c3f[k + 1] + c4f[k + 1] + ptop
            pfd = mu * c3f[k] + c4f[k] + ptop
            phm = mu * c3h[k] + c4h[k] + ptop
            al[..., k] = ((ph_2[..., k + 1] - ph_2[..., k] + phb[..., k + 1] - phb[..., k]) / phm
                          / np.log(pfd / pfu) - alb[..., k])
    rho = 1.0 / (alb + al)
    qvf1 = 1.0 + qv / RD_OVER_RV
    p = base_pres * ((GAS_CONSTANT * (T0 + t_2) * qvf1) / (base_pres * (al + alb))) ** cpovcv
    hf = (phb + ph_2) / GRAVITY
    u = 0.5 * (f["U"][:-1] + f["U"][1:])
    v = 0.5 * (f["V"][:, :-1] + f["V"][:, 1:])
    h = np.empty((nx, ny, nz))
    for k in range(nz):
        if hypsometric_opt == 1:
            h[..., k] = 0.5 * (hf[..., k] + hf[..., k + 1])
        else:
            pf_k = mu * c3f[k] + c4f[k] + ptop
            pf_k1 = mu * c3f[k + 1] + c4f[k + 1] + ptop
            pp_k = mu * c3h[k] + c4h[k] + ptop
            h[..., k] = hf[..., k] + (hf[..., k + 1] - hf[..., k]) * np.log(pf_k / pp_k) / np.log(pf_k / pf_k1)
    qv = np.maximum(qv, QLIMIT)  # num_pseudo == 0
    t = (T0 + t_2) * (p / base_pres) ** KAPPA
    q = qv / (1.0 + qv)
    psac = mub + mu_2
    ds = float(fg.attrs["DX"])
    map_factor = f["MAPFAC_M"]
    tmp = ds / map_factor
    grid_box_area = tmp * tmp
    # da_tpq_to_rh
    t_c = t - T_KELVIN
    es = ES_ALPHA * np.exp(ES_BETA * t_c / (t_c + ES_GAMMA))
    qs = RD_OVER_RV * es / (p - RD_OVER_RV1 * es)
    # da_integrat_dz
    pw = np.zeros((nx, ny))
    for k in range(nz):
        pw = pw + (hf[..., k + 1] - hf[..., k]) * q[..., k] * rho[..., k]
    tpw = 0.1 * pw
    # da_transform_xtogpsref
    partone = 0.776 * p / t
    dividnd = t * (0.622 + 0.378 * q)
    ref = partone * (1.0 + COEFF * q / dividnd)
    mminlu = str(fg.attrs.get("MMINLU", "USGS"))
    rough = roughness_from_lanu(f["LU_INDEX"], mminlu, fg.times, landuse_table)
    coefx = 0.5 * map_factor / ds
    xb = {
        "u": u, "v": v, "w": f["W"].copy(), "t": t, "p": p, "q": q, "h": h, "hf": hf, "rho": rho, "qs": qs,
        "es": es, "ref": ref, "reflog": np.log(ref), "tpw": tpw, "psac": psac, "psfc": f["PSFC"].copy(),
        "terr": ht.copy(), "lat": f.get("XLAT"), "lon": f.get("XLONG"), "map_factor": map_factor.copy(),
        "cori": f.get("F"), "landmask": f.get("LANDMASK"), "xland": f.get("XLAND"), "lanu": f["LU_INDEX"].copy(),
        "rough": rough, "grid_box_area": grid_box_area, "coefx": coefx, "coefy": coefx.copy(),
        "znu": f["ZNU"].copy(), "znw": f["ZNW"].copy(), "dn": f.get("DN"), "dnw": f["DNW"].copy(),
        "sigmah": f["ZNU"].copy(), "sigmaf": f["ZNW"].copy(), "ptop": ptop, "ds": ds, "ztop": float(hf[..., nz].max()),
        **{k: c[k].copy() for k in ("c1h", "c2h", "c1f", "c2f")},
    }
    xb = {k: v for k, v in xb.items() if v is not None}
    grid = {
        "u_2": f["U"].copy(), "v_2": f["V"].copy(), "w_2": f["W"].copy(), "t_2": t_2.copy(), "ph_2": ph_2.copy(),
        "mu_2": mu_2.copy(), "p": f["P"].copy(), "psfc": f["PSFC"].copy(), "qv": qv, "mub": mub, "phb": phb,
        "pb": pb, "ht": ht.copy(), "rdnw": f["RDNW"].copy(), "rdn": f["RDN"].copy(), "dnw": f["DNW"].copy(),
        "p_top": ptop, "base_pres": base_pres, "hypsometric_opt": hypsometric_opt, **c,
    }
    return xb, grid


# --------------------------------------------------------------------------- x_a


def xa_from_cv(b, xb: dict, cv) -> dict:
    """``da_transform_vtox`` and ``da_transform_xtoxa`` in binary64 (the reference ``U``)."""
    from . import reference as ref

    return ref.transform_u(np.asarray(cv, dtype=np.float64), b, xb)["xa"]


def _tp_to_qs(t, p):
    t_c = t - T_KELVIN
    es = ES_ALPHA * np.exp(ES_BETA * t_c / (t_c + ES_GAMMA))
    return RD_OVER_RV * es / (p - RD_OVER_RV1 * es)


def check_rh_simple(xa_q, xb: dict, xa_p, xa_t):
    """``da_check_rh_simple``: ``xa%q`` within ``[1e-6 qs, qs] - xb%q``, zero above the tropopause."""
    nx, ny, nz = xb["t"].shape
    t_tropp = np.full((nx, ny), 999.0)
    k_tropp = np.full((nx, ny), nz - 1)
    for k in range(nz - 1, -1, -1):
        sel = (xb["t"][..., k] < t_tropp) & (xb["p"][..., k] > 5000.0)
        t_tropp = np.where(sel, xb["t"][..., k], t_tropp)
        k_tropp = np.where(sel, k, k_tropp)
    out = xa_q.copy()
    for k in range(nz):
        q_new = _tp_to_qs(xb["t"][..., k] + xa_t[..., k], xb["p"][..., k] + xa_p[..., k])
        dqmax = q_new * 1.0 - xb["q"][..., k]
        dqmin = q_new * 1.0e-6 - xb["q"][..., k]
        lim = np.minimum(np.maximum(out[..., k], dqmin), dqmax)
        out[..., k] = np.where(k > k_tropp, 0.0, lim)
    return out


def xatowrf(grid: dict, xb: dict, xa: dict, *, check_rh: int = 0, var4d: bool = False) -> dict:
    """``da_transfer_xatowrf``: the analysed WRF fields (Fortran order, binary64).

    With ``var4d`` WRFDA's 4D-Var branch: no pressure increment
    (``xa%p = 0``; the boundary tendencies it also adds are not ported).
    Returns ``U V T QVAPOR MU P PH PSFC`` (the full WRF variables, ``T`` and
    ``PH`` as perturbations as in the file) and the internal ``xa%p`` it
    recomputes (``"xa_p"``), ``mu_cgrid``, ``q_cgrid``.
    """
    if check_rh not in (0, 1):
        raise NotImplementedError(f"check_rh = {check_rh} (only 0 and 1, simple)")
    nx, ny, nz = xb["t"].shape
    qv = grid["qv"]
    dnw, rdnw, rdn = grid["dnw"], grid["rdnw"], grid["rdn"]
    c1f, c3f, c4f, c3h, c4h, c1h, c2h = (grid[k] for k in ("c1f", "c3f", "c4f", "c3h", "c4h", "c1h", "c2h"))
    mub, mu_2, p_top = grid["mub"], grid["mu_2"], grid["p_top"]
    base_pres = grid["base_pres"]
    xa_q = np.asarray(xa["q"], dtype=np.float64)
    xa_t = np.asarray(xa["t"], dtype=np.float64)
    xa_psfc = np.asarray(xa["psfc"], dtype=np.float64)
    xbq = xb["q"]
    ph_old = grid["ph_2"].copy()
    # [1.0] mixing ratio
    one = (1.0 - xbq) * (1.0 - xbq)
    q_cgrid = np.where(xbq + xa_q < 0.0, -xbq / one, xa_q / one)
    # [2.0] dry column mass
    sdmd = np.zeros((nx, ny))
    s1md = np.zeros((nx, ny))
    for k in range(nz):
        sdmd = sdmd + q_cgrid[..., k] * dnw[k]
        s1md = s1md + (1.0 + qv[..., k]) * dnw[k]
    mu_cgrid = -(xa_psfc + xb["psac"] * sdmd) / s1md
    # [3.0] pressure (Guo's TL of real.exe's p)
    xa_p = np.zeros((nx, ny, nz))
    if not var4d:
        for k in range(nz - 1, -1, -1):
            kk = k if k == nz - 1 else k + 1
            qvf1 = 0.5 * (q_cgrid[..., k] + q_cgrid[..., kk])
            qvf1_b = 0.5 * (qv[..., k] + qv[..., kk])
            qvf2 = -qvf1 / ((1.0 + qvf1_b) * (1.0 + qvf1_b))
            qvf2_b = 1.0 / (1.0 + qvf1_b)
            qvf1 = qvf1 * qvf2_b + qvf1_b * qvf2
            qvf1_b = qvf1_b * qvf2_b
            term = ((c1f[k] * mu_cgrid + qvf1 * c1f[k] * mub) / qvf2_b
                    - (c1f[k] * mu_2 + qvf1_b * c1f[k] * mub) * qvf2 / (qvf2_b * qvf2_b))
            if k == nz - 1:
                xa_p[..., k] = (-0.5 / rdnw[k]) * term
            else:
                xa_p[..., k] = xa_p[..., k + 1] - (1.0 / rdn[k + 1]) * term
    p_new = grid["p"] + xa_p
    if check_rh == 1:
        xa_q = check_rh_simple(xa_q, xb, xa_p, xa_t)
    q_cgrid = xa_q / ((1.0 - xbq) * (1.0 - xbq))
    # [4.0] theta and geopotential
    t_2 = np.empty((nx, ny, nz))
    ph_2 = grid["ph_2"].copy()
    ht = grid["ht"]
    if grid["hypsometric_opt"] == 1:
        ph_full = ht * GRAVITY
        ph_xb_hd = ht * GRAVITY
        for k in range(nz):
            t_full = xa_t[..., k] + xb["t"][..., k]
            p_full = xa_p[..., k] + xb["p"][..., k]
            q_full = qv[..., k] + q_cgrid[..., k]
            rho_dry = p_full / (GAS_CONSTANT * t_full * (1.0 + q_full / RD_OVER_RV))
            t_2[..., k] = t_full * (base_pres / p_full) ** KAPPA - T0
            ph_full = ph_full - xb["dnw"][k] * ((c1h[k] * xb["psac"] + c2h[k]) + c1h[k] * mu_cgrid) / rho_dry
            ph_xb_hd = ph_xb_hd - xb["dnw"][k] * (c1h[k] * xb["psac"] + c2h[k]) / xb["rho"][..., k]
            ph_2[..., k + 1] = ph_full - grid["phb"][..., k + 1] + (xb["hf"][..., k + 1] * GRAVITY - ph_xb_hd)
    else:
        cvpm = -(1.0 - GAS_CONSTANT / CP)
        mu_full = mub + mu_2 + mu_cgrid
        ph = ht * GRAVITY
        for k in range(nz):
            p_full = xb["p"][..., k] + xa_p[..., k]
            t_full = xb["t"][..., k] + xa_t[..., k]
            q_full = qv[..., k] + q_cgrid[..., k]
            theta = t_full * (base_pres / p_full) ** KAPPA
            t_2[..., k] = theta - T0
            thetam = theta * (1.0 + q_full / RD_OVER_RV)
            ald = (GAS_CONSTANT / base_pres) * thetam * (p_full / base_pres) ** cvpm
            pfu = mu_full * c3f[k + 1] + c4f[k + 1] + p_top
            pfd = mu_full * c3f[k] + c4f[k] + p_top
            phm = mu_full * c3h[k] + c4h[k] + p_top
            ph = ph + ald * phm * np.log(pfd / pfu)
            ph_2[..., k + 1] = ph - grid["phb"][..., k + 1]
    # A grid to C grid, with WRFDA's boundary extrapolation
    ua = np.zeros((nx + 2, ny + 1, nz))  # i = -1..nx, j = 0..ny (j = ny is the unset halo, unused)
    va = np.zeros((nx + 1, ny + 2, nz))
    ua[1:nx + 1, :ny] = xa["u"]
    va[:nx, 1:ny + 1] = xa["v"]
    ua[0, :ny] = 2.0 * ua[1, :ny] - ua[2, :ny]
    ua[nx + 1, :ny] = 2.0 * ua[nx, :ny] - ua[nx - 1, :ny]
    va[:nx, 0] = 2.0 * va[:nx, 1] - va[:nx, 2]
    va[:nx, ny + 1] = 2.0 * va[:nx, ny] - va[:nx, ny - 1]
    u_cgrid = 0.5 * (ua[0:nx + 1, :ny] + ua[1:nx + 2, :ny])  # (nx+1, ny, nz)
    v_cgrid = 0.5 * (va[:nx, 0:ny + 1] + va[:nx, 1:ny + 2])  # (nx, ny+1, nz)
    # [5.0] add to the first guess
    qv_new = np.maximum(qv + q_cgrid, QLIMIT)
    return {
        "U": grid["u_2"] + u_cgrid,
        "V": grid["v_2"] + v_cgrid,
        "T": t_2,
        "QVAPOR": qv_new,
        "MU": mu_2 + mu_cgrid,
        "P": p_new,
        "PH": ph_2,
        "PSFC": grid["psfc"] + xa_psfc,
        "xa_p": xa_p,
        "mu_cgrid": mu_cgrid,
        "q_cgrid": q_cgrid,
        "ph_cgrid": ph_2 - ph_old,
    }


# --------------------------------------------------------------------------- the whole step

#: The background fields ``U`` reads.
_XBU = ("grid_box_area", "coefx", "coefy", "qs", "q", "psac", "dnw", "rho", "p", "t", "c1h", "c2h")


@dataclass
class VarResult:
    """One analysis: per outer loop the CG results, the control vector, ``xa`` and the WRF fields."""

    log: str
    cg: list
    cvt: np.ndarray
    xa: dict
    wrf: dict
    xb: dict
    grid: dict
    nobs: int
    timings: dict = field(default_factory=dict)
    #: The radiance door's report (counts per sensor, Jo, VarBC), or None.
    radiance: dict | None = None
    #: With radiances: O-B and O-A (linearised, ``d - H' U cvt``) per term over the used observations.
    fit: dict = field(default_factory=dict)


def obs_window(fg: FirstGuess, minutes: float = 60.0) -> dict:
    """The observation window around the first guess's time (WRFDA's ``time_window_min/max``)."""
    t = _dt.datetime.strptime(fg.times[:19], "%Y-%m-%d_%H:%M:%S")
    fmt = "%Y-%m-%d_%H:%M:%S"
    return {"analysis_date": t.strftime(fmt),
            "time_window_min": (t - _dt.timedelta(minutes=minutes)).strftime(fmt),
            "time_window_max": (t + _dt.timedelta(minutes=minutes)).strftime(fmt)}


def analyse(be, obs, first_guess, *, tier: str = "reference", ntmax=50, eps=0.01, max_ext_its: int = 1,
            orthonorm_gradient: bool = True, precondition_cg: bool = False, check_rh: int = 0,
            window_minutes: float = 60.0, landuse_table=None, xb_override: dict | None = None,
            write=None, radiances=None, crtm=None, radiance_qc=None, varbc=None, report_fit: bool | None = None,
            ensemble=None, hybrid=None, vertloc=None, fgat=None,
            time_window: tuple[str, str] | None = None, var4d=None) -> VarResult:
    """3D-Var from files: ``be.dat``, OBSPROC's ``ob.ascii`` and a WRF first guess.

    ``tier`` is ``reference`` (NumPy binary64), ``oracle`` (device binary64,
    WRFDA's serial dot) or ``production`` (device binary32, fixed-tree dot).
    ``write`` receives each line of WRFDA's minimiser log.  ``xb_override``
    replaces the background fields :func:`wrftoxb` derives (the oracle's own
    ``xb`` dump, for the oracle tests).  With one outer loop the innovations
    are the nonlinear ones at ``x_b``; later loops relinearise ``H'`` about
    the previous loop's analysis (the linear update ``d - H' U xhat``).

    Radiances (opt-in, plan section 13): ``radiances`` is a list of
    :class:`~woof.da.var.radiance.RadianceObs`; ``crtm`` an open
    :class:`woof.rtm.crtm.ClearSkyOperator` with their sensors (one is opened
    from ``WOOF_CRTM_LIB``/``WOOF_CRTM_COEFFS`` when None); ``radiance_qc``
    a :class:`~woof.da.var.radiance.RadianceQC`; ``varbc`` a
    :class:`~woof.da.var.varbc.VarBC` whose coefficients are updated in
    place.  Without ``radiances`` nothing here changes.  ``report_fit``
    (default: with radiances) fills :attr:`VarResult.fit` at the cost of one more ``H' U``.

    Hybrid 3DEnVar (opt-in, plan V3): ``ensemble`` is a
    :class:`~woof.da.var.hybrid.Ensemble` (e.g. from
    :func:`~woof.da.var.hybrid.ep_from_wrf` or ``read_ep``) on the first
    guess's grid and ``hybrid`` its :class:`~woof.da.var.hybrid.HybridConfig`
    (``je_factor`` given); ``vertloc`` WRFDA's ``be.vertloc.dat`` contents
    for ``alpha_vertloc_opt = 1`` (option 2 computes it).  The alpha control
    variable follows the static one; the analysis is ``U_s v_s + sum_n alpha_n
    o x'_n`` in binary64.

    FGAT (opt-in, plan V5.1; WRFDA ``num_fgat_time > 1``): ``fgat`` is the
    first guesses at the slot times (``fg01`` .. ``fgNN``: the window's
    start, evenly spaced, its end) and ``obs`` the matching ``ob01.ascii``
    .. ``obNN.ascii``.  Each slot's innovations, QC and vertical positions
    are computed against its own first guess
    (:mod:`woof.da.var.fgat`); ``H'``, the increment and the analysis stay
    at ``first_guess``'s time, which must be one of the slot times.  One
    outer loop; no radiances, no hybrid.

    4D-Var (opt-in, EXPERIMENTAL; plan docs/dev/WRFDA-4DVAR-PLAN.md, WRFDA
    ``var4d``): ``var4d`` is a :class:`~woof.da.var.fourd.Var4D` (the
    tangent-linear model, the identity by default) and needs ``fgat``: the
    trajectory's states at the slot times, with ``first_guess`` at the
    window's start.  Each slot's ``H'`` is linearised about its own state
    and applied to the TL model's perturbation there
    (:class:`~woof.da.var.fourd.FourDObsTerm`); the analysis is
    ``first_guess`` plus the increment, with no pressure increment
    (``da_transfer_xatowrf``'s ``var4d`` branch).  The lateral boundary
    perturbation is not ported.  FGAT's refusals apply.  4D-Var's own
    refusals and where each is enforced are listed in :func:`_var4d_refusals`;
    the ``reference`` tier only.  Outer loops (``max_ext_its > 1``) need
    ``var4d.nl_model``: each loop after the first reruns it from the
    previous loop's analysis, relinearises ``var4d.model`` about the run
    and recomputes every slot's innovations, as WRFDA's ``da_solve`` does
    (:mod:`woof.da.var.outer4d`); ``Jb`` stays measured from the first
    guess, :attr:`VarResult.cg` holds each loop's log and ``J``, and
    :attr:`VarResult.xa` is the last loop's increment (WRFDA's ``xa``).

    ``time_window`` is WRFDA's ``(time_window_min, time_window_max)``; by
    default the first guess's time plus and minus ``window_minutes``.
    """
    from . import be_dat as bd
    from . import innov_wrfda as iw
    from . import obs_wrfda as ow
    from .cost import VarCost
    from .minimise import minimise
    from .obsop_wrfda import FlatObsTerm

    if var4d is not None and fgat is None:
        raise ValueError("var4d needs the trajectory: fgat=[fg01, ..., fgNN] at the slot times")
    if fgat is not None:   # every FGAT refusal, before anything is read
        _fgat_refusals(obs, fgat, first_guess, window_minutes=window_minutes, time_window=time_window,
                       max_ext_its=max_ext_its, radiances=radiances, xb_override=xb_override,
                       ensemble=ensemble, hybrid=hybrid, var4d=var4d, tier=tier)
    timings = {}
    t0 = time.perf_counter()
    fg = first_guess if isinstance(first_guess, FirstGuess) else read_first_guess(first_guess)
    xb, grid = wrftoxb(fg, landuse_table=landuse_table)
    if xb_override is not None:
        xb = dict(xb, **xb_override)
    nx, ny, nz = xb["t"].shape
    b = bd.setup_be_regional(be if not isinstance(be, (str, Path)) else bd.read_be_dat(be),
                             ix=nx, jy=ny, kz=nz, ds=float(xb["ds"]))
    proj = ow.LambertWRFDA.from_attributes(fg.attrs)
    window = obs_window(fg, window_minutes)
    if time_window is not None:
        window.update(time_window_min=time_window[0], time_window_max=time_window[1])
    if var4d is not None:
        return _analyse_var4d(b, fg, xb, grid, obs, fgat, proj, window, var4d, max_ext_its=max_ext_its,
                              ntmax=ntmax, eps=eps, check_rh=check_rh, landuse_table=landuse_table, write=write,
                              timings=timings, t0=t0, orthonorm_gradient=orthonorm_gradient,
                              precondition_cg=precondition_cg, report_fit=bool(report_fit))
    if fgat is None:
        wobs = obs if not isinstance(obs, (str, Path)) else ow.read_obs_ascii(obs, proj, **window)
        iv = iw.innovations(wobs, xb)
    else:
        iv, _ = _fgat_innovations(obs, fgat, xb, proj, window, landuse_table=landuse_table)
    term = FlatObsTerm(iv, xb)
    terms = [term]
    rterm = rreport = None
    if radiances:
        from . import radiance as rad

        rterm, rreport = _radiance_term(rad, radiances, xb, fg, proj, crtm, radiance_qc, varbc)
        terms.append(rterm)
    xbu = {k: xb[k] for k in _XBU}
    hu = None
    if ensemble is not None:
        from . import hybrid as hy

        if hybrid is None:
            raise ValueError("an ensemble needs a HybridConfig (hybrid=...) with je_factor")
        if ensemble.shape != (nx, ny, nz):
            raise ValueError(f"the ensemble is on {ensemble.shape}, the first guess on {(nx, ny, nz)}")
        cost = hy.hybrid_cost(b, dict(xbu, ds=xb["ds"], znw=xb["znw"], ptop=xb["ptop"]), terms, ensemble, hybrid,
                              tier=tier, vertloc=vertloc)
        hu = cost.u if tier == "reference" else hy.HybridReferenceU(b, cost.u.alpha, ensemble, xbu, cost.u.jb_factor)
    elif hybrid is not None:
        raise ValueError("a HybridConfig needs an ensemble")
    else:
        cost = VarCost.reference(b, xbu, terms) if tier == "reference" else VarCost.device(b, xbu, terms, tier)
    timings["setup_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    d0 = [np.array(d, dtype=np.float64) for d in cost.d]
    results, cvt = minimise(cost, max_ext_its=max_ext_its, ntmax=ntmax, eps=eps,
                            orthonorm_gradient=orthonorm_gradient, precondition_cg=precondition_cg, write=write)
    if tier != "reference":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    timings["minimise_s"] = time.perf_counter() - t1
    fit = _fit(cost, d0, cvt, terms) if (rterm is not None if report_fit is None else report_fit) else {}
    cvt = np.asarray(cvt.get() if hasattr(cvt, "get") else cvt, dtype=np.float64)
    if rterm is not None:
        rreport["j_final"] = results[-1].j_final
        if varbc is not None and rterm.control_size:
            varbc.update(rterm, cvt[cost.cv_u:])
            rreport["varbc_controls"] = int(rterm.control_size)
        cvt = cvt[:cost.cv_u]
    t2 = time.perf_counter()
    xa = xa_from_cv(b, xbu, cvt) if hu is None else hu.apply(cvt)["xa"]
    wrf = xatowrf(grid, xb, xa, check_rh=check_rh)
    timings["transfer_s"] = time.perf_counter() - t2
    log = "".join(r.log for r in results)
    return VarResult(log=log, cg=results, cvt=cvt, xa=xa, wrf=wrf, xb=xb, grid=grid, nobs=int(term.mask.sum()),
                     timings=timings, radiance=rreport, fit=fit)


def _analysis_state(start: FirstGuess, wrf: dict) -> FirstGuess:
    """``start`` with :data:`WRF_FIELDS` replaced by the analysis's (WRFDA's
    ``grid`` after ``da_transfer_xatowrf``): what an outer loop's nonlinear
    rerun starts from.  ``path`` stays ``start``'s, whose file does not hold
    this state; :func:`write_analysis_file` writes it
    (``write_analysis_file(state, state.fields, out)``)."""
    fields = dict(start.fields)
    for k in WRF_FIELDS:
        fields[k] = np.array(wrf[k], dtype=np.float64)
    return FirstGuess(path=start.path, fields=fields, attrs=dict(start.attrs), times=start.times)


def _analyse_var4d(b, fg, xb, grid, obs, fgat, proj, window, var4d, *, max_ext_its, ntmax, eps, check_rh,
                   landuse_table, write, timings, t0, report_fit=False, **cg_kw) -> VarResult:
    """:func:`analyse`'s 4D-Var: :func:`woof.da.var.outer4d.outer_loops` with
    WRFDA's pieces (after :func:`_fgat_refusals`).

    Loop 1 is the one-loop 4D-Var: the increment transferred about
    ``first_guess`` (``fg``/``xb``/``grid``), the slots ``fgat``'s.  Loop
    ``it > 1`` reruns ``var4d.nl_model`` from the previous loop's analysis
    (:func:`_analysis_state`), and everything (``U``'s background, the
    transfer about the window's start, each slot's innovations, QC and
    ``H'``, the analysis) is about that run: its slot 1 is the start, as
    WRFDA's var4d ``da_get_innov_vector`` leaves slot 1's state in ``grid``.
    The observations are read once; each loop recomputes the innovations
    and QC from them (WRFDA carries the previous loop's QC flags, see
    docs/var.md).  ``report_fit``: :func:`_fit` of the last loop (its
    innovations, about the last rerun, and its own ``xhat``).
    """
    from . import fgat as fm
    from .cost import VarCost
    from .fourd import FourDObsTerm
    from .outer4d import Trajectory, outer_loops

    paths = [obs] if isinstance(obs, (str, Path)) else list(obs)
    wobs = fm.read_obs_ascii_fgat(paths, proj, **window)
    nslots = len(fgat)
    clock = {"setup1_s": 0.0, "setup_s": 0.0, "transfer_s": 0.0, "nl_model_s": 0.0}
    last: dict = {}

    def setup(it, trajectory):
        t = time.perf_counter()
        if it == 1:
            start, (xb0, grid0) = fg, (xb, grid)
            slots = list(trajectory.slots)
        else:
            slots = [f if isinstance(f, FirstGuess) else read_first_guess(f) for f in trajectory.slots]
            if len(slots) != nslots:
                raise ValueError(f"outer loop {it}: the rerun has {len(slots)} slot states, the window {nslots}")
            try:
                fm.check_fgat(nslots, [fm.first_guess_time(f) for f in slots],
                              time_window_min=window["time_window_min"], time_window_max=window["time_window_max"])
            except ValueError as exc:
                raise ValueError(f"outer loop {it}: the rerun's slots: {exc}") from None
            start = slots[0]
        states = _slot_states(slots, xb, proj, landuse_table=landuse_table,
                              what="FGAT slot" if it == 1 else f"outer loop {it}: slot")
        if it > 1:
            xb0, grid0 = states[0]   # the rerun's start (WRFDA leaves slot 1 in grid)
        iv = fm.innovations_fgat(wobs, [x for x, _ in states])
        if it > 1 and var4d.base_rtol is not None and callable(getattr(var4d.model, "check_base", None)):
            var4d.model.check_base(states, rtol=var4d.base_rtol)
        term = FourDObsTerm(iv, states, (xb0, grid0), model=var4d.model, press=var4d.press, ph=var4d.ph)
        xbu = {k: xb0[k] for k in _XBU}
        cost = VarCost.reference(b, xbu, [term])
        last["cost"] = cost
        clock["setup_s" if it > 1 else "setup1_s"] += time.perf_counter() - t
        return cost, (start, xb0, grid0, xbu, term)

    def update(it, context, xhat):
        t = time.perf_counter()
        start, xb0, grid0, xbu, term = context
        xa = xa_from_cv(b, xbu, np.asarray(xhat, dtype=np.float64))
        wrf = xatowrf(grid0, xb0, xa, check_rh=check_rh, var4d=True)
        last.update(xa=xa, wrf=wrf)
        state = _analysis_state(start, wrf) if it < max_ext_its else None
        if state is not None and callable(getattr(var4d.model, "boundary_increment", None)):
            # da_transfer_xatowrf.inc:665-727: the last TL call's table
            # perturbation (da_minimise_cg.inc:258, this loop's xhat) joins
            # the tables; the TL model's now, the nonlinear run's from
            # state.lbc_increment (a run sharing the model's window has it)
            from . import transfer4d as t4
            x0 = {k: np.asarray(xa[k], dtype=np.float64) for k in ("u", "v", "t", "q", "psfc")}
            inc = var4d.model.boundary_increment(t4.xa_to_tl(x0, xb0, grid0, press=var4d.press, ph=var4d.ph))
            if inc:
                var4d.model.add_boundary_increment(inc)
                state.lbc_increment = inc
        clock["transfer_s"] += time.perf_counter() - t
        return state

    def nl_model(it, state):
        t = time.perf_counter()
        run = var4d.nl_model(it, state)
        clock["nl_model_s"] += time.perf_counter() - t
        return run

    t1 = time.perf_counter()
    results, cvt, _, context = outer_loops(
        setup, update, Trajectory(slots=list(fgat)), max_ext_its=max_ext_its,
        nl_model=nl_model if var4d.nl_model is not None else None, model=var4d.model, ntmax=ntmax, eps=eps,
        write=write, **cg_kw)
    total = time.perf_counter() - t1
    timings["setup_s"] = (t1 - t0) + clock["setup1_s"]
    timings["minimise_s"] = total - sum(clock.values())
    timings["transfer_s"] = clock["transfer_s"]
    if max_ext_its > 1:
        timings["outer_setup_s"] = clock["setup_s"]
        timings["nl_model_s"] = clock["nl_model_s"]
    _, xb0, grid0, _, term = context
    log = "".join(r.log for r in results)
    fit = _fit(last["cost"], [term.d], results[-1].xhat, [term]) if report_fit else {}
    return VarResult(log=log, cg=results, cvt=np.asarray(cvt, dtype=np.float64), xa=last["xa"], wrf=last["wrf"],
                     xb=xb0, grid=grid0, nobs=int(term.mask.sum()), timings=timings, fit=fit)


def _fgat_refusals(obs, fgat, first_guess, *, window_minutes, time_window, max_ext_its, radiances, xb_override,
                   ensemble, hybrid, var4d=None, tier="reference"):
    """What FGAT (and 4D-Var) refuses, from the arguments and the files'
    headers alone (``Times``, global attributes, variable names): no field
    and no ``be.dat`` is read."""
    import types

    from . import fgat as fm

    if var4d is not None:
        _var4d_refusals(var4d, tier, [first_guess, *fgat], max_ext_its=max_ext_its)

    if radiances:
        raise NotImplementedError("FGAT with radiances is not ported (WRFDA bins radiances by time slot)")
    if ensemble is not None or hybrid is not None:
        raise NotImplementedError("FGAT with the hybrid (ensemble) B is not ported: WRFDA bins the ensemble "
                                  "by time slot only in 4DEnVar (use_4denvar)")
    if xb_override is not None:
        raise ValueError("xb_override replaces the analysis-time background only; it is not FGAT's slots")
    window = obs_window(types.SimpleNamespace(times=fm.first_guess_time(first_guess)), window_minutes)
    if time_window is not None:
        window.update(time_window_min=time_window[0], time_window_max=time_window[1])
    paths = [obs] if isinstance(obs, (str, Path)) else list(obs)
    fm.check_fgat(len(paths), [fm.first_guess_time(f) for f in fgat], time_window_min=window["time_window_min"],
                  time_window_max=window["time_window_max"], max_ext_its=1 if var4d is not None else max_ext_its,
                  analysis_date=window["analysis_date"])
    if var4d is not None:
        from .fourd import check_var4d_bins
        from .obs_wrfda import _date_minutes

        if abs(_date_minutes(window["analysis_date"]) - _date_minutes(window["time_window_min"])) >= 0.1:
            raise ValueError(f"4D-Var: the first guess is at {window['analysis_date'][:19]}, not at the window's "
                             f"start {window['time_window_min'][:19]} (the TL model starts from the analysis)")
        bins = check_var4d_bins(window["time_window_min"], window["time_window_max"], len(fgat),
                                var4d_bin=var4d.var4d_bin, time_step=var4d.time_step)
        sps = getattr(var4d.model, "sps", None)
        if sps is not None and bins["steps_per_bin"] is not None and sps != bins["steps_per_bin"]:
            raise ValueError(f"4D-Var: the window model reads a slot every {sps} steps, but var4d_bin / time_step "
                             f"= {bins['var4d_bin']} / {var4d.time_step} = {bins['steps_per_bin']}")


def _var4d_refusals(var4d, tier, first_guesses, *, max_ext_its: int = 1) -> None:
    """4D-Var's refusals from the options and the first guesses' headers.

    Where each of the 4D-Var plan's refusals is enforced:

    - here, before anything is read: a device ``tier`` (the 4D-Var term is
      host binary64 and has no device-tier comparison yet), ``press`` and
      ``ph`` values, ``HYBRID_OPT > 0`` and a first guess without ``DN``;
    - here too: more than one outer loop without ``var4d.nl_model`` (or with
      a TL model that cannot be relinearised);
    - :func:`_fgat_refusals` (also before reading): radiances, the hybrid
      (ensemble) B, ``xb_override``, a first guess not at the window's
      start, and :func:`~woof.da.var.fourd.check_var4d_bins` (WRFDA's
      ``var4d_bin`` and ``time_step`` rules, the slot count);
    - :func:`_analyse_var4d`, in each outer loop after the first: a rerun
      whose slots are not at the slot times, off the projection or shape;
    - :class:`~woof.da.var.fourd.FourDObsTerm` (after the files are read):
      :func:`~woof.da.var.transfer4d.check_supported` again on the read
      states (the ``c1``..``c4`` coefficients themselves, ``DN``, binary64,
      at least 2 points each way);
    - not options of :func:`analyse` at all, so nothing can ask for them:
      ``cloud_cv_options`` (always 0 here; the hydrometeor transfers are
      not ported), rain observations (``use_rainobs``), ``var4d_lbc``,
      ``jcdfi_use`` and ``multi_inc``.
    """
    from . import transfer4d as t4

    if tier != "reference":
        raise NotImplementedError(f"4D-Var with tier={tier!r}: the 4D-Var observation term runs on the host in "
                                  "binary64 and has no device-tier comparison yet; use tier='reference'")
    if var4d.press not in t4._PRESS:
        raise ValueError(f"var4d.press must be one of {t4._PRESS}, not {var4d.press!r}")
    if var4d.ph not in t4._PH:
        raise ValueError(f"var4d.ph must be one of {t4._PH}, not {var4d.ph!r}")
    if max_ext_its < 1:
        raise ValueError(f"max_ext_its must be >= 1, not {max_ext_its}")
    if max_ext_its > 1:
        if var4d.nl_model is None:
            raise NotImplementedError(f"4D-Var with max_ext_its = {max_ext_its} needs var4d.nl_model: WRFDA reruns "
                                      "the nonlinear model from the updated analysis in every outer loop "
                                      "(da_nl_model), and there is no nonlinear model to rerun without one")
        if not callable(getattr(var4d.model, "relinearise", None)):
            raise TypeError(f"4D-Var with max_ext_its = {max_ext_its}: {type(var4d.model).__name__} has no "
                            "relinearise(trajectory), so its TL and AD would stay on loop 1's trajectory")
    for f in first_guesses:
        attrs, names = _header(f)
        if int(attrs.get("HYBRID_OPT", 0)) > 0:
            raise NotImplementedError(f"4D-Var: {_name(f)} has HYBRID_OPT = {attrs['HYBRID_OPT']}; WRFDA's TL "
                                      "transfer assumes terrain-following eta (hybrid_opt <= 0)")
        if names is not None and "DN" not in names:
            raise ValueError(f"4D-Var: {_name(f)} has no DN, which da_transfer_xatowrftl integrates with")


def _name(f) -> str:
    return Path(getattr(f, "path", f)).name


def _header(f) -> tuple[dict, set | None]:
    """A first guess's global attributes and variable names (``None``: not
    known without reading, for an in-memory :class:`FirstGuess`)."""
    if isinstance(f, FirstGuess):
        return dict(f.attrs), set(f.fields)
    import netCDF4

    with netCDF4.Dataset(Path(f)) as nc:
        return {k: nc.getncattr(k) for k in nc.ncattrs()}, set(nc.variables)


def _fgat_innovations(obs, fgat, xb, proj, window, *, landuse_table):
    """FGAT's innovation vector, each slot's observations against its own first guess
    (after :func:`_fgat_refusals`), and the slots' ``(xb, grid)``."""
    from . import fgat as fm

    paths = [obs] if isinstance(obs, (str, Path)) else list(obs)
    states = _slot_states(fgat, xb, proj, landuse_table=landuse_table)
    wobs = fm.read_obs_ascii_fgat(paths, proj, **window)
    return fm.innovations_fgat(wobs, [x for x, _ in states]), states


def _slot_states(fgat, xb, proj, *, landuse_table, what: str = "FGAT slot") -> list:
    """Each slot's ``(xb, grid)`` (:func:`wrftoxb`), refusing a slot off the
    first guess's projection or shape."""
    from . import obs_wrfda as ow

    fgs = [f if isinstance(f, FirstGuess) else read_first_guess(f) for f in fgat]
    states = []
    for n, f in enumerate(fgs, start=1):
        if ow.LambertWRFDA.from_attributes(f.attrs) != proj:
            raise ValueError(f"{what} {n}: {f.path.name} is not on the first guess's projection")
        x, g = wrftoxb(f, landuse_table=landuse_table)
        if x["t"].shape != xb["t"].shape:
            raise ValueError(f"{what} {n}: {f.path.name} is {x['t'].shape}, the first guess {xb['t'].shape}")
        states.append((x, g))
    return states


def _fit(cost, d0, cvt, terms) -> dict:
    """RMS of O-B and of the linearised O-A (``d - H' U cvt``) per term, per conventional type and
    variable, and per radiance sensor and channel."""
    ys = cost.vtoy(cvt)
    out = {}
    for n, (t, d, y) in enumerate(zip(terms, d0, ys)):
        m = np.asarray(t.mask, bool)
        d = np.asarray(d, np.float64)
        a = d - np.asarray(y, np.float64)
        name = "radiance" if n else "conventional"
        rms = (lambda v, k: float(np.sqrt(np.mean(v[k] ** 2))) if k.any() else float("nan"))
        out[name] = {"n": int(m.sum()), "omb_rms": rms(d, m), "oma_rms": rms(a, m)}
        if not n:  # per conventional type and variable (``sound:t`` ...), in the term's Jo groups
            for kind, variables in t.jo_split():
                for var, idx in variables:
                    idx = np.asarray(idx, dtype=np.int64)
                    k = np.zeros(m.size, bool)
                    k[idx] = m[idx]
                    if k.any():
                        out[f"{kind}:{var}"] = {"n": int(k.sum()), "omb_rms": rms(d, k), "oma_rms": rms(a, k)}
        if n and hasattr(t, "batches"):
            for k, b in enumerate(t.batches):
                sl = slice(int(t.offsets[k]), int(t.offsets[k + 1]))
                dd, aa, mm = (v[sl].reshape(b.n, -1) for v in (d, a, m))
                for c, ch in enumerate(b.channels.tolist()):
                    out[f"{b.sensor}:{ch}"] = {"n": int(mm[:, c].sum()), "omb_rms": rms(dd[:, c], mm[:, c]),
                                               "oma_rms": rms(aa[:, c], mm[:, c])}
    return out


def _radiance_term(rad, radiances, xb, fg, proj, crtm, qc, varbc):
    """The radiance term at ``x_b`` (and VarBC's block), opening CRTM when the caller did not."""
    surface = rad.read_surface(fg.path)
    sensors = sorted({r.sensor for r in radiances})
    bias = None if varbc is None else varbc.bias
    if crtm is None:
        from woof.rtm.crtm import ClearSkyOperator

        with ClearSkyOperator(sensors) as op:
            term, report = rad.radiance_term(radiances, xb, surface, proj, op, settings=qc, bias=bias)
    else:
        term, report = rad.radiance_term(radiances, xb, surface, proj, crtm, settings=qc, bias=bias)
    if varbc is not None:
        varbc.attach(term)
    report["used"] = int(term.mask.sum())
    return term, report


# --------------------------------------------------------------------------- the bundle


def _to_file_order(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64).transpose())


def write_analysis_file(first_guess, wrf: dict, out) -> Path:
    """The first guess with :data:`WRF_FIELDS` replaced (``da_update_firstguess``), NetCDF classic.

    Written through :func:`woof.io.analysis_bundle.write_analysis`: the
    increments (analysis minus first guess, binary64) added to the first
    guess and rounded to each variable's stored type.  A first guess from
    the middle of a forecast is restamped to start at its valid time
    (:func:`restamp_start`), as WRFDA's output is.
    """
    from woof.io.analysis_bundle import write_analysis

    fg = first_guess if isinstance(first_guess, FirstGuess) else read_first_guess(first_guess)
    import netCDF4

    increments = {}
    with netCDF4.Dataset(fg.path) as nc:
        nc.set_auto_mask(False)
        for name in WRF_FIELDS:
            base = np.asarray(nc.variables[name][0], dtype=np.float64)
            increments[name] = _to_file_order(wrf[name]) - base
    write_analysis(fg.path, increments, out)
    restamp_start(out, fg.times)
    return Path(out)


def restamp_start(path, valid_time: str) -> bool:
    """Stamp a written analysis as starting at its valid time, as WRFDA's
    ``da_med_initialdata_output`` does (``start_date = current_date``, then
    ``GMT``/``JULYR``/``JULDAY`` from it; the run's simulation start is the
    analysis time too).

    A first guess from the middle of a forecast (a ``wrfout`` or a
    ``write_input`` file at 15Z of a run started at 12Z) carries the
    forecast's ``START_DATE``; without this the analysis would claim to start
    at 12Z while holding the 15Z state, and the bundle refuses it.  Files
    whose ``START_DATE`` is already the valid time are not touched, byte for
    byte.  Returns whether the file was changed.
    """
    import netCDF4

    stamp = str(valid_time)[:19]
    t = _dt.datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S")
    with netCDF4.Dataset(path) as nc:
        attrs = {k: nc.getncattr(k) for k in nc.ncattrs()}
    if str(attrs.get("START_DATE", stamp)) == stamp:
        return False
    want = {"START_DATE": stamp, "SIMULATION_START_DATE": stamp,
            "GMT": np.float32(t.hour + t.minute / 60.0 + t.second / 3600.0),
            "JULYR": np.int32(t.year), "JULDAY": np.int32(t.timetuple().tm_yday)}
    with netCDF4.Dataset(path, "a") as nc:
        for k, v in want.items():
            if k in attrs:
                old = attrs[k]
                nc.setncattr(k, v if isinstance(old, str) else np.asarray(old).dtype.type(v))
    return True


def write_bundle(out, first_guess, wrf: dict, *, recipe: dict, grid_id: str, physics_suite: str = "unspecified",
                 iau_window_s: float | None = None, ph: str = "analysis", notes: dict | None = None):
    """Write ``wrf`` (from :func:`xatowrf`) as a ``wrf-analysis-bundle/1`` at ``out``.

    The analysis is a patched copy of the first guess (:func:`write_analysis_file`);
    :func:`woof.io.analysis_bundle.make_bundle` then digests it, checks it and
    writes the manifest (``kind = control``, one member, one cycle).  ``ph``
    ``analysis`` keeps WRFDA's hydrostatic ``PH``; ``balance`` lets the
    bundle writer recompute it.  ``recipe`` is the run's configuration,
    digested into ``analysis.recipe_sha256``.
    """
    from woof.io.analysis_bundle import make_bundle

    fg = first_guess if isinstance(first_guess, FirstGuess) else read_first_guess(first_guess)
    recipe_bytes = json.dumps(recipe, sort_keys=True).encode()
    with tempfile.TemporaryDirectory(prefix="woof-var-") as tmp:
        a = write_analysis_file(fg, wrf, Path(tmp) / "analysis.nc")
        bundle, record = make_bundle(
            out, background=fg.path, analysis=a, grid_id=grid_id, static_builder="wrf first guess",
            physics_suite=physics_suite, physics_mode="correct", kind="control", members=1,
            recipe_id="woof-var-3dvar", recipe_sha256=hashlib.sha256(recipe_bytes).hexdigest(), cycles=1,
            window=None, producer="woof var (WRFDA 4.7.1 3D-Var port)", iau_window_s=iau_window_s, ph=ph,
            hypsometric_opt=int(fg.attrs.get("HYPSOMETRIC_OPT", 2)), notes=notes)
    return bundle, record


def increment_stats(ours: np.ndarray, theirs: np.ndarray) -> dict:
    """O7's numbers: max |difference| over the increment's RMS, and the correlation."""
    a = np.asarray(ours, dtype=np.float64).ravel()
    b = np.asarray(theirs, dtype=np.float64).ravel()
    rms = math.sqrt(float(np.mean(b * b)))
    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        corr = 1.0 if np.array_equal(a, b) else 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    return {"rms": rms, "max_abs_diff": diff, "max_over_rms": diff / rms if rms else (0.0 if diff == 0 else math.inf),
            "corr": corr}
