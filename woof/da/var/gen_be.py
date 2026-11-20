"""gen_be: background error statistics (``be.dat``) from forecast differences.

WRFDA's ``var/gen_be`` estimates the regional ``B`` of cv_options 5 and 7
from NMC pairs (Parrish and Derber 1992: a long and a short forecast valid at
the same time) or from ensemble perturbations (member minus ensemble mean).
This module is that pipeline in binary64, stage by stage, for woof's own
forecasts (WRF-named history fields, as ``wrfout`` files or as in-memory
member dictionaries):

stage 0 (``gen_be_stage0_wrf``)
    psi and chi from the C-grid winds by a sine-transform Poisson solve
    (cv5), or the winds at mass points (cv7); temperature, relative humidity
    (fraction), surface pressure; differences of the two forecasts or
    deviations from the ensemble mean.
stage 1 (``gen_be_stage1``)
    the mean over all samples removed point by point; bins (``bin_type`` 5:
    one bin per level, one 2-D bin).
stage 2 and 2a (``gen_be_stage2``, ``gen_be_stage2a``)
    the balance regressions of chi, T and ps on psi and the unbalanced
    chi_u, t_u, ps_u (cv5 only).
stage 3 (``gen_be_stage3``)
    vertical covariances, their EOFs (LAPACK ``dsyev``, as WRFDA), and each
    3-D field projected on the global modes.
stage 4 (``gen_be_stage4_regional``)
    horizontal length scales per mode: the correlation against distance,
    fitted with Yong-Run's Gaussian through the origin; then
    ``gen_be_diags``' spike removal and smoothing.

The result is a :class:`woof.da.var.be_dat.BeDat`, written with
:func:`woof.da.var.be_dat.write_be_dat` in WRFDA's layout, so the same file
runs in woof's 3D-Var and in WRFDA.

Operation order follows WRFDA statement by statement where it is cheap
(running means, regressions, projections and the fit; every reduction is
left to right, as gfortran ``-O2`` without FMA evaluates them).  Three steps
are not WRFDA's words, and the gate (``tests/test_var_gen_be.py``) states a
tolerance for them instead: the Poisson solve uses an exact discrete sine
transform where WRFDA's ``fft661`` builds its trig tables from
``pi = 3.1415926`` (a relative difference near 1e-7 in psi and chi), the
eigensolver is the host's LAPACK ``dsyev`` rather than WRFDA's bundled copy,
and the stage-4 pair sums are formed per displacement by a fixed-order
autocorrelation rather than WRFDA's pair loop.

Only ``bin_type = 5`` and ``cv_options`` 5 and 7 are implemented; others are
refused by name.

Derived from WRFDA 4.7.1 (``var/gen_be``, ``var/da/da_gen_be``), public
domain; see ``licenses/LICENSE-WRF-public-domain.txt``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from woof.da.var.be_dat import BeDat, BeEigen

__all__ = [
    "GenBeConfig",
    "GenBeResult",
    "read_wrf_history",
    "standard_fields",
    "nmc_perturbation",
    "ensemble_perturbations",
    "gen_be",
]

# WRFDA's da_control constants (real*8 under -fdefault-real-8).
PI = 3.1415926
GAS_CONSTANT = 287.0
GAS_CONSTANT_V = 461.6
CP = 7.0 * GAS_CONSTANT / 2.0
KAPPA = GAS_CONSTANT / CP
RD_OVER_RV = GAS_CONSTANT / GAS_CONSTANT_V
RD_OVER_RV1 = 1.0 - RD_OVER_RV
T_KELVIN = 273.15
GRAVITY = 9.81
ES_ALPHA = 611.2
ES_BETA = 17.67
ES_GAMMA = 243.5
T0 = 300.0
BASE_PRES = 100000.0  # set by gen_be_stage0_wrf itself

# The WRF history fields stage 0 reads.
HISTORY_FIELDS = ("U", "V", "T", "P", "PB", "QVAPOR", "PH", "PHB", "PSFC", "XLAT", "MAPFAC_M", "MAPFAC_U",
                  "MAPFAC_V")

_CV_NAMES = {5: ("psi", "chi_u", "t_u", "rh", "ps_u"), 7: ("u", "v", "t", "rh", "ps")}


@dataclass(frozen=True)
class GenBeConfig:
    """gen_be's namelist keys (``gen_be_set_defaults.ksh`` defaults)."""

    cv_options: int = 5
    bin_type: int = 5
    lat_min: float = -90.0
    lat_max: float = 90.0
    binwidth_lat: float = 10.0
    hgt_min: float = 0.0
    hgt_max: float = 20000.0
    binwidth_hgt: float = 1000.0
    n_smth_sl: int = 2
    stride: int = 1
    num_passes: int = 0
    use_global_eofs: bool = True


@dataclass
class GenBeResult:
    """``be.dat`` plus the intermediate statistics the gates compare."""

    be: BeDat
    regcoeff: dict = field(default_factory=dict)
    stage3: dict = field(default_factory=dict)  # name -> (evec, eval, evec_loc, eval_loc)
    scale_length_raw: dict = field(default_factory=dict)  # name -> (nk,) stage-4 fits, before smoothing
    nsamples: int = 0


# --------------------------------------------------------------------------- stage 0


def read_wrf_history(path, time_index: int = 0) -> dict:
    """The fields stage 0 reads from a WRF/woof history file, ``(i, j, k)`` binary64.

    ``DX`` is returned as ``ds``.  Values are the file's binary32 words
    widened exactly (WRFDA's ``da_get_field`` reads ``real*4``).
    """
    import netCDF4

    out = {}
    with netCDF4.Dataset(Path(path)) as nc:
        nc.set_auto_mask(False)
        for n in HISTORY_FIELDS:
            if n not in nc.variables:
                raise ValueError(f"{Path(path).name}: lacks {n}, which gen_be stage 0 reads")
            a = np.asarray(nc.variables[n][time_index], dtype=np.float32).astype(np.float64)
            out[n] = np.ascontiguousarray(a.transpose())
        out["ds"] = float(np.float32(nc.getncattr("DX")))
    return out


def _find_fft_factors(n: int) -> bool:
    """``da_find_fft_factors``' verdict: ``n`` is a product of 2, 3 and 5."""
    nu = n
    for f in (6, 8, 5, 4, 3, 2):
        while nu % f == 0:
            nu //= f
            if nu == 1:
                return True
    return nu == 1


def _fft_size(dim: int) -> int:
    """``da_fft_initialize1``: the first even 2-3-5 number from ``dim - 1``."""
    n0 = dim - 1
    for n in range(n0, n0 + 1001):
        if _find_fft_factors(n) and n % 2 == 0:
            return n
    raise ValueError(f"no FFT size within 1000 of {n0}")


def _fft_coeffs(n1: int, n2: int, ds: float) -> np.ndarray:
    """``da_fft_initialize2``'s ``fft_coeffs`` (0-based, (n1+1, n2+1))."""
    const = -0.5 * ds * ds
    cx = PI / float(n1)
    cy = PI / float(n2)
    cosx = np.cos(cx * np.arange(n1 + 1, dtype=np.float64))
    cosy = np.cos(cy * np.arange(n2 + 1, dtype=np.float64))
    with np.errstate(divide="ignore"):
        c = const / ((2.0 - cosx[:, None]) - cosy[None, :])
    c[0, 0] = 0.0
    return c


def _del2a_to_a(del2a: np.ndarray, ds: float) -> np.ndarray:
    """``da_del2a_to_a`` with ``fft_method = 2``: inverse 5-point Laplacian, zero boundary.

    Padded to an even 2-3-5 size as WRFDA does; the transform is an exact
    DST-I (WRFDA's ``fft661`` builds its tables from ``pi = 3.1415926``).
    """
    from scipy.fft import dstn, idstn

    dim1, dim2 = del2a.shape
    n1, n2 = _fft_size(dim1), _fft_size(dim2)
    a2d = np.zeros((n1 + 1, n2 + 1))
    a2d[1 : dim1 - 1, 1 : dim2 - 1] = del2a[1 : dim1 - 1, 1 : dim2 - 1]
    inner = a2d[1:n1, 1:n2]
    spec = dstn(inner, type=1)
    spec = spec * _fft_coeffs(n1, n2, ds)[1:n1, 1:n2]
    a2d[1:n1, 1:n2] = idstn(spec, type=1)
    return a2d[:dim1, :dim2].copy()


def _uv_to_vor_div(u, v, mapfac_u, mapfac_v, ds):
    """``da_uv_to_vor_c`` and ``da_uv_to_div_c`` for one level."""
    dim1, dim2 = v.shape[0], u.shape[1]
    ds_inv = 1.0 / ds
    um = u / mapfac_u
    with np.errstate(divide="ignore", invalid="ignore"):
        vm = np.where(mapfac_v > 0.000001, v / mapfac_v, 0.0)
    vor = np.zeros((dim1 + 1, dim2 + 1))
    vor[1:dim1, 1:dim2] = ds_inv * (((vm[1:dim1, 1:dim2] - vm[0 : dim1 - 1, 1:dim2]) - um[1:dim1, 1:dim2])
                                    + um[1:dim1, 0 : dim2 - 1])
    vor[0, 1:dim2] = vor[1, 1:dim2]
    vor[dim1, 1:dim2] = vor[dim1 - 1, 1:dim2]
    vor[:, 0] = vor[:, 1]
    vor[:, dim2] = vor[:, dim2 - 1]
    div = ds_inv * (((um[1 : dim1 + 1, :] - um[0:dim1, :]) + vm[:, 1 : dim2 + 1]) - vm[:, 0:dim2])
    return vor, div


def standard_fields(f: Mapping, cv_options: int = 5, ds: float | None = None) -> dict:
    """``gen_be_stage0_wrf`` for one member: psi/chi (or u/v), t, rh, ps, height, lat.

    ``f`` holds WRF-named fields in ``(i, j, k)`` order (:data:`HISTORY_FIELDS`;
    ``ds`` from the dictionary when not given).  ``rh`` is a fraction, as
    WRFDA's ``0.01 * rh``.
    """
    if cv_options not in _CV_NAMES:
        raise NotImplementedError(f"woof gen_be: cv_options={cv_options} is not implemented (5 and 7 are)")
    ds = float(f["ds"] if ds is None else ds)
    U = np.asarray(f["U"], dtype=np.float64)
    V = np.asarray(f["V"], dtype=np.float64)
    dim1, dim2, dim3 = U.shape[0] - 1, U.shape[1], U.shape[2]
    mfu = np.asarray(f["MAPFAC_U"], dtype=np.float64)
    mfv = np.asarray(f["MAPFAC_V"], dtype=np.float64)
    out = {}
    if cv_options == 7:
        out["u"] = 0.5 * (U[0:dim1] + U[1 : dim1 + 1])
        out["v"] = 0.5 * (V[:, 0:dim2] + V[:, 1 : dim2 + 1])
    else:
        psi = np.empty((dim1, dim2, dim3))
        chi = np.empty((dim1, dim2, dim3))
        for k in range(dim3):
            vor, div = _uv_to_vor_div(U[:, :, k], V[:, :, k], mfu, mfv, ds)
            p2 = _del2a_to_a(vor, ds)
            chi[:, :, k] = _del2a_to_a(div, ds)
            psi[:, :, k] = 0.25 * (((p2[0:dim1, 0:dim2] + p2[1 : dim1 + 1, 0:dim2]) + p2[0:dim1, 1 : dim2 + 1])
                                   + p2[1 : dim1 + 1, 1 : dim2 + 1])
        out["psi"] = psi
        out["chi"] = chi
    theta = T0 + np.asarray(f["T"], dtype=np.float64)
    p = np.asarray(f["PB"], dtype=np.float64) + np.asarray(f["P"], dtype=np.float64)
    temp = theta * np.power(p / BASE_PRES, KAPPA)
    x = np.asarray(f["QVAPOR"], dtype=np.float64)
    q = x / (1.0 + x)
    t_c = temp - T_KELVIN
    es = ES_ALPHA * np.exp((ES_BETA * t_c) / (t_c + ES_GAMMA))
    qs = (RD_OVER_RV * es) / (p - RD_OVER_RV1 * es)
    out["t"] = temp
    out["rh"] = 0.01 * ((100.0 * q) / qs)
    out["ps"] = np.asarray(f["PSFC"], dtype=np.float64).copy()
    ph = np.asarray(f["PHB"], dtype=np.float64) + np.asarray(f["PH"], dtype=np.float64)
    out["height"] = (0.5 * (ph[:, :, 1:] + ph[:, :, :-1])) * (1.0 / GRAVITY)
    out["lat"] = np.asarray(f["XLAT"], dtype=np.float64).copy()
    return out


_PERT = {5: ("psi", "chi", "t", "rh", "ps"), 7: ("u", "v", "t", "rh", "ps")}


def nmc_perturbation(long: Mapping, short: Mapping, cv_options: int = 5) -> dict:
    """An NMC sample: the long forecast's standard fields minus the short one's.

    Both are :func:`standard_fields` results (or WRF-named field dicts, which
    are converted).  Height and latitude are the long forecast's.
    """
    a = long if "t" in long else standard_fields(long, cv_options)
    b = short if "t" in short else standard_fields(short, cv_options)
    out = {n: a[n] - b[n] for n in _PERT[cv_options]}
    out["height"], out["lat"] = b["height"], b["lat"]
    return out


def ensemble_perturbations(members: Sequence[Mapping], cv_options: int = 5) -> list:
    """``be_method = ENS``: each member's standard fields minus their running mean."""
    std = [m if "t" in m else standard_fields(m, cv_options) for m in members]
    if len(std) < 2:
        raise ValueError("the ensemble method needs at least two members")
    mean = {n: np.zeros_like(std[0][n]) for n in _PERT[cv_options]}
    for c, s in enumerate(std, start=1):
        inv = 1.0 / float(c)
        for n in mean:
            mean[n] = (float(c - 1) * mean[n] + s[n]) * inv
    out = []
    for s in std:
        d = {n: s[n] - mean[n] for n in _PERT[cv_options]}
        d["height"], d["lat"] = s["height"], s["lat"]
        out.append(d)
    return out


# --------------------------------------------------------------------------- stages 1-4


def _dsyev(a: np.ndarray):
    """``da_eof_decomposition``: LAPACK ``dsyev('V', 'U')``, modes in decreasing order."""
    from scipy.linalg import lapack

    w, v, info = lapack.dsyev(np.asfortranarray(a), compute_v=1, lower=0)
    if info != 0:
        raise ValueError(f"dsyev failed, info = {info}")
    return np.ascontiguousarray(v[:, ::-1]), w[::-1].copy()


def _running_outer(samples: Sequence[np.ndarray], a_of, b_of, shape) -> np.ndarray:
    """``cov = coeffb * cov + (coeffa * a) * b`` over every point, WRFDA's order.

    Points run (sample, j, i) with ``i`` fastest; ``a_of(s)``/``b_of(s)`` return
    per-point arrays shaped ``(ni, nj, ...)``.
    """
    cov = np.zeros(shape)
    count = 0
    for s in samples:
        a_s, b_s = a_of(s), b_of(s)
        ni, nj = a_s.shape[:2]
        for j in range(nj):
            for i in range(ni):
                count += 1
                coeffa = 1.0 / float(count)
                coeffb = float(count - 1) * coeffa
                ca = coeffa * a_s[i, j]
                cov = coeffb * cov + np.multiply.outer(ca, b_s[i, j]) if ca.ndim else coeffb * cov + ca * b_s[i, j]
    return cov


def _seq_dot_last(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``sum(a * b)`` along the last axis, left to right."""
    acc = np.zeros(np.broadcast_shapes(a.shape, b.shape)[:-1])
    for k in range(a.shape[-1]):
        acc = acc + a[..., k] * b[..., k]
    return acc


def _stage1(samples: Sequence[dict], names) -> list:
    """``gen_be_stage1``: remove the running mean over all samples."""
    mean = {n: np.zeros_like(samples[0][n]) for n in names}
    for c, s in enumerate(samples, start=1):
        inv = 1.0 / float(c)
        for n in names:
            mean[n] = (float(c - 1) * mean[n] + s[n]) * inv
    return [{**{n: s[n] - mean[n] for n in names}, "height": s["height"], "lat": s["lat"]} for s in samples]


def _stage2(samples: Sequence[dict], nk: int):
    """``gen_be_stage2`` for ``bin_type = 5``: psi->chi, psi->ps, psi->T regressions."""
    variance_threshold = 1e-6
    # covar1/var1 per level (bin b = k): the per-k sequence over (sample, j, i).
    covar1 = np.zeros(nk)
    var1 = np.zeros(nk)
    count = 0
    for s in samples:
        psi, chi = s["psi"], s["chi"]
        ni, nj = psi.shape[:2]
        for j in range(nj):
            for i in range(ni):
                count += 1
                coeffa = 1.0 / float(count)
                coeffb = float(count - 1) * coeffa
                cp_ = coeffa * psi[i, j]
                covar1 = coeffb * covar1 + cp_ * chi[i, j]
                var1 = coeffb * var1 + cp_ * psi[i, j]
    covar2 = _running_outer(samples, lambda s: s["ps"], lambda s: s["psi"], (nk,))
    covar3 = _running_outer(samples, lambda s: s["t"], lambda s: s["psi"], (nk, nk))
    var2 = _running_outer(samples, lambda s: s["psi"], lambda s: s["psi"], (nk, nk))
    # Only k2 <= k is accumulated, then mirrored: the same words.
    var2 = np.tril(var2) + np.tril(var2, -1).T
    evec, eval_ = _dsyev(var2)
    total = 0.0
    for m in range(nk):
        total = total + eval_[m]
    cumul = 0.0
    mmax = nk
    for m in range(nk):
        cumul = cumul + eval_[m] / total
        if cumul > 1.0 - variance_threshold:
            mmax = m
            break
    lam = np.zeros((nk, nk))  # LamInvET(m, k)
    lam[:mmax, :] = (evec[:, :mmax] / eval_[:mmax]).T
    inv = np.zeros((nk, nk))
    for m in range(nk):
        inv = inv + np.multiply.outer(evec[:, m], lam[m, :])
    inv = np.tril(inv) + np.tril(inv, -1).T
    reg1 = covar1 / var1
    reg2 = np.zeros(nk)
    for k2 in range(nk):
        reg2 = reg2 + covar2[k2] * inv[k2, :]
    reg3 = np.zeros((nk, nk))
    for k3 in range(nk):
        reg3 = reg3 + np.multiply.outer(covar3[:, k3], inv[k3, :])
    return {"psi_chi": reg1, "psi_ps": reg2[:, None], "psi_t": reg3[:, :, None]}


def _stage2a(samples: Sequence[dict], reg: dict) -> list:
    """``gen_be_stage2a``: chi_u, t_u, ps_u."""
    r1, r2, r3 = reg["psi_chi"], reg["psi_ps"][:, 0], reg["psi_t"][:, :, 0]
    out = []
    for s in samples:
        psi = s["psi"]
        chi_u = s["chi"] - r1[None, None, :] * psi
        t_u = s["t"] - _seq_dot_last(r3[None, None, :, :], psi[:, :, None, :])
        ps_u = s["ps"] - _seq_dot_last(r2[None, None, :], psi)
        out.append({**s, "chi_u": chi_u, "t_u": t_u, "ps_u": ps_u})
    return out


def _stage3(samples: Sequence[dict], name: str, cfg: GenBeConfig):
    """``gen_be_stage3`` for one variable: EOFs and (3-D) the projected modes."""
    fields = []
    for s in samples:
        f = s[name]
        f = f[:, :, None] if f.ndim == 2 else f
        ni, nj, nk = f.shape
        inv_nij = 1.0 / float(ni * nj)
        g = np.empty_like(f)
        for k in range(nk):
            flat = f[:, :, k].ravel(order="F")
            mean = np.cumsum(flat)[-1] * inv_nij
            g[:, :, k] = f[:, :, k] - mean
        fields.append(g)
    bv = _running_outer(fields, lambda g: g, lambda g: g, (nk, nk))
    bv = np.tril(bv) + np.tril(bv, -1).T
    e_vec_loc, e_val_loc = _dsyev(bv)
    work = bv / 1.0  # sum over num_bins2d = 1 bins, divided by num_bins2d
    e_vec, e_val = _dsyev(work)
    stats = (e_vec, e_val, e_vec_loc[:, :, None], e_val_loc[:, None])
    if nk == 1:
        return stats, None
    ev, lv = (e_vec, e_val) if cfg.use_global_eofs else (e_vec_loc, e_val_loc)
    sq = np.sqrt(lv)
    modes = []
    for s in samples:
        vp = s[name]
        vv = np.empty_like(vp)
        for m in range(nk):
            acc = np.zeros(vp.shape[:2])
            for k in range(nk):
                acc = acc + ev[k, m] * vp[:, :, k]
            vv[:, :, m] = acc / sq[m]
        modes.append(vv)
    return stats, modes


def _pair_sums(f: np.ndarray, nn: int) -> np.ndarray:
    """``get_covariance``'s ``bb`` before the division: sum over pairs by squared distance.

    Pairs as WRFDA counts them: the same row with ``m >= i``, and every point
    of every later row.  Each displacement's sum comes from one zero-padded
    autocorrelation; displacements are folded into ``d2`` bins in a fixed order.
    """
    ni, nj = f.shape
    p = np.zeros((2 * ni, 2 * nj))
    p[:ni, :nj] = f
    F = np.fft.rfft2(p)
    ac = np.fft.irfft2(F * np.conj(F), s=p.shape)  # ac[dx, dy] = sum f(i,j) f(i+dx, j+dy)
    bb = np.zeros(nn + 1)
    dx = np.arange(-(ni - 1), ni)
    for dy in range(0, nj):
        row = ac[dx % (2 * ni), dy]
        sel = dx >= 0 if dy == 0 else np.ones_like(dx, dtype=bool)
        np.add.at(bb, dx[sel] ** 2 + dy * dy, row[sel])
    return bb


def _pair_counts(ni: int, nj: int, nn: int) -> np.ndarray:
    """``get_grid_info``'s ``nr`` (stride 1, one latitude bin)."""
    nr = np.zeros(nn + 1, dtype=np.int64)
    dx = np.arange(-(ni - 1), ni)
    for dy in range(0, nj):
        mult = (ni - np.abs(dx)) * (nj - dy)
        sel = dx >= 0 if dy == 0 else np.ones_like(dx, dtype=bool)
        np.add.at(nr, dx[sel] ** 2 + dy * dy, mult[sel])
    return nr


def _fortran_e20_8(x: float) -> float:
    """A value written with ``(e20.8)`` and read back: 8 significant digits."""
    return float(f"{x:.7e}")


def _scale_length(fields: Sequence[np.ndarray]) -> tuple[float, float]:
    """``gen_be_stage4_regional`` for one mode: ``(ml, sl)``, sl in grid lengths."""
    ni, nj = fields[0].shape
    nn = ni * ni + nj * nj
    nr = _pair_counts(ni, nj, nn)
    has = nr != 0
    cov = np.zeros(nn + 1)
    for c, f in enumerate(fields, start=1):
        count = float(c)
        inv = 1.0 / count
        bb = _pair_sums(f, nn)
        bb[has] = bb[has] / nr[has].astype(np.float64)
        cov[has] = ((count - 1.0) * cov[has] + bb[has]) * inv
    yr_cutoff = 3.0
    corr_min = np.exp(-0.125 * yr_cutoff * yr_cutoff)
    coeff1 = 0.0
    coeff2 = 0.0
    for d2 in range(1, nn + 1):
        if nr[d2] > 0 and cov[d2] < cov[0]:
            if cov[d2] / cov[0] < corr_min:
                break
            yr = np.sqrt(8.0 * np.log(cov[0] / cov[d2]))
            d = np.sqrt(float(d2))
            nrr = float(nr[d2])
            coeff1 = coeff1 + (nrr * d) * yr
            coeff2 = coeff2 + (nrr * d) * d
    if coeff2 > 0.0:
        ml = coeff1 / coeff2
        return float(ml), float(1.0 / ml)
    return 0.0, 0.0


def _diags_scale(ml: np.ndarray, sl: np.ndarray, n_smth_sl: int) -> np.ndarray:
    """``da_readwrite_be_stage4`` (``uh_method = 'scale'``): the e20.8 round trip, spikes, smoothing."""
    spike_tolerance = 1.5
    nk = sl.size
    s = np.array([_fortran_e20_8(v) for v in sl])
    m = np.array([_fortran_e20_8(v) for v in ml])
    for k in range(nk):
        if m[k] == 0.0 and s[k] == 0.0 and k > 0:
            s[k] = s[k - 1]
    for k in range(1, nk - 1):
        mean_scale = 0.5 * (s[k - 1] + s[k + 1])
        if s[k] > spike_tolerance * mean_scale:
            s[k] = mean_scale
    sm = s.copy()
    for _ in range(n_smth_sl):
        for k in range(1, nk - 1):
            sm[k] = s[k] + 0.25 * ((s[k - 1] + s[k + 1]) - 2.0 * s[k])
        s = sm.copy()
    return s


def gen_be(samples: Iterable[Mapping], config: GenBeConfig = GenBeConfig()) -> GenBeResult:
    """Stages 1-4 and ``gen_be_diags`` on stage-0 samples, in (time, member) order.

    ``samples`` are :func:`nmc_perturbation` or :func:`ensemble_perturbations`
    results.  Returns ``be.dat`` as a :class:`BeDat` (``be.sha256`` is empty
    until it is written).
    """
    c = config
    if c.bin_type != 5:
        raise NotImplementedError(f"woof gen_be: bin_type={c.bin_type} is not implemented (5 is)")
    if c.cv_options not in _CV_NAMES:
        raise NotImplementedError(f"woof gen_be: cv_options={c.cv_options} is not implemented (5 and 7 are)")
    if c.num_passes != 0:
        raise NotImplementedError("woof gen_be: num_passes > 0 (da_filter_regcoeffs) is not implemented")
    if c.stride != 1:
        raise NotImplementedError("woof gen_be: stride > 1 is not implemented")
    samples = list(samples)
    if len(samples) < 2:
        raise ValueError("gen_be needs at least two samples")
    names = _PERT[c.cv_options]
    for s in samples:
        missing = [n for n in names + ("height", "lat") if n not in s]
        if missing:
            raise ValueError(f"a gen_be sample lacks {missing} (cv_options={c.cv_options})")
    ni, nj, nk = samples[0][names[2]].shape
    pert = _stage1(samples, names)
    bins = np.broadcast_to(np.arange(1, nk + 1, dtype=np.int64), (ni, nj, nk)).copy()
    bins2d = np.ones((ni, nj), dtype=np.int64)
    regcoeff: dict = {}
    if c.cv_options == 5:
        regcoeff = _stage2(pert, nk)
        pert = _stage2a(pert, regcoeff)
    cv = _CV_NAMES[c.cv_options]
    eig, rfls, raw = [], [], {}
    stage3 = {}
    for name in cv:
        stats, modes = _stage3(pert, name, c)
        stage3[name] = stats
        e_vec, e_val, e_vec_loc, e_val_loc = stats
        eig.append(BeEigen(name, e_vec, e_val, e_vec_loc, e_val_loc))
        if modes is None:  # 2-D: stage 4 reads stage 2a's (or stage 1's) field itself
            per_mode = [[s[name] for s in pert]]
        else:
            per_mode = [[m[:, :, k] for m in modes] for k in range(nk)]
        fits = [_scale_length(fs) for fs in per_mode]
        ml = np.array([f[0] for f in fits])
        sl = np.array([f[1] for f in fits])
        raw[name] = sl
        rfls.append(_diags_scale(ml, sl, c.n_smth_sl))
    be = BeDat(
        ni=ni, nj=nj, nk=nk, bin_type=c.bin_type, eta_be=None,
        lat_min=c.lat_min, lat_max=c.lat_max, binwidth_lat=c.binwidth_lat,
        hgt_min=c.hgt_min, hgt_max=c.hgt_max, binwidth_hgt=c.binwidth_hgt,
        num_bins=nk, num_bins2d=1, bin=bins, bin2d=bins2d, cv_options=c.cv_options,
        regcoeff=regcoeff if c.cv_options == 5 else {},
        eigen=tuple(eig[:4]), eigen_ps=eig[4], rf_lengthscale=tuple(rfls[:4]), rf_lengthscale_ps=rfls[4],
        endian=">", real_size=8, sha256="",
    )
    return GenBeResult(be=be, regcoeff=regcoeff, stage3=stage3, scale_length_raw=raw, nsamples=len(samples))
