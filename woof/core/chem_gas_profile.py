"""WRF-Chem's idealized RADM2 background gas profile (chem_in_opt = 0) and the
RADM2 lateral boundary value, word for word against WRF v4.7.1.

Source: chem/module_input_chem_data.F (public domain, FSL notice), whose
numbers are transcribed into woof/core/chem_gas_profile_data.py by
tools/chem_gas_profile/transcribe.py.  Pure NumPy, float32 arithmetic in
WRF's order of operations; no CuPy.  Graded bit for bit by
tests/test_chem_gas_profile_wrf471_parity.py against the oracle in
tools/chem_wrf471_oracle/gasprofile.

Initial condition (real.exe, main/real_em.F:553 -> input_chem_profile)
    :714  si_zsigf = (ph_1 + phb) / grav, grav = 9.80622 (module_data_sorgam.F:575,
          NOT module_model_constants' g = 9.81): :func:`zsigf_from_geopotential`.
    :716-719  si_zsig(k) = 0.5*(zsigf(k)+zsigf(k+1)) for k < kde, and
          si_zsig(kde) = 0.5*(3.*zsigf(kde) - zsigf(kde-1)).
    :726  setup_gasprofile_maps -> setup_gasprofile_map_radm_racm (:409-431).
    :763  make_chem_profile (:775-937): the NALROM profiles xl (top first) are
          flipped onto zprof(k) = 0.5*(zfa(k)+zfa(k+1)) (:812-813); xl rows
          28-34 are divided by dens(k), with k the FLIPPED (bottom-up) index
          while dens is stored top first (:826, as WRF does); vinterp_chem
          (:567-675, extrapolate = .false.) interpolates every row but xl(34)
          (it is called with nch = lo, so chem slot lo+1 is never written;
          no RADM2 row reads it); then chem(l) = fracref(l-1)*stor(iref(l-1))*1.E6
          (:845) for l = p_so2 .. p_ho2.
    The result is ppmv on every mass level (WRF also fills level kde, the
    extrapolated one above the top mass level; ``include_top=True`` returns it).

Lateral boundary (wrf.exe, dyn_em/solve_em.F:2599 -> flow_dep_bdy_chem
(:1531-2031)): with have_bcs_chem = .false., i_bdy_method = 1 for
p_so2 <= ic <= p_ho2 under chem_opt 1 and 2 (:1631-1633; the RACM branch
:1635-1648 does not apply), and every INFLOW cell of the specified zone
(:1739-1745 etc.; outflow cells copy the inner neighbour) gets
bdy_chem_value(chem, z(i,k,j), ic, numgas) (:1415-1504) at the mass-level
height grid%z.  bdy_chem_value flips xl with the CORRECT dens(kx+1-k)
(:1460-1467), interpolates on zfa_bdy, and for chem_opt = 2 (p_nu0 > 1) scales
sulf by (1. - so4vaptoaer) (:1499-1501).  :func:`radm2_boundary_value`.

Registry order and WRF's map.  The gases are the radm2 / radm2sorg package
order (Registry/registry.chem:3929-3930), which puts hcl before ho and ho2,
while setup_gasprofile_map_radm_racm's 41 entries were written without hcl
(their ggnam labels end ..., 'ISO', 'HO', 'HO2').  WRF applies the map by
index, so hcl takes the 'HO' entry (xl 28, OH), ho the 'HO2' entry (xl 29) and
ho2 the iref(:) = 7 / fracref(:) = 1. default (xl 7, isoprene).  This module
reproduces that, as it reproduces the dens flip.

gas_ic_opt / gas_bc_opt = 101 (GAS_IC_PNNL / GAS_BC_PNNL) call
gasprofile_init_pnnl (:2772-3350), which under CASENAME 4 (:45) sets the sulf
profile to conmin; they are refused.  Every other value takes the path here
for chem_opt 1 and 2.
"""
from __future__ import annotations

import numpy as np

from woof.core import chem_gas_profile_data as _d

__all__ = ["GASES", "GRAV", "zsigf_from_geopotential", "radm2_background_profile",
           "radm2_profile_on_mass_levels", "radm2_boundary_value"]

_f32 = np.float32

#: Registry/registry.chem:3929 (radm2), :3930 (radm2sorg's first 42 rows).
GASES = ("so2", "sulf", "no2", "no", "o3", "hno3", "h2o2", "ald", "hcho", "op1", "op2",
         "paa", "ora1", "ora2", "nh3", "n2o5", "no3", "pan", "hc3", "hc5", "hc8", "eth",
         "co", "ol2", "olt", "oli", "tol", "xyl", "aco3", "tpan", "hono", "hno4", "ket",
         "gly", "mgly", "dcb", "onit", "csl", "iso", "hcl", "ho", "ho2")

KX, LO = _d.KX, _d.LO
GRAV = _f32(_d.GRAV)
SO4VAPTOAER = _f32(_d.SO4VAPTOAER)
DENS = np.array(_d.DENS, dtype=np.float32)
ZFA = np.array(_d.ZFA, dtype=np.float32)
ZFA_BDY = np.array(_d.ZFA_BDY, dtype=np.float32)
#: (lo, kx), Fortran xl(n, k) at XL[n-1, k-1]; k = 1 is the top of the profile.
XL = np.array(_d.XL, dtype=np.float32)
#: iref(1:len(GASES)) / fracref(...) after setup_gasprofile_map_radm_racm.
IREF = np.array(list(_d.IREF) + [_d.IREF_DEFAULT] * (len(GASES) - len(_d.IREF)), dtype=np.int32)
FRACREF = np.array(list(_d.FRACREF) + [_d.FRACREF_DEFAULT] * (len(GASES) - len(_d.FRACREF)),
                   dtype=np.float32)

_PNNL = 101  # GAS_IC_PNNL / GAS_BC_PNNL, Registry/registry.chem:4162, :4166
_ONE = _f32(1.0)
_HALF = _f32(0.5)
_PPM = _f32(1.0e6)


def _check_options(chem_opt: int, **opts: int) -> None:
    if chem_opt not in (1, 2):
        raise ValueError(f"chem_opt = {chem_opt}: only RADM2 (1) and RADM2SORG (2) are ported")
    for name, value in opts.items():
        if value == _PNNL:
            raise ValueError(
                f"{name} = 101 ({name.split('_')[0].upper()}_{name.split('_')[1].upper()}_PNNL): "
                "gasprofile_init_pnnl replaces the sulf profile by conmin; not ported")


def zsigf_from_geopotential(ph, phb) -> np.ndarray:
    """input_chem_profile's w-level height, ``(ph_1 + phb) / grav`` (:714),
    float32, with module_data_sorgam's grav = 9.80622."""
    return (np.asarray(ph, dtype=np.float32) + np.asarray(phb, dtype=np.float32)) / GRAV


def _interp(z: np.ndarray, zp: np.ndarray, prof: np.ndarray, *, sentinel: bool) -> np.ndarray:
    """Clamped linear interpolation of the columns of ``prof`` (kx, m) from
    increasing heights ``zp`` (kx,) to every height in ``z`` (any shape):
    vinterp_chem's (:600-668, extrapolate = .false.) and bdy_chem_value's
    (:1470-1493) arithmetic, which agree for finite heights.  Returns
    (m,) + z.shape float32."""
    flat = z.reshape(-1)
    n = flat.size
    out = np.empty((prof.shape[1], n), dtype=np.float32)
    below = flat < zp[0]
    above = flat > zp[-1]
    inside = ~(below | above)
    j = np.searchsorted(zp, flat, side="left")
    j = np.clip(j, 0, KX - 1)
    exact = inside & (zp[j] == flat)
    between = inside & ~exact & ~np.isnan(flat)
    out[:, below] = prof[0][:, None]
    out[:, above] = prof[-1][:, None]
    out[:, exact] = prof[j[exact]].T
    kk = j[between] - 1  # zp[kk] < z < zp[kk+1]
    zb = flat[between]
    wgt0 = (zb - zp[kk + 1]) / (zp[kk] - zp[kk + 1])
    a = prof[kk].T
    b = prof[kk + 1].T
    out[:, between] = np.maximum(wgt0 * a + (_ONE - wgt0) * b, _f32(0.0))
    if sentinel:  # data_out = -99999.9 where no branch fires (NaN heights)
        out[:, inside & ~exact & ~between] = _f32(-99999.9)
    return out.reshape((prof.shape[1],) + z.shape)


def radm2_background_profile(z_at_w, *, chem_opt: int = 1, gas_ic_opt: int = 1,
                             include_top: bool = False) -> dict[str, np.ndarray]:
    """real.exe's chem_in_opt = 0 RADM2 initial gas profile.

    ``z_at_w``: w-level heights in metres, shape (nz+1, ...) with any trailing
    shape (WRF (nz+1, ny, nx), hex (nz+1, 1, ncells) or (nz+1, ncells)).  WRF's
    own value is ``(ph_1 + phb) / 9.80622``, :func:`zsigf_from_geopotential`.
    Returns ``{gas: float32 ppmv}`` in Registry order (:data:`GASES`), each of
    shape (nz, ...) on the mass levels, or (nz+1, ...) with
    ``include_top=True`` (WRF's level kde, 0.5*(3*zf(kde) - zf(kde-1))).
    Identical for chem_opt 1 and 2.
    """
    _check_options(chem_opt, gas_ic_opt=gas_ic_opt)
    zf = np.asarray(z_at_w, dtype=np.float32)
    if zf.ndim < 1 or zf.shape[0] < 2:
        raise ValueError(f"z_at_w needs at least two w levels on axis 0, got shape {zf.shape}")
    nz = zf.shape[0] - 1
    zsig = np.empty((nz + 1,) + zf.shape[1:], dtype=np.float32)
    zsig[:nz] = _HALF * (zf[:nz] + zf[1:])                    # :716-718
    zsig[nz] = _HALF * (_f32(3.0) * zf[nz] - zf[nz - 1])      # :719
    if not include_top:
        zsig = zsig[:nz]
    return radm2_profile_on_mass_levels(zsig)


def radm2_profile_on_mass_levels(z) -> dict[str, np.ndarray]:
    """make_chem_profile (:775-937) for chem_opt 1 / 2 on mass-level heights
    ``z`` (metres, any shape): ``{gas: float32 ppmv}``, each the shape of ``z``."""
    zsig = np.asarray(z, dtype=np.float32)
    # :812-813: chprof(k, 2:lo+1) = xl(1:lo, kx-k+1), zprof(k) = 0.5*(zfa(k)+zfa(k+1));
    # :826: rows lo-5:lo+1 (xl 28..34) / dens(k), k bottom-up.
    chprof = XL[:, ::-1].T.copy()                              # (kx, lo)
    chprof[:, LO - 7:] = chprof[:, LO - 7:] / DENS[:, None]
    zprof = _HALF * (ZFA[:-1] + ZFA[1:])
    refs = sorted({int(r) for r in IREF})
    if max(refs) >= LO:  # vinterp_chem never fills stor(:, lo)
        raise AssertionError("a RADM2 row maps to xl(lo), which make_chem_profile leaves unset")
    stor = _interp(zsig, zprof, chprof[:, [r - 1 for r in refs]], sentinel=True)
    at = {r: i for i, r in enumerate(refs)}
    return {name: (FRACREF[n] * stor[at[int(IREF[n])]]) * _PPM   # :845
            for n, name in enumerate(GASES)}


def radm2_boundary_value(z, *, chem_opt: int = 1, gas_bc_opt: int = 1,
                         gas_ic_opt: int = 1) -> dict[str, np.ndarray]:
    """bdy_chem_value (:1415-1504): the RADM2 inflow value flow_dep_bdy_chem
    puts in specified-zone inflow cells when have_bcs_chem = .false.

    ``z``: mass-level heights in metres (WRF's grid%z), any shape.  Returns
    ``{gas: float32 ppmv}`` in Registry order, each the shape of ``z``.  For
    chem_opt = 2 sulf carries the (1. - so4vaptoaer) factor.  gas_ic_opt is
    checked too: wrf.exe's chemics_init runs gasprofile_init_pnnl when either
    is 101 (chem/chemics_init.F:661-664).
    """
    _check_options(chem_opt, gas_bc_opt=gas_bc_opt, gas_ic_opt=gas_ic_opt)
    zz = np.asarray(z, dtype=np.float32)
    if np.isnan(zz).any():  # bdy_chem_value leaves stor undefined
        raise ValueError("NaN height")
    k = np.arange(1, KX + 1)
    zprof = _HALF * (ZFA_BDY[:-1] + ZFA_BDY[1:])               # :1462
    refs = sorted({int(r) for r in IREF})
    cols = []
    for r in refs:                                             # :1463-1467
        row = XL[r - 1, KX - k]
        cols.append(row if r < LO - 6 else row / DENS[KX - k])
    stor = _interp(zz, zprof, np.stack(cols, axis=1), sentinel=False)
    at = {r: i for i, r in enumerate(refs)}
    out = {}
    for n, name in enumerate(GASES):
        v = (FRACREF[n] * stor[at[int(IREF[n])]]) * _PPM       # :1496
        if name == "sulf" and chem_opt == 2:                   # :1499-1501
            v = v * (_ONE - SO4VAPTOAER)
        out[name] = v
    return out
