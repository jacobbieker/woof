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

RADM2SORG aerosol lateral boundary (chem_opt = 2, have_bcs_chem = .false.):
flow_dep_bdy_chem's i_bdy_method = 2 for p_so4aj <= ic <= p_corn (:1651-1652),
the 38 rows of woof.core.chem_sorgam.AEROSOLS.  Each inflow cell gets
bdy_chem_value_sorgam(chem(i,k,j), z(i,k,j), ic, config_flags, alt(i,k,j),
convfac, g) (:1071-1265; called at :1753, :1823, :1893, :1963 for the
four edges), with
grid%z the mass-level height, grid%alt the inverse dry density (m3 kg-1),
and convfac computed just before each call (:1751-1752, ...) as
    tempfac = (t(i,k,j)+t0)*((p(i,k,j)+pb(i,k,j))/p1000mb)**rcp
    convfac = (p(i,k,j)+pb(i,k,j))/rgasuniv/tempfac
from grid%t_1, grid%p, grid%pb (solve_em.F:2599-2609), t0 = 300, p1000mb =
1.e5, rcp = r_d/cp and module_data_sorgam's rgasuniv = 8.314510: the dry air
molar concentration p/(R T) in mol m-3.  g is unused.  For aer_bc_opt = 1
(AER_BC_DEFAULT) the routine gives (:1124-1261):
    conmin = epsilc = 1.E-16 for every row, except
    nh4aj, nh4ai, no3aj, no3ai: 10.e-5*alt (ug kg-1);
    so4aj / so4ai: chemsulf*convfac*mwso4*splitfac*so4vaptoaer*alt, with
        (1.-splitfac) for so4ai (ug kg-1), chemsulf = fracref*stor*1.E6 the
        RADM2 sulf profile (ppmv) on the zfa_bdy grid, exactly
        bdy_chem_value's sulf for chem_opt = 1;
    nu0 / ac0 / corn: m3/((dgini**3)*es36)*alt (# kg-1), with the 3rd
        moments m3nuc = so4fac*so4ai_conc + conmin*(nh4fac+no3fac+orgfac*9
        +2*anthfac), m3acc likewise with so4aj, m3cor = conmin*(soilfac+
        seasfac+anthfac), and es36 = exp(0.125*log(sgini)**2)**36 built as
        the routine builds it (en1**4, ..., esn16*esn20).
:func:`sorgam_boundary_value`.  aer_bc_opt = 101 (AER_BC_PNNL) calls
sorgam_set_aer_bc_pnnl and is refused; WRF stops on any other value.

gas_ic_opt / gas_bc_opt = 101 (GAS_IC_PNNL / GAS_BC_PNNL) call
gasprofile_init_pnnl (:2772-3350), which under CASENAME 4 (:45) sets the sulf
profile to conmin; they are refused.  Every other value takes the path here
for chem_opt 1 and 2.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from woof.core import chem_gas_profile_data as _d
from woof.core.chem_sorgam import AEROSOLS

__all__ = ["GASES", "AEROSOLS", "GRAV", "zsigf_from_geopotential",
           "radm2_background_profile", "radm2_profile_on_mass_levels",
           "radm2_boundary_value", "sorgam_boundary_value"]

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


def _cube(x: np.float32) -> np.float32:
    """``x**3`` for a default-REAL PARAMETER ``x``: gfortran folds it once,
    correctly rounded to REAL (mpfr_pow_z), not as x*x*x."""
    exact = Fraction(float(x)) ** 3
    lo = np.float32(float(exact))
    for cand in (np.nextafter(lo, np.float32(-np.inf)), np.nextafter(lo, np.float32(np.inf))):
        dc, dl = abs(Fraction(float(cand)) - exact), abs(Fraction(float(lo)) - exact)
        if dc < dl or (dc == dl and int(cand.view(np.uint32)) % 2 == 0):
            lo = cand
    return lo


def _sorgam_constants() -> dict[str, np.float32]:
    """module_data_sorgam's PARAMETERs as gfortran folds them, and the module
    variables bdy_chem_value_sorgam sets (:1197-1255), in float32."""
    c = {}
    # :545-547: REAL*8 pirs from a default-REAL literal (rounded to REAL first);
    # :601 f6dpi = 6.0/pirs in REAL*8, stored REAL; :609 f6dpim9 in REAL.
    pirs = np.float64(np.float32(_d.PIRS))
    f6dpi = np.float32(np.float64(6.0) / pirs)
    f6dpim9 = _f32("1.0E-9") * f6dpi
    fac = {r: f6dpim9 / _f32(v) for r, v in _d.RHO.items()}          # :689-711
    conmin = _f32(_d.EPSILC)                                            # :950
    c.update(pirs=np.float32(pirs), f6dpi=f6dpi, f6dpim9=f6dpim9, conmin=conmin,
             **{f"{r}fac": v for r, v in fac.items()})
    # m3nuc / m3acc / m3cor constant parts (:1191-1193), folded in REAL.
    c["m3fac_na"] = conmin * (fac["nh4"] + fac["no3"] + fac["org"] * _f32(9.0)
                              + _f32(2.0) * fac["anth"])
    c["m3fac_c"] = conmin * (fac["soil"] + fac["seas"] + fac["anth"])
    for m in ("n", "a", "c"):
        c[f"dgini{m}3"] = _cube(_f32(_d.DGINI[m]))
        # :1198-1200 log of a PARAMETER (folded, correctly rounded; no logf is
        # linked); :1202-1204 x**2 = x*x; :1206-1208 expf at run time, here
        # the correctly rounded value, which glibc's expf gives for these three.
        xxlsg = np.float32(np.log(np.float64(_f32(_d.SGINI[m]))))
        l2 = xxlsg * xxlsg
        e1 = np.float32(np.exp(np.float64(_f32(0.125) * l2)))
        e04 = (e1 * e1) * (e1 * e1)    # en1**4: libgcc __powisf2 squares twice
        e08 = e04 * e04
        e16 = e08 * e08
        e20 = e16 * e04
        c[f"xxlsg{m}"], c[f"l2sgini{m}"], c[f"e1{m}"] = xxlsg, l2, e1
        c[f"es36{m}"] = e16 * e20                                        # :1251-1253
    return c


_SORGAM = _sorgam_constants()
_SO4FAC = _SORGAM["so4fac"]
_AER_BC_DEFAULT, _AER_BC_PNNL = 1, 101  # Registry/registry.chem:4168-4169


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
    out = _bdy_gases(np.asarray(z, dtype=np.float32), GASES)
    if chem_opt == 2:                                          # :1499-1501
        out["sulf"] = out["sulf"] * (_ONE - SO4VAPTOAER)
    return out


def _bdy_gases(zz: np.ndarray, names) -> dict[str, np.ndarray]:
    """``fracref(n)*stor(iref(n))*1.E6`` on the zfa_bdy grid for the gases
    ``names``: bdy_chem_value (:1460-1496) and bdy_chem_value_sorgam
    (:1148-1183), which flip, interpolate and scale identically."""
    if np.isnan(zz).any():  # both routines leave stor undefined
        raise ValueError("NaN height")
    k = np.arange(1, KX + 1)
    zprof = _HALF * (ZFA_BDY[:-1] + ZFA_BDY[1:])               # :1462, :1150
    idx = [GASES.index(name) for name in names]
    refs = sorted({int(IREF[n]) for n in idx})
    cols = []
    for r in refs:                                             # :1463-1467
        row = XL[r - 1, KX - k]
        cols.append(row if r < LO - 6 else row / DENS[KX - k])
    stor = _interp(zz, zprof, np.stack(cols, axis=1), sentinel=False)
    at = {r: i for i, r in enumerate(refs)}
    return {GASES[n]: (FRACREF[n] * stor[at[int(IREF[n])]]) * _PPM   # :1496
            for n in idx}


def sorgam_boundary_value(z, alt, convfac, *, aer_bc_opt: int = 1) -> dict[str, np.ndarray]:
    """bdy_chem_value_sorgam (:1071-1265), aer_bc_opt = 1: the RADM2SORG
    aerosol inflow value flow_dep_bdy_chem puts in specified-zone inflow
    cells when have_bcs_chem = .false.

    ``z``: mass-level height (m, WRF's grid%z), any shape; ``alt``: inverse
    dry density (m3 kg-1, grid%alt); ``convfac``: (p+pb)/rgasuniv/T (mol
    m-3), as flow_dep_bdy_chem computes it (:1751-1752).  ``alt`` and
    ``convfac`` broadcast to ``z``'s shape.  Returns ``{aerosol: float32}``
    in :data:`AEROSOLS` order, each the shape of ``z``, in WRF's chem units:
    ug kg-1 dry air for the mass rows, # kg-1 for nu0 / ac0 / corn (the
    routine's ``*alt``); rows the routine does not name are conmin (1.E-16,
    not scaled by alt).
    """
    if aer_bc_opt == _AER_BC_PNNL:
        raise ValueError("aer_bc_opt = 101 (AER_BC_PNNL): sorgam_set_aer_bc_pnnl's "
                         "profile is not ported")
    if aer_bc_opt != _AER_BC_DEFAULT:
        raise ValueError(f"aer_bc_opt = {aer_bc_opt}: WRF stops (bdy_chem_value_sorgam "
                         "-- unable to parse aer_bc_opt); only AER_BC_DEFAULT (1) is ported")
    zz = np.asarray(z, dtype=np.float32)
    alt = np.broadcast_to(np.asarray(alt, dtype=np.float32), zz.shape)
    convfac = np.broadcast_to(np.asarray(convfac, dtype=np.float32), zz.shape)
    c = _SORGAM
    chemsulf = _bdy_gases(zz, ("sulf",))["sulf"]                       # :1183
    split = _f32(_d.SPLITFAC)
    base = chemsulf * convfac * _f32(_d.MWSO4)                         # :1187-1188
    so4aj = (base * split) * SO4VAPTOAER
    so4ai = (base * (_ONE - split)) * SO4VAPTOAER
    m3nuc = _SO4FAC * so4ai + c["m3fac_na"]                            # :1191-1193
    m3acc = _SO4FAC * so4aj + c["m3fac_na"]
    m3cor = np.full(zz.shape, c["m3fac_c"], dtype=np.float32)
    small = _f32(10.e-5) * alt                                         # :1132-1135
    out = {name: np.full(zz.shape, c["conmin"], dtype=np.float32) for name in AEROSOLS}
    for name in ("nh4aj", "nh4ai", "no3aj", "no3ai"):
        out[name] = small.copy()
    out["so4aj"] = so4aj * alt                                         # :1189-1190
    out["so4ai"] = so4ai * alt
    for name, m3, m in (("nu0", m3nuc, "n"), ("ac0", m3acc, "a"), ("corn", m3cor, "c")):
        out[name] = (m3 / (c[f"dgini{m}3"] * c[f"es36{m}"])) * alt      # :1259-1261
    return out
