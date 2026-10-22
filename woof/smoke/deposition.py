"""Removal of smoke particles: dry deposition velocity and settling.

Two WRF v4.7.1 routines, ported word for word, and two small woof-own
pieces that feed and apply them:

* :func:`zhang2001_depvel` -- ``zhang2001_aer_drydepvel_1``
  (``chem/module_aer_drydep.F:1650-1941``), the Zhang, Gong, Padro and
  Barrie (2001, Atmos. Environ. 35, 549-560) size-resolved particle dry
  deposition velocity for a lognormal mode, real(4).  Per cell, for the
  number (moment 0) and mass (moment 3) median diameters
  ``dp = dg exp(m ln^2 sigma)``::

      v_d = v_s + 1 / (R_a + R_s + R_a R_s v_s)
      R_s = 1 / (u* (e0_B E_B + e0_IN E_IN + e0_IM E_IM) R_1)

  with Stokes-Cunningham settling ``v_s`` (times the size-dispersion factor
  ``exp(2 ln^2 sigma)``), Brownian collection ``E_B = Sc^-gamma``,
  impaction ``E_IM = (St/(alpha+St))^2``, interception ``E_IN`` and the
  sticking fraction ``R_1``, the land-use tables of Zhang et al.'s Table 3
  (15 classes, 5 seasons, or the 25 USGS classes mapped onto them), and
  WRF's two variants (``zhang_optaa`` 1 as published, 2 modified).  The
  aerodynamic resistance ``resist_a`` (s m-1) is an input, as in WRF.
* :func:`gocart_settling` -- GOCART's ``settling``
  (``chem/module_gocart_settling.F:206-367``), REAL*8, fed one column at a
  time as ``gocart_settling_driver`` feeds it for dust (``idust = 1``,
  ``module_gocart_settling.F:89-122``): every bin settles through the
  column at its Stokes-Cunningham speed, in up to 12 CFL substeps, the
  bottom level's loss going to the ground (``graset``).  A PM2.5 smoke bin
  is one more bin with its own effective radius and density.
* :func:`aerodynamic_resistance` -- woof-own, no WRF oracle: ``R_a`` from
  u*, the roughness length and the reference height, with an optional
  Obukhov length (log law, Businger-Dyer ``psi_h``), real(4).
* :func:`surface_deposition_tendency` -- woof-own: the lowest layer's
  mixing-ratio tendency ``-v_d q0 / dz0`` and the deposition flux
  ``rho0 v_d q0``, real(4).

Each has a CuPy form (whole-domain arrays in and out, one launch of the
:mod:`woof.smoke._kernels` unit) and a NumPy reference (``_np``) that
repeats the arithmetic operation for operation: real(4) through
``np.float32`` with glibc's ``expf``/``powf``/``logf``
(:mod:`woof.core.noahmp_libm`), REAL*8 through float64 with the C
library's ``exp``/``pow`` (:mod:`woof.core.host_libm`).  The two WRF ports
equal the compiled WRF at zero ULP on every output
(``tests/test_smoke_deposition_wrf471_parity.py``); the woof-own pieces
equal each other at zero ULP and are held to float64 references within a
tolerance (no WRF routine exists to hold them to).
"""
from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np

F = np.float32

# --------------------------------------------------------------------------
# zhang2001_aer_drydepvel_1 tables (module_aer_drydep.F:1745-1810)

#: Schmidt-number exponent gamma, by Zhang land-use class 1-15.
ZHANG_GAMMA = np.array([0.56, 0.58, 0.56, 0.56, 0.56, 0.54, 0.54, 0.54, 0.54,
                        0.54, 0.54, 0.54, 0.50, 0.50, 0.56], dtype=F)
#: Impaction parameter alpha, by class.
ZHANG_ALPHA = np.array([1.0, 0.6, 1.1, 0.8, 0.8, 1.2, 1.2, 50.0, 50.0, 1.3,
                        2.0, 50.0, 100.0, 100.0, 1.5], dtype=F)
#: Collector radius (m), [class, season] (rad_collector(5,15) transposed).
ZHANG_RAD_COLLECTOR = np.array([
    [0.002, 0.002, 0.002, 0.002, 0.002],
    [0.005, 0.005, 0.005, 0.005, 0.005],
    [0.002, 0.002, 0.005, 0.005, 0.002],
    [0.005, 0.005, 0.010, 0.010, 0.005],
    [0.005, 0.005, 0.005, 0.005, 0.005],
    [0.002, 0.002, 0.005, 0.005, 0.002],
    [0.002, 0.002, 0.005, 0.005, 0.002],
    [9.999, 9.999, 9.999, 9.999, 9.999],
    [9.999, 9.999, 9.999, 9.999, 9.999],
    [0.010, 0.010, 0.010, 0.010, 0.010],
    [0.010, 0.010, 0.010, 0.010, 0.010],
    [9.999, 9.999, 9.999, 9.999, 9.999],
    [9.999, 9.999, 9.999, 9.999, 9.999],
    [9.999, 9.999, 9.999, 9.999, 9.999],
    [0.010, 0.010, 0.010, 0.010, 0.010]], dtype=F)
#: USGS land-use class 1-25 -> Zhang class (isurftype_zhang_from_usgs).
ZHANG_FROM_USGS = np.array([15, 7, 7, 7, 7, 10, 6, 6, 6, 6, 4, 3, 2, 1, 5,
                            14, 11, 10, 8, 9, 10, 9, 8, 12, 8], dtype=np.int32)
#: The non-vegetated classes (desert, tundra, ice, inland water, ocean).
ZHANG_SMOOTH = (8, 9, 12, 13, 14)
#: log(5.0), folded by gfortran in real(4), correctly rounded.
_LOG5 = F(float.fromhex("0x1.9c0420p+0"))

#: GOCART's dust dynamic viscosity (phys/module_data_gocart_dust.F:16) and
#: WRF's g (share/module_model_constants.F), both real(4).
GOCART_DYN_VISC = 1.5e-5
GRAVITY = 9.81


class ZhangDepositionVelocity(NamedTuple):
    """zhang2001_aer_drydepvel_1's outputs, float32, per cell."""

    #: Number (moment 0) and mass (moment 3) deposition velocity (m s-1).
    depvel_0: Any
    depvel_3: Any
    #: Surface resistances (s m-1); 1e30 for a skipped moment.
    resist_s0: Any
    resist_s3: Any
    #: Gravitational settling velocities (m s-1).
    vsettl_0: Any
    vsettl_3: Any
    #: The kinematic viscosity (m2 s-1) and mean free path (m) the routine
    #: used: the inputs when positive, else the values it computed.
    airkinvisc: Any
    freepath: Any


class SettlingResult(NamedTuple):
    """GOCART settling of one field: WRF's REAL*8 outputs."""

    #: The settled mixing ratio, REAL*8, the input's shape (ug kg-1).
    tc: Any
    #: Mean ground-level loss rate over the step, per bin and column
    #: (ug kg-1 m s-1; times rho and dt it is the deposited mass).
    graset: Any
    #: The bottom level's settling velocity, per bin and column (m s-1).
    grasetvel: Any


class SurfaceDeposition(NamedTuple):
    """The lowest layer's dry-deposition tendency and flux, float32."""

    #: d q0 / dt (q-units s-1).
    tendency: Any
    #: rho0 v_d q0, downward positive (kg m-2 s-1 per unit q).
    flux: Any


# --------------------------------------------------------------------------
# helpers


def _expf(values) -> np.ndarray:
    """glibc ``expf`` element by element (noahmp_libm transcription)."""
    from woof.core import noahmp_libm as libm

    flat = np.asarray(values, dtype=F).reshape(-1)
    out = np.fromiter((libm.expf(float(v)) for v in flat), dtype=F,
                      count=flat.size)
    return out.reshape(np.shape(values))


def _logf(values) -> np.ndarray:
    from woof.core import noahmp_libm as libm

    flat = np.asarray(values, dtype=F).reshape(-1)
    out = np.fromiter((libm.logf(float(v)) for v in flat), dtype=F,
                      count=flat.size)
    return out.reshape(np.shape(values))


def _powf(base, exponent) -> np.ndarray:
    from woof.core import noahmp_libm as libm

    base = np.asarray(base, dtype=F)
    exponent = np.broadcast_to(np.asarray(exponent, dtype=F), base.shape)
    return np.asarray(libm.powf_array(np.ascontiguousarray(base),
                                      np.ascontiguousarray(exponent)),
                      dtype=F).reshape(base.shape)


def _minf(a, b):
    """gfortran's MIN(a, b) of ordinary reals: b if b < a, else a."""
    return np.where(b < a, b, a)


def _maxf(a, b):
    return np.where(b > a, b, a)


def _cells(*arrays, dtype):
    shape = np.broadcast_shapes(*(np.shape(a) for a in arrays))
    return shape, [np.ascontiguousarray(np.broadcast_to(
        np.asarray(a, dtype=dtype), shape)) for a in arrays]


def _device_cells(cp, shape, arrays, dtype):
    return [cp.ascontiguousarray(cp.broadcast_to(
        cp.asarray(a, dtype=dtype), shape)) for a in arrays]


def _shape_of(*arrays):
    return np.broadcast_shapes(*(tuple(getattr(a, "shape", np.shape(a)))
                                 for a in arrays))


# --------------------------------------------------------------------------
# Zhang et al. (2001): NumPy reference


def _zhang_moment_np(moment, dg_num, lnsig, dens_part, airtemp, ustar,
                     resist_a, isurftype, iseason, iwetsurf, zhang_optaa,
                     e0, airkinvisc, freepath, viscosity, dp_vm):
    gravit, boltzmann, pi = F(9.806), F(1.381e-23), F(3.1415926536)
    dp_mom = dg_num * _expf(F(moment) * lnsig * lnsig)
    dp_mom = _minf(F(100.0e-6), dp_mom)
    tmpa = F(2.0) * freepath / dp_mom
    cuncorrect = F(1.0) + tmpa * (F(1.257) + F(0.4) * _expf(-(F(1.10) / tmpa)))
    dispersion = _expf(F(2.0) * lnsig * lnsig)
    vsettl = (dens_part * (dp_mom * dp_mom) * gravit * cuncorrect
              / (F(18.0) * viscosity)) * dispersion
    diffus_part = boltzmann * airtemp * cuncorrect / (
        (F(3.0) * pi) * viscosity * dp_mom)
    schmidt = airkinvisc / diffus_part
    e_brown = _powf(schmidt, -ZHANG_GAMMA[isurftype - 1])

    smooth = np.isin(isurftype, ZHANG_SMOOTH)
    rc = ZHANG_RAD_COLLECTOR[isurftype - 1, iseason - 1]
    stokes = np.where(smooth, vsettl * ustar * ustar / (gravit * airkinvisc),
                      vsettl * ustar / (gravit * rc))
    if zhang_optaa == 2:
        veg_intcept = F(0.5) * dp_mom / rc
    else:
        ratio = dp_mom / rc
        veg_intcept = F(0.5) * (ratio * ratio)
    e_intcept = np.where(smooth, F(0.0), veg_intcept)
    e_impact = _powf(stokes / (ZHANG_ALPHA[isurftype - 1] + stokes), F(2.0))
    sticky = ((iwetsurf > 0) | (isurftype == 13) | (isurftype == 14)
              | (dp_vm < F(5.0e-6)))
    stickfrac = np.where(sticky, F(1.0), _expf(-np.sqrt(stokes)))
    e0_brown, e0_impact, e0_intcept = e0
    tmpa = ustar * (e0_brown * e_brown + e0_intcept * e_intcept
                    + e0_impact * e_impact) * stickfrac
    resist_sfc = F(1.0) / _maxf(tmpa, F(1.0e-30))
    depvel = vsettl + F(1.0) / (resist_a + resist_sfc
                                + resist_a * resist_sfc * vsettl)
    return depvel, vsettl, resist_sfc


def zhang2001_depvel_np(dg_num, lnsig, dens_part, airtemp, airdens, ustar,
                        resist_a, *, isurftype, iseason=1, iwetsurf=0,
                        modesurftype: int = 1, zhang_optaa: int = 1,
                        airkinvisc=0.0, freepath=0.0, ido_mom0: int = 1,
                        ido_mom3: int = 1) -> ZhangDepositionVelocity:
    """Reference zhang2001_aer_drydepvel_1, float32, every cell at once.

    Per cell (any common broadcast shape): ``dg_num`` number-mode geometric
    mean diameter (m), ``lnsig`` ln of its geometric standard deviation,
    ``dens_part`` particle density (kg m-3), ``airtemp`` (K), ``airdens``
    (kg m-3), ``ustar`` (m s-1), ``resist_a`` aerodynamic resistance
    (s m-1), ``isurftype`` (Zhang class 1-15 when ``modesurftype <= 1``,
    USGS 1-25 when >= 2; anything else is class 8), ``iseason`` (1-5, else
    1), ``iwetsurf`` (> 0 wet), ``airkinvisc`` (m2 s-1) and ``freepath``
    (m), each computed by the routine where <= 0.  ``zhang_optaa`` (1 as
    published, 2 modified), ``modesurftype`` and the moment switches
    ``ido_mom0``/``ido_mom3`` are scalars, as one WRF call has them.  Every
    input is clamped as the routine's "reality checks" clamp it.
    """
    shape = _shape_of(dg_num, lnsig, dens_part, airtemp, airdens, ustar,
                      resist_a, airkinvisc, freepath, isurftype, iseason,
                      iwetsurf)
    (dg_num, lnsig, dens_part, airtemp, airdens, ustar, resist_a, airkinvisc,
     freepath) = (np.ascontiguousarray(np.broadcast_to(
         np.asarray(a, dtype=F), shape)) for a in (
        dg_num, lnsig, dens_part, airtemp, airdens, ustar, resist_a,
        airkinvisc, freepath))
    isurftype, iseason, iwetsurf = (np.broadcast_to(
        np.asarray(a, dtype=np.int32), shape) for a in (
        isurftype, iseason, iwetsurf))
    with np.errstate(all="ignore"):
        airtemp = _maxf(F(50.0), _minf(F(350.0), airtemp))
        airdens = _maxf(F(0.01), _minf(F(5.0), airdens))
        dg_num = _maxf(F(1.0e-9), _minf(F(1.0e-2), dg_num))
        dens_part = _maxf(F(0.1e3), _minf(F(50.0e3), dens_part))
        lnsig = _maxf(F(0.0), _minf(_LOG5, lnsig))
        resist_a = _maxf(F(1.0e-2), _minf(F(1.0e6), resist_a))
        ustar = _maxf(F(1.0e-6), _minf(F(1.0e2), ustar))

        if modesurftype <= 1:
            stype = np.where((isurftype >= 1) & (isurftype <= 15), isurftype,
                             8)
        else:
            inside = (isurftype >= 1) & (isurftype <= 25)
            stype = np.where(inside, ZHANG_FROM_USGS[
                np.clip(isurftype, 1, 25) - 1], 8)
        stype = stype.astype(np.int32)
        season = np.where((iseason >= 1) & (iseason <= 5), iseason,
                          1).astype(np.int32)
        if zhang_optaa == 2:
            e0 = (F(1.0), F(1.0), F(4.0))
        else:
            e0 = (F(3.0), F(3.0), F(3.0))

        freepath = np.where(freepath <= F(0.0),
                            F(6.6328e-8) * (F(1.2255) / airdens), freepath)
        computed = airkinvisc <= F(0.0)
        visc_c = F(1.458e-6) * airtemp * np.sqrt(airtemp) / (airtemp
                                                            + F(110.4))
        viscosity = np.where(computed, visc_c, airkinvisc * airdens)
        airkinvisc = np.where(computed, visc_c / airdens, airkinvisc)
        dp_vm = dg_num * _expf(F(1.5) * lnsig * lnsig)

        outs = []
        for moment, ido in ((0, ido_mom0), (3, ido_mom3)):
            if ido <= 0:
                outs.append((np.zeros(shape, F), np.zeros(shape, F),
                             np.full(shape, F(1.0e30))))
                continue
            outs.append(_zhang_moment_np(
                moment, dg_num, lnsig, dens_part, airtemp, ustar, resist_a,
                stype, season, iwetsurf, zhang_optaa, e0, airkinvisc,
                freepath, viscosity, dp_vm))
    (d0, v0, r0), (d3, v3, r3) = outs
    return ZhangDepositionVelocity(
        *(np.asarray(a, dtype=F).reshape(shape)
          for a in (d0, d3, r0, r3, v0, v3, airkinvisc, freepath)))


def zhang2001_depvel(dg_num, lnsig, dens_part, airtemp, airdens, ustar,
                     resist_a, *, isurftype, iseason=1, iwetsurf=0,
                     modesurftype: int = 1, zhang_optaa: int = 1,
                     airkinvisc=0.0, freepath=0.0, ido_mom0: int = 1,
                     ido_mom3: int = 1) -> ZhangDepositionVelocity:
    """:func:`zhang2001_depvel_np` on the device: one thread per cell.

    Same contract; inputs are CuPy or NumPy arrays (or scalars) broadcast
    to one shape, outputs CuPy float32 arrays of that shape.
    """
    import cupy as cp

    from woof.smoke import _kernels

    shape = _shape_of(dg_num, lnsig, dens_part, airtemp, airdens, ustar,
                      resist_a, isurftype, iseason, iwetsurf, airkinvisc,
                      freepath)
    floats = _device_cells(cp, shape, (dg_num, lnsig, dens_part, airtemp,
                                       airdens, ustar, resist_a), cp.float32)
    ints = _device_cells(cp, shape, (iseason, isurftype, iwetsurf), cp.int32)
    kinv, path = (cp.array(a, copy=True) for a in _device_cells(
        cp, shape, (airkinvisc, freepath), cp.float32))
    outs = [cp.empty(shape, dtype=cp.float32) for _ in range(6)]
    n = int(np.prod(shape, dtype=np.int64))
    if n:
        _kernels.kernel("smoke_deposition_zhang2001")(
            _kernels.grid(n), (_kernels.BLOCK,),
            (*floats, *ints, kinv, path, *outs, np.int32(zhang_optaa),
             np.int32(modesurftype), np.int32(ido_mom0), np.int32(ido_mom3),
             np.int32(n)))
    d0, d3, r0, r3, v0, v3 = outs
    return ZhangDepositionVelocity(d0, d3, r0, r3, v0, v3, kinv, path)


# --------------------------------------------------------------------------
# GOCART settling


def _settling_inputs(chem, t_phy, p_phy, dz8w, rho_phy, reff, den, xp):
    chem = xp.asarray(chem, dtype=xp.float32)
    single = chem.ndim == 3
    if single:
        chem = chem[None]
    if chem.ndim != 4:
        raise ValueError(f"chem must be (nz, ny, nx) or (nbin, nz, ny, nx), "
                         f"got {chem.shape}")
    nbin, nz = chem.shape[0], chem.shape[1]
    cols = chem.shape[2:]
    fields = [xp.ascontiguousarray(xp.broadcast_to(
        xp.asarray(a, dtype=xp.float32), (nz,) + cols))
        for a in (t_phy, p_phy, dz8w, rho_phy)]
    reff = np.broadcast_to(np.asarray(reff, dtype=np.float64), (nbin,)).copy()
    den = np.broadcast_to(np.asarray(den, dtype=np.float64), (nbin,)).copy()
    return single, xp.ascontiguousarray(chem), fields, reff, den, cols


def _int_x86(x: np.ndarray) -> np.ndarray:
    """gfortran INT of a REAL*8 on x86-64: trunc, INT_MIN out of range."""
    inside = (x > -2147483649.0) & (x < 2147483648.0)
    safe = np.where(inside, x, 0.0)
    return np.where(inside, np.trunc(safe), -2147483648.0).astype(np.int64)


def gocart_settling_np(chem, t_phy, p_phy, dz8w, rho_phy, dt, reff, den, *,
                       g: float = GRAVITY, dyn_visc: float = GOCART_DYN_VISC,
                       uoc: int = 0) -> SettlingResult:
    """Reference GOCART settling of one step, REAL*8 as WRF computes it.

    ``chem`` is the tracer mixing ratio (ug kg-1, real(4)) on WRF's mass
    levels in model order, ``(nz, ny, nx)`` for one bin or
    ``(nbin, nz, ny, nx)``; ``t_phy`` (K), ``p_phy`` (Pa), ``dz8w`` (layer
    depth, m) and ``rho_phy`` (kg m-3) are the real(4) ``(nz, ny, nx)``
    fields gocart_settling_driver reads; ``dt`` (s, real(4); the routine
    uses ``INT(dt)`` whole seconds); ``reff`` (m) and ``den`` (kg m-3) the
    REAL*8 effective radius and density of each bin; ``uoc = 1`` is the UoC
    dust schemes' 2650 kg m-3 override of bin 1.

    The driver hands the routine ``p_mid = .01*p_phy`` (hPa, a real(4)
    product) and ``delz = dz8w`` with the level order reversed (top first);
    the routine reads both at ``l2 = lmx - l + 1``, which is model level
    ``l`` again, so here every field is read at the level being settled.
    Columns are independent; each one's substep count comes from its own
    thinnest layer.
    """
    from woof.core import host_libm

    single, chem, (t, p, dz, rho), reff, den, cols = _settling_inputs(
        chem, t_phy, p_phy, dz8w, rho_phy, reff, den, np)
    nbin, nz = chem.shape[:2]
    ncol = int(np.prod(cols, dtype=np.int64))
    chem = chem.reshape(nbin, nz, ncol)
    t, p, dz, rho = (a.reshape(nz, ncol) for a in (t, p, dz, rho))
    tc = chem.astype(np.float64)
    tc = np.where(tc < 0.0, 1.0e-32, tc)
    dzmin = dz.astype(np.float64).min(axis=0)
    g0, visc, dt32 = F(g), F(dyn_visc), F(dt)
    c49 = np.float64(F(F(4.0) / F(9.0)) * g0)
    c29 = np.float64(F(F(2.0) / F(9.0)) * g0)
    ntdt = int(dt32)
    tmp = t.astype(np.float64)
    p_mid = (F(0.01) * p).astype(np.float64)
    delz = dz.astype(np.float64)
    airden = rho.astype(np.float64)
    c_stokes = (np.float64(F(1.458e-6)) * host_libm.power(tmp, 1.5)
                / (tmp + np.float64(F(110.4))))
    free_path = np.float64(F(1.1e-3)) / p_mid / np.sqrt(tmp)
    graset = np.zeros((nbin, ncol))
    grasetvel = np.zeros((nbin, ncol))
    with np.errstate(all="ignore"):
        for b in range(nbin):
            dens = 2650.0 if (uoc == 1 and b == 0) else float(den[b])
            rwet = float(reff[b]) * 1.0
            vsettl = c49 * dens * (rwet * rwet) / np.float64(visc)
            dtmax = dzmin / vsettl
            ndt = np.clip(_int_x86(np.float64(ntdt) / dtmax), 1, 12)
            dt_settl = (F(ntdt) / ndt.astype(F)).astype(np.float64)
            c_cun = 1.0 + free_path / rwet * (
                np.float64(F(1.257)) + np.float64(F(0.4)) * host_libm.exp(
                    -(np.float64(F(1.1)) * rwet / free_path)))
            viscosity = c_stokes / c_cun
            vd_cor = c29 * dens * (rwet * rwet) / viscosity
            vd_wk1 = dt_settl[None, :] * vd_cor / delz
            col = tc[b]
            for step in range(1, int(ndt.max()) + 1):
                live = step <= ndt
                transfer = np.zeros(ncol)
                for k in range(nz - 1, -1, -1):
                    temp_tc = col[k].copy()
                    new = col[k] * (1.0 - vd_wk1[k]) + transfer
                    col[k] = np.where(live, new, col[k])
                    if k == 0:
                        graset[b] = np.where(
                            live, graset[b] + vd_cor[0] * temp_tc / ndt,
                            graset[b])
                        grasetvel[b] = np.where(live, vd_cor[0],
                                                grasetvel[b])
                    else:
                        transfer = (temp_tc * vd_wk1[k]) * (
                            (delz[k] * airden[k])
                            / (delz[k - 1] * airden[k - 1]))
    tc = tc.reshape((nbin, nz) + cols)
    graset = graset.reshape((nbin,) + cols)
    grasetvel = grasetvel.reshape((nbin,) + cols)
    if single:
        return SettlingResult(tc[0], graset[0], grasetvel[0])
    return SettlingResult(tc, graset, grasetvel)


def gocart_settling(chem, t_phy, p_phy, dz8w, rho_phy, dt, reff, den, *,
                    g: float = GRAVITY, dyn_visc: float = GOCART_DYN_VISC,
                    uoc: int = 0) -> SettlingResult:
    """:func:`gocart_settling_np` on the device: one thread per column.

    Same contract; CuPy (or NumPy) inputs, CuPy float64 outputs.
    """
    import cupy as cp

    from woof.smoke import _kernels

    single, chem, (t, p, dz, rho), reff, den, cols = _settling_inputs(
        chem, t_phy, p_phy, dz8w, rho_phy, reff, den, cp)
    nbin, nz = chem.shape[:2]
    ncol = int(np.prod(cols, dtype=np.int64))
    tc = cp.empty((nbin, nz) + cols, dtype=cp.float64)
    graset = cp.empty((nbin,) + cols, dtype=cp.float64)
    grasetvel = cp.empty((nbin,) + cols, dtype=cp.float64)
    if ncol:
        _kernels.kernel("smoke_deposition_gocart_settling")(
            _kernels.grid(ncol), (_kernels.BLOCK,),
            (chem, t, p, dz, rho, cp.asarray(reff), cp.asarray(den), tc,
             graset, grasetvel, F(dt), F(g), F(dyn_visc), np.int32(uoc),
             np.int32(nbin), np.int32(nz), np.int32(ncol)))
    if single:
        return SettlingResult(tc[0], graset[0], grasetvel[0])
    return SettlingResult(tc, graset, grasetvel)


# --------------------------------------------------------------------------
# woof-own: aerodynamic resistance


def _obukhov(obukhov_length):
    """``inf`` (neutral) for ``None``; an Obukhov length of 0 is refused."""
    if obukhov_length is None:
        return np.inf
    host = (obukhov_length.get() if hasattr(obukhov_length, "get")
            and not isinstance(obukhov_length, np.ndarray)
            else np.asarray(obukhov_length))
    if np.any(host == 0) or np.any(np.isnan(host)):
        raise ValueError("an Obukhov length must be non-zero and not NaN; "
                         "pass None (or inf) for neutral")
    return obukhov_length


def aerodynamic_resistance_np(ustar, z0, z_ref,
                              obukhov_length=None) -> np.ndarray:
    """Reference aerodynamic resistance R_a (s m-1), float32.

    Woof-own, not a WRF routine: no oracle holds it.  It is the device
    form's word-for-word twin (zero ULP) and is held to the float64
    :func:`aerodynamic_resistance_f64` within a tolerance::

        zeta  = min(z_ref / L, 1)                 (L infinite: neutral;
                                                   held at 1, the top of
                                                   the fitted stable range)
        psi_h = -5 zeta                           zeta >= 0
              = 2 ln((1 + sqrt(1 - 16 zeta)) / 2) zeta <  0
        R_a   = max((ln(z_ref / z0) - psi_h) / (0.4 u*), 0)

    ``ustar`` (m s-1), ``z0`` roughness length (m), ``z_ref`` the height
    the resistance runs to (m, normally the lowest mass level) and
    ``obukhov_length`` (m, ``None`` neutral), broadcast together.
    """
    length = _obukhov(obukhov_length)
    shape, (us, z0, zr, ob) = _cells(ustar, z0, z_ref, length, dtype=F)
    with np.errstate(all="ignore"):
        zeta = np.minimum(zr / ob, F(1.0))
        unstable = F(2.0) * _logf((F(1.0) + np.sqrt(
            F(1.0) - F(16.0) * np.minimum(zeta, F(0.0)))) * F(0.5))
        psi = np.where(zeta >= F(0.0), F(-5.0) * zeta, unstable)
        ra = (_logf(zr / z0) - psi) / (F(0.4) * us)
    return np.where(ra > F(0.0), ra, F(0.0)).astype(F).reshape(shape)


def aerodynamic_resistance_f64(ustar, z0, z_ref,
                               obukhov_length=None) -> np.ndarray:
    """The same formula in float64 (the woof-own form's tolerance check)."""
    length = _obukhov(obukhov_length)
    shape, (us, z0, zr, ob) = _cells(ustar, z0, z_ref, length,
                                     dtype=np.float64)
    out = np.empty(shape)
    for i in np.ndindex(shape):
        zeta = min(zr[i] / ob[i], 1.0)
        if zeta >= 0.0:
            psi = -5.0 * zeta
        else:
            psi = 2.0 * math.log((1.0 + math.sqrt(1.0 - 16.0 * zeta)) / 2.0)
        out[i] = max((math.log(zr[i] / z0[i]) - psi) / (0.4 * us[i]), 0.0)
    return out


def aerodynamic_resistance(ustar, z0, z_ref, obukhov_length=None):
    """:func:`aerodynamic_resistance_np` on the device, float32."""
    import cupy as cp

    from woof.smoke import _kernels

    length = _obukhov(obukhov_length)
    shape = _shape_of(ustar, z0, z_ref, length)
    us, z0, zr, ob = _device_cells(cp, shape, (ustar, z0, z_ref, length),
                                   cp.float32)
    out = cp.empty(shape, dtype=cp.float32)
    n = int(np.prod(shape, dtype=np.int64))
    if n:
        _kernels.kernel("smoke_deposition_resist_a")(
            _kernels.grid(n), (_kernels.BLOCK,), (us, z0, zr, ob, out,
                                                  np.int32(n)))
    return out


# --------------------------------------------------------------------------
# woof-own: the lowest layer's deposition tendency


def surface_deposition_tendency_np(q0, dz0, v_d,
                                   rho0=1.0) -> SurfaceDeposition:
    """Reference lowest-layer dry-deposition tendency and flux, float32.

    Woof-own (WRF-Chem applies its deposition velocities as the lower
    boundary of its vertical mixing instead).  ``q0`` the lowest layer's
    mixing ratio, ``dz0`` its depth (m), ``v_d`` the deposition velocity
    (m s-1, e.g. :func:`zhang2001_depvel`'s ``depvel_3``) and ``rho0`` the
    air density (kg m-3)::

        tendency = -(v_d q0) / dz0        flux = (rho0 v_d) q0

    A forward step with it is stable for ``v_d dt <= dz0``; over one hour
    a well-mixed layer with no source keeps ``exp(-v_d t / dz0)`` of its
    tracer.  Equal to the device form at zero ULP and held to a float64
    evaluation of the same two lines within a tolerance.
    """
    shape, (q0, dz0, v_d, rho0) = _cells(q0, dz0, v_d, rho0, dtype=F)
    return SurfaceDeposition((-(v_d * q0) / dz0).astype(F),
                             ((rho0 * v_d) * q0).astype(F))


def surface_deposition_tendency(q0, dz0, v_d, rho0=1.0) -> SurfaceDeposition:
    """:func:`surface_deposition_tendency_np` on the device."""
    import cupy as cp

    from woof.smoke import _kernels

    shape = _shape_of(q0, dz0, v_d, rho0)
    arrays = _device_cells(cp, shape, (q0, dz0, v_d, rho0), cp.float32)
    tendency = cp.empty(shape, dtype=cp.float32)
    flux = cp.empty(shape, dtype=cp.float32)
    n = int(np.prod(shape, dtype=np.int64))
    if n:
        _kernels.kernel("smoke_deposition_surface_tendency")(
            _kernels.grid(n), (_kernels.BLOCK,), (*arrays, tendency, flux,
                                                  np.int32(n)))
    return SurfaceDeposition(tendency, flux)
