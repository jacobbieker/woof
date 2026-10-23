"""The Freitas et al. (2007) 1-D plume-rise model of WRF-Chem v4.7.1.

A port of ``chem/module_chem_plumerise_scalar.F`` -- ``plumerise`` and
everything it calls: ``get_env_condition`` (the sounding interpolated onto
the plume model's own 100 m grid, ``set_grid``), ``get_fire_properties``
(the biome's heat flux, the fire's radius and the 53-minute burn with its
4-minute ramp), ``makeplume`` (the time-dependent 1-D cloud model of a
buoyant plume: vertical advection of momentum and scalars, entrainment,
the horizontal-wind drag of Freitas et al. 2010, Kessler-type warm and ice
microphysics, Rayleigh damping, run until the plume top settles) and the
``k1``/``k2`` part of ``set_flam_vert`` that places the flaming emission.

THE OUTPUT, AS WRF FORMS IT.  For one column, with the four aggregated
biomes (1 tropical forest, 2 boreal forest, 3 savanna, 4 grassland) each
carrying a fire size ``firesize`` (m2) and a flaming fraction
``mean_fct``::

    eburn_out(1) = eburn_in
    eburn_out(k) += mean_fct(b) * eburn_in / (zm(k2+1) - zm(k1)),
                    k = k1(b) .. k2(b), for every biome b with
                    mean_fct(b) >= 1e-6

where ``k1(b)``/``k2(b)`` are the model levels just above the plume tops of
the biome's minimum and maximum heat flux (``ztopmax``; grassland has one
heat flux and injects from the ground).  ``eburn_in`` is the smouldering
emission and ``eburn_out`` is in its units per metre, as
``chem/module_plumerise1.F`` hands them on (it multiplies level ``k >= 2``
by its depth).

THE INPUTS, AS ``plumerise`` TAKES THEM (``chem/module_plumerise1.F``
:240-252 forms them from the WRF state): on the column's ``m1`` mass levels,
``u``, ``v`` (m s-1), ``theta`` (K), the Exner function ``pi = cp *
(p/p1000mb)**rcp`` (J kg-1 K-1), the vapour mixing ratio ``qv`` (kg kg-1),
the mass-level heights ``zt`` and the w-level heights ``zm`` above the
ground (``zm(1) = 0``; m).  Columns are skipped as the driver skips them
(lines 213-227): a flaming-fraction sum or a fire-size sum below 1e-6, or a
zero ``eburn_in``, leaves ``eburn_out = (eburn_in, 0, ..., 0)``.

PRECISION AND ORDER.  Everything is real(4), as WRF compiles it.  The CuPy
path (:func:`plumerise`) is one NVRTC kernel launch
(``woof/core/kernels/smoke_plumerise.cu``, :mod:`woof.smoke._kernels`):
one thread block per burning column, the plume model's module arrays in the
block's shared memory, its threads sharing the levels wherever WRF's loop
over levels carries no dependence.  :func:`plumerise_np` is the NumPy
reference: the same statements in the same order on ``np.float32``
(glibc's ``expf`` and ``powf`` from :mod:`woof.core.noahmp_libm`),
vectorised over levels only where WRF's loop carries no dependence.  Both
equal the compiled WRF word for word
(``tests/test_smoke_plumerise_wrf471_parity.py``).

STATE.  WRF keeps the plume model's state in module variables that it
zeroes once per process, so a column inherits scalars (WBAR, DQSDZ) and
levels above the plume grid's top from the column WRF computed before it.
Here every column starts from that first-call state, the state the oracle
runs in (``tools/smoke_wrf471_oracle/PROVENANCE.md``).

``heat_flux`` (kW m-2, ``(2, 4)``: minimum and maximum per biome) defaults
to the table of ``get_fire_properties``; another table runs the same
arithmetic on other data.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

F = np.float32

#: Plume grid levels and minutes of heating (module_zero_plumegen_coms.F:4).
NKP = 200
NTIME = 200
#: get_fire_properties:425-435, kW m-2: (min, max) x (tropical forest,
#: boreal forest, savanna, grassland).
HEAT_FLUX = np.array([[30.0, 30.0, 4.4, 3.3],
                      [80.0, 80.0, 23.0, 3.3]], dtype=np.float32)


def _c(text: str) -> np.float32:
    """A Fortran real(4) literal (each is checked against gfortran's word
    by tests/test_smoke_plumerise_wrf471_parity.py)."""
    return F(float(text))


# share/module_model_constants.F, folded in real(4) as gfortran folds them.
_G_MC = F(9.81)
_R_D = F(287.)
_CP_MC = F(F(F(7.) * _R_D) / F(2.))
_RCP = F(_R_D / _CP_MC)
_CPOR = F(F(1.) / _RCP)
_P00 = F(100000.)

_ONE, _ZERO, _HALF, _TWO = F(1.), F(0.), F(.5), F(2.)
_G2 = F(_G_MC * F(2.))                          # g*2. (get_env_condition)
_P61 = _c(".61")
_QVMIN = _c("1e-8")
_E3 = _c("1.e-3")
_RHOC = _c("3483.8")
_EPS = _c("0.622")
# lbound's local parameters
_LB_G, _LB_R, _LB_CP, _LB_PI = (_c("9.80796"), _c("287.04"), _c("1004.6"),
                                _c("3.14159"))
_LB_GR = F(_LB_G * _LB_R)
_E1 = F(F(1.) / F(3.))
_E2 = F(F(5.) / F(3.))
_HEAT = _c("19.3e6")
# esat_pr
_CI1, _CI2, _CI3 = _c("6.1115"), _c("22.542"), _c("273.48")
_CW1, _CW2, _CW3, _CW4 = (_c("6.1121"), _c("18.729"), _c("257.87"),
                          _c("227.3"))
_TMELT_ES = _c("273.3")
# microphysics
_FRC_EV = F(_c("2.5E3") / _c("1.004"))
_SRC = F(_c("2834.") / _c("1.004"))
_FRC_GL = F(_c("334.") / _c("1.004"))
_HERC = _c("5.44E-4")
_TFREEZE = _c("269.3")
_C2PI = F(F(2.) / _c("3.1416"))
_PI_DYN = _c("3.1416")
_UMGAMAI = F(F(1.) / F(F(1.) + F(.5)))
_F0 = _c("0.75")
_TINY = _c("1.0E-10")

#: Every real(4) literal the port spells, for the word check against
#: gfortran's own conversion.
LITERALS = (".61", "1e-8", "1.e-3", "3483.8", "0.622", "9.80796", "287.04",
            "1004.6", "3.14159", "19.3e6", "6.1115", "22.542", "273.48",
            "6.1121", "18.729", "257.87", "227.3", "273.3", "2.5E3",
            "1.004", "2834.", "334.", "5.44E-4", "269.3", "3.1416", "0.05",
            "0.55", "0.1", "0.0001", "1.0E-10", "1.E6", "0.475", "0.525",
            "1.13", "7.E5", "4.1E6", "0.025", "2.27", "1.E-6", "332.27",
            "0.0052", ".875", "0.146", "0.0366", "0.000001", "0.9", "1.2",
            "9.81", "1004.", "9.8", "1.6", "0.57E-3", "-0.42", ".65", ".2",
            "0.75", "1.5", "1.e-6", "-40.0")


def folded_constants() -> tuple:
    """The constant expressions gfortran folds, as this port forms them, in
    the order of run_plumerise.F90's ``folded``: g*2., cp, rcp, cpor, G*R,
    1./3., 5./3., HEATCOND/CP, HEATSUBL/CP, HEATFUS/CP, 2./3.1416,
    (2./3.1416)*0.5, 1./(1.+gama), F0**(-0.42), 0.0366*ANBASE, MOIST/100."""
    return (_G2, _CP_MC, _RCP, _CPOR, _LB_GR, _E1, _E2, _FRC_EV, _SRC,
            _FRC_GL, _C2PI, F(_C2PI * _HALF), _UMGAMAI,
            _powf(_F0, _c("-0.42")), F(_c("0.0366") * F(100000.)),
            F(F(10) / F(100.)))


def _powf(x, y) -> np.float32:
    from woof.core.noahmp_libm import powf
    return powf(x, y)


def _expf(x) -> np.float32:
    from woof.core.noahmp_libm import expf
    return expf(x)


def _fmax(a, b):
    """Fortran MAX(a, b) as gfortran evaluates it: ``a > b ? a : b``."""
    return a if a > b else b


def _fmin(a, b):
    """Fortran MIN(a, b): ``a < b ? a : b``."""
    return a if a < b else b


def _esat_pr(tem) -> np.float32:
    """ESAT_PR (kPa), lines 2436-2463."""
    temc = F(tem - _TMELT_ES)
    if temc > _c("-40.0"):
        esatm = F(_CW1 * _expf(F(F(F(_CW2 - F(temc / _CW4)) * temc)
                                 / F(temc + _CW3))))
    else:
        esatm = F(_CI1 * _expf(F(F(_CI2 * temc) / F(temc + _CI3))))
    return F(esatm / F(10.))


def _esat_v(tem: np.ndarray) -> np.ndarray:
    return np.array([_esat_pr(x) for x in tem], dtype=np.float32)


def _powf_v(x: np.ndarray, y) -> np.ndarray:
    return np.array([_powf(a, y) for a in x], dtype=np.float32)


class _Plume:
    """module_zero_plumegen_coms in its first-call state, 1-based."""

    ARRAYS = ("w", "t", "qv", "qc", "qh", "qi", "vth", "vti", "rho", "est",
              "qsat", "wc", "wt", "tt", "qvt", "qct", "qht", "qit", "dzm",
              "dzt", "zm", "zt", "vctr1", "vctr2", "vt3dc", "vt3df",
              "vt3dk", "vt3dg", "scr1", "pke", "the", "thve", "te", "pe",
              "qvenv", "visc", "cvi", "radius", "upe", "vpe", "vel_e",
              "vel_p", "rad_p", "vel_t", "rad_t", "ztop_", "heating")

    def __init__(self, heat_flux: np.ndarray):
        for name in self.ARRAYS:
            setattr(self, name, np.zeros(NKP + 2, dtype=np.float32))
        self.heat_flux = heat_flux
        self.wbar = self.dqsdz = self.adiabat = F(0.)
        self.viscosity = self.tstpf = F(0.)
        self.ztop = self.area = self.rsurf = self.alpha = F(0.)
        self.fmoist = self.dt = self.time = self.tdur = F(0.)
        self.mintime = self.mdur = self.maxtime = 0
        self.nm1 = 0
        self.L = 0
        self.set_grid()

    # ---------------------------------------------------------------- grid
    def set_grid(self):
        dz = F(100.)
        zt, zm = self.zt, self.zm
        zt[1] = F(0.)
        zm[1] = F(0.)
        zt[2] = F(zt[1] + F(_HALF * dz))
        zm[2] = F(zm[1] + dz)
        for k in range(3, NKP + 1):
            zt[k] = F(zt[k - 1] + dz)
            zm[k] = F(zm[k - 1] + dz)
        for k in range(1, NKP):
            self.dzm[k] = F(_ONE / F(zt[k + 1] - zt[k]))
        self.dzm[NKP] = self.dzm[NKP - 1]
        for k in range(2, NKP + 1):
            self.dzt[k] = F(_ONE / F(zm[k] - zm[k - 1]))
        self.dzt[1] = F(F(self.dzt[2] * self.dzt[2]) / self.dzt[3])

    # ---------------------------------------------------------- environment
    @staticmethod
    def htint(nzz1, vctra, eleva, nzz2, vctrb, elevb):
        """htint, lines 2395-2434 (vctra/eleva 1-based)."""
        l = 1
        for k in range(1, nzz2 + 1):
            while True:
                if (elevb[k] < eleva[1]) or (elevb[k] >= eleva[l]
                                             and elevb[k] <= eleva[l + 1]):
                    wt = F(F(elevb[k] - eleva[l]) / F(eleva[l + 1] - eleva[l]))
                    vctrb[k] = F(vctra[l] + F(F(vctra[l + 1] - vctra[l]) * wt))
                    break
                elif elevb[k] > eleva[nzz1]:
                    wt = F(F(elevb[k] - eleva[nzz1])
                           / F(eleva[nzz1 - 1] - eleva[nzz1]))
                    vctrb[k] = F(vctra[nzz1] + F(F(vctra[nzz1 - 1]
                                                   - vctra[nzz1]) * wt))
                    break
                l += 1
                if l == nzz1:
                    raise RuntimeError("htint: the plume grid is outside "
                                       "the column (WRF stops 'htint')")

    def get_env_condition(self, col) -> int:
        m1 = col["m1"]
        znz = col["zcon"][m1]
        for k in range(NKP, 0, -1):
            if self.zt[k] < znz:
                break
        else:
            raise RuntimeError("get_env_condition: ' envir stop 12'")
        kmt = min(k, NKP - 1)
        zt = self.zt
        self.htint(m1, col["ucon"], col["zcon"], kmt, self.upe, zt)
        self.htint(m1, col["vcon"], col["zcon"], kmt, self.vpe, zt)
        self.htint(m1, col["thtcon"], col["zcon"], kmt, self.the, zt)
        self.htint(m1, col["rvcon"], col["zcon"], kmt, self.qvenv, zt)
        s = slice(1, kmt + 1)
        qvenv = self.qvenv[s]
        self.qvenv[s] = np.where(qvenv > _QVMIN, qvenv, _QVMIN)
        self.pke[1] = col["picon"][1]
        self.thve[s] = self.the[s] * (_ONE + _P61 * self.qvenv[s])
        for k in range(2, kmt + 1):
            self.pke[k] = F(self.pke[k - 1]
                            - F(F(_G2 * F(zt[k] - zt[k - 1]))
                                / F(self.thve[k] + self.thve[k - 1])))
        self.te[s] = (self.the[s] * self.pke[s]) / _CP_MC
        self.pe[s] = _powf_v(self.pke[s] / _CP_MC, _CPOR) * _P00
        self.vel_e[s] = np.sqrt(self.upe[s] * self.upe[s]
                                + self.vpe[s] * self.vpe[s])
        self.pe[s] = self.pe[s] * _E3
        return kmt

    def get_fire_properties(self, imm, iveg, burnt_area):
        self.area = F(burnt_area)
        heat_flux_w = F(self.heat_flux[imm - 1, iveg - 1] * F(1000.))
        self.mdur = 53
        self.maxtime = self.mdur + 2
        self.alpha = _c("0.05")
        self.maxtime = self.maxtime * 60
        self.rsurf = F(np.sqrt(F(self.area / _c("3.14159"))))
        self.fmoist = F(F(10) / F(100.))
        self.heating[1:NTIME + 1] = _c("0.0001")
        self.tdur = F(F(self.mdur) * F(60.))
        self.heating[1:self.mdur + 1] = F(heat_flux_w * _c("0.55"))
        hinc = F(self.heating[1] / F(4.))
        self.heating[1] = _c("0.1")
        self.heating[2] = hinc
        self.heating[3] = F(F(2.) * hinc)
        self.heating[4] = F(F(3.) * hinc)

    # ------------------------------------------------------- microphysics
    def waterbal(self):
        L = self.L
        if self.qc[L] <= _TINY:
            self.qc[L] = F(0.)
        if self.qh[L] <= _TINY:
            self.qh[L] = F(0.)
        if self.qi[L] <= _TINY:
            self.qi[L] = F(0.)
        self.evaporate()
        self.sublimate()
        self.glaciate()
        self.melt()
        self.convert()

    def evaporate(self):
        L = self.L
        qsat, qv, qc, qh, qi, t = (self.qsat, self.qv, self.qc, self.qh,
                                   self.qi, self.t)
        dt = self.dt
        sd = F(qsat[L] - qv[L])
        if sd == F(0.0):
            return
        evrate = F(abs(F(self.wbar * self.dqsdz)))
        evap = F(evrate * dt)
        if sd <= F(0.0):
            if evap >= F(abs(sd)):
                qc[L] = F(qc[L] - sd)
                qv[L] = qsat[L]
                t[L] = F(t[L] - F(sd * _FRC_EV))
                return
            qc[L] = F(qc[L] + evap)
            qv[L] = F(qv[L] - evap)
            t[L] = F(t[L] + F(evap * _FRC_EV))
            return
        if evap <= qc[L]:
            if sd <= evap:
                qc[L] = F(qc[L] - sd)
                qv[L] = qsat[L]
                t[L] = F(t[L] - F(sd * _FRC_EV))
                return
            sd = F(sd - evap)
            qv[L] = F(qv[L] + evap)
            t[L] = F(t[L] - F(evap * _FRC_EV))
            qc[L] = F(qc[L] - evap)
        else:
            if sd <= qc[L]:
                qv[L] = qsat[L]
                qc[L] = F(qc[L] - sd)
                t[L] = F(t[L] - F(sd * _FRC_EV))
                return
            sd = F(sd - qc[L])
            qv[L] = F(qv[L] + qc[L])
            t[L] = F(t[L] - F(qc[L] * _FRC_EV))
            qc[L] = F(0.0)
        if qh[L] > _c("1.E-10"):
            rho = self.rho[L]
            quant = F(F(F(qsat[L] - qc[L]) - qv[L]) * rho)
            evhdt = F(F(F(F(dt * _HERC) * quant)
                        * _powf(F(qh[L] * rho), _c(".65"))) / rho)
            if evhdt <= qh[L]:
                if sd <= evhdt:
                    qh[L] = F(qh[L] - sd)
                    qv[L] = qsat[L]
                    t[L] = F(t[L] - F(sd * _FRC_EV))
                    return
                sd = F(sd - evhdt)
                qv[L] = F(qv[L] + evhdt)
                t[L] = F(t[L] - F(evhdt * _FRC_EV))
                qh[L] = F(qh[L] - evhdt)
            else:
                if sd <= qh[L]:
                    qv[L] = qsat[L]
                    qh[L] = F(qh[L] - sd)
                    t[L] = F(t[L] - F(sd * _FRC_EV))
                    return
                sd = F(sd - qh[L])
                qv[L] = F(qv[L] + qh[L])
                t[L] = F(t[L] - F(qh[L] * _FRC_EV))
                qh[L] = F(0.0)
        if qi[L] <= _c("1.E-10"):
            return
        dividend = F(F(F(_powf(F(_c("1.E6") / self.rho[L]), _c("0.475"))
                         * F(F(sd / qsat[L]) - F(1.)))
                       * _powf(qi[L], _c("0.525"))) * _c("1.13"))
        divisor = F(_c("7.E5") + F(_c("4.1E6") / F(F(10.) * self.est[L])))
        devidt = F(-F(F(self.cvi[L] * dividend) / divisor))
        evidt = F(devidt * dt)
        if evidt <= qi[L]:
            if sd <= evidt:
                qi[L] = F(qi[L] - sd)
                qv[L] = qsat[L]
                t[L] = F(t[L] - F(sd * _SRC))
                return
            sd = F(sd - evidt)
            qv[L] = F(qv[L] + evidt)
            t[L] = F(t[L] - F(evidt * _SRC))
            qi[L] = F(qi[L] - evidt)
        else:
            if sd <= qi[L]:
                qv[L] = qsat[L]
                qi[L] = F(qi[L] - sd)
                t[L] = F(t[L] - F(sd * _SRC))
                return
            sd = F(sd - qi[L])
            qv[L] = F(qv[L] + qi[L])
            t[L] = F(t[L] - F(qi[L] * _SRC))
            qi[L] = F(0.0)

    def sublimate(self):
        L = self.L
        if self.t[L] > _TFREEZE:
            return
        if self.qv[L] <= self.qsat[L]:
            return
        dividend = F(F(F(_powf(F(_c("1.E6") / self.rho[L]), _c("0.475"))
                         * F(F(self.qv[L] / self.qsat[L]) - F(1.)))
                       * _powf(self.qi[L], _c("0.525"))) * _c("1.13"))
        divisor = F(_c("7.E5") + F(_c("4.1E6") / F(F(10.) * self.est[L])))
        dtsubh = F(abs(F(dividend / divisor)))
        subl = F(dtsubh * self.dt)
        if subl < self.qv[L]:
            self.qv[L] = F(self.qv[L] - subl)
            self.qi[L] = F(self.qi[L] + subl)
            self.t[L] = F(self.t[L] + F(subl * _SRC))
            return
        # WRF's own: the ice becomes the vapour, it is not added to it.
        self.qi[L] = self.qv[L]
        self.t[L] = F(self.t[L] + F(self.qv[L] * _SRC))
        self.qv[L] = F(0.0)

    def glaciate(self):
        L = self.L
        if self.qh[L] <= F(0.):
            return
        if self.qv[L] < self.qsat[L]:
            return
        if self.t[L] > _TFREEZE:
            return
        dfrzh = F(F(self.dt * _c("0.025")) * self.qh[L])
        if dfrzh < self.qh[L]:
            self.qi[L] = F(self.qi[L] + dfrzh)
            self.qh[L] = F(self.qh[L] - dfrzh)
            self.t[L] = F(self.t[L] + F(_FRC_GL * dfrzh))
            return
        self.qi[L] = F(self.qi[L] + self.qh[L])
        self.t[L] = F(self.t[L] + F(_FRC_GL * self.qh[L]))
        self.qh[L] = F(0.0)

    def melt(self):
        L = self.L
        if self.qi[L] <= F(0.0):
            return
        tmelt = F(273.)
        if self.t[L] < tmelt:
            return
        rho = self.rho[L]
        # F0**(-0.42) is a constant expression gfortran folds; its word is
        # powf's (checked by the literal test).
        f0pow = _powf(_F0, _c("-0.42"))
        dtmelt = F(F(F(F(F(self.dt * F(_c("2.27") / rho)) * self.cvi[L])
                       * F(self.t[L] - tmelt))
                     * _powf(F(F(rho * self.qi[L]) * _c("1.E-6")),
                             _c("0.525"))) * f0pow)
        frc = _c("332.27")
        if dtmelt < self.qi[L]:
            self.qh[L] = F(self.qh[L] + dtmelt)
            self.qi[L] = F(self.qi[L] - dtmelt)
            self.t[L] = F(self.t[L] - F(frc * dtmelt))
            return
        self.qh[L] = F(self.qh[L] + self.qi[L])
        self.t[L] = F(self.t[L] - F(frc * self.qi[L]))
        self.qi[L] = F(0.0)

    def convert(self):
        L = self.L
        if self.t[L] <= _TFREEZE:
            return
        if self.qc[L] == F(0.):
            return
        accrete = F(0.)
        rho = self.rho[L]
        q = F(rho * self.qc[L])
        h = F(rho * self.qh[L])
        if self.qh[L] > F(0.):
            accrete = F(F(_c("0.0052") * q) * _powf(h, _c(".875")))
        bdisp = _c("0.146")
        con = F(F(F(F(q * q) * q) * bdisp)
                / F(F(60.) * F(F(F(F(5.) * q) * bdisp)
                               + F(_c("0.0366") * F(100000.)))))
        total = F(F(F(con + accrete) * self.dt) / rho)
        if total < self.qc[L]:
            self.qc[L] = F(self.qc[L] - total)
            self.qh[L] = F(self.qh[L] + total)
            return
        self.qh[L] = F(self.qh[L] + self.qc[L])
        self.qc[L] = F(0.0)

    # ---------------------------------------------------- boundary, start
    def lbound(self):
        self.qh[1] = self.qh[2]
        self.qi[1] = self.qi[2]
        self.qc[1] = F(0.)
        if self.time > self.tdur:
            eflux = _c("0.000001")
            water = F(0.)
        else:
            eflux = self.heating[self.mintime]
            water = F(F(F(eflux * F(self.dt / _HEAT))
                        * F(F(.5) + self.fmoist)) / _c("0.55"))
            water = F(water * F(1000.))
        pres = F(self.pe[1] * F(1000.))
        c1 = F(F(5.) / F(F(6.) * self.alpha))
        c2 = F(_c("0.9") * self.alpha)
        f = F(eflux / F(F(pres * _LB_CP) * _LB_PI))
        f = F(F(_LB_GR * f) * self.area)
        zv = F(c1 * self.rsurf)
        c2f_e1 = _powf(F(c2 * f), _E1)
        self.w[1] = F(F(c1 * c2f_e1) / _powf(zv, _E1))
        denscor = F(F(F(F(c1 * f) / _LB_G) / c2f_e1) / _powf(zv, _E2))
        self.t[1] = F(self.te[1] / F(F(1.) - denscor))
        self.wc[1] = self.w[1]
        self.vel_p[1] = F(0.)
        self.rad_p[1] = self.rsurf
        self.vth[1] = F(-4.)
        self.vti[1] = F(-3.)
        self.visc[1] = self.viscosity
        self.rho[1] = F(F(_RHOC * self.pe[1]) / self.t[1])
        xwater = F(water / F(F(self.w[1] * self.dt) * self.rho[1]))
        self.qv[1] = F(xwater + self.qvenv[1])
        es = _esat_pr(self.t[1])
        self.est[1] = es
        self.qsat[1] = F(F(_EPS * es) / F(self.pe[1] - es))
        if self.qv[1] > self.qsat[1]:
            self.qc[1] = F(F(self.qv[1] - self.qsat[1]) + self.qc[1])
            self.qv[1] = self.qsat[1]
        self.L = 1
        self.waterbal()

    def initial(self, kmt):
        n = kmt
        s = slice(1, n + 1)
        self.w[s] = 0.
        self.t[s] = self.te[s]
        self.wc[s] = 0.
        self.wt[s] = 0.
        self.qv[s] = self.qvenv[s]
        self.vth[s] = 0.
        self.vti[s] = 0.
        self.qh[s] = 0.
        self.qi[s] = 0.
        self.qc[s] = 0.
        es = _esat_v(self.t[s])
        self.est[s] = es
        self.qsat[s] = (_c(".622") * es) / (self.pe[s] - es)
        self.rho[s] = (_RHOC * self.pe[s]) / self.t[s]
        self.vel_p[s] = 0.
        self.rad_p[s] = 0.
        grow = F(_c("1.2") * self.alpha)
        self.radius[1] = self.rsurf
        for k in range(2, n + 1):
            self.radius[k] = F(self.radius[k - 1]
                               + F(grow * F(self.zt[k] - self.zt[k - 1])))
        self.radius[1] = self.rsurf
        self.rad_p[1] = self.rsurf
        for k in range(2, n + 1):
            self.radius[k] = F(self.radius[k - 1]
                               + F(grow * F(self.zt[k] - self.zt[k - 1])))
            self.rad_p[k] = self.radius[k]
        self.visc[1] = self.viscosity
        step = F(F(F(1.) * self.viscosity) / F(float(NKP)))
        for k in range(2, n + 1):
            self.visc[k] = _fmax(_c("1.e-3"), F(self.visc[k - 1] - step))
        self.lbound()

    # ------------------------------------------------------------ dynamics
    def tend0(self):
        s = slice(1, self.nm1 + 1)
        for name in ("wt", "tt", "qvt", "qct", "qht", "qit", "vel_t",
                     "rad_t"):
            getattr(self, name)[s] = 0.

    def vel_advectc(self, m1):
        wc, wt, rho, dzm = self.wc, self.wt, self.rho, self.dzm
        dn0 = np.zeros(NKP + 2, dtype=np.float32)
        dn0[1:m1 + 1] = rho[1:m1 + 1] * _E3
        flxw = np.zeros(NKP + 2, dtype=np.float32)
        flxw[1] = wc[1] * dn0[1]
        k = slice(2, m1)                      # 2 .. m1-1
        flxw[k] = (wc[k] * _HALF) * (dn0[2:m1] + dn0[3:m1 + 1])
        k = slice(2, m1 - 1)                  # 2 .. m1-2
        kp, km = slice(3, m1), slice(1, m1 - 2)
        a = (flxw[k] + flxw[km]) * (wc[k] + wc[km])
        b = (flxw[k] + flxw[kp]) * (wc[k] + wc[kp])
        c = ((flxw[kp] - flxw[km]) * _TWO) * wc[k]
        wt[k] = wt[k] + ((_HALF * dzm[k]) / (dn0[k] + dn0[kp])) * ((a - b) + c)

    def fa_zc(self, m1, scp, scr1):
        vt3dc, vt3df, vt3dg, vt3dk = (self.vt3dc, self.vt3df, self.vt3dg,
                                      self.vt3dk)
        k, kp = slice(1, m1), slice(2, m1 + 1)          # 1 .. m1-1
        vt3dg[k] = vt3dc[k] * ((self.vctr1[k] * scr1[k]
                                + self.vctr2[k] * scr1[kp])
                               + vt3df[k] * (scr1[k] - scr1[kp]))
        c, g = vt3dc[k], vt3dg[k].copy()
        up = (c > 0) & (g * vt3dk[k] > _HALF * scr1[k])
        down = (c < 0) & ((-g) * vt3dk[kp] > _HALF * scr1[kp])
        g = np.where(up, c * scr1[k], g)
        g = np.where(down, c * scr1[kp], g)
        vt3dg[k] = g
        k, km = slice(2, m1), slice(1, m1 - 1)          # 2 .. m1-1
        scr1[k] = scr1[k] + vt3dk[k] * ((vt3dg[km] - vt3dg[k])
                                        + scp[k] * (vt3dc[k] - vt3dc[km]))

    def scl_advectc(self, mzp):
        w, wc, rho, dzm = self.w, self.wc, self.rho, self.dzm
        dtlto2 = F(_HALF * self.dt)
        self.vt3dc[1] = F(F(F(F(w[1] + wc[1]) * dtlto2) * rho[1]) * _E3)
        self.vt3df[1] = F(F(F(_HALF * F(w[1] + wc[1])) * dtlto2) * dzm[1])
        k = slice(2, mzp + 1)
        s = (w[k] + wc[k]) * dtlto2
        self.vt3dc[k] = ((s * _HALF) * (rho[k] + rho[3:mzp + 2])) * _E3
        self.vt3df[k] = (s * _HALF) * dzm[k]
        k = slice(1, mzp + 1)
        self.vctr1[k] = (self.zt[2:mzp + 2] - self.zm[k]) * dzm[k]
        self.vctr2[k] = (self.zm[k] - self.zt[k]) * dzm[k]
        self.vt3dk[k] = self.dzt[k] / (rho[k] * _E3)
        dtli = F(_ONE / self.dt)
        k = slice(2, mzp)
        for var, tend in (("t", "tt"), ("qv", "qvt"), ("qc", "qct"),
                          ("qi", "qit"), ("qh", "qht"), ("vel_p", "vel_t"),
                          ("rad_p", "rad_t")):
            scp = getattr(self, var)
            scr1 = scp.copy()
            self.fa_zc(mzp, scp, scr1)
            self.scr1[:] = scr1
            sct = getattr(self, tend)
            sct[k] = sct[k] + (scr1[k] - scp[k]) * dtli

    def scl_misc(self, m1):
        k, km = slice(2, m1), slice(1, m1 - 1)
        wbar = _HALF * (self.w[k] + self.w[km])
        adiabat = -((wbar * _c("9.81")) / _c("1004."))
        dmdtm = ((_TWO * self.alpha) * np.abs(wbar)) / self.radius[k]
        self.tt[k] = (self.tt[k] + adiabat) - dmdtm * (self.t[k] - self.te[k])
        self.qvt[k] = self.qvt[k] - dmdtm * (self.qv[k] - self.qvenv[k])
        self.qct[k] = self.qct[k] - dmdtm * self.qc[k]
        self.qht[k] = self.qht[k] - dmdtm * self.qh[k]
        self.qit[k] = self.qit[k] - dmdtm * self.qi[k]
        self.vel_t[k] = self.vel_t[k] - dmdtm * (self.vel_p[k] - self.vel_e[k])
        self.rad_t[k] = (self.rad_t[k]
                         + ((_HALF * dmdtm) * _c("1.2")) * self.radius[k])
        if m1 - 1 >= 2:
            self.wbar = wbar[-1]
            self.adiabat = adiabat[-1]

    def scl_dyn_entrain(self, m1):
        k = slice(2, m1)
        dv = np.abs(self.vel_e[k] - self.vel_p[k])
        self.rad_t[k] = self.rad_t[k] + dv / _PI_DYN
        dmdtm = (_C2PI * dv) / self.radius[k]
        self.vel_t[k] = self.vel_t[k] - dmdtm * (self.vel_p[k] - self.vel_e[k])
        self.tt[k] = self.tt[k] - dmdtm * (self.t[k] - self.te[k])
        self.qvt[k] = self.qvt[k] - dmdtm * (self.qv[k] - self.qvenv[k])
        self.qct[k] = self.qct[k] - dmdtm * self.qc[k]
        self.qht[k] = self.qht[k] - dmdtm * self.qh[k]
        self.qit[k] = self.qit[k] - dmdtm * self.qi[k]

    def friction(self, ifrom, nm1, deltak, dt):
        kf = nm1 - deltak
        zmkf = self.zm[kf]
        ztop = self.zm[nm1]
        distim = _fmin(F(F(3.) * dt), F(60.))
        c1 = F(_ONE / F(distim * F(ztop - zmkf)))
        c2 = F(dt * c1)
        k = np.arange(2, nm1 + 1)
        k = k[self.zt[k] > zmkf]
        dzk = self.zt[k] - zmkf
        if ifrom == 1:
            self.tt[k] = self.tt[k] + (c1 * dzk) * (self.te[k] - self.t[k])
        else:
            self.w[k] = self.w[k] + (c2 * dzk) * (_ZERO - self.w[k])

    def fallpart(self, m1):
        k, km, kp = slice(2, m1), slice(1, m1 - 1), slice(3, m1 + 1)
        w, rho = self.w, self.rho
        # VTH(k+1), VTI(k+1) are read before the loop reaches k+1.
        vth_up = self.vth[kp].copy()
        vti_up = self.vti[kp].copy()
        self.vth[k] = F(-4.)
        vhrel = w[k] + self.vth[k]
        self.vti[k] = F(-3.)
        virel = w[k] + self.vti[k]
        self.cvi[k] = _c("1.6") + ((_c("0.57E-3")
                                    * _powf_v(np.abs(virel), _c("1.5")))
                                   / _F0)
        vth_dn, vti_dn = self.vth[km], self.vti[km]
        qh, qi = self.qh[k], self.qi[k]
        with np.errstate(all="ignore"):
            up_h = (qh * (rho[k] * self.vth[k] - rho[km] * vth_dn)) / rho[km]
            dn_h = (qh * (rho[kp] * vth_up - rho[k] * self.vth[k])) / rho[k]
            up_i = (qi * (rho[k] * self.vti[k] - rho[km] * vti_dn)) / rho[km]
            dn_i = (qi * (rho[kp] * vti_up - rho[k] * self.vti[k])) / rho[k]
        dfhz = np.where(vhrel >= 0, up_h, dn_h)
        dfiz = np.where(virel >= 0, up_i, dn_i)
        dz1 = self.zm[k] - self.zm[km]
        self.qht[k] = self.qht[k] - dfhz / dz1
        self.qit[k] = self.qit[k] - dfiz / dz1

    def visc_w(self, m1, kmt):
        m2 = min(m1, kmt)
        k, km, kp = slice(2, m2), slice(1, m2 - 1), slice(3, m2 + 1)
        dz1t = _HALF * (self.zt[kp] - self.zt[km])
        dz2t = self.visc[k] / (dz1t * dz1t)
        dz1m = _HALF * (self.zm[kp] - self.zm[km])
        dz2m = self.visc[k] / (dz1m * dz1m)

        def d2(a, scale):
            return ((a[kp] - _TWO * a[k]) + a[km]) * scale

        d2w = d2(self.w, dz2m)
        d2t, d2qv = d2(self.t, dz2t), d2(self.qv, dz2t)
        d2qh, d2qc, d2qi = (d2(self.qh, dz2t), d2(self.qc, dz2t),
                            d2(self.qi, dz2t))
        d2vel, d2rad = d2(self.vel_p, dz2t), d2(self.rad_p, dz2t)
        self.wt[k] = self.wt[k] + d2w
        self.tt[k] = self.tt[k] + d2t
        self.qvt[k] = self.qvt[k] + d2qv
        self.qct[k] = self.qct[k] + d2qc
        self.qht[k] = self.qht[k] + d2qh
        self.qit[k] = self.qit[k] + d2qi
        self.vel_t[k] = self.vel_t[k] + d2vel
        self.rad_t[k] = self.rad_t[k] + d2rad

    def update(self, m1, varn):
        k = slice(2, m1)
        dt = self.dt
        if varn == "W":
            self.w[k] = self.w[k] + self.wt[k] * dt
            return
        self.t[k] = self.t[k] + self.tt[k] * dt
        self.qv[k] = self.qv[k] + self.qvt[k] * dt
        self.qc[k] = self.qc[k] + self.qct[k] * dt
        self.qh[k] = self.qh[k] + self.qht[k] * dt
        self.qi[k] = self.qi[k] + self.qit[k] * dt
        for name in ("qv", "qc", "qh", "qi"):
            a = getattr(self, name)
            a[k] = np.where(_ZERO > a[k], _ZERO, a[k])
        self.vel_p[k] = self.vel_p[k] + self.vel_t[k] * dt
        self.rad_p[k] = self.rad_p[k] + self.rad_t[k] * dt

    def hadvance(self, iac, m1):
        eps = _c(".2")
        if self.mintime == 1:
            eps = F(0.5)
        m = slice(1, m1 + 1)
        if iac == 1:
            self.wc[m] = self.wc[m] + eps * (self.w[m] - _TWO * self.wc[m])
        else:
            af = self.w[m].copy()
            self.w[m] = self.wc[m] + eps * af
            self.wc[m] = af

    def buoyancy(self, m1):
        k = slice(2, m1)
        g, eps = _c("9.8"), _c("0.622")
        qv, qve = self.qv[k], self.qvenv[k]
        tv = (self.t[k] * (_ONE + (qv / eps))) / (_ONE + qv)
        tve = (self.te[k] * (_ONE + (qve / eps))) / (_ONE + qve)
        qw = (self.qh[k] + self.qi[k]) + self.qc[k]
        self.scr1[k] = F(g * _UMGAMAI) * (((tv - tve) / tve) - qw)
        k, kp = slice(2, m1 - 1), slice(3, m1)
        self.wt[k] = self.wt[k] + _HALF * (self.scr1[k] + self.scr1[kp])

    def entrainment(self, m1):
        k, km = slice(2, m1), slice(1, m1 - 1)
        wbar = self.w[k]
        rbar = _HALF * (self.radius[k] + self.radius[km])
        dmdtm = ((F(_UMGAMAI * _TWO) * self.alpha) * np.abs(wbar)) / rbar
        self.wt[k] = self.wt[k] - dmdtm * np.abs(wbar)
        dv = ((self.vel_p[k] - self.vel_e[k]) + self.vel_p[km]) \
            - self.vel_e[km]
        dyn = (F(_C2PI * _HALF) * np.abs(dv)) / rbar
        self.wt[k] = self.wt[k] - dyn * np.abs(wbar)

    def microphysics(self, nm1):
        dt_save = self.dt
        self.dt = F(self.dt / F(3.))
        for _ in range(3):
            self.fallpart(nm1)
            # The loop's only carried dependence is QSAT(L-1), set by the
            # iteration before (QSAT(L+1) is still last step's); T(L) and
            # PE(L) are untouched until WATERBAL(L), so ES, QSAT, EST and
            # RHO of every level can be formed first.
            k = slice(2, nm1)
            es = _esat_v(self.t[k])
            qsat_up = self.qsat[3:nm1 + 1].copy()
            wbar = _HALF * (self.w[k] + self.w[1:nm1 - 1])
            self.qsat[k] = (_EPS * es) / (self.pe[k] - es)
            self.est[k] = es
            self.rho[k] = (_RHOC * self.pe[k]) / self.t[k]
            for L in range(2, nm1):
                self.L = L
                self.wbar = wbar[L - 2]
                self.dqsdz = F(F(qsat_up[L - 2] - self.qsat[L - 1])
                               / F(self.zt[L + 1] - self.zt[L - 1]))
                self.waterbal()
            self.L = nm1
        self.dt = dt_save

    # ------------------------------------------------------------- the run
    def makeplume(self, kmt) -> np.float32:
        self.tstpf = F(2.0)
        self.viscosity = F(500.)
        self.mintime = 1
        ztopmax = F(0.)
        self.ztop = F(0.)
        self.time = F(0.)
        self.dt = F(1.)
        wmax = F(1.)
        kkmax = 10
        deltak = 20
        self.L = 1
        self.initial(kmt)
        rmaxtime = F(float(self.maxtime))
        while self.time <= rmaxtime:
            nm1 = min(kmt, kkmax + deltak)
            self.nm1 = nm1
            self.dt = _fmin(F(5.), F(F(self.zm[2] - self.zm[1])
                                     / F(self.tstpf * wmax)))
            self.time = F(self.time + self.dt)
            self.mintime = 1 + int(self.time) // 60
            wmax = F(1.)
            self.tend0()
            self.L = 1
            self.lbound()
            self.vel_advectc(nm1)
            self.scl_advectc(nm1)
            self.scl_misc(nm1)
            self.scl_dyn_entrain(nm1)
            self.friction(1, nm1, deltak, self.dt)
            self.microphysics(nm1)
            self.visc_w(nm1, kmt)
            self.update(nm1, "S")
            self.hadvance(1, nm1)
            self.buoyancy(nm1)
            self.entrainment(nm1)
            self.update(nm1, "W")
            self.hadvance(2, nm1)
            k = slice(2, nm1 + 1)
            es = _esat_v(self.t[k])
            self.qsat[k] = (_EPS * es) / (self.pe[k] - es)
            self.est[k] = es
            self.rho[k] = (_RHOC * self.pe[k]) / self.t[k]
            for value in np.abs(self.wc[k]):
                if value > wmax:
                    wmax = value
            self.friction(2, nm1, deltak, self.dt)
            self.radius[k] = self.rad_p[k]
            kk = 1
            while self.w[kk] > F(1.):
                kk += 1
                self.ztop = self.zm[kk]
            self.ztop_[self.mintime] = self.ztop
            ztopmax = _fmax(self.ztop, ztopmax)
            kkmax = max(kk, kkmax)
            if self.mintime > 10:
                if F(abs(F(self.ztop_[self.mintime]
                           - self.ztop_[self.mintime - 10]))) < F(100.):
                    break
        return ztopmax


def _k_lim(zzcon: np.ndarray, ztop) -> int:
    """set_flam_vert's search: the first level above ztop, nkp if none."""
    for k in range(1, NKP):
        if zzcon[k] > ztop:
            return k
    return NKP


#: The plume-model state recorded as each makeplume returns, in the order
#: of tools/smoke_wrf471_oracle/run_plumerise.F90 (mode 'tops').
STATE_FIELDS = ("w", "t", "qv", "qc", "qh", "qi", "rho", "radius", "vel_p",
                "rad_p", "wc", "qsat")
STATE_SCALARS = ("time", "dt", "wbar", "dqsdz")


def _plumerise_column(col: dict, heat_flux: np.ndarray, trace=None):
    """``plumerise`` for one column; returns (eburn_out, ztopmax, k1, k2).

    ``trace``, a dict, receives the plume state after every makeplume:
    ``state`` (nkp, 12, 8), ``scalars`` (4, 8) and ``mintime`` (8,), slot
    ``imm + 2*(biome - 1)`` (one-based), zeros where no plume ran.
    """
    if trace is not None:
        trace["state"] = np.zeros((NKP, len(STATE_FIELDS), 8), np.float32)
        trace["scalars"] = np.zeros((len(STATE_SCALARS), 8), np.float32)
        trace["mintime"] = np.zeros(8, np.int32)
    m1 = col["m1"]
    plume = _Plume(heat_flux)
    # The module's nkp-long zzcon: the column's w levels, then zeros.
    zzcon = np.zeros(NKP + 3, dtype=np.float32)
    zzcon[1:m1 + 1] = col["zzcon"][1:m1 + 1]
    eburn_out = np.zeros(NKP + 3, dtype=np.float32)
    eburn_out[1] = col["eburn_in"]
    kmt = plume.get_env_condition(col)
    tops = np.zeros((4, 2), dtype=np.float32)
    k1s = np.zeros(4, dtype=np.int32)
    k2s = np.zeros(4, dtype=np.int32)
    ztopmax = np.zeros(3, dtype=np.float32)
    for iveg in range(1, 5):
        if col["mean_fct"][iveg - 1] < _c("1.e-6"):
            continue
        burnt_area = col["firesize"][iveg - 1]
        for imm in (1, 2):
            plume.get_fire_properties(imm, iveg, burnt_area)
            if iveg == 4 and imm == 2:
                ztopmax[2] = ztopmax[1]
                ztopmax[1] = zzcon[1]
                continue
            ztopmax[imm] = plume.makeplume(kmt)
            if trace is not None:
                slot = imm + 2 * (iveg - 1) - 1
                for j, name in enumerate(STATE_FIELDS):
                    trace["state"][:, j, slot] = getattr(plume,
                                                         name)[1:NKP + 1]
                for j, name in enumerate(STATE_SCALARS):
                    trace["scalars"][j, slot] = getattr(plume, name)
                trace["mintime"][slot] = plume.mintime
        k1 =max(3, _k_lim(zzcon, ztopmax[1]))
        k2 = max(3, _k_lim(zzcon, ztopmax[2]))
        if k2 < k1:
            k2 = k1
        if k2 >= col["m1"]:
            # dz_flam would read zzcon(m1+1), past the column (and k2 > m1
            # write past eburn_out): the device's status 3.
            raise RuntimeError(
                "plumerise: the injection layer leaves the column "
                f"(k2 = {k2}, m1 = {col['m1']}); status [3]")
        dz_flam = F(zzcon[k2 + 1] - zzcon[k1])
        dzi = F(_ONE / dz_flam)
        add = F(F(col["mean_fct"][iveg - 1] * col["eburn_in"]) * dzi)
        for k in range(k1, k2 + 1):
            eburn_out[k] = F(eburn_out[k] + add)
        tops[iveg - 1] = ztopmax[1:3]
        k1s[iveg - 1] = k1
        k2s[iveg - 1] = k2
    return eburn_out[1:m1 + 1], tops, k1s, k2s


class PlumeriseResult(NamedTuple):
    """``plumerise``'s output and the plume heights it keeps local.

    ``eburn_out`` is ``(m1, ...)``; ``ztopmax`` is ``(4, 2, ...)``, the plume
    top (m above ground) of each biome at its minimum and maximum heat flux
    (grassland: the ground, then its one plume top); ``k1``, ``k2`` are
    ``(4, ...)`` int32, the one-based first and last level of each biome's
    flaming injection.  A biome the column does not burn has zeros.
    """

    eburn_out: Any
    ztopmax: Any
    k1: Any
    k2: Any


def _heat_table(heat_flux) -> np.ndarray:
    table = HEAT_FLUX if heat_flux is None else np.asarray(heat_flux,
                                                          dtype=np.float32)
    if table.shape != (2, 4):
        raise ValueError(f"heat_flux must be (2, 4) kW m-2, got {table.shape}")
    if not np.all(table > 0):
        raise ValueError("heat_flux must be positive")
    return np.ascontiguousarray(table)


def _burning(firesize, mean_fct, eburn_in):
    """module_plumerise1.F:213-227: which columns reach plumerise."""
    sum_fct = ((mean_fct[0] + mean_fct[1]) + mean_fct[2]) + mean_fct[3]
    sum_size = ((firesize[0] + firesize[1]) + firesize[2]) + firesize[3]
    small = _c("1.e-6")
    return (~(sum_fct < small)) & (~(sum_size < small)) & (eburn_in != 0)


def _check(u, v, theta, pi, qv, zt, zm, firesize, mean_fct, eburn_in):
    m1 = u.shape[0]
    cols = u.shape[1:]
    for name, a in (("v", v), ("theta", theta), ("pi", pi), ("qv", qv),
                    ("zt", zt), ("zm", zm)):
        if a.shape != u.shape:
            raise ValueError(f"{name} has shape {a.shape}, u has {u.shape}")
    for name, a in (("firesize", firesize), ("mean_fct", mean_fct)):
        if a.shape != (4,) + cols:
            raise ValueError(f"{name} must be (4,) + {cols}, got {a.shape}")
    if eburn_in.shape != cols:
        raise ValueError(f"eburn_in must be {cols}, got {eburn_in.shape}")
    if not 3 <= m1 <= NKP - 2:
        raise ValueError(f"plumerise takes 3 to {NKP - 2} levels, got {m1}")
    return m1, cols


def plumerise_np(u, v, theta, pi, qv, zt, zm, firesize, mean_fct, eburn_in,
                 heat_flux=None) -> PlumeriseResult:
    """NumPy reference of :func:`plumerise`, column by column, real(4).

    Seconds per burning column: it is the reference, not a path.
    """
    arrays = [np.asarray(a, dtype=np.float32) for a in (
        u, v, theta, pi, qv, zt, zm, firesize, mean_fct, eburn_in)]
    u, v, theta, pi, qv, zt, zm, firesize, mean_fct, eburn_in = arrays
    m1, cols = _check(*arrays)
    table = _heat_table(heat_flux)
    burning = _burning(firesize, mean_fct, eburn_in)
    eburn_out = np.zeros((m1,) + cols, dtype=np.float32)
    eburn_out[0] = eburn_in
    ztopmax = np.zeros((4, 2) + cols, dtype=np.float32)
    k1 = np.zeros((4,) + cols, dtype=np.int32)
    k2 = np.zeros((4,) + cols, dtype=np.int32)

    def pad(a):
        out = np.zeros(NKP + 3, dtype=np.float32)
        out[1:m1 + 1] = a
        return out

    for c in np.ndindex(cols):
        if not burning[c]:
            continue
        s = (slice(None),) + c
        col = {"m1": m1, "ucon": pad(u[s]), "vcon": pad(v[s]),
               "thtcon": pad(theta[s]), "picon": pad(pi[s]),
               "rvcon": pad(qv[s]), "zcon": pad(zt[s]), "zzcon": pad(zm[s]),
               "firesize": firesize[s], "mean_fct": mean_fct[s],
               "eburn_in": eburn_in[c]}
        out, tops, a, b = _plumerise_column(col, table)
        eburn_out[s] = out
        ztopmax[(slice(None), slice(None)) + c] = tops
        k1[s] = a
        k2[s] = b
    return PlumeriseResult(eburn_out, ztopmax, k1, k2)


# --------------------------------------------------------------------------
# CuPy path

#: The plume model's module arrays per column in the block's shared memory
#: (smoke_plumerise.cu: SP_NARR arrays of SP_LEV = nkp + 3 levels).
SHARED_ARRAYS = 44
SHARED_LEVELS = NKP + 3
SHARED_BYTES = SHARED_ARRAYS * SHARED_LEVELS * 4
#: Threads per column (block); the kernel holds at most four levels per
#: thread in registers, so it needs at least 50.
THREADS = 128


def _f32(xp, a, name, shape=None):
    out = xp.ascontiguousarray(xp.asarray(a, dtype=xp.float32))
    if shape is not None and out.shape != tuple(shape):
        raise ValueError(f"{name} has shape {out.shape}, expected {shape}")
    return out


def plumerise(u, v, theta, pi, qv, zt, zm, firesize, mean_fct, eburn_in,
              heat_flux=None, trace: dict | None = None) -> PlumeriseResult:
    """The plume-rise injection over a whole domain, on the device.

    Arrays as :func:`plumerise_np` takes them: ``(m1, ...)`` column fields,
    ``(4, ...)`` fire sizes (m2) and flaming fractions, ``(...)``
    ``eburn_in``; any trailing column shape.  The burning columns are
    compacted on the device and run one thread block each (one launch),
    the block's threads sharing the column's levels.

    ``trace``, a dict (for the parity tests), receives the plume state after
    every makeplume as :func:`_plumerise_column` records it, with the
    columns last: ``state`` (nkp, 12, 8, ...), ``scalars`` (4, 8, ...),
    ``mintime`` (8, ...).
    """
    import cupy as cp

    from woof.smoke import _kernels

    u = _f32(cp, u, "u")
    m1, cols = u.shape[0], u.shape[1:]
    ncol = int(np.prod(cols, dtype=np.int64))
    v, theta, pi, qv, zt, zm = (_f32(cp, a, n, u.shape) for a, n in (
        (v, "v"), (theta, "theta"), (pi, "pi"), (qv, "qv"), (zt, "zt"),
        (zm, "zm")))
    firesize = _f32(cp, firesize, "firesize", (4,) + cols)
    mean_fct = _f32(cp, mean_fct, "mean_fct", (4,) + cols)
    eburn_in = _f32(cp, eburn_in, "eburn_in", cols)
    _check(u, v, theta, pi, qv, zt, zm, firesize, mean_fct, eburn_in)
    table = cp.asarray(_heat_table(heat_flux))

    eburn_out = cp.zeros((m1,) + cols, dtype=cp.float32)
    eburn_out[0] = eburn_in
    ztopmax = cp.zeros((4, 2) + cols, dtype=cp.float32)
    k1 = cp.zeros((4,) + cols, dtype=cp.int32)
    k2 = cp.zeros((4,) + cols, dtype=cp.int32)
    status = cp.zeros(cols, dtype=cp.int32)
    burning = _burning(firesize.reshape(4, ncol), mean_fct.reshape(4, ncol),
                       eburn_in.reshape(ncol))
    index = cp.flatnonzero(burning).astype(cp.int32)
    nburn = int(index.size)
    null = np.uintp(0)
    tstate = tscal = tmin = null
    if trace is not None:
        tstate = cp.zeros((8, len(STATE_FIELDS), NKP, ncol), cp.float32)
        tscal = cp.zeros((8, len(STATE_SCALARS), ncol), cp.float32)
        tmin = cp.zeros((8, ncol), cp.int32)
    if nburn:
        launch = _kernels.kernel("smoke_plumerise_columns")
        launch((nburn,), (THREADS,),
               (index, np.int32(m1), np.int32(ncol), u, v, theta, pi, qv, zt,
                zm, firesize, mean_fct, eburn_in, table, eburn_out, ztopmax,
                k1, k2, status, tstate, tscal, tmin),
               shared_mem=SHARED_BYTES)
        if bool(cp.any(status != 0)):
            codes = sorted(set(cp.asnumpy(status[status != 0]).tolist()))
            raise RuntimeError(
                "plumerise: a column's plume grid fell outside it, where "
                f"WRF stops (htint or get_env_condition) or its injection "
                f"layer leaves the column; status {codes}")
    if trace is not None:
        trace["state"] = cp.transpose(tstate, (2, 1, 0, 3)).reshape(
            (NKP, len(STATE_FIELDS), 8) + cols)
        trace["scalars"] = cp.transpose(tscal, (1, 0, 2)).reshape(
            (len(STATE_SCALARS), 8) + cols)
        trace["mintime"] = tmin.reshape((8,) + cols)
    return PlumeriseResult(eburn_out, ztopmax, k1, k2)


__all__ = ["HEAT_FLUX", "LITERALS", "NKP", "PlumeriseResult", "plumerise",
           "plumerise_np"]
