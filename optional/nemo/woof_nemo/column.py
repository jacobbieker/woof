# WOOF-OPTIN: nemo
"""A coupled NEMO tracer column on device: :class:`NemoColumnOcean`.

The ported NEMO 5.0.2 pieces stepped in src/OCE/stpmlf.F90's order, as one
ocean behind WRF's coupler (:class:`woof.ocean.interface.ExternalOcean`):

1. ``sbc``: sbc_ssm's nn_fsbc = 1 lines (sst_m from eos_pt_from_ct under
   TEOS-10 or S-EOS, the model temperature under EOS-80), then sbc_cpl_rcv's
   ocean-only mapping of WRF's ``cpl_snd2`` fields (``column.cu``);
2. eos_rab and bn2 for the before and now states (:mod:`woof_nemo.eos`);
3. zdf_phy's TKE path: zdf_sh2, zdf_drg 'BOTTOM' (``column.cu``), zdf_tke
   (:mod:`woof_nemo.tke`), then avt = avt_k and avs = avt;
4. tra_sbc (``column.cu``), tra_qsr (:mod:`woof_nemo.traqsr`), tra_zdf
   (:mod:`woof_nemo.trazdf`), tra_atf_qco's tra_atf_fix_lf (``column.cu``);
5. the time-level swap, sbc_cpl_snd's sea-surface temperature, and the
   switch from the first Euler step to leapfrog.

Graded word for word against ``optional/nemo/oracle/build_column.sh``,
which compiles those NEMO files whole and runs this sequence.

What the column is: one NEMO water column under every ocean point of the
WRF grid, on a z-coordinate (1-D levels, full steps) with a linear free
surface, the ocean at rest.  Momentum, advection, lateral diffusion and the
free surface are not stepped, so the column couples SST only: it hands WRF
``SST`` and zero ``UOCE``/``VOCE`` (which WRF merges onto zero anyway).
Everything else is refused by name: sea ice, runoff, restarts, mixed
forced/coupled forcing, the diurnal-cycle reconstruction, partial steps and
s-coordinates, background-profile and equatorial avtb shapes,
enhanced vertical diffusion, double diffusion, drag boost files, and an
averaged (LOCTRANS AVERAGE) SST.

Coupling: the column receives the fields the coupler delivers once per
coupling interval (``set_fluxes``) and, as with namcouple's LAG = rn_Dt and
LOCTRANS INSTANT for O_SSTSST, hands back the SST after the last NEMO step
of the interval (``surface``).  Before the first step it hands back the
initial state's SST.
"""

from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path

import numpy as np

from .eos import EquationOfState
from .tke import TkeMixing
from .traqsr import TraQsr2BD
from .trazdf import TracerVerticalDiffusion

_TPB = 64
_MODULE = None
#: phycst.F90 and eos_init (eosbn2.F90 :1898-1899, :2331-2334)
RHO0, RCP, RT0, VKARMN = 1026.0, 3991.86795711963, 273.15, 0.4
#: namdrg_bot (cfgs/SHARED/namelist_ref, NEMO 5.0.2)
NAMDRG_BOT = dict(rn_Cd0=1.e-3, rn_Uc0=0.4, rn_Cdmax=0.1, rn_ke0=2.5e-3, rn_z0=3.e-3)
DRAG = {"off": 0, "lin": 1, "non_lin": 2, "loglayer": 3}
#: namzdf's background coefficients (namelist_ref)
RN_AVM0, RN_AVT0 = 1.2e-4, 1.2e-5


def _module():
    global _MODULE
    if _MODULE is None:
        import cupy as cp

        kernels = Path(__import__("woof.core.kernels", fromlist=["x"]).__file__).parent
        here = Path(__file__).resolve().parent
        source = ((kernels / "glibc_flt64.cuh").read_text(encoding="utf-8") + "\n"
                  + (here / "column.cu").read_text(encoding="utf-8"))
        _MODULE = cp.RawModule(code=source, options=("--fmad=false",))
    return _MODULE


def _refuse(name, value, default=False, why="is not ported in the NEMO column"):
    if value != default:
        raise NotImplementedError(f"{name} {why}")


def _pad(a):
    """The WRF-grid interior with NEMO's halo: one cell, repeating the edge."""
    pad = [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)]
    return np.pad(a, pad, mode="edge")


class NemoColumnOcean:
    """NEMO tracer columns under a WRF grid, coupled SST-only.

    ``mbkt`` is the bottom level of each ``(ny, nx)`` point (0 = land),
    ``latitude`` its latitude (degrees; zdf_tke's htau), ``e3t``/``e3w``/
    ``gdept``/``gdepw`` the 1-D z-coordinate levels (m), ``temperature`` and
    ``salinity`` the ``(jpk, ny, nx)`` initial state (CT and absolute
    salinity under TEOS-10, potential temperature and practical salinity
    otherwise).  ``rn_Dt`` is NEMO's step, ``coupling_interval`` the
    coupler's period, a whole number of steps.
    """

    def __init__(self, *, mbkt, latitude, e3t, e3w, gdept, gdepw, temperature, salinity,
                 rn_Dt, coupling_interval, eos="teos10", traqsr=True, qsr_scheme="2bd",
                 drag="non_lin", drag_params=None, tke=None, rn_atfp=0.1,
                 nn_avb=0, nn_havtb=0, ln_zdfevd=False, ln_zdfddm=False, ln_boost=False,
                 nn_ice=0, ln_rnf=False, ln_rstart=False, ln_mixcpl=False, ln_dm2dc=False,
                 vertical_coordinate="zco", sst_loctrans="INSTANT", emp_receive="oce only",
                 stress_orientation="local grid"):
        import cupy as cp

        _refuse("sea ice (nn_ice)", nn_ice, 0)
        _refuse("river runoff (ln_rnf)", ln_rnf)
        _refuse("restarting from a NEMO restart (ln_rstart)", ln_rstart)
        _refuse("mixed forced-coupled forcing (ln_mixcpl)", ln_mixcpl)
        _refuse("the diurnal-cycle reconstruction of qsr (ln_dm2dc)", ln_dm2dc)
        _refuse("enhanced vertical diffusion (ln_zdfevd)", ln_zdfevd)
        _refuse("double-diffusive mixing (ln_zdfddm)", ln_zdfddm)
        _refuse("the regional drag boost file (ln_boost)", ln_boost)
        _refuse("the background avt profile (nn_avb = 1)", nn_avb, 0)
        _refuse("the equatorial avtb shape (nn_havtb = 1)", nn_havtb, 0)
        _refuse(f"vertical coordinate {vertical_coordinate!r} (partial steps ln_zps, "
                "s-coordinates ln_sco)", vertical_coordinate, "zco")
        _refuse(f"OASIS LOCTRANS {sst_loctrans} of the sent SST", sst_loctrans, "INSTANT")
        _refuse(f"sn_rcv_emp%cldes = {emp_receive!r} (WRF sends one E-P)", emp_receive,
                "oce only")
        _refuse(f"sn_rcv_tau%clvor = {stress_orientation!r} (WRF sends grid-relative stress)",
                stress_orientation, "local grid")
        if drag not in DRAG:
            raise ValueError(f"namdrg offers {sorted(DRAG)}")
        unknown = sorted(set(drag_params or {}) - set(NAMDRG_BOT))
        if unknown:
            raise ValueError(f"not &namdrg_bot keys: {unknown}")
        dp = {**NAMDRG_BOT, **(drag_params or {})}

        mbkt = np.asarray(mbkt)
        ny, nx = mbkt.shape
        e3t, e3w, gdept, gdepw = (np.asarray(a, np.float64) for a in (e3t, e3w, gdept, gdepw))
        jpk = int(e3t.shape[0])
        for name, a in (("e3w", e3w), ("gdept", gdept), ("gdepw", gdepw)):
            if a.shape != (jpk,):
                raise ValueError(f"{name}: expected ({jpk},) levels")
        if mbkt.min() < 0 or mbkt.max() > jpk - 1:
            raise ValueError("mbkt must lie in 0..jpk-1 (level jpk is never wet)")
        if gdepw[0] != 0.0:
            raise ValueError("gdepw[0] must be 0 (the surface)")
        rn_Dt = float(rn_Dt)
        nsub = coupling_interval / rn_Dt
        if not (rn_Dt > 0 and nsub >= 1 and abs(nsub - round(nsub)) < 1e-9):
            raise ValueError("the coupling interval must be a whole number of rn_Dt steps")
        self.shape = (ny, nx)
        self.jpk = jpk
        self.rn_Dt = rn_Dt
        self.coupling_interval = float(coupling_interval)
        self.nsub = int(round(nsub))
        # isec = (kt - nit000) * NINT(rn_Dt) (sbccpl.F90), the coupling period in s
        self._nint_dt = int(math.floor(rn_Dt + 0.5))
        self._ncpl_sec = int(round(self.coupling_interval))
        self.rn_atfp = float(rn_atfp)
        self.ln_traqsr = bool(traqsr)

        # grid with NEMO's one-cell halo repeating the edge (usrdef_zgr style)
        mb = _pad(mbkt.astype(np.int32))
        jpj, jpi = mb.shape
        lev = np.arange(1, jpk + 1)[:, None, None]
        tmask = (lev <= mb[None]).astype(np.float64)
        umask = tmask * np.concatenate([tmask[:, :, 1:], tmask[:, :, -1:]], axis=2)
        vmask = tmask * np.concatenate([tmask[:, 1:, :], tmask[:, -1:, :]], axis=1)

        def wof(m):
            w = m.copy()
            w[1:] = m[1:] * m[:-1]
            return w

        full = np.ones((jpk, jpj, jpi))
        g = {k: cp.asarray(np.ascontiguousarray(v)) for k, v in dict(
            tmask=tmask, umask=umask, vmask=vmask, wmask=wof(tmask), wumask=wof(umask),
            wvmask=wof(vmask), e3t=e3t[:, None, None] * full, e3w=e3w[:, None, None] * full,
            gdept=gdept[:, None, None] * full, gdepw=gdepw[:, None, None] * full).items()}
        g["ssmask"] = cp.ascontiguousarray(g["tmask"][0])
        g["mbkt"] = cp.asarray(np.ascontiguousarray(np.maximum(mb, 1)))
        self.g = g
        self._n = (jpi - 2) * (jpj - 2)
        self._dims = (np.int32(jpi), np.int32(jpj))

        # istate_init (:81-87, :135): the initial state masked, in before and now
        ts = np.zeros((3, 2, jpk, jpj, jpi))
        for jn, a in enumerate((temperature, salinity)):
            a = np.asarray(a, np.float64)
            if a.shape != (jpk, ny, nx):
                raise ValueError(f"initial state: expected ({jpk}, {ny}, {nx})")
            ts[0, jn] = _pad(a) * tmask
        ts[1] = ts[0]
        self.ts = cp.asarray(ts)
        self.rhs = cp.zeros((2, jpk, jpj, jpi), cp.float64)
        self.nbb, self.nnn, self.naa = 0, 1, 2
        z3 = lambda: cp.zeros((jpk, jpj, jpi), cp.float64)  # noqa: E731
        z2 = lambda: cp.zeros((jpj, jpi), cp.float64)  # noqa: E731
        self.uu = z3()                       # the ocean at rest (no momentum stepped)
        self.vv = z3()
        self.rab = {k: z3() for k in ("ab", "bb", "an", "bn")}
        self.rn2, self.rn2b, self.sh2 = z3(), z3(), z3()
        for name in ("sst_m", "sss_m", "utau", "vtau", "taum", "wndm", "emp", "qns", "qsr",
                     "sfx", "rCdU_bot", "pCd0"):
            setattr(self, name, z2())
        self.sbc_tsc = cp.zeros((2, jpj, jpi), cp.float64)
        self.sbc_tsc_b = cp.zeros((2, jpj, jpi), cp.float64)
        self.frcv = cp.zeros((7, jpj, jpi), cp.float64)

        # nemo_init: eos_init; zdf_phy_init (:151-179, nn_avb = nn_havtb = 0)
        self.eos = EquationOfState(eos)
        self.use_ct = eos in ("teos10", "seos")       # l_useCT
        self.r1_rho0 = 1.0 / RHO0
        self.r1_rho0_rcp = 1.0 / (RHO0 * RCP)
        self.avmb = cp.full((jpk,), RN_AVM0, cp.float64)
        self.avtb = cp.full((jpk,), RN_AVT0, cp.float64)
        self.avtb_2d = cp.ones((jpj, jpi), cp.float64)
        self.avt_k = cp.ascontiguousarray(self.avtb_2d[None] * self.avtb[:, None, None] * g["wmask"])
        self.avm_k = cp.ascontiguousarray(self.avmb[:, None, None] * g["wmask"])
        self.avt, self.avm = z3(), z3()
        # zdf_tke_init, zdf_drg_init, tke_rst 'READ' from rest
        self.tke = TkeMixing(rn_Dt=rn_Dt, gphit=cp.asarray(_pad(np.asarray(latitude, np.float64))),
                             **(tke or {}))
        self.en, self.dissl = self.tke.start_from_rest(g["wmask"])
        self.drag = DRAG[drag]
        self.drag_params = dp
        k = _module()
        k.get_function("nemo_drg_cd0")(self._grid(), (_TPB,), (
            g["ssmask"], g["e3t"], g["mbkt"], np.int32(self.drag), np.float64(dp["rn_Cd0"]),
            np.float64(dp["rn_Cdmax"]), np.float64(dp["rn_z0"]), np.float64(VKARMN), self.pCd0,
            *self._dims))
        if self.drag in (0, 1):                       # l_zdfdrg = F: set once
            self._drag_cdu()
        # tra_qsr_init sees domain.F90's rDt = 2 rn_Dt
        self.qsr_solver = None
        if self.ln_traqsr:
            self.qsr_solver = TraQsr2BD(gdepw=g["gdepw"], e3t=g["e3t"], tmask=g["tmask"],
                                        wmask=g["wmask"], rDt=2.0 * rn_Dt, scheme=qsr_scheme)
        self.zdf = TracerVerticalDiffusion(rDt=2.0 * rn_Dt, ln_zdfddm=ln_zdfddm,
                                           ln_seos=(eos == "seos"))
        self.kt = 0
        self.l_1st_euler = True
        self.rDt = 2.0 * rn_Dt
        self._fluxes = None
        self._sst = cp.zeros((ny, nx), cp.float32)
        self._uoce = cp.zeros((ny, nx), cp.float32)
        self._voce = cp.zeros((ny, nx), cp.float32)
        self._send_sst()

    def _grid(self):
        return (((self._n + _TPB - 1) // _TPB),)

    def _drag_cdu(self):
        _module().get_function("nemo_drg_cdu")(self._grid(), (_TPB,), (
            self.pCd0, self.uu, self.vv, self.g["mbkt"], np.int32(self.drag),
            np.float64(self.drag_params["rn_Uc0"]), np.float64(self.drag_params["rn_ke0"]),
            self.rCdU_bot, *self._dims))

    def _send_sst(self):
        n = self.nnn
        _module().get_function("nemo_cpl_snd_sst")(self._grid(), (_TPB,), (
            self.ts[n, 0], self.ts[n, 1], self.g["tmask"], np.int32(self.use_ct),
            np.float64(RT0), self._sst, *self._dims))

    # -- ExternalOcean -----------------------------------------------------
    def set_fluxes(self, fluxes) -> None:
        """The coupling-interval averages of WRF's ``cpl_snd2`` fields
        (:class:`woof.ocean.wrf_cpl.AtmosphereToOcean`), float32 ``(ny, nx)``."""
        import cupy as cp

        for f in fields(fluxes):
            a = getattr(fluxes, f.name)
            if tuple(a.shape) != self.shape or a.dtype != cp.float32:
                raise ValueError(f"{f.name}: expected float32 {self.shape}")
        self._fluxes = fluxes

    def step(self, dt: float) -> None:
        """One coupling interval: ``coupling_interval / rn_Dt`` NEMO steps."""
        if abs(float(dt) - self.coupling_interval) > 1e-9 * self.coupling_interval:
            raise ValueError(f"the column steps one coupling interval "
                             f"({self.coupling_interval} s) at a time, not {dt}")
        for _ in range(self.nsub):
            self.nemo_step()

    @property
    def surface(self):
        from woof.ocean.interface import OceanSurface

        return OceanSurface(sst=self._sst, uoce=self._uoce, voce=self._voce)

    # -- one stp_MLF -------------------------------------------------------
    def nemo_step(self) -> None:
        """One NEMO time step (stpmlf.F90's order, the column's subset)."""
        isec = self.kt * self._nint_dt              # (kt - nit000) * NINT(rn_Dt)
        received = isec % self._ncpl_sec == 0
        if received and self._fluxes is None:
            raise RuntimeError("no fluxes delivered for this coupling time (set_fluxes)")
        self.kt += 1
        if self.l_1st_euler:                                     # :114-116
            self.rDt = self.rn_Dt
        g, ts, k = self.g, self.ts, _module()
        bb, nn, aa = self.nbb, self.nnn, self.naa
        grid, dims = self._grid(), self._dims
        f = self._fluxes
        k.get_function("nemo_sbc_rcv")(grid, (_TPB,), (
            ts[nn, 0], ts[nn, 1], g["tmask"], np.int32(self.use_ct), np.int32(received),
            f.evap_precip, f.surf_net_solar, f.surf_net_non_solar, f.taux, f.tauy, f.taumod,
            self.frcv, self.sst_m, self.sss_m, self.utau, self.vtau, self.taum, self.wndm,
            self.emp, self.qns, self.qsr, np.float64(RCP), *dims))
        r = self.rab
        self.eos.rab(ts[bb, 0], ts[bb, 1], g["gdept"], g["tmask"], r["ab"], r["bb"])
        self.eos.rab(ts[nn, 0], ts[nn, 1], g["gdept"], g["tmask"], r["an"], r["bn"])
        self.eos.bn2(ts[bb, 0], ts[bb, 1], r["ab"], r["bb"], g["gdept"], g["gdepw"], g["e3w"],
                     g["wmask"], self.rn2b)
        self.eos.bn2(ts[nn, 0], ts[nn, 1], r["an"], r["bn"], g["gdept"], g["gdepw"], g["e3w"],
                     g["wmask"], self.rn2)
        # zdf_phy
        self.tke.sh2(avm=self.avm_k, uu_b=self.uu, uu_n=self.uu, vv_b=self.vv, vv_n=self.vv,
                     e3uw=g["e3w"], e3vw=g["e3w"], wumask=g["wumask"], wvmask=g["wvmask"],
                     umask=g["umask"], vmask=g["vmask"], sh2=self.sh2)
        if self.drag in (2, 3):
            self._drag_cdu()
        self.tke.step(sh2=self.sh2, avm=self.avm_k, avt=self.avt_k, en=self.en, dissl=self.dissl,
                      rn2=self.rn2, rn2b=self.rn2b, taum=self.taum, rCdU_bot=self.rCdU_bot,
                      uu_b=self.uu, vv_b=self.vv, umask=g["umask"], vmask=g["vmask"],
                      tmask=g["tmask"], wmask=g["wmask"], ssmask=g["ssmask"], mbkt=g["mbkt"],
                      e3t=g["e3t"], e3w=g["e3w"], gdepw=g["gdepw"], avmb=self.avmb,
                      avtb=self.avtb, avtb_2d=self.avtb_2d)
        jpk = self.jpk
        self.avt[1:jpk - 1, 1:-1, 1:-1] = self.avt_k[1:jpk - 1, 1:-1, 1:-1]
        self.avm[1:jpk - 1, 1:-1, 1:-1] = self.avm_k[1:jpk - 1, 1:-1, 1:-1]
        # (avs = avt: tra_zdf reuses avt's matrix for salinity without ddm)
        # the TRA loop: ts(Nrhs) = 0, held apart from ts(Naa) (tra_zdf reads
        # each RHS word before it writes the after word at that level)
        self.rhs[:, :jpk - 1, 1:-1, 1:-1] = 0.0
        k.get_function("nemo_tra_sbc")(grid, (_TPB,), (
            self.qns, self.qsr, self.emp, self.sfx, self.sbc_tsc, self.sbc_tsc_b, ts[nn, 0],
            ts[nn, 1], g["e3t"], self.rhs[0], self.rhs[1], np.int32(self.ln_traqsr),
            np.int32(self.kt == 1), np.float64(self.r1_rho0_rcp), np.float64(self.r1_rho0),
            *dims))
        if self.ln_traqsr:
            self.qsr_solver.step(self.qsr, self.rhs[0])
        self.zdf.p2dt = self.rDt
        self.zdf.step(avt=self.avt, e3t=g["e3t"], e3w=g["e3w"], tmask=g["tmask"], tb=ts[bb, 0],
                      sb=ts[bb, 1], trhs=self.rhs[0], srhs=self.rhs[1], ta=ts[aa, 0],
                      sa=ts[aa, 1])
        if not self.l_1st_euler:
            k.get_function("nemo_tra_atf")(grid, (_TPB,), (
                ts[bb, 0], ts[bb, 1], ts[nn, 0], ts[nn, 1], ts[aa, 0], ts[aa, 1],
                np.float64(self.rn_atfp), *dims, np.int32(jpk)))
        # :403-406 (Naa and Nrhs share a level in NEMO)
        self.nbb, self.nnn, self.naa = nn, aa, bb
        self._send_sst()
        if self.l_1st_euler:                                     # :467-470
            self.rDt = 2.0 * self.rn_Dt
            self.l_1st_euler = False
