"""WRF-Solar FARMS: ``swint_opt = 2`` surface irradiance on every step.

Authority: WRF v4.7.1 (commit f52c197ed) ``phys/module_ra_farms.F`` (sha256
1b9e060d..., byte-identical in v4.6.1) and the radiation driver that calls
it, ``phys/module_radiation_driver.F`` (sha256 77987565..., also identical in
v4.6.1).  Graded word for word by tools/farms_wrf471_oracle and
tests/test_farms_wrf471_parity.py.

What WRF does under ``swint_opt = 2``
------------------------------------
FARMS is NOT a shortwave scheme (``ra_sw_physics`` keeps its own choice and
the radiation call still produces SWDOWN, GSW and the heating rates).  It is a
surface-irradiance model the driver evaluates on EVERY model step, radiation
steps included, after ``ENDIF Radiation_step`` (:2983-3101):

* ``coszen_loc``: ``radconst`` + ``calc_coszen`` at the step's own xtime
  (:1158-1170, shared with ``swint_opt = 1``);
* the "Solar_step" block (:1176-1609) runs on every step too, so the cloud
  fraction CLDFRA (``icloud``), and the temporary hydrometeor additions that
  go with it (MYNN's QC_BL/QI_BL under ``icloud_bl``), are those of the
  CURRENT state, not the last radiation call's.  The heating rates and the
  radiation call's fluxes are untouched;
* ``Farms_driver`` (:3064-3068) reads, per column: the surface full-level
  pressure ``p8w(i,kts,j)``, ``rho``, ``dz8w``, ``qv/qc/qi/qs``, the
  microphysics' effective radii ``re_cloud/re_ice/re_snow`` in metres (with
  ``has_reqc/i/s``; without them the 8/24/24 um climatology), CLDFRA, the
  live ALBEDO, and four 2-D aerosol words: ``aer_opt = 0`` zeros them
  (:2996-3011); ``aer_opt = 1`` takes the last radiation call's AODTOT and
  the namelist ``aer_angexp_val``, ``aer_ssa_val``, ``aer_asy_val``
  (:3013-3030).  It needs nothing else from RRTMG: FARMS computes its own
  cloud optical depths from the paths and path-weighted radii (Matrosov
  et al. 2002 for ice and snow) and its own clear sky (Bird 1981);
* it writes SWDOWN2, SWDDIR2, SWDDNI2, SWDDIF2, SWDOWNC2 and SWDDNIC2 --
  separate fields; SWDOWN/SWDDIR/SWDDNI/SWDDIF/GSW are NOT overwritten --
  and, only under ``couple_farms = .true.`` (which module_check_a_mundo
  admits only with ``swint_opt = 2``), copies SWDOWN2 into SWDOWN
  (:3084-3099).  That copy is FARMS's only feedback: the land surface reads
  SWDOWN (Noah: SOLDN = SWDOWN, SOLNET = SOLDN*(1-ALBEDO),
  module_sf_noahdrv.F:830-834), not GSW, which keeps the radiation call's
  value.

Two side effects of ``swint_opt = 2`` that are not FARMS: the Solar_step
block also rewrites the COSZEN state at ``xtime + radt/2`` and CLDFRA on
every step (between radiation calls WRF leaves both at the last call's
values otherwise).  :meth:`FarmsShortwave.between_calls` reproduces the
COSZEN rewrite for a driver that carries COSZEN (Noah-MP, RUC, urban read
it) and the CLDFRA rewrite for the UW PBL's ``uw_cldfra``, the one carried
CLDFRA a scheme reads between radiation calls.

The radii seam
--------------
WRF hands FARMS the microphysics' metre words.  woof's state carries the
radiation-facing MICRON words (``effc/effi/effs``, the metre value times
1e6 rounded once, as WRF's RRTMG wrapper forms them); this module takes
``micron / 1e6`` (rounded once), which returns the metre word exactly for
95.6 % of the radii a scheme's clamps admit and is one ULP off for the
rest (every binary32 radius in Thompson's [2.49, 50], [4.99, 125] and
[9.99, 999] um ranges; ``* 1e-6`` would be one ULP off for 5.2 %).  The
kernel itself is exact on metre inputs; the seam is the measured distance.

Numerics: woof/core/kernels/farms.cu, explicit single-rounded arithmetic and
woof's own binary32 libm (glibc's expf/powf/log10f/cosf/sinf/acosf and a
correctly rounded sinhf), compiled without FMA contraction or FTZ.
"""
from __future__ import annotations

import numpy as np

from woof.core.swint import ShortwaveInterpolation, calendar_scalars

F = np.float32

#: The six 2-D float32 fields FARMS writes, WRF's names (Registry.EM_COMMON
#: :1710-1726, all restart-carried; SWDOWN2, SWDOWNC2, SWDDNIC2 also in
#: history), in the driver's ``fields`` dict so the restart stream carries
#: them in place.
STATE_FIELDS = ("swdown2", "swddir2", "swddni2", "swddif2", "swdownc2",
                "swddnic2")

#: module_physics_init.F:1039-1057: FARMS needs both effective radii and the
#: mass of cloud, ice and snow.  These schemes pass by name (THOMPSON,
#: RCON_MP_SCHEME, THOMPSONAERO, WSM3/5/6/7, WDM5/6/7); any other passes
#: only with has_reqc = has_reqi = has_reqs = 0 and F_QC, F_QI and F_QS.
FARMS_NAMED_MP = frozenset({8, 28, 3, 4, 6, 24, 14, 16, 26})
RCON_MP_SCHEME = 38

_BLOCK = 128


def farms_mp_admitted(mp_physics: int, use_mp_re: int = 1) -> bool:
    """WRF's FARMS microphysics check (module_physics_init.F:1039-1057)."""
    from woof.core.rrtmg_legacy import (legacy_cloud_fraction_flags,
                                        legacy_scheme_has_req)
    mp = int(mp_physics)
    if mp in FARMS_NAMED_MP or mp == RCON_MP_SCHEME:
        return True
    try:
        has_req = legacy_scheme_has_req(mp, use_mp_re)
        f_qi, f_qs = legacy_cloud_fraction_flags(mp)
    except NotImplementedError:
        return False
    return tuple(has_req) == (0, 0, 0) and mp != 0 and f_qi and f_qs


def radius_metres(micron):
    """The metre radius FARMS reads, from woof's micron state word:
    ``micron / 1e6`` rounded once (see the module docstring)."""
    import cupy as cp
    return cp.divide(cp.asarray(micron, dtype=cp.float32), F(1.0e6))


def farms_columns(*, p8w_sfc, albedo, coszen, aerssa, aerasy, aod550,
                  angexp, rho, dz8w, qv, qc, qi, qs, re_cloud, re_ice,
                  re_snow, cldfra, has_req, juday, out=None):
    """``Farms_driver`` on device columns.

    2-D inputs are float32 device arrays of ``ncol`` words (any shape);
    3-D inputs are ``(nz, *columns)`` float32, C-contiguous, the radii in
    METRES.  ``has_req`` is WRF's ``(has_reqc, has_reqi, has_reqs)`` and
    ``juday`` WRF's ``INT(julian)``.  Returns a dict of the six outputs
    (shape of ``albedo``), written into ``out`` when given.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel

    shape2 = tuple(albedo.shape)
    ncol = int(np.prod(shape2)) if shape2 else 1
    nz = int(rho.shape[0])
    if nz < 1:
        raise ValueError("FARMS needs at least one model layer")

    def flat2(a, name):
        if not isinstance(a, cp.ndarray) or a.dtype != cp.float32 \
                or a.size != ncol:
            raise ValueError(f"FARMS 2-D input {name!r} must be float32 with "
                             f"{ncol} words")
        return cp.ascontiguousarray(a).reshape(-1)

    def flat3(a, name):
        if not isinstance(a, cp.ndarray) or a.dtype != cp.float32 \
                or a.size != nz * ncol or int(a.shape[0]) != nz:
            raise ValueError(f"FARMS 3-D input {name!r} must be float32 "
                             f"({nz}, ...) with {ncol} columns")
        return cp.ascontiguousarray(a).reshape(nz, ncol)

    if out is None:
        out = {name: cp.empty(shape2, dtype=cp.float32)
               for name in STATE_FIELDS}
    for name in STATE_FIELDS:
        a = out[name]
        if a.dtype != cp.float32 or a.size != ncol or not a.flags.c_contiguous:
            raise ValueError(f"FARMS output {name!r} must be contiguous "
                             f"float32 with {ncol} words")
    hc, hi, hs = (int(v) for v in has_req)
    if ncol:
        get_kernel("farms", "farms_columns")(
            ((ncol + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
            (np.int32(ncol), np.int32(nz), np.int64(ncol), np.int32(hc),
             np.int32(hi), np.int32(hs), np.int32(juday),
             flat2(p8w_sfc, "p8w_sfc"), flat2(albedo, "albedo"),
             flat2(coszen, "coszen"), flat2(aerssa, "aerssa"),
             flat2(aerasy, "aerasy"), flat2(aod550, "aod550"),
             flat2(angexp, "angexp"),
             flat3(rho, "rho"), flat3(dz8w, "dz8w"), flat3(qv, "qv"),
             flat3(qc, "qc"), flat3(qi, "qi"), flat3(qs, "qs"),
             flat3(re_cloud, "re_cloud"), flat3(re_ice, "re_ice"),
             flat3(re_snow, "re_snow"), flat3(cldfra, "cldfra"),
             *(out[name].reshape(-1) for name in (
                 "swdown2", "swddir2", "swddni2", "swddif2", "swdownc2",
                 "swddnic2"))))
    return out


def wrf_juday(julian) -> int:
    """Fortran ``INT(julian)`` of the float32 julian (truncation)."""
    return int(np.trunc(F(julian)))


class FarmsShortwave(ShortwaveInterpolation):
    """The ``swint_opt = 2`` half of WRF's radiation driver on one domain.

    Inherits :class:`~woof.core.swint.ShortwaveInterpolation`'s per-step
    zenith cosine (the same ``radconst`` + ``calc_coszen`` at the step's
    xtime, read from ``latitude_deg``/``longitude_deg`` at each call so a
    streamed tile's gathered geography is the one used) and holds no value a
    step writes: the six outputs live in the driver's ``fields``.  Rebuilt,
    never serialized (woof/io/restart.py).
    """

    def __init__(self, *, start_time, latitude_deg, longitude_deg, shape,
                 generation="wrf_480", mp_physics, use_mp_re=1, icloud=1,
                 aer_opt=0, aer_angexp_val=1.3, aer_ssa_val=0.85,
                 aer_asy_val=0.9, couple_farms=False, bl_merge=False,
                 icloud_cu=0, deng_cloud=False, gridkm=None):
        super().__init__(start_time=start_time, latitude_deg=latitude_deg,
                         longitude_deg=longitude_deg, shape=shape,
                         generation=generation)
        from woof.core.rrtmg_legacy import (legacy_cloud_fraction_flags,
                                            legacy_scheme_has_req)
        self.mp_physics = int(mp_physics)
        if not farms_mp_admitted(self.mp_physics, use_mp_re):
            raise ValueError(
                f"swint_opt = 2 (FARMS) with mp_physics={self.mp_physics}: "
                "WRF stops with 'FARMS (swint_opt==2) requires a different "
                "MP scheme' (module_physics_init.F:1039-1057)")
        self.has_req = tuple(int(v) for v in
                             legacy_scheme_has_req(self.mp_physics, use_mp_re))
        self.f_qi, self.f_qs = legacy_cloud_fraction_flags(self.mp_physics)
        self.icloud = int(icloud)
        if self.icloud not in (0, 1, 2, 3):
            raise NotImplementedError(
                f"swint_opt = 2 with icloud={self.icloud}: FARMS reads the "
                "per-step CLDFRA, and only icloud 0, 1, 2 and 3 are ported")
        if self.icloud == 3 and gridkm is None:
            raise ValueError("swint_opt = 2 with icloud = 3 needs the "
                             "driver's gridkm (woof.core.cldfra3.cldfra3_gridkm)")
        # icloud = 3 (WRF-Solar cal_cldfra3, woof.core.cldfra3): the
        # Solar_step runs it on every step and Farms_driver
        # (module_radiation_driver.F:3064-3068) reads its CLDFRA and the qc/qi
        # it rewrote -- the restore of qc/qi comes after (:3137-3155).
        self.gridkm = None if gridkm is None else F(gridkm)
        self.aer_opt = int(aer_opt)
        if self.aer_opt not in (0, 1, 2):
            raise NotImplementedError(
                f"swint_opt = 2 with aer_opt={self.aer_opt}: FARMS's 2-D "
                "aerosol (AOD5502D/ANGEXP2D/AERSSA2D/AERASY2D) is wired for "
                "aer_opt 0, 1 and 2; under aer_opt = 3 WRF reads the "
                "radiation call's TAOD5502D and the taer_* 2-D optics")
        self.aer_values = (F(aer_angexp_val), F(aer_ssa_val), F(aer_asy_val))
        self.couple_farms = bool(couple_farms)
        self.bl_merge = bool(bl_merge)
        # The Solar_step's convective cloud (woof.core.cumulus_radiation):
        # cu_rad_feedback's ICLOUD_CU and Deng's shallow cloud, which WRF
        # forms before farms_driver reads QC/QI/CLDFRA
        # (module_radiation_driver.F:1264-1281, :1337-1348, :1566-1575,
        # then :3064), on every step under swint_opt = 2 (:1176).
        self.icloud_cu = int(icloud_cu)
        self.deng_cloud = bool(deng_cloud)

    # -- inputs ------------------------------------------------------------

    def cloud_fraction(self, atmosphere, fields, *, itimestep, eps=None):
        """The Solar_step's CLDFRA and hydrometeors for this step:
        ``(cldfra, qc, qi)`` on the device, (nz, ny, nx)."""
        import cupy as cp
        qv = atmosphere["qv"]
        qc = atmosphere["qc"]
        qi = atmosphere["qi"]
        qs = atmosphere["qs"]
        nz = qv.shape[0]
        cols = qv.size // nz
        from woof.core import cumulus_radiation as curad
        if self.icloud_cu == 1:
            # :1264-1281, before the cloud fraction (F_QC is true here).
            qc, qi = curad.add_convective_condensate(
                qc, qi, fields["qc_cu"], fields["qi_cu"], f_qc=True,
                f_qi=self.f_qi)
        cldfra = cp.zeros(qv.shape, dtype=cp.float32)
        if self.icloud == 1:
            from woof.core.rrtmgp import cal_cldfra1
            flat = lambda a: cp.ascontiguousarray(a).reshape(nz, cols)
            cldfra = cal_cldfra1(
                flat(qv), flat(qc), flat(qi), flat(qs),
                flat(atmosphere["temperature"]), flat(atmosphere["pressure"]),
                f_qc=True, f_qi=self.f_qi, f_qs=self.f_qs,
                mp_physics=self.mp_physics).reshape(qv.shape)
            if self.icloud_cu == 2:
                # :1337-1348, right after cal_cldfra1, before the MYNN merge.
                cldfra, qc, qi = curad.merge_kf_cloud(
                    cldfra, qc, qi, cldfra_dp=fields["cldfra_dp"],
                    cldfra_sh=fields["cldfra_sh"], qc_cu=fields["qc_cu"],
                    qi_cu=fields["qi_cu"])
            if self.bl_merge:
                # module_radiation_driver.F:1406-1428 (icloud_bl > 0)
                cf_bl = fields["cldfra_bl"]
                if int(itimestep) != 1:
                    cldfra = cp.ascontiguousarray(cf_bl, dtype=cp.float32).copy()
                active = cf_bl > F(0.001)
                qc = cp.where((qc < F(1.0e-6)) & active, qc + fields["qc_bl"], qc)
                qi = cp.where((qi < F(1.0e-8)) & active, qi + fields["qi_bl"], qi)
        elif self.icloud == 2:
            from woof.core.rrtmgp import cal_cldfra2
            cldfra = cal_cldfra2(cp.ascontiguousarray(qc, dtype=cp.float32),
                                 cp.ascontiguousarray(qi, dtype=cp.float32),
                                 f_qc=True, f_qi=self.f_qi)
        elif self.icloud == 3:
            from woof.core.cldfra3 import cal_cldfra3
            from woof.core.rrtm_lw import _stream_is_capturing
            ncol = cols
            tr = lambda a: cp.ascontiguousarray(
                cp.ascontiguousarray(a, dtype=cp.float32).reshape(nz, ncol).T)
            c3, q3c, q3i = cal_cldfra3(
                tr(qv), tr(qc), tr(qi), tr(qs), tr(atmosphere["dz"]),
                tr(atmosphere["pressure"]), tr(atmosphere["temperature"]),
                fields["xland"].reshape(-1), self.gridkm,
                perturbation=(None if eps is None else eps.cld3_columns(tr)),
                check=not _stream_is_capturing(cp))
            back = lambda a: cp.ascontiguousarray(a.T).reshape(qv.shape)
            cldfra, qc, qi = back(c3), back(q3c), back(q3i)
        if self.deng_cloud:
            # :1566-1575, after every icloud branch.
            cldfra, qc = curad.merge_deng_cloud(
                cldfra, qc, cldfra_sh=fields["cldfra_sh"],
                cw_rad=fields["cw_rad"])
        return cldfra, qc, qi

    def _radii(self, state, nz):
        import cupy as cp
        radii = []
        for flag, name, clim in zip(self.has_req, ("effc", "effi", "effs"),
                                    (8.0e-6, 24.0e-6, 24.0e-6)):
            value = getattr(state, name, None) if flag else None
            if flag and value is None:
                raise ValueError(
                    f"swint_opt = 2: mp_physics={self.mp_physics} declares "
                    f"{name} (has_req) but the state carries none")
            if value is None:
                # never read: the kernel takes the climatology when has_req=0
                radii.append(cp.full((nz, *self.shape), F(clim), cp.float32))
            else:
                radii.append(radius_metres(value).reshape(nz, *self.shape))
        return radii

    def _aerosol(self, aodtot, fields):
        import cupy as cp
        zeros = cp.zeros(self.shape, dtype=cp.float32)
        if self.aer_opt == 0:
            # :2996-3011, zeroed every step
            return zeros, zeros, zeros, zeros
        if self.aer_opt == 2:
            # case (2): the 2-D fields calc_aerosol_rrtmg_sw left at the
            # last radiation call (:2293-2297), read as they stand
            return tuple(fields[name] for name in (
                "AOD5502D", "ANGEXP2D", "AERSSA2D", "AERASY2D"))
        if aodtot is None:
            raise ValueError("swint_opt = 2 with aer_opt = 1 reads the last "
                             "radiation call's AODTOT, and none is carried")
        angexp, ssa, asy = self.aer_values
        return (cp.ascontiguousarray(aodtot, dtype=cp.float32),
                cp.full(self.shape, angexp, cp.float32),
                cp.full(self.shape, ssa, cp.float32),
                cp.full(self.shape, asy, cp.float32))

    # -- the every-step call -------------------------------------------------

    def step(self, fields, atmosphere, state, *, elapsed_seconds, itimestep,
             aodtot=None, eps=None):
        """FARMS for this step: the six outputs into ``fields``, then the
        ``couple_farms`` copy into SWDOWN.  Returns the step's CLDFRA.

        ``eps`` is the domain's WRF-Solar EPS provider
        (woof.core.wrfsolar_eps.MultiPerturb) or None: pert_cld3 perturbs
        cal_cldfra3's inputs; pert_farms wraps farms_driver in WRF's Add_/
        Remove_ pair (module_radiation_driver.F:3052-3082) on the arrays
        WRF's own would touch -- the albedo, qv and qs in place (what the
        round trip leaves stays, as in WRF), the 2-D aerosol in place where
        it is carried (aer_opt = 2) and on fresh copies where WRF rebuilds
        it each call, and a copy of the radiation's qc, whose Remove_ the
        driver's qc restore (:3137-3145) discards."""
        import cupy as cp
        julian, xtime, gmt = calendar_scalars(self.start_time, elapsed_seconds)
        coszen_loc = self.coszen_loc_at(julian, xtime, gmt)
        cldfra, qc, qi = self.cloud_fraction(atmosphere, fields,
                                             itimestep=itimestep, eps=eps)
        nz = atmosphere["qv"].shape[0]
        re_cloud, re_ice, re_snow = self._radii(state, nz)
        aod, angexp, ssa, asy = self._aerosol(aodtot, fields)
        if "albsol" in fields:
            albedo = fields["albsol"]
        else:
            albedo = fields["albedo"]
        farms_fac = None if eps is None else eps.factors("pert_farms")
        if farms_fac is not None:
            from woof.core import wrfsolar_eps as _eps
            if self.aer_opt != 2:
                # rebuilt by the radiation driver on every call: perturb
                # private copies (four distinct arrays, never the carried
                # AODTOT)
                aod, angexp, asy = (cp.array(a, dtype=cp.float32, copy=True)
                                    for a in (aod, angexp, asy))
            qc = cp.array(qc, dtype=cp.float32, copy=True)
            pats = eps.patterns()
            hooked = dict(albedo=albedo, aod=aod, angexp=angexp, aerasy=asy,
                          qv=atmosphere["qv"], qc=qc, qs=atmosphere["qs"])
            saved = _eps.farms_add(pats, farms_fac, **hooked)
        farms_columns(
            p8w_sfc=atmosphere["p_interface"][0], albedo=albedo,
            coszen=coszen_loc, aerssa=ssa, aerasy=asy, aod550=aod,
            angexp=angexp, rho=atmosphere["rho"], dz8w=atmosphere["dz"],
            qv=atmosphere["qv"], qc=qc, qi=qi, qs=atmosphere["qs"],
            re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow,
            cldfra=cldfra, has_req=self.has_req, juday=wrf_juday(julian),
            out={name: fields[name] for name in STATE_FIELDS})
        if farms_fac is not None:
            _eps.farms_remove(pats, farms_fac, saved, **hooked)
        if self.couple_farms:
            fields["swdown"][...] = fields["swdown2"]
        return cldfra

    def radiation_coszen_at(self, elapsed_seconds, radt_minutes):
        """The Solar_step's COSZEN on this step: ``calc_coszen`` at the
        CURRENT julian and the hour angle at ``xtime + radt*0.5``
        (module_radiation_driver.F:1206-1208), (ny, nx)."""
        julian, xtime, gmt = calendar_scalars(self.start_time, elapsed_seconds)
        xt_mid = F(xtime) + F(radt_minutes) * F(0.5)
        return self.coszen_loc_at(julian, xt_mid, gmt).reshape(self.shape)
