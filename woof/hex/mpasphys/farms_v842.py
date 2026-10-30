"""WRF-Solar FARMS on the MPAS-A v8.4.2 physics path (WOOF-only service).

MPAS v8.4.2 has no FARMS.  This is WRF's ``swint_opt = 2``
(woof.core.farms; WRF v4.7.1 phys/module_ra_farms.F) carried to MPAS as a
glue service in the physics-suite profile
(``mpasphys_v842.services.farms_sw``, backend selector
``config_farms_sw_backend`` in ``&hex_mpasphys``, default ``off``).  With it
on, the physics driver calls ``farms_sw`` on EVERY step, radiation steps
included, right after the radiation slots (driver_v842.begin_step), and
FARMS overwrites the four surface shortwave diagnostics the RRTMG
shortwave slot writes:

    swdnb   <- FARMS GHI     (WRF SWDOWN2)
    swddir  <- FARMS direct  (WRF SWDDIR2)
    swddni  <- FARMS DNI     (WRF SWDDNI2)
    swddif  <- FARMS diffuse (WRF SWDDIF2)

In MPAS v8.4.2 no scheme reads these four between radiation calls: Noah
takes ``gsw`` (the radiation call's net shortwave, untouched here) and the
shortwave accumulators are summed from the radiation call's own module
arrays, so the service is diagnostic -- no prognostic word moves.  (WRF's
``couple_farms`` copies SWDOWN2 into SWDOWN, which WRF's Noah reads; MPAS's
Noah has no such input, so there is no coupled variant here.)

Inputs, per step, MPAS's own words:

* the prepared physics arrays MPAS_to_physics fills every step: the surface
  hydrostatic interface pressure ``pres2_hyd_p(k=1)`` (the p8w the MPAS
  RRTMG wrapper takes), the moist density ``rho_p`` (= zz*rho_zz*(1+qv), so
  FARMS's ``rho*dz/(1+qv)`` is the dry layer mass as in WRF), ``dz_p``,
  ``qv_p``, ``qc_p``, ``qi_p``, ``qs_p``;
* the last radiation call's cloud fraction ``diag_physics.cldfrac`` (MPAS
  forms it only on radiation steps -- WRF re-forms CLDFRA on every step
  under swint_opt = 2, MPAS has no per-step cloudiness to call);
* the microphysics' effective radii ``re_cloud/re_ice/re_snow`` in metres,
  exactly the words the MPAS shortwave reads (``config_microp_re`` with
  Thompson or WSM6: has_reqc/i/s = 1; otherwise FARMS's 8/24/24 um
  climatology);
* ``sfc_albedo``; zero aerosol (WRF aer_opt = 0);
* the sun at the step's own time by MPAS's radiation geometry
  (``farms_mpas_coszen``), and ``juday = INT(curr_julday)``.

The column kernel is woof/core/kernels/farms.cu's ``farms_columns``, held to
compiled WRF v4.7.1 word for word (tests/test_farms_wrf471_parity.py) and,
in MPAS's [level, cell] layout through :func:`farms_mpas`, by
components/hex/tests/test_mpas_farms_v842.py.
"""
from __future__ import annotations

import numpy as np

from woof.core.farms import farms_columns

F = np.float32
SERVICE = "farms_sw"
OPERATION = "farms_sw"
#: FARMS output -> the MPAS diag_physics field it overwrites.
OVERWRITES = (("swdown2", "swdnb"), ("swddir2", "swddir"),
              ("swddni2", "swddni"), ("swddif2", "swddif"))


def mpas_coszen(cp, *, lat_cell, lon_cell, julian, xtime_min, gmt, degrad, dpd):
    """The step's zenith cosine by MPAS's radiation geometry, (nc,)."""
    from woof.core.kernels import get_kernel
    lat = cp.ascontiguousarray(lat_cell, dtype=cp.float32)
    lon = cp.ascontiguousarray(lon_cell, dtype=cp.float32)
    nc = int(lat.size)
    out = cp.empty(nc, dtype=cp.float32)
    if nc:
        get_kernel("farms", "farms_mpas_coszen")(
            ((nc + 127) // 128,), (128,),
            (np.int32(nc), lat, lon, F(julian), F(xtime_min), F(gmt),
             F(degrad), F(dpd), out))
    return out


def farms_mpas(cp, *, nc, nl, p8w, rho, dz, qv, qc, qi, qs, cldfrac,
               re_cloud, re_ice, re_snow, albedo, coszen, has_req, juday):
    """FARMS on MPAS [level, cell] arrays: the first ``nc`` cells of every
    array, levels 1..nl (nl = nVertLevels, the WRF kte).  ``p8w`` is the
    surface interface pressure per cell.  Returns the six outputs, (nc,)."""
    def prof(a):
        return cp.ascontiguousarray(cp.asarray(a, dtype=cp.float32)[:nl, :nc])

    def surf(a):
        return cp.ascontiguousarray(cp.asarray(a, dtype=cp.float32)[:nc])

    zeros = cp.zeros(nc, dtype=cp.float32)
    return farms_columns(
        p8w_sfc=surf(p8w), albedo=surf(albedo), coszen=surf(coszen),
        aerssa=zeros, aerasy=zeros, aod550=zeros, angexp=zeros,
        rho=prof(rho), dz8w=prof(dz), qv=prof(qv), qc=prof(qc), qi=prof(qi),
        qs=prof(qs), re_cloud=prof(re_cloud), re_ice=prof(re_ice),
        re_snow=prof(re_snow), cldfra=prof(cldfrac), has_req=has_req,
        juday=juday)


def radii_declared(options) -> bool:
    """The MPAS shortwave's own rule for reading the scheme radii."""
    return bool(options["config_microp_re"]) and options["config_microp_scheme"] in (
        "mp_thompson", "mp_thompson_aerosols", "mp_wsm6")


def farms_sw_step(provider, call):
    """The ``farms_sw`` glue operation on the session's device pool views."""
    cp = provider.cp
    nl = int(provider.session.layout.dimensions["nVertLevels"])
    nc = int(provider.session.layout.dimensions["nCellsSolve"])
    prepared, diag, mesh = provider["prepared"], provider["diag_physics"], provider["mesh"]
    constants = provider["constants"]
    scalars = provider["scalar_state"]
    julian = provider.native.get_table("mpas_atmphys_manager.curr_julday")[0][()]
    gmt = provider.native.get_table("mpas_atmphys_manager.gmt")[0][()]
    degrad, dpd = F(constants["degrad"]), F(constants["dpd"])
    coszen = mpas_coszen(cp, lat_cell=mesh["latCell"][:nc], lon_cell=mesh["lonCell"][:nc],
                         julian=julian, xtime_min=F(call.time_seconds) / F(60.0),
                         gmt=gmt, degrad=degrad, dpd=dpd)
    declared = radii_declared(call.options)
    has_req = tuple(int(scalars[n]) if declared else 0
                    for n in ("has_reqc", "has_reqi", "has_reqs"))
    zeros = cp.zeros((nl, nc), dtype=cp.float32)
    radii = {n: (diag[n] if declared else zeros) for n in ("re_cloud", "re_ice", "re_snow")}
    out = farms_mpas(
        cp, nc=nc, nl=nl, p8w=prepared["pres2_hyd_p"][0], rho=prepared["rho_p"],
        dz=prepared["dz_p"], qv=prepared["qv_p"], qc=prepared["qc_p"],
        qi=prepared["qi_p"], qs=prepared["qs_p"], cldfrac=diag["cldfrac"],
        albedo=diag["sfc_albedo"], coszen=coszen, has_req=has_req,
        juday=int(np.trunc(F(julian))), **radii)
    for source, target in OVERWRITES:
        diag[target][:nc] = out[source]
    return out


class FarmsService:
    """Binds ``farms_sw`` to a physics session's device pool views."""

    def __init__(self, session):
        self.session = session

    def __call__(self, call):
        provider = self.session.device_pools
        provider.begin(call, {})
        try:
            return farms_sw_step(provider, call)
        finally:
            provider.commit()

    def callbacks(self):
        return {OPERATION: self}
