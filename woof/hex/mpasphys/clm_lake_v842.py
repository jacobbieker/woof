"""WRF's CLM lake on the MPAS-A v8.4.2 physics path (WOOF-only service).

MPAS-A v8.4.2 has no lake model: its runtime folds the with-lakes MODIS lake class
(21) to water (runtime_v842.fold_lake_class) and its schemes treat a lake as any
other water cell.  This service is WRF's ``sf_lake_physics = 1``
(phys/module_sf_lake.F v4.7.1, through woof.core.lake: ``initialize_lake`` is
WRF's lakeini, ``LakeModel.step`` WRF's ``Lake``) carried to MPAS as a glue
service in the physics-suite profile (``mpasphys_v842.services.clm_lake``,
backend selector ``config_clm_lake_backend`` in ``&hex_mpasphys``, default
``off``).  The lake column words are woof.core.lake's, with no copy of the lake
physics here: this module only maps MPAS's words to the lake's arguments and the
lake's outputs back to MPAS's pools.

Placement.  WRF's surface driver runs the lake after the land surface, the sea
ice and the surface diagnostics, on the lake columns only, and its outputs
replace the surface fluxes and skin there before the PBL reads them
(WRF v4.7.1 phys/module_surface_driver.F:4394-4426, after the LSM blocks that
end with their SFCDIAGS; RAINBL is reset after the lake, :4428-4440).  In the
MPAS driver (driver_v842.begin_step) the ``clm_lake`` glue therefore follows the
land-surface block (Noah or Noah-MP, then sea ice, then
``atmphys_sfc_diagnostics``; MPAS v8.4.2 mpas_atmphys_driver.F) and the TC spray,
and precedes the PBL, on the steps the land surface runs.  The lake's own
T2/TH2/Q2 therefore stand over the MPAS surface diagnostics, as WRF's overwrite
SFCDIAGS's.  ``clm_lake_init`` (WRF's lakeini, module_sf_lake.F lakeini, called
from phy_init) runs once, at the start of the first step, before any surface
scheme: lakes stay water for MPAS's own schemes, and a lake frozen in the initial
state is handed from the sea-ice branch to lake water there (lakeini sets
ivgtyp = ISWATER, xland = 2, xice = 0 where xice > xice_threshold), exactly
as WRF real + lakeini do.

Inputs, the words MPAS's own land-surface driver hands Noah
(MPAS v8.4.2 mpas_atmphys_driver_lsm.F lsm_from_MPAS :357-480 and the
``lsm`` call :704-726) in the place of WRF's Lake arguments
(module_surface_driver.F:4400-4403):

* ``t_phy``    <- ``t_p(k=1)``         (Noah's t3d = t_p)
* ``p8w(1:2)`` <- ``pres2_hyd_p(k=1:2)`` (Noah's p8w3d = pres2_hyd_p: MPAS hands
  its LSM the hydrostatic interface pressure where WRF passes p8w)
* ``dz8w``     <- ``dz_p(k=1)``        (Noah's dz8w = dz_p)
* ``qv_curr``  <- ``qv_p(k=1)``        (Noah's qv3d = qv_p)
* ``u_phy, v_phy`` <- ``u_p(k=1), v_p(k=1)`` (the cell-centre winds MPAS_to_physics
  forms, as the surface layer reads them; Noah reads no wind)
* ``glw``      <- diag_physics ``glw``
* ``emiss``    <- diag_physics ``sfc_emiss``
* ``swdown``   <- ``gsw/(1 - sfc_albedo)``, MPAS's own SWDOWN for its LSM
  (lsm_from_MPAS :470-474; MPAS keeps no other downward surface shortwave the
  LSM reads), in single precision
* ``rainbl``   <- ``0 + rainncv + raincv`` in mm, each term only when its
  scheme is on, MPAS's own RAINBL for its LSM (lsm_from_MPAS :441-466): the
  precipitation of the step, the land surface's interval when it runs every step
* ``dtbl``     <- the land surface's step (Noah's ``dt = dt_pbl``, :726), which is
  the dynamics step here (config_pbl_interval is refused with the lake on)
* ``albedo``   <- diag_physics ``sfc_albedo`` (INOUT in WRF's Lake)
* ``xlat_urb2d`` <- ``latCell`` in degrees (WRF XLAT; Lake converts back,
  module_sf_lake.F:324)
* ``xice_threshold``: MPAS's own (0.5, or 0.02 with config_frac_seaice);
  ``iswater``: sfc_input ``iswater``.

Outputs, MPAS's names for WRF's (Lake's INOUT/OUT list :4422-4423, plus
ALBEDO): ``hfx, lh, grdflx, qfx`` (diag_physics), ``tsk -> skintemp``
(sfc_input: the surface layer reads skintemp over water next step,
mpas_atmphys_driver_sfclayer.F:428, :481, and over a lake cell, whose xice is
zero after lakeini, it is not replaced by sst), ``t2 -> t2m``,
``th2 -> th2m``, ``q2`` and ``albedo -> sfc_albedo`` (diag_physics).

Statics.  The lake mask and bathymetry are the run's surface statics sidecar
(``<mesh>.surface_statics.nc``, ``woof hex surface-statics``; LAKEMASK,
LAKEFLAG, LAKE_DEPTH, LAKE_DEPTH_FLAG), read by woof.hex.surface_inputs.bind,
the same reader and checks as the WOOF column route's CLM lake, with WRF's
namelist defaults use_lakedepth = 1 and lakedepth_default = 50 m.  A run with
the service on and no sidecar is refused by name.  Every LAKEMASK cell must be
water (or ice-covered water) in the run's own state.

Refused with the service on (by name, at configuration or at run start): sst
and sst-skin updates (they rewrite the lake skin and ice: WRF's own SST update
hands lake ice back to the lake every interval, module_surface_driver.F:1644-1661,
which this route does not carry), the OML1D/PWP ocean slot, config_pbl_interval
(RP6's precipitation window), no land surface, native-file restarts, checkpoints
and moving regions (the lake columns have no carrier yet), same-host ranks,
local time stepping and DA cycling.  The ERA5/GLSEA lake initial state
(lake_state_source) is not bound on this route: lakeini starts every column from
the run's skin temperature, WRF's default.
"""
from __future__ import annotations

import numpy as np

F = np.float32
SERVICE = "clm_lake"
INITIALIZE = "clm_lake_init"
OPERATION = "clm_lake"
#: WRF's namelist defaults for the lake (Registry.EM_COMMON use_lakedepth, lakedepth_default).
USE_LAKEDEPTH = 1
LAKEDEPTH_DEFAULT = 50.0
#: Lake output (woof.core.lake_schema.LAKE_OUTPUT_LAYOUT) -> the MPAS pool field it replaces on lake cells.
OUTPUTS = (("hfx", "diag_physics", "hfx"), ("lh", "diag_physics", "lh"),
           ("grdflx", "diag_physics", "grdflx"), ("tsk", "sfc_input", "skintemp"),
           ("qfx", "diag_physics", "qfx"), ("t2", "diag_physics", "t2m"),
           ("th2", "diag_physics", "th2m"), ("q2", "diag_physics", "q2"),
           ("albedo", "diag_physics", "sfc_albedo"))
#: lakeini's initial-state inputs and the sfc_input field each is (seed rows of woof.core.lake).
SEED = (("tsk", "skintemp"), ("isltyp", "isltyp"), ("snow", "snow"), ("xice", "xice"))
#: The water conversion lakeini makes on a frozen lake cell (woof.core.lake.initialize_lake).
CONVERTED = ("ivgtyp", "xland", "xice")


def enabled(config) -> bool:
    """The service is selected (``config_clm_lake_backend = 'hex-cuda'``)."""
    return str(getattr(config, "service_backends", {}).get(SERVICE, "off")) != "off"


def check(config) -> None:
    """The configurations the lake service runs beside (driver_v842.resolve_namelist)."""
    if not enabled(config):
        return
    from .driver_v842 import interval_seconds
    name = "config_clm_lake_backend"
    if not config.active("land_surface"):
        raise ValueError(f"{name}: WRF runs the CLM lake inside its land-surface step, after the LSM; this "
                         "run has no land surface (config_lsm_scheme = 'off')")
    o = config.options
    if o.get("config_sst_update"):
        raise ValueError(f"{name} with config_sst_update = true is refused: MPAS's SST update rewrites the lake "
                         "cells' ice, land class and skin from the surface series, and WRF's own lake hand-back "
                         "on an SST update (module_surface_driver.F:1644-1661) is not carried on this route")
    if o.get("config_sstdiurn_update"):
        raise ValueError(f"{name} with config_sstdiurn_update = true is refused: the diurnal SST update "
                         "rewrites the skin of every water cell, the lake's skin included")
    if config.active("oml"):
        raise ValueError(f"{name} with the ocean mixed layer (config_oml1d) is refused: the OML1D/PWP slot "
                         "would step the lake cells as ocean before the lake replaces them")
    if interval_seconds(o.get("config_pbl_interval", "none")) is not None:
        raise ValueError(f"{name} with config_pbl_interval is refused: the lake reads the step's own "
                         "precipitation, not RP6's land-surface window")


def refusal(arguments, config, layout, *, moving, execution) -> str | None:
    """Why a run with the lake service cannot carry its lake columns, or None (runtime_v842)."""
    if getattr(config, "ocean_pwp", None):
        return ("config_clm_lake_backend with --ocean-mode pwp is refused: the PWP columns run in the OML slot "
                "on every open-water cell this route does not mark as a lake")
    if layout.configs.get("config_do_restart"):
        return ("config_clm_lake_backend: a native-file restart would start the lake columns again from "
                "lakeini, discarding their water, ice and sediment heat; the Registry restart stream holds no "
                "lake state. Run the lake from a cold start")
    if moving is not None:
        return ("config_clm_lake_backend: checkpoints, restarts from them and moving regions carry no lake "
                "columns yet (moving_v842's carrier holds none); drop --checkpoint-hours/--restart-from and "
                "the moving switches, or the lake service")
    if execution is not None or getattr(arguments, "samehost_devices", None):
        return ("config_clm_lake_backend: the lake service runs on one card's owned cells; same-host ranks, "
                "local time stepping and DA cycling carry no lake columns yet")
    return None


def statics_path(arguments):
    """The run's surface statics sidecar: --surface-statics, else beside --grid/--static."""
    from pathlib import Path
    from ..surface_inputs import default_candidates
    given = getattr(arguments, "surface_statics", None)
    if given is not None:
        return Path(given).expanduser().absolute()
    grid = getattr(arguments, "grid", None) or getattr(arguments, "init", None)
    mesh = getattr(arguments, "mesh", None)
    if grid is None or mesh is None:
        return None
    candidates = default_candidates(grid, getattr(arguments, "static", None), mesh)
    return next((path for path in candidates if path.is_file()), None)


def bind_statics(native, mesh, layout, arguments) -> dict:
    """Read LAKEMASK/LAKE_DEPTH and their flags for this run onto ``native.clm_lake_statics``.

    The reader is the WOOF column route's (woof.hex.surface_inputs.bind, CLM lake with WRF's defaults): the
    sidecar's schema, its cells against this mesh's and its parent static against --static are held there."""
    from woof.core.mpas_column_suite import MPAS_DEFAULT_PHYSICS
    from ..surface_inputs import bind
    path = statics_path(arguments)
    if path is None or not path.is_file():
        raise ValueError(
            "config_clm_lake_backend = 'hex-cuda' reads the lake mask and bathymetry (LAKEMASK, LAKE_DEPTH) from "
            f"the mesh's surface statics, and {'none was found' if path is None else str(path) + ' does not exist'}"
            "; build it with `woof hex surface-statics --static <static> --geog-root <WPS_GEOG> -o "
            "<mesh>.surface_statics.nc` and pass --surface-statics")
    physics = dict(MPAS_DEFAULT_PHYSICS, sf_lake_physics=1, use_lakedepth=USE_LAKEDEPTH)
    inputs, receipt = bind(physics, path, latitude_rad=mesh.arrays["latCell"],
                           longitude_rad=mesh.arrays["lonCell"],
                           static_path=getattr(arguments, "static", None))
    count = int(layout.dimensions["nCellsSolve"])
    statics = dict(lakemask=np.ascontiguousarray(inputs["lakemask"][:count], dtype=F),
                   lake_depth=np.ascontiguousarray(inputs["lake_depth"][:count], dtype=F),
                   lake_depth_flag=int(inputs["lake_depth_flag"]),
                   lake_mask_flag=int(inputs["lake_mask_flag"]))
    native.clm_lake_statics = statics
    return dict(path=receipt.get("path"), file_sha256=receipt.get("file_sha256"),
                lake_cells=int(np.count_nonzero(statics["lakemask"] == 1)),
                use_lakedepth=USE_LAKEDEPTH, lakedepth_default=LAKEDEPTH_DEFAULT,
                initial_state="lakeini from the run's skin temperature (lake_state_source not bound)")


def _owned(cp, array, nc):
    """An owned-cell copy of a pool vector (the lake writes its own working copies)."""
    return cp.array(cp.asarray(array)[:nc], copy=True)


def _surface(cp, array, nc):
    return cp.ascontiguousarray(cp.asarray(array, dtype=cp.float32)[:nc])


class LakeColumns:
    """One rank's CLM lake columns on the MPAS owned cells: woof.core.lake's state and kernels."""

    def __init__(self, cp, *, statics, nc):
        self.cp = cp
        self.nc = int(nc)
        mask = np.asarray(statics["lakemask"], dtype=F)
        if mask.shape != (self.nc,):
            raise ValueError(f"the lake mask covers {mask.shape} cells, this rank owns {self.nc}")
        self.statics = statics
        self.indices = np.flatnonzero(mask == 1)
        self.fields = None
        self.model = None
        self._swdown = self._rain = None
        self.steps = 0

    @property
    def column_count(self) -> int:
        return int(self.indices.size)

    def initialize(self, provider):
        """WRF lakeini on the lake cells from MPAS's initial surface (woof.core.lake.initialize_lake)."""
        from woof.core.lake import initialize_lake
        cp, nc = self.cp, self.nc
        sfc, mesh = provider["sfc_input"], provider["mesh"]
        fields = {"lakemask": cp.asarray(self.statics["lakemask"])}
        for name, source in SEED:
            fields[name] = _owned(cp, sfc[source], nc)
        for name in ("ivgtyp", "xland"):
            fields[name] = _owned(cp, sfc[name], nc)
        lat = cp.asnumpy(cp.asarray(mesh["latCell"])[:nc]).astype(np.float64)
        latitude = cp.asarray(np.degrees(lat).astype(F))
        threshold = float(provider["scalar_state"]["xice_threshold"])
        iswater = int(provider.native.get_field("sfc_input", "iswater").item())
        xland = cp.asnumpy(fields["xland"])[self.indices]
        xice = cp.asnumpy(fields["xice"])[self.indices]
        land = (xland < 1.5) & ~(xice >= threshold)
        if land.any():
            raise ValueError(f"the surface statics' LAKEMASK marks {int(land.sum())} cells this run holds as "
                             "land (xland = 1, no ice); a lake is a water cell (real.exe's ISLAKE -> ISWATER), so "
                             "the sidecar was built for another land use")
        self.model = initialize_lake(
            fields, latitude=latitude, lake_depth=cp.asarray(self.statics["lake_depth"]),
            lake_depth_flag=int(self.statics["lake_depth_flag"]), use_lakedepth=USE_LAKEDEPTH,
            lakedepth_default=LAKEDEPTH_DEFAULT, iswater=iswater, xice_threshold=threshold)
        for name in CONVERTED:
            target = sfc[name]
            target[:nc] = fields[name]
        self.fields = {name: fields[name] for name in ("lakemask", "lake_columns", "lake_static",
                                                       "lake_latitude")}
        self.lake_indices = cp.asarray(self.indices.astype(np.int32))
        self._swdown = cp.zeros(nc, cp.float32)
        self._rain = cp.zeros(nc, cp.float32)
        return self.model

    def forcing(self, provider, call):
        """MPAS's own SWDOWN and RAINBL for its land surface (lsm_from_MPAS), on the lake cells, in single
        precision: ``gsw/(1 - sfc_albedo)`` and ``0 + rainncv + raincv``."""
        cp = self.cp
        diag = provider["diag_physics"]
        at = self.lake_indices
        take = lambda a: cp.asnumpy(cp.asarray(a, dtype=cp.float32)[:self.nc][at])
        swdown = take(diag["gsw"]) / (F(1.0) - take(diag["sfc_albedo"]))
        rain = np.zeros(self.column_count, F)
        if call.options["config_microp_scheme"] != "off":
            rain = rain + take(diag["rainncv"])
        if call.options["config_convection_scheme"] != "off":
            rain = rain + take(diag["raincv"])
        self._swdown[self.lake_indices] = cp.asarray(swdown.astype(F))
        self._rain[self.lake_indices] = cp.asarray(rain.astype(F))
        return self._swdown, self._rain

    def step(self, provider, call):
        """WRF's Lake on the lake cells; its outputs replace MPAS's surface words there."""
        cp, nc = self.cp, self.nc
        if self.model is None:
            raise RuntimeError("the CLM lake stepped before clm_lake_init")
        if not self.column_count:
            self.steps += 1
            return None
        prepared, diag = provider["prepared"], provider["diag_physics"]
        pools = {"diag_physics": diag, "sfc_input": provider["sfc_input"]}

        def levels(name, count):
            return cp.ascontiguousarray(cp.asarray(prepared[name], dtype=cp.float32)[:count, :nc])
        atmosphere = {"temperature": levels("t_p", 1), "p_interface": levels("pres2_hyd_p", 2),
                      "dz": levels("dz_p", 1), "qv": levels("qv_p", 1),
                      "u": levels("u_p", 1), "v": levels("v_p", 1)}
        swdown, rain = self.forcing(provider, call)
        fields = dict(self.fields, glw=_surface(cp, diag["glw"], nc),
                      emiss=_surface(cp, diag["sfc_emiss"], nc), swdown=swdown)
        for name, pool, source in OUTPUTS:
            fields[name] = _owned(cp, pools[pool][source], nc)
        self.model.step(fields, atmosphere, dt=float(call.dt), precipitation=rain)
        for name, pool, source in OUTPUTS:
            target = pools[pool][source]
            target[:nc] = fields[name]
        self.steps += 1
        return fields

    def receipt(self):
        return dict(service=SERVICE, lake_columns=self.column_count, steps=self.steps,
                    use_lakedepth=USE_LAKEDEPTH, lakedepth_default=LAKEDEPTH_DEFAULT,
                    source="WRF v4.7.1 phys/module_sf_lake.F through woof.core.lake")


def lake_init_step(provider, call):
    """The ``clm_lake_init`` glue operation on the session's device pool views."""
    session = provider.session
    statics = getattr(session.native, "clm_lake_statics", None)
    if statics is None:
        raise RuntimeError("config_clm_lake_backend = 'hex-cuda' has no lake statics bound: the run did not read "
                           "the mesh's surface statics (LAKEMASK, LAKE_DEPTH); pass --surface-statics")
    columns = LakeColumns(provider.cp, statics=statics,
                          nc=int(session.layout.dimensions["nCellsSolve"]))
    columns.initialize(provider)
    return columns


class ClmLakeService:
    """Binds ``clm_lake_init`` and ``clm_lake`` to a physics session's device pool views."""

    def __init__(self, session):
        self.session = session
        self.columns = None

    def _run(self, call, body):
        provider = self.session.device_pools
        provider.begin(call, {})
        try:
            return body(provider)
        finally:
            provider.commit()

    def initialize(self, call):
        if self.columns is not None:
            raise RuntimeError("clm_lake_init ran twice: lakeini would start the lake columns again")
        self.columns = self._run(call, lambda provider: lake_init_step(provider, call))

    def step(self, call):
        if self.columns is None:
            raise RuntimeError("the CLM lake stepped before clm_lake_init")
        return self._run(call, lambda provider: self.columns.step(provider, call))

    def receipt(self):
        return None if self.columns is None else self.columns.receipt()

    def callbacks(self):
        return {INITIALIZE: self.initialize, OPERATION: self.step}
