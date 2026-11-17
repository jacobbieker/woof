"""WRF-Hydro routing coupled to the hex land surface, through the conservative cell <-> grid exchange.

WRF built with ``WRF_HYDRO`` couples the LSM to WRF-Hydro's routing cell for cell on the WRF mass grid
(woof.wrf_hydro_route, the opt-in's ``woof_wrf_hydro.coupled.WrfHydroCoupled``).  The hex WOOF route runs the
same WRF PhysicsDriver on ``ny = 1`` columns (woof.core.mpas_column_batch), so its Noah and Noah-MP calls take
the same ``WRF_HYDRO`` planes and its surface step calls the same ``exchange`` after the land surface; what
differs is the grid.  The routing domain (its ``hydro.namelist``, ``geo_em`` and ``Fulldom_hires``) is a WRF
domain, so this seam keeps the routing on that domain's LSM grid and crosses to the hex cells through
:class:`woof.hex.hydro_remap.CellGridExchange`:

* to the routing: SMOIS, SH2O, INFXSRT and SOLDRAIN as overlap-weighted means of the cells under each LSM
  grid cell (only the covered part of a cell's runoff enters the routing; the rest leaves as it does without
  the coupling);
* back: SMOIS and SH2O as the cells' share of the routing's change (a cell keeps its own soil where the
  routing changed nothing), SFCHEADRT as the cells' share of the routing's ponded head; every one conserves
  the grid integral exactly;
* the routing's IVGTYP/ISLTYP are the dominant hex classes under each grid cell, with WRF-Hydro's water rule
  (lsm_input: a water or lake land use takes water soil, water soil takes the water land use) applied to that
  choice so the routing reads one consistent class pair.

The exchange runs on the device by default (woof.hex.hydro_remap_gpu): the weights and every plane stay there,
and the words are the host route's on the host it runs beside.  ``--no-wrf-hydro-gpu-exchange`` (``[wrf_hydro]
gpu_exchange = false``) keeps the host route (scipy, float64 weights); without a GPU backend, or on a host whose
scipy mixes fused and plain loops, the host route runs and the log says why.

Nothing here runs without a hydro namelist: :func:`attach` returns None and the column driver keeps its
default LSM calls, so a run without the coupling is byte-identical.  The routing stays the opt-in
(``WOOF_ENABLE_WRF_HYDRO=1``), reached through :func:`woof.optin.load`; this module carries no UCAR code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from woof.wrf_hydro_route import LSM_PLANES

#: the water budget of one coupling cycle (HexWrfHydroSeam._budget_row), m3
BUDGET_KEYS = ("soil_change_cells_m3", "soil_change_cells_exact_m3", "soil_cast_envelope_m3",
               "soil_water_cells_m3", "soil_change_grid_m3", "head_cells_m3", "head_grid_m3", "runoff_cells_m3",
               "runoff_grid_m3", "runoff_grid_exact_m3", "runoff_cast_envelope_m3")


class HexWrfHydroSeam:
    """The hex column driver's ``wrf_hydro`` seam: cell planes, the routing on its grid, the exchange."""

    def __init__(self, routing, exchange, *, columns, lsm, dzs, device=None):
        import cupy as cp
        self.routing = routing
        self.remap = exchange
        #: the device route (woof.hex.hydro_remap_gpu.DeviceExchange), or None: the host route
        self.device = device
        self.host_reason = None
        #: the driver's columns; the first ``ncells`` are the mesh cells (a regional run carries native's garbage
        #: column after them, which the exchange leaves alone)
        self.columns, self.ncells = int(columns), len(exchange.cell_area)
        self.lsm = int(lsm)
        self.dzs = np.asarray(dzs, np.float64)
        self.planes = {name: cp.zeros((1, self.columns), cp.float32) for name in LSM_PLANES[self.lsm]}
        self.head = cp.zeros(exchange.shape, cp.float32)
        self._budget, self._pending = [], []
        self._work = None

    #: the device route fetches its budget rows to the host every this many cycles (one copy, not one a cycle)
    PENDING = 256

    @property
    def budget(self) -> list:
        """One dict per coupling cycle (:meth:`_budget_row`); the device route's rows are fetched here, at once."""
        if self._pending:
            import cupy as cp
            rows = cp.asnumpy(cp.stack(self._pending))
            self._pending.clear()
            self._budget.extend(dict(zip(BUDGET_KEYS, map(float, row))) for row in rows)
        return self._budget

    def lsm_kwargs(self) -> dict:
        return {"wrf_hydro": self.planes}

    def _cells(self, value):
        import cupy as cp
        return np.asarray(cp.asnumpy(value), np.float64).reshape(-1, self.columns)[:, :self.ncells]

    def exchange(self, fields) -> None:
        """wrf_drv_HYDRO after the surface driver, across the cell <-> grid exchange."""
        if self.device is not None:
            return self._exchange_device(fields)
        import cupy as cp
        f32 = np.float32
        ex, p = self.remap, self.planes
        nsoil = fields["smois"].shape[0]
        if tuple(fields["smois"].shape[1:]) != (1, self.columns):
            raise ValueError(f"smois has shape {tuple(fields['smois'].shape)}, the seam {self.columns} columns")
        soil = {name: self._cells(fields[name]) for name in ("smois", "sh2o")}
        flux = {name: self._cells(p[name])[0] for name in ("infxsrt", "soldrain")}
        grid = {name: ex.to_grid(value).astype(f32) for name, value in (*soil.items(), *flux.items())}
        before = {name: grid[name].copy() for name in soil}
        device = {name: cp.asarray(value) for name, value in grid.items()}
        self.routing.exchange(smois=device["smois"], sh2o=device["sh2o"], infxsrt=device["infxsrt"],
                              soldrain=device["soldrain"], sfcheadrt=self.head)
        after = {name: cp.asnumpy(device[name]) for name in soil}
        head = cp.asnumpy(self.head)
        out, exact = {}, {}
        for name in soil:
            exact[name] = soil[name] + ex.increment(before[name], after[name])
            out[name] = exact[name].astype(f32)
            fields[name][:, 0, :self.ncells] = cp.asarray(out[name].reshape(nsoil, self.ncells))
        p["sfcheadrt"][0, :self.ncells] = cp.asarray(ex.to_cells(head).astype(f32))
        self._budget.append(self._budget_row(soil, out, exact, flux, grid, before, after, head))

    def _budget_row(self, soil, out, exact, flux, grid, before, after, head):
        """The exchange's water on both sides (m3): soil change, ponded head returned and runoff handed over,
        each as the cells see it and as the routing's grid sees it.  The cells' soil change is given before
        (``_exact``) and after the float32 state cast, with the cast's own envelope
        (``sum_c A_c dz |q32 - q64|``)."""
        ex, dz = self.remap, self.dzs[:, None]
        cells = lambda q: float(np.sum(ex.integral_cells(q)))  # noqa: E731
        on_grid = lambda q: float(np.sum(ex.integral_grid(q)))  # noqa: E731
        covered = np.asarray(ex.overlap.sum(axis=0)).ravel()
        return {
            "soil_change_cells_m3": cells((out["smois"].astype(np.float64) - soil["smois"]) * dz),
            "soil_change_cells_exact_m3": cells((exact["smois"] - soil["smois"]) * dz),
            "soil_cast_envelope_m3": cells(np.abs(out["smois"].astype(np.float64) - exact["smois"]) * dz),
            "soil_water_cells_m3": cells(soil["smois"] * dz),
            "soil_change_grid_m3": on_grid((after["smois"].astype(np.float64) - before["smois"]) * dz[..., None]),
            "head_cells_m3": cells(ex.to_cells(head)[None] * 1e-3),
            "head_grid_m3": on_grid(head[None].astype(np.float64) * 1e-3),
            "runoff_cells_m3": float((flux["infxsrt"] + flux["soldrain"]) @ covered) * 1e-3,
            "runoff_grid_m3": on_grid((grid["infxsrt"].astype(np.float64) + grid["soldrain"])[None] * 1e-3),
            "runoff_grid_exact_m3": on_grid((ex.to_grid(flux["infxsrt"]) + ex.to_grid(flux["soldrain"]))[None] * 1e-3),
            "runoff_cast_envelope_m3": on_grid(sum(np.abs(grid[n] - ex.to_grid(flux[n]))
                                                   for n in ("infxsrt", "soldrain"))[None] * 1e-3),
        }

    def _workspace(self, nsoil):
        """The device route's grid planes and binary64 scratch, allocated at the first cycle."""
        import cupy as cp
        if self._work is None or self._work["smois"].shape[0] != nsoil:
            ny, nx = self.remap.shape
            grid = lambda n=None, t=cp.float32: cp.zeros((ny, nx) if n is None else (n, ny, nx), t)  # noqa: E731
            cells = lambda n, t: cp.zeros((n, self.ncells), t)  # noqa: E731
            w = {}
            for name in ("smois", "sh2o"):
                w[name], w[name + "_before"] = grid(nsoil), grid(nsoil)
                w[name + "_grid64"] = grid(nsoil, cp.float64)
                w[name + "_prior"], w[name + "_exact"] = cells(nsoil, cp.float32), cells(nsoil, cp.float64)
            for name in ("infxsrt", "soldrain"):
                w[name], w[name + "_grid64"] = grid(), grid(None, cp.float64)
            w["head64"] = cp.zeros(self.ncells, cp.float64)
            self._work = w
        return self._work

    def _exchange_device(self, fields) -> None:
        """The exchange with the weights and every plane on the device (woof.hex.hydro_remap_gpu): the host
        route's operations in its order, no copy to the host; the budget stays on the device until read."""
        import cupy as cp
        dev, p = self.device, self.planes
        nsoil = fields["smois"].shape[0]
        for name in ("smois", "sh2o"):
            field = fields[name]
            if tuple(field.shape[1:]) != (1, self.columns):
                raise ValueError(f"{name} has shape {tuple(field.shape)}, the seam {self.columns} columns")
            if field.dtype != cp.float32 or not field.flags.c_contiguous:
                raise ValueError(f"the device exchange updates {name} in place: it must be contiguous float32")
        w = self._workspace(nsoil)
        for name in ("smois", "sh2o"):
            dev.to_grid(fields[name], fields=nsoil, stride=self.columns, exact=w[name + "_grid64"], out=w[name])
            w[name + "_before"][...] = w[name]
        for name in ("infxsrt", "soldrain"):
            dev.to_grid(p[name], fields=1, stride=self.columns, exact=w[name + "_grid64"], out=w[name])
        self.routing.exchange(smois=w["smois"], sh2o=w["sh2o"], infxsrt=w["infxsrt"], soldrain=w["soldrain"],
                              sfcheadrt=self.head)
        for name in ("smois", "sh2o"):
            dev.increment(w[name + "_before"], w[name], fields[name], fields=nsoil, stride=self.columns,
                          prior=w[name + "_prior"], exact=w[name + "_exact"])
        dev.to_cells(self.head, exact=w["head64"], out=p["sfcheadrt"])
        self._pending.append(self._budget_device(fields, w))
        if len(self._pending) >= self.PENDING:
            self.budget  # noqa: B018 -- folds the pending rows into the host list

    def _budget_device(self, fields, w):
        """:meth:`_budget_row`'s terms as one device vector (the sums run in the device's order)."""
        import cupy as cp
        if "dz" not in w:
            w["dz"] = cp.asarray(np.ascontiguousarray(self.dzs, np.float64))
        p = self.planes
        cells, grid = self.device.budget(
            w["dz"], prior=w["smois_prior"], state=fields["smois"], stride=self.columns, exact=w["smois_exact"],
            head_cells=w["head64"], infx=p["infxsrt"], sold=p["soldrain"], before=w["smois_before"],
            after=w["smois"], head=self.head, infx_grid=w["infxsrt"], sold_grid=w["soldrain"],
            infx_exact=w["infxsrt_grid64"], sold_exact=w["soldrain_grid64"])
        # BUDGET_KEYS: cells 0-3, grid 0, cells 4, grid 1, cells 5, grid 2-4
        return cp.concatenate([cells[:4], grid[:1], cells[4:5], grid[1:2], cells[5:], grid[2:]])

    def history_fields(self) -> dict:
        p = self.planes
        return {"SFCHEADRT": p["sfcheadrt"], "INFXSRT": p["infxsrt"], "SOLDRAIN": p["soldrain"]}

    def digest_arrays(self) -> dict:
        out = {f"wrf_hydro/lsm/{k}": v for k, v in self.planes.items()}
        out.update(self.routing.digest_arrays())
        return out

    def receipt(self) -> dict:
        out = {"exchange": self.remap.receipt, "cycles": len(self.budget),
               "budget": self.budget[-1] if self.budget else None}
        out["exchange_route"] = ({"device": True, "host_scipy_fused": list(map(bool, self.device.fuse))}
                                 if self.device is not None else {"device": False, "reason": self.host_reason})
        return out


def _water_rule(veg, sol, geo_static):
    """WRF-Hydro lsm_input's water rule on the dominant classes (module_lsm_forcing.F90 sys_cpl = 2)."""
    from netCDF4 import Dataset
    with Dataset(geo_static) as ds:
        att = lambda name, default: int(ds.getncattr(name)) if name in ds.ncattrs() else default  # noqa: E731
        iswater, islake, isoilwater = att("ISWATER", 16), att("ISLAKE", -1), att("ISOILWATER", 14)
    sol = np.where((veg == iswater) | (veg == islake), isoilwater, sol).astype(np.int32)
    veg = np.where(sol == isoilwater, iswater, veg).astype(np.int32)
    return veg, sol


#: the backends the device exchange is built and checked on (woof.backend): CUDA through the no-FTZ compile,
#: Metal; hip and hip-nvidia keep the host route until they are (hydro_remap_gpu compiles no-FTZ on cuda only)
DEVICE_BACKENDS = ("cuda", "mlx")
#: the receipt's reason when the launch switched the device exchange off
SWITCHED_OFF = ("switched off (--no-wrf-hydro-gpu-exchange, [wrf_hydro] gpu_exchange = false or "
                "WOOF_WRF_HYDRO_GPU_EXCHANGE=0)")


def _device_exchange(exchange, asked, fields=None):
    """The device route for ``exchange`` and why not, as ``(DeviceExchange or None, reason or None)``: any
    failure to build it (no GPU backend, a host scipy that mixes loops, an overlap past int32, a kernel that
    does not compile, soil fields it cannot update in place) leaves the host route, with the reason."""
    if not asked:
        return None, SWITCHED_OFF
    from woof import backend
    name = backend.active()
    if name not in DEVICE_BACKENDS:
        return None, f"no device exchange on woof backend {name!r}"
    for key in ("smois", "sh2o"):
        field = (fields or {}).get(key)
        if field is not None and (str(field.dtype) != "float32" or not field.flags.c_contiguous):
            return None, f"{key} is not contiguous float32, which the device exchange updates in place"
    from .hydro_remap_gpu import DeviceExchange
    try:
        return DeviceExchange(exchange), None
    except Exception as exc:  # noqa: BLE001 -- any failure here leaves the host route, which the log names
        return None, f"{type(exc).__name__}: {exc}"


def attach(driver, cells, *, namelist, start_time, dzs, lsm, output_dir=None, gpu_exchange=True):
    """Couple the routing named by ``namelist`` to the hex column ``driver`` (cells ``cells``, a
    :class:`woof.hex.moving_remap.SphericalCells`); None, touching nothing, without a namelist.
    The exchange runs on the device (woof.hex.hydro_remap_gpu) unless ``gpu_exchange`` is false or no device
    route is available, which the log says."""
    if namelist is None:
        return None
    if int(lsm) not in LSM_PLANES:
        raise NotImplementedError(f"--wrf-hydro with sf_surface_physics = {lsm}: only Noah (2) and Noah-MP (4) "
                                  "carry the ported WRF_HYDRO hooks")
    if getattr(driver, "wrf_hydro", None) is not None:
        raise ValueError("the hex column driver already carries a WRF-Hydro coupling")
    from importlib import import_module
    from woof import optin
    from woof.wrf_hydro_route import soil_centres
    from .hydro_remap import CellGridExchange, geo_em_grid

    package = optin.load("wrf_hydro").__name__
    parse = import_module(package + ".namelist").parse
    path = Path(namelist).resolve()
    groups = parse(path.read_text(encoding="utf-8", errors="replace"))
    value = str(groups.get("hydro_nlist", {}).get("geo_static_flnm", "") or "")
    if not value:
        raise ValueError("GEO_STATIC_FLNM is empty: the hex exchange needs the routing domain's LSM grid")
    geo_static = Path(value) if Path(value).is_absolute() else path.parent / value
    grid = geo_em_grid(geo_static, radius_m=cells.radius_m)
    from netCDF4 import Dataset
    with Dataset(geo_static) as ds:
        shape = (len(ds.dimensions["south_north"]), len(ds.dimensions["west_east"]))
    exchange = CellGridExchange.build(cells, grid, shape)
    fields = driver.fields
    columns, ncells = int(fields["smois"].shape[-1]), len(cells.area)
    if columns not in (ncells, ncells + 1):
        raise ValueError(f"the column driver carries {columns} columns and the mesh {ncells} cells")
    import cupy as cp
    device, reason = _device_exchange(exchange, gpu_exchange, fields)
    if device is None and gpu_exchange:
        import logging
        logging.getLogger(__name__).warning("WRF-Hydro on hex: the cell <-> grid exchange runs on the host: %s",
                                            reason)
    if device is not None:
        veg, sol = (cp.asnumpy(device.dominant(fields[name].reshape(-1)[:ncells])) for name in ("ivgtyp", "isltyp"))
    else:
        veg = exchange.dominant(cp.asnumpy(fields["ivgtyp"]).reshape(-1)[:ncells])
        sol = exchange.dominant(cp.asnumpy(fields["isltyp"]).reshape(-1)[:ncells])
    veg, sol = _water_rule(veg, sol, geo_static)
    coupled = import_module(package + ".coupled")
    routing = coupled.WrfHydroCoupled(
        path, ix=shape[1], jx=shape[0], zs=soil_centres(dzs), vegtyp=veg, soltyp=sol,
        hydro_dt=float(driver.bldt_seconds), start=start_time,
        output_dir=output_dir or str(Path.cwd() / "wrf_hydro_output"))
    seam = HexWrfHydroSeam(routing, exchange, columns=columns, lsm=lsm, dzs=dzs, device=device)
    seam.host_reason = reason
    driver.wrf_hydro = seam
    return seam


def attach_host(backend, host, *, start_time, cycling=None):
    """The hex WOOF route's attach (woof.hex.drivers.run_cuda_v841_forecast): the routing the launch names
    (``--wrf-hydro``, woof.wrf_hydro_route.configured) on the column physics driver of ``backend``, or None."""
    import os
    from woof.wrf_hydro_route import ENV_OUTPUT, configured, gpu_exchange
    namelist = configured()
    if namelist is None:
        return None
    if cycling is not None and (cycling.get("restart_from") is not None or cycling.get("checkpoint_steps")):
        raise NotImplementedError(
            "a hex run coupled to WRF-Hydro neither writes nor resumes checkpoints: the routing state (soil and "
            "surface heads on the routing grid, channel, lakes and buckets) is not checkpointed")
    from woof.ingest.soil import NOAH_LAYER_THICKNESS_M
    from .hydro_remap import mesh_cells
    seam = backend._seam
    driver, cfg = seam._driver, seam._cfg
    lsm = int(getattr(cfg, "sf_surface_physics", 0))
    dzs = driver.noahmp_soil_thickness_m if lsm == 4 else NOAH_LAYER_THICKNESS_M
    return attach(driver, mesh_cells(host["prepared"].mesh), namelist=namelist, start_time=start_time, dzs=dzs,
                  lsm=lsm, output_dir=os.environ.get(ENV_OUTPUT), gpu_exchange=gpu_exchange())


def refuse_unwired(mode):
    """Refuse a launch mode that never reaches :func:`attach_host` when the launch asks for WRF-Hydro, rather
    than run uncoupled under the coupling's name."""
    from woof.wrf_hydro_route import configured
    if configured() is not None:
        raise NotImplementedError(f"--wrf-hydro with {mode}: that path does not attach the routing yet; run the "
                                  "coupled forecast on one card, as a single member, without active local time "
                                  "stepping")
