"""WRF's per-step station time series (tslist) on the MPAS-A v8.4.2 suite route.

A WOOF-only output.  MPAS-A v8.4.2 has none; the hourly point products of a hex run are the column store
(``.wcol``, from the converted frames) and the history frames.  WRF's ``tslist`` samples named stations at EVERY
model step and writes eight text files per station (``<name>.d01.TS`` surface series, ``.UU .VV .WW .PH .TH .QV
.PR`` profiles of the lowest ``max_ts_level`` levels).  This module is that, on mesh columns.

The WOOF namelist group ``&diagnostics`` (the group config_solar_diagnostics, config_nwp_diagnostics and the
p_lev / z_lev options share) takes::

    config_tslist        character  the WRF station file (wrf_tsin.F fixed columns, ``tslist`` in a WRF run dir);
                                    geographic (lat, lon) stations only -- an i,j station has no meaning on a mesh
    config_max_ts_locs   integer    WRF max_ts_locs, stations read (default 5)
    config_max_ts_level  integer    WRF max_ts_level, levels of each profile file (default 15; <= nVertLevels)
    config_ts_buf_size   integer    WRF ts_buf_size, samples held before the writer is called (default 200)

With ``config_tslist`` unset nothing here is constructed, allocated or written.

Nothing is re-derived.  The sampler is the WRF route's own: ``woof/core/tslist_sampler.TslistSampler`` with its CUDA
kernel ``tslist_gather`` (woof/core/kernels/tslist.cu: wrf_timeseries.F's ``calc_ts`` arithmetic, ``calc_p8w``'s
interface pressures, the column water of five species) and the Rust writer ``woof.tslist_bridge.RustTslistWriter``
(the native text layouts, file by file).  A station is a one-point WRF grid: one mesh cell is one WRF column
(``ny = 1``, ``nx`` = the stations), so the kernel sees the carriers it was written for, gathered once a step from
the pools by the station's cell:

* **placement**: a station goes to the NEAREST CELL CENTRE (great circle), and a station whose nearest centre is
  farther than 1.5 times that cell's widest neighbour distance is refused by name (it lies outside the mesh).  The
  file's latitude, longitude and height are the cell's (as WRF's are the grid point's);
* the winds are MPAS's cell-centre reconstructed ``uReconstructZonal/Meridional`` (so WRF's staggered-face
  ``tslist_unstagger_winds`` has no counterpart here, ``cosa = 1``, ``sina = 0``), ``w`` is the state's,
  the pressure is ``pressure_base + pressure_p``, the potential temperature the DRY one (``theta_m / (1 + R_v/R_d
  qv)``, so the kernel's ``theta_m`` conversion is off), ``fzm``/``fzp`` are WRF's ``fnm``/``fnp``, and the
  geopotential is ``PH = 0``, ``PHB = 9.81 * zgrid`` (as nwp_diag_v842);
* the surface words are the 13 the kernel reads (t2m, q2, u10, v10, surface_pressure, glw, gsw, hfx, lh, skintemp,
  tslb level 1, rainc, rainnc); a field this route's physics does not carry reads as zero, the way the WRF route's
  absent species do.

The sample is taken after each completed step from the state that step left (WRF calls ``calc_ts`` at the end of
the step), never at t = 0, with ``ts_hour`` the elapsed hours as WRF forms them (minutes as a REAL, divided by REAL 60).

Not carried: a checkpoint continuation of a run with tslist output is refused by name (the Rust writer's resume
receipt is not in hex's checkpoint), and the WRF-Chem and solar file sets of the Rust writer are not requested.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

GROUP = "diagnostics"
OPTION = "config_tslist"
OPTIONS = ("config_tslist", "config_max_ts_locs", "config_max_ts_level", "config_ts_buf_size")
_F32 = np.float32


@dataclass(frozen=True)
class TslistConfig:
    """The resolved ``&diagnostics`` tslist options."""

    path: str
    max_locs: int = 5
    max_level: int = 15
    buf_size: int = 200


def resolve(groups: Mapping[str, Mapping[str, Any]]) -> TslistConfig | None:
    """The tslist options from the namelist's &diagnostics group (None when ``config_tslist`` is unset)."""
    supplied = {str(k).casefold(): v for k, v in groups.get(GROUP, {}).items()}
    path = supplied.get(OPTION)
    if path is None or (isinstance(path, str) and not path.strip()):
        stray = sorted(key for key in supplied if key in OPTIONS and key != OPTION)
        if stray:
            raise ValueError(f"&{GROUP} {stray[0]} is set without {OPTION}; it would be read by nothing")
        return None
    if not isinstance(path, str):
        raise ValueError(f"{OPTION} names the WRF station file (a quoted path), not {path!r}")
    values = {}
    for key, default, lower, upper in (("config_max_ts_locs", 5, 1, None), ("config_max_ts_level", 15, 1, 999),
                                       ("config_ts_buf_size", 200, 1, None)):
        value = supplied.get(key, default)
        if type(value) is not int or value < lower or (upper is not None and value > upper):
            raise ValueError(f"{key} is an integer from {lower}" + (f" through {upper}" if upper else "")
                             + f", not {value!r}")
        values[key] = value
    return TslistConfig(path.strip(), values["config_max_ts_locs"], values["config_max_ts_level"],
                        values["config_ts_buf_size"])


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------
def nearest_cells(lat_cell, lon_cell, neighbours, lat_deg, lon_deg):
    """The nearest cell centre of each (lat, lon) in degrees, and whether it lies inside the mesh.

    ``lat_cell``/``lon_cell`` are the cell centres in radians (nCells,), ``neighbours`` the one-based
    ``cellsOnCell`` (maxEdges, nCells) with 0 for no neighbour.  Returns ``(index, inside)`` per station; a
    station is inside when its great-circle distance to that centre is at most 1.5 times the centre's widest
    distance to a neighbouring centre.
    """
    lat_cell = np.asarray(lat_cell, dtype=np.float64)
    lon_cell = np.asarray(lon_cell, dtype=np.float64)
    unit = np.stack([np.cos(lat_cell) * np.cos(lon_cell), np.cos(lat_cell) * np.sin(lon_cell), np.sin(lat_cell)], axis=1)
    lat, lon = np.radians(np.asarray(lat_deg, dtype=np.float64)), np.radians(np.asarray(lon_deg, dtype=np.float64))
    point = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)
    index = np.argmax(point @ unit.T, axis=1)
    inside = np.zeros(len(index), dtype=bool)
    neighbours = np.asarray(neighbours)
    for station, cell in enumerate(index):
        around = neighbours[:, cell]
        around = around[(around > 0) & (around <= len(lat_cell))] - 1
        angle = lambda a, b: float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))
        reach = max((angle(unit[cell], unit[n]) for n in around), default=0.0)
        inside[station] = angle(point[station], unit[cell]) <= 1.5 * reach if reach else False
    return index.astype(np.int64), inside


def placement_plan(station_file: str, max_locs: int, lat_cell, lon_cell, height, neighbours):
    """The Rust writer's placement plan for ``station_file`` on this mesh, and the station cells.

    ``read_stations`` is the native fixed-column READ; ``place_stations`` rounds the (cell, 1) one-based
    coordinates this module hands it onto a one-row grid of ``nCells`` points; the file's latitude, longitude and
    height columns are then the cell's, as WRF's are the grid point's.
    """
    from woof.tslist_bridge import place_stations, read_stations

    geographic, source = read_stations(station_file, max_locs, ideal=False)
    rows = [row.split("\t") for row in source.splitlines()]
    if not geographic:
        raise ValueError(f"{station_file} lists (i, j) stations; a mesh has no i,j grid, so config_tslist needs "
                         "geographic (latitude, longitude) stations")
    first = np.array([int(row[7], 16) for row in rows], dtype=np.uint32).view(np.float32).astype(np.float64)
    second = np.array([int(row[8], 16) for row in rows], dtype=np.uint32).view(np.float32).astype(np.float64)
    names = [row[0] for row in rows]
    lon = (second + 180.0) % 360.0 - 180.0
    cell, inside = nearest_cells(lat_cell, lon_cell, neighbours, first, lon)
    outside = [f"{names[s]} ({first[s]:.4f}, {second[s]:.4f})" for s in np.flatnonzero(~inside)]
    if outside:
        raise ValueError(f"tslist station(s) {', '.join(outside)} lie outside this mesh (the nearest cell centre is "
                         "farther than 1.5 times its widest neighbour distance)")
    n = len(cell)
    placed = place_stations(source, (cell + 1).astype(np.float64), np.ones(n, dtype=np.float64), len(lat_cell), 1)
    out_rows = [row.split("\t") for row in placed.splitlines()]
    if len(out_rows) != n:
        raise ValueError("tslist placement lost a station on the mesh's one-row grid")
    lat_deg = np.degrees(np.asarray(lat_cell, dtype=np.float64)[cell])
    lon_deg = np.degrees(np.asarray(lon_cell, dtype=np.float64)[cell])
    bits = np.column_stack((lat_deg, lon_deg, np.asarray(height, dtype=np.float64)[cell])).astype(np.float32).view(np.uint32)
    for row, words in zip(out_rows, bits, strict=True):
        row[3:6] = [f"{int(word):08x}" for word in words]
    return "".join("\t".join(row) + "\n" for row in out_rows), cell


# ---------------------------------------------------------------------------
# the route's module
# ---------------------------------------------------------------------------
_SURFACE = (("t2m",), ("q2",), ("u10",), ("v10",), ("surface_pressure",), ("glw",), ("gsw",), ("hfx",), ("lh",),
            ("skintemp",), ("tslb",), ("rainc",), ("rainnc",))
_MOISTURE = ("qv", "qc", "qr", "qi", "qs", "qg")


class TslistModule:
    """Sample the stations after every step and hand the batches to WRF's own writer.

    ``owner`` is the route's ResidentOutputDiagnostics (its device pools, buffers and stream writer).  Its
    ``accumulates`` flag keeps the per-step update from being skipped on steps no stream is due.
    """

    accumulates = True

    def __init__(self, owner, config: TslistConfig, *, writer_factory=None):
        self.owner = owner
        self.config = config
        self._writer_factory = writer_factory
        self._due = False
        self.samples = 0

    # -- setup -------------------------------------------------------------------------------------------------
    def setup(self):
        import cupy as cp
        from woof.core.tslist_sampler import TslistSampler
        owner, config = self.owner, self.config
        layout = owner.session.layout
        dims = layout.dimensions
        self.cp = cp
        self.nc, self.nl = int(dims["nCellsSolve"]), int(dims["nVertLevels"])
        if config.max_level > self.nl:
            raise ValueError(f"config_max_ts_level {config.max_level} exceeds the mesh's {self.nl} mass levels")
        host = lambda pool, name: np.asarray(cp.asnumpy(owner.authority(pool, name)))
        nc = self.nc
        lat_cell, lon_cell = host("mesh", "latCell")[:nc], host("mesh", "lonCell")[:nc]
        neighbours = host("mesh", "cellsOnCell")[:, :nc]
        height = host("mesh", "zgrid")[0, :nc]
        plan, cells = placement_plan(config.path, config.max_locs, lat_cell, lon_cell, height, neighbours)
        self.plan, self.cells = plan, cells
        n = len(cells)
        self.n = n
        self.cols = cp.asarray(cells, dtype=cp.int64)
        writer = self._make_writer(plan)
        f32 = cp.float32
        z = lambda *shape: cp.zeros(shape, dtype=f32)
        nl = self.nl
        self.carriers = dict(
            u=z(nl, 1, n + 1), v=z(nl, 2, n), w=z(nl + 1, 1, n), phb=z(nl + 1, 1, n), ph=z(nl + 1, 1, n),
            theta=z(nl, 1, n), pressure=z(nl, 1, n), cosa=cp.ones((1, n), dtype=f32), sina=z(1, n),
            fnm=cp.ascontiguousarray(owner.authority("mesh", "fzm")[:nl], dtype=f32),
            fnp=cp.ascontiguousarray(owner.authority("mesh", "fzp")[:nl], dtype=f32))
        self.surface = [z(1, n) for _ in _SURFACE]
        self.moisture = [z(nl, 1, n) for _ in _MOISTURE]
        self.sampler = TslistSampler(
            ij=np.column_stack((np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.int32))),
            surface=self.surface, moisture=self.moisture, u=self.carriers["u"], v=self.carriers["v"],
            w=self.carriers["w"], phb=self.carriers["phb"], ph=self.carriers["ph"], theta=self.carriers["theta"],
            pressure=self.carriers["pressure"], cosa=self.carriers["cosa"], sina=self.carriers["sina"],
            fnm=self.carriers["fnm"], fnp=self.carriers["fnp"], levels=config.max_level,
            buffer_size=config.buf_size, unstagger=False, theta_m=False, writer=writer)

    def _make_writer(self, plan):
        start = self.owner.writer.start.strftime("%Y-%m-%d_%H:%M:%S")
        directory = str(self.owner.writer.output)
        if self._writer_factory is not None:
            return self._writer_factory(directory, plan, self.config.max_level, start)
        from woof.tslist_bridge import RustTslistWriter
        return RustTslistWriter(directory, plan, geographic=True, domain=1, levels=self.config.max_level, start=start)

    # -- the per-step sample -----------------------------------------------------------------------------------
    def begin(self, step, when):
        """Called with the step the next :meth:`update` folds and the valid time its state has."""
        self._due = int(step) > 0
        self._elapsed_minutes = (when - self.owner.writer.start).total_seconds() / 60.0

    def _field(self, name, pools=("diag_physics", "diag", "sfc_input")):
        layout = self.owner.session.layout.fields
        for pool in pools:
            if any(row["pool"] == pool and row.get("pool_name", row.get("name")) == name for row in layout):
                return self.owner.authority(pool, name)
        return None

    def _scalar(self, name):
        indices = self.owner.session.layout.indices
        key = ("state", "index_" + name)
        if key not in indices or int(indices[key]) < 1:
            return None
        return self.owner.authority("state", "scalars")[int(indices[key]) - 1]

    def update(self):
        """Gather this step's station columns into the sampler's carriers and take one sample."""
        if not self._due or not self.n:
            return
        cp, nl, n, cols = self.cp, self.nl, self.n, self.cols
        f32 = cp.float32
        own = self.owner.authority
        c = self.carriers

        def mass(pool, name):
            return cp.ascontiguousarray(own(pool, name)[:nl][:, cols], dtype=f32)

        east = mass("diag", "uReconstructZonal")
        c["u"][:, 0, :n] = east
        c["u"][:, 0, n] = east[:, n - 1]
        north = mass("diag", "uReconstructMeridional")
        c["v"][:, 0, :] = north
        c["v"][:, 1, :] = north
        c["w"][:, 0, :] = cp.ascontiguousarray(own("state", "w")[:nl + 1][:, cols], dtype=f32)
        c["phb"][:, 0, :] = cp.ascontiguousarray(own("mesh", "zgrid")[:nl + 1][:, cols], dtype=f32) * f32(9.81)
        c["pressure"][:, 0, :] = mass("diag", "pressure_base") + mass("diag", "pressure_p")
        qv = self._scalar("qv")
        qv_cols = cp.ascontiguousarray(qv[:nl][:, cols], dtype=f32)
        theta_m = cp.ascontiguousarray(own("state", "theta_m")[:nl][:, cols], dtype=f32)
        c["theta"][:, 0, :] = theta_m / (f32(1.0) + (f32(461.6) / f32(287.0)) * qv_cols) - f32(300.0)
        for slot, (name,) in zip(self.surface, _SURFACE):
            source = self._field(name)
            if source is None:
                slot[...] = f32(0.0)
            elif name == "tslb":
                slot[0, :] = cp.ascontiguousarray(source[0][cols], dtype=f32)
            else:
                slot[0, :] = cp.ascontiguousarray(source[cols], dtype=f32)
        for slot, name in zip(self.moisture, _MOISTURE):
            source = self._scalar(name)
            slot[:, 0, :] = f32(0.0) if source is None else cp.ascontiguousarray(source[:nl][:, cols], dtype=f32)
        # WRF forms ts_hour from elapsed minutes as a REAL, then divides by REAL 60.
        minutes = _F32(self._elapsed_minutes)
        self.sampler.begin_sample()
        self.sampler.gather(_F32(minutes / _F32(60.0)))
        self.sampler.end_sample()
        self.samples += 1

    def close(self):
        sampler = getattr(self, "sampler", None)
        if sampler is not None:
            sampler.close()
            self.sampler = None
