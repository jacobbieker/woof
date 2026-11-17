"""Time hex's WRF-Hydro cell <-> LSM grid exchange (woof.hex.wrf_hydro_hex) per LSM step, host and device.

Builds the synthetic WRF-Hydro domain of tests/wrf_hydro_synthetic.py at ``--nx`` x ``--ny`` LSM cells (1 km,
routing at 250 m) and a hex mesh around it (great-circle Voronoi cells in the gnomonic plane): uniform at
``--spacing-km``, or variable resolution (``--outer-km``: the spacing grows linearly from the core over the
grid to that at ``--grade-km`` beyond it).  Attaches the coupling with the host exchange and with the
device exchange (``gpu_exchange``), runs ``--cycles`` coupling cycles of each after a warm-up, and prints
JSON: ms per cycle (the whole ``exchange`` call; the routing alone; the exchange around it), the host round
trips per cycle (count and bytes, device->host and host->device, outside the routing), and the exchange's
share of the cycle (the exchange timed again with the routing a no-op, over exchange + routing).  Needs the WRF-Hydro opt-in (WOOF_ENABLE_WRF_HYDRO=1)::

    WOOF_ENABLE_WRF_HYDRO=1 python tools/bench_hex_hydro_exchange.py --nx 128 --ny 128 --spacing-km 3
    WOOF_ENABLE_WRF_HYDRO=1 python tools/bench_hex_hydro_exchange.py --nx 128 --ny 128 --spacing-km 1 \\
        --outer-km 9 --grade-km 150
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAT0, DLAT, LON0, DLON = 40.0, 0.008, -100.0, 0.012
DZS = [0.1, 0.3, 0.6, 1.0]
EARTH_KM = 6371.229


def _domain(path, nx, ny):
    import netCDF4
    sys.path.insert(0, str(ROOT / "tests"))
    import wrf_hydro_synthetic as synthetic
    synthetic.write(path, nx=nx, ny=ny)
    j, i = np.meshgrid(np.arange(ny + 1) - 0.5, np.arange(nx + 1) - 0.5, indexing="ij")
    with netCDF4.Dataset(path / "geo_em.d01.nc", "a") as ds:
        ds.createDimension("south_north_stag", ny + 1)
        ds.createDimension("west_east_stag", nx + 1)
        for name, value in (("XLAT_C", LAT0 + DLAT * j), ("XLONG_C", LON0 + DLON * i)):
            ds.createVariable(name, "f4", ("Time", "south_north_stag", "west_east_stag"))[0] = value
    return path / "hydro.namelist"


def _mesh(nx, ny, spacing_km, outer_km, grade_km, margin_km, seed=9):
    """Gnomonic-plane Voronoi cells (great-circle edges) around the grid; points thinned to the spacing."""
    from scipy.spatial import Voronoi
    lat_c, lon_c = np.radians(LAT0 + DLAT * ny / 2), np.radians(LON0 + DLON * nx / 2)
    centre = np.array([np.cos(lat_c) * np.cos(lon_c), np.cos(lat_c) * np.sin(lon_c), np.sin(lat_c)])
    east = np.array([-np.sin(lon_c), np.cos(lon_c), 0.0])
    north = np.cross(centre, east)
    core = np.radians(np.array([DLON * nx * np.cos(lat_c), DLAT * ny])) * 0.5 + 3 * spacing_km / EARTH_KM
    outer = outer_km or spacing_km
    half = core + (grade_km + margin_km) / EARTH_KM if outer_km else core + margin_km / EARTH_KM
    rng = np.random.default_rng(seed)
    count = int(np.prod(2 * half) / (spacing_km / EARTH_KM) ** 2)
    points = rng.uniform(-1.0, 1.0, (count, 2)) * half
    if outer_km:
        beyond = np.max(np.maximum(np.abs(points) - core, 0.0), axis=1) * EARTH_KM
        local = spacing_km + (outer - spacing_km) * np.clip(beyond / grade_km, 0.0, 1.0)
        points = points[rng.uniform(size=len(points)) < (spacing_km / local) ** 2]
    vor = Voronoi(points)
    to_sphere = lambda xy: centre + xy[:, :1] * east + xy[:, 1:] * north  # noqa: E731
    keep, regions = [], []
    edge = half - 1.5 * outer / EARTH_KM
    for k, r in enumerate(vor.point_region):
        region = vor.regions[r]
        if -1 in region or not region or np.any(np.abs(points[k]) > edge):
            continue
        v = vor.vertices[region] - points[k]
        regions.append(np.asarray(region)[np.argsort(np.arctan2(v[:, 1], v[:, 0]))])
        keep.append(k)
    cells, vertex = to_sphere(points[keep]), to_sphere(vor.vertices)
    latlon = lambda xyz: (np.arcsin(xyz[:, 2] / np.linalg.norm(xyz, axis=1)),  # noqa: E731
                          np.arctan2(xyz[:, 1], xyz[:, 0]))
    counts = np.array([len(r) for r in regions])
    on_cell = np.full((len(regions), counts.max()), -1, np.int64)
    for i, r in enumerate(regions):
        on_cell[i, :len(r)] = r
    (lat, lon), (latv, lonv) = latlon(cells), latlon(vertex)
    return SimpleNamespace(n_cells=len(regions), attrs={}, arrays=dict(
        latCell=lat, lonCell=lon, latVertex=latv, lonVertex=lonv, nEdgesOnCell=counts, verticesOnCell=on_cell))


class _Transfers:
    """Counts the host round trips the seam makes (cupy.asnumpy; cupy.asarray of a host array)."""

    def __init__(self, cp):
        self.cp, self.on, self.d2h, self.h2d = cp, False, [], []
        self._asnumpy, self._asarray = cp.asnumpy, cp.asarray

        def asnumpy(a, *args, **kw):
            out = self._asnumpy(a, *args, **kw)
            if self.on:
                self.d2h.append(out.nbytes)
            return out

        def asarray(a, *args, **kw):
            if self.on and isinstance(a, np.ndarray):
                self.h2d.append(a.nbytes)
            return self._asarray(a, *args, **kw)
        cp.asnumpy, cp.asarray = asnumpy, asarray


def _run(seam, driver, cp, transfers, cycles, warm, rng_seed=3):
    sync = cp.cuda.Device().synchronize
    rng = np.random.default_rng(rng_seed)
    nc, f32 = seam.ncells, np.float32
    inputs = []
    for _ in range(warm + cycles):
        smois = rng.uniform(0.2, 0.4, (4, 1, seam.columns)).astype(f32)
        inputs.append((cp.asarray(smois), cp.asarray((smois * f32(0.9)).astype(f32)),
                       cp.asarray(rng.uniform(0.0, 6.0, (1, seam.columns)).astype(f32)),
                       cp.asarray(rng.uniform(0.0, 0.2, (1, seam.columns)).astype(f32))))
    routing_exchange, routing_ms = seam.routing.exchange, []

    def timed_routing(**kw):
        transfers.on = False
        sync()
        t = time.perf_counter()
        routing_exchange(**kw)
        sync()
        routing_ms.append(1e3 * (time.perf_counter() - t))
        transfers.on = True
    seam.routing.exchange = timed_routing
    total = []
    for n, (smois, sh2o, infx, sold) in enumerate(inputs):
        driver.fields["smois"][...] = smois
        driver.fields["sh2o"][...] = sh2o
        seam.planes["infxsrt"][...] = infx
        seam.planes["soldrain"][...] = sold
        if n == warm:
            transfers.d2h.clear(), transfers.h2d.clear()
            routing_ms.clear()
        sync()
        transfers.on = n >= warm
        t = time.perf_counter()
        seam.exchange(driver.fields)
        sync()
        transfers.on = False
        if n >= warm:
            total.append(1e3 * (time.perf_counter() - t))
    seam.budget                                         # the device route's budget rows, fetched once
    # The exchange alone: the routing a no-op (no syncs around it), the same planes and kernels.
    seam.routing.exchange = lambda **kw: None
    alone = []
    for n, (smois, sh2o, infx, sold) in enumerate(inputs):
        driver.fields["smois"][...] = smois
        driver.fields["sh2o"][...] = sh2o
        sync()
        t = time.perf_counter()
        seam.exchange(driver.fields)
        sync()
        if n >= warm:
            alone.append(1e3 * (time.perf_counter() - t))
    seam.budget
    del nc
    whole, route, ex = float(np.median(total)), float(np.median(routing_ms)), float(np.median(alone))
    return {"cycle_ms": round(whole, 3), "routing_ms": round(route, 3), "exchange_ms": round(ex, 3),
            "exchange_share": round(ex / (ex + route), 4),
            "d2h_per_cycle": len(transfers.d2h) / cycles, "d2h_bytes_per_cycle": sum(transfers.d2h) / cycles,
            "h2d_per_cycle": len(transfers.h2d) / cycles, "h2d_bytes_per_cycle": sum(transfers.h2d) / cycles,
            "cycle_ms_all": [round(x, 3) for x in total]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--spacing-km", type=float, default=3.0)
    p.add_argument("--outer-km", type=float, default=None)
    p.add_argument("--grade-km", type=float, default=150.0)
    p.add_argument("--margin-km", type=float, default=200.0)
    p.add_argument("--cycles", type=int, default=20)
    p.add_argument("--warm", type=int, default=3)
    p.add_argument("--work", default=None)
    args = p.parse_args(argv)
    import woof.backend
    woof.backend.install()
    import cupy as cp
    from woof.hex.hydro_remap import mesh_cells
    from woof.hex.wrf_hydro_hex import attach
    work = Path(args.work or tempfile.mkdtemp(prefix="hex-hydro-bench-"))
    namelist = _domain(work / "dom", args.nx, args.ny)
    t = time.perf_counter()
    mesh = _mesh(args.nx, args.ny, args.spacing_km, args.outer_km, args.grade_km, args.margin_km)
    cells = mesh_cells(mesh)
    mesh_s = time.perf_counter() - t
    nc = len(cells.area)
    transfers = _Transfers(cp)
    out = {"backend": woof.backend.active(),
           "lsm_grid": [args.ny, args.nx], "cells": nc, "spacing_km": args.spacing_km, "outer_km": args.outer_km,
           "mesh_s": round(mesh_s, 2)}
    for label, gpu in (("host", False), ("device", True)):
        fields = {"smois": cp.zeros((4, 1, nc), cp.float32), "sh2o": cp.zeros((4, 1, nc), cp.float32),
                  "ivgtyp": cp.full((1, nc), 10, cp.int32), "isltyp": cp.full((1, nc), 6, cp.int32)}
        driver = SimpleNamespace(fields=fields, bldt_seconds=60.0)
        t = time.perf_counter()
        seam = attach(driver, cells, namelist=namelist, start_time=datetime.datetime(2020, 1, 1), dzs=DZS, lsm=2,
                      output_dir=str(work / label), gpu_exchange=gpu)
        if (seam.device is not None) != gpu:
            raise SystemExit(f"the {label} run attached the other route: {seam.receipt()['exchange_route']}")
        out[label] = {"attach_s": round(time.perf_counter() - t, 2)}
        if label == "host":
            out["overlaps"] = int(seam.remap.overlap.nnz)
            out["covered_cells"] = seam.remap.receipt["covered_cells"]
        out[label].update(_run(seam, driver, cp, transfers, args.cycles, args.warm))
    out["speedup_exchange"] = round(out["host"]["exchange_ms"] / max(out["device"]["exchange_ms"], 1e-6), 1)
    out["speedup_cycle"] = round(out["host"]["cycle_ms"] / out["device"]["cycle_ms"], 2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
