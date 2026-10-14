"""Process bodies and fixtures for tests/test_energy_hindcast_icechunk.py.

Kept in their own importable module because a spawned child process
imports its target by module name, and a test module's name depends on
how pytest collected it.
"""

from __future__ import annotations

from pathlib import Path
import traceback

import numpy as np

T0 = np.datetime64("2025-01-01T00:00:00", "s")
STEP_S = 900
NZ, NY, NX = 2, 3, 4


def frame_time(index: int) -> np.datetime64:
    return T0 + np.timedelta64(index * STEP_S, "s")


def write_wrfout(path: Path, indices, *, with_refl: bool = True) -> Path:
    """A tiny wrfout holding the frames at axis ``indices``: ``U`` and
    ``T2`` equal the frame's axis index, ``REFL_10CM`` (only when
    ``with_refl``) is ``-index``, ``HGT_STATIC`` has no Time dimension."""

    import netCDF4

    path = Path(path)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("Time", None)
        nc.createDimension("DateStrLen", 19)
        nc.createDimension("bottom_top", NZ)
        nc.createDimension("south_north", NY)
        nc.createDimension("west_east", NX)
        nc.DX = 100.0
        nc.SIMULATION_START_DATE = str(frame_time(indices[0])).replace("T", "_")
        times = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times.set_auto_chartostring(False)
        xtime = nc.createVariable("XTIME", "f4", ("Time",))
        xtime.units = "minutes since " + str(frame_time(indices[0])).replace("T", " ")
        u = nc.createVariable("U", "f4", ("Time", "bottom_top", "south_north",
                                          "west_east"))
        u.units = "m s-1"
        u.coordinates = "XLONG XLAT XTIME"
        t2 = nc.createVariable("T2", "f4", ("Time", "south_north", "west_east"))
        t2.units = "K"
        step = nc.createVariable("ITIMESTEP", "i4", ("Time",))
        hgt = nc.createVariable("HGT_STATIC", "f4", ("south_north", "west_east"))
        hgt[:] = np.arange(NY * NX, dtype=np.float32).reshape(NY, NX)
        refl = None
        if with_refl:
            refl = nc.createVariable("REFL_10CM", "f4", ("Time", "bottom_top",
                                                         "south_north",
                                                         "west_east"))
        for n, index in enumerate(indices):
            stamp = str(frame_time(index)).replace("T", "_")
            times[n] = np.frombuffer(stamp.encode("ascii"), dtype="S1")
            xtime[n] = 15.0 * (index - indices[0])
            u[n] = np.full((NZ, NY, NX), float(index), dtype=np.float32)
            t2[n] = np.full((NY, NX), float(index), dtype=np.float32)
            step[n] = index
            if refl is not None:
                refl[n] = np.full((NZ, NY, NX), -float(index), np.float32)
    return path


def write_segment_files(directory: Path, first: int, count: int,
                        *, per_file: int = 1) -> list[Path]:
    """wrfout files for frames ``first .. first+count-1``; the segment's
    first file lacks REFL_10CM, as WRF's t=0 frame does."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(first, first + count, per_file):
        indices = list(range(start, min(start + per_file, first + count)))
        stamp = str(frame_time(start)).replace("T", "_").replace(":", "-")
        paths.append(write_wrfout(directory / f"wrfout_d01_{stamp}", indices,
                                  with_refl=start != first))
    return paths


def gridded_writer(store: str, paths: list[str], group: str, axis_text: str,
                   job_id: str, gate, results) -> None:
    from woof.energy.icechunk_store import (RetryPolicy, TimeAxis,
                                            write_gridded_segment)

    gate.wait(timeout=120)
    try:
        summary = write_gridded_segment(
            store, paths, group=group, axis=TimeAxis.parse(axis_text),
            job_id=job_id, segment_id=job_id,
            policy=RetryPolicy(max_attempts=40, base_delay_s=0.05,
                               max_delay_s=0.5))
    except BaseException:
        results.put({"error": traceback.format_exc()})
        raise
    results.put(summary)


def _hold(barrier):
    def hook(attempt: int) -> None:
        if attempt == 1:
            barrier.wait(timeout=120)
    return hook


def site_writer(store: str, data_path: str, job_id: str, gate, barrier,
                results) -> None:
    """Write the ForecastData pickled at ``data_path``; both writers wait at
    ``barrier`` just before their first commit so they always collide."""

    import pickle

    from woof.energy.icechunk_store import RetryPolicy, write_site_region

    with open(data_path, "rb") as handle:
        data = pickle.load(handle)
    gate.wait(timeout=120)
    try:
        summary = write_site_region(
            store, data, job_id=job_id,
            policy=RetryPolicy(max_attempts=40, base_delay_s=0.05,
                               max_delay_s=0.5),
            before_commit=_hold(barrier))
    except BaseException:
        barrier.abort()
        results.put({"error": traceback.format_exc()})
        raise
    results.put(summary)
