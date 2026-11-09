# WOOF-OPTIN: wrf-hydro
"""Write the lake_option 3 reservoir oracle's inputs (synthetic, no real data):
the reservoir parameter file, USGS and USACE 15-minute timeslice files, an
RFC time-series file, LAKEPARM-style lake properties and the inflow series.

    pixi run -e default python optional/wrf_hydro/oracle/gen_reservoir_da_inputs.py OUT_DIR

Five lakes: 1 a USGS persistence/level-pool hybrid (type 2), 2 a USACE hybrid
(type 3), 3 an RFC-forecast outflow reservoir (type 4), 4 an RFC
inflow-plus-forecast reservoir (type 5), 5 a level pool (type 1).  The
observations cover the first day with gaps, missing gages and low-quality
values; the RFC series run out (persist days) to fall back to the level pool.
tests/test_wrf_hydro_reservoir_da_parity.py regenerates these and checks
their hash against the build's.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

START = datetime(2020, 6, 1, 0, 0, 0)
NCALLS, LOOKBACK_HOURS, UPDATE_SECONDS, RFC_LOOKBACK_HOURS = 900, 2, 1800, 28
LAKE_IDS = (101, 202, 303, 404, 505)
TYPES = (2, 3, 4, 5, 1)
F = np.float32


def lakes():
    """LAKEPARM-style properties, one row per lake."""
    n = len(LAKE_IDS)
    return dict(
        lake_area=np.array([0.8, 2.5, 4.0, 1.2, 0.6], F),
        max_depth=np.array([112.0, 215.0, 318.0, 409.0, 506.0], F),
        weir_elevation=np.array([110.5, 213.0, 316.0, 407.5, 504.8], F),
        weir_coefficient=np.full(n, 0.4, F), weir_length=np.array([10, 30, 50, 15, 8], F),
        dam_length=np.array([10, 40, 60, 20, 10], F),
        orifice_coefficient=np.full(n, 0.1, F), orifice_area=np.array([1.0, 3.0, 5.0, 2, 1], F),
        orifice_elevation=np.array([104.0, 205.0, 305.0, 401.0, 500.0], F),
        ifd=np.array([0.7, 0.4, 0.5, 0.9, 0.6], F))


def calls():
    """Per call: routing period (s), previous inflow, inflow, lateral inflow per lake."""
    rows = []
    prev = np.zeros(len(LAKE_IDS), F)
    for k in range(NCALLS):
        period = F(300.0) if k < 600 else F(450.0)
        # a triangle wave of period 96 calls, phase-shifted per lake: binary32
        # arithmetic only (no libm), so every platform writes the same inputs
        phase = ((k + 19 * np.arange(5)) % 96).astype(F) / F(48.0)
        tri = np.where(phase <= F(1.0), phase, F(2.0) - phase).astype(F) - F(0.5)
        inflow = (np.array([3.0, 40.0, 80.0, 6.0, 2.0], F)
                  * (F(1.0) + F(1.2) * tri).astype(F))
        inflow = inflow.astype(F)
        if 300 <= k < 320:
            inflow[0] = F(-0.5)                     # a negative inflow: warned, not changed
        lateral = (F(0.05) * inflow).astype(F)
        rows.append((period, prev.copy(), inflow, lateral))
        prev = inflow
    return rows


def _chars(values, width, pad):
    return np.array([[c.encode() for c in v.ljust(width, pad)] for v in values], "S1")


def _stamp(d):
    return d.strftime("%Y-%m-%d_%H:%M:%S")


def write(out_dir):
    import netCDF4

    out = Path(out_dir)
    for sub in ("usgs", "usace", "rfc"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    n = len(LAKE_IDS)
    usgs_gages, usace_gages = ["01234567", "07654321"], ["ACE1", "ACE2", "ACE3"]
    with netCDF4.Dataset(out / "reservoir_parm.nc", "w") as ds:
        ds.createDimension("lake", n)
        ds.createDimension("usgs_lake", 2)
        ds.createDimension("usace_lake", 3)
        ds.createDimension("rfc_lake", 2)
        ds.createDimension("nchar15", 15)
        ds.createDimension("nchar5", 5)
        ds.createDimension("nweights", 3)
        ds.createVariable("lake_id", "i8", ("lake",))[:] = LAKE_IDS
        ds.createVariable("reservoir_type", "i4", ("lake",))[:] = TYPES

        def chars(name, dim, values, width):
            v = ds.createVariable(name, "S1", (dim, f"nchar{width}"))
            v[:] = _chars(values, width, "\0")

        ds.createVariable("usgs_lake_id", "i4", ("usgs_lake",))[:] = [999, 101]
        chars("usgs_gage_id", "usgs_lake", ["07654321", "01234567"], 15)
        ds.createVariable("usgs_persistence_coefficients", "f4", ("usgs_lake", "nweights"))[:] = \
            [[0.5, 0.25, 0.1], [0.9, 0.6, 0.3]]
        ds.createVariable("usace_lake_id", "i4", ("usace_lake",))[:] = [202, 888, 777]
        chars("usace_gage_id", "usace_lake", ["ACE2", "ACE1", "ACE3"], 15)
        ds.createVariable("usace_persistence_coefficients", "f4", ("usace_lake", "nweights"))[:] = \
            [[0.8, 0.4, 0.2], [0.7, 0.7, 0.7], [0.1, 0.1, 0.1]]
        ds.createVariable("rfc_lake_id", "i4", ("rfc_lake",))[:] = [303, 404]
        chars("rfc_gage_id", "rfc_lake", ["RFCA1", "RFCB2"], 5)
        ds.createVariable("rfc_forecast_persist", "i4", ("rfc_lake",))[:] = [2, 1]
        ds.createVariable("rfc_timeslice_offset", "i4", ("rfc_lake",))[:] = [6, 0]
    # 15-minute timeslices over the first 26 hours, some absent
    rng = np.random.default_rng(20200601)
    t = START - timedelta(hours=LOOKBACK_HOURS)
    k = 0
    while t <= START + timedelta(hours=26):
        k += 1
        if k % 5 == 3:                                   # a missing file
            t += timedelta(minutes=15)
            continue
        for source, gages in (("usgs", usgs_gages + ["99999999"]), ("usace", usace_gages)):
            m = len(gages)
            q = (rng.uniform(1.0, 60.0, m)).astype(F)
            qual = rng.choice([100.0, 95.0, 50.0, 100.0], m).astype(F)
            if k % 7 == 0:
                q[0] = F(-999.0)
            if k % 11 == 0:
                q[-1] = F(95000.0)
            name = f"{_stamp(t)}.15min.{source}TimeSlice.ncdf"
            with netCDF4.Dataset(out / source / name, "w") as ds:
                ds.createDimension("stationIdInd", m)
                ds.createDimension("stationIdStrLen", 15)
                v = ds.createVariable("stationId", "S1", ("stationIdInd", "stationIdStrLen"))
                v[:] = _chars(gages, 15, " ")
                ds.createVariable("discharge", "f4", ("stationIdInd",))[:] = q
                ds.createVariable("discharge_quality", "f4", ("stationIdInd",))[:] = qual
        t += timedelta(minutes=15)
    # RFC: lake 3's file six hours ahead of the start (its timeslice offset),
    # lake 4's three hours back (found by looking back)
    for gage, when, total, observed, bad in (("RFCA1", START + timedelta(hours=6), 72, 12, None),
                                              ("RFCB2", START - timedelta(hours=3), 48, 6, None)):
        q = (F(5.0) + F(0.3) * np.arange(total, dtype=F)
             + F(2.0) * (np.arange(total) % 5).astype(F)).astype(F)
        if bad is not None:
            q[bad] = F(-1.0)
        name = f"{when.strftime('%Y-%m-%d_%H')}.60min.{gage}.RFCTimeSeries.ncdf"
        with netCDF4.Dataset(out / "rfc" / name, "w") as ds:
            ds.createDimension("forecastInd", total)
            ds.createVariable("totalCounts", "i4")[...] = total
            ds.createVariable("observedCounts", "i4")[...] = observed
            ds.createVariable("timeSteps", "i4")[...] = 3600
            ds.createVariable("discharges", "f4", ("forecastInd",))[:] = q
            syn = np.zeros(total, np.int32)
            syn[-4:] = 1
            ds.createVariable("synthetic_values", "i4", ("forecastInd",))[:] = syn
    L = lakes()
    bits = lambda a: [str(int(b)) for b in np.asarray(a, F).view(np.int32).ravel()]  # noqa: E731
    lines = [f"{n} {NCALLS} {LOOKBACK_HOURS} {UPDATE_SECONDS} {RFC_LOOKBACK_HOURS} {_stamp(START)}"]
    for i in range(n):
        lines.append(f"{LAKE_IDS[i]} {TYPES[i]}")
        lines += bits([L[k][i] for k in ("lake_area", "weir_elevation", "weir_coefficient",
                                         "weir_length", "dam_length", "orifice_elevation",
                                         "orifice_coefficient", "orifice_area", "max_depth",
                                         "ifd")])
    for period, prev, inflow, lateral in calls():
        lines += bits([period]) + bits(prev) + bits(inflow) + bits(lateral)
    (out / "reservoir-inputs.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write(sys.argv[1])
