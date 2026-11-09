# WOOF-OPTIN: wrf-hydro
"""lake_option 3: WRF-Hydro's persistence/level-pool hybrid and RFC-forecast
reservoirs (``reservoir_da.cu``) with their readers on the host.

WRF 4.7.1's read_rt_nlst turns lake_option = 3 into the level pool plus
``reservoir_type_specified`` when any of reservoir_persistence_usgs,
reservoir_persistence_usace or reservoir_rfc_forecasts is set
(config.F90 :762-769); LandRT_ini then gives each lake the type the
reservoir parameter file names (``read_reservoir_type``,
module_reservoir_utilities.F90 :169-251) and builds a hybrid for type 2/3
when its flag is on, an RFC reservoir for type 4/5 when that flag is on,
and a level pool otherwise (module_RT.F90 :1184-1263).

Host side, as WRF-Hydro reads them (Reservoirs/, WRF 4.7.1):

* the hybrid's properties (``hybrid_properties_init``): the gage id and
  persistence coefficients of the lake from ``usgs_*``/``usace_*`` in the
  reservoir parameter file; storage limits; its state init and the
  initial-storage clamps (``hybrid_init``);
* the timeslice reader (``module_reservoir_read_timeslice_data``): the k-th
  read is at start + k * update interval, floored to 15 minutes, looking back
  15 minutes at a time through ``observation_lookback_hours`` of
  ``<date>.15min.<usgs|usace>TimeSlice.ncdf`` files for the first gage value
  of quality above 0.9; :meth:`ReservoirDA.ensure_reads` tabulates them
  ahead of the device, which consumes them in order;
* the RFC properties and time-series reader
  (``module_reservoir_read_rfc_time_series_data``): the newest
  ``<YYYY-MM-DD_HH>.60min.<gage>.RFCTimeSeries.ncdf`` within the lookback
  from start + rfc_timeslice_offset hours; ``rfc_forecasts_init``'s index and
  update time; the first call's rejections (all synthetic, a negative value,
  a value >= 90000) made here.

Missing files and lakes are refused by name.  The device state is per lake
(``ri``/``rf``, layout in reservoir_da.cu).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

F = np.float32
RES_NI, RES_NF = 16, 16
(RI_KIND, RI_TYPE, RI_TIME, RI_UPDATE, RI_WUPDATE, RI_INDEX, RI_EVENT, RI_COUNT, RI_OFFSET,
 RI_INTERVAL, RI_FOUND, RI_PERSIST, RI_DYNAMIC, RI_WINTERVAL) = range(14)
(RF_AREA, RF_MINS, RF_MAX, RF_STORAGE, RF_ELEV, RF_PW, RF_LW, RF_PERSISTED, RF_GAGE, RF_ASSIM,
 RF_ORIFICE) = range(11)
SECONDS_IN_DAY = 86400
OBS_MINUTES, QUALITY_THRESHOLD = 15, F(0.9)


class MissingReservoirInput(FileNotFoundError):
    pass


def _open(path, what):
    import netCDF4

    p = Path(path)
    if not p.is_file():
        raise MissingReservoirInput(f"{what}: {p} does not exist")
    return netCDF4.Dataset(str(p), "r")


def _strings(var):
    """A char array read as Fortran does: NULs become blanks, ADJUSTL(TRIM())."""
    a = np.asarray(var[...])
    if a.dtype.kind == "S" and a.ndim == 2:
        rows = [b"".join(r) for r in a]
    else:
        rows = [bytes(r) for r in np.atleast_1d(a)]
    return [r.replace(b"\0", b" ").decode().strip() for r in rows]


def _stamp(d):
    return d.strftime("%Y-%m-%d_%H:%M:%S")


def read_reservoir_types(param_file, lake_ids):
    """read_reservoir_type: the parameter file's lake ids must be LAKEPARM's."""
    with _open(param_file, "reservoir_parameter_file") as ds:
        ids = np.asarray(ds["lake_id"][...]).astype(np.int64)
        if len(ids) != len(lake_ids) or (ids != np.asarray(lake_ids, np.int64)).any():
            raise ValueError("Reservoir ID array in the reservoir parameter file does not match "
                             "the Lake ID array in the lake parameter file.")
        return np.asarray(ds["reservoir_type"][...]).astype(np.int32)


class TimesliceReader:
    """module_reservoir_read_timeslice_data for one source (usgs or usace)."""

    def __init__(self, param_file, source, path, start, lookback_hours, update_seconds):
        with _open(param_file, "reservoir_parameter_file") as ds:
            self.gage_ids = _strings(ds[f"{source}_gage_id"])
        self.source, self.path, self.start = source, Path(path), start
        self.lookback_hours, self.update = int(lookback_hours), int(update_seconds)
        self.reads = []                      # (discharges, lookbacks) per read, gage order

    def read(self, k):
        while len(self.reads) <= k:
            self.reads.append(self._read(self.start + timedelta(
                seconds=self.update * len(self.reads))))
        return self.reads[k]

    def _read(self, current):
        import netCDF4

        n = len(self.gage_ids)
        q = np.full(n, F(-1.0), F)
        lb = np.zeros(n, np.int32)
        when = current.replace(minute=current.minute // OBS_MINUTES * OBS_MINUTES, second=0)
        lookback = 0
        for _ in range(self.lookback_hours * (60 // OBS_MINUTES)):
            if not (q == F(-1.0)).any():
                break
            f = self.path / f"{_stamp(when)}.{OBS_MINUTES}min.{self.source}TimeSlice.ncdf"
            if f.is_file():
                with netCDF4.Dataset(str(f), "r") as ds:
                    ids = _strings(ds["stationId"])
                    dq = np.asarray(ds["discharge"][...], F)
                    qual = (np.asarray(ds["discharge_quality"][...], F) / F(100)).astype(F)
                qual = np.where((qual < 0) | (qual > 1), F(0), qual)
                qual = np.where(dq < F(0.0), F(0), qual)
                qual = np.where(dq >= F(90000.0), F(0), qual).astype(F)
                for g, gid in enumerate(self.gage_ids):
                    if q[g] != F(-1.0):
                        continue
                    for s, sid in enumerate(ids):
                        if sid == gid:
                            if qual[s] > QUALITY_THRESHOLD:
                                q[g], lb[g] = dq[s], lookback
                            break
            when -= timedelta(minutes=OBS_MINUTES)
            lookback += OBS_MINUTES * 60
        return q, lb

    def lake(self, k, gage):
        """The read's value for a lake's gage (the last matching row, as WRF)."""
        q, lb = self.read(k)
        idx = [i for i, g in enumerate(self.gage_ids) if g == gage]
        if not idx:
            raise ValueError(f"gage {gage!r} is not among the reservoir parameter file's "
                             f"{self.source} gages")
        return q[idx[-1]], lb[idx[-1]]


def _lake_index(ds, var, lake_id, what):
    ids = np.asarray(ds[var][...]).astype(np.int64)
    hits = np.nonzero(ids == int(lake_id))[0]
    if not len(hits):
        raise ValueError(f"{what}: lake {lake_id} is not in {var}")
    return int(hits[-1])


def read_rfc_series(path, gage, start, offset_hours, lookback_hours):
    """setup_read_time_series: (found, discharges, synthetic, total, observed,
    step, lookback seconds, file name)."""
    import netCDF4

    when = start + timedelta(seconds=int(offset_hours) * 3600)
    lookback = 0
    for _ in range(int(lookback_hours)):
        name = f"{when.strftime('%Y-%m-%d_%H')}.60min.{gage}.RFCTimeSeries.ncdf"
        f = Path(path) / name
        if f.is_file():
            with netCDF4.Dataset(str(f), "r") as ds:
                total = int(ds["totalCounts"][...])
                observed = int(ds["observedCounts"][...])
                step = int(ds["timeSteps"][...])
                q = np.asarray(ds["discharges"][...], F)[:total]
                syn = np.asarray(ds["synthetic_values"][...])[:total] != 0
            return True, q, syn, total, observed, step, lookback, name
        when -= timedelta(hours=1)
        lookback += 3600
    return False, None, None, 0, 0, 0, lookback, ""


class ReservoirDA:
    """Device state of every lake for reservoir_da.cu (level-pool lakes too)."""

    def __init__(self, *, lake_ids, props, ifd, nl, start, base):
        """``props`` maps levelpool._PROPS to per-lake float32 arrays; ``nl``
        the namelist dict (hydro + reservoir groups); paths resolve at ``base``."""
        n = len(lake_ids)
        self.n = n
        path = lambda key, default="": (Path(base) / str(nl.get(key, default)))  # noqa: E731
        flags = {2: bool(nl.get("reservoir_persistence_usgs")),
                 3: bool(nl.get("reservoir_persistence_usace")),
                 4: bool(nl.get("reservoir_rfc_forecasts")),
                 5: bool(nl.get("reservoir_rfc_forecasts"))}
        param_file = path("reservoir_parameter_file")
        types = read_reservoir_types(param_file, lake_ids)
        ri = np.zeros((n, RES_NI), np.int32)
        rf = np.zeros((n, RES_NF), F)
        tab = []
        self.readers, self.lake_gage = {}, {}
        lookback = int(nl.get("reservoir_observation_lookback_hours", 24))
        update = int(nl.get("reservoir_observation_update_time_interval_seconds", 3600))
        rfc_lookback = int(nl.get("reservoir_rfc_forecasts_lookback_hours", 28))
        A, oe, maxh = props["lake_area"], props["orifice_elevation"], props["max_depth"]
        self.resht = (oe + ((maxh - oe) * ifd)).astype(F)
        with _open(param_file, "reservoir_parameter_file") as ds:
            for l in range(n):
                t = int(types[l])
                ri[l, RI_TYPE] = t
                ri[l, RI_DYNAMIC] = t if t in (2, 3, 4) else 1
                rf[l, RF_ASSIM] = F(-9999.0)
                if t in (2, 3) and flags[t]:
                    src = "usgs" if t == 2 else "usace"
                    i = _lake_index(ds, f"{src}_lake_id", lake_ids[l], "hybrid_properties_init")
                    gage = _strings(ds[f"{src}_gage_id"])[i]
                    coef = np.asarray(ds[f"{src}_persistence_coefficients"][...], F)[i]
                    if src not in self.readers:
                        self.readers[src] = TimesliceReader(
                            param_file, src, path(f"reservoir_{src}_timeslice_path"), start,
                            lookback, update)
                    self.lake_gage[l] = (src, gage)
                    ri[l, RI_KIND] = 1
                    ri[l, RI_COUNT], ri[l, RI_OFFSET] = len(coef), sum(len(x) for x in tab)
                    tab.append(coef)
                    ri[l, RI_INTERVAL], ri[l, RI_WINTERVAL] = update, SECONDS_IN_DAY
                    ri[l, RI_INDEX] = len(coef) + 1
                    rf[l, RF_AREA] = F(A[l] * F(1.0E6))
                    rf[l, RF_MINS] = F(0.0)
                    rf[l, RF_MAX] = F(F((maxh[l] - oe[l]) * A[l]) * F(1.0E6))
                    rf[l, RF_ORIFICE] = oe[l]
                    we = F(oe[l] + F((maxh[l] - oe[l]) * ifd[l]))
                    rf[l, RF_ELEV] = we
                    s = F(F(F(we - oe[l]) * A[l]) * F(1.0E6))
                    if s > rf[l, RF_MAX]:
                        s = rf[l, RF_MAX]
                    elif s < rf[l, RF_MINS]:
                        s = rf[l, RF_MINS]
                    rf[l, RF_STORAGE] = s
                elif t in (4, 5) and flags[t]:
                    i = _lake_index(ds, "rfc_lake_id", lake_ids[l], "rfc_forecasts_properties_init")
                    gage = _strings(ds["rfc_gage_id"])[i][:5]
                    persist = int(np.asarray(ds["rfc_forecast_persist"][...])[i])
                    offset = int(np.asarray(ds["rfc_timeslice_offset"][...])[i])
                    found, q, syn, total, observed, step, lb, _ = read_rfc_series(
                        path("reservoir_rfc_forecasts_time_series_path"), gage, start, offset,
                        rfc_lookback)
                    ri[l, RI_KIND] = 2
                    ri[l, RI_PERSIST] = persist * SECONDS_IN_DAY
                    rf[l, RF_AREA] = F(A[l] * F(1.0E6))
                    rf[l, RF_MAX] = maxh[l]
                    rf[l, RF_ELEV] = F(oe[l] + F((maxh[l] - oe[l]) * ifd[l]))
                    ri[l, RI_INDEX] = 1
                    if found:
                        ri[l, RI_UPDATE] = step - (lb % step)
                        idx = lb // step + 1 + observed - offset
                        if idx <= 0 or idx >= total:
                            found, idx = False, 1
                        ri[l, RI_INDEX] = idx
                        # run_rfc_forecasts_reservoir's checks at current_time == 0
                        if found and (syn.all() or q.min() < F(0.0)
                                      or q.max() >= F(90000.0)):
                            found = False
                        ri[l, RI_COUNT], ri[l, RI_OFFSET] = total, sum(len(x) for x in tab)
                        ri[l, RI_INTERVAL] = step
                        tab.append(q)
                    ri[l, RI_FOUND] = int(found)
        import cupy as cp

        self.ri = cp.asarray(ri.ravel())
        self.rf = cp.asarray(rf.ravel())
        flat = np.concatenate(tab).astype(F) if tab else np.zeros(1, F)
        self.rtab = cp.asarray(flat)
        self.err = cp.zeros(1, cp.int32)
        self.nev = 0
        self.obs_q = cp.zeros(1, cp.float32)
        self.obs_lb = cp.zeros(1, cp.int32)
        self.update = update
        self.time = 0.0                      # reservoir time the host has handed out
        self.any_da = bool((ri[:, RI_KIND] != 0).any())

    def ensure_reads(self, until_seconds):
        """Tabulate the timeslice reads the device can reach by ``until_seconds``
        of reservoir time (the k-th read happens no earlier than k * interval)."""
        import cupy as cp

        if not self.lake_gage:
            return
        need = int(until_seconds) // self.update + 2
        if need <= self.nev:
            return
        q = np.full((need, self.n), F(-1.0), F)
        lb = np.zeros((need, self.n), np.int32)
        for l, (src, gage) in self.lake_gage.items():
            for k in range(need):
                q[k, l], lb[k, l] = self.readers[src].lake(k, gage)
        self.obs_q, self.obs_lb, self.nev = cp.asarray(q.ravel()), cp.asarray(lb.ravel()), need

    def check(self):
        if int(self.err.get()[0]) & 64:
            raise RuntimeError("lake_option 3: the device ran past the tabulated timeslice reads")


_MODULE = None


def _k(name):
    global _MODULE
    if _MODULE is None:
        from ._compile import raw_module

        kernels = Path(__import__("woof.core.kernels", fromlist=["x"]).__file__).parent
        here = Path(__file__).resolve().parent
        source = ((kernels / "glibc_flt32.cuh").read_text(encoding="utf-8") + "\n"
                  + (here / "levelpool.cu").read_text(encoding="utf-8") + "\n"
                  + (here / "reservoir_da.cu").read_text(encoding="utf-8"))
        _MODULE = raw_module(source, ("--fmad=false", "--ftz=false"))
    return _MODULE.get_function(name)


def run(da, lakes, *, previous_inflow, inflow, lateral_inflow, routing_period, elapsed):
    """One reservoir call on every lake (the standalone kernel): updates
    ``lakes.resht`` and ``lakes.qlakeo``; ``elapsed`` is the reservoir time
    before the call, for the timeslice table."""
    import cupy as cp

    da.ensure_reads(elapsed + float(routing_period))
    _k("wrfhydro_reservoir_da")((((da.n + 127) // 128),), (128,), (
        cp.asarray(previous_inflow, cp.float32), cp.asarray(inflow, cp.float32),
        cp.asarray(lateral_inflow, cp.float32), lakes.resht, lakes.qlakeo, lakes.props,
        lakes.lake_opt, da.ri, da.rf, da.rtab, da.obs_q, da.obs_lb, np.int32(da.nev),
        np.float32(routing_period), da.err, np.int32(da.n)))
    da.check()
