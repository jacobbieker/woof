"""lake_option 3's reservoirs (the opt-in), bitwise against WRF 4.7.1.

The fixture is ``optional/wrf_hydro/oracle/build_reservoir_da.sh``: WRF
4.7.1's hydro/Routing/Reservoirs compiled whole -- the persistence /
level-pool hybrid with its USGS and USACE timeslice reader, the
RFC-forecast reservoir with its time-series reader, the level pool -- each
lake initialised from the synthetic reservoir parameter file, timeslices and
RFC series of ``gen_reservoir_da_inputs.py`` and run for 900 calls (300 s,
then 450 s routing periods; 3.6 days, past the observations and the RFC
persistence).  ``woof_wrf_hydro.reservoir_da`` reads the same files on the
host and runs the reservoirs on device.  Graded: every call's outflow and
water elevation of all five lakes.  Every test skips, naming the opt-in
rule, unless ``WOOF_ENABLE_WRF_HYDRO=1``.
"""

from __future__ import annotations

import csv
import hashlib
import sys

import numpy as np
import pytest

from conftest import requires_gpu
from woof import optin

REASON = optin.unavailable_reason("wrf_hydro")
pytestmark = pytest.mark.skipif(REASON is not None, reason=f"opt-in rule: {REASON}")

ORACLE = optin.optional_root() / "wrf_hydro" / "oracle"
FIX = ORACLE / "fixtures" / "reservoir_da"


def _gen():
    if str(ORACLE) not in sys.path:
        sys.path.insert(0, str(ORACLE))
    import gen_reservoir_da_inputs

    return gen_reservoir_da_inputs


def _sums():
    rows = (FIX / "reservoir-da-sha256sums.txt").read_text().splitlines()
    return {line.split()[1].rsplit("/", 1)[-1]: line.split()[0] for line in rows}


def test_fixtures_are_the_ones_the_build_hashed(tmp_path):
    sums = _sums()
    assert sums["reservoir-steps.csv"] == hashlib.sha256(
        (FIX / "reservoir-steps.csv").read_bytes()).hexdigest()
    for name in ("reservoir_da_harness.F90", "build_reservoir_da.sh", "gen_reservoir_da_inputs.py"):
        assert sums[name] == hashlib.sha256((ORACLE / name).read_bytes()).hexdigest(), name
    _gen().write(tmp_path)
    assert sums["reservoir-inputs.txt"] == hashlib.sha256(
        (tmp_path / "reservoir-inputs.txt").read_bytes()).hexdigest()


def _da(tmp_path, mod, g, **flags):
    from importlib import import_module

    rd = import_module(mod.__name__ + ".reservoir_da")
    lp = import_module(mod.__name__ + ".levelpool")
    L = g.lakes()
    props = {k: L[k] for k in lp._PROPS}
    nl = dict(reservoir_parameter_file="reservoir_parm.nc", reservoir_persistence_usgs=True,
              reservoir_persistence_usace=True, reservoir_rfc_forecasts=True,
              reservoir_usgs_timeslice_path="usgs", reservoir_usace_timeslice_path="usace",
              reservoir_rfc_forecasts_time_series_path="rfc",
              reservoir_observation_lookback_hours=g.LOOKBACK_HOURS,
              reservoir_observation_update_time_interval_seconds=g.UPDATE_SECONDS,
              reservoir_rfc_forecasts_lookback_hours=g.RFC_LOOKBACK_HOURS)
    nl.update(flags)
    da = rd.ReservoirDA(lake_ids=g.LAKE_IDS, props=props, ifd=L["ifd"], nl=nl, start=g.START,
                        base=tmp_path)
    return rd, da, lp.LevelPool(water_elevation=da.resht, lake_opt=1, **props)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("flags,kinds", [
    ({}, [1, 1, 2, 2, 0]),
    ({"reservoir_persistence_usace": False}, [1, 0, 2, 2, 0]),
    ({"reservoir_rfc_forecasts": False, "reservoir_persistence_usgs": False}, [0, 1, 0, 0, 0]),
])
def test_kinds_follow_the_types_and_the_flags(tmp_path, flags, kinds):
    """LandRT_ini :1184-1263: a DA type without its flag is a level pool."""
    g = _gen()
    g.write(tmp_path)
    _, da, _ = _da(tmp_path, optin.load("wrf_hydro"), g, **flags)
    assert da.ri.get().reshape(da.n, -1)[:, 0].tolist() == kinds


@pytest.mark.gpu
@requires_gpu
def test_missing_inputs_are_refused_by_name(tmp_path):
    g = _gen()
    g.write(tmp_path)
    mod = optin.load("wrf_hydro")
    (tmp_path / "reservoir_parm.nc").rename(tmp_path / "elsewhere.nc")
    from importlib import import_module

    rd = import_module(mod.__name__ + ".reservoir_da")
    with pytest.raises(rd.MissingReservoirInput, match="reservoir_parameter_file"):
        _da(tmp_path, mod, g)


def test_an_rfc_series_with_a_negative_value_falls_back_to_the_level_pool(tmp_path):
    """run_rfc_forecasts_reservoir's first call: negative, all-synthetic or
    >= 90000 discharges disable the forecast (made at init here)."""
    import netCDF4

    g = _gen()
    g.write(tmp_path)
    f = next((tmp_path / "rfc").glob("*RFCA1*"))
    with netCDF4.Dataset(f, "a") as ds:
        q = ds["discharges"][:]
        q[5] = -1.0
        ds["discharges"][:] = q
    if optin.unavailable_reason("wrf_hydro") is None:
        try:
            import cupy  # noqa: F401
        except ImportError:
            pytest.skip("needs the device arrays")
        _, da, _ = _da(tmp_path, optin.load("wrf_hydro"), g)
        ri = da.ri.get().reshape(da.n, -1)
        assert ri[2, 0] == 2 and ri[2, 10] == 0 and ri[3, 10] == 1   # RI_FOUND


@pytest.mark.gpu
@requires_gpu
def test_reservoir_calls_are_bitwise_against_wrf_hydro(tmp_path):
    g = _gen()
    g.write(tmp_path)
    mod = optin.load("wrf_hydro")
    rd, da, lakes = _da(tmp_path, mod, g)
    want = {}
    with (FIX / "reservoir-steps.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            want[(int(r["call"]), int(r["lake"]))] = (int(r["outflow"]), int(r["water_elevation"]))
    elapsed = 0.0
    for c, (period, prev, inflow, lateral) in enumerate(g.calls(), start=1):
        rd.run(da, lakes, previous_inflow=prev, inflow=inflow, lateral_inflow=lateral,
               routing_period=period, elapsed=elapsed)
        elapsed += float(period)
        q = lakes.qlakeo.get().view(np.int32)
        h = lakes.resht.get().view(np.int32)
        for lake in range(da.n):
            assert (int(q[lake]), int(h[lake])) == want[(c, lake + 1)], (c, lake + 1)
