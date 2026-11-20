"""A mid-forecast first guess: the analysis starts at its valid time, as
WRFDA's ``wrfvar_output`` does (``da_med_initialdata_output``)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from woof.da.var.analysis import restamp_start  # noqa: E402


def _file(path: Path, start: str, valid: str) -> Path:
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as nc:
        nc.createDimension("Time", None)
        nc.createDimension("DateStrLen", 19)
        nc.createDimension("west_east", 3)
        nc.START_DATE = start
        nc.SIMULATION_START_DATE = start
        nc.GMT = np.float32(12.0)
        nc.JULYR = np.int32(2008)
        nc.JULDAY = np.int32(36)
        nc.TITLE = "OUTPUT FROM WRF V4.7.1 MODEL"
        t = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[0] = np.array(list(valid), dtype="S1")
        v = nc.createVariable("T", "f4", ("Time", "west_east"))
        v[0] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    return path


def test_mid_forecast_first_guess_is_restamped(tmp_path):
    f = _file(tmp_path / "a.nc", "2008-02-05_12:00:00", "2008-02-06_15:30:00")
    assert restamp_start(f, "2008-02-06_15:30:00")
    with netCDF4.Dataset(f) as nc:
        assert nc.START_DATE == "2008-02-06_15:30:00"
        assert nc.SIMULATION_START_DATE == "2008-02-06_15:30:00"
        assert nc.GMT == np.float32(15.5) and nc.getncattr("GMT").dtype == np.float32
        assert nc.JULYR == 2008 and nc.JULDAY == 37 and nc.getncattr("JULDAY").dtype == np.int32
        assert nc.TITLE == "OUTPUT FROM WRF V4.7.1 MODEL"
        assert nc.file_format == "NETCDF3_64BIT_OFFSET"
        np.testing.assert_array_equal(nc.variables["T"][0], [1.0, 2.0, 3.0])


def test_first_guess_at_its_start_is_untouched(tmp_path):
    f = _file(tmp_path / "a.nc", "2008-02-05_12:00:00", "2008-02-05_12:00:00")
    before = f.read_bytes()
    assert not restamp_start(f, "2008-02-05_12:00:00")
    assert f.read_bytes() == before


def test_tutorial_mid_forecast_analysis_passes_the_bundle_check(tmp_path):
    """fg04 of the FGAT lane is WRF's 15Z ``write_input`` of a run started at 12Z."""
    d = os.environ.get("WOOF_WRFDA471_FGAT_DIR")
    if not d:
        pytest.skip("WOOF_WRFDA471_FGAT_DIR is not set")
    from conftest import cpu_preprocess_gap, netcdf_bridge_gap, netcdf_writer_gap

    for gap in (netcdf_bridge_gap(), netcdf_writer_gap(), cpu_preprocess_gap("gpuwm_analysis_binary_f64")):
        if gap:
            pytest.skip(gap)
    from woof.da.var.analysis import WRF_FIELDS, read_first_guess, write_analysis_file
    from woof.io.analysis_bundle import check_analysis_file

    fg = read_first_guess(Path(d) / "fg04")
    assert fg.attrs["START_DATE"] == "2008-02-05_12:00:00" and fg.times == "2008-02-05_15:00:00"
    wrf = {k: fg.fields[k] for k in WRF_FIELDS}
    out = write_analysis_file(fg, wrf, tmp_path / "analysis.nc")
    check_analysis_file(out, valid_time="2008-02-05_15:00:00")
    with netCDF4.Dataset(out) as nc:
        assert nc.START_DATE == nc.SIMULATION_START_DATE == "2008-02-05_15:00:00"
        assert nc.GMT == np.float32(15.0) and nc.JULDAY == 36
