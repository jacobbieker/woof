"""rrtmg_spectral_output in wrfout: each band field on its own named axis
(never bottom_top, whatever the band count), with the band edges."""

import numpy as np
import pytest


def _write(tmp_path, fields, ny=3, nx=4, nz=14, engine="python"):
    netCDF4 = pytest.importorskip("netCDF4")
    from woof.io.wrfout import WrfoutWriter
    path = tmp_path / f"wrfout_spectral_{engine}"
    try:
        writer = WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0, engine=engine)
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        pytest.skip(f"wrfout engine {engine} unavailable: {exc}")
    writer.write_frame("1974-04-03_12:00:00", fields)
    writer.close()
    return netCDF4.Dataset(path)


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_band_fields_land_on_their_spectral_axes(tmp_path, engine):
    ny, nx = 3, 4
    fields = {"SWDNB_BAND": np.full((14, ny, nx), 2.0, np.float32),
              "SWUPT_GPT": np.full((112, ny, nx), 1.0, np.float32),
              "LWUPT_BAND": np.full((16, ny, nx), 3.0, np.float32)}
    # nz = 14: a band count equal to nz must not be read as levels
    with _write(tmp_path, fields, ny, nx, nz=14, engine=engine) as ds:
        sw, gpt, lw = ds["SWDNB_BAND"], ds["SWUPT_GPT"], ds["LWUPT_BAND"]
        assert sw.dimensions == ("Time", "rrtmg_sw_band", "south_north", "west_east")
        assert gpt.dimensions == ("Time", "rrtmg_sw_gpt", "south_north", "west_east")
        assert lw.dimensions == ("Time", "rrtmg_lw_band", "south_north", "west_east")
        assert sw.units == "W m-2" and "PER RRTMG BAND" in sw.description
        from woof.core.rrtmg_sw import band_wavenumbers, gpoint_bands
        low, high = band_wavenumbers()
        assert np.array_equal(sw.wavenumber_low, np.float32(low))
        assert np.array_equal(sw.wavenumber_high, np.float32(high))
        assert list(gpt.band_of_gpoint) == list(gpoint_bands())
        assert float(lw.wavenumber_low[0]) == 10.0 and float(lw.wavenumber_high[-1]) == 3250.0
        assert np.all(np.asarray(sw[0]) == 2.0)


def test_lw_band_edges_are_the_ports_band_widths():
    from woof.core.rrtmg_legacy import _lw_coeffs
    from woof.io.wrf_output_schema import RRTMG_LW_WAVENUMBERS
    widths = np.float32([high - low for low, high in RRTMG_LW_WAVENUMBERS])
    assert np.array_equal(widths, np.asarray(_lw_coeffs()["wvn/delwave"], np.float32))


def test_frame_sink_routes_band_fields_by_name():
    from woof.io.frame_sink import frame_dims
    assert frame_dims("SWDNB_BAND", (14, 3, 4), nx=4, ny=3, nz=14) == ("rrtmg_sw_band", "south_north", "west_east")
    assert frame_dims("LWUPT_BAND", (16, 3, 4), nx=4, ny=3, nz=16) == ("rrtmg_lw_band", "south_north", "west_east")
    assert frame_dims("T", (14, 3, 4), nx=4, ny=3, nz=14) == ("bottom_top", "south_north", "west_east")


def test_config_admits_legacy_rrtmg_only():
    import dataclasses
    from woof.config import ConfigurationRefused, RunConfig, validate_run_config
    from woof.core.physics_inventory import radiation_spectral_fields
    base = RunConfig(nx=8, ny=8, nz=30, dx=3000.0, dy=3000.0, ztop=20000.0, dt=18.0, run_seconds=60.0)
    bands = validate_run_config(dataclasses.replace(
        base, ra_physics=4, ra_rrtmg_variant="rrtmg_legacy", rrtmg_spectral_output="bands"))
    fields = radiation_spectral_fields(bands)
    assert fields["swdnb_band"] == 14 and fields["lwupt_band"] == 16 and len(fields) == 16
    gpts = radiation_spectral_fields(dataclasses.replace(bands, rrtmg_spectral_output="gpoints"))
    assert gpts["swdnb_gpt"] == 112 and gpts["lwupt_band"] == 16
    for bad in (dict(ra_physics=4, ra_rrtmg_variant="rte-rrtmgp", rrtmg_spectral_output="bands"),
                dict(rrtmg_spectral_output="all")):
        with pytest.raises(ConfigurationRefused):
            validate_run_config(dataclasses.replace(base, **bad))
