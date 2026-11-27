"""Observed snow analysis at the start (woof/ingest/snow_analysis.py)."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import io
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from woof.ingest import snow_analysis as sa

DATE = datetime(2023, 4, 27, 6, tzinfo=timezone.utc)
X0, Y0, RES = -110.0, 40.0, 0.01


def _header(code: str) -> str:
    return "\n".join([
        f"Description: synthetic {code}",
        "Data units: Meters / 1000.000000",
        "Data bytes per pixel: 2",
        "Data intercept: 0.0",
        "Data slope: 1.0",
        "No data value: -9999.0",
        "Number of columns: 40",
        "Number of rows: 30",
        f"Benchmark x-axis coordinate: {X0}",
        f"Benchmark y-axis coordinate: {Y0}",
        f"X-axis resolution: {RES}",
        f"Y-axis resolution: {RES}",
    ]) + "\n"


def _tar(path, swe_mm, depth_mm):
    row = sa.load_rules()["analyses"][0]
    with tarfile.open(path, "w") as archive:
        for key, values in (("snow_water", swe_mm), ("snow_depth", depth_mm)):
            stem = sa._fill(row["members"][key], DATE)
            for suffix, payload in ((".txt.gz", _header(key).encode()),
                                    (".dat.gz", values.astype(">i2").tobytes())):
                data = gzip.compress(payload)
                info = tarfile.TarInfo(stem + suffix)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))


@pytest.fixture
def cached(tmp_path, monkeypatch):
    monkeypatch.setenv("WOOF_SNOW_ANALYSIS_CACHE", str(tmp_path))
    monkeypatch.delenv("WOOF_SNOW_ANALYSIS", raising=False)
    swe = np.full((30, 40), 500, np.int16)
    swe[:, 20:] = 100
    swe[0:5, 0:5] = -9999
    depth = (swe.astype(np.int32) * 2).astype(np.int16)
    depth[swe < 0] = -9999
    (tmp_path / "snodas-conus").mkdir()
    _tar(tmp_path / "snodas-conus" / "SNODAS_20230427.tar", swe, depth)
    return tmp_path


def _context(lat, lon, hour=12):
    grid = SimpleNamespace(latlon_mass=lambda: (lat, lon))
    return SimpleNamespace(valid_time=datetime(2023, 4, 27, hour), latitude=lat, longitude=lon,
                           grid=grid)


def _grid(ny=4, nx=6, lat0=39.85, lon0=-109.95, step=0.03):
    lat = lat0 - step * np.arange(ny)[:, None] + 0 * np.arange(nx)[None, :]
    lon = lon0 + step * np.arange(nx)[None, :] + 0 * np.arange(ny)[:, None]
    return lat, lon


def test_analysis_time_is_the_newest_06z_at_or_before_the_start():
    row = sa.load_rules()["analyses"][0]
    assert sa.analysis_time(row, datetime(2023, 4, 27, 12)) == DATE
    assert sa.analysis_time(row, datetime(2023, 4, 27, 3)) == datetime(2023, 4, 26, 6, tzinfo=timezone.utc)
    assert sa.analysis_time(row, datetime(2003, 1, 1, 12)) is None


def test_archive_url_uses_english_month_folders():
    row = sa.load_rules()["analyses"][0]
    assert sa._fill(row["transport"]["url"], DATE).endswith("/2023/04_Apr/SNODAS_20230427.tar")


def test_covered_land_takes_the_footprint_mean_and_density(cached):
    lat, lon = _grid()
    snow = np.full(lat.shape, 7.0, np.float32)
    fields = {"SNOW": snow, "SNOWH": snow * np.float32(0.005)}
    out, receipt = sa.apply_snow_analysis(fields, context=_context(lat, lon),
                                          landmask=np.ones(lat.shape))
    assert receipt["analysis"] == "snodas-conus"
    assert receipt["analysis_time"] == "2023-04-27T06Z"
    # 0.03 deg cells over 0.01 deg pixels: a 3 x 3 (lat) footprint.
    assert np.allclose(out["SNOW"][:, 0], 500.0)
    assert np.allclose(out["SNOWH"][:, 0], 1.0)  # depth 2x SWE in mm: 500 kg m-3
    # The last column sits on the 500/100 edge (pixel 20): one 500 and two
    # 100 pixels across its footprint.
    assert np.allclose(out["SNOW"][:, -1], 700.0 / 3.0)


def test_water_and_uncovered_cells_keep_the_forcing_exactly(cached):
    lat, lon = _grid()
    lat[0, 0], lon[0, 0] = 39.98, -109.98      # inside the no-data corner
    lat[1, 1], lon[1, 1] = 45.0, -100.0        # off the analysis grid
    snow = np.linspace(1, 24, lat.size).reshape(lat.shape).astype(np.float32)
    land = np.ones(lat.shape)
    land[2, 2] = 0.0
    out, receipt = sa.apply_snow_analysis({"SNOW": snow, "SNOWH": snow * np.float32(0.004)},
                                          context=_context(lat, lon), landmask=land)
    for j, i in ((0, 0), (1, 1), (2, 2)):
        assert out["SNOW"][j, i] == np.float64(snow[j, i])
        assert out["SNOWH"][j, i] == np.float64(snow[j, i] * np.float32(0.004))
    assert receipt["land_cells_not_covered"] == 2


def test_era5_snow_ec_gets_the_forcing_depth_rule_where_not_covered(cached):
    lat, lon = _grid()
    lat[0, 0], lon[0, 0] = 45.0, -100.0
    ec = np.full(lat.shape, 0.2, np.float32)
    out, receipt = sa.apply_snow_analysis({"SNOW_EC": ec}, context=_context(lat, lon),
                                          landmask=np.ones(lat.shape))
    assert receipt["replaced"] == "SNOW_EC"
    expected = 1000.0 * np.float64(ec[0, 0])
    assert out["SNOW"][0, 0] == expected
    assert out["SNOWH"][0, 0] == expected / 1000.0 * 5.0


def test_opt_out_returns_the_input_mapping_untouched(cached, monkeypatch):
    monkeypatch.setenv("WOOF_SNOW_ANALYSIS", "off")
    lat, lon = _grid()
    fields = {"SNOW": np.ones(lat.shape)}
    out, receipt = sa.apply_snow_analysis(fields, context=_context(lat, lon))
    assert out is fields
    assert receipt["analysis"].startswith("off")


def test_domain_outside_coverage_is_not_fetched(tmp_path, monkeypatch):
    monkeypatch.setenv("WOOF_SNOW_ANALYSIS_CACHE", str(tmp_path))
    monkeypatch.delenv("WOOF_SNOW_ANALYSIS", raising=False)
    lat, lon = _grid(lat0=50.0, lon0=10.0)  # Europe
    fields = {"SNOW": np.ones(lat.shape)}
    out, receipt = sa.apply_snow_analysis(fields, context=_context(lat, lon))
    assert out is fields
    assert "outside its coverage" in receipt["analysis"]
    assert not any(tmp_path.iterdir())


def test_unpublished_newest_analysis_falls_back_to_the_previous_one(cached, monkeypatch):
    # NSIDC posts the 06Z analysis at ~14:00Z: a 12Z start's same-day file
    # is a 404, so the run takes the previous day's 06Z (30 h old), not the
    # forcing's snow.
    real_fetch = sa._fetch

    def fetch(row, date, root):
        if date.day == 28:
            raise OSError("HTTP Error 404: Not Found")
        return real_fetch(row, date, root)

    monkeypatch.setattr(sa, "_fetch", fetch)
    row = sa.load_rules()["analyses"][0]
    start = datetime(2023, 4, 28, 13)
    assert sa.analysis_times(row, start)[:2] == [
        datetime(2023, 4, 28, 6, tzinfo=timezone.utc), DATE]
    lat, lon = _grid()
    fields = {"SNOW": np.ones(lat.shape)}
    context = SimpleNamespace(valid_time=start, latitude=lat, longitude=lon,
                              grid=SimpleNamespace(latlon_mass=lambda: (lat, lon)))
    out, receipt = sa.apply_snow_analysis(fields, context=context)
    assert receipt["analysis_time"] == "2023-04-27T06Z"
    assert "404" in receipt["newer_analyses_unreadable"][0]
    assert out["SNOW"].max() == pytest.approx(500.0)


@pytest.fixture
def glacier(tmp_path, monkeypatch):
    """The 2026-11-06 Mount Rainier shape: a summit block of glacier pixels
    at the 2-byte ceiling (32 767 mm, depth likewise) and an apron at
    25 000 mm, in seasonal snow of 500 mm."""
    monkeypatch.setenv("WOOF_SNOW_ANALYSIS_CACHE", str(tmp_path))
    monkeypatch.delenv("WOOF_SNOW_ANALYSIS", raising=False)
    swe = np.full((30, 40), 500, np.int16)
    swe[12:18, 3:9] = 25000
    swe[14:17, 4:7] = 32767
    depth = np.full((30, 40), 1000, np.int16)
    depth[12:18, 3:9] = 30000
    depth[14:17, 4:7] = 32767
    (tmp_path / "snodas-conus").mkdir()
    _tar(tmp_path / "snodas-conus" / "SNODAS_20230427.tar", swe, depth)
    return tmp_path


def test_glacier_pixels_enter_on_the_snow_water_cap_density_kept(glacier):
    from woof.ingest.soil import (SNOW_WATER_CAP_KG_M2, _SNOW_WATER_CEILING_KG_M2,
                                  _admitted_snow_field)

    lat, lon = _grid()
    snow = np.full(lat.shape, 0.2, np.float32)
    out, receipt = sa.apply_snow_analysis({"SNOW": snow, "SNOWH": snow * np.float32(0.005)},
                                          context=_context(lat, lon),
                                          landmask=np.ones(lat.shape))
    # Cell (0, 0)'s footprint is all summit pixels: exactly the cap, and the
    # pixel density (32 767 / 32 767 -> 1000 kg m-3) is kept, not invented.
    assert out["SNOW"][0, 0] == SNOW_WATER_CAP_KG_M2
    assert out["SNOWH"][0, 0] == pytest.approx(SNOW_WATER_CAP_KG_M2 / 1000.0)
    assert out["SNOW"].max() <= SNOW_WATER_CAP_KG_M2
    # Seasonal snow off the glacier is untouched.
    assert out["SNOW"][:, -1] == pytest.approx(500.0)
    assert receipt["perennial_cap_cells"] > 0
    assert receipt["perennial_raw_max_kg_m2"] == pytest.approx(32767.0)
    # The preparation's plausibility ceiling now admits the field; before
    # the cap the summit cell carried 32 767 kg m-2 and was refused.
    admitted = _admitted_snow_field("snow water", out["SNOW"], lat.shape,
                                    _SNOW_WATER_CEILING_KG_M2, "kg m-2",
                                    land_ice=np.zeros(lat.shape, bool))
    assert admitted.max() == SNOW_WATER_CAP_KG_M2


def test_cap_perennial_pixels_leaves_rasters_below_the_cap_untouched():
    swe = np.array([[0.1, np.nan], [4.9, 5.0]])
    depth = swe * 3
    out_swe, out_depth, capped = sa.cap_perennial_pixels(swe, depth)
    assert out_swe is swe and out_depth is depth
    assert not capped.any()
    over = np.array([[25.0, np.nan], [32.767, 0.5]])
    deep = np.array([[30.0, np.nan], [32.767, 1.0]])
    out_swe, out_depth, capped = sa.cap_perennial_pixels(over, deep)
    assert capped.tolist() == [[True, False], [True, False]]
    assert out_swe[0, 0] == 5.0 and out_depth[0, 0] == pytest.approx(30.0 * 5.0 / 25.0)
    assert np.isnan(out_swe[0, 1]) and np.isnan(out_depth[0, 1])
    assert out_swe[1, 1] == 0.5 and out_depth[1, 1] == 1.0
