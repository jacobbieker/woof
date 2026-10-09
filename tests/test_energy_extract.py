"""``woof energy extract``: derivations, merging, writers and refusals.

The samplers are replaced by fakes returning synthetic ``SampleResult``s,
so these tests read no history and need no native library.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import types

import numpy as np
import pytest

from woof.cli import build_parser
from woof.energy import extract, sample, sample_hex
from woof.energy.contracts import (
    FORECAST_COORDINATES,
    FORECAST_SCHEMA,
    FORECAST_VARIABLES,
    Plan,
    PlanDomain,
    dump_plan,
    file_ref,
    load_sites,
)
from woof.energy.sample import PROFILE_VARS, SURFACE_VARS, SampleResult

FIXTURES = Path(__file__).parent / "fixtures" / "energy"
RING = ((-4.5, 51.3), (-2.5, 51.3), (-2.5, 52.2), (-4.5, 52.2), (-4.5, 51.3))
T0 = np.datetime64("2026-06-21T12:00:00", "s")
WRF_KEYS = set(PROFILE_VARS) | {"U10", "V10", "T2", "Q2", "PSFC", "SWDOWN",
                                "RAINNC", "RAINC"}


def _times(count, start=T0, step_s=900):
    return start + np.arange(count) * np.timedelta64(step_s, "s")


def fake_result(times, lat, heights, *, profile_vars=PROFILE_VARS,
                surface_vars=SURFACE_VARS, u=3.0, v=4.0, theta=290.0,
                pres=90000.0, qv=0.005, rainnc=None, inside=None):
    nt, ns, nh = len(times), len(lat), len(heights)
    base = {
        "U": u, "V": v, "W": 0.1, "THETA": theta, "PRES": pres, "QVAPOR": qv,
        "QCLOUD": 1e-5, "QRAIN": 2e-5, "QICE": 3e-6, "QSNOW": 4e-6,
        "QGRAUP": 0.0,
    }
    profile = {k: np.full((nt, ns, nh), base[k], dtype=np.float64)
               for k in profile_vars}
    surface_base = {"U10": 2.0, "V10": -1.0, "T2": 288.0, "Q2": 0.007,
                    "PSFC": 100000.0, "SWDOWN": 600.0, "SWDDNI": 500.0,
                    "SWDDIF": 120.0, "COSZEN": 0.5, "RAINC": 0.0}
    surface = {}
    for key in surface_vars:
        if key == "RAINNC":
            values = (np.arange(nt, dtype=np.float64) * 0.9
                      if rainnc is None else np.asarray(rainnc, float))
            surface[key] = np.repeat(values[:, None], ns, axis=1)
        else:
            surface[key] = np.full((nt, ns), surface_base[key])
    return SampleResult(
        times=np.asarray(times, dtype="datetime64[s]"),
        heights_m=np.asarray(heights, dtype=np.float64),
        profile=profile, surface=surface,
        inside=np.ones(ns, bool) if inside is None else inside,
        terrain_m=np.full(ns, 50.0), dx_m=100.0, source="fake")


class FakeSampler:
    """Monkeypatched over ``woof.energy.sample``; per-domain behaviour is
    keyed by the run directory name."""

    def __init__(self, *, available=None, times=None, **fields):
        self.available = available or {}
        self.times = times or {}
        self.fields = fields
        self.calls = []

    def _domain(self, paths):
        return Path(paths[0]).parent.name

    def available_variables(self, paths, **_):
        return set(self.available.get(self._domain(paths), WRF_KEYS))

    def sample(self, paths, lat, lon, heights_m, profile_vars=PROFILE_VARS,
               surface_vars=SURFACE_VARS, **kwargs):
        domain = self._domain(paths)
        self.calls.append(dict(domain=domain, n=len(lat),
                               profile=list(profile_vars),
                               surface=list(surface_vars), kwargs=kwargs))
        times = self.times.get(domain, _times(4))
        return fake_result(times, lat, heights_m, profile_vars=profile_vars,
                           surface_vars=surface_vars, **self.fields)


@pytest.fixture
def fake(monkeypatch):
    def install(**kwargs):
        sampler = FakeSampler(**kwargs)
        monkeypatch.setattr(sample, "available_variables",
                            sampler.available_variables)
        monkeypatch.setattr(sample, "sample_wrfout", sampler.sample)
        monkeypatch.setattr(sample_hex, "available_variables",
                            sampler.available_variables)
        monkeypatch.setattr(sample_hex, "sample_mpas", sampler.sample)
        return sampler
    return install


def make_plan(tmp_path, *, split=1, files=True, topology="wrf-tiles",
              drop=0, sites_sha=None):
    sites_path = tmp_path / "sites.json"
    sites_path.write_text((FIXTURES / "sites_wales.json").read_text())
    ids = [s.site_id for s in load_sites(sites_path).sites]
    ids = ids[:len(ids) - drop]
    chunks = [ids[i::split] for i in range(split)]
    domains = []
    for n, chunk in enumerate(chunks):
        did = f"tile{n}"
        run = tmp_path / "runs" / did
        run.mkdir(parents=True)
        if files:
            for t in range(2):
                (run / f"wrfout_d01_{t}").write_bytes(b"")
        if topology == "hex-swath":
            domains.append(PlanDomain(
                domain_id=did, topology="hex-swath", role="mesh", dx_m=100.0,
                run_dir=f"runs/{did}", output_glob="wrfout_d01_*",
                footprint=RING, site_ids=tuple(chunk),
                mesh={"mesh_path": "mesh/static.nc"}))
        else:
            domains.append(PlanDomain(
                domain_id=did, topology="wrf-tiles", role="child", dx_m=100.0,
                run_dir=f"runs/{did}", output_glob="wrfout_d01_*",
                footprint=RING, config=f"{did}.toml", grid_id=1,
                site_ids=tuple(chunk)))
    ref = file_ref(sites_path, relative_to=tmp_path)
    if sites_sha is not None:
        ref["sha256"] = sites_sha
    plan = Plan(topology=topology, dx_m=100.0, start="2026-06-21T12",
                hours=1.0, domains=domains, sites_ref=ref)
    return dump_plan(plan, tmp_path / "plan.json")


def _args(plan, output, **overrides):
    record = dict(plan=str(plan), sites=None, heights_m=None, vars=None,
                  format="netcdf", output=str(output))
    record.update(overrides)
    return types.SimpleNamespace(**record)


# ---------------------------------------------------------------- derivations


def test_wind_direction_known_vectors():
    u = np.array([0.0, -1.0, 0.0, 1.0, 0.0])
    v = np.array([-1.0, 0.0, 1.0, 0.0, 0.0])
    got = extract.wind_from_direction(u, v)
    np.testing.assert_allclose(got[:4], [0.0, 90.0, 180.0, 270.0], atol=1e-9)
    assert np.isnan(got[4])


def test_line_normal_and_attack_angle_for_east_west_line():
    # (T=1, S=4): east-west conductors (bearing 90) and one without bearing.
    bearing = np.array([90.0, 90.0, 90.0, np.nan])
    u = np.array([[5.0, 0.0, 3.0, 5.0]])
    v = np.array([[0.0, 5.0, 3.0, 5.0]])
    normal = extract.line_normal_wind(u, v, bearing)
    np.testing.assert_allclose(normal[0, :3], [0.0, 5.0, 3.0], atol=1e-9)
    assert np.isnan(normal[0, 3])
    angle = extract.wind_attack_angle(u, v, bearing)
    np.testing.assert_allclose(angle[0, :3], [0.0, 90.0, 45.0], atol=1e-6)
    assert np.isnan(angle[0, 3])
    calm = extract.wind_attack_angle(np.zeros((1, 1)), np.zeros((1, 1)),
                                     np.array([90.0]))
    assert np.isnan(calm[0, 0])


def test_attack_angle_on_profile_shape_and_north_south_line():
    u = np.full((2, 1, 3), 4.0)
    v = np.zeros((2, 1, 3))
    angle = extract.wind_attack_angle(u, v, np.array([0.0]))
    np.testing.assert_allclose(angle, 90.0)


def test_temperature_from_theta():
    assert extract.air_temperature(np.array(300.0), np.array(1.0e5)) == 300.0
    got = extract.air_temperature(np.array(300.0), np.array(5.0e4))
    np.testing.assert_allclose(got, 300.0 * 0.5 ** (2.0 / 7.0), rtol=1e-12)


def test_density_and_humidity():
    rho = extract.air_density(np.array(1.0e5), np.array(300.0),
                              np.array(0.0))
    np.testing.assert_allclose(rho, 1.0e5 / (287.0 * 300.0))
    np.testing.assert_allclose(extract.specific_humidity(np.array(0.01)),
                               0.01 / 1.01)
    es = extract.saturation_vapour_pressure(np.array(273.15))
    np.testing.assert_allclose(es, 611.2)
    eps = 287.0 / 461.6
    q_sat = eps * es / (1.0e5 - es)
    rh, clipped = extract.relative_humidity(np.array([1.0e5, 1.0e5]),
                                            np.array([273.15, 273.15]),
                                            np.array([q_sat, 2 * q_sat]))
    np.testing.assert_allclose(rh, [100.0, 100.0], rtol=1e-9)
    assert clipped == 1


def test_precipitation_diff_and_reset():
    times = _times(4)
    acc = np.array([[0.0], [0.9], [0.2], [1.1]])
    rate, resets = extract.precipitation_rate(times, acc)
    assert np.isnan(rate[0, 0]) and np.isnan(rate[2, 0])
    np.testing.assert_allclose(rate[1, 0], 0.9 / 900.0)
    np.testing.assert_allclose(rate[3, 0], 0.9 / 900.0)
    assert resets == 1
    single, _ = extract.precipitation_rate(times[:1], acc[:1])
    assert np.isnan(single).all()


def test_cos_solar_zenith_noaa():
    # Equator, near the March equinox at local solar noon: sun overhead.
    times = np.array(["2026-03-20T12:07:00"], dtype="datetime64[s]")
    cosz = extract.cos_solar_zenith(times, np.array([0.0, 0.0]),
                                    np.array([0.0, 180.0]))
    assert cosz[0, 0] > 0.999
    assert cosz[0, 1] < -0.999
    # Midsummer noon at 51.5 N: zenith about 28 degrees.
    times = np.array(["2026-06-21T12:00:00"], dtype="datetime64[s]")
    cosz = extract.cos_solar_zenith(times, np.array([51.5]), np.array([0.0]))
    assert abs(np.degrees(np.arccos(cosz[0, 0])) - (51.5 - 23.44)) < 0.5


# ---------------------------------------------------------------- selection


def test_select_unknown_variable_refused():
    with pytest.raises(extract.ExtractRefused, match="valid names"):
        extract.select_variables(["u", "gust"], {"d": WRF_KEYS})


def test_select_unsuppliable_variable_refused_with_reason():
    with pytest.raises(extract.ExtractRefused, match="dni needs SWDDNI"):
        extract.select_variables(["dni"], {"d": WRF_KEYS})


def test_select_default_drops_and_notes():
    names, choice, notes = extract.select_variables(None, {"d": WRF_KEYS})
    assert "dni" not in names and "dhi" not in names
    assert any(n.startswith("dni not written") for n in notes)
    assert choice["d"]["cos_solar_zenith"] == ()
    assert choice["d"]["precipitation_rate"] == ("RAINNC", "RAINC")


# ---------------------------------------------------------------- end to end


def test_netcdf_round_trip(tmp_path, fake):
    netCDF4 = pytest.importorskip("netCDF4")
    sampler = fake()
    plan = make_plan(tmp_path)
    out = tmp_path / "forecast.nc"
    assert extract.main(_args(plan, out)) == 0
    assert not (tmp_path / "forecast.nc.partial").exists()
    call = sampler.calls[0]
    assert "SWDDNI" not in call["surface"] and "QGRAUP" not in call["profile"]
    with netCDF4.Dataset(out) as nc:
        nc.set_auto_mask(False)
        assert nc.schema == FORECAST_SCHEMA
        assert len(nc.plan_sha256) == 64 and len(nc.sites_sha256) == 64
        assert nc.created_utc and nc.woof_version
        assert {"time": 4, "site": 33, "height": 4} == {
            k: len(v) for k, v in nc.dimensions.items()}
        for name in FORECAST_COORDINATES:
            assert name in nc.variables, name
        assert nc["inside"].dtype == np.int8
        assert list(nc["height"][:]) == [10.0, 30.0, 90.0, 100.0]
        speed = nc["wind_speed"][:]
        np.testing.assert_allclose(speed, 5.0, rtol=1e-6)
        assert nc["wind_speed"].units == "m s-1"
        direction = nc["wind_from_direction"][0, 0, 0]
        np.testing.assert_allclose(direction,
                                   (270 - np.degrees(np.arctan2(4, 3))) % 360,
                                   rtol=1e-5)
        bearing = nc["bearing_deg"][:]
        normal = nc["line_normal_wind"][0, :, 0]
        has = np.isfinite(bearing)
        assert has.any() and (~has).any()
        assert np.isnan(normal[~has]).all()
        b = np.radians(bearing[has])
        np.testing.assert_allclose(normal[has],
                                   np.abs(3 * np.cos(b) - 4 * np.sin(b)),
                                   rtol=1e-5, atol=1e-5)
        precip = nc["precipitation_rate"][:, 0]
        assert np.isnan(precip[0])
        np.testing.assert_allclose(precip[1:], 0.9 / 900.0, rtol=1e-5)
        assert "COSZEN" in nc["cos_solar_zenith"].notes
        hub = nc["hub_height_m"][:]
        assert np.isfinite(hub).any() and np.isnan(hub).any()
        assert nc["voltage_kv"].units == "kV"
        notes = json.loads(nc.notes)
        assert any("dni not written" in n for n in notes)
        assert str(nc["site_id"][0]) == "synthetic:line/1#0"
        assert nc["time"][1] - nc["time"][0] == 900


def test_csv_round_trip(tmp_path, fake, capsys):
    fake()
    plan = make_plan(tmp_path)
    out = tmp_path / "forecast.csv"
    rc = extract.main(_args(plan, out, format="csv",
                            vars=("wind_speed", "t2", "air_temperature")))
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["variables"] == ["wind_speed", "t2", "air_temperature"]
    assert summary["output"] == str(out)
    with open(out, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["time", "site_id", "height_m", "wind_speed",
                             "t2", "air_temperature"]
    assert len(rows) == 4 * 33 * (1 + 4)
    surface = rows[0]
    assert surface["height_m"] == "" and float(surface["t2"]) == 288.0
    assert surface["wind_speed"] == ""
    profile = rows[1]
    assert profile["height_m"] == "10" and profile["t2"] == ""
    np.testing.assert_allclose(float(profile["air_temperature"]),
                               290.0 * 0.9 ** (2 / 7), rtol=1e-6)
    assert rows[0]["time"] == "2026-06-21T12:00:00Z"


def test_multi_domain_merge_intersects_times(tmp_path, fake, capsys):
    pytest.importorskip("netCDF4")
    fake(times={"tile0": _times(5), "tile1": _times(4, start=T0 + 900)})
    plan = make_plan(tmp_path, split=2)
    out = tmp_path / "forecast.nc"
    assert extract.main(_args(plan, out, vars=("u", "precipitation_rate"))) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["domains"] == ["tile0", "tile1"]
    assert summary["times"] == 4
    assert any("intersection" in n for n in summary["notes"])
    import netCDF4
    with netCDF4.Dataset(out) as nc:
        nc.set_auto_mask(False)
        domains = list(nc["domain_id"][:])
        assert set(domains) == {"tile0", "tile1"}
        assert domains[0] == "tile0" and domains[1] == "tile1"
        precip = nc["precipitation_rate"][:]
        # Rates are differenced on the common axis: every site's first kept
        # step is NaN, whichever run started earlier.
        assert np.isnan(precip[0]).all()
        np.testing.assert_allclose(precip[1:], 0.9 / 900.0, rtol=1e-5)


def test_merge_differences_precipitation_on_common_axis(tmp_path, fake):
    pytest.importorskip("netCDF4")
    # tile0 writes every 15 min, tile1 every 30 min: both rates must cover
    # the same 30-minute interval in the merged forecast.
    fake(times={"tile0": _times(5), "tile1": _times(3, step_s=1800)})
    plan = make_plan(tmp_path, split=2)
    out = extract.extract_forecast(plan, output=tmp_path / "f.nc",
                                   variables=["precipitation_rate"])
    import netCDF4
    with netCDF4.Dataset(out) as nc:
        nc.set_auto_mask(False)
        assert len(nc.dimensions["time"]) == 3
        precip = nc["precipitation_rate"][:]
        domains = list(nc["domain_id"][:])
    tile0 = np.array([d == "tile0" for d in domains])
    # tile0's accumulation steps 0.9 per 15 min -> 1.8 per 30 min.
    np.testing.assert_allclose(precip[1:, tile0], 1.8 / 1800.0, rtol=1e-5)
    np.testing.assert_allclose(precip[1:, ~tile0], 0.9 / 1800.0, rtol=1e-5)


def test_requirements_cover_every_variable_and_derive(fake):
    assert set(extract.REQUIREMENTS) == set(FORECAST_VARIABLES)
    times = _times(3)
    lat = np.array([51.5, 51.6])
    every = set(PROFILE_VARS) | set(SURFACE_VARS)
    names, choice, _ = extract.select_variables(None, {"d": every})
    assert names == list(FORECAST_VARIABLES)
    result = fake_result(times, lat, [10.0, 50.0])
    derived = extract.derive_domain(
        result, choice["d"], lat=lat, lon=np.array([-3.0, -3.1]),
        bearing_deg=np.array([90.0, np.nan]), notes=lambda *a: None)
    for name, (dims, _, _) in FORECAST_VARIABLES.items():
        assert derived[name].shape == ((3, 2, 2) if len(dims) == 3
                                       else (3, 2)), name
    np.testing.assert_allclose(derived["dhi"], 120.0)
    np.testing.assert_allclose(derived["cos_solar_zenith"], 0.5)


def test_unwritten_helper_note_is_kept(tmp_path, fake, capsys):
    fake(available={"tile0": WRF_KEYS | {"SWDDNI"}})
    plan = make_plan(tmp_path)
    assert extract.main(_args(plan, tmp_path / "f.csv", format="csv",
                              vars=("dhi",))) == 0
    notes = json.loads(capsys.readouterr().out)["notes"]
    assert any(n.startswith("cos_solar_zenith (used, not written)")
               and "NOAA" in n for n in notes)


def test_refuses_directory_output_and_keeps_it(tmp_path, fake, capsys):
    fake()
    plan = make_plan(tmp_path)
    keep = tmp_path / "results"
    keep.mkdir()
    (keep / "precious.txt").write_text("x")
    assert extract.main(_args(plan, keep, vars=("u",))) == 2
    assert "is a directory" in capsys.readouterr().err
    assert (keep / "precious.txt").read_text() == "x"


def test_zarr_refuses_non_store_directory(tmp_path, fake):
    pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    fake()
    plan = make_plan(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("x")
    with pytest.raises(extract.ExtractRefused, match="not a Zarr store"):
        extract.extract_forecast(plan, output=other, fmt="zarr",
                                 variables=["u"])
    assert (other / "notes.txt").exists()


def test_icechunk_refuses_existing_file(tmp_path, fake, capsys):
    pytest.importorskip("xarray")
    pytest.importorskip("icechunk")
    fake()
    plan = make_plan(tmp_path)
    existing = tmp_path / "forecast.nc"
    existing.write_bytes(b"netcdf")
    assert extract.main(_args(plan, existing, format="icechunk",
                              vars=("u",))) == 2
    assert "not an Icechunk repository" in capsys.readouterr().err
    assert existing.read_bytes() == b"netcdf"


def test_write_failure_is_reported_not_raised(tmp_path, fake, capsys,
                                              monkeypatch):
    fake()
    plan = make_plan(tmp_path)

    def broken(data, path):
        Path(str(path) + ".partial").write_text("half")
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(extract, "write_csv", broken)
    out = tmp_path / "f.csv"
    assert extract.main(_args(plan, out, format="csv", vars=("u",))) == 1
    assert "Permission denied" in capsys.readouterr().err
    assert not (tmp_path / "f.csv.partial").exists()


def test_zarr_rewrite_replaces_store(tmp_path, fake):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    fake()
    plan = make_plan(tmp_path)
    out = tmp_path / "f.zarr"
    extract.extract_forecast(plan, output=out, fmt="zarr", variables=["u"])
    extract.extract_forecast(plan, output=out, fmt="zarr", variables=["t2"])
    assert not (tmp_path / "f.zarr.previous").exists()
    with xr.open_zarr(out, consolidated=False) as ds:
        assert "t2" in ds and "u" not in ds


def test_disjoint_domain_times_refused(tmp_path, fake):
    fake(times={"tile0": _times(2), "tile1": _times(2, start=T0 + 86400)})
    plan = make_plan(tmp_path, split=2)
    with pytest.raises(extract.ExtractRefused, match="share no valid time"):
        extract.extract_forecast(plan, output=tmp_path / "f.nc")


def test_dhi_fallback_note(tmp_path, fake, capsys):
    keys = WRF_KEYS | {"SWDDNI"}
    fake(available={"tile0": keys})
    plan = make_plan(tmp_path)
    out = tmp_path / "f.csv"
    assert extract.main(_args(plan, out, format="csv",
                              vars=("ghi", "dni", "dhi"))) == 0
    summary = json.loads(capsys.readouterr().out)
    assert any("SWDDIF not in the history" in n
               for n in summary["variable_notes"]["dhi"])
    with open(out, newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["dni"]) == 500.0
    assert 0.0 <= float(row["dhi"]) <= 600.0


def test_dhi_direct_when_swddif_present(tmp_path, fake, capsys):
    fake(available={"tile0": WRF_KEYS | {"SWDDNI", "SWDDIF", "COSZEN"}})
    plan = make_plan(tmp_path)
    assert extract.main(_args(plan, tmp_path / "f.csv", format="csv",
                              vars=("dhi", "cos_solar_zenith"))) == 0
    summary = json.loads(capsys.readouterr().out)
    assert "dhi" not in summary["variable_notes"]
    assert "cos_solar_zenith" not in summary["variable_notes"]


def test_hex_swath_passes_mesh_path(tmp_path, fake):
    sampler = fake()
    plan = make_plan(tmp_path, topology="hex-swath")
    extract.extract_forecast(plan, output=tmp_path / "f.csv", fmt="csv",
                             variables=["u"])
    assert sampler.calls[0]["kwargs"]["mesh_path"] == \
        tmp_path / "mesh" / "static.nc"


def test_heights_override(tmp_path, fake, capsys):
    fake()
    plan = make_plan(tmp_path)
    assert extract.main(_args(plan, tmp_path / "f.csv", format="csv",
                              heights_m=(20.0, 50.0), vars=("u",))) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["heights_m"] == [20.0, 50.0]
    assert any("--heights-m" in n for n in summary["notes"])


def test_cli_parses_extract():
    args = build_parser().parse_args(
        ["energy", "extract", "plan.json", "--vars", "u,v", "--format", "csv",
         "-o", "x.csv"])
    assert args.vars == ("u", "v") and args.format == "csv"


# ---------------------------------------------------------------- refusals


def test_refuses_sites_sha_mismatch(tmp_path, fake, capsys):
    fake()
    plan = make_plan(tmp_path, sites_sha="0" * 64)
    assert extract.main(_args(plan, tmp_path / "f.nc")) == 2
    assert "sha256" in capsys.readouterr().err
    assert not (tmp_path / "f.nc").exists()


def test_refuses_unowned_sites(tmp_path, fake):
    fake()
    plan = make_plan(tmp_path, drop=7)
    with pytest.raises(extract.ExtractRefused,
                       match=r"7 site\(s\) have no owning domain"):
        extract.extract_forecast(plan, output=tmp_path / "f.nc")


def test_refuses_missing_output_files(tmp_path, fake, capsys):
    fake()
    plan = make_plan(tmp_path, files=False)
    assert extract.main(_args(plan, tmp_path / "f.nc")) == 2
    assert "run woof energy run first" in capsys.readouterr().err


def test_refuses_unknown_var_via_main(tmp_path, fake, capsys):
    fake()
    plan = make_plan(tmp_path)
    assert extract.main(_args(plan, tmp_path / "f.nc", vars=("gust",))) == 2
    err = capsys.readouterr().err
    assert "gust" in err and "wind_speed" in err


def test_refuses_unsuppliable_var_via_main(tmp_path, fake, capsys):
    fake(available={"tile0": WRF_KEYS - {"W"}})
    plan = make_plan(tmp_path)
    assert extract.main(_args(plan, tmp_path / "f.nc", vars=("w",))) == 2
    assert "w needs W" in capsys.readouterr().err


def test_refuses_plan_without_sites_reference(tmp_path, fake):
    fake()
    plan_path = make_plan(tmp_path)
    document = json.loads(plan_path.read_text())
    document["sites"] = None
    plan_path.write_text(json.dumps(document))
    with pytest.raises(extract.ExtractRefused, match="--sites"):
        extract.extract_forecast(plan_path, output=tmp_path / "f.nc")
    # --sites rescues it.
    extract.extract_forecast(plan_path, output=tmp_path / "f.csv", fmt="csv",
                             sites_path=tmp_path / "sites.json",
                             variables=["u"])


def test_refuses_non_monotonic_times(tmp_path, fake):
    fake(times={"tile0": np.array([T0, T0], dtype="datetime64[s]")})
    plan = make_plan(tmp_path)
    with pytest.raises(extract.ExtractRefused, match="strictly increasing"):
        extract.extract_forecast(plan, output=tmp_path / "f.nc")


def test_refuses_sampler_height_mismatch(tmp_path, fake, monkeypatch):
    sampler = fake()
    original = sampler.sample

    def wrong(paths, lat, lon, heights_m, **kwargs):
        return original(paths, lat, lon, [h + 1 for h in heights_m], **kwargs)

    monkeypatch.setattr(sample, "sample_wrfout", wrong)
    plan = make_plan(tmp_path)
    with pytest.raises(extract.ExtractRefused, match="returned heights"):
        extract.extract_forecast(plan, output=tmp_path / "f.nc",
                                 variables=["u"])


def test_zarr_refuses_without_xarray(tmp_path, fake, monkeypatch):
    import builtins

    fake()
    plan = make_plan(tmp_path)
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "xarray":
            raise ModuleNotFoundError("No module named 'xarray'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(extract.ExtractRefused, match="needs xarray"):
        extract.extract_forecast(plan, output=tmp_path / "f.zarr",
                                 fmt="zarr", variables=["u"])


def test_zarr_round_trip(tmp_path, fake):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    fake()
    plan = make_plan(tmp_path)
    out = extract.extract_forecast(plan, output=tmp_path / "f.zarr",
                                   fmt="zarr", variables=["u", "t2"])
    with xr.open_zarr(out, consolidated=False) as ds:
        assert ds.attrs["schema"] == FORECAST_SCHEMA
        assert ds["u"].dims == FORECAST_VARIABLES["u"][0]
        assert str(ds["site_id"].values[0]) == "synthetic:line/1#0"
        np.testing.assert_allclose(ds["u"].values, 3.0)


def test_icechunk_round_trip(tmp_path, fake, capsys):
    xr = pytest.importorskip("xarray")
    ic = pytest.importorskip("icechunk")
    fake()
    plan = make_plan(tmp_path)
    out = tmp_path / "repo"
    assert extract.main(_args(plan, out, format="icechunk",
                              vars=("wind_speed",))) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["snapshot"]
    repo = ic.Repository.open(ic.local_filesystem_storage(str(out)))
    session = repo.readonly_session("main")
    with xr.open_zarr(session.store, consolidated=False) as ds:
        np.testing.assert_allclose(ds["wind_speed"].values, 5.0, rtol=1e-6)
