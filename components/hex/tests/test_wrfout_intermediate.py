"""The wrfout forcing source: a WOOF WRF run's history onto hex intermediates.

Every test writes a tiny synthetic Lambert wrfout itself, runs the door over
it, and reads the result back through this tree's own WPS reader.  The
interpolation operator is the engine's; here it runs its NumPy reference
(the backend handed in has no native entry), which is the authority the Rust
entry is pinned to.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

netCDF4 = pytest.importorskip("netCDF4")

from woof.hex import hrrr_intermediate as hi  # noqa: E402
from woof.hex import wrfout_intermediate as wi  # noqa: E402
from woof.hex.cli import build_parser  # noqa: E402
from woof.hex.hrrr_intermediate import IntermediateRefusal  # noqa: E402
from woof.hex.wps_intermediate import WpsIntermediateReader  # noqa: E402
from woof.static.projection import projection_class  # noqa: E402

NX, NY, NZ = 40, 36, 8
CEN = (39.0, -98.0)
DX = 3000.0
STAND_LON = -80.0  # far from the centre, so alpha is ~13 degrees everywhere
U_EARTH, V_EARTH = 10.0, 0.0
U10_EARTH = 5.0


def _grid(cen=CEN):
    return projection_class("lambert")(cen[0], cen[1], 30.0, 60.0, STAND_LON, DX, DX, NX + 1, NY + 1)


def _theta(k):
    return 290.0 + 4.0 * k


def _pressure(k):
    return 95000.0 - 8000.0 * k


def _qv(k):
    return 0.008 * np.exp(-k / 3.0)


def write_wrfout(path: Path, times, *, drop=(), attrs=None, optional=True, cen_attr=None,
                 soil_layers=4) -> Path:
    grid = _grid()
    lat, lon = grid.latlon_mass()
    _, lon_u = grid.latlon_u()
    _, lon_v = grid.latlon_v()
    sina, cosa = grid.rotation(lon)
    sin_u, cos_u = grid.rotation(lon_u)
    sin_v, cos_v = grid.rotation(lon_v)
    nt = len(times)
    with netCDF4.Dataset(str(path), "w") as ds:
        for name, size in (("Time", None), ("DateStrLen", 19), ("west_east", NX),
                           ("south_north", NY), ("bottom_top", NZ), ("west_east_stag", NX + 1),
                           ("south_north_stag", NY + 1), ("bottom_top_stag", NZ + 1),
                           ("soil_layers_stag", soil_layers)):
            ds.createDimension(name, size)
        base = {"MAP_PROJ": 1, "MAP_PROJ_CHAR": "Lambert Conformal", "TRUELAT1": 30.0,
                "TRUELAT2": 60.0, "STAND_LON": STAND_LON,
                "CEN_LAT": CEN[0] if cen_attr is None else cen_attr[0],
                "CEN_LON": CEN[1] if cen_attr is None else cen_attr[1],
                "DX": DX, "DY": DX, "WEST-EAST_GRID_DIMENSION": NX + 1,
                "SOUTH-NORTH_GRID_DIMENSION": NY + 1, "SF_SURFACE_PHYSICS": 2}
        base.update(attrs or {})
        for key, value in base.items():
            ds.setncattr(key, value)

        def var(name, dims, values, dtype="f4"):
            if name in drop:
                return
            v = ds.createVariable(name, dtype, dims)
            v[:] = values

        times_arr = np.array([list(t.strftime("%Y-%m-%d_%H:%M:%S")) for t in times], dtype="S1")
        var("Times", ("Time", "DateStrLen"), times_arr, dtype="S1")
        m2 = ("Time", "south_north", "west_east")
        m3 = ("Time", "bottom_top", "south_north", "west_east")

        def plane(values):
            return np.broadcast_to(np.asarray(values, dtype=np.float32), (nt, NY, NX))

        def column(per_level):
            col = np.array([per_level(k) for k in range(NZ)], dtype=np.float32)
            return np.broadcast_to(col[None, :, None, None], (nt, NZ, NY, NX))

        var("XLAT", m2, plane(lat))
        var("XLONG", m2, plane(lon))
        var("SINALPHA", m2, plane(sina))
        var("COSALPHA", m2, plane(cosa))
        # Grid-relative winds for a uniform earth-relative flow: the inverse
        # of u_e = u cos - v sin, v_e = v cos + u sin.
        u_grid = U_EARTH * cos_u + V_EARTH * sin_u
        v_grid = -U_EARTH * sin_v + V_EARTH * cos_v
        var("U", ("Time", "bottom_top", "south_north", "west_east_stag"),
            np.broadcast_to(u_grid[None, None], (nt, NZ, NY, NX + 1)))
        var("V", ("Time", "bottom_top", "south_north_stag", "west_east"),
            np.broadcast_to(v_grid[None, None], (nt, NZ, NY + 1, NX)))
        var("T", m3, column(lambda k: _theta(k) - 300.0))
        var("P", m3, column(lambda k: 0.0))
        var("PB", m3, column(_pressure))
        z_w = 200.0 + 1000.0 * np.arange(NZ + 1)
        var("PH", ("Time", "bottom_top_stag", "south_north", "west_east"),
            np.zeros((nt, NZ + 1, NY, NX), dtype=np.float32))
        var("PHB", ("Time", "bottom_top_stag", "south_north", "west_east"),
            np.broadcast_to((9.81 * z_w).astype(np.float32)[None, :, None, None], (nt, NZ + 1, NY, NX)))
        var("QVAPOR", m3, column(_qv))
        var("PSFC", m2, plane(97000.0))
        var("T2", m2, plane(288.0))
        var("Q2", m2, plane(0.008))
        var("U10", m2, plane(U10_EARTH * cosa))
        var("V10", m2, plane(-U10_EARTH * sina))
        var("TSK", m2, plane(289.0))
        var("HGT", m2, plane(200.0))
        land = np.zeros((NY, NX), dtype=np.float32)
        land[:, : NX // 2] = 1.0
        var("LANDMASK", m2, plane(land))
        var("SEAICE", m2, plane(0.0))
        var("SNOW", m2, plane(0.0))
        soil = ("Time", "soil_layers_stag", "south_north", "west_east")
        var("TSLB", soil, np.full((nt, soil_layers, NY, NX), 285.0, dtype=np.float32))
        var("SMOIS", soil, np.full((nt, soil_layers, NY, NX), 0.3, dtype=np.float32))
        if optional:
            var("SST", m2, plane(290.0))
            var("QCLOUD", m3, column(lambda k: 1.0e-4 if k == 2 else 0.0))
    return path


class _NoNative:
    """A preprocessing backend without the indexed-donor entry: the engine's
    NumPy reference operator runs."""


def _request(paths, *, point=CEN, reach_km=20.0, margin_km=5.0, out_dir, **extra):
    bounds = wi._cap_bounds(point, reach_km + margin_km)
    return wi.WrfoutRequest(
        source=hi.source_row("wrfout"), paths=tuple(paths), bounds=bounds,
        out_dir=out_dir, backend=_NoNative(), **extra,
    )


def _read(path: Path):
    with WpsIntermediateReader(path) as reader:
        return list(reader.iter_fields())


T0 = datetime(2026, 10, 10, 0)


# ---------------------------------------------------------------------------
# the round trip
# ---------------------------------------------------------------------------
def test_a_synthetic_lambert_wrfout_becomes_readable_intermediates(tmp_path: Path) -> None:
    files = [write_wrfout(tmp_path / f"wrfout_d01_{t:%Y-%m-%d_%H_%M_%S}", [t])
             for t in (T0, T0 + timedelta(hours=1))]
    out = tmp_path / "met"
    receipt = wi.build_wrfout_intermediates(_request(files, out_dir=out), log=lambda _m: None)

    names = sorted(p.name for p in out.glob("MET:*"))
    assert names == ["MET:2026-10-10_00", "MET:2026-10-10_01"]
    assert receipt["source"] == "wrfout"
    assert receipt["levels"]["count"] == NZ + 1
    assert receipt["levels"]["init_switches"]["--nfgsoillevels"] == 4
    assert [s["sha256"] for s in receipt["wrfout"]] == [hi.sha256_file(p) for p in files]
    assert receipt["hydrometeors_carried"] == ["QC"]
    assert receipt["wrf_grid"]["projection"] == "lambert"

    fields = _read(out / "MET:2026-10-10_00")
    assert {f.valid_time for f in fields} == {"2026-10-10_00:00:00"}
    assert all(f.projection.name == "latlon" and not f.is_wind_grid_relative for f in fields)
    by = {}
    for f in fields:
        by.setdefault(f.field, {})[f.level] = f
    for name in ("PRESSURE", "GHT", "TT", "SPECHUMD", "UU", "VV", "QC"):
        assert sorted(level for level in by[name] if level != hi.SURFACE_LEVEL) == \
            [float(k + 1) for k in range(NZ)], name
    for name in ("TT", "SPECHUMD", "UU", "VV", "PSFC", "SKINTEMP", "SOILHGT", "SNOW",
                 "LANDSEA", "SEAICE", "SST", "ST000010", "ST010040", "ST040100", "ST100200",
                 "SM000010", "SM010040", "SM040100", "SM100200"):
        assert hi.SURFACE_LEVEL in by[name], name
    assert len({f.level for f in fields}) == NZ + 1
    units = {f.field: f.units for f in fields}
    assert units["PRESSURE"] == "Pa" and units["GHT"] == "m" and units["TT"] == "K"
    assert units["SPECHUMD"] == "kg kg-1" and units["UU"] == "m s-1" and units["SM000010"] == "m3 m-3"

    for k in range(NZ):
        tag = float(k + 1)
        p = _pressure(k)
        assert np.allclose(by["PRESSURE"][tag].values, p, rtol=1e-6)
        assert np.allclose(by["TT"][tag].values, _theta(k) * (p / 1.0e5) ** (2.0 / 7.0), atol=1e-3)
        assert np.allclose(by["GHT"][tag].values, 700.0 + 1000.0 * k, atol=1e-2)
        q = _qv(k)
        assert np.allclose(by["SPECHUMD"][tag].values, q / (1.0 + q), rtol=1e-5)
    surface = {name: by[name][hi.SURFACE_LEVEL].values for name in by if hi.SURFACE_LEVEL in by[name]}
    assert np.allclose(surface["SPECHUMD"], 0.008 / 1.008, rtol=1e-5)
    assert np.allclose(surface["SOILHGT"], 200.0) and np.allclose(surface["PSFC"], 97000.0)
    assert set(np.unique(surface["LANDSEA"])) <= {0.0, 1.0}
    assert np.allclose(surface["ST000010"], 285.0) and np.allclose(surface["SM100200"], 0.3)


def test_the_winds_come_out_earth_relative(tmp_path: Path) -> None:
    files = [write_wrfout(tmp_path / "wrfout_d01_a", [T0])]
    out = tmp_path / "met"
    wi.build_wrfout_intermediates(_request(files, out_dir=out), log=lambda _m: None)
    fields = _read(out / "MET:2026-10-10_00")
    winds = {(f.field, f.level): f.values for f in fields if f.field in ("UU", "VV")}
    # The grid-relative V is ~ -2.3 m/s here (alpha ~ 13 deg); a door that
    # skipped the rotation would write that, not zero.
    grid = _grid()
    sina, _ = grid.rotation(np.array([CEN[1]]))
    assert abs(U_EARTH * float(sina[0])) > 1.5
    for k in range(NZ):
        assert np.allclose(winds[("UU", float(k + 1))], U_EARTH, atol=2e-2)
        assert np.allclose(winds[("VV", float(k + 1))], V_EARTH, atol=2e-2)
    assert np.allclose(winds[("UU", hi.SURFACE_LEVEL)], U10_EARTH, atol=1e-2)
    assert np.allclose(winds[("VV", hi.SURFACE_LEVEL)], 0.0, atol=1e-2)


def test_sub_hourly_history_is_named_with_minutes_and_read_by_header(tmp_path: Path) -> None:
    times = [T0, T0 + timedelta(minutes=30), T0 + timedelta(minutes=60)]
    files = [write_wrfout(tmp_path / "wrfout_d01_all", times)]
    out = tmp_path / "met"
    receipt = wi.build_wrfout_intermediates(_request(files, out_dir=out), log=lambda _m: None)
    assert sorted(p.name for p in out.glob("MET:*")) == [
        "MET:2026-10-10_00:00", "MET:2026-10-10_00:30", "MET:2026-10-10_01:00"]
    assert receipt["interval_seconds"] == [1800]
    found = hi.intermediate_valid_times(out)
    assert [t for t, _ in found] == times
    # The forecast hour in each header is the lead from the first wrfout time.
    leads = [_read(p)[0].forecast_hour for _, p in found]
    assert leads == pytest.approx([0.0, 0.5, 1.0])


def test_a_polygon_cull_region_bounds_the_target(tmp_path: Path) -> None:
    region = {"kind": "polygon", "vertices_deg": [[38.9, -98.1], [38.9, -97.9], [39.1, -97.9]]}
    south, west, north, east = wi.region_bounds(region, halo_km=0.0, margin_km=11.1195, origin="t")
    assert south == pytest.approx(38.8) and north == pytest.approx(39.2)
    assert west < -98.1 and east > -97.9
    path = tmp_path / "cull_region.json"
    path.write_text(json.dumps(region))
    arguments = build_parser().parse_args([
        "intermediate", "--source", "wrfout", "--wrfout-glob", str(tmp_path / "w*"),
        "--cull-region", str(path), "--out-dir", str(tmp_path / "o"),
    ])
    # The boundary rings sit outside the cut and the region file does not
    # say how wide they are: --halo-km is required, not guessed.
    with pytest.raises(IntermediateRefusal) as refusal:
        wi.bounds_from_arguments(arguments)
    assert "needs --halo-km" in str(refusal.value)
    arguments.halo_km = 10.0
    (bounds, basis) = wi.bounds_from_arguments(arguments)
    assert basis["region_kind"] == "polygon" and basis["halo_km"] == 10.0
    assert bounds[0] == pytest.approx(38.9 - (10.0 + hi.DEFAULT_MARGIN_KM) / hi.KM_PER_DEG)
    # A document whose cull_region is a path, not a shape, is refused by name.
    pointer = tmp_path / "mesh.json"
    pointer.write_text(json.dumps({"cull_region": "cull_region.json"}))
    arguments.cull_region = pointer
    with pytest.raises(IntermediateRefusal) as refusal:
        wi.bounds_from_arguments(arguments)
    assert "not a cap or polygon object" in str(refusal.value)
    with pytest.raises(IntermediateRefusal) as refusal:
        wi.region_bounds({"kind": "raster"}, halo_km=0.0, margin_km=0.0, origin="t")
    assert "cap" in str(refusal.value) and "polygon" in str(refusal.value)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
def test_a_cull_outside_the_wrfout_interior_is_refused(tmp_path: Path) -> None:
    files = [write_wrfout(tmp_path / "wrfout_d01_a", [T0])]
    lat, lon = _grid().latlon_mass()
    corner = (float(lat[2, 2]), float(lon[2, 2]))
    with pytest.raises(IntermediateRefusal) as refusal:
        wi.build_wrfout_intermediates(_request(files, point=corner, reach_km=5.0, margin_km=0.0,
                                               out_dir=tmp_path / "met"), log=lambda _m: None)
    assert "not inside the parent" in str(refusal.value)
    # The relaxed boundary rows count as outside: a box that clears the
    # array edge but reaches into the spec zone is refused too.
    near = (float(lat[NY // 2, 6]), float(lon[NY // 2, 6]))
    with pytest.raises(IntermediateRefusal):
        wi.build_wrfout_intermediates(_request(files, point=near, reach_km=1.0, margin_km=0.0,
                                               out_dir=tmp_path / "met2"), log=lambda _m: None)
    wi.build_wrfout_intermediates(_request(files, point=near, reach_km=1.0, margin_km=0.0,
                                           out_dir=tmp_path / "met3", edge_cells=0),
                                  log=lambda _m: None)
    assert not list((tmp_path / "met").glob("MET:*"))


def test_a_missing_variable_is_refused_by_name(tmp_path: Path) -> None:
    files = [write_wrfout(tmp_path / "wrfout_d01_a", [T0], drop=("QVAPOR", "SINALPHA"))]
    with pytest.raises(IntermediateRefusal) as refusal:
        wi.build_wrfout_intermediates(_request(files, out_dir=tmp_path / "met"), log=lambda _m: None)
    assert "QVAPOR" in str(refusal.value) and "SINALPHA" in str(refusal.value)


def test_duplicate_times_are_refused(tmp_path: Path) -> None:
    files = [write_wrfout(tmp_path / f"wrfout_{n}", [T0]) for n in ("a", "b")]
    with pytest.raises(IntermediateRefusal) as refusal:
        wi.build_wrfout_intermediates(_request(files, out_dir=tmp_path / "met"), log=lambda _m: None)
    assert "2026-10-10_00:00:00 appears in both" in str(refusal.value)
    same = [write_wrfout(tmp_path / "wrfout_c", [T0, T0])]
    with pytest.raises(IntermediateRefusal):
        wi.scan_wrfout(same)


def test_attributes_that_do_not_reproduce_the_coordinates_are_refused(tmp_path: Path) -> None:
    files = [write_wrfout(tmp_path / "wrfout_d01_a", [T0], cen_attr=(39.5, -98.0))]
    with pytest.raises(IntermediateRefusal) as refusal:
        wi.build_wrfout_intermediates(_request(files, out_dir=tmp_path / "met"), log=lambda _m: None)
    assert "XLAT/XLONG" in str(refusal.value)


def test_a_mixed_grid_and_a_non_noah_soil_column_are_refused(tmp_path: Path) -> None:
    a = write_wrfout(tmp_path / "wrfout_a", [T0])
    b = write_wrfout(tmp_path / "wrfout_b", [T0 + timedelta(hours=1)], attrs={"DX": 3001.0, "DY": 3001.0})
    with pytest.raises(IntermediateRefusal) as refusal:
        wi.scan_wrfout([a, b])
    assert "different WRF grid" in str(refusal.value)
    ruc = write_wrfout(tmp_path / "wrfout_ruc", [T0], attrs={"SF_SURFACE_PHYSICS": 3})
    with pytest.raises(IntermediateRefusal) as refusal:
        wi.build_wrfout_intermediates(_request([ruc], out_dir=tmp_path / "met"), log=lambda _m: None)
    assert "SF_SURFACE_PHYSICS=3" in str(refusal.value)
    nine = write_wrfout(tmp_path / "wrfout_nine", [T0], soil_layers=9)
    with pytest.raises(IntermediateRefusal) as refusal:
        wi.build_wrfout_intermediates(_request([nine], out_dir=tmp_path / "met"), log=lambda _m: None)
    assert "9 soil layers" in str(refusal.value)
    latlon = write_wrfout(tmp_path / "wrfout_ll", [T0], attrs={"MAP_PROJ": 6})
    with pytest.raises(IntermediateRefusal) as refusal:
        wi.build_wrfout_intermediates(_request([latlon], out_dir=tmp_path / "met"), log=lambda _m: None)
    assert "MAP_PROJ 6" in str(refusal.value)


def test_an_out_dir_already_holding_intermediates_is_refused(tmp_path: Path) -> None:
    files = [write_wrfout(tmp_path / "wrfout_d01_a", [T0])]
    out = tmp_path / "met"
    wi.build_wrfout_intermediates(_request(files, out_dir=out), log=lambda _m: None)
    with pytest.raises(IntermediateRefusal) as refusal:
        wi.build_wrfout_intermediates(_request(files, out_dir=out), log=lambda _m: None)
    assert "already holds intermediates" in str(refusal.value)


def test_every_optional_wrfout_field_has_its_row_entry() -> None:
    row = hi.source_row("wrfout")
    assert set(wi.OPTIONAL_3D.values()) <= set(row.optional_3d)
    assert set(wi.OPTIONAL_SURFACE.values()) <= set(row.optional_surface)


def test_optional_fields_absent_are_not_invented(tmp_path: Path) -> None:
    files = [write_wrfout(tmp_path / "wrfout_d01_a", [T0], optional=False)]
    out = tmp_path / "met"
    receipt = wi.build_wrfout_intermediates(_request(files, out_dir=out), log=lambda _m: None)
    names = {f.field for f in _read(out / "MET:2026-10-10_00")}
    assert "QC" not in names and "SST" not in names
    assert receipt["hydrometeors_carried"] == [] and "QC" in receipt["not_carried"]


# ---------------------------------------------------------------------------
# the console door
# ---------------------------------------------------------------------------
def _parse(*argv: str) -> argparse.Namespace:
    return build_parser().parse_args(["intermediate", *argv])


def test_the_wrfout_door_runs_from_the_console_script(tmp_path: Path, monkeypatch, capsys) -> None:
    write_wrfout(tmp_path / "wrfout_d01_2026-10-10_00_00_00", [T0])
    import woof.ingest.preprocess_backend as backend_module

    monkeypatch.setattr(backend_module, "resolve_preprocess_backend", lambda *_a, **_k: _NoNative())
    arguments = _parse("--source", "wrfout", "--wrfout-glob", str(tmp_path / "wrfout_d01_*"),
                       "--point", f"{CEN[0]},{CEN[1]}", "--radius-km", "10", "--margin-km", "5",
                       "--out-dir", str(tmp_path / "met"))
    assert arguments.handler(arguments) == 0
    printed = capsys.readouterr().out
    assert f"--nfglevels {NZ + 1}" in printed
    receipt = json.loads((tmp_path / "met" / "intermediate-receipt.json").read_text())
    assert receipt["target"]["dlat"] == pytest.approx(round(3.0 / hi.KM_PER_DEG, 6))
    assert receipt["target"]["dlon"] > receipt["target"]["dlat"]


def test_the_console_flags_belong_to_their_rows() -> None:
    with pytest.raises(IntermediateRefusal) as refusal:
        wi.request_from_arguments(_parse("--source", "wrfout", "--wrfout-glob", "x*",
                                         "--grib-dir", "d", "--point", "39,-98",
                                         "--radius-km", "5", "--out-dir", "o"))
    assert "--grib-dir belongs to a GRIB row" in str(refusal.value)
    # --hours selects GRIB forecast hours; a wrfout door converts every time
    # its glob holds, so a --hours that would silently do nothing is refused.
    with pytest.raises(IntermediateRefusal) as refusal:
        wi.request_from_arguments(_parse("--source", "wrfout", "--wrfout-glob", "x*",
                                         "--hours", "0-6", "--point", "39,-98",
                                         "--radius-km", "5", "--out-dir", "o"))
    assert "--hours belongs to a GRIB row" in str(refusal.value)
    # The GRIB road refuses the wrfout row rather than failing deep inside.
    with pytest.raises(IntermediateRefusal) as refusal:
        hi.request_from_arguments(_parse("--source", "wrfout", "--grib-dir", "d",
                                         "--cycle", "2026-09-13T15", "--out-dir", "o"))
    assert "wrfout door" in str(refusal.value)
    with pytest.raises(IntermediateRefusal) as refusal:
        wi.request_from_arguments(_parse("--source", "wrfout", "--point", "39,-98",
                                         "--radius-km", "5", "--out-dir", "o"))
    assert "needs --wrfout-glob" in str(refusal.value)
    with pytest.raises(IntermediateRefusal) as refusal:
        hi.request_from_arguments(_parse("--wrfout-glob", "x*", "--grib-dir", "d",
                                         "--cycle", "2026-09-13T15", "--out-dir", "o"))
    assert "belongs to --source wrfout" in str(refusal.value)
    with pytest.raises(IntermediateRefusal) as refusal:
        hi.request_from_arguments(_parse("--cycle", "2026-09-13T15", "--out-dir", "o",
                                         "--point", "39,-98", "--radius-km", "5"))
    assert "needs --grib-dir" in str(refusal.value)
    with pytest.raises(IntermediateRefusal) as refusal:
        wi.request_from_arguments(_parse("--source", "wrfout", "--wrfout-glob", "/nonexistent/w*",
                                         "--point", "39,-98", "--radius-km", "5", "--out-dir", "o"))
    assert "matches no file" in str(refusal.value)
    with pytest.raises(IntermediateRefusal) as refusal:
        wi.request_from_arguments(_parse("--source", "wrfout", "--wrfout-glob", "x*",
                                         "--wrf-edge-cells", "-1", "--point", "39,-98",
                                         "--radius-km", "5", "--out-dir", "o"))
    assert "negative" in str(refusal.value)
