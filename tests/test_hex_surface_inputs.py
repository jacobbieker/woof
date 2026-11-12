"""The hex surface-input contract for the generic column seam (host only).

woof.hex.surface_inputs declares which per-cell inputs each WRF column scheme
reads beyond the init's dozen; the sealed constructor holds a mapping to
exactly that set, the forecast binds it from the mesh's surface statics file,
and woof.hex.surface_statics builds that file the way geogrid/real.exe would.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import surface_inputs as si
from woof.hex.errors import ConfigurationRefusal

def _lake_physics(**extra):
    from woof.core.mpas_column_suite import MPAS_DEFAULT_PHYSICS

    return dict(MPAS_DEFAULT_PHYSICS, sf_lake_physics=1, **extra)


def _cfg(**values):
    return SimpleNamespace(**values)


def test_a_selection_without_extra_inputs_reads_none():
    assert si.required_inputs(_cfg(sf_surface_physics=4)) == ()
    assert si.seal(_cfg(sf_surface_physics=4), None, n_columns=3) == {}
    assert si.bind(None, None, latitude_rad=None, longitude_rad=None) == (
        {}, {"required": []})


def test_clm_lake_reads_its_mask_and_bathymetry_by_its_own_default():
    lake = [need.input.keyword for need in si.required_inputs(
        _cfg(sf_lake_physics=1, use_lakedepth=1))]
    assert lake == ["lakemask", "lake_mask_flag", "lake_depth", "lake_depth_flag"]
    # use_lakedepth = 0 is WRF's declared default depth: no bathymetry is read.
    assert [need.input.keyword for need in si.required_inputs(
        _cfg(sf_lake_physics=1, use_lakedepth=0))] == ["lakemask", "lake_mask_flag"]
    # RunConfig's own default (use_lakedepth = 1) decides when the run is silent.
    config = si.plan_config(_lake_physics())
    assert int(config.use_lakedepth) == 1
    assert len(si.required_inputs(config)) == 4


def test_every_scheme_that_reads_an_unstaged_input_is_refused_by_name():
    for cfg, keyword in (
            (_cfg(sf_surface_physics=1), "thermal_inertia"),
            (_cfg(sf_surface_physics=7), "landusef"),
            (_cfg(sf_urban_physics=1), "frc_urb2d"),
            (_cfg(sf_surf_irr_scheme=1), "irrigation"),
            (_cfg(lightning_option=1), "lightning_inputs"),
            (_cfg(sf_surface_physics=3, mosaic_lu=1), "landusef")):
        problems = si.unstaged_problems(cfg)
        assert problems and keyword in problems[0], cfg
    with pytest.raises(ConfigurationRefusal, match="thermal_inertia"):
        si.bind(dict(_lake_physics(), sf_lake_physics=0, sf_surface_physics=1), None, latitude_rad=[0.0],
                longitude_rad=[0.0])


def test_the_seal_refuses_a_missing_input_an_extra_input_and_a_wrong_array():
    cfg = _cfg(sf_lake_physics=1, use_lakedepth=1)
    good = {"lakemask": np.array([1, 0, 0], np.float32), "lake_mask_flag": 1,
            "lake_depth": np.array([12.5, 10, 10], np.float32),
            "lake_depth_flag": 1}
    sealed = si.seal(cfg, good, n_columns=3)
    assert sealed["lake_mask_flag"] == 1
    assert not sealed["lake_depth"].flags.writeable
    with pytest.raises(ValueError, match="missing \\['lake_depth'"):
        si.seal(cfg, {k: v for k, v in good.items() if k != "lake_depth"}, n_columns=3)
    with pytest.raises(ValueError, match="extra \\['frc_urb2d'\\]"):
        si.seal(cfg, dict(good, frc_urb2d=np.zeros(3, np.float32)), n_columns=3)
    with pytest.raises(ValueError, match="lake_depth must be float32\\[3\\]"):
        si.seal(cfg, dict(good, lake_depth=np.zeros(3)), n_columns=3)
    with pytest.raises(ValueError, match="missing"):
        si.seal(cfg, None, n_columns=3)
    with pytest.raises(ValueError, match="extra"):
        si.seal(_cfg(), good, n_columns=3)


def _sidecar(path, lat, lon, *, depth=True, flag=1, depth_flag=1,
             mask=(1, 0, 0), provenance="{}"):
    import netCDF4

    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("nCells", lat.size)
        dataset.createVariable("LAKEMASK", "i4", ("nCells",))[:] = list(mask)
        if depth:
            dataset.createVariable("LAKE_DEPTH", "f4", ("nCells",))[:] = [7.5, 10, 10]
            dataset.setncattr("LAKE_DEPTH_FLAG", np.int32(depth_flag))
        dataset.setncattr("LAKEFLAG", np.int32(flag))
        dataset.setncattr("schema", si.SCHEMA)
        dataset.createVariable("latCell", "f8", ("nCells",))[:] = lat
        # Stored in [0, 2 pi) as a static stores it; the mesh's are (-pi, pi].
        dataset.createVariable("lonCell", "f8", ("nCells",))[:] = np.mod(lon, 2 * np.pi)
        dataset.setncattr("provenance_json", provenance)
    return path


def test_bind_reads_the_sidecar_and_refuses_by_name(tmp_path):
    lat = np.radians([40.0, 41.0, 42.0])
    lon = np.radians([-80.0, -81.0, -82.0])
    physics = _lake_physics()
    path = _sidecar(tmp_path / "m.surface_statics.nc", lat, lon)
    inputs, receipt = si.bind(physics, path, latitude_rad=lat, longitude_rad=lon)
    assert sorted(inputs) == ["lake_depth", "lake_depth_flag", "lake_mask_flag", "lakemask"]
    assert inputs["lakemask"].dtype == np.float32
    np.testing.assert_array_equal(inputs["lake_depth"], np.float32([7.5, 10, 10]))
    assert receipt["inputs"]["lakemask"]["cells_set"] == 1
    # The seal takes exactly what bind returns.
    si.seal(si.plan_config(physics), inputs, n_columns=3)
    with pytest.raises(ConfigurationRefusal, match="no surface statics file"):
        si.bind(physics, None, latitude_rad=lat, longitude_rad=lon)
    with pytest.raises(ConfigurationRefusal, match="other cells"):
        si.bind(physics, path, latitude_rad=lat[::-1], longitude_rad=lon[::-1])
    with pytest.raises(ConfigurationRefusal, match="no LAKE_DEPTH"):
        si.bind(physics, _sidecar(tmp_path / "d.nc", lat, lon, depth=False),
                latitude_rad=lat, longitude_rad=lon)
    with pytest.raises(ConfigurationRefusal, match="LAKEFLAG is 0"):
        si.bind(physics, _sidecar(tmp_path / "f.nc", lat, lon, flag=0),
                latitude_rad=lat, longitude_rad=lon)
    # Given for a selection that reads nothing from it: accepted, never read.
    with pytest.raises(ConfigurationRefusal, match="never read"):
        si.bind(dict(_lake_physics(), sf_lake_physics=0), path, latitude_rad=lat, longitude_rad=lon)
    with pytest.raises(ConfigurationRefusal, match="never read"):
        si.bind(None, path, latitude_rad=lat, longitude_rad=lon)


def test_the_statics_matcher_serves_a_superset_reading_only_the_named_fields():
    static = {"file": "static", "variables": {"VAR": "var2d", "CON": "con", "OA1": "oa1"}}
    gsl = {"file": "ugwp", "variables": {"VAR": "var", "OL1": "ol1"}}
    assert si.match_statics_source(("VAR", "CON", "OA1"), [gsl, static]) is static
    cut = si.match_statics_source(("VAR", "CON"), [static, gsl])
    assert cut["variables"] == {"VAR": "var2d", "CON": "con"}
    assert static["variables"] == {"VAR": "var2d", "CON": "con", "OA1": "oa1"}
    assert si.match_statics_source(("VAR", "OL1", "CON"), [static, gsl]) is None


class _Source:
    """A WPS-index-shaped regular_ll source (1-based pixel centres)."""

    def __init__(self, values, *, x0_lon=-100.0, y0_lat=40.0, dx=0.01):
        self.values = np.asarray(values, np.float64)
        self.index = SimpleNamespace(dx=dx, islake=21, category_min=1, category_max=21)
        self.nx_global = self.values.shape[1]
        self.ny_global = self.values.shape[0]
        self.x0_lon, self.y0_lat, self.dx = x0_lon, y0_lat, dx
        self.path = SimpleNamespace(name="synthetic")

    def latlon_to_xy(self, lat, lon):
        return ((np.asarray(lon) - self.x0_lon) / self.dx + 1.0,
                (np.asarray(lat) - self.y0_lat) / self.dx + 1.0)

    def xy_to_latlon(self, x, y):
        return (self.y0_lat + (np.asarray(y) - 1.0) * self.dx,
                self.x0_lon + (np.asarray(x) - 1.0) * self.dx)

    def read_window(self, x0, x1, y0, y1):
        out = np.full((y1 - y0 + 1, x1 - x0 + 1), np.nan)
        for j in range(y0, y1 + 1):
            for i in range(x0, x1 + 1):
                if 1 <= i <= self.nx_global and 1 <= j <= self.ny_global:
                    out[j - y0, i - x0] = self.values[j - 1, i - 1]
        raw = np.where(np.isfinite(out), out, 0).astype(np.int64)
        return SimpleNamespace(values=lambda: out, raw=raw[None])


def _xyz(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                     np.sin(lat)), axis=-1)


class _NearestAssigner:
    def __init__(self, centres):
        self.centres = centres
        self.cells = np.arange(len(centres))

    def assign(self, xyz):
        return np.argmax(np.asarray(xyz) @ self.centres.T, axis=1)


def test_lake_depth_is_geogrids_cell_average_then_search_then_fill():
    from woof.hex import surface_statics as ss

    # A 20 x 20 source at 0.01 degrees; three cell centres.
    values = np.full((20, 20), np.nan)
    values[0:5, 0:5] = 4.0          # a lake under cell 0
    values[2, 2] = 8.0
    values[10, 12] = 30.0           # one valid pixel 3 pixels from cell 1's centre
    lat = np.array([40.02, 40.10, 40.18])
    lon = np.array([-99.98, -99.85, -99.82])
    centres = _xyz(lat, lon)
    geometry = SimpleNamespace(n_cells=3, xyz=centres, vertex_xyz=centres)
    # Cell 1's pixels are given to cell 2 here, so cell 1 holds none and searches.
    assigner = _NearestAssigner(centres)
    assigner.assign = lambda xyz: np.where(
        np.argmax(np.asarray(xyz) @ centres.T, axis=1) == 1, 2,
        np.argmax(np.asarray(xyz) @ centres.T, axis=1))
    water = np.array([True, True, False])
    depth, receipt = ss.lake_depth_cells(geometry, assigner, water, _Source(values))
    assert depth.dtype == np.float32
    assert depth[0] == np.float32((24 * 4.0 + 8.0) / 25)
    assert depth[1] == np.float32(30.0)          # search(5) reached it
    assert depth[2] == np.float32(ss.LAKE_DEPTH_FILL_M)  # land: masked, filled
    assert receipt["cells_searched"] == 1 and receipt["cells_search_found"] == 1


def test_the_lake_mask_vote_is_the_dominant_with_lakes_category_on_water():
    from woof.hex import surface_statics as ss

    categories = np.full((20, 20), 10.0)
    categories[0:12, 0:12] = 21.0     # lake dominates around cell 0
    categories[12:14, 0:4] = 17.0
    lat = np.array([40.02, 40.15])
    lon = np.array([-99.97, -99.85])
    centres = _xyz(lat, lon)
    geometry = SimpleNamespace(n_cells=2, xyz=centres, vertex_xyz=centres)
    mask, receipt = ss.lake_mask_vote(
        geometry, _NearestAssigner(centres), np.array([True, True]),
        _Source(categories))
    assert mask.tolist() == [True, False]
    assert receipt["islake"] == 21


def test_the_forecast_door_resolves_the_sidecar_beside_the_grid(tmp_path):
    from woof.hex import forecast_door

    grid = tmp_path / "m.grid.nc"
    grid.write_bytes(b"")
    profile = {"physics": _lake_physics()}
    physics = SimpleNamespace(column_profile=profile)
    arguments = SimpleNamespace(surface_statics=None, grid=grid)
    collected: list[str] = []
    found = forecast_door.resolve_surface_statics(arguments, physics, "m", grid, collected)
    assert found == tmp_path / f"m{si.FILE_SUFFIX}"
    assert collected and "surface-statics" in collected[0]
    plain = SimpleNamespace(column_profile={"physics": dict(_lake_physics(), sf_lake_physics=0)})
    assert forecast_door.resolve_surface_statics(arguments, plain, "m", grid, None) is None
    collected = []
    forecast_door.resolve_surface_statics(
        SimpleNamespace(surface_statics=grid, grid=grid), plain, "m", grid, collected)
    assert collected and "never read" in collected[0]
    collected = []
    slab = SimpleNamespace(column_profile={"physics": dict(
        _lake_physics(), sf_lake_physics=0, sf_surface_physics=1)})
    forecast_door.resolve_surface_statics(arguments, slab, "m", grid, collected)
    assert collected and "thermal_inertia" in collected[0]


def test_the_contract_table_lists_every_scheme():
    rows = si.contract_table()
    schemes = {row["scheme"] for row in rows}
    assert {"clm_lake", "slab", "pleim_xiu_lsm", "urban", "irrigation",
            "lightning"} <= schemes
    assert all(row["staged"] for row in rows if row["scheme"] == "clm_lake")
    assert not any(row["staged"] for row in rows if row["scheme"] != "clm_lake")


def test_topo_wind_2_reads_var_from_the_static_by_the_superset_match():
    from woof.hex.physics_suites import resolve_physics

    resolved = resolve_physics("arwen_thompson_column", {
        "config_gwdo_scheme": "off", "topo_wind": 2})
    assert resolved.seam == "engine"
    spec = resolved.column_profile["orographic_statistics"]
    assert spec["file"] == "static" and dict(spec["variables"]) == {"VAR": "var2d"}


def test_a_lake_selection_resolves_to_the_generic_seam_and_names_its_inputs():
    from woof.hex.physics_suites import resolve_physics

    resolved = resolve_physics("arwen_thompson_column", {
        "config_gwdo_scheme": "off", "sf_lake_physics": 1})
    assert resolved.seam == "engine"
    physics = resolved.column_profile["physics"]
    assert si.reads_surface_statics(physics)
    assert not si.unstaged_problems(si.plan_config(physics))


def test_bind_holds_the_sidecar_to_the_runs_static_and_its_values(tmp_path):
    import hashlib
    import json

    lat = np.radians([40.0, 41.0, 42.0])
    lon = np.radians([-80.0, -81.0, -82.0])
    physics = _lake_physics()
    static = tmp_path / "m.static.nc"
    static.write_bytes(b"the run's static")
    digest = hashlib.sha256(static.read_bytes()).hexdigest()
    good = _sidecar(tmp_path / "good.nc", lat, lon, provenance=json.dumps(
        {"parent_static": static.name, "parent_static_sha256": digest}))
    si.bind(physics, good, latitude_rad=lat, longitude_rad=lon, static_path=static)
    other = tmp_path / "other.static.nc"
    other.write_bytes(b"another static, same cell centres")
    with pytest.raises(ConfigurationRefusal, match="another static file"):
        si.bind(physics, good, latitude_rad=lat, longitude_rad=lon, static_path=other)
    with pytest.raises(ConfigurationRefusal, match="admits only \\[0, 1\\]"):
        si.bind(physics, _sidecar(tmp_path / "m2.nc", lat, lon, mask=(2, 0, 0)),
                latitude_rad=lat, longitude_rad=lon)
    with pytest.raises(ConfigurationRefusal, match="LAKE_DEPTH_FLAG is 0"):
        si.bind(physics, _sidecar(tmp_path / "f0.nc", lat, lon, depth_flag=0),
                latitude_rad=lat, longitude_rad=lon)


def test_the_seal_refuses_a_non_binary_mask_and_an_unstaged_flag():
    cfg = _cfg(sf_lake_physics=1, use_lakedepth=1)
    good = {"lakemask": np.array([1, 0, 0], np.float32), "lake_mask_flag": 1,
            "lake_depth": np.array([12.5, 10, 10], np.float32),
            "lake_depth_flag": 1}
    with pytest.raises(ValueError, match="admits only"):
        si.seal(cfg, dict(good, lakemask=np.array([2, 0, 0], np.float32)), n_columns=3)
    with pytest.raises(ValueError, match="lake_depth_flag=0"):
        si.seal(cfg, dict(good, lake_depth_flag=0), n_columns=3)


def test_the_forecast_finds_the_builders_default_beside_the_static(tmp_path):
    grid_dir, static_dir = tmp_path / "grid", tmp_path / "static"
    grid_dir.mkdir()
    static_dir.mkdir()
    static = static_dir / "point-x-cull.static.nc"
    built = si.builder_default_path(static)
    assert built == static_dir / f"point-x-cull{si.FILE_SUFFIX}"
    candidates = si.default_candidates(grid_dir / "g.grid.nc", static, "mesh")
    assert candidates[0] == grid_dir / f"mesh{si.FILE_SUFFIX}"
    assert built in candidates


def test_the_builder_bands_are_bounded_by_pixel_count():
    from woof.hex import surface_statics as ss

    window = (1, 43200, 1, 21600)
    bands = list(ss._bands(window))
    assert all((x1 - x0 + 1) * (y1 - y0 + 1) <= ss._BAND_PIXELS
               for x0, x1, y0, y1 in bands)
    assert bands[0][2] == 1 and bands[-1][3] == 21600
