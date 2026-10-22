"""Fuel statics on nested domains, through every child static route.

A child asks for fuel in its own ``[[domain]] static = { fuel = {...} }``.
The request rides the static carrier, and each route that builds a child's
statics builds the child's fuel on the child's own grid and fire mesh; the
``[static.highres]`` overlay every route goes through
(``apply_highres_statics``) rebuilds it from the overlaid fields.  The routes are nest_init, spawn, the
case-data relocation footprint, the moving nest's corridor crop and the
offline child.  Each route's fields must be byte-identical to building that
child's grid directly as a root with the same request and the same overlay.
A configuration that asks for no fuel builds and keys every child exactly as
before.

The geography is the tiny synthetic WPS_GEOG tree tests/test_statics_corridor
.py builds.  The overlay is a stand-in that changes terrain and land use per
cell (no tile fetch), so "after the overlay" is visible in the fuel.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from test_statics_corridor import (_RATIO, _REF_I, _REF_J,
                                   _catalog as _corridor_catalog, _child_dc, _parent_grid, _parent_run,
                                   _reference_grid, _synthetic_wps_geog)
from woof.static.build import (GeogSelection, build_static,
                               build_static_for_domain,
                               geog_selection_from_catalog)
from woof.static.fuel import (FUEL_FIELDS,
                              FuelStaticsRequest, fuel_for, fuel_row,
                              refresh_fuel_fields)
from woof.static.highres_production import HighresStaticConfig

START = datetime(2026, 10, 1)
REQUEST = FuelStaticsRequest(sr_x=3, sr_y=2)


@pytest.fixture(autouse=True)
def _host_and_python(monkeypatch):
    """CPU fuel body (the GPU body is held to it elsewhere); the synthetic
    tree runs on either static builder."""
    from woof.static import fuel
    monkeypatch.setattr(fuel, "gpu_available", lambda: False)


def _overlay(fields, grid, *, config, domain_id, case_date, landuse_attrs,
             urlopen=None):
    """A per-cell stand-in for the high-resolution overlay: new terrain, and
    land that rises above 700 m turned urban (no fuel)."""
    out = dict(fields)
    hgt = np.asarray(fields["HGT_M"], dtype=np.float64)
    out["HGT_M"] = hgt * 1.013 + 3.0
    out["LU_INDEX"] = np.where(hgt > 700.0, 13.0,
                               np.asarray(fields["LU_INDEX"]))
    return out, {"status": "APPLIED"}


@pytest.fixture(scope="module")
def geog_root(tmp_path_factory):
    return _synthetic_wps_geog(tmp_path_factory.mktemp("wps-geog"))


def _catalog(root, directory):
    directory.mkdir(parents=True, exist_ok=True)
    return _corridor_catalog(root, directory)


def _carrier(tmp_path, rows, *, enabled=True):
    return HighresStaticConfig(enabled=enabled, cache_root=tmp_path / "hr",
                               fuel=tuple(rows))


def _child(i_parent_start=_REF_I, j_parent_start=_REF_J):
    dc = _child_dc(i_parent_start, j_parent_start)
    dc.start_time = START
    dc.run.sf_lake_physics = 0
    dc.run.dx = dc.run.dy = _parent_grid().dx / _RATIO
    return dc


def _with_carrier(catalog, carrier):
    return SimpleNamespace(**{**vars(catalog), "static_highres": carrier})


def _direct_root(grid, geog_root, tmp_path, request=REQUEST, *,
                 overlay=True):
    """The child's grid built directly as a root (domain 1) asking for the
    same fuel, through the root's own selection, overlay and refresh."""
    carrier = _carrier(tmp_path, [fuel_row(1, request)])
    catalog = _with_carrier(_catalog(geog_root, tmp_path), carrier)
    namelist = next(item.path for item in catalog.files
                    if item.role == "wps_namelist")
    selection = GeogSelection.from_case_data(SimpleNamespace(
        wps_namelist=namelist, geog_root=geog_root,
        static_highres=carrier), domain_id=1)
    assert selection.fuel == request
    fields = build_static(grid, geog_root, selection=replace(
        selection, fuel=None) if overlay else selection)
    if overlay:
        # The overlay first, then one fuel build from what it left.
        attrs = selection.landuse_global_attrs()
        fields, _ = _overlay(fields, grid, config=carrier, domain_id=1,
                             case_date=START.date(), landuse_attrs=attrs)
        fields = refresh_fuel_fields(fields, grid, selection,
                                     landuse_attrs=attrs)
    return fields


def _assert_bytes(got, want, names):
    for name in names:
        a, b = np.asarray(got[name]), np.asarray(want[name])
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert a.tobytes() == b.tobytes(), name


# --- the request -------------------------------------------------------------

def test_a_nest_can_ask_for_fuel_alone(tmp_path):
    from woof.static.highres_production import (parse_sealed_static_highres,
                                                resolve_static_highres,
                                                static_highres_identity)
    raw = {"domain": [
        {"grid_id": 1},
        {"grid_id": 2, "static": {"fuel": {"sr_x": 4, "sr_y": 4}}},
        {"grid_id": 3, "static": {"smooth_option": "none",
                                  "fuel": {"sr_x": 10, "sr_y": 10,
                                           "landfire": "fbfm40.tif"}}}]}
    carrier = resolve_static_highres(raw, source="case.toml",
                                     base_dir=tmp_path, spacings_m=(3000.0,))
    assert carrier.fuel == ((2, 4, 4, "", "auto"),
                            (3, 10, 10, str(tmp_path / "fbfm40.tif"),
                             "auto"))
    assert fuel_for(carrier, 1) is None
    assert fuel_for(carrier, 2) == FuelStaticsRequest(sr_x=4, sr_y=4)
    assert fuel_for(carrier, 3).landfire == tmp_path / "fbfm40.tif"
    # a sealed preparation restores exactly the rows it recorded
    sealed = static_highres_identity(carrier)
    restored = parse_sealed_static_highres(sealed, source="seal",
                                           base_dir=tmp_path)
    assert restored.fuel == carrier.fuel
    tampered = dict(sealed, fuel=[[2, 4, 4, "", "fbfm99"]])
    with pytest.raises(ValueError):
        parse_sealed_static_highres(tampered, source="seal",
                                    base_dir=tmp_path)


def test_a_malformed_domain_request_is_refused():
    from woof.static.fuel import parse_domain_fuel
    from woof.static.terrain_smoothing import WPS_DEFAULT, parse_domain_static
    assert parse_domain_static({"fuel": {"sr_x": 2, "sr_y": 2}},
                               source="c", grid_id=2) == WPS_DEFAULT
    for table, match in (({"sr_x": 2}, "missing"),
                         ({"sr_x": 2, "sr_y": 2, "sr": 3}, "unknown"),
                         ({"sr_x": 0, "sr_y": 2}, "sr_x"),
                         ({"sr_x": 2, "sr_y": 2, "landfire": 7}, "path"),
                         ("4", "must be a table")):
        with pytest.raises(ValueError, match=match):
            parse_domain_fuel(table, source="c", grid_id=2)


# --- no request: every child keyed and built as before ------------------------

def test_no_request_leaves_every_child_identity_and_field_as_before(
        geog_root, tmp_path, monkeypatch):
    from woof.ensemble.runtime_preparation import _digest, _plain
    from woof.static.terrain_smoothing import (catalog_with_smoothing,
                                               selection_carrier)
    plain = HighresStaticConfig(enabled=False, cache_root=tmp_path)
    assert selection_carrier(plain) is None
    assert "fuel" not in _plain(plain) and "fuel" not in plain.echo()
    legacy = {key: value for key, value in asdict(plain).items()
              if key != "fuel"}
    assert _digest(plain) == _digest(legacy)
    inner = _catalog(geog_root, tmp_path)
    assert catalog_with_smoothing(inner, plain) is inner
    enabled = HighresStaticConfig(enabled=True, cache_root=tmp_path)
    catalog = _with_carrier(inner, enabled)
    selection = geog_selection_from_catalog(catalog, 2)
    assert selection.fuel is None
    assert selection == geog_selection_from_catalog(inner, 2)
    assert hash(selection) == hash(geog_selection_from_catalog(inner, 2))
    # the child route with an overlay and no request: no fuel field, and
    # the overlay's own dict comes back untouched
    from woof.ingest import nest_init
    from woof.static import highres_production
    seen = []

    def overlay(fields, grid, **kwargs):
        out = _overlay(fields, grid, **kwargs)
        seen.append(out[0])
        return out

    monkeypatch.setattr(highres_production, "_apply_highres_statics",
                        overlay)
    grid = _reference_grid()
    fields = build_static_for_domain(grid, nest_init._static_catalog(catalog),
                                     2)
    assert not set(FUEL_FIELDS) & set(fields)
    out, _ = highres_production.apply_highres_statics(
        fields, grid, config=enabled, domain_id=2, case_date=START.date(),
        landuse_attrs=selection.landuse_global_attrs())
    assert out is seen[-1]


# --- every child route builds the child's own fuel, as a root build would ----

@pytest.fixture
def overlay(monkeypatch):
    """The overlay's body stood in for; apply_highres_statics itself (and
    its fuel rebuild) runs as every route runs it."""
    from woof.static import highres_production
    monkeypatch.setattr(highres_production, "_apply_highres_statics",
                        _overlay)


def _child_catalog(geog_root, tmp_path):
    carrier = _carrier(tmp_path, [fuel_row(2, REQUEST)])
    return _with_carrier(_catalog(geog_root, tmp_path / "child"), carrier)


def test_nest_init_builds_the_child_fuel_after_its_overlay(
        geog_root, tmp_path, overlay):
    """nest_init's child statics, as its initializer makes them: the
    catalog build, then the catalog's overlay."""
    from woof.ingest import nest_init
    from woof.static.highres_production import apply_highres_statics
    catalog = _child_catalog(geog_root, tmp_path)
    grid = _reference_grid()
    static_catalog = nest_init._static_catalog(catalog)
    fields = build_static_for_domain(grid, static_catalog, 2)
    assert set(FUEL_FIELDS) <= set(fields)  # the baseline's own fuel
    selection = geog_selection_from_catalog(static_catalog, 2)
    assert selection.fuel == REQUEST
    out, _ = apply_highres_statics(
        fields, grid, config=catalog.static_highres, domain_id=2,
        case_date=START.date(),
        landuse_attrs=selection.landuse_global_attrs())
    direct = _direct_root(grid, geog_root, tmp_path / "root")
    _assert_bytes(out, direct, sorted(direct))
    # the fuel is the overlaid land use's: high ground turned urban burns not
    urban = np.asarray(out["LU_INDEX"]) == 13.0
    assert urban.any()
    np.testing.assert_array_equal(np.asarray(out["NFUEL_CAT"])[urban], 14.0)
    assert out["NFUEL_CAT_FIRE"].shape == (9 * 2, 9 * 3)
    # its own geometry: the fire-mesh slope is at the child's spacing
    assert grid.dx == _parent_grid().dx / _RATIO


def test_a_spawned_nest_builds_its_own_fuel(geog_root, tmp_path, overlay):
    from woof.ingest.nest_init import _child_grid_from_parent
    from woof.ingest.nest_spawn_init import prepare_spawn_statics
    catalog = _child_catalog(geog_root, tmp_path)
    child = _child(6, 2)
    parent = SimpleNamespace(cfg=SimpleNamespace(grid_id=1),
                             grid=_parent_grid())
    result = prepare_spawn_statics(child, parent, catalog,
                                   valid_date=START.date())
    assert result["receipt"]["highres_overlay"] is True
    grid = _child_grid_from_parent(child, 1, _parent_grid())
    direct = _direct_root(grid, geog_root, tmp_path / "root")
    _assert_bytes(result["static_fields"], direct, sorted(direct))


def test_a_relocated_footprint_builds_its_own_fuel(geog_root, tmp_path,
                                                   overlay):
    from woof.ingest.relocation_init import _build_footprint_statics
    catalog = _child_catalog(geog_root, tmp_path)
    grid = _reference_grid().translated(2 * _RATIO, -1 * _RATIO)
    fields, selection, _, applied = _build_footprint_statics(
        grid, catalog, _child(_REF_I + 2, _REF_J - 1))
    assert applied and selection.fuel == REQUEST
    direct = _direct_root(grid, geog_root, tmp_path / "root")
    _assert_bytes(fields, direct, sorted(direct))


def test_a_moving_nest_corridor_crops_its_fuel_exactly(geog_root, tmp_path,
                                                       overlay):
    """The corridor carries no fuel plane: a footprint's fuel is built per
    placement from the crop's own LU_INDEX and HGT_M (exactly the
    footprint's), because the fire mesh continues the footprint's terrain
    past the footprint's own edge."""
    from woof.static.corridor import (build_child_statics_corridor,
                                      corridor_cost,
                                      corridor_footprint_statics_builder,
                                      load_child_statics_corridor,
                                      write_statics_corridor_set)
    plain = build_child_statics_corridor(
        child_dc=_child(), parent_run=_parent_run(),
        reference_grid=_reference_grid(),
        static_catalog=_catalog(geog_root, tmp_path / "plain"),
        static_highres=_carrier(tmp_path, []))
    carrier = _carrier(tmp_path, [fuel_row(2, REQUEST)])
    catalog = _catalog(geog_root, tmp_path / "child")
    built = build_child_statics_corridor(
        child_dc=_child(), parent_run=_parent_run(),
        reference_grid=_reference_grid(), static_catalog=catalog,
        static_highres=carrier)
    assert not set(FUEL_FIELDS) & set(built.fields)
    assert sorted(built.fields) == sorted(plain.fields)
    for name in built.fields:  # the request changes no corridor byte
        assert built.fields[name].tobytes() == plain.fields[name].tobytes()
    assert built.entry["fuel"]["row"] == list(fuel_row(2, REQUEST))
    assert built.entry["fuel"]["landuse"]["MMINLU"] == \
        "MODIFIED_IGBP_MODIS_NOAH"
    cost = corridor_cost(_child(), _parent_run())
    assert cost["host_bytes"] == built.entry["host_bytes"]
    receipt = write_statics_corridor_set(tmp_path / "corridor", [built])
    corridor = load_child_statics_corridor(
        tmp_path / "corridor", expected_set_receipt=receipt, grid_id=2,
        child_dc=_child(), parent_run=_parent_run(),
        reference_grid=_reference_grid())
    builder = corridor_footprint_statics_builder(corridor)
    for ip, jp in ((_REF_I, _REF_J), (1, 1), (8, 8), (2, 6)):
        grid = _reference_grid().translated((ip - _REF_I) * _RATIO,
                                            (jp - _REF_J) * _RATIO)
        fields = builder(grid, _child(ip, jp))
        direct = _direct_root(grid, geog_root, tmp_path / f"root{ip}{jp}")
        assert set(FUEL_FIELDS) <= set(fields)
        _assert_bytes(fields, direct, FUEL_FIELDS + ("HGT_M", "LU_INDEX"))
    # a LANDFIRE request cannot ride a corridor, and says why
    landfire = _carrier(tmp_path, [fuel_row(2, replace(
        REQUEST, landfire=tmp_path / "fbfm40.tif"))])
    with pytest.raises(ValueError, match="LANDFIRE fuel map"):
        build_child_statics_corridor(
            child_dc=_child(), parent_run=_parent_run(),
            reference_grid=_reference_grid(), static_catalog=catalog,
            static_highres=landfire)


def test_a_corridor_without_a_request_is_as_before(geog_root, tmp_path):
    from woof.static.corridor import build_child_statics_corridor
    catalog = _catalog(geog_root, tmp_path / "child")
    plain = build_child_statics_corridor(
        child_dc=_child(), parent_run=_parent_run(),
        reference_grid=_reference_grid(), static_catalog=catalog)
    assert not set(FUEL_FIELDS) & set(plain.fields)
    assert "fuel" not in plain.entry


def test_an_offline_child_builds_its_own_fuel(geog_root, tmp_path, overlay,
                                              monkeypatch):
    from woof import offline_child_geography as ocg
    from woof.static.terrain_smoothing import WPS_DEFAULT
    grid = _reference_grid()
    monkeypatch.setattr(ocg, "child_projected_grid",
                        lambda parent_frame, placement: (grid, {}))
    policy = ocg.ChildStaticPolicy(
        terrain=ocg.CHILD_TERRAIN_OWN, geog_root=geog_root,
        smoothing=WPS_DEFAULT,
        highres=_carrier(tmp_path, [fuel_row(2, REQUEST)]))
    cfg = SimpleNamespace(grid_id=2, nx=9, ny=9, topo_wind=0, gwd_opt=0)
    child = ocg.build_child_geography(None, None, cfg, policy,
                                      valid_time=START)
    direct = _direct_root(grid, geog_root, tmp_path / "root")
    _assert_bytes(child.fields, direct, FUEL_FIELDS + ("HGT_M",))


def test_the_case_loader_carries_a_domain_request_outside_the_experiment(
        tmp_path):
    from test_case_data import _EXPERIMENT_TOML, make_case_toml
    from woof.case_data import load_experiment_case
    old, data = load_experiment_case(make_case_toml(tmp_path))
    request = _EXPERIMENT_TOML.replace(
        "[[domain]]", "[[domain]]\nstatic = { fuel = { sr_x = 5, sr_y = 5 } }")
    new, data2 = load_experiment_case(
        make_case_toml(tmp_path, experiment=request))
    assert new == old  # the experiment (and its restart identity) unchanged
    assert data.static_highres is None
    assert fuel_for(data2.static_highres, 1) == FuelStaticsRequest(
        sr_x=5, sr_y=5)
    bad = _EXPERIMENT_TOML.replace(
        "[[domain]]", "[[domain]]\nstatic = { fuel = { sr_x = 5, sr = 5 } }")
    with pytest.raises(ValueError, match="unknown keys"):
        load_experiment_case(make_case_toml(tmp_path, experiment=bad))



def test_an_offline_child_config_asks_for_fuel_in_its_own_static(tmp_path):
    from woof import offline_child_geography as ocg
    config = tmp_path / "child.toml"
    cfg = SimpleNamespace(grid_id=1, dx=1000.0, sf_urban_physics=0,
                          use_wudapt_lcz=0, topo_wind=0, gwd_opt=0)
    config.write_text('[static]\nterrain = "own"\n'
                      'fuel = { sr_x = 4, sr_y = 2 }\n')
    policy = ocg.load_child_static_policy(config, cfg)
    assert fuel_for(policy.highres, 1) == FuelStaticsRequest(sr_x=4, sr_y=2)
    config.write_text('[static]\nterrain = "parent"\n'
                      'fuel = { sr_x = 4, sr_y = 2 }\n')
    with pytest.raises(ValueError, match="own land use and terrain"):
        ocg.load_child_static_policy(config, cfg)
    config.write_text('[static]\nterrain = "own"\n')
    assert fuel_for(ocg.load_child_static_policy(config, cfg).highres,
                    1) is None


def test_prepared_caches_carry_the_fuel_in_its_own_words(tmp_path):
    """A prepared route's native static cache stores float64; the fuel comes
    back in the build's dtypes with the build's bytes."""
    from test_native_wrf_contract import _complete_native_static, _grid
    from woof.native_wrf_contract import (load_native_static_cache,
                                          native_static_export_fields,
                                          write_native_static_cache)
    from woof.static.fuel import build_fuel_fields
    grid = _grid()
    fields = _complete_native_static(grid)
    fields["HGT_M"] = 100.0 + 7.3 * np.arange(12.0).reshape(3, 4)
    fields["LU_INDEX"] = np.array([[1, 7, 10, 17]] * 3, dtype=np.float64)
    attrs = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
             "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13}
    fuel = build_fuel_fields(grid, fields, REQUEST, landuse_attrs=attrs)
    path = tmp_path / "native-static.npz"
    write_native_static_cache(path, native_static_export_fields(
        {**fields, **fuel}, grid))
    loaded = load_native_static_cache(path, grid, 3, 4)
    _assert_bytes(loaded, fuel, FUEL_FIELDS)
    with np.load(path) as stored:  # the fire mesh is stored as built
        assert stored["NFUEL_CAT_FIRE"].dtype == np.uint8
        assert stored["ZSF"].dtype == np.float32
    plain_path = tmp_path / "plain.npz"
    write_native_static_cache(plain_path,
                              native_static_export_fields(fields, grid))
    assert not set(FUEL_FIELDS) & set(load_native_static_cache(
        plain_path, grid, 3, 4))
    partial = {**fields, "NFUEL_CAT": fuel["NFUEL_CAT"]}
    with pytest.raises(ValueError, match="incomplete"):
        from woof.native_wrf_contract import validate_native_static_fields
        validate_native_static_fields(partial, grid, 3, 4)



def test_a_cache_prepared_before_the_request_gains_its_fuel(geog_root,
                                                            tmp_path):
    """A prepared route's static cache made before the domain asked for
    fuel: the prepared overlay seam builds the requested fuel from the
    cache's own fields instead of running without it."""
    from woof.static.highres_production import apply_prepared_highres
    grid = _reference_grid()
    catalog = _catalog(geog_root, tmp_path / "c")
    plain = build_static_for_domain(grid, catalog, 2)
    assert not set(FUEL_FIELDS) & set(plain)
    selection = geog_selection_from_catalog(catalog, 2)
    carrier = _carrier(tmp_path, [fuel_row(2, REQUEST)], enabled=False)
    fields, _ = apply_prepared_highres(
        plain, grid, config=carrier, domain_id=2, case_date=START.date(),
        landuse_attrs=selection.landuse_global_attrs())
    direct = _direct_root(grid, geog_root, tmp_path / "root", overlay=False)
    _assert_bytes(fields, direct, FUEL_FIELDS)
    none, _ = apply_prepared_highres(
        plain, grid, config=_carrier(tmp_path, [], enabled=False),
        domain_id=2, case_date=START.date(),
        landuse_attrs=selection.landuse_global_attrs())
    assert none is plain


def test_a_windowed_relocation_keeps_the_footprints_fire_mesh_words():
    from woof.core.nest_relocation import RelocationRefusal
    from woof.ingest.relocation_init import window_static_fields
    cfg = SimpleNamespace(nx=6, ny=4)
    fields = {"HGT_M": np.arange(24.0).reshape(4, 6),
              "NFUEL_CAT": np.ones((4, 6)),
              "NFUEL_CAT_FIRE": np.arange(144, dtype=np.uint8)
              .reshape(8, 18),
              "ZSF": np.arange(144, dtype=np.float32).reshape(8, 18),
              "DZDXF": np.zeros((8, 18), np.float32),
              "DZDYF": np.zeros((8, 18), np.float32)}
    window = (slice(1, 3), slice(2, 5))
    out = window_static_fields(fields, window, cfg)
    np.testing.assert_array_equal(out["HGT_M"], fields["HGT_M"][1:3, 2:5])
    np.testing.assert_array_equal(out["ZSF"], fields["ZSF"][2:6, 6:15])
    np.testing.assert_array_equal(out["NFUEL_CAT_FIRE"],
                                  fields["NFUEL_CAT_FIRE"][2:6, 6:15])
    fields["ZSF"] = np.zeros((9, 18), np.float32)
    with pytest.raises(RelocationRefusal, match="fire-mesh extent"):
        window_static_fields(fields, window, cfg)


def test_a_rebuild_reads_the_landfire_window_once(tmp_path, monkeypatch):
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("pyproj")
    from rasterio.transform import from_origin
    from woof.static import fuel
    grid = _reference_grid()
    lat, lon = grid.latlon_mass()
    path = tmp_path / "fbfm13.tif"
    with rasterio.open(path, "w", driver="GTiff", width=400, height=400,
                       count=1, dtype="int16", crs="EPSG:4326",
                       transform=from_origin(lon.min() - 0.1, lat.max() + 0.1,
                                             0.002, 0.002)) as out:
        out.write(np.full((400, 400), 9, np.int16), 1)
    reads = []
    real = fuel._read_landfire_window

    def counted(*args, **kwargs):
        reads.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(fuel, "_read_landfire_window", counted)
    monkeypatch.setattr(fuel, "_WINDOW_CACHE", {})
    first = fuel.landfire_window(path, grid, 2, 2)
    second = fuel.landfire_window(path, grid, 2, 2)
    assert len(reads) == 1 and first is second
    fuel.landfire_window(path, grid.translated(3, 0), 2, 2)  # other cells
    assert len(reads) == 2
