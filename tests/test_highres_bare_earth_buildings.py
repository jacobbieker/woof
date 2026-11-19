"""Building domains take bare-earth terrain, not a surface model.

The grid-spacing default (HIGHRES_DEFAULT_BY_DX) names Copernicus GLO-30,
a radar surface model whose heights over a city include the buildings.  A
domain that resolves buildings (obstacle_opt) or runs an urban canopy
under LES on top of it counts them twice (phase-1 Manhattan LES,
2026-11-17: the default took GLO-30 on the HRRR route).  Such a DOMAIN
takes bare-earth USGS 3DEP inside its envelope under the default or
"auto", and is refused by name outside it; a surface model named
explicitly is kept with a warning.  Other domains of the same tree, and
every configuration without buildings, resolve exactly as before.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from woof.static import highres_production as production
from woof.static.highres_fetch import FootprintBBox, terrain_source_coverage
from woof.static.highres_production import (
    HIGHRES_DEFAULT_BY_DX, HighresRefusal, _experimental_terrain,
    _resolve_plan, bare_earth_domains, default_row_of,
    parse_sealed_static_highres, resolve_static_highres,
    static_highres_identity)

NYC = FootprintBBox(lat_min=40.5, lat_max=40.9, lon_min=-74.1, lon_max=-73.8)
PARIS = FootprintBBox(lat_min=48.7, lat_max=49.0, lon_min=2.2, lon_max=2.5)


def _raw(shared=None, **d2):
    raw = {"domain": [{"grid_id": 1, "parent_id": 0, "dx": 250.0},
                      {"grid_id": 2, "parent_id": 1, "parent_grid_ratio": 5,
                       **d2}]}
    if shared is not None:
        raw["shared"] = dict(shared)
    return raw


def _resolve(raw, **kw):
    return resolve_static_highres(raw, source="c.toml", base_dir=".", **kw)


@pytest.mark.parametrize("raw", [
    _raw(), _raw(obstacle_opt=0), _raw({"sf_urban_physics": 1}),
    _raw(km_opt=3), _raw({"sf_urban_physics": 1}, km_opt=4)])
def test_no_buildings_keeps_the_default_row_and_its_echo(raw):
    assert bare_earth_domains(raw) == ()
    config = _resolve(raw)
    assert default_row_of(config) is HIGHRES_DEFAULT_BY_DX[0]
    assert config.bare_earth == ()
    assert "bare_earth_domains" not in config.echo()
    assert config == _resolve(_raw())
    _, coverage = _resolve_plan(config, NYC, 2)
    assert coverage.source_id == "copernicus-dem-glo30"


@pytest.mark.parametrize("shared,keys", [
    (None, {"obstacle_opt": 1}),
    # sf_urban_physics is a [shared] key; km_opt is per domain.
    ({"sf_urban_physics": 1}, {"km_opt": 3}),
    ({"sf_urban_physics": 2}, {"km_opt": 2}),
])
def test_a_building_domain_takes_3dep_and_only_it(shared, keys):
    config = _resolve(_raw(shared, **keys))
    (row,) = config.bare_earth
    assert row[0] == 2
    # The default row stays the default (plan lines, child land cover).
    assert default_row_of(config) is HIGHRES_DEFAULT_BY_DX[0]
    assert config.echo()["bare_earth_domains"] == [list(row)]
    mode, coverage = _resolve_plan(config, NYC, 2)
    assert mode == "terrain" and coverage.source_id == "usgs-3dep-13as"
    with pytest.raises(HighresRefusal, match="surface-model-under-buildings"):
        _resolve_plan(config, PARIS, 2)
    # The parent has no buildings: the default, wherever it is.
    assert _resolve_plan(config, PARIS, 1)[1].source_id == "copernicus-dem-glo30"
    assert _resolve_plan(config, NYC, 1)[1].source_id == "copernicus-dem-glo30"


def test_built_runs_carry_the_reason_when_the_raw_table_does_not():
    # The case-bytes door passes the domain tables without [shared]; the
    # offline child passes only smoothing keys.  The built runs decide.
    runs = [SimpleNamespace(grid_id=1, obstacle_opt=0, sf_urban_physics=1,
                            km_opt=1),
            SimpleNamespace(grid_id=2, obstacle_opt=0, sf_urban_physics=1,
                            km_opt=3)]
    config = _resolve(_raw(), domain_runs=runs)
    assert [gid for gid, _ in config.bare_earth] == [2]
    child = SimpleNamespace(grid_id=5, obstacle_opt=1, sf_urban_physics=0,
                            use_wudapt_lcz=0, km_opt=3)
    config = resolve_static_highres({}, source="c.toml", base_dir=".",
                                    spacings_m=(25.0,), run_config=child)
    assert [gid for gid, _ in config.bare_earth] == [5]


def test_shared_keys_reach_every_domain_and_a_bad_grid_id_still_counts():
    assert [g for g, _ in bare_earth_domains(_raw({"obstacle_opt": 1}))] == [1, 2]
    raw = _raw(obstacle_opt=1)
    raw["domain"][1]["grid_id"] = "two"
    assert [g for g, _ in bare_earth_domains(raw)] == [2]


def test_a_named_surface_model_is_kept_with_one_warning(capsys):
    raw = _raw(obstacle_opt=1)
    raw["static"] = {"highres": {"enabled": True, "cache_root": "/tmp/x",
                                 "terrain_source": "copernicus-dem-glo30",
                                 "fields": "terrain"}}
    config = _resolve(raw)
    production._SURFACE_MODEL_WARNED.clear()
    for _ in range(3):
        assert _resolve_plan(config, PARIS, 2)[1].source_id == "copernicus-dem-glo30"
    assert capsys.readouterr().err.count("count twice") == 1
    _resolve_plan(config, PARIS, 1)
    assert "count twice" not in capsys.readouterr().err


def test_a_gem15_fallback_to_glo30_is_refused_under_buildings(monkeypatch):
    from woof.static import gem15
    monkeypatch.setattr(gem15, "fallback_reason", lambda bbox, source: "excluded")
    coverage = terrain_source_coverage("gem15-dtm")
    with pytest.raises(HighresRefusal, match="surface-model-under-buildings"):
        _experimental_terrain(coverage, PARIS, None, 2,
                              bare_earth_reason="it resolves buildings")
    fallback, selection = _experimental_terrain(coverage, PARIS, None, 2)
    assert fallback.source_id == "copernicus-dem-glo30"


def test_the_bare_earth_rows_survive_the_sealed_echo():
    config = _resolve(_raw(obstacle_opt=1))
    back = parse_sealed_static_highres(static_highres_identity(config), source="seal",
                                       base_dir=".")
    assert back.bare_earth == config.bare_earth
    assert back.echo() == config.echo()
    plain = _resolve(_raw())
    assert parse_sealed_static_highres(static_highres_identity(plain), source="seal",
                                       base_dir=".").echo() == plain.echo()
    assert replace(plain, bare_earth=config.bare_earth) != plain


def test_only_surface_models_carry_the_flag():
    flagged = {name for name in ("usgs-3dep-13as", "copernicus-dem-glo30",
                                 "srtm-gl1", "gem15-dtm", "gem15-dsm")
               if terrain_source_coverage(name).surface_model}
    assert flagged == {"copernicus-dem-glo30", "srtm-gl1", "gem15-dsm"}
