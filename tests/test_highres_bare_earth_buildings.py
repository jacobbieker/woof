"""Building runs take bare-earth terrain, not a surface model.

The grid-spacing default (HIGHRES_DEFAULT_BY_DX) names Copernicus GLO-30,
a radar surface model whose heights over a city include the buildings.  A
domain that resolves buildings (obstacle_opt) or runs an urban canopy
under LES on top of it counts them twice (phase-1 Manhattan LES,
2026-11-17: "auto" took GLO-30 on the HRRR route).  Such a configuration
takes "auto", which is bare-earth USGS 3DEP inside its envelope, and
refuses by name outside it; a surface model named explicitly is kept with
a warning.  Every other configuration resolves exactly as before.
"""
from __future__ import annotations

import pytest

from woof.static.highres_fetch import FootprintBBox
from woof.static.highres_production import (
    HIGHRES_DEFAULT_BY_DX, HighresRefusal, _resolve_plan, bare_earth_reason,
    default_row_of, resolve_static_highres)

NYC = FootprintBBox(lat_min=40.5, lat_max=40.9, lon_min=-74.1, lon_max=-73.8)
PARIS = FootprintBBox(lat_min=48.7, lat_max=49.0, lon_min=2.2, lon_max=2.5)


def _raw(**d2):
    return {"domain": [{"grid_id": 1, "parent_id": 0, "dx": 250.0},
                       {"grid_id": 2, "parent_id": 1, "parent_grid_ratio": 5,
                        **d2}]}


def test_no_buildings_keeps_the_default_row_exactly():
    for raw in (_raw(), _raw(obstacle_opt=0), _raw(sf_urban_physics=1),
                _raw(km_opt=3), _raw(sf_urban_physics=1, km_opt=4)):
        assert bare_earth_reason(raw) == ""
        config = resolve_static_highres(raw, source="c.toml", base_dir=".")
        assert default_row_of(config) is HIGHRES_DEFAULT_BY_DX[0]
        assert config.terrain_source == "copernicus-dem-glo30"
        assert config.bare_earth_reason == ""


@pytest.mark.parametrize("keys", [
    {"obstacle_opt": 1},
    {"sf_urban_physics": 1, "km_opt": 3},
    {"sf_urban_physics": 2, "km_opt": 2},
])
def test_buildings_turn_the_default_into_auto(keys):
    raw = _raw(**keys)
    assert bare_earth_reason(raw).startswith("d02 ")
    config = resolve_static_highres(raw, source="c.toml", base_dir=".")
    assert config.terrain_source == "auto"
    assert config.bare_earth_reason
    mode, coverage = _resolve_plan(config, NYC)
    assert mode == "terrain" and coverage.source_id == "usgs-3dep-13as"
    with pytest.raises(HighresRefusal, match="surface-model-under-buildings"):
        _resolve_plan(config, PARIS)


def test_shared_keys_reach_every_domain():
    raw = _raw()
    raw["shared"] = {"obstacle_opt": 1}
    assert bare_earth_reason(raw).startswith("d01 ")
    raw["domain"][0]["obstacle_opt"] = 0
    assert bare_earth_reason(raw).startswith("d02 ")


def test_a_named_surface_model_is_kept_with_a_warning(capsys):
    raw = _raw(obstacle_opt=1)
    raw["static"] = {"highres": {"enabled": True, "cache_root": "/tmp/x",
                                 "terrain_source": "copernicus-dem-glo30",
                                 "fields": "terrain"}}
    config = resolve_static_highres(raw, source="c.toml", base_dir=".")
    assert config.terrain_source == "copernicus-dem-glo30"
    _, coverage = _resolve_plan(config, PARIS)
    assert coverage.source_id == "copernicus-dem-glo30"
    assert "count twice" in capsys.readouterr().err


def test_the_reason_is_not_an_operand_of_the_seal():
    plain = resolve_static_highres(_raw(), source="c.toml", base_dir=".")
    from dataclasses import replace
    assert replace(plain, bare_earth_reason="x") == plain
    assert "bare_earth_reason" not in plain.echo()
