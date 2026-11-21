"""``woof go --lakes``: one opt-in for the lake model and its initial state."""
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from woof.lake_launch import MODES, lake_settings, prepare_launch


def test_clm_binds_every_domain_and_keeps_the_original(tmp_path):
    original = tmp_path / "forecast.toml"
    original.write_text('[experiment]\nname="demo"\n[shared]\nlake_state_path="x.nc"\nmp_physics=8\nsf_surface_physics=2\n'
                        '[[domain]]\ngrid_id=1\n[[domain]]\ngrid_id=2\nsf_lake_physics=0\nuse_lakedepth=0\n'
                        '[fetch]\nout="relative/forcing"\n')
    before = original.read_bytes()
    args = SimpleNamespace(config=original, lakes="clm")
    prepare_launch(args)
    raw = tomllib.loads(Path(args.config).read_text())
    assert original.read_bytes() == before and Path(args.config).parent == original.parent
    assert Path(args.config).name.startswith("forecast.lakes-clm-")
    assert raw["fetch"]["out"] == "relative/forcing" and raw["shared"]["mp_physics"] == 8
    assert {k: raw["shared"][k] for k in MODES["clm"]} == MODES["clm"]
    assert raw["shared"]["lake_state_source"] == "auto+era5" and "lake_state_path" not in raw["shared"]
    assert all("sf_lake_physics" not in d for d in raw["domain"])
    assert raw["domain"][1]["use_lakedepth"] == 0
    # The same request names the same file; a second launch reuses it.
    again = SimpleNamespace(config=original, lakes="clm")
    prepare_launch(again)
    assert again.config == args.config


@pytest.mark.parametrize("mode", ["ic", "off"])
def test_ic_and_off_leave_the_lake_model_off(mode):
    raw = lake_settings({"shared": {"sf_lake_physics": 1}}, mode)
    assert raw["shared"]["sf_lake_physics"] == 0
    assert raw["shared"]["lake_state_source"] == ("auto" if mode == "off" else "auto+era5")


def test_absent_switch_reads_nothing_and_hex_is_refused_by_name():
    args = SimpleNamespace(config=Path("absent.toml"))
    prepare_launch(args)
    assert args.config == Path("absent.toml")
    with pytest.raises(ValueError, match="engine = 'hex'"):
        lake_settings({"model": {"engine": "hex"}}, "clm")


def test_every_mode_is_a_valid_run_configuration():
    from woof.config import validate_run_config
    from test_ruc_mosaic_options import _config
    for mode, settings in MODES.items():
        validate_run_config(_config(**settings))


def test_clm_refuses_inactive_land_surface_before_writing(tmp_path):
    cfg = tmp_path / "f.toml"
    cfg.write_text('[experiment]\nname="d"\n[shared]\nsf_surface_physics=0\n[[domain]]\ngrid_id=1\n')
    with pytest.raises(ValueError, match="--lakes clm"):
        prepare_launch(SimpleNamespace(config=cfg, lakes="clm"))
    assert list(tmp_path.iterdir()) == [cfg]


def test_bathymetry_tuning_survives():
    raw = lake_settings({"shared": {"sf_surface_physics": 2, "lake_min_elev": 7},
                         "domain": [{"lakedepth_default": 20, "use_lakedepth": 0}]}, "clm")
    assert raw["domain"][0] == {"lakedepth_default": 20, "use_lakedepth": 0}
    assert raw["shared"]["lake_min_elev"] == 7


def test_invalid_engine_is_named_by_the_config_validator():
    with pytest.raises(ValueError, match="bogus"):
        lake_settings({"model": {"engine": "bogus"}}, "off")


@pytest.mark.parametrize("route", ["dir", "namelist", "wrfinput"])
def test_lakes_is_refused_on_directory_routes(tmp_path, route):
    """Review (#587): a directory route carries its own namelist; --lakes
    there would be ignored, so it is refused before anything launches."""
    from woof.go_cli import GoRefusal, _refuse_lakes_off_configuration
    config = {"dir": tmp_path, "namelist": tmp_path / "namelist.input",
              "wrfinput": tmp_path / "forecast.toml"}[route]
    args = SimpleNamespace(config=config, lakes="clm",
                           wrfinput=tmp_path if route == "wrfinput" else None)
    with pytest.raises(GoRefusal, match="drop --lakes"):
        _refuse_lakes_off_configuration(args)
    _refuse_lakes_off_configuration(SimpleNamespace(config=config, lakes=None, wrfinput=None))
    _refuse_lakes_off_configuration(SimpleNamespace(config=tmp_path / "f.toml", lakes="clm", wrfinput=None))


def test_derived_configuration_keeps_the_umask_mode(tmp_path):
    import os
    original = tmp_path / "forecast.toml"
    original.write_text('[experiment]\nname="demo"\n[shared]\nsf_surface_physics=2\n[[domain]]\ngrid_id=1\n')
    args = SimpleNamespace(config=original, lakes="clm")
    prepare_launch(args)
    umask = os.umask(0)
    os.umask(umask)
    assert Path(args.config).stat().st_mode & 0o777 == 0o666 & ~umask


@pytest.mark.parametrize("use_lakedepth, needed", [(1, True), (0, False)])
def test_go_preflight_asks_for_lake_depth_only_with_use_lakedepth(tmp_path, use_lakedepth, needed):
    """Review (#599): use_lakedepth = 0 runs on lakedepth_default, which the
    builder (woof.static.lake) reads no dataset for, so go must not ask."""
    from woof.geog_assets import geography_for_run
    from woof.toml_document import emit_experiment_toml
    recipe = Path(__file__).resolve().parents[1] / "woof/data/recipes/hrrr_v4_gsd41.toml"
    raw = lake_settings(tomllib.loads(recipe.read_text()), "clm")
    raw.pop("static", None)  # WPS geography, not the native HRRR statics
    raw.setdefault("shared", {})["use_lakedepth"] = use_lakedepth
    for domain in raw.get("domain", ()):
        domain.pop("use_lakedepth", None)
    config = tmp_path / "lakes.toml"
    config.write_bytes(emit_experiment_toml(raw).encode())
    assert ("lake_depth" in geography_for_run(config)["datasets"]) is needed


def _lakes_config(tmp_path, use_lakedepth=1):
    from woof.toml_document import emit_experiment_toml
    recipe = Path(__file__).resolve().parents[1] / "woof/data/recipes/hrrr_v4_gsd41.toml"
    raw = lake_settings(tomllib.loads(recipe.read_text()), "clm")
    raw.pop("static", None)
    raw.setdefault("shared", {})["use_lakedepth"] = use_lakedepth
    for domain in raw.get("domain", ()):
        domain.pop("use_lakedepth", None)
    config = tmp_path / "lakes.toml"
    config.write_bytes(emit_experiment_toml(raw).encode())
    return config


def test_go_stages_lake_depth_itself_when_the_lake_model_needs_it(tmp_path, monkeypatch):
    """`woof go --lakes clm` fetched nothing for lake_depth and refused; go now stages it."""
    from woof import geog_assets, go_cli
    calls = []
    monkeypatch.setattr(geog_assets, "fetch_geog", lambda **kw: calls.append(kw) or 1)
    root = tmp_path / "WPS_GEOG"
    assert go_cli.stage_lake_depth(root, _lakes_config(tmp_path)) is True
    assert [(c["root"], c["datasets"], c["source"]) for c in calls] == [(root, ("lake_depth",), "ncar")]
    # Only lake_depth: every other missing dataset still refuses with its remedy.
    assert "lake_depth/index" in go_cli.geography_refusal(root, config=_lakes_config(tmp_path))
    flagged = go_cli.go_stages_lake_depth_refusal(root, config=_lakes_config(tmp_path))
    assert flagged is not None and "lake_depth/index" not in flagged


def test_go_stages_nothing_when_lake_depth_is_present_or_not_read(tmp_path, monkeypatch):
    from woof import geog_assets, go_cli
    calls = []
    monkeypatch.setattr(geog_assets, "fetch_geog", lambda **kw: calls.append(kw) or 1)
    root = tmp_path / "WPS_GEOG"
    assert go_cli.stage_lake_depth(root, _lakes_config(tmp_path, use_lakedepth=0)) is False
    assert go_cli.stage_lake_depth(root, None) is False
    (root / "lake_depth").mkdir(parents=True)
    (root / "lake_depth" / "index").write_text("x")
    assert go_cli.stage_lake_depth(root, _lakes_config(tmp_path)) is False
    assert calls == []


def test_a_failed_lake_depth_fetch_refuses_with_both_remedies(tmp_path, monkeypatch):
    from woof import geog_assets, go_cli

    def fail(**kw):
        raise geog_assets.GeogFetchError("no route to host")

    monkeypatch.setattr(geog_assets, "fetch_geog", fail)
    with pytest.raises(go_cli.GoRefusal, match="no route to host") as refused:
        go_cli.stage_lake_depth(tmp_path / "WPS_GEOG", _lakes_config(tmp_path))
    assert "fetch-geog --datasets lake_depth" in str(refused.value) and "use_lakedepth = 0" in str(refused.value)
