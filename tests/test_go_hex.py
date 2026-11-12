"""``woof go`` on the hex model: ``[model] engine = "hex"`` (woof.go_hex).

Host-only: the engine switch, the stage map and the regional command a go TOML
makes (a dry-run plan golden), and the refusals that name what hex cannot run
yet.  The end-to-end run is tests/test_go_hex_smoke.py (a declared must-run gate).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from woof import cli
from woof import go_hex
from woof.config import MODEL_ENGINES, model_engine

REPO = Path(__file__).resolve().parents[1]

HEX_TOML = """\
[model]
engine = "hex"

[experiment]
name = "hexgo"
start_time = 2025-05-30T12:00:00
run_seconds = 7200.0

[projection]
map_proj = "lambert"
ref_lat = 39.5
ref_lon = -98.5
truelat1 = 30.0
truelat2 = 60.0
stand_lon = -98.5

[[domain]]
nx = 40
ny = 30
dx = 12000.0
history_interval_s = 1800.0
{domain_extra}
{shared}
[fetch]
source = "gfs"
cycle = "2025-05-30T12"
{tail}"""


def _write(tmp_path: Path, *, domain_extra: str = "", shared: str = "", tail: str = "",
           text: str | None = None) -> Path:
    path = tmp_path / "hexgo.toml"
    path.write_text(text if text is not None else
                    HEX_TOML.format(domain_extra=domain_extra, shared=shared, tail=tail))
    return path


def _args(*argv: str):
    return cli.build_parser().parse_args(["go", *argv])


# ---------------------------------------------------------------------------
# the engine key
# ---------------------------------------------------------------------------
def test_model_engine_defaults_to_wrf_and_reads_hex():
    assert MODEL_ENGINES == ("wrf", "hex")
    assert model_engine(None) == "wrf"
    assert model_engine({}) == "wrf"
    assert model_engine({"engine": "wrf"}) == "wrf"
    assert model_engine({"engine": "hex"}) == "hex"


@pytest.mark.parametrize("table,words", [
    ({"engine": "mpas"}, "must be one of"),
    ({"engine": 1}, "must be a string"),
    ({"engin": "hex"}, "has no key 'engin'"),
    ("hex", "must be a table"),
])
def test_model_engine_refuses_a_typo_by_name(table, words):
    with pytest.raises(ValueError, match=words):
        model_engine(table)


def test_every_shipped_config_stays_on_the_wrf_engine():
    """The default engine is WRF: no shipped TOML reaches the hex route."""

    configs = sorted((REPO / "configs").rglob("*.toml"))
    assert configs
    routed = [str(path) for path in configs if go_hex.configured_engine(path) != "wrf"]
    assert routed == []


def test_an_unreadable_config_is_left_to_the_wrf_path(tmp_path):
    broken = tmp_path / "broken.toml"
    broken.write_text("[experiment\n")
    assert go_hex.configured_engine(broken) == "wrf"
    assert go_hex.configured_engine(tmp_path) == "wrf"
    assert go_hex.configured_engine(None) == "wrf"
    assert go_hex.configured_engine(tmp_path / "missing.toml") == "wrf"


def test_a_wrf_experiment_loads_the_same_with_model_engine_wrf(tmp_path):
    """``[model] engine = "wrf"`` is the default spelled out: the WRF loader builds the same experiment."""

    from woof.experiment import load_experiment

    source = (REPO / "configs" / "conus3km.toml").read_text()
    plain = tmp_path / "plain.toml"
    plain.write_text(source)
    named = tmp_path / "named.toml"
    named.write_text('[model]\nengine = "wrf"\n\n' + source)
    assert repr(load_experiment(named)) == repr(load_experiment(plain))
    hexed = tmp_path / "hexed.toml"
    hexed.write_text('[model]\nengine = "hex"\n\n' + source)
    with pytest.raises(ValueError, match="woof go CONFIG"):
        load_experiment(hexed)


def test_go_main_dispatches_hex_and_only_hex(tmp_path, monkeypatch):
    from woof import go_cli

    seen = []
    monkeypatch.setattr(go_hex, "launch", lambda args, observer=None: seen.append(args.config) or 0)
    config = _write(tmp_path)
    assert go_cli.go_main(_args(str(config))) == 0
    assert seen == [config]
    with pytest.raises(go_cli.GoRefusal, match="must be one of"):
        go_cli.go_main(_args(str(_write(tmp_path, text=HEX_TOML.replace('"hex"', '"mpas"', 1)
                                        .format(domain_extra="", shared="", tail="")))))


# ---------------------------------------------------------------------------
# the stage map and the plan
# ---------------------------------------------------------------------------
def test_stage_map_covers_every_regional_stage_once():
    from woof.hex.regional_door import STAGES

    mapped = [stage for stages in go_hex.STAGE_MAP.values() for stage in stages]
    assert tuple(go_hex.STAGE_MAP) == ("fetch", "prepare", "forecast", "render")
    assert sorted(mapped) == sorted(STAGES)


def test_dry_run_plan_golden(tmp_path):
    config = _write(tmp_path)
    value = go_hex.plan(_args(str(config), "--dry-run"))
    work = tmp_path / "hexgo-go"
    assert value["engine"] == "hex"
    assert value["stages"] == {"fetch": ["intermediate"], "prepare": ["mesh", "init", "lbc", "contract"],
                               "forecast": ["forecast"], "render": ["products"]}
    assert value["regional_argv"] == [
        "regional", "--shape", str(work / "go-region.json"), "--cycle", "2025-05-30T12",
        "--source", "gfs", "--hours", "2", "--history-every-minutes", "30",
        "--fine-dx-m", "12000.0", "--background-km", "96.0", "--work-dir", str(work)]
    assert value["command"].startswith("woof hex regional --shape ")
    assert json.loads(json.dumps(value)) == value


def test_the_region_is_the_domain_outline(tmp_path):
    polygon = go_hex.region_polygon({"map_proj": "lambert", "ref_lat": 39.5, "ref_lon": -98.5,
                                     "truelat1": 30.0, "truelat2": 60.0, "stand_lon": -98.5},
                                    40, 30, 12000.0)
    assert polygon["kind"] == "polygon"
    lats = [vertex[0] for vertex in polygon["vertices_deg"]]
    lons = [vertex[1] for vertex in polygon["vertices_deg"]]
    # 30 x 12 km north-south (about 3.2 degrees), 40 x 12 km east-west at 39.5 N (about 5.6 degrees).
    assert 3.0 < max(lats) - min(lats) < 3.5
    assert 5.3 < max(lons) - min(lons) < 6.0
    assert abs((max(lats) + min(lats)) / 2 - 39.5) < 0.2
    assert abs((max(lons) + min(lons)) / 2 + 98.5) < 1e-6
    from woof.hex import mesh_shape
    assert mesh_shape.shape_from_document(polygon)["kind"] == "polygon"


@pytest.mark.parametrize("dx,background", [(12000.0, 96.0), (3000.0, 96.0), (937.5, 120.0),
                                           (15000.0, 120.0), (50000.0, 100.0)])
def test_background_ladder_holds_the_spacing(dx, background):
    assert go_hex.background_km(dx) == background


def test_physics_selectors_go_to_hex_resolve(tmp_path):
    config = _write(tmp_path, domain_extra="cu_physics = 0", shared="[shared]\nmp_physics = 8\n")
    argv = go_hex.plan(_args(str(config)))["regional_argv"]
    assert "--forecast-arg=--physics=cu_physics=0" in argv
    assert "--forecast-arg=--physics=mp_physics=8" in argv


def test_go_flags_the_route_reads(tmp_path):
    config = _write(tmp_path, tail="forecast_start_hour = 0\n")
    out = tmp_path / "elsewhere"
    value = go_hex.plan(_args(str(config), "--outdir", str(out), "--products", "none",
                              "--cycle", "2025-05-31T00", "--card", "16gb"))
    argv = value["regional_argv"]
    assert argv[argv.index("--work-dir") + 1] == str(out)
    assert argv[argv.index("--cycle") + 1] == "2025-05-31T00"
    assert "--no-products" in argv and value["stages"]["render"] == []
    assert argv[argv.index("--card") + 1] == "16gb"
    prepared = go_hex.plan(_args(str(config), "--prepare-only"))
    assert prepared["stages"]["forecast"] == [] and prepared["stages"]["render"] == []
    assert prepared["regional_argv"][-4:] == ["--skip", "forecast", "--skip", "products"]


def test_output_format_table_rides_to_the_forecast_door(tmp_path):
    config = _write(tmp_path, tail='\n[output]\nformats = "netcdf"\n')
    argv = go_hex.plan(_args(str(config)))["regional_argv"]
    assert argv[-2:] == ["--output-config", str(config)]


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("edit,words", [
    (dict(tail="\n[[domain]]\ngrid_id = 2\nparent_id = 1\nnx = 10\nny = 10\ndx = 4000.0\n"
               "history_interval_s = 3600.0\n"), "nests are N/A on the mesh"),
    (dict(shared="[shared]\nchem_opt = 300\n"), "P4"),
    (dict(shared='[shared]\nchem_sets = ["gocart_simple"]\n'), "6.2"),
    (dict(shared="[shared]\neta_levels = [1.0, 0.5, 0.0]\n"), "vertical_spec"),
    (dict(domain_extra="time_step = 60"), "4.1"),
    (dict(domain_extra="partial_nest = 1"), "5.6"),
    (dict(shared="[shared]\nspp_pbl = 1\n"), "5.4"),
    (dict(shared="[shared]\nkm_opt = 4\n"), r"\[shared\] km_opt: it has no hex mapping"),
    (dict(tail="\n[ensemble]\nmembers = 3\n"), "3.3"),
    (dict(tail="\n[nudging]\ngrid = true\n"), "5.1"),
    (dict(tail="\n[wrf_hydro]\nenabled = true\n"), "P7"),
    (dict(tail="\n[simulated_radar]\nenabled = true\n"), "2.7"),
    (dict(tail="\n[tiles]\nmode = \"auto\"\n"), "N/A on the mesh"),
    (dict(tail='\n[output]\nfields = ["T2"]\n'), r"\[output\] fields"),
    (dict(domain_extra="parent_grid_ratio = 3"), "one root domain"),
])
def test_unreadable_keys_and_tables_are_refused_with_their_matrix_item(tmp_path, edit, words):
    with pytest.raises(go_hex.HexGoRefusal, match=words):
        go_hex.plan(_args(str(_write(tmp_path, **edit))))


def test_every_refusal_is_named_at_once(tmp_path):
    config = _write(tmp_path, domain_extra="time_step = 60", shared="[shared]\nnz = 50\nchem_opt = 300\n")
    with pytest.raises(go_hex.HexGoRefusal) as error:
        go_hex.plan(_args(str(config)))
    text = str(error.value)
    for key in ("[domain] time_step", "[shared] nz", "[shared] chem_opt"):
        assert key in text


@pytest.mark.parametrize("argv,words", [
    (("--restart", "x"), "3.2"),
    (("--members", "3"), "3.3"),
    (("--storm-nests", "2"), "A.9"),
    (("--nudging", "grid"), "5.1"),
    (("--grib2",), "2.8"),
    (("--products", "maps"), "other than 'none'"),
    (("--tiles", "auto"), "A.17"),
])
def test_go_flags_without_a_hex_route_are_refused_by_name(tmp_path, argv, words):
    config = _write(tmp_path)
    with pytest.raises(go_hex.HexGoRefusal, match=words):
        go_hex.plan(_args(str(config), *argv))


def test_woof_ensemble_and_the_run_plan_stream_are_refused(tmp_path):
    config = _write(tmp_path)
    args = cli.build_parser().parse_args(["ensemble", str(config)])
    with pytest.raises(go_hex.HexGoRefusal, match="3.3"):
        go_hex.plan(args)
    with pytest.raises(go_hex.HexGoRefusal, match="run-plan event stream"):
        go_hex.launch(_args(str(config)), observer=lambda *a, **k: None)


@pytest.mark.parametrize("edit,words", [
    (("run_seconds = 7200.0", "run_seconds = 5400.0"), "whole hours"),
    (("history_interval_s = 1800.0", "history_interval_s = 90.5"), "whole minutes"),
    (('map_proj = "lambert"', 'map_proj = "lat-lon"'), "woof hex global"),
    (("start_time = 2025-05-30T12:00:00", "start_time = 2025-05-30T15:00:00"), "forecast_start_hour"),
])
def test_run_shape_refusals(tmp_path, edit, words):
    text = HEX_TOML.format(domain_extra="", shared="", tail="").replace(*edit)
    with pytest.raises(go_hex.HexGoRefusal, match=words):
        go_hex.plan(_args(str(_write(tmp_path, text=text))))


# ---------------------------------------------------------------------------
# review fixes (#316)
# ---------------------------------------------------------------------------
def test_an_unknown_projection_key_is_refused(tmp_path):
    text = HEX_TOML.format(domain_extra="", shared="", tail="").replace(
        'stand_lon = -98.5', 'stand_lon = -98.5\npole_lat = 90.0')
    with pytest.raises(go_hex.HexGoRefusal, match=r"\[projection\] pole_lat"):
        go_hex.plan(_args(str(_write(tmp_path, text=text))))


@pytest.mark.parametrize("start", ['"2025-05-30T12:00:00"', "2025-05-30T12:00:00Z", "2025-05-30"])
def test_start_time_must_be_an_offset_free_datetime(tmp_path, start):
    text = HEX_TOML.format(domain_extra="", shared="", tail="").replace(
        "start_time = 2025-05-30T12:00:00", f"start_time = {start}")
    with pytest.raises(go_hex.HexGoRefusal, match="offset-free"):
        go_hex.plan(_args(str(_write(tmp_path, text=text))))


@pytest.mark.parametrize("lead", ["true", "-1", "1.5"])
def test_forecast_start_hour_is_checked_under_cycle_too(tmp_path, lead):
    config = _write(tmp_path, tail=f"forecast_start_hour = {lead}\n")
    with pytest.raises(go_hex.HexGoRefusal, match="nonnegative integer"):
        go_hex.plan(_args(str(config), "--cycle", "2025-05-31T00"))


def test_frame_deletion_is_refused_while_products_run(tmp_path):
    config = _write(tmp_path, tail='\n[output]\nformats = "netcdf,icechunk"\nicechunk_delete_after_commit = true\n')
    with pytest.raises(go_hex.HexGoRefusal, match="--products none"):
        go_hex.plan(_args(str(config)))
    assert "--no-products" in go_hex.plan(_args(str(config), "--products", "none"))["regional_argv"]
    plain = _write(tmp_path)
    with pytest.raises(go_hex.HexGoRefusal, match="--products none"):
        go_hex.plan(_args(str(plain), "--icechunk-delete-after-commit"))
