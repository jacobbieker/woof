"""``woof energy plan --topology hex-swath``: the corridor mesh planner."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pytest

from woof.energy import plan_hex
from woof.energy.contracts import Site, SiteSet, load_plan, load_sites

FIXTURES = Path(__file__).parent / "fixtures" / "energy"
SITES = FIXTURES / "sites_wales.json"


@pytest.fixture(autouse=True)
def _no_generator(monkeypatch):
    """Plans here never depend on a staged ``rw_mpas_mesh``."""

    monkeypatch.setattr(plan_hex, "_generator_sizing", lambda spec: None)


def _plan(tmp_path, **overrides):
    kwargs = dict(outdir=tmp_path / "plan", dx_m=937.5, start="2026-10-10T00")
    kwargs.update(overrides)
    return plan_hex.build_plan(load_sites(SITES), **kwargs)


def _args(tmp_path, **overrides):
    values = dict(sites=str(SITES), topology="hex-swath", dx_m=937.5,
                  corridor_km=2.0, parent_dx_m=None, start="2026-10-10T00",
                  hours=24.0, source=None, card=None, vram_gib=None,
                  max_domains=None, nz=None, outdir=str(tmp_path / "plan"))
    values.update(overrides)
    return argparse.Namespace(**values)


def test_fixture_plan_documents(tmp_path):
    plan = _plan(tmp_path)
    outdir = tmp_path / "plan"
    loaded = load_plan(outdir / "plan.json")
    assert loaded.topology == "hex-swath" and len(loaded.domains) == 1
    domain = loaded.domains[0]
    assert domain.role == "mesh" and domain.config is None
    assert domain.grid_id is None and domain.dx_m == 937.5
    assert domain.run_dir == "runs/hex"
    assert domain.output_glob == "forecast/cuda-history.*.nc"
    sites = load_sites(SITES)
    assert set(domain.site_ids) == {s.site_id for s in sites.sites}
    for key in ("mesh_spec", "cull_region", "mesh_path", "grid", "static"):
        assert key in domain.mesh
    assert domain.mesh["mesh_path"].endswith("corridor.init.nc")
    assert (outdir / domain.mesh["mesh_spec"]).is_file()
    assert (outdir / domain.mesh["cull_region"]).is_file()
    assert (outdir / "runs" / "hex").is_dir()
    assert domain.extra["runnable"] == "mesh-and-cull"
    assert {row["stage"] for row in domain.extra["blocked_stages"]} == {
        "vertical", "register", "met", "forecast"}
    assert plan.start == "2026-10-10T00"

    spec = json.loads((outdir / "mesh_spec.json").read_text())
    assert spec["background_km"] == 120.0
    spacings = sorted({r["spacing_km"] for r in spec["regions"]})
    assert spacings == [0.9375, 1.875, 3.75, 7.5, 15.0, 30.0, 60.0]
    fine = [r for r in spec["regions"] if r["spacing_km"] == 0.9375]
    assert any(r["shape"]["kind"] == "polygon" for r in fine)
    for region in spec["regions"]:
        assert region["transition_km"] > 0.0
        if region["shape"]["kind"] == "polygon":
            ring = region["shape"]["vertices_deg"]
            assert len(ring) >= 4 and ring[0] != ring[-1]

    cull = json.loads((outdir / "cull_region.json").read_text())
    assert cull["kind"] == "polygon"
    ring = [tuple(v) for v in cull["vertices_deg"]]
    footprint = domain.footprint
    assert footprint[0] == footprint[-1]
    assert [(lat, lon) for lon, lat in footprint[:-1]] == ring
    # every site lies inside the cut
    from woof.hex.swath.geometry import ring_containment

    contains = ring_containment(ring)
    assert all(contains((s.lat, s.lon)) for s in sites.sites)


def test_commands_parse_with_the_real_parsers(tmp_path):
    from woof.hex.cli import build_parser as hex_parser
    from woof.mpas_mesh import build_parser as mesh_parser

    domain = _plan(tmp_path).domains[0]
    commands = domain.extra["commands"]
    assert [c[:2] for c in commands] == [
        ["hex", "mesh-plan"], ["mesh", "--spec"], ["hex", "mesh-check"],
        ["hex", "cull"], ["hex", "mesh-check"]]
    for argv in commands:
        if argv[0] == "hex":
            parsed = hex_parser().parse_args(argv[1:])
            assert callable(parsed.handler)
        else:
            parsed = mesh_parser().parse_args(argv[1:])
            assert parsed.spec == Path("mesh_spec.json")
            assert parsed.cells > 0
    cull = hex_parser().parse_args(commands[3][1:])
    assert cull.region == Path("cull_region.json")
    assert cull.out_dir == Path(domain.run_dir)
    for argv in commands:
        for token in argv:
            assert not Path(token).is_absolute()


def test_ladder_clears_the_smoothness_bound(tmp_path):
    domain = _plan(tmp_path).domains[0]
    gates = domain.extra["gates"]
    bound = min(gates["woof_mesh_refuse_above_percent_per_cell"],
                gates["transition_band_ceiling_percent_per_cell"])
    assert gates["steepest_gradient_percent_per_cell_estimate"] <= \
        plan_hex.GRADIENT_MARGIN * bound
    # the registered 18x ramp alone would not
    rungs = plan_hex.ladder([0.9375 * 2 ** k for k in range(7)], 2.0)
    assert plan_hex.steepest_gradient_percent(rungs, 120.0) > bound
    assert domain.extra["transition_factor"] > plan_hex.MIN_TRANSITION_FACTOR


def test_spacing_is_near_dx_within_the_corridor(tmp_path):
    domain = _plan(tmp_path).domains[0]
    rungs = [plan_hex.Rung(r["spacing_km"], r["transition_km"], r["reach_km"])
             for r in domain.extra["rungs"]]
    h = plan_hex.spacing_at_distance(np.array([0.0, 1.0, 2.0]), rungs, 120.0)
    assert np.all(h <= 0.9375 * 1.06)
    assert plan_hex.spacing_at_distance(np.array([1e5]), rungs, 120.0)[0] == \
        pytest.approx(120.0, rel=1e-3)


def test_dx_below_the_hex_floor_is_refused(tmp_path):
    for dx in (100.0, 50.0):
        with pytest.raises(plan_hex.HexPlanRefusal) as caught:
            _plan(tmp_path, dx_m=dx)
        message = str(caught.value)
        assert "wrf-tiles" in message and "819 m" in message
        assert "anchored" in message
    assert not (tmp_path / "plan" / "plan.json").exists()
    floor = plan_hex.spacing_floor_m(100.0)
    assert floor["smallest_anchored_dt_s"] == 5.0
    assert floor["min_dc_edge_m"] == pytest.approx(5.0 * 125.0 / 0.9)


def test_capacity_refusal(tmp_path):
    with pytest.raises(plan_hex.HexPlanRefusal, match="cells") as caught:
        _plan(tmp_path, vram_gib=2.0)
    assert "wrf-tiles" in str(caught.value)
    verdict = plan_hex.capacity_verdict(1000.0, None, 64.0)
    assert verdict["fits"]
    assert any("most demanding" in note for note in verdict["notes"])


def test_card_names(tmp_path):
    domain = _plan(tmp_path, card="rtx-5090").domains[0]
    assert domain.extra["capacity"]["row"] == "limited-area/170sm"
    assert domain.extra["capacity"]["budget_basis"].startswith("--card")
    with pytest.raises(plan_hex.HexPlanRefusal, match="not a card"):
        _plan(tmp_path, card="voodoo-2")


def test_parent_dx_off_the_ladder_is_refused(tmp_path):
    with pytest.raises(plan_hex.HexPlanRefusal, match="halving ladder"):
        _plan(tmp_path, parent_dx_m=100_000.0)
    domain = _plan(tmp_path, parent_dx_m=60_000.0).domains[0]
    assert domain.mesh["background_km"] == 60.0


def test_other_controls_are_refused(tmp_path):
    with pytest.raises(plan_hex.HexPlanRefusal, match="--nz"):
        _plan(tmp_path, nz=80)
    with pytest.raises(plan_hex.HexPlanRefusal, match="YYYY-MM-DDTHH"):
        _plan(tmp_path, start="2026-10-10 00:00")
    with pytest.raises(plan_hex.HexPlanRefusal, match="no sites"):
        plan_hex.build_plan(SiteSet(sites=[], heights_m=(10.0,)),
                            outdir=tmp_path / "plan", dx_m=937.5)
    assert _plan(tmp_path, nz=55).domains


def test_generator_refusal_refuses_the_plan(tmp_path, monkeypatch):
    def refuse(spec):
        raise plan_hex.HexPlanRefusal("rw_mpas_mesh refuses the corridor "
                                      "spec: transition band")

    monkeypatch.setattr(plan_hex, "_generator_sizing", refuse)
    with pytest.raises(plan_hex.HexPlanRefusal, match="transition band"):
        _plan(tmp_path)

    monkeypatch.setattr(plan_hex, "_generator_sizing", lambda spec: {
        "predicted_cells": 12345.0,
        "gates_applied_by_hexcore": {"transition_band": {
            "steepest_gradient_percent_per_cell": 9.0}}})
    with pytest.raises(plan_hex.HexPlanRefusal, match="woof mesh refuses"):
        _plan(tmp_path)

    monkeypatch.setattr(plan_hex, "_generator_sizing", lambda spec: {
        "predicted_cells": 12345.0,
        "gates_applied_by_hexcore": {"transition_band": {
            "steepest_gradient_percent_per_cell": 2.5}}})
    domain = _plan(tmp_path).domains[0]
    assert domain.extra["commands"][1][3:5] == ["--cells", "12345"]


def test_long_corridor_stays_corridor_shaped(tmp_path):
    lat = 52.0 + 0.0 * np.arange(60)
    lon = -4.0 + 0.05 * np.arange(60)      # about 200 km of line
    sites = SiteSet(sites=[Site(site_id=f"s{i}", asset_id="line/9",
                                kind="line_sample", lat=float(a),
                                lon=float(b), chainage_m=3400.0 * i)
                           for i, (a, b) in enumerate(zip(lat, lon))],
                    heights_m=(10.0, 30.0))
    plan = plan_hex.build_plan(sites, outdir=tmp_path / "plan", dx_m=1000.0)
    domain = plan.domains[0]
    assert domain.mesh["background_km"] == 128.0
    spec = json.loads((tmp_path / "plan" / "mesh_spec.json").read_text())
    fine = [r for r in spec["regions"] if r["spacing_km"] == 1.0]
    assert fine and all(r["shape"]["kind"] == "polygon" for r in fine)
    estimate = domain.extra["estimate"]
    assert 0 < estimate["fine_cells"] < estimate["cull_cells"]


def test_main_prints_a_summary_and_refuses_cleanly(tmp_path, capsys):
    assert plan_hex.main(_args(tmp_path)) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["topology"] == "hex-swath" and record["domains"] == 1
    plan = load_plan(tmp_path / "plan" / "plan.json")
    assert plan.sites_ref["sha256"]
    assert plan.notes and any("floor" in n for n in plan.notes)

    assert plan_hex.main(_args(tmp_path, dx_m=100.0)) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_cli_dispatches_hex_swath(tmp_path, capsys):
    from woof.cli import build_parser

    args = build_parser().parse_args([
        "energy", "plan", str(SITES), "--topology", "hex-swath",
        "--dx-m", "937.5", "--start", "2026-10-10T00",
        "-o", str(tmp_path / "plan")])
    assert args.func(args) == 0
    assert (tmp_path / "plan" / "plan.json").is_file()


def test_default_start_is_a_six_hour_cycle():
    from datetime import datetime, timezone

    assert plan_hex.default_start(datetime(2026, 10, 10, 17, 42,
                                           tzinfo=timezone.utc)) == \
        "2026-10-10T12"
    assert math.isclose(plan_hex.CULL_PAD_SCALE, 1.35)


def test_rings_cover_their_reach_at_sharp_bends():
    from woof.hex.swath.geometry import destination, ring_containment

    projection = plan_hex._Aeqd(52.0, -4.0)
    lat = np.array([52.0, 52.0, 52.0, 52.3, 52.6])
    lon = np.array([-4.6, -4.3, -4.0, -4.0, -4.0])   # a right angle at -4.0
    x, y = projection.forward(lat, lon)
    chain = plan_hex.Chain("line/1", [f"s{i}" for i in range(5)], lat, lon,
                           x, y)
    reach = 30.0
    rings = plan_hex._ring_regions(chain, reach, 0.2)
    assert len(rings) == 2
    tests = [ring_containment(ring) for ring in rings]
    for bearing in range(0, 360, 15):
        point = destination(52.0, -4.0, bearing, 0.98 * reach)
        assert any(test(point) for test in tests), bearing


def test_clusters_scale_with_length_not_site_count():
    import time

    n = 20000
    lat = 52.0 + 0.0 * np.arange(n)
    lon = -4.0 + 1e-4 * np.arange(n)
    projection = plan_hex._Aeqd(52.0, -3.0)
    x, y = projection.forward(lat, lon)
    chains = [plan_hex.Chain(f"tower/{i}", [f"t{i}"], lat[i:i + 1],
                             lon[i:i + 1], x[i:i + 1], y[i:i + 1])
              for i in range(n)]
    started = time.perf_counter()
    groups = plan_hex._clusters(chains, 200.0)
    assert time.perf_counter() - started < 30.0
    assert len(groups) == 1


def test_main_refuses_hex_engine_errors(tmp_path, monkeypatch, capsys):
    from woof.hex.errors import MpasPortError

    def broken(spec):
        raise MpasPortError("engine says no")

    monkeypatch.setattr(plan_hex, "_generator_sizing", broken)
    assert plan_hex.main(_args(tmp_path)) == 2
    assert "engine says no" in capsys.readouterr().err


# --------------------------------------------------------------------------
# the experimental lane: --hex-experimental-dt


def _compact_sites() -> SiteSet:
    """Two turbines and a short line: a cut that fits the reference card
    at the LES vertical's level count at 50 m and 100 m."""

    return SiteSet(sites=[
        Site(site_id="t1", asset_id="generator/1", kind="turbine",
             lat=51.70, lon=-3.64, hub_height_m=80.0),
        Site(site_id="t2", asset_id="generator/2", kind="turbine",
             lat=51.71, lon=-3.62, hub_height_m=80.0),
        Site(site_id="l0", asset_id="line/1", kind="line_sample",
             lat=51.705, lon=-3.66, chainage_m=0.0),
        Site(site_id="l1", asset_id="line/1", kind="line_sample",
             lat=51.712, lon=-3.645, chainage_m=1300.0),
    ], heights_m=(10.0, 80.0))


def _xplan(tmp_path, **overrides):
    kwargs = dict(outdir=tmp_path / "xplan", dx_m=100.0, corridor_km=1.0,
                  start="2026-10-10T00", hours=6.0, experimental_dt=True)
    kwargs.update(overrides)
    return plan_hex.build_plan(_compact_sites(), **kwargs)


RASTER_STAGES = [
    "density", "mesh", "mesh-check", "static-highres", "register-parent",
    "vertical", "cull", "mesh-check-cull", "register-cull", "fetch",
    "intermediate", "init", "lbc", "forecast"]


def _by_stage(domain):
    return dict(zip(domain.extra["stages"], domain.extra["commands"]))


def _flag(argv, flag):
    return argv[argv.index(flag) + 1]


def test_floor_refusal_names_the_experimental_lane(tmp_path):
    with pytest.raises(plan_hex.HexPlanRefusal) as caught:
        _plan(tmp_path, dx_m=100.0)
    message = str(caught.value)
    assert "--hex-experimental-dt" in message
    assert "experimental-unanchored" in message
    # today's refusal is kept word for word ahead of the new sentence
    assert message.index("or ask hex-swath for a coarser --dx-m.") < \
        message.index("--hex-experimental-dt")


@pytest.mark.parametrize("dx_m,dt", [(100.0, 600.0 / 1024.0),
                                     (50.0, 600.0 / 2048.0)])
def test_experimental_plan_lifts_the_floor(tmp_path, dx_m, dt):
    plan = _xplan(tmp_path, dx_m=dx_m)
    outdir = tmp_path / "xplan"
    domain = load_plan(outdir / "plan.json").domains[0]
    extra = domain.extra
    assert extra["timestep"]["dt_seconds"] == dt
    assert extra["timestep_evidence"] == "experimental-unanchored"
    assert extra["timestep"]["timestep_evidence"] == "experimental-unanchored"
    # Courant still binds: dt sits under 0.9 x 0.8484 dx / 125
    assert dt <= extra["timestep"]["courant_limit_seconds"]
    assert extra["timestep"]["courant_limit_seconds"] == pytest.approx(
        0.9 * 0.8484 * dx_m / 125.0)
    assert 600.0 / dt == round(600.0 / dt)
    assert extra["runnable"] == "full" and "blocked_stages" not in extra
    assert extra["stages"] == RASTER_STAGES
    assert domain.mesh["mesh_path"] == "runs/hex/corridor.init.nc"
    assert domain.output_glob == "forecast/cuda-history.*.nc"
    assert extra["env_paths"] == {
        "WOOF_HEX_MESH_ROWS": "runs/hex/mesh-rows.json"}
    for name in ("plan.json", "mesh_spec.json", "cull_region.json",
                 "regional_window.json", "vertical_spec.json", "sites.json"):
        assert (outdir / name).is_file(), name
    assert len(load_sites(outdir / "sites.json")) == 4
    assert extra["capacity"]["fits"] and extra["capacity"]["levels"] == 100
    assert "INTERIM" in extra["capacity"]["bytes_per_cell_basis"]
    assert extra["estimate"]["window_cells"] > extra["estimate"]["cull_cells"]
    assert plan.notes[0].startswith("EXPERIMENTAL (experimental-unanchored)")
    assert plan.source == "gfs"
    register = _by_stage(domain)["register-parent"]
    assert float(_flag(register, "--dt-seconds")) == dt


def test_experimental_chain_uses_the_frozen_contracts(tmp_path):
    domain = _xplan(tmp_path).domains[0]
    stage = _by_stage(domain)
    run = "runs/hex"
    assert stage["density"] == [
        "hex", "density", "--sites", "sites.json", "--fine-km", "0.1",
        "--background-km", f"{domain.mesh['background_km']:g}",
        "-o", f"{run}/density.nc"]
    mesh = stage["mesh"]
    assert mesh[:3] == ["mesh", "--density-raster", f"{run}/density.nc"]
    assert _flag(mesh, "--regional-window") == "regional_window.json"
    assert int(_flag(mesh, "--cells")) >= domain.extra["estimate"][
        "window_cells"]
    assert stage["static-highres"] == [
        "hex", "static-highres", "--static", f"{run}/parent.static.nc",
        "-o", f"{run}/parent.static-highres.nc", "--terrain", "glo30",
        "--landuse", "cglc"]
    parent_row = domain.mesh["parent_row"]
    cull_row = domain.mesh["cull_row"]
    assert stage["register-parent"][:2] == ["hex", "register"]
    assert _flag(stage["register-parent"], "--name") == parent_row
    assert _flag(stage["register-parent"], "--static") == \
        f"{run}/parent.static-highres.nc"
    assert stage["vertical"] == [
        "hex", "vertical", "--grid", f"{run}/parent.grid.nc", "--static",
        f"{run}/parent.static-highres.nc", "--vertical-spec",
        "vertical_spec.json", "-o", f"{run}/parent.vertical.nc"]
    cull = stage["cull"]
    assert _flag(cull, "--parent-vertical") == f"{run}/parent.vertical.nc"
    assert _flag(cull, "--parent-static") == f"{run}/parent.static-highres.nc"
    assert _flag(cull, "--region") == "cull_region.json"
    reg = stage["register-cull"]
    assert _flag(reg, "--parent-row") == parent_row
    assert _flag(reg, "--cull-receipt") == f"{run}/corridor.cull.json"
    assert _flag(reg, "--name") == cull_row
    assert _flag(reg, "--rows") == f"{run}/mesh-rows.json"
    assert stage["fetch"][:3] == ["fetch", "--source", "gfs"]
    assert _flag(stage["fetch"], "--cadence") == "1"
    assert _flag(stage["fetch"], "--hours") == "6"
    assert _flag(stage["intermediate"], "--source") == "gfs"
    assert _flag(stage["intermediate"], "--hours") == "0-6"
    assert int(_flag(stage["fetch"], "--radius-km")) > \
        int(_flag(stage["intermediate"], "--radius-km")) + \
        float(_flag(stage["intermediate"], "--margin-km"))
    init = stage["init"]
    assert _flag(init, "--capsule") == _flag(init, "--reference") == \
        f"{run}/corridor.vertical.nc"
    assert _flag(init, "--out") == domain.mesh["mesh_path"]
    assert _flag(init, "--met") == f"{run}/met/MET:2026-10-10_00"
    assert _flag(stage["lbc"], "--stop-time") == "2026-10-10_06:00:00"
    forecast = stage["forecast"]
    assert _flag(forecast, "--mesh") == cull_row
    assert "--experimental-dt" in forecast
    assert _flag(forecast, "--les-model") == "3d_smagorinsky"
    assert _flag(forecast, "--pbl") == "off"
    assert _flag(forecast, "--history-preset") == "energy"
    assert _flag(forecast, "--out") == f"{run}/forecast"
    assert float(_flag(forecast, "--hours")) == 6.0
    for argv in domain.extra["commands"]:
        for token in argv:
            assert not Path(token).is_absolute(), token
    window = json.loads((tmp_path / "xplan" / "regional_window.json")
                        .read_text())
    assert window["kind"] == "polygon" and len(window["vertices_deg"]) >= 3


def test_experimental_chain_variants(tmp_path, monkeypatch):
    domain = _xplan(tmp_path, density="polygons",
                    source="ecmwf-open-data").domains[0]
    stage = _by_stage(domain)
    assert domain.extra["stages"][0] == "mesh-plan"
    assert stage["mesh"][:3] == ["mesh", "--spec", "mesh_spec.json"]
    assert "--density-raster" not in stage["mesh"]
    assert not (tmp_path / "xplan" / "sites.json").exists()
    assert _flag(stage["intermediate"], "--source") == "ecmwf-open-data"
    mesh_needs = next(r for r in domain.extra["requires"]
                      if r["stage"] == "mesh")["needs"]
    assert [n["unit"] for n in mesh_needs] == [9]
    # a 3-hourly source: fetched on its own ladder, the met window rounded
    # up to whole leads, and the coarse ladder named as a need
    assert _flag(stage["fetch"], "--cadence") == "3"
    assert _flag(stage["intermediate"], "--hours") == "0-6"
    met_needs = next(r for r in domain.extra["requires"]
                     if r["stage"] == "intermediate")["needs"]
    assert [n["unit"] for n in met_needs] == [11, 11]
    domain = _xplan(tmp_path, source="aifs", hours=7.0).domains[0]
    assert _flag(_by_stage(domain)["intermediate"], "--hours") == "0-12"

    # a relative WRF glob is rewritten against the plan directory, the cwd
    # its commands run in
    here = tmp_path / "here"
    here.mkdir()
    monkeypatch.chdir(here)
    domain = _xplan(tmp_path, wrfout_glob="wrf/wrfout_d01_*").domains[0]
    stage = _by_stage(domain)
    assert "fetch" not in stage
    assert stage["intermediate"][:6] == [
        "hex", "intermediate", "--source", "wrfout", "--wrfout-glob",
        "../here/wrf/wrfout_d01_*"]
    assert (tmp_path / "xplan" / "../here/wrf").resolve() == \
        (here / "wrf").resolve()
    assert "--grib-dir" not in stage["intermediate"]
    assert domain.extra["forcing"]["source"] == "wrfout"
    needs = next(r for r in domain.extra["requires"]
                 if r["stage"] == "intermediate")["needs"]
    assert [n["unit"] for n in needs] == [12]


def test_requirements_name_each_feature_and_parse_with_real_parsers(
        tmp_path):
    from woof.cli import build_parser
    from woof.hex.cli import build_parser as hex_parser

    domain = _xplan(tmp_path).domains[0]
    requires = domain.extra["requires"]
    assert [r["command"] for r in requires] == \
        list(range(len(domain.extra["commands"])))
    units = {r["stage"]: sorted(n["unit"] for n in r["needs"])
             for r in requires}
    assert units["density"] == [8]
    assert units["mesh"] == [7, 9]
    assert units["static-highres"] == [13]
    assert units["register-parent"] == units["register-cull"] == [10]
    assert units["vertical"] == units["cull"] == [10]
    assert units["intermediate"] == [11]
    assert units["forecast"] == [1, 5, 6, 16]
    for stage in ("mesh-check", "mesh-check-cull", "fetch", "init", "lbc"):
        assert units[stage] == [], stage
    for row, argv in zip(requires, domain.extra["commands"]):
        if not row["needs"]:
            # nothing new: this build's own parser must take it as written
            assert row["available_in_this_build"], row
            if argv[0] == "hex":
                parsed = hex_parser().parse_args(argv[1:])
                assert callable(parsed.handler)
            else:
                build_parser().parse_args(argv)
        elif not row["available_in_this_build"]:
            assert row["why_not"], row
            assert "choose from" not in row["why_not"]


def test_main_reports_the_missing_features(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(plan_hex, "load_sites", lambda path: _compact_sites())
    args = _args(tmp_path, dx_m=100.0, corridor_km=1.0, hours=6.0,
                 hex_experimental_dt=True, hex_density=None,
                 hex_forcing_wrfout=None)
    assert plan_hex.main(args) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["timestep_evidence"] == "experimental-unanchored"
    assert record["dt_seconds"] == 600.0 / 1024.0
    assert record["runnable"] == "full"
    assert record["capacity"]["levels"] == 100
    missing = {row["argv_head"] for row in
               record["commands_missing_in_this_build"]}
    plan = load_plan(tmp_path / "plan" / "plan.json")
    expected = {r["argv_head"] for r in plan.domains[0].extra["requires"]
                if not r["available_in_this_build"]}
    assert missing == expected


def test_experimental_capacity_refusal(tmp_path):
    with pytest.raises(plan_hex.HexPlanRefusal) as caught:
        plan_hex.build_plan(load_sites(SITES), outdir=tmp_path / "plan",
                            dx_m=100.0, start="2026-10-10T00",
                            vram_gib=48.0, experimental_dt=True)
    message = str(caught.value)
    assert "cells" in message and "wrf-tiles" in message
    assert "100 levels" in message and "shorter corridor" in message
    assert not (tmp_path / "plan" / "plan.json").exists()


def test_capacity_scales_with_levels():
    base = plan_hex.capacity_verdict(100_000.0, None, 48.0)
    same = plan_hex.capacity_verdict(100_000.0, None, 48.0, levels=55)
    assert {k: v for k, v in same.items() if k != "levels"} == base
    deep = plan_hex.capacity_verdict(100_000.0, None, 48.0, levels=100)
    assert deep["levels"] == 100
    assert deep["required_mib"] > base["required_mib"] * 1.5
    assert deep["cells_that_fit"] < base["cells_that_fit"]
    assert "INTERIM" in deep["bytes_per_cell_basis"]


def test_windowed_gradient_matches_the_uniform_estimate(monkeypatch):
    background, rungs_km = plan_hex.background_km_for(100.0, None)
    rungs = plan_hex.ladder(rungs_km, 1.0, 0.1, 80.0)
    monkeypatch.setattr(plan_hex, "MAX_GRADIENT_SAMPLES", 10 ** 12)
    uniform = plan_hex.steepest_gradient_percent(rungs, background)
    monkeypatch.setattr(plan_hex, "MAX_GRADIENT_SAMPLES", 1)
    windowed = plan_hex.steepest_gradient_percent(rungs, background)
    assert windowed == pytest.approx(uniform, rel=1e-9)


def test_les_vertical_spec():
    from woof.hex.vertical_spec import VerticalSpec

    doc = plan_hex.les_vertical_spec()
    VerticalSpec.from_mapping(doc).validate()
    z = np.asarray(doc["specified_interfaces_m"])
    assert doc["scheme"] == "specified" and doc["n_vert_levels"] == 100
    assert len(z) == 101 and z[0] == 0.0 and z[-1] == doc["ztop_m"]
    dz = np.diff(z)
    assert np.all(dz > 0.0)
    assert dz[0] == pytest.approx(plan_hex.LES_SURFACE_DZ_M, rel=0.02)
    assert np.all(dz[1:] >= dz[:-1] - 1e-2)
    assert plan_hex.les_vertical_spec(60)["n_vert_levels"] == 60
    for n in (40, 55, 60, 200):
        few = plan_hex.les_vertical_spec(n)
        VerticalSpec.from_mapping(few).validate()
        z = np.asarray(few["specified_interfaces_m"])
        assert len(z) == n + 1 and z[-1] == few["ztop_m"]
        # the LES surface layer holds however few levels are asked for
        assert z[1] == pytest.approx(plan_hex.LES_SURFACE_DZ_M, rel=0.02), n
    with pytest.raises(plan_hex.HexPlanRefusal, match="--nz 20"):
        plan_hex.les_vertical_spec(20)


def test_experimental_controls_are_refused(tmp_path):
    with pytest.raises(plan_hex.HexPlanRefusal, match="--hex-density"):
        _plan(tmp_path, density="raster")
    with pytest.raises(plan_hex.HexPlanRefusal, match="--hex-forcing-wrfout"):
        _plan(tmp_path, wrfout_glob="wrfout_d01_*")
    with pytest.raises(plan_hex.HexPlanRefusal, match="give one"):
        _xplan(tmp_path, source="gfs", wrfout_glob="wrfout_d01_*")
    with pytest.raises(plan_hex.HexPlanRefusal, match="no hex regional"):
        _xplan(tmp_path, source="cmc")
    with pytest.raises(plan_hex.HexPlanRefusal, match="not one of"):
        _xplan(tmp_path, density="voronoi")
    with pytest.raises(plan_hex.HexPlanRefusal, match="whole number"):
        _xplan(tmp_path, hours=0.0002)
    with pytest.raises(plan_hex.HexPlanRefusal, match="no timestep"):
        _xplan(tmp_path, dx_m=30.0)
    with pytest.raises(plan_hex.HexPlanRefusal, match="drop the flag"):
        _xplan(tmp_path, dx_m=937.5)
    with pytest.raises(plan_hex.HexPlanRefusal, match="f009"):
        _xplan(tmp_path, source="gdas", hours=12.0)
    with pytest.raises(plan_hex.HexPlanRefusal, match="--nz 500"):
        _xplan(tmp_path, nz=500)
    assert not (tmp_path / "xplan" / "plan.json").exists()
    domain = _xplan(tmp_path, nz=80).domains[0]
    assert domain.extra["vertical"]["n_vert_levels"] == 80
    assert domain.extra["capacity"]["levels"] == 80


def test_cli_hex_lane_flags(tmp_path, capsys):
    from woof.cli import build_parser

    args = build_parser().parse_args([
        "energy", "plan", str(SITES), "--topology", "hex-swath",
        "--dx-m", "100", "--hex-experimental-dt", "--hex-density",
        "polygons", "--hex-forcing-wrfout", "wrfout_d01_*",
        "-o", str(tmp_path / "plan")])
    assert args.hex_experimental_dt and args.hex_density == "polygons"
    assert args.hex_forcing_wrfout == "wrfout_d01_*"
    default = build_parser().parse_args([
        "energy", "plan", str(SITES), "--topology", "hex-swath",
        "-o", str(tmp_path / "plan")])
    assert default.hex_experimental_dt is False
    assert default.hex_density is None and default.hex_forcing_wrfout is None
    other = build_parser().parse_args([
        "energy", "plan", str(SITES), "--topology", "wrf-nests",
        "--hex-experimental-dt", "-o", str(tmp_path / "nests")])
    assert other.func(other) == 2
    assert "only to --topology hex-swath" in capsys.readouterr().err
    assert not (tmp_path / "nests").exists()


def test_run_exports_the_rows_path(tmp_path, monkeypatch):
    from woof.energy import run
    from woof.energy.contracts import PlanDomain

    ring = ((-3.6, 51.7), (-3.5, 51.7), (-3.5, 51.8), (-3.6, 51.7))

    def domain(env_paths):
        return PlanDomain(
            domain_id="hex", topology="hex-swath", role="mesh", dx_m=100.0,
            run_dir="runs/hex", output_glob="forecast/cuda-history.*.nc",
            footprint=ring, site_ids=("s",), mesh={"grid": "g.nc"},
            extra={"commands": [["hex", "version"]],
                   "env_paths": env_paths})

    from woof.energy.contracts import Plan, dump_plan

    rows = {"WOOF_HEX_MESH_ROWS": "runs/hex/mesh-rows.json"}
    step = run._hex_step(domain(rows))
    assert step.env_paths == rows
    assert any("WOOF_HEX_MESH_ROWS" in note for note in step.notes)
    monkeypatch.delenv("WOOF_HEX_MESH_ROWS", raising=False)
    exported = run._step_env(step, tmp_path)
    expected = str((tmp_path / "runs/hex/mesh-rows.json").resolve())
    assert run._child_env(exported)["WOOF_HEX_MESH_ROWS"] == expected
    assert "WOOF_HEX_MESH_ROWS" not in run._child_env()

    # through woof energy run: every command of the step gets the path
    plan_path = dump_plan(Plan(topology="hex-swath", dx_m=100.0,
                               start="2026-10-10T00", hours=6.0,
                               domains=[domain(rows)]),
                          tmp_path / "plan.json")
    seen = []

    def fake(argv, cwd, log, env=None):
        seen.append(env)
        out = Path(cwd) / "runs/hex/forecast"
        out.mkdir(parents=True, exist_ok=True)
        (out / "cuda-history.2026-10-10_00.nc").write_bytes(b"x")
        return 0

    monkeypatch.setattr(run, "_execute", fake)
    assert run.main(argparse.Namespace(plan=str(plan_path), dry_run=False,
                                       only=None, resume=False)) == 0
    assert seen == [{"WOOF_HEX_MESH_ROWS": expected}]
    for bad in ({"WOOF_HEX_MESH_ROWS": "/abs/rows.json"},
                {"WOOF_HEX_MESH_ROWS": 3}, ["WOOF_HEX_MESH_ROWS"]):
        with pytest.raises(run.RunRefusal, match="env_paths"):
            run._hex_step(domain(bad))
