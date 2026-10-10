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
        _plan(tmp_path, nz=2)
    with pytest.raises(plan_hex.HexPlanRefusal, match="--nz"):
        _plan(tmp_path, nz=plan_hex.HEX_MAX_LEVELS + 1)
    with pytest.raises(plan_hex.HexPlanRefusal, match="YYYY-MM-DDTHH"):
        _plan(tmp_path, start="2026-10-10 00:00")
    with pytest.raises(plan_hex.HexPlanRefusal, match="no sites"):
        plan_hex.build_plan(SiteSet(sites=[], heights_m=(10.0,)),
                            outdir=tmp_path / "plan", dx_m=937.5)
    native = _plan(tmp_path, nz=55).domains[0]
    assert "levels" not in native.mesh


def test_a_declared_column_is_priced_at_its_level_count(tmp_path):
    native = _plan(tmp_path / "a").domains[0]
    deep = _plan(tmp_path / "b", nz=80).domains[0]
    assert deep.mesh["levels"] == 80
    assert "preset:les:levels=80" in deep.mesh["vertical_spec"]
    assert deep.extra["capacity"]["levels"] == 80
    assert (deep.extra["capacity"]["required_mib"]
            > native.extra["capacity"]["required_mib"])


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
