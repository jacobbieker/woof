"""``woof energy plan --topology wrf-tiles`` (woof.energy.plan_wrf_tiles).

The parent is emitted by the real ``woof domain`` wizard and every tile is
priced by the real estimator; only the rectangle cover is a stand-in (a
greedy slab cover with the frozen ``cover_with_rectangles`` contract), so
these tests do not depend on the geometry unit having landed.  CPU only.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pytest

from woof.energy import plan_wrf_tiles as tiles
from woof.energy.contracts import (
    EnergyNotImplemented,
    Site,
    SiteSet,
    load_plan,
    load_sites,
)
from woof.energy.geometry import Rect

FIXTURES = Path(__file__).parent / "fixtures" / "energy"
SITES_WALES = FIXTURES / "sites_wales.json"
START = "2026-10-01T00"


def fake_cover(x, y, *, margin_m, dx_m, max_nx, max_ny, align_m=None):
    """Greedy cover honouring the frozen contract: slabs along x, each
    rectangle the members' box plus ``margin_m``, snapped outward to
    ``align_m`` and no larger than ``max_nx`` x ``max_ny`` cells."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    align = align_m or dx_m

    def box(idx):
        x0 = math.floor((x[idx].min() - margin_m) / align) * align
        x1 = math.ceil((x[idx].max() + margin_m) / align) * align
        y0 = math.floor((y[idx].min() - margin_m) / align) * align
        y1 = math.ceil((y[idx].max() + margin_m) / align) * align
        return x0, y0, x1, y1, round((x1 - x0) / dx_m), round((y1 - y0) / dx_m)

    rects, group = [], []
    for index in np.argsort(x, kind="stable"):
        trial = group + [int(index)]
        *_, nx, ny = box(np.array(trial))
        if group and (nx > max_nx or ny > max_ny):
            x0, y0, x1, y1, gnx, gny = box(np.array(group))
            rects.append(Rect(x0, y0, x1, y1, gnx, gny, np.array(group)))
            group = [int(index)]
        else:
            group = trial
    x0, y0, x1, y1, gnx, gny = box(np.array(group))
    rects.append(Rect(x0, y0, x1, y1, gnx, gny, np.array(group)))
    return rects


@pytest.fixture
def cover(monkeypatch):
    monkeypatch.setattr("woof.energy.geometry.cover_with_rectangles",
                        fake_cover)


def _plan_args(sites, outdir, *extra):
    from woof.cli import build_parser

    return build_parser().parse_args(
        ["energy", "plan", str(sites), "--topology", "wrf-tiles",
         "--dx-m", "100", "--start", START, "--hours", "6",
         "-o", str(outdir), *extra])


@pytest.fixture(scope="module")
def wales(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("wales") / "plan-tiles"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("woof.energy.geometry.cover_with_rectangles",
                      fake_cover)
        args = _plan_args(SITES_WALES, outdir)
        import contextlib
        import io

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = args.func(args)
    return code, json.loads(out.getvalue()), outdir


def _line_sites(n: int, *, lat: float = 52.3, lon0: float = -4.0,
                lon1: float = -1.0) -> SiteSet:
    lons = np.linspace(lon0, lon1, n)
    return SiteSet(sites=[
        Site(site_id=f"s{k:04d}", asset_id="line:1", kind="line_sample",
             lat=lat, lon=float(value), bearing_deg=90.0,
             chainage_m=float(k) * 100.0, voltage_kv=400.0)
        for k, value in enumerate(lons)], heights_m=(10.0, 30.0))


# --------------------------------------------------------------------------
# pure helpers


@pytest.mark.parametrize("total, chain", [
    (5, (5,)), (8, (8,)), (25, (5, 5)), (30, (6, 5)), (10, (5, 2)),
    (125, (5, 5, 5)), (2, (2,))])
def test_factor_chain(total, chain):
    assert tiles.factor_chain(total) == chain


@pytest.mark.parametrize("total", [1, 11, 13 * 3])
def test_factor_chain_refusals(total):
    with pytest.raises(tiles.TilePlanRefusal):
        tiles.factor_chain(total)


def test_default_start_is_latest_synoptic_cycle():
    now = datetime(2026, 10, 10, 13, 45, tzinfo=timezone.utc)
    assert tiles.default_start(now) == "2026-10-10T12"
    assert tiles.default_start(
        datetime(2026, 10, 10, 5, 59, tzinfo=timezone.utc)) == "2026-10-10T00"


def test_placement_point_matches_downscale_centering():
    from woof.downscale import _centered_placement

    parent = {"nx": 200, "ny": 200}
    for start in (20, 33, 57):
        for span in (6, 7, 10, 13, 40):
            i0 = tiles._placement_point(start, span)
            placement = _centered_placement(parent, j0=i0, i0=i0, ratio=5,
                                            child_nx=span * 5,
                                            child_ny=span * 5)
            assert placement.i_parent_start == start
            assert placement.j_parent_start == start


# --------------------------------------------------------------------------
# the Wales fixture plan


def test_wales_plan_is_valid(wales):
    code, record, outdir = wales
    assert code == 0
    plan = load_plan(outdir / "plan.json")
    assert plan.topology == "wrf-tiles"
    assert plan.start == START and plan.hours == 6.0
    parent = plan.domain("parent")
    assert parent.role == "parent" and parent.grid_id == 1
    assert parent.run_dir == "runs/parent"
    assert parent.output_glob == "run-*/run/wrfout/wrfout_d01_*"
    assert parent.parent is None and parent.dx_m == 500.0
    assert (outdir / parent.config).is_file()
    assert (outdir / parent.wps_namelist).is_file()
    children = [d for d in plan.domains if d.role == "child"]
    assert children and record["tile_count"] == len(children)
    for domain in children:
        assert domain.parent == "parent" and domain.grid_id == 1
        assert domain.dx_m == 100.0
        assert domain.run_dir == f"runs/{domain.domain_id}"
        assert domain.output_glob == "wrfout_d02_*"
        spec = json.loads((outdir / domain.config).read_text())
        assert spec["schema"] == tiles.TILE_SCHEMA
        assert spec["nx"] % spec["ratio"] == 0
        assert spec["downscale_args"] == domain.extra["downscale_args"]
        # The orchestrator requires --out at (or under) the tile's run_dir.
        args = domain.extra["downscale_args"]
        assert args[args.index("--out") + 1] == domain.run_dir
        assert spec["parent_frames_glob"] == "runs/parent/run-*/run/wrfout"
        assert spec["physics_overrides"]["applied"] is False
        assert spec["physics_overrides"]["recipe"]["km_opt"] == 3
        assert spec["history"] == {**spec["history"], "preset": "energy",
                                   "applied": False}
    assert plan.sites_ref["sha256"]
    assert (outdir / plan.sites_ref["path"]).resolve() == SITES_WALES.resolve()
    assert record["parent"]["nx"] == parent.extra["nx"]
    assert any("NOT applied" in note for note in plan.notes)


def test_wales_parent_toml_loads(wales):
    from woof.experiment import load_experiment

    _, _, outdir = wales
    plan = load_plan(outdir / "plan.json")
    exp = load_experiment(outdir / plan.domain("parent").config)
    assert exp.root.history_interval_s == 900.0
    assert exp.root.run.restart_interval_s > 0.0


def test_every_site_owned_exactly_once(wales):
    _, _, outdir = wales
    plan = load_plan(outdir / "plan.json")
    owned = [s for d in plan.domains for s in d.site_ids]
    expected = [s.site_id for s in load_sites(SITES_WALES).sites]
    assert sorted(owned) == sorted(expected)
    assert len(owned) == len(set(owned))


def test_owned_sites_inside_tile_footprint(wales):
    from woof.static.projection import grids_from_wps_namelist

    _, _, outdir = wales
    plan = load_plan(outdir / "plan.json")
    sites = {s.site_id: s for s in load_sites(SITES_WALES).sites}
    grid = grids_from_wps_namelist(outdir / plan.domain("parent").wps_namelist)[0]
    for domain in plan.domains:
        if domain.role != "child":
            continue
        spec = json.loads((outdir / domain.config).read_text())
        child = grid.nest(spec["i_parent_start"], spec["j_parent_start"],
                          spec["ratio"], spec["nx"] + 1, spec["ny"] + 1)
        for site_id in domain.site_ids:
            i, j = child.latlon_to_ij(sites[site_id].lat, sites[site_id].lon)
            # At least the corridor (2 km = 20 cells at 100 m) inside.
            assert 20.0 <= float(i) <= spec["nx"] - 19.0
            assert 20.0 <= float(j) <= spec["ny"] - 19.0


def test_downscale_args_parse_with_the_real_parser(wales):
    from woof.cli import build_parser

    _, _, outdir = wales
    plan = load_plan(outdir / "plan.json")
    parser = build_parser()
    for domain in plan.domains:
        if domain.role == "parent" and domain.domain_id == "parent":
            continue
        parent_dir = plan.domain(domain.parent).run_dir
        args = parser.parse_args(
            ["downscale", parent_dir, *domain.extra["downscale_args"]])
        spec = json.loads((outdir / domain.config).read_text())
        assert args.parent == [parent_dir]
        assert args.out == Path(domain.run_dir)
        assert args.ratio == spec["ratio"] == 5
        assert args.child_size == f"{spec['nx']},{spec['ny']}"
        lat, lon = (float(v) for v in args.point.split(","))
        assert lat == pytest.approx(spec["centre"]["lat"], abs=1e-6)
        assert lon == pytest.approx(spec["centre"]["lon"], abs=1e-6)
        assert args.parent_restart == "latest"
        assert args.parent_domain == 1
        assert args.child_levels == "60,2.5"
        assert args.max_boundary_interval_seconds == 900.0
        assert args.card == "24gb"
        assert args.render_products == "none"


def test_downscale_point_lands_on_the_planned_parent_cell(wales):
    """The --point downscale receives resolves to the planned placement
    through downscale's own nearest-point search and centring."""

    from woof.downscale import _centered_placement, _nearest_parent_index
    from woof.static.projection import grids_from_wps_namelist

    _, _, outdir = wales
    plan = load_plan(outdir / "plan.json")
    grid = grids_from_wps_namelist(outdir / plan.domain("parent").wps_namelist)[0]
    xlat, xlong = grid.latlon_mass()
    parent = {"nx": grid.e_we - 1, "ny": grid.e_sn - 1}
    for domain in plan.domains:
        if domain.role != "child":
            continue
        spec = json.loads((outdir / domain.config).read_text())
        j0, i0 = _nearest_parent_index(xlat.astype(np.float32),
                                       xlong.astype(np.float32),
                                       spec["centre"]["lat"],
                                       spec["centre"]["lon"])
        placement = _centered_placement(parent, j0=j0, i0=i0, ratio=5,
                                        child_nx=spec["nx"],
                                        child_ny=spec["ny"])
        assert placement.i_parent_start == spec["i_parent_start"]
        assert placement.j_parent_start == spec["j_parent_start"]


# --------------------------------------------------------------------------
# many tiles, and the refusals


def test_long_line_yields_many_tiles(tmp_path, cover, monkeypatch):
    monkeypatch.setattr(tiles, "_tile_capacity",
                        lambda *args, ratio, **kwargs: 60)
    sites = _line_sites(400)
    plan = tiles.build_plan(sites, outdir=tmp_path / "plan", dx_m=100.0,
                            corridor_km=1.0, start=START, hours=6)
    children = [d for d in plan.domains if d.role == "child"]
    assert len(children) > 21
    assert all(d.extra["nx"] <= 60 and d.extra["ny"] <= 60 for d in children)
    owned = [s for d in children for s in d.site_ids]
    assert sorted(owned) == sorted(s.site_id for s in sites.sites)
    assert len({d.domain_id for d in plan.domains}) == len(plan.domains)
    assert len(list((tmp_path / "plan" / "tiles").glob("*.json"))) == len(children)


def test_max_domains_refused(tmp_path, cover, monkeypatch):
    monkeypatch.setattr(tiles, "_tile_capacity",
                        lambda *args, ratio, **kwargs: 60)
    with pytest.raises(tiles.TooManyTiles, match="--max-domains 2"):
        tiles.build_plan(load_sites(SITES_WALES), outdir=tmp_path / "plan",
                         dx_m=100.0, start=START, hours=6, max_domains=2)
    assert not (tmp_path / "plan" / "plan.json").exists()


def test_sites_outside_parent_refused(tmp_path, cover, monkeypatch):
    real = tiles._sites_region

    def western_half(lon, lat):
        keep = lon < np.median(lon)
        return real(lon[keep], lat[keep])

    monkeypatch.setattr(tiles, "_sites_region", western_half)
    with pytest.raises(tiles.SiteOutsideParent):
        tiles.build_plan(load_sites(SITES_WALES), outdir=tmp_path / "plan",
                         dx_m=100.0, start=START, hours=6)
    assert not (tmp_path / "plan" / "plan.json").exists()


def test_parent_dx_not_a_multiple_refused(tmp_path):
    with pytest.raises(tiles.TilePlanRefusal, match="whole multiple"):
        tiles.build_plan(load_sites(SITES_WALES), outdir=tmp_path,
                         dx_m=100.0, parent_dx_m=333.0, start=START)


def test_parent_dx_with_large_prime_ratio_refused(tmp_path):
    with pytest.raises(tiles.TilePlanRefusal, match="prime factor"):
        tiles.build_plan(load_sites(SITES_WALES), outdir=tmp_path,
                         dx_m=100.0, parent_dx_m=1100.0, start=START)


def test_fractional_hours_refused(tmp_path):
    with pytest.raises(tiles.TilePlanRefusal, match="whole number of hours"):
        tiles.build_plan(load_sites(SITES_WALES), outdir=tmp_path,
                         dx_m=100.0, hours=1.5, start=START)


def test_bad_start_refused(tmp_path):
    with pytest.raises(tiles.TilePlanRefusal, match="YYYY-MM-DDTHH"):
        tiles.build_plan(load_sites(SITES_WALES), outdir=tmp_path,
                         dx_m=100.0, start="tomorrow")


def test_empty_sites_refused(tmp_path):
    with pytest.raises(tiles.TilePlanRefusal, match="no sites"):
        tiles.build_plan(SiteSet(sites=[], heights_m=(10.0,)),
                         outdir=tmp_path, start=START)


def test_geometry_contract_violation_refused(tmp_path, monkeypatch):
    def overlapping(x, y, **kwargs):
        rects = fake_cover(x, y, **kwargs)
        members = np.arange(np.asarray(x).size)
        return rects + [Rect(rects[0].x_min, rects[0].y_min, rects[0].x_max,
                             rects[0].y_max, rects[0].nx, rects[0].ny,
                             members[:1])]

    monkeypatch.setattr("woof.energy.geometry.cover_with_rectangles",
                        overlapping)
    with pytest.raises(tiles.GeometryContractError, match="exactly one"):
        tiles.build_plan(load_sites(SITES_WALES), outdir=tmp_path / "plan",
                         dx_m=100.0, start=START, hours=6)


def test_main_reports_missing_geometry(tmp_path, monkeypatch, capsys):
    def stub(*args, **kwargs):
        raise EnergyNotImplemented("woof.energy.geometry.cover_with_rectangles")

    monkeypatch.setattr("woof.energy.geometry.cover_with_rectangles", stub)
    args = _plan_args(SITES_WALES, tmp_path / "plan")
    assert args.func(args) == 2
    assert "cover_with_rectangles" in capsys.readouterr().err


def test_parent_too_big_falls_back_to_two_step_chain(tmp_path, cover,
                                                     monkeypatch):
    real = tiles._emit_parent

    def refuse_500(**kwargs):
        if kwargs["parent_dx_m"] == 500.0:
            raise tiles._ParentDoesNotFit("d01 EXCEEDS the budget")
        return real(**kwargs)

    monkeypatch.setattr(tiles, "_emit_parent", refuse_500)
    plan = tiles.build_plan(load_sites(SITES_WALES), outdir=tmp_path / "plan",
                            dx_m=100.0, start=START, hours=6)
    assert plan.domain("parent").dx_m == 2500.0
    inter = [d for d in plan.domains if d.domain_id.startswith("inter-")]
    leaves = [d for d in plan.domains if d.role == "child"]
    assert inter and leaves
    assert all(d.role == "parent" and d.parent == "parent" and not d.site_ids
               and d.dx_m == 500.0 and d.output_glob == "wrfout_d02_*"
               for d in inter)
    ids = {d.domain_id for d in inter}
    for leaf in leaves:
        assert leaf.parent in ids and leaf.output_glob == "wrfout_d03_*"
        args = leaf.extra["downscale_args"]
        assert args[args.index("--parent-domain") + 1] == "2"
    assert any("did not fit" in note for note in plan.notes)
    order = [d.domain_id for d in plan.run_order()]
    for leaf in leaves:
        assert order.index(leaf.parent) < order.index(leaf.domain_id)


def test_per_tile_budget_below_smallest_tile_refused(wales):
    import dataclasses

    from woof.experiment import load_experiment

    _, _, outdir = wales
    plan = load_plan(outdir / "plan.json")
    exp = load_experiment(outdir / plan.domain("parent").config)
    pricer = tiles._Pricer(2.0)
    with pytest.raises(tiles.TilePlanRefusal, match="per-tile budget"):
        pricer.largest_square(
            ("parent", 5, 60), dataclasses.asdict(exp.root.run), ratio=5,
            parent_dx=500.0, run_seconds=21600.0, output_interval_s=900.0,
            levels=None, centre_lat=51.7)
