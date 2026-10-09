"""``woof energy plan --topology wrf-nests``: one config, sibling LES leaves."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import tomllib

import numpy as np
import pytest

from woof.energy import geometry
from woof.energy import plan_wrf_nests as nests
from woof.energy.contracts import (EnergyNotImplemented, load_plan,
                                   load_sites, sha256_file)

FIXTURES = Path(__file__).parent / "fixtures" / "energy"
SITES = FIXTURES / "sites_wales.json"
START = "2026-10-10T00"


def _fake_cover(x, y, *, margin_m, dx_m, max_nx, max_ny, align_m=None):
    """Deterministic grid-bin cover standing in for the geometry unit."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    width = max((max_nx - 1) * dx_m - 2 * margin_m, dx_m)
    height = max((max_ny - 1) * dx_m - 2 * margin_m, dx_m)
    bx = np.floor((x - x.min()) / width).astype(int)
    by = np.floor((y - y.min()) / height).astype(int)
    rects = []
    for key in sorted(set(zip(by.tolist(), bx.tolist()))):
        members = np.flatnonzero((by == key[0]) & (bx == key[1]))
        xs, ys = x[members], y[members]
        x0, x1 = xs.min() - margin_m, xs.max() + margin_m
        y0, y1 = ys.min() - margin_m, ys.max() + margin_m
        if align_m:
            x0 = math.floor(x0 / align_m) * align_m
            y0 = math.floor(y0 / align_m) * align_m
            x1 = math.ceil(x1 / align_m) * align_m
            y1 = math.ceil(y1 / align_m) * align_m
        rects.append(geometry.Rect(x0, y0, x1, y1,
                                   int(round((x1 - x0) / dx_m)) + 1,
                                   int(round((y1 - y0) / dx_m)) + 1,
                                   members))
    return rects


def _geometry_is_stub() -> bool:
    try:
        geometry.cover_with_rectangles(np.zeros(1), np.zeros(1), margin_m=1.0,
                                       dx_m=1.0, max_nx=8, max_ny=8)
    except EnergyNotImplemented:
        return True
    return False


@pytest.fixture(scope="module")
def cover():
    with pytest.MonkeyPatch.context() as patch:
        if _geometry_is_stub():
            patch.setattr(geometry, "cover_with_rectangles", _fake_cover)
        yield patch


@pytest.fixture(autouse=True)
def _cover_each(cover):
    yield


@pytest.fixture(scope="module", params=[100.0, 50.0], ids=["dx100", "dx50"])
def built(request, cover, tmp_path_factory):
    dx = request.param
    outdir = tmp_path_factory.mktemp(f"nests{int(dx)}")
    sites = load_sites(SITES)
    if dx == 100.0:
        plan, summary = nests._build(
            sites, outdir=outdir, dx_m=dx, corridor_km=2.0, parent_dx_m=None,
            start=START, hours=24.0, source=None, card=None, vram_gib=24.0,
            max_domains=None, nz=None, sites_path=SITES, name=None, now=None)
    else:
        # The public entry point, with no sites path: it binds a copy.
        plan = nests.build_plan(sites, outdir=outdir, dx_m=dx, start=START,
                                vram_gib=24.0)
        summary = None
    return dx, outdir, plan, summary, sites


def test_plan_validates_and_shares_one_config(built):
    dx, outdir, plan, _, _ = built
    again = load_plan(outdir / "plan.json")
    assert again.topology == "wrf-nests" and again.dx_m == dx
    assert again.start == "2026-10-10T00:00:00Z" and again.hours == 24.0
    configs = {d.config for d in again.domains}
    assert len(configs) == 1
    config = configs.pop()
    assert (outdir / config).is_file()
    assert (outdir / again.domains[0].wps_namelist).is_file()
    assert [d.grid_id for d in again.domains] == list(
        range(1, len(again.domains) + 1))
    for domain in again.domains:
        assert domain.topology == "wrf-nests" and domain.parent is None
        assert domain.run_dir == f"runs/{Path(config).stem}"
        assert domain.output_glob == (
            f"run-*/run/wrfout/wrfout_d{domain.grid_id:02d}_*")
        assert domain.footprint[0] == domain.footprint[-1]
        assert len(domain.footprint) == 5
    leaves = [d for d in again.domains if d.role == "child"]
    parents = [d for d in again.domains if d.role == "parent"]
    assert leaves and parents
    assert all(d.dx_m == dx for d in leaves)
    assert all(not d.site_ids for d in parents)
    assert parents[0].dx_m == dx * 25 and parents[0].grid_id == 1
    assert sorted({d.dx_m for d in again.domains}) == [dx, dx * 5, dx * 25]
    assert len(again.domains) <= 21


def test_every_site_owned_exactly_once(built):
    _, outdir, plan, _, sites = built
    owned = [s for d in plan.domains for s in d.site_ids]
    assert len(owned) == len(set(owned)) == len(sites)
    assert set(owned) == {s.site_id for s in sites.sites}


def test_sites_lie_inside_their_leaf_footprint(built):
    _, _, plan, _, sites = built
    by_id = {s.site_id: s for s in sites.sites}
    for domain in plan.domains:
        lons = [p[0] for p in domain.footprint]
        lats = [p[1] for p in domain.footprint]
        for site_id in domain.site_ids:
            site = by_id[site_id]
            assert min(lons) < site.lon < max(lons)
            assert min(lats) < site.lat < max(lats)


def test_sites_ref_binds_the_sites_document(built):
    dx, outdir, plan, _, _ = built
    ref = plan.sites_ref
    if dx == 100.0:
        assert Path(ref["path"]).is_absolute()
        assert ref["sha256"] == sha256_file(SITES)
    else:
        assert ref["path"] == "sites.json"
        assert ref["sha256"] == sha256_file(outdir / "sites.json")


def test_emitted_toml_loads_and_carries_the_recipe(built):
    from woof.experiment import load_experiment
    from woof.io.history_selection import HISTORY_PRESETS

    dx, outdir, plan, _, _ = built
    config = outdir / plan.domains[0].config
    exp = load_experiment(config)
    assert len(exp.domains) == len(plan.domains)
    raw = tomllib.loads(config.read_text())
    assert raw["experiment"]["feedback"] == 0
    assert raw["experiment"]["run_seconds"] == 24 * 3600.0
    assert raw["static"]["highres"]["enabled"] is True
    assert raw["static"]["highres"]["max_dx_m"] == 1000.0
    assert raw["fetch"]["source"] == "gfs"
    assert raw["fetch"]["cycle"] == START
    tables = {t["grid_id"]: t for t in raw["domain"]}
    leaf_ids = {d.grid_id for d in plan.domains if d.role == "child"}
    for grid_id, table in tables.items():
        if grid_id > 1:
            assert table["parent_id"] < grid_id
            assert table["nx"] % table["parent_grid_ratio"] == 0
            assert table["parent_grid_ratio"] in (3, 5)
        spacing = plan.domain(f"d{grid_id:02d}").dx_m
        if spacing < 1000.0:
            assert table["bl_pbl_physics"] == 0
            assert table["km_opt"] == 3
            assert table["mix_isotropic"] == 1
            assert table["cu_physics"] == 0
        else:
            assert "km_opt" not in table
        if grid_id in leaf_ids:
            assert table["history_interval_s"] == 900.0
            if "energy" in HISTORY_PRESETS:
                assert table["output"] == {"preset": "energy"}
            else:
                assert "U" in table["output"]["history_vars"]
                assert "T2" in table["output"]["history_vars"]
        else:
            assert table["history_interval_s"] == 3600.0
            assert table["output"] == {"preset": "minimal"}
    wps = (outdir / plan.domains[0].wps_namelist).read_text()
    assert f"max_dom = {len(plan.domains)}," in wps
    assert "GRAY ZONE:" not in config.read_text()
    assert "LES GRAY-ZONE RECIPE" in config.read_text()


def test_summary_record_prices_the_tree(built):
    dx, _, plan, summary, _ = built
    if summary is None:
        pytest.skip("public entry point returns the plan only")
    assert summary["peak_bytes"] <= summary["target_bytes"]
    assert any("estimated peak VRAM" in note for note in plan.notes)
    assert any("LES gray-zone recipe" in note for note in plan.notes)


def test_main_prints_a_json_summary(tmp_path, capsys):
    from woof.cli import build_parser

    args = build_parser().parse_args([
        "energy", "plan", str(SITES), "--topology", "wrf-nests",
        "--dx-m", "100", "--vram-gib", "24", "--start", START,
        "--hours", "6", "-o", str(tmp_path / "plan")])
    assert args.func(args) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["topology"] == "wrf-nests"
    assert record["dx_chain_m"] == [2500.0, 500.0, 100.0]
    assert record["domains"] == sum(record["domains_by_level"])
    assert record["vram"]["peak_envelope_gib"] <= record["vram"]["fit_target_gib"]
    assert Path(record["plan"]).is_file() and Path(record["config"]).is_file()
    assert load_plan(record["plan"]).hours == 6.0


def test_domain_limit_refusal_names_wrf_tiles(tmp_path):
    with pytest.raises(nests.DomainLimitRefused, match="wrf-tiles") as error:
        nests.build_plan(load_sites(SITES), outdir=tmp_path, dx_m=100.0,
                         start=START, vram_gib=24.0, max_domains=2)
    assert "leaf nest(s)" in str(error.value)
    assert not (tmp_path / "plan.json").exists()


def test_domain_limit_refusal_through_main(tmp_path, capsys):
    from woof.cli import build_parser

    args = build_parser().parse_args([
        "energy", "plan", str(SITES), "--topology", "wrf-nests",
        "--vram-gib", "24", "--start", START, "--max-domains", "2",
        "-o", str(tmp_path)])
    assert args.func(args) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["refused"] == "DomainLimitRefused"
    assert "wrf-tiles" in captured.err


def test_vram_refusal_on_a_tiny_card(tmp_path):
    with pytest.raises(nests.VramRefused, match="--vram-gib"):
        nests.build_plan(load_sites(SITES), outdir=tmp_path, dx_m=100.0,
                         start=START, vram_gib=0.5)
    assert not (tmp_path / "plan.json").exists()


def test_vram_refusal_when_no_leaf_size_fits(tmp_path, monkeypatch):
    # Every candidate prices over the target: the cap shrinks to its floor
    # and the planner refuses rather than emit a tree the card cannot hold.
    monkeypatch.setattr(nests, "_price",
                        lambda exp, *, budget, source: (1 << 50, "forecast", 0))
    with pytest.raises(nests.VramRefused, match="wrf-tiles"):
        nests.build_plan(load_sites(SITES), outdir=tmp_path, dx_m=100.0,
                         start=START, vram_gib=24.0)


def test_default_start_cycle():
    assert nests.default_start_cycle(datetime(2026, 10, 10, 17, 59)) == \
        datetime(2026, 10, 10, 12)
    assert nests.default_start_cycle(datetime(2026, 10, 10, 0, 0)) == \
        datetime(2026, 10, 10, 0)
    aware = datetime(2026, 10, 10, 5, 30, tzinfo=timezone(timedelta(hours=-3)))
    assert nests.default_start_cycle(aware) == datetime(2026, 10, 10, 6)
    assert nests.parse_start(None, now=datetime(2026, 1, 1, 23, 1)) == \
        datetime(2026, 1, 1, 18)
    now = nests.default_start_cycle()
    assert now.hour % 6 == 0 and now.minute == 0
    assert nests.parse_start("2026-10-10T06") == datetime(2026, 10, 10, 6)


def test_start_off_the_hour_is_refused():
    with pytest.raises(nests.PlanRefused, match="whole hour"):
        nests.parse_start("2026-10-10T06:30")
    with pytest.raises(nests.PlanRefused, match="YYYY-MM-DDTHH"):
        nests.parse_start("tomorrow")


def test_chain_ratios():
    assert nests.chain_ratios(100.0, 2500.0) == (5, 5)
    assert nests.chain_ratios(50.0, 1250.0) == (5, 5)
    assert nests.chain_ratios(100.0, 1500.0) == (5, 3)
    assert nests.chain_ratios(100.0, 500.0) == (5,)
    with pytest.raises(nests.PlanRefused, match="3s and 5s"):
        nests.chain_ratios(100.0, 700.0)
    with pytest.raises(nests.PlanRefused, match="at least 3"):
        nests.chain_ratios(100.0, 200.0)


def test_unsupported_ratio_refused_before_emitting(tmp_path):
    with pytest.raises(nests.PlanRefused, match="3s and 5s"):
        nests.build_plan(load_sites(SITES), outdir=tmp_path, dx_m=100.0,
                         parent_dx_m=700.0, start=START, vram_gib=24.0)


def test_hrrr_source_refused(tmp_path):
    with pytest.raises(nests.PlanRefused, match="hrrr"):
        nests.build_plan(load_sites(SITES), outdir=tmp_path, start=START,
                         source="hrrr", vram_gib=24.0)


def test_nz_below_stencil_refused(tmp_path):
    with pytest.raises(nests.PlanRefused, match="--nz"):
        nests.build_plan(load_sites(SITES), outdir=tmp_path, start=START,
                         vram_gib=24.0, nz=3)


def test_cover_that_drops_a_site_is_refused(tmp_path, monkeypatch):
    def dropping(x, y, **kwargs):
        rects = _fake_cover(x, y, **kwargs)
        first = rects[0]
        rects[0] = geometry.Rect(first.x_min, first.y_min, first.x_max,
                                 first.y_max, first.nx, first.ny,
                                 first.members[1:])
        return rects

    monkeypatch.setattr(geometry, "cover_with_rectangles", dropping)
    with pytest.raises(nests.PlanRefused, match="exactly one leaf"):
        nests.build_plan(load_sites(SITES), outdir=tmp_path, start=START,
                         vram_gib=24.0)


def test_leaf_output_prefers_the_energy_preset(monkeypatch):
    from woof.io import history_selection

    notes: list[str] = []
    output, _ = nests._leaf_output(notes)
    if "energy" not in history_selection.HISTORY_PRESETS:
        assert "history_vars" in output
        monkeypatch.setitem(history_selection.HISTORY_PRESETS, "energy",
                            frozenset({"U", "V"}))
    output, words = nests._leaf_output([])
    assert output == {"preset": "energy"} and "energy" in words


def test_layout_respects_parent_clearance():
    x = np.array([0.0, 30_000.0, 31_000.0, -40_000.0])
    y = np.array([0.0, 5_000.0, -2_000.0, 10_000.0])
    layout = nests.build_layout(x, y, dx_m=100.0, ratios=(5, 5),
                                margin_m=2000.0, cap=200, min_axis=11,
                                clearance_rows=10)
    owned = sorted(int(i) for leaf in layout.leaves for i in leaf.members)
    assert owned == [0, 1, 2, 3]
    for box in layout.domains[1:]:
        ratio = layout.ratios[box.level - 1]
        parent = box.parent
        assert box.lo_x % ratio == 0 and box.hi_x % ratio == 0
        assert box.lo_x // ratio - parent.lo_x >= 10
        assert parent.hi_x - box.hi_x // ratio >= 10
        assert box.lo_y // ratio - parent.lo_y >= 10
        assert parent.hi_y - box.hi_y // ratio >= 10
    root = layout.levels[0][0]
    assert root.lo_x == -root.hi_x and root.lo_y == -root.hi_y
    for leaf in layout.leaves:
        xs = x[leaf.members]
        assert leaf.lo_x * 100.0 <= xs.min() - 2000.0
        assert leaf.hi_x * 100.0 >= xs.max() + 2000.0


def test_overlapping_siblings_are_merged():
    a = nests._Box(level=2, lo_x=0, hi_x=50, lo_y=0, hi_y=50,
                   members=np.array([0]))
    b = nests._Box(level=2, lo_x=40, hi_x=90, lo_y=10, hi_y=60,
                   members=np.array([1]))
    c = nests._Box(level=2, lo_x=90, hi_x=120, lo_y=0, hi_y=50,
                   members=np.array([2]))
    merged = nests._merge_overlapping([a, b, c])
    assert len(merged) == 2  # c only touches b's edge
    union = next(box for box in merged if box.members.size == 2)
    assert (union.lo_x, union.hi_x, union.lo_y, union.hi_y) == (0, 90, 0, 60)
    assert sorted(union.members.tolist()) == [0, 1]


def test_layout_leaves_never_overlap():
    rng = np.random.default_rng(7)
    x = rng.uniform(-30_000.0, 30_000.0, 60)
    y = rng.uniform(-10_000.0, 10_000.0, 60)
    layout = nests.build_layout(x, y, dx_m=100.0, ratios=(5, 5),
                                margin_m=2000.0, cap=120, min_axis=11,
                                clearance_rows=10)
    for level in layout.levels:
        for i, a in enumerate(level):
            for b in level[i + 1:]:
                assert not nests._overlaps(a, b)


@pytest.mark.parametrize("corridor", [0.0, -1.0, float("nan")])
def test_bad_corridor_refused(tmp_path, corridor):
    with pytest.raises(nests.PlanRefused, match="--corridor-km"):
        nests.build_plan(load_sites(SITES), outdir=tmp_path, start=START,
                         vram_gib=24.0, corridor_km=corridor)
