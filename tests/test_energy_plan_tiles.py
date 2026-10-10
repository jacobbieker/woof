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
        assert spec["physics_overrides"]["applied"] is True
        assert spec["physics_overrides"]["recipe"] == tiles.LES_PHYSICS_RECIPE
        assert spec["physics_overrides"]["verified_keys"] == sorted(
            tiles.LES_PHYSICS_RECIPE)
        assert spec["history"] == {**spec["history"], "preset": "energy",
                                   "applied": True}
        assert spec["child_levels"]["applied"] is True
        assert spec["child_config"]["path"] == f"tiles/{domain.domain_id}.toml"
        assert domain.extra["child_config"] == spec["child_config"]["path"]
    assert plan.sites_ref["sha256"]
    assert (outdir / plan.sites_ref["path"]).resolve() == SITES_WALES.resolve()
    assert record["parent"]["nx"] == parent.extra["nx"]
    assert not any("NOT applied" in note for note in plan.notes)
    assert any(note.startswith("LES gray-zone closure applied")
               for note in plan.notes)
    assert any(note.startswith("history preset 'energy' applied")
               for note in plan.notes)


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
        # The --child-config route: explicit placement, no --point
        # derivation flags (the vertical is in the TOML).
        assert args.child_config == Path(spec["child_config"]["path"])
        assert args.child_config_sha256 == spec["child_config"]["sha256"]
        assert args.i_parent_start == spec["i_parent_start"]
        assert args.j_parent_start == spec["j_parent_start"]
        assert args.point is None and args.child_size is None
        assert args.child_levels is None
        assert args.hours is None and args.output_interval_seconds is None
        assert args.parent_restart == "latest"
        assert args.parent_domain == 1
        assert args.max_boundary_interval_seconds == 900.0
        assert args.card == "24gb"
        assert args.render_products == "none"


def test_tile_centre_lands_on_the_planned_parent_cell(wales):
    """The recorded centre resolves to the planned placement through
    downscale's own nearest-point search and centring, so the explicit
    --i/--j-parent-start are the start --point would have reached."""

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


# --------------------------------------------------------------------------
# the --child-config TOMLs: LES closure, history preset, sha binding


def _leaf_specs(outdir):
    plan = load_plan(outdir / "plan.json")
    for domain in plan.domains:
        if domain.role == "child":
            yield domain, json.loads((outdir / domain.config).read_text())


def test_les_leaf_tomls_carry_the_recipe_and_energy_preset(wales):
    import tomllib

    _, _, outdir = wales
    for domain, spec in _leaf_specs(outdir):
        path = outdir / spec["child_config"]["path"]
        raw = tomllib.loads(path.read_text())
        for key, value in tiles.LES_PHYSICS_RECIPE.items():
            assert raw["run"][key] == value, key
        assert raw["output"] == {"preset": "energy"}
        assert raw["grid"]["nx"] == spec["nx"] == domain.extra["nx"]
        assert raw["grid"]["nz"] == 60 == len(raw["run"]["eta_levels"]) - 1
        assert raw["run"]["specified"] is True
        assert raw["run"]["nested"] is False
        assert raw["static"] == {"terrain": "own"}


def test_leaf_tomls_pass_the_child_config_door(wales):
    """The door's own loader reads every planned key back, and the
    RunConfig wraps as an experiment (load_experiment itself reads only
    the experiment schema and refuses a legacy [grid]/[run] file)."""

    from woof.config import load_history_selection
    from woof.experiment import experiment_from_run_config, load_experiment
    from woof.offline_child import (require_offline_child_root_forcing,
                                    resolve_child_run_config)

    _, _, outdir = wales
    for _, spec in _leaf_specs(outdir):
        path = outdir / spec["child_config"]["path"]
        cfg = resolve_child_run_config(path)
        require_offline_child_root_forcing(cfg)
        for key, value in tiles.LES_PHYSICS_RECIPE.items():
            assert getattr(cfg, key) == value, key
        assert (cfg.nx, cfg.ny, cfg.grid_id) == (spec["nx"], spec["ny"], 2)
        assert cfg.dx == pytest.approx(spec["dx_m"])
        assert load_history_selection(path).preset == "energy"
        exp = experiment_from_run_config(
            cfg, datetime(2026, 10, 1, tzinfo=timezone.utc))
        assert exp.root.run.km_opt == 3
        with pytest.raises(ValueError):
            load_experiment(path)


def test_child_config_sha_recorded_and_verified(wales, tmp_path):
    import hashlib
    import shutil

    from woof.downscale import _verify_child_config_hash
    from woof.offline_child import OfflineChildContractError

    _, _, outdir = wales
    for domain, spec in _leaf_specs(outdir):
        path = outdir / spec["child_config"]["path"]
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert spec["child_config"]["sha256"] == sha
        assert domain.extra["child_config_sha256"] == sha
        args = domain.extra["downscale_args"]
        assert args[args.index("--child-config-sha256") + 1] == sha
        assert _verify_child_config_hash(path, sha) == sha
        edited = tmp_path / path.name
        shutil.copy(path, edited)
        edited.write_text(edited.read_text().replace("km_opt = 3",
                                                     "km_opt = 4"))
        with pytest.raises(OfflineChildContractError, match="changed"):
            _verify_child_config_hash(edited, sha)


def test_leaf_args_drop_point_flags_and_keep_out_under_run_dir(wales):
    _, _, outdir = wales
    for domain, spec in _leaf_specs(outdir):
        args = domain.extra["downscale_args"]
        for flag in ("--point", "--child-size", "--child-levels", "--hours"):
            assert flag not in args
        assert args[args.index("--out") + 1] == domain.run_dir
        assert spec["command_template"][:3] == ["woof", "downscale",
                                                "{parent_frames}"]
        assert spec["command_template"][3:] == args


def test_intermediate_tiles_keep_parent_physics_and_full_history(tmp_path,
                                                                  cover,
                                                                  monkeypatch):
    """A two-step chain: the 500 m intermediate tiles force the 100 m
    leaves, so they keep the parent's physics and the full inventory."""

    import tomllib

    from woof.config import load_history_selection

    real = tiles._emit_parent

    def refuse_500(**kwargs):
        if kwargs["parent_dx_m"] == 500.0:
            raise tiles._ParentDoesNotFit("d01 EXCEEDS the budget")
        return real(**kwargs)

    monkeypatch.setattr(tiles, "_emit_parent", refuse_500)
    outdir = tmp_path / "plan"
    plan = tiles.build_plan(load_sites(SITES_WALES), outdir=outdir,
                            dx_m=100.0, start=START, hours=6)
    parent_cfg = tomllib.loads(
        (outdir / plan.domain("parent").config).read_text())["shared"]
    inter = [d for d in plan.domains if d.domain_id.startswith("inter-")]
    leaves = [d for d in plan.domains if d.role == "child"]
    assert inter and leaves
    for domain in inter:
        spec = json.loads((outdir / domain.config).read_text())
        path = outdir / spec["child_config"]["path"]
        raw = tomllib.loads(path.read_text())
        assert "output" not in raw
        assert load_history_selection(path).preset == "full"
        assert raw["run"]["km_opt"] == parent_cfg["km_opt"] != 3
        assert raw["run"]["bl_pbl_physics"] == parent_cfg["bl_pbl_physics"]
        assert spec["physics_overrides"]["applied"] is None
        assert spec["physics_overrides"]["recipe"] == {}
        assert spec["history"]["preset"] == "full"
        assert spec["history"]["applied"] is None
    for domain in leaves:
        spec = json.loads((outdir / domain.config).read_text())
        raw = tomllib.loads((outdir / spec["child_config"]["path"])
                            .read_text())
        assert raw["run"]["km_opt"] == 3 and raw["output"]["preset"] == "energy"
        assert raw["run"]["grid_id"] == 3
        args = domain.extra["downscale_args"]
        assert args[args.index("--parent-domain") + 1] == "2"
    assert any("intermediate tiles write the full inventory" in note
               for note in plan.notes)


def test_coarse_leaf_tiles_get_preset_but_no_les(tmp_path, cover):
    import tomllib

    outdir = tmp_path / "plan"
    plan = tiles.build_plan(load_sites(SITES_WALES), outdir=outdir,
                            dx_m=300.0, start=START, hours=6)
    leaves = [d for d in plan.domains if d.role == "child"]
    assert leaves
    for domain in leaves:
        spec = json.loads((outdir / domain.config).read_text())
        raw = tomllib.loads((outdir / spec["child_config"]["path"])
                            .read_text())
        assert raw["run"]["km_opt"] != 3
        assert raw["output"] == {"preset": "energy"}
        assert spec["physics_overrides"]["applied"] is None
        assert spec["history"]["applied"] is True
    assert any("coarser than the 250 m LES regime" in note
               for note in plan.notes)


def test_recipe_key_downscale_cannot_carry_refuses_the_plan(tmp_path, cover,
                                                           monkeypatch):
    monkeypatch.setattr(tiles, "LES_PHYSICS_RECIPE",
                        {**tiles.LES_PHYSICS_RECIPE, "not_a_field": 1})
    with pytest.raises(tiles.TilePlanRefusal, match="not_a_field"):
        tiles.build_plan(load_sites(SITES_WALES), outdir=tmp_path / "plan",
                         dx_m=100.0, start=START, hours=6)
    assert not (tmp_path / "plan" / "plan.json").exists()


def test_invalid_recipe_value_refuses_the_plan(tmp_path, cover, monkeypatch):
    monkeypatch.setattr(tiles, "LES_PHYSICS_RECIPE",
                        {**tiles.LES_PHYSICS_RECIPE, "km_opt": 99})
    with pytest.raises(tiles.TilePlanRefusal, match="does not validate"):
        tiles.build_plan(load_sites(SITES_WALES), outdir=tmp_path / "plan",
                         dx_m=100.0, start=START, hours=6)


def test_override_lost_on_read_back_refuses_the_plan(tmp_path, cover,
                                                     monkeypatch):
    """A loader that silently changed a key (here mix_isotropic) must not
    leave a tile spec claiming the closure was applied."""

    import dataclasses

    import woof.offline_child as offline_child

    real = offline_child.resolve_child_run_config

    def drops_mix_isotropic(path, **kwargs):
        return dataclasses.replace(real(path, **kwargs), mix_isotropic=0)

    monkeypatch.setattr(offline_child, "resolve_child_run_config",
                        drops_mix_isotropic)
    with pytest.raises(tiles.TilePlanRefusal,
                       match="mix_isotropic = 1 planned, 0 read back"):
        tiles.build_plan(load_sites(SITES_WALES), outdir=tmp_path / "plan",
                         dx_m=100.0, start=START, hours=6)
    assert not (tmp_path / "plan" / "plan.json").exists()


def test_preset_lost_on_read_back_refuses_the_plan(tmp_path, cover,
                                                   monkeypatch):
    real = tiles._child_config_text

    def no_output(tile):
        return real(tile).replace('[output]\npreset = "energy"\n', "")

    monkeypatch.setattr(tiles, "_child_config_text", no_output)
    with pytest.raises(tiles.TilePlanRefusal,
                       match="preset = 'energy' planned, 'full' read back"):
        tiles.build_plan(load_sites(SITES_WALES), outdir=tmp_path / "plan",
                         dx_m=100.0, start=START, hours=6)


def test_auto_epssm_parent_hands_the_label_down(wales):
    """A parent whose epssm was the model's choice hands it down as
    ``epssm = { auto = VALUE }``, exactly as --point does from the
    checkpoint label."""

    import tomllib

    _, _, outdir = wales
    _, spec = next(_leaf_specs(outdir))
    assert spec["child_config"]["epssm_auto_label"] is False
    raw = tomllib.loads((outdir / spec["child_config"]["path"]).read_text())
    assert isinstance(raw["run"]["epssm"], float)
    tile = tiles._Tile(
        domain_id="tile-x", role="child", level=1, ratio=5, dx=100.0,
        nx=spec["nx"], ny=spec["ny"], i_parent_start=1, j_parent_start=1,
        i0=0, j0=0, centre_lat=0.0, centre_lon=0.0, grid=None,
        parent_id="parent", parent_wrf_grid_id=1, wrf_grid_id=2,
        sites=np.arange(1), run_config={**raw["grid"], **raw["run"]},
        peak_envelope_gib=0.0, levels=None, parent_auto_epssm=True)
    labelled = tomllib.loads(tiles._child_config_text(tile))
    assert labelled["run"]["epssm"] == {"auto": raw["run"]["epssm"]}


def test_epssm_label_lost_on_read_back_refuses_the_plan(tmp_path, cover,
                                                        monkeypatch):
    import woof.offline_child as offline_child

    monkeypatch.setattr(offline_child, "child_epssm_is_auto",
                        lambda path: True)
    with pytest.raises(tiles.TilePlanRefusal,
                       match="epssm auto label = False planned, True"):
        tiles.build_plan(load_sites(SITES_WALES), outdir=tmp_path / "plan",
                         dx_m=100.0, start=START, hours=6)


def test_tile_toml_header_does_not_claim_restart_evidence(wales):
    _, _, outdir = wales
    for _, spec in _leaf_specs(outdir):
        text = (outdir / spec["child_config"]["path"]).read_text()
        assert "restart evidence" not in text
        assert "km_opt = 3" in text.split("[grid]")[0]


@pytest.mark.parametrize("parent_cu, dx, leaf, expected", [
    # LES leaf under a cumulus-free parent: the recipe, nothing else.
    (0, 100.0, True, dict(tiles.LES_PHYSICS_RECIPE)),
    # Coarse leaf and intermediate under a cumulus-free parent: nothing.
    (0, 300.0, True, {}),
    (0, 500.0, False, {}),
    # Kain-Fritsch parent: retired below the 4 km bound on every tile.
    (1, 2500.0, False, {"cu_physics": 0, "cudt_minutes": 0.0}),
    (1, 300.0, True, {"cu_physics": 0, "cudt_minutes": 0.0}),
    (1, 100.0, True, {**tiles.LES_PHYSICS_RECIPE, "cudt_minutes": 0.0}),
    # ... and kept above it.
    (1, 5000.0, False, {}),
    # Grell-Freitas parent: its family keys go back to their defaults.
    (3, 500.0, False, {"cu_physics": 0, "cudt_minutes": 0.0,
                       "clos_choice": 0, "ishallow": 0}),
])
def test_tile_overrides(parent_cu, dx, leaf, expected):
    assert tiles.tile_overrides({"cu_physics": parent_cu}, child_dx_m=dx,
                                leaf=leaf) == expected


def test_grell_parent_les_leaf_validates(wales):
    """A Grell-Freitas parent with non-default family keys: the LES leaf's
    cu_physics = 0 would be refused by validate_run_config unless the
    family keys are reset with it."""

    import dataclasses

    from woof.experiment import load_experiment

    _, _, outdir = wales
    plan = load_plan(outdir / "plan.json")
    exp = load_experiment(outdir / plan.domain("parent").config)
    parent = {**dataclasses.asdict(exp.root.run), "cu_physics": 3,
              "cudt_minutes": 0.0, "clos_choice": 1, "ishallow": 1}
    overrides = tiles.tile_overrides(parent, child_dx_m=100.0, leaf=True)
    merged = tiles._Pricer(24.0).derive(
        parent, parent_dx=500.0, ratio=5, nx=100, ny=100,
        run_seconds=21600.0, output_interval_s=900.0, levels=None,
        centre_lat=51.7, overrides=overrides)
    assert (merged["cu_physics"], merged["clos_choice"],
            merged["ishallow"]) == (0, 0, 0)
    assert merged["km_opt"] == 3
