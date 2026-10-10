"""``woof hex adapt`` and the adaptive cycle: criteria, raster, hysteresis.

Card-free and generator-free: the history is a tiny MPAS-like netCDF built
here, and the adaptive chain runs against a fake runner that writes the
files each stage would have written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import adapt  # noqa: E402
from woof.hex.cycle import chain  # noqa: E402
from woof.hex.cycle.errors import CycleRefusal  # noqa: E402

netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("scipy")

REPO = Path(__file__).resolve().parents[3]
SITES = REPO / "tests" / "fixtures" / "energy" / "sites_wales.json"
LEVELS = 4


# ---------------------------------------------------------------------------
# a synthetic history
# ---------------------------------------------------------------------------
def _cells(step: float = 0.06):
    lat = np.arange(50.6, 52.8, step)
    lon = np.arange(-5.8, -1.6, step * 1.6)
    grid_lat, grid_lon = np.meshgrid(lat, lon, indexing="ij")
    rng = np.random.default_rng(3)
    grid_lat = grid_lat + rng.uniform(-0.15, 0.15, grid_lat.shape) * step
    grid_lon = grid_lon + rng.uniform(-0.15, 0.15, grid_lon.shape) * step
    return grid_lat.ravel(), grid_lon.ravel()


def write_frame(path: Path, lat, lon, *, jet=(51.7, -4.6), cold=(51.8, -3.7),
                rain_mm: float = 0.0, omit: tuple[str, ...] = ()) -> Path:
    n = lat.size
    d_jet = np.hypot(lat - jet[0], (lon - jet[1]) * 0.62)
    speed = 8.0 + 22.0 * np.exp(-(d_jet / 0.25) ** 2)
    d_cold = np.hypot(lat - cold[0], (lon - cold[1]) * 0.62)
    theta0 = 285.0 - 14.0 * np.exp(-(d_cold / 0.3) ** 2)
    theta = np.repeat(theta0[:, None], LEVELS, 1) + 0.5 * np.arange(LEVELS)[None, :]
    pressure = np.repeat(np.linspace(1.0e5, 0.95e5, LEVELS)[None, :], n, 0)
    qc = np.repeat((3.0e-4 * np.exp(-(d_cold / 0.35) ** 2))[:, None], LEVELS, 1)
    fields2 = {"u10": speed * 0.8, "v10": speed * 0.6, "t2": theta0 - 1.0,
               "rainnc": np.full(n, rain_mm), "rainc": np.zeros(n)}
    fields3 = {"u_zonal": np.repeat((speed * 0.8)[:, None], LEVELS, 1),
               "v_meridional": np.repeat((speed * 0.6)[:, None], LEVELS, 1),
               "theta": theta, "pressure": pressure, "qc": qc,
               "qr": np.zeros((n, LEVELS))}
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(path), "w", format="NETCDF4_CLASSIC") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("nCells", n)
        ds.createDimension("nVertLevels", LEVELS)
        ds.createVariable("latCell", "f4", ("nCells",))[:] = np.radians(lat)
        ds.createVariable("lonCell", "f4", ("nCells",))[:] = np.radians(lon)
        for name, value in fields2.items():
            if name not in omit:
                ds.createVariable(name, "f4", ("Time", "nCells"))[:] = value[None]
        for name, value in fields3.items():
            if name not in omit:
                ds.createVariable(name, "f4", ("Time", "nCells", "nVertLevels"))[:] = value[None]
    return path


def write_grid(path: Path, lat, lon) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(path), "w", format="NETCDF4_CLASSIC") as ds:
        ds.createDimension("nCells", lat.size)
        ds.createVariable("latCell", "f8", ("nCells",))[:] = np.radians(lat)
        ds.createVariable("lonCell", "f8", ("nCells",))[:] = np.radians(lon)
    return path


def frames(directory: Path, start: datetime, count: int = 3, **kwargs) -> list[Path]:
    lat, lon = _cells()
    out = []
    for k in range(count):
        moment = start + timedelta(hours=k)
        out.append(write_frame(
            directory / f"cuda-history.{moment:%Y-%m-%d_%H.%M.%S}.nc", lat, lon,
            rain_mm=2.0 * k, **kwargs))
    return out


def _request(tmp_path: Path, history, **overrides) -> adapt.AdaptRequest:
    values = dict(
        history=[str(item) for item in history],
        criteria=["wind,icing,assets"], out=tmp_path / "next-spec",
        fine_km=0.5, background_km=3.0, dt_seconds=3.0, experimental_dt=True,
        sites=SITES, raster_km=2.0,
    )
    values.update(overrides)
    return adapt.AdaptRequest(**values)


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------
def test_criteria_read_the_fields_they_name(tmp_path):
    paths = frames(tmp_path / "h", datetime(2026, 10, 10, 0))
    (group,) = adapt.group_frames(paths)
    fields, notes = adapt.criterion_fields(
        group, ["wind", "icing", "precip", "theta-gradient"], adapt.FieldOptions())
    assert notes["wind"]["source"] == "u10/v10"
    # hypot(0.8 s, 0.6 s) == s, and s peaks near 30 m/s on the jet.
    assert 28.0 < float(np.nanmax(fields["wind"])) <= 30.0
    # Icing only where T < 0 C: the cold core.  Away from it theta is 285 K.
    lat = np.degrees(group.lat)
    lon = np.degrees(group.lon)
    far = np.hypot(lat - 51.8, (lon + 3.7) * 0.62) > 1.0
    assert np.all(fields["icing"][far] == 0.0)
    assert float(fields["icing"].max()) > 1.0e-4
    # 2 mm per hour of accumulation between consecutive frames.
    assert np.allclose(fields["precip"], 2.0)
    assert float(np.nanmax(fields["theta-gradient"])) > 0.0


def test_a_criterion_without_its_field_refuses_by_name(tmp_path):
    lat, lon = _cells()
    path = write_frame(tmp_path / "cuda-history.2026-10-10_00.00.00.nc", lat, lon,
                       omit=("qc",))
    (group,) = adapt.group_frames([path])
    with pytest.raises(adapt.AdaptRefusal, match="criterion icing needs"):
        adapt.criterion_fields(group, ["icing"], adapt.FieldOptions())


def test_a_mesh_with_one_frame_forms_no_rate(tmp_path):
    paths = frames(tmp_path / "h", datetime(2026, 10, 10, 0), count=1)
    (group,) = adapt.group_frames(paths)
    fields, notes = adapt.criterion_fields(group, ["precip"], adapt.FieldOptions())
    assert np.all(np.isnan(fields["precip"]))
    assert "skipped" in notes["precip"]


def test_unknown_criteria_and_stray_tuning_refuse():
    with pytest.raises(adapt.AdaptRefusal, match="not a criterion"):
        adapt.resolve_criteria(["wind,sunspots"])
    assert adapt.resolve_criteria(["gradients"]) == ("theta-gradient", "wind-gradient")
    with pytest.raises(adapt.AdaptRefusal, match="did not select"):
        adapt.criterion_ramps(("wind",), thresholds=["icing=1:2"])
    with pytest.raises(adapt.AdaptRefusal, match="LO < HI"):
        adapt.criterion_ramps(("wind",), thresholds=["wind=5:5"])


def test_multipoint_assets_are_points_not_a_line():
    lines = adapt._geometry_lines({"type": "MultiPoint",
                                   "coordinates": [[-4.0, 51.5], [-3.0, 51.9]]})
    assert lines == [[(-4.0, 51.5)], [(-3.0, 51.9)]]


def test_precip_skips_a_mesh_with_one_frame_when_another_has_two(tmp_path):
    two = frames(tmp_path / "a", datetime(2026, 10, 10, 0), count=2)
    lat, lon = _cells(step=0.09)
    one = [write_frame(tmp_path / "b" / "cuda-history.2026-10-10_01.00.00.nc", lat, lon)]
    plan = adapt.build_plan(_request(tmp_path, two + one, criteria=["precip,assets"]))
    assert plan["criteria"]["precip"]["field_max"] == pytest.approx(2.0)
    with pytest.raises(adapt.AdaptRefusal, match="RATE"):
        adapt.build_plan(_request(tmp_path / "x", one, criteria=["precip"]))


def test_over_resolution_outside_the_new_raster_counts():
    """A fine corridor the new raster no longer reaches is wasted cells."""

    lat = np.linspace(51.0, 52.0, 20)
    lon = np.linspace(-5.0, -3.0, 30)
    current = np.full((20, 30), 3.0)
    current[:, :10] = 0.5                  # the old corridor, west
    new_lon = lon[15:]                     # the new raster covers the east only
    target = np.full((20, 15), 3.0)
    numbers = adapt.compare_targets((lat, lon, current), (lat, new_lon, target),
                                    tolerance_ratio=1.25, background_km=3.0)
    assert numbers["over_resolved_cell_fraction"] > 0.9


def test_assets_demand_is_full_on_the_line_and_zero_far_away():
    lat = np.linspace(51.0, 52.5, 40)
    lon = np.linspace(-5.5, -3.0, 50)
    distance = adapt.asset_distance_km(np.array([51.7]), np.array([-4.2]), lat, lon)
    demand = adapt.ramp(distance, 3.0, 6.0, inverted=True)
    near = np.unravel_index(np.argmin(distance), distance.shape)
    assert demand[near] == 1.0
    assert demand[0, 0] == 0.0


# ---------------------------------------------------------------------------
# the raster contract and the limiter
# ---------------------------------------------------------------------------
def test_the_density_raster_meets_the_contract(tmp_path):
    lat = np.linspace(50.0, 51.0, 11)
    lon = np.linspace(-4.0, -3.0, 21)
    spacing = np.full((11, 21), 3.0)
    spacing[5, 10] = 0.25
    adapt.write_density_raster(tmp_path / "density.nc", lat, lon, spacing)
    with netCDF4.Dataset(str(tmp_path / "density.nc")) as ds:
        assert ds.schema == "woof-hex.density.v1"
        assert ds.min_spacing_km == 0.25
        assert ds.variables["spacing_km"].dtype == np.float64
        assert ds.variables["spacing_km"].dimensions == ("lat", "lon")
        assert ds.variables["lat"].dimensions == ("lat",)
        assert ds.variables["lat"].units == "degrees_north"
        assert ds.variables["lon"].units == "degrees_east"
    back = adapt.read_density_raster(tmp_path / "density.nc")
    assert np.array_equal(back[2], spacing)


def test_a_raster_off_contract_refuses(tmp_path):
    lat = np.linspace(50.0, 51.0, 3)
    with pytest.raises(adapt.AdaptRefusal, match="ascending"):
        adapt.write_density_raster(tmp_path / "d.nc", lat[::-1], lat, np.ones((3, 3)))
    adapt.write_density_raster(tmp_path / "d.nc", lat, lat, np.ones((3, 3)))
    with netCDF4.Dataset(str(tmp_path / "d.nc"), "a") as ds:
        ds.schema = "something-else"
    with pytest.raises(adapt.AdaptRefusal, match="declares schema"):
        adapt.read_density_raster(tmp_path / "d.nc")


def test_the_limiter_bounds_the_gradient_and_only_refines():
    lat = np.linspace(51.0, 52.0, 60)
    lon = np.linspace(-5.0, -3.0, 80)
    spacing = np.full((60, 80), 6.0)
    spacing[30, 40] = 0.1
    spacing[10:12, 5:60] = 0.5
    limited, sweeps = adapt.limit_gradient(spacing, lat, lon, 0.0306)
    assert sweeps >= 1
    assert np.all(limited <= spacing)
    assert limited[30, 40] == 0.1
    assert adapt.delivered_gradient(limited, lat, lon) <= 0.0306 * (1 + 1e-9)


# ---------------------------------------------------------------------------
# hysteresis
# ---------------------------------------------------------------------------
POLICY = adapt.RemeshPolicy()


def test_no_state_generates():
    assert adapt.decide(None, None, policy=POLICY, domain_inside=None)["action"] == "generate"


@pytest.mark.parametrize(
    "held, under, over, inside, expected",
    [
        (5, 0.01, 0.05, True, "keep"),
        (5, 0.10, 0.05, True, "remesh"),
        (5, 0.01, 0.50, True, "remesh"),
        (1, 0.10, 0.50, True, "keep"),      # dwell protects
        (1, 0.25, 0.00, True, "remesh"),    # dwell does not hide weather
        (1, 0.00, 0.00, False, "remesh"),   # domain left the mesh
    ],
)
def test_hysteresis_keep_and_remesh(held, under, over, inside, expected):
    state = {"mesh": {"cycles_held": held}}
    comparison = {"under_resolved_cell_fraction": under,
                  "over_resolved_cell_fraction": over}
    decision = adapt.decide(state, comparison, policy=POLICY, domain_inside=inside)
    assert decision["action"] == expected, decision["reason"]


def test_a_policy_that_contradicts_itself_refuses():
    with pytest.raises(adapt.AdaptRefusal):
        adapt.RemeshPolicy(remesh_under_fraction=0.3, force_under_fraction=0.1).validate()


def test_compare_targets_weights_by_cells():
    lat = np.linspace(51.0, 52.0, 20)
    lon = np.linspace(-5.0, -3.0, 30)
    current = np.full((20, 30), 3.0)
    target = current.copy()
    target[10, 15] = 0.5   # one pixel, but 36x the cells per km2
    numbers = adapt.compare_targets((lat, lon, current), (lat, lon, target),
                                    tolerance_ratio=1.25, background_km=3.0)
    assert numbers["under_resolved_cell_fraction"] > 1.0 / 600.0 * 10
    assert numbers["over_resolved_cell_fraction"] == 0.0


def test_two_cycles_keep_then_remesh_when_weather_moves(tmp_path):
    first = frames(tmp_path / "c1", datetime(2026, 10, 10, 0))
    plan1 = adapt.build_plan(_request(tmp_path / "1", first))
    assert plan1["decision"]["action"] == "generate"
    assert [c["stage"] for c in plan1["commands"]][:1] == ["mesh"]
    assert plan1["timestep_evidence"] == "experimental-unanchored"
    forecast = next(c for c in plan1["commands"] if c["stage"] == "forecast")
    assert "--experimental-dt" in forecast["argv"]
    assert forecast["env"] == {"WOOF_HEX_MESH_ROWS": plan1["mesh"]["rows"]}

    # The planned state is PROPOSED until the plan's stages have run.
    with pytest.raises(adapt.AdaptRefusal, match="PROPOSED"):
        adapt.load_state(Path(plan1["paths"]["proposed_state"]))
    adapt.commit_state(tmp_path / "1" / "next-spec")
    same = frames(tmp_path / "c2", datetime(2026, 10, 10, 6))
    plan2 = adapt.build_plan(_request(
        tmp_path / "2", same, state=Path(plan1["paths"]["state"]),
        policy=adapt.RemeshPolicy(minimum_dwell_cycles=0)))
    assert plan2["decision"]["action"] == "keep", plan2["decision"]
    assert "mesh" not in [c["stage"] for c in plan2["commands"]]
    assert plan2["mesh"]["name"] == plan1["mesh"]["name"]
    assert plan2["mesh"]["cycles_held"] == 2

    adapt.commit_state(tmp_path / "2" / "next-spec")
    moved = frames(tmp_path / "c3", datetime(2026, 10, 10, 12),
                   jet=(52.0, -3.3), cold=(51.4, -5.0))
    plan3 = adapt.build_plan(_request(
        tmp_path / "3", moved, state=Path(plan2["paths"]["state"]),
        policy=adapt.RemeshPolicy(minimum_dwell_cycles=0)))
    assert plan3["decision"]["action"] == "remesh", plan3["decision"]
    assert plan3["decision"]["under_resolved_cell_fraction"] > 0.05
    assert plan3["mesh"]["name"] != plan1["mesh"]["name"]


def test_a_state_whose_raster_moved_refuses(tmp_path):
    plan = adapt.build_plan(_request(tmp_path / "1", frames(tmp_path / "h", datetime(2026, 10, 10))))
    adapt.commit_state(tmp_path / "1" / "next-spec")
    # Reusing the state's own folder as -o would overwrite its raster.
    with pytest.raises(adapt.AdaptRefusal, match="fresh -o"):
        adapt.build_plan(_request(tmp_path / "1", frames(tmp_path / "h1", datetime(2026, 10, 10, 6)),
                                  state=Path(plan["paths"]["state"])))
    Path(plan["raster"]["path"]).write_bytes(b"other bytes")
    with pytest.raises(adapt.AdaptRefusal, match="changed since"):
        adapt.build_plan(_request(tmp_path / "2", frames(tmp_path / "h2", datetime(2026, 10, 10, 6)),
                                  state=Path(plan["paths"]["state"])))


# ---------------------------------------------------------------------------
# timestep
# ---------------------------------------------------------------------------
def test_a_sub_anchor_dt_needs_the_experimental_lane(tmp_path):
    with pytest.raises(adapt.AdaptRefusal, match="--experimental-dt"):
        adapt.timestep_preflight(3.0, 0.5, experimental=False, hours=6.0,
                                 history_every_minutes=30)
    receipt = adapt.timestep_preflight(3.0, 0.5, experimental=True, hours=6.0,
                                       history_every_minutes=30)
    assert receipt["timestep_evidence"] == "experimental-unanchored"


def test_the_courant_and_clock_checks_stay_enforced():
    with pytest.raises(adapt.AdaptRefusal, match="Courant"):
        adapt.timestep_preflight(4.0, 0.5, experimental=True, hours=6.0,
                                 history_every_minutes=30)
    with pytest.raises(adapt.AdaptRefusal, match="whole number"):
        adapt.timestep_preflight(1.7, 0.5, experimental=True, hours=6.0,
                                 history_every_minutes=30)


# ---------------------------------------------------------------------------
# the plan's argv against this build's parsers
# ---------------------------------------------------------------------------
def test_planned_argv_parse_with_the_real_parsers(tmp_path):
    history = frames(tmp_path / "h", datetime(2026, 10, 10, 0))
    lat, lon = _cells()
    grid = write_grid(tmp_path / "coarse.grid.nc", lat, lon)
    plan = adapt.build_plan(_request(
        tmp_path / "p", history, from_grid=grid, met_dir=tmp_path,
        nfglevels=34, extrap_airtemp="lapse-rate", use_spechumd="no",
        vertical_spec=tmp_path / "v.json"))
    stages = [c["stage"] for c in plan["commands"]]
    assert stages == ["mesh", "static-highres", "register-parent", "vertical",
                      "cull", "register", "remap", "lbc", "forecast"]
    from woof.hex.cli import build_parser

    subcommands = next(a for a in build_parser()._actions
                       if isinstance(a, argparse._SubParsersAction)).choices
    checked = 0
    for command in plan["commands"]:
        argv = command["argv"]
        status = adapt.check_argv(argv)
        assert status == command["contract"]
        if argv[0] == "hex" and argv[1] not in subcommands:
            assert status["status"] == "command-not-in-this-build"
            continue  # owned by another unit; not in this branch yet
        if status["status"] == "flags-not-in-this-build":
            continue  # a flag another unit adds (--density-raster, --parent-vertical, ...)
        assert status["status"] == "parsed", (command["stage"], status)
        checked += 1
    assert checked >= 1  # at least woof hex lbc parses here today


def test_check_argv_does_not_accept_a_prefix_as_a_flag():
    """argparse would take --pbl as --pbl-cadence; the contract check must not."""

    status = adapt.check_argv(["hex", "forecast", "--pbl", "ysu"])
    assert status["status"] == "flags-not-in-this-build", status


def test_check_argv_tells_missing_commands_from_missing_flags():
    assert adapt.check_argv(["hex", "no-such-door"])["status"] == "command-not-in-this-build"
    assert adapt.check_argv(["hex", "cull", "--no-such-flag", "x"])["status"] == "flags-not-in-this-build"
    assert adapt.check_argv(["hex", "cull", "--region", "r.json"])["status"] == "parsed"
    assert adapt.check_argv(["hex", "lbc"])["status"] == "rejected"


def test_the_door_is_registered_and_writes_a_plan(tmp_path, capsys):
    from woof.hex.cli import main

    history = frames(tmp_path / "h", datetime(2026, 10, 10, 0))
    code = main(["adapt", "--history", str(tmp_path / "h" / "cuda-history.*.nc"),
                 "--criteria", "wind,icing,assets", "--sites", str(SITES),
                 "--fine-km", "0.5", "--background-km", "3", "--dt-seconds", "3",
                 "--experimental-dt", "--raster-km", "2", "-o", str(tmp_path / "o")])
    assert code == 0
    plan = json.loads((tmp_path / "o" / "adapt-plan.json").read_text())
    assert plan["schema"] == "woof-hex.adapt-plan.v1"
    assert len(plan["history"]["files"]) == len(history)
    assert plan["estimate"]["forecast_domain_cells"] > 0
    assert "GENERATE" in capsys.readouterr().out
    assert main(["adapt", "--commit", str(tmp_path / "o")]) == 0
    assert adapt.load_state(tmp_path / "o" / "adapt-state.json")["committed"] is True
    assert main(["adapt", "--history", "x"]) == 2   # the rest are required


def test_max_cells_refuses_in_the_plan(tmp_path):
    plan = adapt.build_plan(_request(tmp_path, frames(tmp_path / "h", datetime(2026, 10, 10)),
                                     max_cells=10))
    assert plan["refusals"] and plan["runnable"] is False
    with pytest.raises(adapt.AdaptRefusal, match="never"):
        adapt.commit_state(tmp_path / "next-spec")


# ---------------------------------------------------------------------------
# the adaptive chain, against a fake runner
# ---------------------------------------------------------------------------
def _flag(argv, name):
    return argv[argv.index(name) + 1]


@pytest.fixture
def coarse_receipt(tmp_path):
    lat, lon = _cells()
    grid = write_grid(tmp_path / "coarse" / "coarse.grid.nc", lat, lon)
    files, labels = {}, {}
    start = datetime(2026, 10, 10, 0)
    for hour in range(8):
        moment = start + timedelta(hours=hour)
        path = write_frame(
            tmp_path / "coarse" / f"cuda-history.{moment:%Y-%m-%d_%H.%M.%S}.nc",
            lat, lon, rain_mm=float(hour))
        files[str(hour * 3600)] = {"path": str(path), "sha256": ""}
        labels[str(hour * 3600)] = f"{moment:%Y-%m-%d_%H.%M.%S}"
    receipt = tmp_path / "coarse" / "receipt.json"
    receipt.write_text(json.dumps({"forecast": {
        "snapshot_files": files, "history_labels": labels,
        "authority": {"files": {"grid": {"path": str(grid)},
                                "static": {"path": str(grid)}}}}}))
    return receipt, grid


def test_the_adaptive_chain_regenerates_then_keeps(tmp_path, monkeypatch, coarse_receipt):
    receipt, coarse_grid = coarse_receipt
    lat, lon = _cells()
    ran: list[tuple[int, list[str]]] = []
    cycle_of = {"n": 0}

    def fake_run(argv, *, log, env=None):
        argv = [str(item) for item in argv]
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("$ " + " ".join(argv) + "\n")
        if argv[1:3] == ["-m", "woof"] and argv[3:5] == ["hex", "adapt"]:
            from woof.hex.cli import main

            cycle_of["n"] += 1
            return main(argv[4:])
        ran.append((cycle_of["n"], argv))
        if "woof.hex.drivers.run_cuda_regional_contract" in argv:
            Path(_flag(argv, "--out")).write_text(json.dumps({"n_cells": lat.size}))
            return 0
        if argv[0] == "rw_mpas_lbc":
            out = Path(_flag(argv, "--out-dir"))
            out.mkdir(parents=True, exist_ok=True)
            (out / "lbc.2026-10-10_00.00.00.nc").write_bytes(b"x")
            return 0
        words = argv[3:]
        if words[:2] == ["hex", "cull"]:
            out = Path(_flag(words, "--out-dir"))
            write_grid(out / f"{_flag(words, '--name')}.grid.nc", lat, lon)
        elif words[:2] == ["hex", "remap"]:
            target = Path(_flag(words, "-o"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"init")
        elif words[:2] == ["hex", "forecast"]:
            out = Path(_flag(words, "--out"))
            start = datetime.strptime(_flag(words, "--start-time"), "%Y-%m-%d_%H:%M:%S")
            for hour in range(0, 7):
                moment = start + timedelta(hours=hour)
                write_frame(out / f"cuda-history.{moment:%Y-%m-%d_%H.%M.%S}.nc",
                            lat, lon, rain_mm=float(hour))
        return 0

    monkeypatch.setattr(chain, "_run", fake_run)
    monkeypatch.setattr(chain, "resolve_lbc_engine", lambda explicit: Path("rw_mpas_lbc"))
    monkeypatch.setattr(adapt, "check_argv", lambda argv: {"status": "parsed"})
    config = chain.CascadeConfig(
        out=tmp_path / "cascade", parent_row="", parent_grid=Path(""),
        parent_static=Path(""), parent_init=Path(""), parent_history=None,
        coarse_history=receipt, coarse_parent_grid=coarse_grid,
        gpuwm_checkout=None, repo=REPO, cycles=2, cycle_hours=3.0,
        plan_window_hours=2.0, fine_hours=6.0, history_every_minutes=60,
        dt_seconds=3.0, render=False, adaptive=True,
        adapt_criteria=("wind", "icing", "assets"),
        adapt_fine_km=0.5, adapt_background_km=3.0,
        adapt_options=("--sites", str(SITES), "--raster-km", "2",
                       "--experimental-dt", "--vertical-spec", str(tmp_path / "v.json")),
    )
    document = chain.run_cascade(config)
    assert document["schema"] == chain.ADAPTIVE_SCHEMA
    first, second = document["cycles"]
    assert first["decision"]["action"] == "generate"
    assert second["decision"]["action"] == "keep"        # dwell: held 1 < 2
    assert second["mesh_row"] == first["mesh_row"]
    assert first["timestep_evidence"] == "experimental-unanchored"
    stages = {1: [], 2: []}
    for cycle, argv in ran:
        if argv[1:3] == ["-m", "woof"]:
            stages[cycle].append(" ".join(argv[3:5]))
    assert stages[1][0] == "mesh --density-raster"
    assert "hex cull" in stages[1] and "hex remap" in stages[1]
    assert stages[1][-1] == "hex forecast"
    assert "mesh --density-raster" not in stages[2]
    assert stages[2] == ["hex remap", "hex forecast"]
    # Cycle 2 remaps from cycle 1's fine forecast, not the coarse parent.
    assert second["remap_source"]["grid"] == str(
        Path(first["plan"]).parent / "mesh" / f"{first['mesh_row']}.grid.nc")
    assert second["remap_source"]["role"] == "from"
    forecast = next(argv for cycle, argv in ran
                    if cycle == 2 and argv[3:5] == ["hex", "forecast"])
    assert "--experimental-dt" in forecast


def test_the_adaptive_chain_refuses_a_plan_with_missing_doors(tmp_path):
    plan = {"refusals": [], "blocked_stages": [{"stage": "lbc", "blocked_by": "x"}],
            "commands": [
                {"stage": "remap", "owner": "unit 14",
                 "contract": {"status": "command-not-in-this-build", "detail": "no remap"}},
                {"stage": "forecast", "owner": "door", "contract": {"status": "parsed"}},
            ]}
    with pytest.raises(CycleRefusal, match="remap"):
        chain.admit_adaptive_plan(plan, tmp_path / "adapt-plan.json")
    plan["commands"][0]["contract"] = {"status": "parsed"}
    chain.admit_adaptive_plan(plan, tmp_path / "adapt-plan.json")  # lbc is substituted
    plan["blocked_stages"].append({"stage": "vertical", "blocked_by": "no spec"})
    with pytest.raises(CycleRefusal, match="vertical"):
        chain.admit_adaptive_plan(plan, tmp_path / "adapt-plan.json")


def test_the_cascade_engine_pins_reach_the_planned_argv():
    config = chain.CascadeConfig(
        out=Path("o"), parent_row="", parent_grid=Path(""), parent_static=Path(""),
        parent_init=Path(""), parent_history=None, coarse_history=Path("r"),
        coarse_parent_grid=Path("g"), gpuwm_checkout=Path("/src/engine"), repo=REPO,
        mesh_exe=Path("/opt/rw_mpas_mesh"), adaptive=True)
    cull = chain.with_engine_overrides(config, ["hex", "cull", "--region", "r.json"])
    assert cull[-2:] == ["--engine", "/opt/rw_mpas_mesh"]
    forecast = chain.with_engine_overrides(config, ["hex", "forecast", "--mesh", "m"])
    assert forecast[-2:] == ["--gpuwm-checkout", "/src/engine"]
    assert chain.with_engine_overrides(config, ["hex", "remap"]) == ["hex", "remap"]


def test_adaptive_without_criteria_refuses(tmp_path):
    config = chain.CascadeConfig(
        out=tmp_path, parent_row="", parent_grid=Path(""), parent_static=Path(""),
        parent_init=Path(""), parent_history=None, coarse_history=tmp_path / "r",
        coarse_parent_grid=tmp_path / "g", gpuwm_checkout=None, repo=REPO,
        adaptive=True)
    with pytest.raises(CycleRefusal, match="--adapt-criteria"):
        chain.run_cascade(config)


def test_the_cycle_door_takes_adaptive_without_a_parent(tmp_path):
    from woof.hex.cycle.door import _add_arguments, _config, run_cycle_plan

    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    arguments = parser.parse_args([
        "--out", str(tmp_path), "--coarse-history", "r.json",
        "--coarse-parent-grid", "g.nc", "--adaptive",
        "--adapt-criteria", "wind,assets", "--adapt-fine-km", "0.1",
        "--adapt-background-km", "3", "--adapt-option", "--sites s.json",
        "--adapt-option=--experimental-dt",
    ])
    config = _config(arguments)
    assert config.adaptive is True
    assert config.adapt_criteria == ("wind", "assets")
    assert config.adapt_options == ("--sites", "s.json", "--experimental-dt")
    with pytest.raises(CycleRefusal, match="woof hex adapt"):
        run_cycle_plan(arguments)
    # Without --adaptive the fixed-parent requirements are unchanged.
    plain = parser.parse_args(["--out", str(tmp_path), "--coarse-history", "r.json"])
    with pytest.raises(CycleRefusal, match="--parent-row"):
        _config(plain)
